"""Phase 18 submission construction with strict pre/post-write validation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_test
from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)


KEYS = ["ts_code", "trade_date"]
EXPECTED_COLUMNS = [*KEYS, "pred"]


def _key_hash(frame: pd.DataFrame) -> str:
    value = int(pd.util.hash_pandas_object(frame[KEYS], index=False).sum()) & ((1 << 64) - 1)
    return f"{value:016x}"


def validate_submission_frame(
    frame: pd.DataFrame,
    expected_keys: pd.DataFrame,
    *,
    require_order: bool = True,
) -> dict[str, Any]:
    """Validate the exact competition prediction contract in memory."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Submission columns/order must be {EXPECTED_COLUMNS}, got {list(frame.columns)}")
    if list(expected_keys.columns) != KEYS:
        raise ValueError("Expected test keys must contain ts_code and trade_date only")
    if len(frame) != len(expected_keys):
        raise ValueError(f"Submission row count differs from test: {len(frame)} != {len(expected_keys)}")
    if frame[KEYS].isna().any().any() or frame["pred"].isna().any():
        raise ValueError("Submission contains missing keys or predictions")
    if frame.duplicated(KEYS).any() or expected_keys.duplicated(KEYS).any():
        raise ValueError("Submission and raw test keys must each be unique")
    codes = frame["ts_code"].astype("string")
    if not codes.str.fullmatch(r"[0-9]{6}\.[A-Z]{2}", na=False).all():
        raise ValueError("Stock codes do not match the required six-digit.exchange format")
    dates = pd.to_numeric(frame["trade_date"], errors="coerce")
    if dates.isna().any() or not np.equal(dates.to_numpy(dtype="float64") % 1.0, 0.0).all():
        raise ValueError("trade_date must contain finite integer values")
    if not dates.astype("int64").astype("string").str.fullmatch(r"[0-9]{8}").all():
        raise ValueError("trade_date must use YYYYMMDD integer format")
    predictions = pd.to_numeric(frame["pred"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(predictions).all():
        raise ValueError("pred must be numeric, finite and non-missing")
    left = frame[KEYS].copy(); left["ts_code"] = left["ts_code"].astype("string")
    right = expected_keys[KEYS].copy(); right["ts_code"] = right["ts_code"].astype("string")
    left["trade_date"] = left["trade_date"].astype("int64")
    right["trade_date"] = right["trade_date"].astype("int64")
    if not left.sort_values(KEYS, kind="stable").reset_index(drop=True).equals(
        right.sort_values(KEYS, kind="stable").reset_index(drop=True)
    ):
        raise ValueError("Submission key set differs from the untouched test key set")
    order_match = left.reset_index(drop=True).equals(right.reset_index(drop=True))
    if require_order and not order_match:
        raise ValueError("Submission does not preserve the configured raw test row order")
    merged_rows = len(left.merge(right, on=KEYS, how="inner", validate="one_to_one"))
    if merged_rows != len(right):
        raise ValueError("Submission/test inner merge changed row count")
    return {
        "columns_exact": True, "rows": int(len(frame)), "keys_unique": True,
        "key_set_exact": True, "raw_order_preserved": order_match,
        "inner_merge_rows": merged_rows, "key_hash": _key_hash(left),
        "pred_numeric": True, "pred_finite": True,
        "pred_min": float(predictions.min()), "pred_max": float(predictions.max()),
    }


def validate_csv_roundtrip(
    path: Path,
    source: pd.DataFrame,
    expected_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reload a written CSV and require exact float32 and key preservation."""
    reloaded = pd.read_csv(
        path,
        dtype={"ts_code": "string", "trade_date": "int32", "pred": "float64"},
        low_memory=False,
    )
    report = validate_submission_frame(reloaded, expected_keys, require_order=True)
    source_values = source["pred"].to_numpy(dtype="float32")
    reloaded_values = reloaded["pred"].to_numpy(dtype="float32")
    if not np.array_equal(source_values, reloaded_values):
        raise ValueError("CSV round-trip changed float32 prediction values")
    report.update({
        "roundtrip_float32_exact": True,
        "reloaded_dtypes": {name: str(dtype) for name, dtype in reloaded.dtypes.items()},
    })
    return reloaded, report


def build_submission(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Create one validated submission CSV from the immutable Phase 17 Parquet."""
    started = time.perf_counter()
    settings = config["submission"]
    root = Path(config["paths"]["root"])
    source_path = Path(settings["prediction_input"])
    target = Path(settings["output"])
    audit_path = Path(settings["audit_output"])
    if (target.exists() or audit_path.exists()) and not overwrite:
        raise FileExistsError(f"Phase 18 output exists; explicit --overwrite required: {target}")
    raw_before = {name: sha256_file(Path(config["paths"][name])) for name in ("train", "test")}
    source = pd.read_parquet(source_path, engine="pyarrow")
    test = load_test(config, use_cache=True)
    expected_keys = test[KEYS].copy()
    prewrite = validate_submission_frame(source, expected_keys, require_order=True)
    if str(source["pred"].dtype) != str(settings["prediction_dtype"]):
        raise ValueError(f"Prediction dtype must be {settings['prediction_dtype']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.csv")
    if temporary.exists():
        temporary.unlink()
    try:
        source.to_csv(temporary, index=False, columns=EXPECTED_COLUMNS, lineterminator="\n", encoding="utf-8")
        reloaded, postwrite = validate_csv_roundtrip(temporary, source, expected_keys)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Submission appeared during build: {target}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    raw_after = {name: sha256_file(Path(config["paths"][name])) for name in ("train", "test")}
    if raw_before != raw_after:
        raise ValueError("Raw train/test files changed during submission construction")
    config_path = root / "config.yaml"; config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase18_submission_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source_tree = build_source_manifest(root)
    source_manifest = Path(config["experiments"]["manifests_dir"]) / f"source_{source_tree['tree_sha256'][:12]}.json"
    write_immutable(source_manifest, json.dumps(source_tree, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [
        config_snapshot, source_path, Path(config["prediction"]["audit_output"]), target,
        Path(config["paths"]["test"]), Path(config["paths"]["official_evaluator"]),
    ]
    artifact = {
        "phase": 18, "test_labels_used": False, "hidden_test_metrics_computed": False,
        "source_tree_sha256": source_tree["tree_sha256"],
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase18_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    git = git_state(root)
    registry = append_registry_rows(Path(config["experiments"]["registry"]), [{
        "experiment_id": settings["experiment_id"], "timestamp": timestamp,
        "record_type": "summary", "fold": "submission_validation",
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "source_tree_sha256": source_tree["tree_sha256"], "source_manifest_path": str(source_manifest),
        "config_path": str(config_snapshot), "config_sha256": config_sha,
        "feature_version": "phase17_locked_prediction", "feature_manifest_sha256": sha256_file(source_path),
        "model": "submission_contract_validation", "model_params": json.dumps({
            "columns": EXPECTED_COLUMNS, "prediction_dtype": settings["prediction_dtype"],
            "preserve_raw_test_order": True,
        }, sort_keys=True, separators=(",", ":")),
        "train_period": "not_applicable", "val_period": "hidden_test_unscored",
        "seed": config["project"]["seed"],
        "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(target)]),
        "notes": "csv_roundtrip_validated_no_hidden_labels_no_metric",
    }])
    report = {
        "phase": 18, "experiment_id": settings["experiment_id"],
        "source_prediction": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "submission": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)},
        "prewrite_validation": prewrite, "postwrite_validation": postwrite,
        "contracts": {
            "column_order_exact": True, "test_key_set_exact": True,
            "raw_test_order_preserved": True, "csv_roundtrip_passed": True,
            "float32_values_roundtrip_exact": True, "test_labels_read_or_inferred": False,
            "hidden_test_metrics_computed": False, "raw_files_unchanged": True,
            "silent_overwrite_disabled": True,
        },
        "raw_sha256_before": raw_before, "raw_sha256_after": raw_after,
        "run_seconds": time.perf_counter() - started,
        "source_tree_sha256": source_tree["tree_sha256"], "source_manifest": str(source_manifest),
        "config_sha256": config_sha, "config_snapshot": str(config_snapshot),
        "artifact_manifest": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_count": len(artifact["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact["artifacts"]),
        "registry": registry,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_path.with_suffix(".tmp.json")
    temporary_audit.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_audit, audit_path)
    return report
