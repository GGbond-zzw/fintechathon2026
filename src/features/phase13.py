"""Build, audit, fingerprint and register Phase 13 advanced feature stores."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_test, load_train_columns, load_train_tail
from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)
from src.features.advanced import (
    advanced_feature_dictionary, build_advanced_year_store,
    compute_advanced_features, maximum_advanced_history,
)
from src.features.pipeline import FEATURE_INPUTS
from src.utils.paths import ProjectPaths


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _hand_checks(base: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    targets = ["rsi_6", "atr_norm_14", "close_vwap_ratio_20", "trend_efficiency_20", "volume_breakout_20"]
    chosen = None
    features = None
    for _, stock in base.groupby("ts_code", sort=False, observed=True):
        current = compute_advanced_features(stock, settings)
        valid = current[targets].notna().all(axis=1)
        if valid.any():
            row = int(np.flatnonzero(valid.to_numpy())[-1])
            chosen = stock.sort_values("trade_date", kind="stable").reset_index(drop=True)
            features = current
            break
    if chosen is None or features is None:
        raise ValueError("No stock row supports advanced feature hand checks")
    row = int(np.flatnonzero(features[targets].notna().all(axis=1).to_numpy())[-1])
    close = chosen["close"].astype("float64")
    high = chosen["high"].astype("float64")
    low = chosen["low"].astype("float64")
    vol = chosen["vol"].astype("float64")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    average_gain = gain.iloc[row-5:row+1].mean()
    average_loss = loss.iloc[row-5:row+1].mean()
    previous_close = close.shift(1)
    true_range = pd.concat([high-low, (high-previous_close).abs(), (low-previous_close).abs()], axis=1).max(axis=1, skipna=False)
    ret_1 = close / previous_close - 1.0
    manual = {
        "rsi_6": average_gain / (average_gain + average_loss) if average_gain + average_loss else 0.5,
        "atr_norm_14": true_range.iloc[row-13:row+1].mean() / close.iloc[row],
        "close_vwap_ratio_20": close.iloc[row] / (
            (close.iloc[row-19:row+1] * vol.iloc[row-19:row+1]).sum()
            / vol.iloc[row-19:row+1].sum()
        ) - 1.0,
        "trend_efficiency_20": (close.iloc[row] / close.iloc[row-20] - 1.0) / ret_1.abs().iloc[row-19:row+1].sum(),
        "volume_breakout_20": vol.iloc[row] / vol.iloc[row-20:row].max() - 1.0,
    }
    return pd.DataFrame([{
        "ts_code": str(chosen.loc[row, "ts_code"]), "trade_date": int(chosen.loc[row, "trade_date"]),
        "feature": name, "manual": float(value), "computed": float(features.loc[row, name]),
        "abs_error": abs(float(value) - float(features.loc[row, name])),
    } for name, value in manual.items()])


def build_phase13(config: dict[str, Any], *, overwrite: bool = False, max_stocks: int | None = None) -> dict[str, Any]:
    settings = config["advanced_features"]
    paths = ProjectPaths.from_config(config)
    train_output = Path(settings["train_output"])
    test_output = Path(settings["test_output"])
    raw_before = {"train": sha256_file(paths.train), "test": sha256_file(paths.test)}
    train = load_train_columns(config, FEATURE_INPUTS, use_cache=True)
    train_report = build_advanced_year_store(
        train, train_output, settings, overwrite=overwrite, max_stocks=max_stocks
    )

    test = load_test(config, use_cache=True)
    if config["data"]["label"] in test.columns:
        raise ValueError("Test input unexpectedly contains the training label")
    if max_stocks is not None:
        selected = test["ts_code"].drop_duplicates().head(max_stocks)
        test = test[test["ts_code"].isin(selected)].copy()
    warmup_days = int(config["data"]["warmup_days"])
    required_history = maximum_advanced_history(settings)
    if warmup_days < required_history:
        raise ValueError("Configured warm-up is shorter than advanced feature history")
    tail = load_train_tail(config, warmup_days, use_cache=True)
    if max_stocks is not None:
        tail = tail[tail["ts_code"].isin(test["ts_code"].unique())]
    test_start = int(test["trade_date"].min())
    combined = pd.concat([tail[FEATURE_INPUTS], test[FEATURE_INPUTS]], ignore_index=True)
    test_report = build_advanced_year_store(
        combined, test_output, settings, overwrite=overwrite,
        output_start_date=test_start, max_stocks=max_stocks,
    )
    if train_report["feature_names"] != test_report["feature_names"] or train_report["schema"] != test_report["schema"]:
        raise ValueError("Advanced train/test schemas differ")

    dictionary = advanced_feature_dictionary(settings)
    dictionary_path = Path(settings["dictionary_output"])
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_tmp = dictionary_path.with_suffix(".tmp.csv")
    dictionary.to_csv(dictionary_tmp, index=False, lineterminator="\n")
    os.replace(dictionary_tmp, dictionary_path)
    hand_checks = _hand_checks(train, settings)
    hand_path = Path(settings["handcheck_output"])
    hand_tmp = hand_path.with_suffix(".tmp.csv")
    hand_checks.to_csv(hand_tmp, index=False, lineterminator="\n")
    os.replace(hand_tmp, hand_path)

    probe = train[["close", "vol", "amount"]].head(250000).astype("float64")
    required_scale = probe["close"] * probe["vol"] / probe["amount"].where(probe["amount"] != 0)
    raw_after = {"train": sha256_file(paths.train), "test": sha256_file(paths.test)}
    report = {
        "phase": 13, "scope": "full" if max_stocks is None else f"smoke_{max_stocks}_stocks",
        "version": settings["version"], "feature_count": len(train_report["feature_names"]),
        "feature_names": train_report["feature_names"], "train": train_report, "test": test_report,
        "warmup_days": warmup_days, "maximum_required_history": required_history,
        "warmup_sufficient": warmup_days >= required_history,
        "train_test_schema_equal": True,
        "dictionary": {"path": str(dictionary_path), "rows": len(dictionary)},
        "hand_checks": {"path": str(hand_path), "rows": len(hand_checks),
                        "max_abs_error": float(hand_checks["abs_error"].max())},
        "vwap_proxy_diagnostic": {
            "definition": "rolling_sum(adjusted_close*vol)/rolling_sum(vol)",
            "raw_amount_implied_price_used": False,
            "reason": "OHLC is adjusted while raw amount/vol price scale is not stable",
            "sample_rows": int(required_scale.notna().sum()),
            "raw_amount_required_scale_quantiles": {
                str(q): float(value)
                for q, value in required_scale.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()
            },
        },
        "contracts": {
            "label_used": False, "future_shift_used": False,
            "test_labels_created_or_inferred": False, "cross_sectional_screening_performed": False,
            "model_training_performed": False, "raw_files_unchanged": raw_before == raw_after,
        },
        "raw_sha256_before": raw_before, "raw_sha256_after": raw_after,
    }

    root = Path(config["paths"]["root"])
    config_path = root / "config.yaml"
    config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase13_advanced_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root)
    source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [
        paths.train, paths.test, config_snapshot, train_output / "manifest.json", test_output / "manifest.json",
        *sorted(train_output.glob("year=*/features.parquet")),
        *sorted(test_output.glob("year=*/features.parquet")), dictionary_path, hand_path,
    ]
    artifact_manifest = {
        "phase": 13, "test_data_used_for_supervised_fit": False,
        "source_tree_sha256": source["tree_sha256"], "feature_version": settings["version"],
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase13_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)
    git = git_state(root)
    registry_result = append_registry_rows(Path(config["experiments"]["registry"]), [{
        "experiment_id": settings["experiment_id"],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "record_type": "artifact", "fold": "not_applicable",
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
        "config_path": str(config_snapshot), "config_sha256": config_sha,
        "feature_version": settings["version"],
        "feature_manifest_sha256": sha256_file(train_output / "manifest.json"),
        "model": "feature_build_only_no_model", "model_params": json.dumps({
            "feature_count": len(train_report["feature_names"]), "warmup_days": warmup_days,
        }, sort_keys=True, separators=(",", ":")),
        "train_period": f"{train_report['date_range'][0]}-{train_report['date_range'][1]}",
        "val_period": "not_applicable", "seed": config["project"]["seed"],
        "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(train_output), str(test_output), str(dictionary_path)], separators=(",", ":")),
        "notes": "causal_incremental_features_no_label_screening_no_model_training",
    }])
    audit = {
        **report, "source_tree_sha256": source["tree_sha256"], "source_file_count": source["file_count"],
        "source_manifest": str(source_path), "config_sha256": config_sha,
        "config_snapshot": str(config_snapshot), "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha, "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "git": git, "registry": {"path": config["experiments"]["registry"], **registry_result},
    }
    _atomic_json(Path(settings["audit_output"]), audit)
    return audit
