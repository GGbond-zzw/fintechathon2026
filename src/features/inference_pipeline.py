"""Build label-free test features with real training history as causal warm-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.loader import load_test, load_train_tail
from src.features.cross_sectional import build_cross_sectional_store_from_paths
from src.features.market import compute_market_features
from src.features.pipeline import FEATURE_INPUTS, _hash_sum, build_time_feature_store
from src.features.storage import repartition_parquet_by_year
from src.utils.paths import ProjectPaths


def maximum_required_history(settings: dict[str, Any]) -> int:
    """Return the largest configured lag/rolling lookback in trading rows."""
    candidates = [1]
    for key in ("return_windows", "volatility_windows", "volume_windows", "position_windows"):
        candidates.extend(int(value) for value in settings[key])
    candidates.extend(int(long_window) for _, long_window in settings["ma_pairs"])
    return max(candidates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _with_suffix(path: Path, max_stocks: int | None) -> Path:
    if max_stocks is None:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}_smoke_{max_stocks}{path.suffix}")
    return path.with_name(f"{path.name}_smoke_{max_stocks}")


def _parquet_schema(path: Path) -> list[tuple[str, str]]:
    schema = pq.ParquetFile(path).schema_arrow
    return [(field.name, str(field.type)) for field in schema]


def _first_year_file(directory: Path) -> Path:
    candidates = sorted(directory.glob("year=*/features.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No yearly feature files found: {directory}")
    return candidates[0]


def build_test_feature_artifacts(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    max_stocks: int | None = None,
) -> dict[str, Any]:
    """Build Phase 5 artifacts without creating or consuming test labels."""
    paths = ProjectPaths.from_config(config)
    feature_settings = config["features"]
    test_settings = config["test_features"]
    cross_settings = config["cross_sectional"]
    warmup_days = int(config["data"]["warmup_days"])
    required_history = maximum_required_history(feature_settings)
    if warmup_days < required_history:
        raise ValueError(
            f"warmup_days={warmup_days} is shorter than maximum feature history={required_history}"
        )

    output_paths = {
        name: _with_suffix(Path(test_settings[name]), max_stocks)
        for name in (
            "time_output", "market_output", "model_base_by_year_output",
            "cross_sectional_by_year_output", "audit_output",
        )
    }
    material_outputs = [
        output_paths["time_output"], output_paths["market_output"],
        output_paths["model_base_by_year_output"], output_paths["cross_sectional_by_year_output"],
    ]
    existing = [str(path) for path in material_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Test feature outputs exist; pass overwrite explicitly: {existing}")

    raw_hash_before = {"train": _sha256(paths.train), "test": _sha256(paths.test)}
    train_tail = load_train_tail(config, warmup_days, use_cache=True)
    test = load_test(config, use_cache=True)
    label = str(config["data"]["label"])
    if label in test.columns:
        raise ValueError(f"Configured test data unexpectedly contains label column: {label}")
    if max_stocks is not None:
        if max_stocks < 1:
            raise ValueError("max_stocks must be positive")
        selected = test["ts_code"].drop_duplicates().head(max_stocks)
        test = test[test["ts_code"].isin(selected)].copy()
        train_tail = train_tail[train_tail["ts_code"].isin(selected)].copy()

    test_start = int(test["trade_date"].min())
    test_dates = set(int(value) for value in test["trade_date"].unique())
    if int(train_tail["trade_date"].max()) >= test_start:
        raise ValueError("Training warm-up overlaps the test period")
    if train_tail.duplicated(["ts_code", "trade_date"]).any() or test.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Warm-up and test keys must be unique")

    combined = pd.concat(
        [train_tail[FEATURE_INPUTS], test[FEATURE_INPUTS]], ignore_index=True
    ).sort_values(["ts_code", "trade_date"], kind="stable")
    time_report = build_time_feature_store(
        combined,
        output_paths["time_output"],
        feature_settings,
        output_start_date=test_start,
    )

    market = compute_market_features(combined)
    market = market[market["trade_date"].isin(test_dates)].reset_index(drop=True)
    output_paths["market_output"].parent.mkdir(parents=True, exist_ok=True)
    market_tmp = output_paths["market_output"].with_suffix(".tmp.parquet")
    market.to_parquet(
        market_tmp, engine="pyarrow", compression=str(feature_settings["parquet_compression"]), index=False
    )
    market_tmp.replace(output_paths["market_output"])

    model_columns = ["ts_code", "trade_date", *list(cross_settings["source_features"])]
    model_report = repartition_parquet_by_year(
        output_paths["time_output"],
        output_paths["model_base_by_year_output"],
        compression=str(feature_settings["parquet_compression"]),
        overwrite=overwrite,
        columns=model_columns,
    )
    cross_report = build_cross_sectional_store_from_paths(
        output_paths["model_base_by_year_output"],
        output_paths["cross_sectional_by_year_output"],
        cross_settings,
        overwrite=overwrite,
        phase=5,
    )

    expected_hash = _hash_sum(test[["ts_code", "trade_date"]])
    first_date_features = pd.read_parquet(
        output_paths["time_output"],
        columns=["trade_date", *list(test_settings["first_date_probe_features"])],
        filters=[("trade_date", "=", test_start)],
        engine="pyarrow",
    )
    probe_finite_counts = {
        column: int(np.isfinite(first_date_features[column].to_numpy(dtype="float64", na_value=np.nan)).sum())
        for column in test_settings["first_date_probe_features"]
    }
    train_time_schema = _parquet_schema(Path(feature_settings["time_output"]))
    test_time_schema = _parquet_schema(output_paths["time_output"])
    train_model_schema = _parquet_schema(_first_year_file(Path(feature_settings["model_base_by_year_output"])))
    test_model_schema = _parquet_schema(_first_year_file(output_paths["model_base_by_year_output"]))
    train_cross_schema = _parquet_schema(_first_year_file(Path(cross_settings["output"])))
    test_cross_schema = _parquet_schema(_first_year_file(output_paths["cross_sectional_by_year_output"]))
    raw_hash_after = {"train": _sha256(paths.train), "test": _sha256(paths.test)}

    report = {
        "phase": 5,
        "scope": "full" if max_stocks is None else f"smoke_{max_stocks}_stocks",
        "warmup_days": warmup_days,
        "maximum_required_history": required_history,
        "warmup_sufficient": warmup_days >= required_history,
        "train_tail_date_range": [int(train_tail["trade_date"].min()), int(train_tail["trade_date"].max())],
        "test_date_range": [test_start, int(test["trade_date"].max())],
        "test_rows": int(len(test)),
        "test_dates": int(test["trade_date"].nunique()),
        "test_stocks": int(test["ts_code"].nunique()),
        "label_column_present_in_test_input": label in test.columns,
        "label_column_present_in_any_output": any(
            name == label for name, _ in test_time_schema + test_model_schema + test_cross_schema
        ),
        "time_features": time_report,
        "market_features": {
            "rows": int(len(market)),
            "dates": int(market["trade_date"].nunique()),
            "first_date_valid_return_count": int(market.iloc[0]["market_valid_return_count"]),
            "inf_count": int(np.isinf(market.drop(columns="trade_date").to_numpy(dtype="float64", na_value=np.nan)).sum()),
            "file_bytes": output_paths["market_output"].stat().st_size,
        },
        "model_base": model_report,
        "cross_sectional": cross_report,
        "expected_test_key_hash": f"{expected_hash:016x}",
        "all_key_hashes_equal": all([
            time_report["output_key_hash"] == f"{expected_hash:016x}",
            model_report["partition_key_hash"] == f"{expected_hash:016x}",
            cross_report["output_key_hash"] == f"{expected_hash:016x}",
        ]),
        "first_test_date_probe_finite_counts": probe_finite_counts,
        "first_test_date_probes_not_all_missing": all(value > 0 for value in probe_finite_counts.values()),
        "schema_equal_to_train": {
            "time": train_time_schema == test_time_schema,
            "model_base": train_model_schema == test_model_schema,
            "cross_sectional": train_cross_schema == test_cross_schema,
        },
        "raw_sha256_before": raw_hash_before,
        "raw_sha256_after": raw_hash_after,
        "raw_files_unchanged": raw_hash_before == raw_hash_after,
        "future_shift_used": False,
        "test_labels_created_or_inferred": False,
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    output_paths["audit_output"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["audit_output"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
