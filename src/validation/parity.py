"""Deterministic synthetic fixture for official/local evaluator parity checks."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import ProjectPaths
from src.validation.evaluator import (
    METRIC_KEYS,
    evaluate_files,
    evaluation_diagnostics,
    merge_evaluation_inputs,
    validate_official_contract,
)


def build_official_parity_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cover exact 30/100 cutoffs, exclusions, floor top sizes and turnover reset."""
    dates = [20250102, 20250103, 20250106, 20250107, 20250108, 20250109]
    non_limit_counts = [99, 100, 99, 100, 101, 110]
    label_counts = [29, 30, 99, 100, 100, 109]
    prediction_rows, label_rows, feature_rows = [], [], []
    for date_index, (date, non_limit_count, label_count) in enumerate(
        zip(dates, non_limit_counts, label_counts)
    ):
        for stock_index in range(120):
            code = f"S{stock_index:03d}"
            base = (stock_index - 60) / 1000.0
            if date_index == 0:
                prediction = float(stock_index)
            elif date_index == 1:
                prediction = base
            elif date_index == 2:
                prediction = -base
            elif date_index == 3:
                prediction = float((stock_index + 7) % 120)
            elif date_index == 4:
                prediction = float((stock_index + 23) % 120)
            else:
                prediction = float(np.sin(stock_index / 7.0))
            label = base + date_index * 0.0001 if stock_index < label_count else np.nan
            prediction_rows.append({"ts_code": code, "trade_date": date, "pred": prediction})
            label_rows.append({"ts_code": code, "trade_date": date, "y_ret_1d": label})
            feature_rows.append({
                "ts_code": code,
                "trade_date": date,
                "flag_limit_up": 0 if stock_index < non_limit_count else 1,
                "unused_feature": stock_index,
            })
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(feature_rows),
    )


def _load_official_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("vendored_official_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_diff(left: float, right: float) -> float:
    if np.isnan(left) and np.isnan(right):
        return 0.0
    return abs(float(left) - float(right))


def _key_hash_sum(frame: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(frame[["ts_code", "trade_date"]], index=False).sum()) & ((1 << 64) - 1)


def run_official_parity_check(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    settings = config["evaluator"]
    output_path = Path(settings["audit_output"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Evaluator audit exists; pass overwrite explicitly: {output_path}")
    contract = validate_official_contract(config)
    predictions, labels, features = build_official_parity_fixture()
    paths = ProjectPaths.from_config(config)
    paths.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase7_parity_", dir=paths.cache) as directory:
        fixture_dir = Path(directory)
        predictions.to_csv(fixture_dir / "submission.csv", index=False)
        labels.to_csv(fixture_dir / "测试集_Y.csv", index=False)
        features.to_csv(fixture_dir / "测试集_X.csv", index=False)
        official_module = _load_official_module(Path(contract["path"]))
        official = official_module.evaluate(str(fixture_dir / "submission.csv"), str(fixture_dir))
        local = evaluate_files(fixture_dir / "submission.csv", fixture_dir)
    contract_after = validate_official_contract(config)

    differences = {
        key: _metric_diff(float(official[key]), float(local[key])) for key in METRIC_KEYS
    }
    tolerance = float(settings.get("parity_absolute_tolerance", 1e-12))
    merged = merge_evaluation_inputs(predictions, labels, features)
    diagnostics = evaluation_diagnostics(merged)
    branch_checks = {
        "rank_ic_exact_30_included": diagnostics["by_date"][1]["rank_ic_eligible"],
        "rank_ic_29_skipped": not diagnostics["by_date"][0]["rank_ic_eligible"],
        "portfolio_exact_100_included": diagnostics["by_date"][3]["portfolio_eligible"],
        "portfolio_99_skipped": not diagnostics["by_date"][2]["portfolio_eligible"],
        "portfolio_109_uses_top_10": diagnostics["by_date"][5]["portfolio_top_n"] == 10,
        "turnover_110_uses_top_11": diagnostics["by_date"][5]["turnover_top_n"] == 11,
        "turnover_sub_100_resets": diagnostics["turnover_comparisons"] == 2,
        "limit_up_filter_pools_differ": (
            diagnostics["by_date"][1]["rank_ic_valid_rows"]
            != diagnostics["by_date"][1]["turnover_valid_rows"]
        ),
        "missing_label_filter_pools_differ": (
            diagnostics["by_date"][1]["portfolio_valid_rows"]
            != diagnostics["by_date"][1]["turnover_valid_rows"]
        ),
    }
    report = {
        "phase": 7,
        "fixture": "deterministic_synthetic_parity_only_not_training_data",
        "official_contract": contract,
        "official_sha256_after": contract_after["sha256"],
        "metric_keys": list(METRIC_KEYS),
        "official_metrics": {key: float(official[key]) for key in METRIC_KEYS},
        "local_metrics": {key: float(local[key]) for key in METRIC_KEYS},
        "absolute_differences": differences,
        "maximum_absolute_difference": max(differences.values()),
        "absolute_tolerance": tolerance,
        "parity_passed": all(value <= tolerance for value in differences.values()),
        "branch_checks": branch_checks,
        "all_branch_checks_passed": all(branch_checks.values()),
        "diagnostics": diagnostics,
        "fixture_key_hash": f"{_key_hash_sum(predictions):016x}",
        "official_file_modified": contract["sha256"] != contract_after["sha256"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return report
