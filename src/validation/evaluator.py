"""Local wrapper that preserves the vendored official evaluator semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRIC_KEYS = (
    "ic_mean", "ic_std", "icir", "ic_positive_ratio",
    "annual_excess", "top1_annual_ret", "mean_turnover", "final_score",
)

OFFICIAL_CONSTANTS = {
    "ic_min_samples": 30,
    "portfolio_min_samples": 100,
    "top_fraction_denominator": 10,
    "annualization_days": 252,
    "weights": {"rank_ic": 0.4, "annual_excess": 0.3, "one_minus_turnover": 0.3},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_official_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Reject config or vendored-script drift from the official scoring contract."""
    settings = config["evaluator"]
    if not bool(settings.get("constants_locked")):
        raise ValueError("Evaluator constants must remain locked")
    for key in ("ic_min_samples", "portfolio_min_samples", "top_fraction_denominator", "annualization_days"):
        if int(settings[key]) != OFFICIAL_CONSTANTS[key]:
            raise ValueError(f"Official evaluator constant drift: {key}")
    actual_weights = {key: float(value) for key, value in settings["weights"].items()}
    if actual_weights != OFFICIAL_CONSTANTS["weights"]:
        raise ValueError("Official evaluator weight drift")
    official_path = Path(config["paths"]["official_evaluator"])
    actual_hash = _sha256(official_path)
    expected_hash = str(settings["official_sha256"]).upper()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Official evaluator hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return {"path": str(official_path), "sha256": actual_hash, "constants": OFFICIAL_CONSTANTS}


def merge_evaluation_inputs(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """Perform the same two inner merges as the official file evaluator."""
    merged = predictions.merge(labels, on=["ts_code", "trade_date"], how="inner")
    return merged.merge(test_features, on=["ts_code", "trade_date"], how="inner")


def evaluate_merged_frame(df: pd.DataFrame) -> dict[str, float]:
    """Compute the eight official metrics without changing any filter or threshold."""
    required = {"ts_code", "trade_date", "pred", "y_ret_1d", "flag_limit_up"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing evaluator columns: {missing}")

    ic_list = []
    for _, group in df.groupby("trade_date"):
        valid = group.dropna(subset=["y_ret_1d"])
        if len(valid) < 30:
            continue
        ic, _ = spearmanr(valid["pred"], valid["y_ret_1d"])
        ic_list.append(ic)
    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_ratio = np.mean(np.array(ic_list) > 0)

    excess_list = []
    for _, group in df.groupby("trade_date"):
        valid = group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()].copy()
        if len(valid) < 100:
            continue
        valid = valid.sort_values("pred", ascending=False).reset_index(drop=True)
        n_top = max(len(valid) // 10, 1)
        top1_ret = valid["y_ret_1d"].iloc[:n_top].mean()
        market_ret = valid["y_ret_1d"].mean()
        excess_list.append(top1_ret - market_ret)
    annual_excess = np.mean(excess_list) * 252
    top1_annual_ret = np.mean([
        group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()]
        .sort_values("pred", ascending=False)["y_ret_1d"]
        .iloc[:max(len(group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()]) // 10, 1)]
        .mean()
        for _, group in df.groupby("trade_date")
        if len(group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()]) >= 100
    ]) * 252

    dates = sorted(df["trade_date"].unique())
    turnover_list = []
    prev_set = None
    for date in dates:
        group = df[df["trade_date"] == date]
        valid = group[group["flag_limit_up"] == 0].copy()
        if len(valid) < 100:
            prev_set = None
            continue
        valid = valid.sort_values("pred", ascending=False)
        n_top = max(len(valid) // 10, 1)
        curr_set = set(valid["ts_code"].iloc[:n_top])
        if prev_set is not None and len(prev_set) > 0:
            intersection = len(curr_set & prev_set)
            union = len(curr_set | prev_set)
            turnover_list.append(1.0 - intersection / union)
        prev_set = curr_set
    mean_turnover = np.mean(turnover_list)
    final_score = ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_ratio": ic_positive_ratio,
        "annual_excess": annual_excess,
        "top1_annual_ret": top1_annual_ret,
        "mean_turnover": mean_turnover,
        "final_score": final_score,
    }


def evaluate_frames(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict[str, float]:
    return evaluate_merged_frame(merge_evaluation_inputs(predictions, labels, test_features))


def evaluate_files(submission_path: str | Path, data_dir: str | Path) -> dict[str, float]:
    """Read the same CSV files and columns as official/evaluate.py."""
    directory = Path(data_dir)
    predictions = pd.read_csv(submission_path)
    labels = pd.read_csv(directory / "测试集_Y.csv")
    features = pd.read_csv(
        directory / "测试集_X.csv", usecols=["ts_code", "trade_date", "flag_limit_up"]
    )
    return evaluate_frames(predictions, labels, features)


def evaluation_diagnostics(merged: pd.DataFrame) -> dict[str, Any]:
    """Report branch coverage without altering the official metric return schema."""
    by_date = []
    turnover_comparisons = 0
    previous_eligible = False
    for date, group in merged.groupby("trade_date"):
        label_count = int(group["y_ret_1d"].notna().sum())
        portfolio_count = int(((group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()).sum())
        turnover_count = int((group["flag_limit_up"] == 0).sum())
        turnover_eligible = turnover_count >= 100
        if turnover_eligible and previous_eligible:
            turnover_comparisons += 1
        previous_eligible = turnover_eligible
        by_date.append({
            "trade_date": int(date),
            "rank_ic_valid_rows": label_count,
            "rank_ic_eligible": label_count >= 30,
            "portfolio_valid_rows": portfolio_count,
            "portfolio_eligible": portfolio_count >= 100,
            "portfolio_top_n": max(portfolio_count // 10, 1) if portfolio_count >= 100 else None,
            "turnover_valid_rows": turnover_count,
            "turnover_eligible": turnover_eligible,
            "turnover_top_n": max(turnover_count // 10, 1) if turnover_eligible else None,
        })
    return {
        "merged_rows": int(len(merged)),
        "dates": int(merged["trade_date"].nunique()),
        "rank_ic_eligible_dates": sum(item["rank_ic_eligible"] for item in by_date),
        "portfolio_eligible_dates": sum(item["portfolio_eligible"] for item in by_date),
        "turnover_eligible_dates": sum(item["turnover_eligible"] for item in by_date),
        "turnover_comparisons": turnover_comparisons,
        "by_date": by_date,
    }
