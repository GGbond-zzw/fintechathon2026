"""Phase 2 label-only diagnostics.

Negative shifts and future closes are strictly confined to this audit module.
Nothing produced here is a model feature or a replacement training label.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_label_audit_columns(path: Path, label: str) -> pd.DataFrame:
    """Read only the four columns needed, preserving return precision."""
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    return pd.read_csv(
        path,
        usecols=["ts_code", "trade_date", "close", label],
        dtype={
            "ts_code": "category",
            "trade_date": "int32",
            "close": "float64",
            label: "float64",
        },
        low_memory=False,
    )


def load_price_audit_columns(path: Path) -> pd.DataFrame:
    """Read label-free prices used only to close the train/test boundary."""
    if not path.is_file():
        raise FileNotFoundError(f"Price CSV not found: {path}")
    return pd.read_csv(
        path,
        usecols=["ts_code", "trade_date", "close"],
        dtype={"ts_code": "category", "trade_date": "int32", "close": "float64"},
        low_memory=False,
    )


def label_distribution(
    frame: pd.DataFrame,
    label: str,
    quantiles: list[float],
    extreme_thresholds: list[float],
) -> dict[str, Any]:
    values = frame[label]
    finite = values[np.isfinite(values)]
    return {
        "rows": int(len(values)),
        "non_missing": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "missing_rate": float(values.isna().mean()),
        "inf": int(np.isinf(values.fillna(0.0)).sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "quantiles": {str(q): float(finite.quantile(q)) for q in quantiles},
        "absolute_extremes": {
            str(threshold): int((finite.abs() >= threshold).sum())
            for threshold in extreme_thresholds
        },
    }


def time_breakdowns(
    frame: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame[["trade_date", label]].copy()
    work["year"] = work["trade_date"] // 10000

    def summarize(grouped: Any) -> pd.DataFrame:
        base = grouped[label].agg(["size", "count", "mean", "std", "min", "max"])
        base = base.rename(columns={"size": "rows", "count": "non_missing"})
        base["missing"] = base["rows"] - base["non_missing"]
        base["missing_rate"] = base["missing"] / base["rows"]
        quantile_table = grouped[label].quantile([0.01, 0.5, 0.99]).unstack()
        quantile_table.columns = ["q01", "q50", "q99"]
        return base.join(quantile_table).reset_index()

    yearly = summarize(work.groupby("year", sort=True, observed=True))
    daily = summarize(work.groupby("trade_date", sort=True, observed=True))
    return yearly, daily


def _alignment_metrics(
    actual: pd.Series,
    expected: pd.Series,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    comparable = actual.notna() & expected.notna()
    errors = (actual - expected).abs()
    matching = pd.Series(False, index=actual.index)
    matching.loc[comparable] = np.isclose(
        actual.loc[comparable],
        expected.loc[comparable],
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )
    missing_agreement = actual.isna() == expected.isna()
    metrics = {
        "comparable_rows": int(comparable.sum()),
        "matching_rows": int(matching.sum()),
        "match_rate_on_comparable": float(matching.loc[comparable].mean()) if comparable.any() else None,
        "mean_abs_error": float(errors.loc[comparable].mean()) if comparable.any() else None,
        "max_abs_error": float(errors.loc[comparable].max()) if comparable.any() else None,
        "missing_mask_agreement_rows": int(missing_agreement.sum()),
        "missing_mask_agreement_rate": float(missing_agreement.mean()),
        "label_missing_expected_present": int((actual.isna() & expected.notna()).sum()),
        "label_present_expected_missing": int((actual.notna() & expected.isna()).sum()),
    }
    return metrics, matching, comparable


def audit_label_alignment(
    frame: pd.DataFrame,
    label: str,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    mismatch_sample_rows: int,
    future_prices: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare labels with immediate-row and next-non-null close returns."""
    ordered = frame.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    train_prices = ordered[["ts_code", "trade_date", "close"]].copy()
    train_prices["_training_index"] = np.arange(len(train_prices), dtype="int64")
    if future_prices is not None:
        boundary_prices = future_prices[["ts_code", "trade_date", "close"]].copy()
        boundary_prices["_training_index"] = -1
        timeline = pd.concat([train_prices, boundary_prices], ignore_index=True)
    else:
        timeline = train_prices
    timeline = timeline.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    grouped = timeline.groupby("ts_code", sort=False, observed=True)["close"]
    timeline["_immediate_next_close"] = grouped.shift(-1)
    timeline["_next_valid_close"] = grouped.transform(lambda series: series.shift(-1).bfill())
    training_timeline = timeline.loc[timeline["_training_index"] >= 0].sort_values("_training_index")
    immediate_next_close = training_timeline["_immediate_next_close"].reset_index(drop=True)
    next_valid_close = training_timeline["_next_valid_close"].reset_index(drop=True)
    current_close = ordered["close"]
    immediate_expected = immediate_next_close / current_close - 1.0
    next_valid_expected = next_valid_close / current_close - 1.0

    immediate, immediate_matching, immediate_comparable = _alignment_metrics(
        ordered[label], immediate_expected,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    next_valid, next_valid_matching, next_valid_comparable = _alignment_metrics(
        ordered[label], next_valid_expected,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    methods = {"immediate_next_row": immediate, "next_non_null_close": next_valid}
    supported = max(
        methods,
        key=lambda name: (
            methods[name]["matching_rows"],
            methods[name]["missing_mask_agreement_rows"],
        ),
    )
    mismatch = (
        (immediate_comparable & ~immediate_matching)
        | (ordered[label].isna() != immediate_expected.isna())
        | (next_valid_comparable & ~next_valid_matching)
        | (ordered[label].isna() != next_valid_expected.isna())
    )
    samples = ordered.loc[mismatch, ["ts_code", "trade_date", "close", label]].copy()
    samples["immediate_expected"] = immediate_expected.loc[mismatch]
    samples["next_valid_expected"] = next_valid_expected.loc[mismatch]
    samples = samples.head(mismatch_sample_rows)
    report = {
        "ordered_by": ["ts_code", "trade_date"],
        "negative_shift_scope": "label audit only; not exported as a feature",
        "future_price_rows_for_boundary": int(len(future_prices)) if future_prices is not None else 0,
        "test_labels_created": False,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "methods": methods,
        "best_supported_definition": supported,
        "sampled_disagreement_rows": int(len(samples)),
    }
    return report, samples
