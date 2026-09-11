"""Dataset integrity checks that do not mutate input frames."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _json_number(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _key_hash(frame: pd.DataFrame, id_cols: list[str]) -> str:
    hashed = pd.util.hash_pandas_object(frame[id_cols], index=False)
    return f"{int(hashed.sum()) & ((1 << 64) - 1):016x}"


def validate_frame(frame: pd.DataFrame, *, dataset_name: str, required_columns: Iterable[str], id_cols: list[str]) -> dict[str, Any]:
    """Return a JSON-serializable integrity report."""
    required = list(required_columns)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        return {"dataset": dataset_name, "status": "invalid_schema", "missing_columns": missing_columns}
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    null_counts = frame.isna().sum()
    inf_counts = {column: int(np.isinf(frame[column].to_numpy(dtype="float64", na_value=np.nan)).sum()) for column in numeric_columns}
    duplicate_rows = int(frame.duplicated(id_cols, keep=False).sum())
    daily = frame.groupby("trade_date", sort=True, observed=True).size()
    prices = ["open", "high", "low", "close"]
    complete_prices = frame[prices].notna().all(axis=1)
    high_bound = frame["high"] >= frame[["open", "low", "close"]].max(axis=1)
    low_bound = frame["low"] <= frame[["open", "high", "close"]].min(axis=1)
    violations = {
        "ohlc": int((complete_prices & ~(high_bound & low_bound)).sum()),
        "negative_vol": int((frame["vol"].dropna() < 0).sum()),
        "negative_amount": int((frame["amount"].dropna() < 0).sum()),
        "invalid_limit_up": int((~frame["flag_limit_up"].dropna().isin([0, 1])).sum()),
        "invalid_limit_down": int((~frame["flag_limit_down"].dropna().isin([0, 1])).sum()),
    }
    sorted_copy = frame.sort_values(id_cols, kind="stable")[id_cols].reset_index(drop=True)
    checks = {
        "required_columns_present": True,
        "key_has_no_null": not bool(frame[id_cols].isna().any().any()),
        "key_unique": duplicate_rows == 0,
        "sorted_by_key": frame[id_cols].reset_index(drop=True).equals(sorted_copy),
        "all_numeric_values_finite_or_missing": sum(inf_counts.values()) == 0,
        "ohlc_consistent": violations["ohlc"] == 0,
        "nonnegative_volume": violations["negative_vol"] == 0,
        "nonnegative_amount": violations["negative_amount"] == 0,
        "limit_up_binary": violations["invalid_limit_up"] == 0,
        "limit_down_binary": violations["invalid_limit_down"] == 0,
    }
    return {
        "dataset": dataset_name, "status": "valid" if all(checks.values()) else "issues_found",
        "shape": [int(frame.shape[0]), int(frame.shape[1])], "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "date_range": [_json_number(frame["trade_date"].min()), _json_number(frame["trade_date"].max())],
        "stock_count": int(frame["ts_code"].nunique(dropna=True)), "key_hash": _key_hash(frame, id_cols),
        "duplicate_key_rows": duplicate_rows,
        "missing_count": {column: int(null_counts[column]) for column in frame.columns},
        "missing_rate": {column: float(null_counts[column] / len(frame)) if len(frame) else None for column in frame.columns},
        "inf_count": inf_counts,
        "daily_stock_count": {"trading_days": int(len(daily)), "min": _json_number(daily.min()), "median": _json_number(daily.median()), "max": _json_number(daily.max())},
        "violations": violations, "checks": checks,
    }


def compare_cache_consistency(raw_frame: pd.DataFrame, cached_frame: pd.DataFrame, *, id_cols: list[str]) -> dict[str, Any]:
    """Compare row/key identity and critical per-column statistics."""
    common_numeric = [column for column in raw_frame.select_dtypes(include=[np.number]).columns if column in cached_frame.columns]
    numeric_stats_equal = {
        column: bool(raw_frame[column].min(skipna=True) == cached_frame[column].min(skipna=True) and raw_frame[column].max(skipna=True) == cached_frame[column].max(skipna=True))
        for column in common_numeric
    }
    checks = {
        "row_count_equal": len(raw_frame) == len(cached_frame),
        "columns_equal": list(raw_frame.columns) == list(cached_frame.columns),
        "key_hash_equal": _key_hash(raw_frame, id_cols) == _key_hash(cached_frame, id_cols),
        "missing_counts_equal": raw_frame.isna().sum().equals(cached_frame.isna().sum()),
        "numeric_min_max_equal": all(numeric_stats_equal.values()),
    }
    return {"status": "consistent" if all(checks.values()) else "mismatch", "checks": checks, "numeric_stats_equal": numeric_stats_equal}

