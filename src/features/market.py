"""Same-day market context features built only from allowed competition fields."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["ts_code", "trade_date", "close", "amount", "flag_limit_up", "flag_limit_down"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing market feature inputs: {missing}")
    work = frame[required].sort_values(["ts_code", "trade_date"], kind="stable").copy()
    previous_close = work.groupby("ts_code", sort=False, observed=True)["close"].shift(1)
    work["ret_1"] = work["close"] / previous_close.where(previous_close != 0) - 1.0
    valid_return = work["ret_1"].notna()
    work["is_up"] = (work["ret_1"] > 0).astype("float32").where(valid_return)
    work["is_down"] = (work["ret_1"] < 0).astype("float32").where(valid_return)
    grouped = work.groupby("trade_date", sort=True, observed=True)
    market = grouped.agg(
        market_count=("ts_code", "size"),
        market_valid_return_count=("ret_1", "count"),
        market_mean_ret_1=("ret_1", "mean"),
        market_median_ret_1=("ret_1", "median"),
        market_std_ret_1=("ret_1", "std"),
        market_up_ratio=("is_up", "mean"),
        market_down_ratio=("is_down", "mean"),
        market_limit_up_ratio=("flag_limit_up", "mean"),
        market_limit_down_ratio=("flag_limit_down", "mean"),
        market_amount=("amount", "sum"),
        market_median_amount=("amount", "median"),
    ).reset_index()
    numeric = [column for column in market.columns if column != "trade_date"]
    market[numeric] = market[numeric].replace([np.inf, -np.inf], np.nan).astype("float32")
    market["trade_date"] = market["trade_date"].astype("int32")
    return market
