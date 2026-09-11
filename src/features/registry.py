"""Feature dictionary generation from deterministic naming conventions."""

from __future__ import annotations

import re

import pandas as pd


def _describe(name: str) -> tuple[str, str, str]:
    fixed = {
        "intraday_return": ("price", "close/open - 1", "0"),
        "overnight_gap": ("price", "open/close[t-1] - 1", "1"),
        "high_low_range": ("price", "(high-low)/close", "0"),
        "upper_shadow": ("price", "(high-max(open,close))/close", "0"),
        "lower_shadow": ("price", "(min(open,close)-low)/close", "0"),
        "close_position_in_bar": ("price", "(close-low)/(high-low)", "0"),
        "open_to_high": ("price", "high/open - 1", "0"),
        "open_to_low": ("price", "open/low - 1", "0"),
        "close_to_high": ("price", "high/close - 1", "0"),
        "close_to_low": ("price", "close/low - 1", "0"),
        "volume_change_1": ("volume", "vol/vol[t-1] - 1", "1"),
        "amount_change_1": ("volume", "amount/amount[t-1] - 1", "1"),
        "amihud_illiq_1": ("liquidity", "abs(ret_1)/amount", "1"),
    }
    if name in fixed:
        return fixed[name]
    numbers = [int(value) for value in re.findall(r"\d+", name)]
    lookback = str(max(numbers)) if numbers else "0"
    if name.startswith(("ret_mean", "ret_std", "ret_mean_abs", "downside", "upside", "ret_skew", "ret_kurt")):
        family = "volatility"
    elif name.startswith(("ret_", "log_ret_", "close_ma", "close_ema")):
        family = "momentum_trend"
    elif name.startswith(("vol_", "amount_", "corr_ret")):
        family = "volume"
    elif name.startswith("amihud"):
        family = "liquidity"
    elif name.startswith(("position_", "distance_", "breakout_", "range_width_")):
        family = "price_position"
    elif name.startswith("market_"):
        family = "market"
    else:
        family = "other"
    return family, name.replace("_", " "), lookback


def build_feature_dictionary(feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in feature_names:
        family, formula, lookback = _describe(name)
        rows.append({
            "feature": name,
            "family": family,
            "formula_or_definition": formula,
            "max_lookback_rows": lookback,
            "uses_current_row": True,
            "uses_future_data": False,
            "min_periods_policy": "full_window",
        })
    return pd.DataFrame(rows)
