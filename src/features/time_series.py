"""Causal per-stock price, momentum, volatility, volume and liquidity features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype("float64") / denominator.astype("float64").where(denominator != 0)
    return result.replace([np.inf, -np.inf], np.nan)


def _rolling(series: pd.Series, window: int) -> Any:
    return series.rolling(window=window, min_periods=window)


def _causal_skew_kurt(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Population skew/kurtosis from window-local raw moments only."""
    mean_1 = _rolling(series, window).mean()
    mean_2 = _rolling(series.pow(2), window).mean()
    mean_3 = _rolling(series.pow(3), window).mean()
    mean_4 = _rolling(series.pow(4), window).mean()
    central_2 = (mean_2 - mean_1.pow(2)).clip(lower=0.0)
    central_3 = mean_3 - 3.0 * mean_1 * mean_2 + 2.0 * mean_1.pow(3)
    central_4 = mean_4 - 4.0 * mean_1 * mean_3 + 6.0 * mean_1.pow(2) * mean_2 - 3.0 * mean_1.pow(4)
    valid_variance = central_2.where(central_2 > 1e-20)
    skew = safe_divide(central_3, valid_variance.pow(1.5))
    kurt = safe_divide(central_4, valid_variance.pow(2)) - 3.0
    return skew, kurt


def compute_stock_features(stock: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Compute features for one stock using only its current and previous rows."""
    required = [
        "ts_code", "trade_date", "open", "high", "low", "close",
        "vol", "amount", "flag_limit_up", "flag_limit_down",
    ]
    missing = [column for column in required if column not in stock.columns]
    if missing:
        raise ValueError(f"Missing feature inputs: {missing}")
    ordered = stock.sort_values("trade_date", kind="stable").reset_index(drop=True)
    open_ = ordered["open"].astype("float64")
    high = ordered["high"].astype("float64")
    low = ordered["low"].astype("float64")
    close = ordered["close"].astype("float64")
    vol = ordered["vol"].astype("float64")
    amount = ordered["amount"].astype("float64")
    previous_close = close.shift(1)
    ret_1 = safe_divide(close, previous_close) - 1.0
    vol_change = safe_divide(vol, vol.shift(1)) - 1.0
    amount_change = safe_divide(amount, amount.shift(1)) - 1.0
    bar_range = high - low
    features: dict[str, pd.Series] = {
        "intraday_return": safe_divide(close, open_) - 1.0,
        "overnight_gap": safe_divide(open_, previous_close) - 1.0,
        "high_low_range": safe_divide(bar_range, close),
        "upper_shadow": safe_divide(high - pd.concat([open_, close], axis=1).max(axis=1), close),
        "lower_shadow": safe_divide(pd.concat([open_, close], axis=1).min(axis=1) - low, close),
        "close_position_in_bar": safe_divide(close - low, bar_range),
        "open_to_high": safe_divide(high, open_) - 1.0,
        "open_to_low": safe_divide(open_, low) - 1.0,
        "close_to_high": safe_divide(high, close) - 1.0,
        "close_to_low": safe_divide(close, low) - 1.0,
        "volume_change_1": vol_change,
        "amount_change_1": amount_change,
    }
    amihud = safe_divide(ret_1.abs(), amount)
    features["amihud_illiq_1"] = amihud

    for value in settings["return_windows"]:
        window = int(value)
        lag_close = close.shift(window)
        ma = _rolling(close, window).mean()
        ema = close.ewm(span=window, min_periods=window, adjust=False).mean()
        features[f"ret_{window}"] = safe_divide(close, lag_close) - 1.0
        features[f"log_ret_{window}"] = np.log(close.where(close > 0)) - np.log(lag_close.where(lag_close > 0))
        features[f"close_ma_ratio_{window}"] = safe_divide(close, ma) - 1.0
        features[f"close_ema_ratio_{window}"] = safe_divide(close, ema) - 1.0

    negative_squared = ret_1.clip(upper=0.0).pow(2)
    positive_squared = ret_1.clip(lower=0.0).pow(2)
    for value in settings["volatility_windows"]:
        window = int(value)
        returns = _rolling(ret_1, window)
        skew, kurt = _causal_skew_kurt(ret_1, window)
        features[f"ret_mean_{window}"] = returns.mean()
        features[f"ret_std_{window}"] = returns.std()
        features[f"ret_mean_abs_{window}"] = _rolling(ret_1.abs(), window).mean()
        features[f"downside_rms_{window}"] = np.sqrt(_rolling(negative_squared, window).mean())
        features[f"upside_rms_{window}"] = np.sqrt(_rolling(positive_squared, window).mean())
        features[f"ret_skew_{window}"] = skew
        features[f"ret_kurt_{window}"] = kurt

    for value in settings["volume_windows"]:
        window = int(value)
        vol_roll = _rolling(vol, window)
        amount_roll = _rolling(amount, window)
        vol_mean = vol_roll.mean()
        amount_mean = amount_roll.mean()
        features[f"vol_ma_ratio_{window}"] = safe_divide(vol, vol_mean)
        features[f"vol_cv_{window}"] = safe_divide(vol_roll.std(), vol_mean)
        features[f"amount_ma_ratio_{window}"] = safe_divide(amount, amount_mean)
        features[f"amount_cv_{window}"] = safe_divide(amount_roll.std(), amount_mean)
        features[f"amihud_mean_{window}"] = _rolling(amihud, window).mean()
        features[f"corr_ret_vol_change_{window}"] = _rolling(ret_1, window).corr(vol_change)
        features[f"corr_ret_amount_change_{window}"] = _rolling(ret_1, window).corr(amount_change)

    for value in settings["position_windows"]:
        window = int(value)
        rolling_high = _rolling(high, window).max()
        rolling_low = _rolling(low, window).min()
        prior_high = _rolling(high.shift(1), window).max()
        prior_low = _rolling(low.shift(1), window).min()
        range_width = rolling_high - rolling_low
        features[f"position_{window}"] = safe_divide(close - rolling_low, range_width)
        features[f"distance_to_high_{window}"] = safe_divide(close, rolling_high) - 1.0
        features[f"distance_to_low_{window}"] = safe_divide(close, rolling_low) - 1.0
        features[f"breakout_high_{window}"] = safe_divide(close, prior_high) - 1.0
        features[f"breakout_low_{window}"] = safe_divide(close, prior_low) - 1.0
        features[f"range_width_{window}"] = safe_divide(range_width, close)

    for short_value, long_value in settings["ma_pairs"]:
        short, long = int(short_value), int(long_value)
        close_short, close_long = _rolling(close, short).mean(), _rolling(close, long).mean()
        vol_short, vol_long = _rolling(vol, short).mean(), _rolling(vol, long).mean()
        amount_short, amount_long = _rolling(amount, short).mean(), _rolling(amount, long).mean()
        features[f"close_ma_{short}_{long}"] = safe_divide(close_short, close_long) - 1.0
        features[f"vol_ma_{short}_{long}"] = safe_divide(vol_short, vol_long) - 1.0
        features[f"amount_ma_{short}_{long}"] = safe_divide(amount_short, amount_long) - 1.0

    feature_frame = pd.DataFrame(features, index=ordered.index)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).astype("float32")
    return pd.concat([ordered[["ts_code", "trade_date"]], feature_frame], axis=1)
