"""Causal advanced price/volume features and direct yearly Parquet storage."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.pipeline import FEATURE_INPUTS, _hash_sum
from src.features.time_series import safe_divide

KEYS = ["ts_code", "trade_date"]


def maximum_advanced_history(settings: dict[str, Any]) -> int:
    values = [1]
    for key in (
        "rsi_windows", "atr_windows", "bollinger_windows", "obv_flow_windows",
        "mfi_windows", "vwap_windows", "trend_efficiency_windows",
        "reversal_windows", "breakout_windows", "volatility_breakout_windows",
    ):
        values.extend(int(value) for value in settings[key])
    values.extend(int(value) for value in settings["momentum_reversal"].values())
    macd = settings["macd"]
    values.append(int(macd["slow"]) + int(macd["signal"]) - 1)
    return max(values)


def _neutral_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = safe_divide(numerator, denominator)
    return result.mask(denominator.eq(0), 0.5)


def compute_advanced_features(stock: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Compute 32 advanced features from one stock's current and prior rows only."""
    missing = [column for column in FEATURE_INPUTS if column not in stock.columns]
    if missing:
        raise ValueError(f"Missing advanced feature inputs: {missing}")
    ordered = stock[FEATURE_INPUTS].sort_values("trade_date", kind="stable").reset_index(drop=True)
    high = ordered["high"].astype("float64")
    low = ordered["low"].astype("float64")
    close = ordered["close"].astype("float64")
    vol = ordered["vol"].astype("float64")
    amount = ordered["amount"].astype("float64")
    previous_close = close.shift(1)
    ret_1 = safe_divide(close, previous_close) - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    features: dict[str, pd.Series] = {}

    for value in settings["rsi_windows"]:
        window = int(value)
        average_gain = gain.rolling(window, min_periods=window).mean()
        average_loss = loss.rolling(window, min_periods=window).mean()
        features[f"rsi_{window}"] = _neutral_ratio(average_gain, average_gain + average_loss)

    macd = settings["macd"]
    fast, slow, signal = int(macd["fast"]), int(macd["slow"]), int(macd["signal"])
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_diff = safe_divide(ema_fast, ema_slow) - 1.0
    macd_signal = macd_diff.ewm(span=signal, min_periods=signal, adjust=False).mean()
    features[f"macd_diff_{fast}_{slow}"] = macd_diff
    features[f"macd_signal_{signal}"] = macd_signal
    features[f"macd_hist_{fast}_{slow}_{signal}"] = macd_diff - macd_signal

    true_range = pd.concat([
        high - low, (high - previous_close).abs(), (low - previous_close).abs()
    ], axis=1).max(axis=1, skipna=False)
    for value in settings["atr_windows"]:
        window = int(value)
        atr = true_range.rolling(window, min_periods=window).mean()
        features[f"atr_norm_{window}"] = safe_divide(atr, close)

    for value in settings["bollinger_windows"]:
        window = int(value)
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()
        features[f"bollinger_position_{window}"] = safe_divide(close - (mean - 2.0 * std), 4.0 * std)
        features[f"bollinger_width_{window}"] = safe_divide(4.0 * std, mean)

    signed_volume = np.sign(delta).fillna(0.0) * vol
    for value in settings["obv_flow_windows"]:
        window = int(value)
        numerator = signed_volume.rolling(window, min_periods=window).sum()
        denominator = vol.rolling(window, min_periods=window).sum()
        features[f"obv_flow_ratio_{window}"] = safe_divide(numerator, denominator)

    typical = (high + low + close) / 3.0
    raw_money_flow = typical * vol
    typical_direction = np.sign(typical.diff())
    for value in settings["mfi_windows"]:
        window = int(value)
        positive = raw_money_flow.where(typical_direction > 0, 0.0).rolling(window, min_periods=window).sum()
        negative = raw_money_flow.where(typical_direction < 0, 0.0).rolling(window, min_periods=window).sum()
        features[f"mfi_{window}"] = _neutral_ratio(positive, positive + negative)

    for value in settings["vwap_windows"]:
        window = int(value)
        # Raw amount/volume implies an unadjusted price, while competition OHLC
        # is adjusted. Weight adjusted close by volume to preserve price scale.
        rolling_amount = (close * vol).rolling(window, min_periods=window).sum()
        rolling_volume = vol.rolling(window, min_periods=window).sum()
        vwap = safe_divide(rolling_amount, rolling_volume)
        features[f"close_vwap_ratio_{window}"] = safe_divide(close, vwap) - 1.0

    for value in settings["trend_efficiency_windows"]:
        window = int(value)
        net_return = safe_divide(close, close.shift(window)) - 1.0
        path = ret_1.abs().rolling(window, min_periods=window).sum()
        features[f"trend_efficiency_{window}"] = safe_divide(net_return, path)

    for value in settings["reversal_windows"]:
        window = int(value)
        features[f"reversal_{window}"] = -(safe_divide(close, close.shift(window)) - 1.0)
    reversal = settings["momentum_reversal"]
    short, long = int(reversal["short"]), int(reversal["long"])
    short_return = safe_divide(close, close.shift(short)) - 1.0
    long_return = safe_divide(close, close.shift(long)) - 1.0
    features[f"momentum_reversal_{short}_{long}"] = long_return - short_return

    for value in settings["breakout_windows"]:
        window = int(value)
        prior_volume_max = vol.shift(1).rolling(window, min_periods=window).max()
        prior_amount_max = amount.shift(1).rolling(window, min_periods=window).max()
        features[f"volume_breakout_{window}"] = safe_divide(vol, prior_volume_max) - 1.0
        features[f"amount_breakout_{window}"] = safe_divide(amount, prior_amount_max) - 1.0

    for value in settings["volatility_breakout_windows"]:
        window = int(value)
        prior_volatility = ret_1.shift(1).rolling(window, min_periods=window).std()
        features[f"volatility_breakout_{window}"] = safe_divide(ret_1.abs(), prior_volatility)

    result = pd.DataFrame(features, index=ordered.index).replace([np.inf, -np.inf], np.nan)
    result = result.astype("float32")
    if len(result.columns) != 32:
        raise ValueError(f"Expected 32 advanced features, got {len(result.columns)}")
    return pd.concat([ordered[["ts_code", "trade_date"]], result], axis=1)


def advanced_feature_dictionary(settings: dict[str, Any]) -> pd.DataFrame:
    """Return explicit family, formula and lookback metadata for configured features."""
    sample = pd.DataFrame({
        "ts_code": ["X"] * (maximum_advanced_history(settings) + 2),
        "trade_date": np.arange(maximum_advanced_history(settings) + 2),
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
        "vol": 1.0, "amount": 0.1, "flag_limit_up": 0, "flag_limit_down": 0,
    })
    names = [column for column in compute_advanced_features(sample, settings) if column not in ("ts_code", "trade_date")]
    formulas = {
        "rsi": "rolling_mean(gain)/(rolling_mean(gain)+rolling_mean(loss))",
        "macd_diff": "EMA_fast(close)/EMA_slow(close)-1",
        "macd_signal": "EMA_signal(macd_diff)",
        "macd_hist": "macd_diff-macd_signal",
        "atr_norm": "rolling_mean(true_range)/close",
        "bollinger_position": "(close-(MA-2*STD))/(4*STD)",
        "bollinger_width": "4*STD/MA",
        "obv_flow_ratio": "rolling_sum(sign(delta_close)*vol)/rolling_sum(vol)",
        "mfi": "positive_money_flow/(positive_money_flow+negative_money_flow)",
        "close_vwap_ratio": "close/(rolling_sum(close*vol)/rolling_sum(vol))-1",
        "trend_efficiency": "return_window/rolling_sum(abs(ret_1))",
        "reversal": "-return_window",
        "momentum_reversal": "long_return-short_return",
        "volume_breakout": "vol/prior_rolling_max(vol)-1",
        "amount_breakout": "amount/prior_rolling_max(amount)-1",
        "volatility_breakout": "abs(ret_1)/prior_rolling_std(ret_1)",
    }
    rows = []
    for name in names:
        key = next(prefix for prefix in formulas if name.startswith(prefix))
        numbers = [int(value) for value in name.split("_") if value.isdigit()]
        rows.append({
            "feature": name, "family": key, "formula": formulas[key],
            "max_lookback_rows": max(numbers) if numbers else maximum_advanced_history(settings),
            "uses_current_row": True, "uses_future_data": False,
            "min_periods_policy": settings["min_periods_policy"],
        })
    return pd.DataFrame(rows)


def build_advanced_year_store(
    base: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
    *,
    overwrite: bool = False,
    output_start_date: int | None = None,
    max_stocks: int | None = None,
) -> dict[str, Any]:
    """Compute stock features and write yearly partitions without a monolithic copy."""
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Advanced feature output exists; pass overwrite: {output_dir}")
    source = base[FEATURE_INPUTS].sort_values(["ts_code", "trade_date"], kind="stable")
    if max_stocks is not None:
        selected = source["ts_code"].drop_duplicates().head(max_stocks)
        source = source[source["ts_code"].isin(selected)]
    expected = source if output_start_date is None else source[source["trade_date"] >= output_start_date]
    expected_hash = _hash_sum(expected[["ts_code", "trade_date"]])
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    writers: dict[int, pq.ParquetWriter] = {}
    row_counts: Counter[int] = Counter()
    key_hashes: Counter[int] = Counter()
    null_counts: Counter[str] = Counter()
    inf_counts: Counter[str] = Counter()
    feature_names: list[str] = []
    grouped = iter(source.groupby("ts_code", sort=False, observed=True))
    try:
        with ThreadPoolExecutor(max_workers=int(settings["feature_workers"])) as executor:
            while True:
                stocks = [stock for _, stock in islice(grouped, int(settings["stocks_per_write_batch"]))]
                if not stocks:
                    break
                frames = list(executor.map(lambda stock: compute_advanced_features(stock, settings), stocks))
                if not feature_names:
                    feature_names = [c for c in frames[0] if c not in ("ts_code", "trade_date")]
                batch = pd.concat(frames, ignore_index=True)
                if output_start_date is not None:
                    batch = batch[batch["trade_date"] >= output_start_date].reset_index(drop=True)
                if batch.empty:
                    continue
                values = batch[feature_names]
                null_counts.update({name: int(value) for name, value in values.isna().sum().items()})
                inf_counts.update({
                    name: int(np.isinf(values[name].to_numpy(dtype="float64", na_value=np.nan)).sum())
                    for name in feature_names
                })
                years = (batch["trade_date"] // 10000).astype("int32")
                for year in sorted(int(value) for value in years.unique()):
                    part = batch.loc[years == year].reset_index(drop=True)
                    table = pa.Table.from_pandas(part, preserve_index=False)
                    year_dir = temporary / f"year={year}"
                    year_dir.mkdir(parents=True, exist_ok=True)
                    if year not in writers:
                        writers[year] = pq.ParquetWriter(
                            year_dir / "features.parquet", table.schema,
                            compression=str(settings["parquet_compression"]),
                        )
                    writers[year].write_table(table)
                    row_counts[year] += len(part)
                    key_hashes[year] = (key_hashes[year] + _hash_sum(part[KEYS])) & ((1 << 64) - 1)
        for writer in writers.values():
            writer.close()
        writers.clear()

        years_report: dict[str, Any] = {}
        total_rows = 0
        total_hash = 0
        all_unique = True
        schema: list[tuple[str, str]] | None = None
        for year in sorted(row_counts):
            path = temporary / f"year={year}" / "features.parquet"
            metadata = pq.read_metadata(path)
            keys = pd.read_parquet(path, columns=KEYS, engine="pyarrow")
            duplicate_rows = int(keys.duplicated(KEYS, keep=False).sum())
            current_schema = [
                (field.name, str(field.type))
                for field in metadata.schema.to_arrow_schema()
            ]
            num_row_groups = metadata.num_row_groups
            del metadata
            schema = schema or current_schema
            if current_schema != schema:
                raise ValueError("Advanced yearly schemas differ")
            total_rows += len(keys)
            total_hash = (total_hash + key_hashes[year]) & ((1 << 64) - 1)
            all_unique = all_unique and duplicate_rows == 0
            years_report[str(year)] = {
                "rows": len(keys), "row_groups": num_row_groups,
                "bytes": path.stat().st_size, "key_hash": f"{key_hashes[year]:016x}",
                "duplicate_key_rows": duplicate_rows,
            }
        report = {
            "version": settings["version"], "input_rows": int(len(source)),
            "warmup_rows": int(len(source) - len(expected)), "rows": total_rows,
            "stocks": int(expected["ts_code"].nunique()), "dates": int(expected["trade_date"].nunique()),
            "date_range": [int(expected["trade_date"].min()), int(expected["trade_date"].max())],
            "feature_count": len(feature_names), "feature_names": feature_names,
            "source_key_hash": f"{expected_hash:016x}", "output_key_hash": f"{total_hash:016x}",
            "key_hash_equal": expected_hash == total_hash, "all_keys_unique": all_unique,
            "schema": schema, "years": years_report,
            "null_count": dict(null_counts), "inf_count": dict(inf_counts),
            "finite_ratio": {
                name: float((total_rows - null_counts[name] - inf_counts[name]) / total_rows)
                for name in feature_names
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if output_dir.exists():
            backup = output_dir.with_name(f"{output_dir.name}.backup")
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(output_dir, backup)
            try:
                os.replace(temporary, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output_dir)
        return report
    except Exception:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
