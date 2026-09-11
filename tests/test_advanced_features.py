from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from src.features.advanced import (
    build_advanced_year_store, compute_advanced_features, maximum_advanced_history,
)


SETTINGS = {
    "version": "test", "rsi_windows": [6, 12, 24],
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "atr_windows": [14, 28], "bollinger_windows": [20, 60],
    "obv_flow_windows": [5, 20, 60], "mfi_windows": [14, 28],
    "vwap_windows": [5, 20, 60], "trend_efficiency_windows": [10, 20, 60],
    "reversal_windows": [1, 5], "momentum_reversal": {"short": 5, "long": 20},
    "breakout_windows": [20, 60], "volatility_breakout_windows": [20, 60],
    "min_periods_policy": "full_window",
    "stocks_per_write_batch": 2, "feature_workers": 1, "parquet_compression": "zstd",
}


def stock_frame(code: str = "A", rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows, dtype="float64")
    close = 10.0 + 0.05 * index + 0.02 * np.sin(index)
    vol = 1000.0 + 5.0 * index
    return pd.DataFrame({
        "ts_code": [code] * rows,
        "trade_date": np.where(index < 70, 20200101 + index, 20210101 + index - 70).astype("int32"),
        "open": close - 0.01, "high": close + 0.10, "low": close - 0.10,
        "close": close, "vol": vol, "amount": close * vol / 10.0,
        "flag_limit_up": np.zeros(rows, dtype="int8"),
        "flag_limit_down": np.zeros(rows, dtype="int8"),
    })


class AdvancedFeatureTest(TestCase):
    def test_feature_count_history_and_finite_contract(self) -> None:
        result = compute_advanced_features(stock_frame(), SETTINGS)
        features = [c for c in result if c not in ("ts_code", "trade_date")]
        self.assertEqual(len(features), 32)
        self.assertEqual(maximum_advanced_history(SETTINGS), 60)
        self.assertEqual(int(np.isinf(result[features].to_numpy(dtype="float64", na_value=np.nan)).sum()), 0)

    def test_selected_formulas(self) -> None:
        source = stock_frame()
        result = compute_advanced_features(source, SETTINGS)
        row = 80
        close, vol = source["close"], source["vol"]
        expected_vwap = (close.iloc[row-19:row+1] * vol.iloc[row-19:row+1]).sum() / vol.iloc[row-19:row+1].sum()
        self.assertAlmostEqual(float(result.loc[row, "close_vwap_ratio_20"]), close.iloc[row] / expected_vwap - 1.0, places=7)
        self.assertAlmostEqual(float(result.loc[row, "volume_breakout_20"]), vol.iloc[row] / vol.iloc[row-20:row].max() - 1.0, places=7)

    def test_appending_changed_future_does_not_change_past(self) -> None:
        source = stock_frame()
        prefix = source.iloc[:80].copy()
        changed = source.copy()
        changed.loc[80:, ["open", "high", "low", "close", "vol", "amount"]] *= 1000.0
        before = compute_advanced_features(prefix, SETTINGS)
        after = compute_advanced_features(changed, SETTINGS).iloc[:80].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_year_store_preserves_keys_and_warmup_rows(self) -> None:
        source = pd.concat([stock_frame("A"), stock_frame("B")], ignore_index=True)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "advanced"
            report = build_advanced_year_store(
                source, output, SETTINGS, output_start_date=20200121
            )
            files = sorted(output.glob("year=*/features.parquet"))
            result = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        expected = source[source["trade_date"] >= 20200121]
        self.assertEqual(len(result), len(expected))
        self.assertTrue(report["key_hash_equal"])
        self.assertTrue(report["all_keys_unique"])
        self.assertEqual(report["warmup_rows"], len(source) - len(expected))
