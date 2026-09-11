from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from src.features.inference_pipeline import maximum_required_history
from src.features.market import compute_market_features
from src.features.pipeline import build_time_feature_store


SETTINGS = {
    "return_windows": [1, 5, 20, 120],
    "volatility_windows": [5, 20, 60],
    "volume_windows": [5, 20, 60],
    "position_windows": [5, 20, 120],
    "ma_pairs": [[5, 20], [40, 120]],
    "stocks_per_write_batch": 2,
    "feature_workers": 1,
    "parquet_compression": "zstd",
}


def stock_frame(code: str, rows: int = 130) -> pd.DataFrame:
    index = np.arange(rows, dtype="float64")
    close = 10.0 + 0.05 * index
    return pd.DataFrame({
        "ts_code": [code] * rows,
        "trade_date": np.arange(20200001, 20200001 + rows, dtype="int32"),
        "open": close - 0.01,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "vol": 1000.0 + index,
        "amount": close * (1000.0 + index),
        "flag_limit_up": np.zeros(rows, dtype="int8"),
        "flag_limit_down": np.zeros(rows, dtype="int8"),
    })


class InferenceFeatureTest(TestCase):
    def test_maximum_required_history_covers_all_window_families(self) -> None:
        self.assertEqual(maximum_required_history(SETTINGS), 120)
        changed = dict(SETTINGS, ma_pairs=[[5, 250]])
        self.assertEqual(maximum_required_history(changed), 250)

    def test_warmup_rows_are_used_but_not_written(self) -> None:
        source = pd.concat([stock_frame("A"), stock_frame("B")], ignore_index=True)
        output_start = 20200121
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.parquet"
            report = build_time_feature_store(
                source, path, SETTINGS, output_start_date=output_start
            )
            result = pd.read_parquet(path)
        expected = source[source["trade_date"] >= output_start]
        self.assertEqual(len(result), len(expected))
        self.assertEqual(report["warmup_rows"], len(source) - len(expected))
        self.assertGreaterEqual(int(result["trade_date"].min()), output_start)
        first_a = result[(result["ts_code"] == "A") & (result["trade_date"] == output_start)].iloc[0]
        original_a = source[source["ts_code"] == "A"].reset_index(drop=True)
        row = int(original_a.index[original_a["trade_date"] == output_start][0])
        expected_ret = original_a.loc[row, "close"] / original_a.loc[row - 1, "close"] - 1.0
        self.assertAlmostEqual(float(first_a["ret_1"]), expected_ret, places=7)
        self.assertTrue(pd.notna(first_a["ret_20"]))

    def test_market_first_test_date_uses_last_training_close(self) -> None:
        a = stock_frame("A", 3)
        b = stock_frame("B", 3)
        combined = pd.concat([a, b], ignore_index=True)
        market = compute_market_features(combined)
        first_test = int(a.iloc[2]["trade_date"])
        test_market = market[market["trade_date"] >= first_test].reset_index(drop=True)
        self.assertEqual(int(test_market.iloc[0]["market_valid_return_count"]), 2)
        self.assertTrue(pd.notna(test_market.iloc[0]["market_mean_ret_1"]))
