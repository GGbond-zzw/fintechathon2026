from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.features.market import compute_market_features
from src.features.time_series import compute_stock_features


SETTINGS = {
    "return_windows": [1, 2, 3, 5, 10, 20, 40, 60, 120],
    "volatility_windows": [5, 10, 20, 60],
    "volume_windows": [5, 10, 20, 60],
    "position_windows": [5, 10, 20, 60, 120],
    "ma_pairs": [[5, 20], [10, 20], [20, 60], [40, 120]],
}


def stock_frame(rows: int = 130, code: str = "A") -> pd.DataFrame:
    index = np.arange(rows, dtype="float64")
    close = 10.0 + index * 0.1
    return pd.DataFrame({
        "ts_code": [code] * rows,
        "trade_date": np.arange(20200101, 20200101 + rows),
        "open": close - 0.02,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "vol": 1000.0 + index * 10.0,
        "amount": close * (1000.0 + index * 10.0),
        "flag_limit_up": np.zeros(rows, dtype="int8"),
        "flag_limit_down": np.zeros(rows, dtype="int8"),
    })


class FeatureTest(TestCase):
    def test_feature_count_and_hand_formulas(self) -> None:
        source = stock_frame()
        result = compute_stock_features(source, SETTINGS)
        feature_columns = [c for c in result.columns if c not in ("ts_code", "trade_date")]
        self.assertGreaterEqual(len(feature_columns), 100)
        self.assertLessEqual(len(feature_columns), 300)
        self.assertAlmostEqual(float(result.loc[1, "ret_1"]), source.loc[1, "close"] / source.loc[0, "close"] - 1.0, places=7)
        self.assertAlmostEqual(float(result.loc[4, "position_5"]), (source.loc[4, "close"] - source.loc[:4, "low"].min()) / (source.loc[:4, "high"].max() - source.loc[:4, "low"].min()), places=7)

    def test_appending_future_rows_does_not_change_past_features(self) -> None:
        full = stock_frame()
        prefix = full.iloc[:100].copy()
        future_changed = full.copy()
        future_changed.loc[100:, ["open", "high", "low", "close", "vol", "amount"]] *= 1000.0
        before = compute_stock_features(prefix, SETTINGS)
        after = compute_stock_features(future_changed, SETTINGS).iloc[:100].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_market_features_do_not_mix_dates_or_use_future(self) -> None:
        a = stock_frame(3, "A")
        b = stock_frame(3, "B")
        b[["open", "high", "low", "close"]] *= 2.0
        base = pd.concat([a, b], ignore_index=True)
        first = compute_market_features(base.iloc[[0, 1, 2, 3, 4, 5]])
        changed = base.copy()
        changed.loc[changed["trade_date"] == changed["trade_date"].max(), "close"] *= 10.0
        second = compute_market_features(changed)
        pd.testing.assert_frame_equal(first.iloc[:-1].reset_index(drop=True), second.iloc[:-1].reset_index(drop=True))
        self.assertTrue(pd.isna(first.loc[0, "market_up_ratio"]))
        self.assertTrue(pd.isna(first.loc[0, "market_down_ratio"]))
