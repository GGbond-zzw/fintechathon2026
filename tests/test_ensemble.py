from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.models.ensemble import blend_daily_rank_predictions


def model_frame(predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["A", "B", "C", "A", "B", "C"],
        "trade_date": [20260101] * 3 + [20260102] * 3,
        "pred": predictions,
    })


class EnsembleTest(TestCase):
    def test_endpoints_equal_each_models_daily_rank(self) -> None:
        left = model_frame([3, 1, 2, 1, 3, 2])
        right = model_frame([1, 2, 3, 3, 2, 1])
        left_only = blend_daily_rank_predictions(
            {"lightgbm": left, "ridge": right}, {"lightgbm": 1.0, "ridge": 0.0}
        )
        expected = left.groupby("trade_date")["pred"].rank(method="average", pct=True)
        np.testing.assert_array_equal(left_only["pred"], expected)

    def test_weighted_blend_uses_same_date_ranks_and_is_scale_invariant(self) -> None:
        left = model_frame([3, 1, 2, 1, 3, 2])
        right = model_frame([1, 2, 3, 3, 2, 1])
        original = blend_daily_rank_predictions(
            {"lightgbm": left, "ridge": right}, {"lightgbm": 0.6, "ridge": 0.4}
        )
        scaled = left.copy()
        scaled["pred"] = scaled["pred"] * 1000 + 77
        changed = blend_daily_rank_predictions(
            {"lightgbm": scaled, "ridge": right}, {"lightgbm": 0.6, "ridge": 0.4}
        )
        np.testing.assert_array_equal(original["pred"], changed["pred"])

    def test_future_date_change_does_not_change_past_blend(self) -> None:
        left = model_frame([3, 1, 2, 1, 3, 2])
        right = model_frame([1, 2, 3, 3, 2, 1])
        original = blend_daily_rank_predictions(
            {"lightgbm": left, "ridge": right}, {"lightgbm": 0.5, "ridge": 0.5}
        )
        changed = right.copy()
        changed.loc[changed["trade_date"] == 20260102, "pred"] *= -100
        later = blend_daily_rank_predictions(
            {"lightgbm": left, "ridge": changed}, {"lightgbm": 0.5, "ridge": 0.5}
        )
        np.testing.assert_array_equal(original.loc[:2, "pred"], later.loc[:2, "pred"])

    def test_rejects_bad_weights_and_misaligned_keys(self) -> None:
        left = model_frame([3, 1, 2, 1, 3, 2])
        right = model_frame([1, 2, 3, 3, 2, 1])
        with self.assertRaises(ValueError):
            blend_daily_rank_predictions(
                {"lightgbm": left, "ridge": right}, {"lightgbm": 0.7, "ridge": 0.4}
            )
        shifted = right.copy()
        shifted.loc[0, "ts_code"] = "Z"
        with self.assertRaises(ValueError):
            blend_daily_rank_predictions(
                {"lightgbm": left, "ridge": shifted}, {"lightgbm": 0.5, "ridge": 0.5}
            )
