from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.features.screening import aggregate_daily_metrics, daily_rank_ic, select_features


class FeatureScreeningTest(TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for date, offset in ((20200102, 0.0), (20200103, 100.0)):
            for index in range(40):
                rows.append({
                    "ts_code": f"S{index:03d}", "trade_date": date,
                    "y_ret_1d": float(index), "good": float(index) + offset,
                    "bad": -float(index) + offset,
                    "market_mean_ret_1": 0.01, "market_std_ret_1": 0.02,
                })
        frame = pd.DataFrame(rows)
        frame.loc[0, "good"] = np.nan
        return frame

    def test_daily_rank_ic_direction_and_neutral_missing_policy(self) -> None:
        daily, corr_sum, corr_count = daily_rank_ic(
            self._frame(), ["good", "bad"], label="y_ret_1d",
            min_samples=30, missing_rank_fill=0.5, collect_rank_correlation=True,
        )
        good = daily[daily["feature"] == "good"]["ic"]
        bad = daily[daily["feature"] == "bad"]["ic"]
        self.assertTrue((good > 0.95).all())
        self.assertLess(float(good.iloc[0]), 1.0)
        self.assertTrue((bad < -0.99).all())
        self.assertEqual(corr_sum.shape, (2, 2))
        self.assertTrue((corr_count > 0).all())

    def test_future_date_does_not_change_prior_daily_metrics(self) -> None:
        original = self._frame()
        before, _, _ = daily_rank_ic(
            original[original["trade_date"] == 20200102], ["good", "bad"],
            label="y_ret_1d", min_samples=30, missing_rank_fill=0.5,
        )
        changed = original.copy()
        changed.loc[changed["trade_date"] == 20200103, ["good", "bad", "y_ret_1d"]] *= -1000
        after, _, _ = daily_rank_ic(
            changed, ["good", "bad"], label="y_ret_1d",
            min_samples=30, missing_rank_fill=0.5,
        )
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after[after["trade_date"] == 20200102].reset_index(drop=True))

    def test_selection_uses_only_passed_screen_metrics(self) -> None:
        daily, _, _ = daily_rank_ic(
            self._frame(), ["good", "bad"], label="y_ret_1d",
            min_samples=30, missing_rank_fill=0.5,
        )
        screen = aggregate_daily_metrics(daily, "screen")
        yearly = pd.concat([
            aggregate_daily_metrics(daily, f"year_{year}").assign(year=year)
            for year in (2018, 2019, 2020, 2021)
        ], ignore_index=True)
        corr = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], index=["good", "bad"], columns=["good", "bad"])
        settings = {
            "screen_start": 20180102, "screen_end": 20211231,
            "min_coverage": 0.9, "min_abs_ic_mean": 0.1, "min_abs_icir": 0.1,
            "min_aligned_positive_ratio": 0.5, "min_consistent_screen_years": 3,
            "max_abs_rank_correlation": 0.9, "max_selected_per_family": 1,
            "max_selected_total": 2,
        }
        selected = select_features(screen, yearly, {"good": "a", "bad": "b"}, corr, settings)
        self.assertEqual(len(selected["selected_features"]), 1)
        self.assertFalse(selected["holdout_metrics_used_for_selection"])
        chosen = selected["selected_features"][0]
        self.assertEqual(selected["directions"][chosen], 1 if chosen == "good" else -1)
