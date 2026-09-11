from unittest import TestCase

import pandas as pd

from src.features.cross_sectional import transform_cross_sectional


class CrossSectionalTest(TestCase):
    def test_dates_are_independent_and_ties_are_average(self) -> None:
        frame = pd.DataFrame({
            "ts_code": ["A", "B", "C", "A", "B", "C"],
            "trade_date": [20240102] * 3 + [20240103] * 3,
            "factor": [1.0, 1.0, 3.0, 100.0, 200.0, 300.0],
        })
        output, audit = transform_cross_sectional(
            frame, ["factor"], min_group_size=3, zscore_ddof=0,
            zero_variance_zscore=0.0,
        )
        first = output[output["trade_date"] == 20240102]
        self.assertAlmostEqual(float(first.iloc[0]["factor__rank"]), 0.5)
        self.assertAlmostEqual(float(first.iloc[1]["factor__rank"]), 0.5)
        self.assertAlmostEqual(float(first.iloc[2]["factor__rank"]), 1.0)
        self.assertTrue(audit["key_unique"])
        self.assertLess(audit["max_daily_zscore_std_deviation"], 1e-6)

    def test_future_date_change_does_not_change_past(self) -> None:
        frame = pd.DataFrame({
            "ts_code": ["A", "B", "C", "A", "B", "C"],
            "trade_date": [20240102] * 3 + [20240103] * 3,
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        first, _ = transform_cross_sectional(frame, ["factor"], min_group_size=3, zscore_ddof=0, zero_variance_zscore=0.0)
        changed = frame.copy()
        changed.loc[changed["trade_date"] == 20240103, "factor"] *= 1000.0
        second, _ = transform_cross_sectional(changed, ["factor"], min_group_size=3, zscore_ddof=0, zero_variance_zscore=0.0)
        pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])

    def test_zero_variance_and_small_group_rules(self) -> None:
        frame = pd.DataFrame({
            "ts_code": ["A", "B", "C"],
            "trade_date": [20240102] * 3,
            "factor": [5.0, 5.0, 5.0],
        })
        output, _ = transform_cross_sectional(frame, ["factor"], min_group_size=3, zscore_ddof=0, zero_variance_zscore=0.0)
        self.assertTrue((output["factor__rank"] == 2.0 / 3.0).all())
        self.assertTrue((output["factor__zscore"] == 0.0).all())
        small, _ = transform_cross_sectional(frame.iloc[:2], ["factor"], min_group_size=3, zscore_ddof=0, zero_variance_zscore=0.0)
        self.assertTrue(small[["factor__rank", "factor__zscore"]].isna().all().all())
