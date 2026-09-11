from __future__ import annotations

from unittest import TestCase

import pandas as pd

from src.models.stability import _daily_oof_diagnostics, trailing_train_dates


class StabilityTest(TestCase):
    def test_trailing_windows_preserve_purge_and_chronology(self) -> None:
        train = (20180102, 20191231, 20200102, 20211230)
        validation = (20220103, 20221230)
        self.assertEqual(trailing_train_dates(train, validation, None), train)
        self.assertEqual(trailing_train_dates(train, validation, 2), (20200102, 20211230))
        self.assertTrue(max(trailing_train_dates(train, validation, 2)) < min(validation))

    def test_future_validation_year_does_not_enter_training_window(self) -> None:
        train = (20190102, 20200102, 20210104, 20220104, 20231228)
        validation = (20240102, 20241231)
        selected = trailing_train_dates(train, validation, 2)
        self.assertEqual(selected, (20220104, 20231228))
        self.assertNotIn(20240102, selected)

    def test_daily_diagnostics_assign_turnover_to_current_full_calendar_date(self) -> None:
        rows = []
        for date, reverse in ((20220103, False), (20220104, True)):
            for index in range(100):
                score = 99 - index if reverse else index
                rows.append({
                    "ts_code": f"S{index:03d}", "trade_date": date,
                    "y_ret_1d": index / 1000.0, "flag_limit_up": 0, "pred": score,
                })
        result = _daily_oof_diagnostics(pd.DataFrame(rows), "fold", "source")
        self.assertTrue(pd.isna(result.loc[0, "turnover"]))
        self.assertEqual(float(result.loc[1, "turnover"]), 1.0)
        self.assertGreater(float(result.loc[0, "ic"]), 0.99)
        self.assertLess(float(result.loc[1, "ic"]), -0.99)
