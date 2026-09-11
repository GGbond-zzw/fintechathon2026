from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores


def fixture(dates: int = 3, stocks: int = 20) -> pd.DataFrame:
    rows = []
    for date_index in range(dates):
        for stock in range(stocks):
            rows.append({
                "ts_code": f"S{stock:03d}", "trade_date": 20260101 + date_index,
                "pred": float(stock + date_index * ((stock % 3) - 1)),
                "flag_limit_up": 0,
            })
    return pd.DataFrame(rows)


class TurnoverControlTest(TestCase):
    def test_future_rows_do_not_change_past_smoothed_scores(self) -> None:
        frame = fixture()
        first = frame[frame["trade_date"] < 20260103].reset_index(drop=True)
        first_scores, _ = causal_rank_ema(first, 0.7)
        changed = frame.copy()
        changed.loc[changed["trade_date"] == 20260103, "pred"] *= -1000
        all_scores, _ = causal_rank_ema(changed, 0.7)
        np.testing.assert_array_equal(first_scores, all_scores[:len(first)])

    def test_prior_state_continues_the_same_recurrence(self) -> None:
        frame = fixture()
        full, _ = causal_rank_ema(frame, 0.6)
        left = frame[frame["trade_date"] < 20260103].reset_index(drop=True)
        right = frame[frame["trade_date"] == 20260103].reset_index(drop=True)
        _, state = causal_rank_ema(left, 0.6)
        continued, _ = causal_rank_ema(right, 0.6, prior_state=state)
        np.testing.assert_allclose(continued, full[len(left):], rtol=0, atol=1e-15)

    def test_hysteresis_retains_buffer_member_and_returns_unique_scores(self) -> None:
        frame = fixture(dates=2, stocks=20)
        # Day two moves S018 just outside top 10% but inside the 20% exit buffer.
        frame.loc[(frame["trade_date"] == 20260102) & (frame["ts_code"] == "S018"), "pred"] = 17.5
        smoothed, _ = causal_rank_ema(frame, 1.0)
        predictions, selected = hysteresis_rank_scores(
            frame, smoothed, 0.20, top_fraction_denominator=10, minimum_pool_size=2
        )
        result = frame.assign(stable=predictions)
        day_two = result[result["trade_date"] == 20260102].nlargest(2, "stable")
        self.assertIn("S018", set(day_two["ts_code"]))
        self.assertIn("S018", selected)
        self.assertTrue(result.groupby("trade_date")["stable"].apply(lambda x: x.is_unique).all())
        self.assertTrue(np.isfinite(predictions).all())

    def test_subthreshold_day_resets_membership(self) -> None:
        frame = fixture(dates=1, stocks=5)
        smoothed, _ = causal_rank_ema(frame, 1.0)
        _, selected = hysteresis_rank_scores(
            frame, smoothed, 0.5, top_fraction_denominator=2,
            minimum_pool_size=10, prior_top_set={"S004"},
        )
        self.assertEqual(selected, set())
