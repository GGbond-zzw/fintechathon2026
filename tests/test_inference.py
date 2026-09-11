from __future__ import annotations
from unittest import TestCase
import numpy as np
import pandas as pd
from src.models.inference import boundary_postprocess, validate_exact_keys
from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores

def frame(dates: list[int]) -> pd.DataFrame:
    rows = []
    for date in dates:
        for i, code in enumerate(("A", "B", "C", "D")):
            rows.append({"ts_code": code, "trade_date": date, "flag_limit_up": 0,
                         "pred": float((date % 7) * (i + 1) - i)})
    return pd.DataFrame(rows)

class InferenceTest(TestCase):
    def test_explicit_boundary_state_equals_one_continuous_run(self) -> None:
        warm, test = frame([20241230, 20241231]), frame([20250102, 20250103])
        split, state = boundary_postprocess(
            warm, test, alpha=0.5, exit_fraction=0.5, denominator=2, minimum_pool=1
        )
        combined = pd.concat([warm, test], ignore_index=True)
        smooth, _ = causal_rank_ema(combined, 0.5)
        continuous, _ = hysteresis_rank_scores(
            combined, smooth, 0.5, top_fraction_denominator=2, minimum_pool_size=1
        )
        np.testing.assert_array_equal(split, continuous[len(warm):])
        self.assertEqual(state["warmup_ema_state_size"], 4)

    def test_exact_key_contract_rejects_missing_and_duplicates(self) -> None:
        expected = pd.DataFrame({"ts_code": ["A", "B"], "trade_date": [1, 1]})
        prediction = expected.assign(pred=[0.1, 0.2])
        self.assertTrue(validate_exact_keys(prediction, expected)["exact_match"])
        with self.assertRaises(ValueError):
            validate_exact_keys(prediction.iloc[:1], expected)
        with self.assertRaises(ValueError):
            validate_exact_keys(pd.concat([prediction, prediction.iloc[:1]]), expected)
