from __future__ import annotations
from unittest import TestCase
import pandas as pd
from src.models.finalization import select_candidate, simplex_weights

class FinalizationTest(TestCase):
    def test_simplex_grid_is_complete_and_valid(self) -> None:
        grid = simplex_weights(0.2)
        self.assertEqual(len(grid), 21)
        self.assertEqual(len(set(grid)), 21)
        self.assertTrue(all(min(x) >= 0 and abs(sum(x) - 1) < 1e-12 for x in grid))
        self.assertIn((1.0, 0.0, 0.0), grid)
        self.assertIn((0.0, 1.0, 0.0), grid)
        self.assertIn((0.0, 0.0, 1.0), grid)

    def test_selection_prefers_mean_then_worst_then_turnover(self) -> None:
        frame = pd.DataFrame([
            {"candidate": "a", "final_score": .4, "final_score_worst": .2, "mean_turnover": .1},
            {"candidate": "b", "final_score": .4, "final_score_worst": .3, "mean_turnover": .2},
            {"candidate": "c", "final_score": .39, "final_score_worst": .35, "mean_turnover": .05},
        ])
        self.assertEqual(select_candidate(frame)["candidate"], "b")
