from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.models.ranking import complete_date_chunks, relevance_labels


class RankingTest(TestCase):
    def test_relevance_is_within_date_ordinal_and_ties_share_level(self) -> None:
        frame = pd.DataFrame({
            "trade_date": [1, 1, 1, 1, 2, 2, 2, 2],
            "y": [1.0, 2.0, 2.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        })
        result = relevance_labels(frame, label="y", bins=2)
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 1, 1, 1, 0, 0], dtype="uint8"))

    def test_relevance_rejects_missing_labels(self) -> None:
        frame = pd.DataFrame({"trade_date": [1, 1], "y": [1.0, np.nan]})
        with self.assertRaises(ValueError):
            relevance_labels(frame, label="y", bins=10)

    def test_complete_date_chunks_repairs_batch_splits(self) -> None:
        first = pd.DataFrame({"trade_date": [1, 1, 2], "value": [1, 2, 3]})
        second = pd.DataFrame({"trade_date": [2, 2, 3], "value": [4, 5, 6]})
        third = pd.DataFrame({"trade_date": [3, 4], "value": [7, 8]})
        chunks = list(complete_date_chunks([first, second, third]))
        combined = pd.concat(chunks, ignore_index=True)
        self.assertEqual(combined.to_dict("list"), {
            "trade_date": [1, 1, 2, 2, 2, 3, 3, 4],
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
        })
        seen = set()
        for chunk in chunks:
            current = set(chunk["trade_date"])
            self.assertFalse(seen & current)
            seen |= current

    def test_complete_date_chunks_rejects_reverse_dates(self) -> None:
        bad = pd.DataFrame({"trade_date": [2, 1], "value": [1, 2]})
        with self.assertRaises(ValueError):
            list(complete_date_chunks([bad]))
