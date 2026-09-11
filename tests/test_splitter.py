from __future__ import annotations

from unittest import TestCase

import pandas as pd

from src.validation.splitter import ExpandingWindowSplitter


DATES = [
    20180102, 20211230, 20211231,
    20220104, 20221229, 20221230,
    20230103, 20231228, 20231229,
    20240102, 20241231,
]

FOLDS = [
    {"name": "fold_2022", "train_start": 20180102, "train_end": 20211231, "val_start": 20220101, "val_end": 20221231},
    {"name": "fold_2023", "train_start": 20180102, "train_end": 20221231, "val_start": 20230101, "val_end": 20231231},
    {"name": "fold_2024", "train_start": 20180102, "train_end": 20231231, "val_start": 20240101, "val_end": 20241231},
]


class SplitterTest(TestCase):
    def test_exact_last_training_trade_date_is_purged(self) -> None:
        splits = ExpandingWindowSplitter(
            FOLDS, purge_trade_days=1, label_horizon_trade_days=1
        ).split(DATES)
        self.assertEqual([split.purge_dates for split in splits], [
            (20211231,), (20221230,), (20231229,),
        ])
        self.assertEqual([split.train_dates[-1] for split in splits], [
            20211230, 20221229, 20231228,
        ])
        self.assertEqual([split.val_dates[0] for split in splits], [
            20220104, 20230103, 20240102,
        ])
        for split in splits:
            self.assertTrue(set(split.train_dates).isdisjoint(split.purge_dates))
            self.assertTrue(set(split.train_dates).isdisjoint(split.val_dates))
            self.assertTrue(set(split.purge_dates).isdisjoint(split.val_dates))

    def test_folds_are_expanding(self) -> None:
        splits = ExpandingWindowSplitter(
            FOLDS, purge_trade_days=1, label_horizon_trade_days=1
        ).split(DATES)
        self.assertTrue(set(splits[0].train_dates).issubset(splits[1].train_dates))
        self.assertTrue(set(splits[1].train_dates).issubset(splits[2].train_dates))

    def test_masks_exclude_purge_and_missing_labels(self) -> None:
        split = ExpandingWindowSplitter(
            FOLDS[:1], purge_trade_days=1, label_horizon_trade_days=1
        ).split(DATES)[0]
        frame = pd.DataFrame({
            "trade_date": [20211230, 20211231, 20220104, 20220104],
            "y_ret_1d": [0.01, 0.02, 0.03, None],
        })
        masks = split.date_masks(frame)
        labeled = split.labeled_masks(frame, "y_ret_1d")
        self.assertEqual(masks["train"].tolist(), [True, False, False, False])
        self.assertEqual(masks["purge"].tolist(), [False, True, False, False])
        self.assertEqual(labeled["validation"].tolist(), [False, False, True, False])

    def test_rejects_insufficient_purge_and_calendar_overlap(self) -> None:
        with self.assertRaises(ValueError):
            ExpandingWindowSplitter(FOLDS, purge_trade_days=0, label_horizon_trade_days=1)
        overlapping = [{
            "name": "bad", "train_start": 20180102, "train_end": 20220104,
            "val_start": 20220101, "val_end": 20221231,
        }]
        with self.assertRaises(ValueError):
            ExpandingWindowSplitter(
                overlapping, purge_trade_days=1, label_horizon_trade_days=1
            ).split(DATES)

