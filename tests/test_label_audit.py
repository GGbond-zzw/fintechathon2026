from unittest import TestCase

import pandas as pd

from src.data.label_audit import audit_label_alignment, label_distribution, time_breakdowns


class LabelAuditTest(TestCase):
    def test_exact_immediate_next_row_alignment(self) -> None:
        frame = pd.DataFrame({
            "ts_code": ["A", "A", "A", "B", "B"],
            "trade_date": [20240102, 20240103, 20240104, 20240102, 20240103],
            "close": [10.0, 11.0, 12.0, 20.0, 18.0],
            "y_ret_1d": [0.1, 12.0 / 11.0 - 1.0, None, -0.1, None],
        })
        original = frame.copy(deep=True)
        report, samples = audit_label_alignment(
            frame, "y_ret_1d", absolute_tolerance=1e-12,
            relative_tolerance=1e-10, mismatch_sample_rows=10,
        )
        self.assertEqual(report["best_supported_definition"], "immediate_next_row")
        self.assertEqual(report["methods"]["immediate_next_row"]["match_rate_on_comparable"], 1.0)
        self.assertTrue(samples.empty)
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_next_close_distinguishes_definitions(self) -> None:
        frame = pd.DataFrame({
            "ts_code": ["A", "A", "A"],
            "trade_date": [20240102, 20240103, 20240104],
            "close": [10.0, None, 12.0],
            "y_ret_1d": [None, None, None],
        })
        report, _ = audit_label_alignment(
            frame, "y_ret_1d", absolute_tolerance=1e-12,
            relative_tolerance=1e-10, mismatch_sample_rows=10,
        )
        self.assertEqual(report["methods"]["immediate_next_row"]["missing_mask_agreement_rate"], 1.0)
        self.assertLess(report["methods"]["next_non_null_close"]["missing_mask_agreement_rate"], 1.0)

    def test_test_price_closes_training_boundary_without_test_label(self) -> None:
        train = pd.DataFrame({
            "ts_code": ["A", "A"],
            "trade_date": [20241230, 20241231],
            "close": [10.0, 11.0],
            "y_ret_1d": [0.1, 1.0 / 11.0],
        })
        test_prices = pd.DataFrame({
            "ts_code": ["A"], "trade_date": [20250102], "close": [12.0],
        })
        report, samples = audit_label_alignment(
            train, "y_ret_1d", absolute_tolerance=1e-12,
            relative_tolerance=1e-10, mismatch_sample_rows=10,
            future_prices=test_prices,
        )
        immediate = report["methods"]["immediate_next_row"]
        self.assertEqual(immediate["comparable_rows"], 2)
        self.assertEqual(immediate["match_rate_on_comparable"], 1.0)
        self.assertEqual(immediate["missing_mask_agreement_rate"], 1.0)
        self.assertFalse("y_ret_1d" in test_prices.columns)
        self.assertTrue(samples.empty)

    def test_distribution_and_time_breakdowns(self) -> None:
        frame = pd.DataFrame({
            "trade_date": [20230102, 20230103, 20240102],
            "y_ret_1d": [0.1, None, -0.2],
        })
        summary = label_distribution(frame, "y_ret_1d", [0.5], [0.1])
        yearly, daily = time_breakdowns(frame, "y_ret_1d")
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(len(yearly), 2)
        self.assertEqual(len(daily), 3)
