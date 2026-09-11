from unittest import TestCase

import pandas as pd

from src.data.validator import compare_cache_consistency, validate_frame

REQUIRED = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "flag_limit_up", "flag_limit_down"]


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({"ts_code": ["000001.SZ", "000001.SZ"], "trade_date": [20240102, 20240103], "open": [10.0, 10.1], "high": [10.3, 10.4], "low": [9.9, 10.0], "close": [10.2, 10.3], "vol": [100.0, 110.0], "amount": [1000.0, 1133.0], "flag_limit_up": [0, 1], "flag_limit_down": [0, 0]})


class ValidatorTest(TestCase):
    def test_valid_frame_passes(self) -> None:
        report = validate_frame(valid_frame(), dataset_name="fixture", required_columns=REQUIRED, id_cols=["ts_code", "trade_date"])
        self.assertEqual(report["status"], "valid")

    def test_detects_integrity_violations(self) -> None:
        frame = valid_frame()
        frame.loc[1, ["trade_date", "high", "vol", "flag_limit_down"]] = [20240102, 9.0, -1.0, 2]
        report = validate_frame(frame, dataset_name="fixture", required_columns=REQUIRED, id_cols=["ts_code", "trade_date"])
        self.assertEqual(report["status"], "issues_found")
        self.assertGreater(report["duplicate_key_rows"], 0)
        self.assertEqual(report["violations"]["ohlc"], 1)
        self.assertEqual(report["violations"]["negative_vol"], 1)
        self.assertEqual(report["violations"]["invalid_limit_down"], 1)

    def test_cache_consistency(self) -> None:
        frame = valid_frame()
        report = compare_cache_consistency(frame, frame.copy(), id_cols=["ts_code", "trade_date"])
        self.assertEqual(report["status"], "consistent")

