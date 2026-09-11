from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import numpy as np
import pandas as pd
from src.submission.validator import validate_csv_roundtrip, validate_submission_frame

class SubmissionTest(TestCase):
    def setUp(self) -> None:
        self.keys = pd.DataFrame({
            "ts_code": pd.Series(["000001.SZ", "600000.SH"], dtype="string"),
            "trade_date": pd.Series([20250102, 20250102], dtype="int32"),
        })
        self.frame = self.keys.assign(pred=pd.Series([0.1, 0.9], dtype="float32"))

    def test_valid_contract_and_float32_csv_roundtrip(self) -> None:
        report = validate_submission_frame(self.frame, self.keys)
        self.assertTrue(report["key_set_exact"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            self.frame.to_csv(path, index=False)
            _, roundtrip = validate_csv_roundtrip(path, self.frame, self.keys)
            self.assertTrue(roundtrip["roundtrip_float32_exact"])

    def test_rejects_column_key_and_value_failures(self) -> None:
        with self.assertRaises(ValueError):
            validate_submission_frame(self.frame[["trade_date", "ts_code", "pred"]], self.keys)
        with self.assertRaises(ValueError):
            validate_submission_frame(self.frame.iloc[:1], self.keys)
        invalid = self.frame.copy(); invalid.loc[0, "pred"] = np.inf
        with self.assertRaises(ValueError):
            validate_submission_frame(invalid, self.keys)
