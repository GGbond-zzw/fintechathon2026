from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from src.models.baselines import _evaluate_predictions, _fit_ridge, _predict_ridge, _save_oof


class BaselineTest(TestCase):
    def test_ridge_fit_uses_only_explicit_training_dates(self) -> None:
        with TemporaryDirectory() as directory:
            view = Path(directory)
            year_dir = view / "year=2024"
            year_dir.mkdir()
            frame = pd.DataFrame({
                "ts_code": ["A", "B", "A", "B"],
                "trade_date": [20240102, 20240102, 20240103, 20240103],
                "y_ret_1d": [0.1, -0.1, 99.0, -99.0],
                "factor__zscore": [1.0, -1.0, 10.0, -10.0],
            })
            frame.to_parquet(year_dir / "data.parquet", index=False)
            first_coef, first_intercept, rows = _fit_ridge(
                view, (20240102,), ["factor__zscore"], "y_ret_1d", 1.0, 10
            )
            changed = frame.copy()
            changed.loc[changed["trade_date"] == 20240103, "y_ret_1d"] *= 1000.0
            changed.to_parquet(year_dir / "data.parquet", index=False)
            second_coef, second_intercept, _ = _fit_ridge(
                view, (20240102,), ["factor__zscore"], "y_ret_1d", 1.0, 10
            )
            np.testing.assert_array_equal(first_coef, second_coef)
            self.assertEqual(first_intercept, second_intercept)
            self.assertEqual(rows, 2)

    def test_ridge_missing_cross_sectional_zscore_maps_to_neutral_zero(self) -> None:
        frame = pd.DataFrame({"factor__zscore": [1.0, np.nan, -1.0]})
        predictions = _predict_ridge(
            frame, ["factor__zscore"], np.array([2.0]), intercept=0.5
        )
        np.testing.assert_allclose(predictions, [2.5, 0.5, -1.5])

    def test_oof_audit_is_unique_finite_and_float32(self) -> None:
        rows = 200
        validation = pd.DataFrame({
            "ts_code": [f"S{i % 100:03d}" for i in range(rows)],
            "trade_date": [20240102] * 100 + [20240103] * 100,
            "y_ret_1d": np.linspace(-0.1, 0.1, rows),
            "flag_limit_up": np.zeros(rows, dtype="int8"),
        })
        predictions = np.linspace(-1.0, 1.0, rows, dtype="float64") + 1e-10
        metrics, _ = _evaluate_predictions(validation, predictions)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oof.parquet"
            audit = _save_oof(path, validation, predictions)
            saved = pd.read_parquet(path)
        self.assertTrue(audit["key_unique"])
        self.assertTrue(audit["pred_finite"])
        self.assertEqual(audit["pred_dtype"], "float32")
        self.assertEqual(saved["pred"].dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(list(metrics.values())).all())
