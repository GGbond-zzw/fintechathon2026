from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from src.models.dataset import build_training_model_view


class ModelDatasetTest(TestCase):
    def test_year_view_joins_features_market_flags_and_real_label_by_keys(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base" / "year=2024"
            cross_dir = root / "cross" / "year=2024"
            base_dir.mkdir(parents=True)
            cross_dir.mkdir(parents=True)
            keys = pd.DataFrame({"ts_code": ["A", "B"], "trade_date": [20240102, 20240102]})
            keys.assign(factor=[1.0, 2.0]).to_parquet(base_dir / "features.parquet", index=False)
            keys.iloc[::-1].assign(
                factor__rank=[1.0, 0.5], factor__zscore=[1.0, -1.0]
            ).to_parquet(cross_dir / "features.parquet", index=False)
            market_path = root / "market.parquet"
            pd.DataFrame({"trade_date": [20240102], "market_mean": [0.01]}).to_parquet(market_path, index=False)
            raw = root / "train.csv"
            keys.assign(flag_limit_up=[0, 1], y_ret_1d=[0.1, None]).to_csv(raw, index=False)
            config = {
                "paths": {
                    "root": str(root), "train": str(raw), "test": str(root / "test.csv"),
                    "cache": str(root / "cache"), "processed": str(root), "interim": str(root),
                    "models": str(root / "models"), "experiments": str(root),
                    "submissions": str(root), "reports": str(root), "logs": str(root),
                },
                "data": {
                    "label": "y_ret_1d", "id_cols": ["ts_code", "trade_date"],
                    "float_columns": [],
                    "test_required_columns": ["ts_code", "trade_date", "flag_limit_up", "y_ret_1d"],
                    "float_dtype": "float32", "parquet_cache": False,
                },
                "features": {"model_base_by_year_output": str(root / "base"), "market_output": str(market_path)},
                "cross_sectional": {"source_features": ["factor"], "output": str(root / "cross")},
                "baseline": {"model_view_output": str(root / "view")},
            }
            report = build_training_model_view(config)
            output = pd.read_parquet(root / "view" / "year=2024" / "data.parquet")
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["feature_count"], 4)
            self.assertEqual(output.loc[output["ts_code"] == "A", "y_ret_1d"].iloc[0], 0.1)
            self.assertEqual(output.loc[output["ts_code"] == "B", "flag_limit_up"].iloc[0], 1)
            self.assertTrue(output[["factor", "factor__rank", "factor__zscore", "market_mean"]].notna().all().all())

