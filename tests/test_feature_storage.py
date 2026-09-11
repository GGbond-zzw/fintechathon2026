import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd

from src.features.storage import repartition_parquet_by_year


class FeatureStorageTest(TestCase):
    def test_year_partition_preserves_rows_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            output = root / "by_year"
            frame = pd.DataFrame({
                "ts_code": ["A", "A", "B", "B"],
                "trade_date": [20231229, 20240102, 20231229, 20240102],
                "ret_1": [0.1, 0.2, -0.1, -0.2],
            })
            frame.to_parquet(source, index=False)
            report = repartition_parquet_by_year(
                source, output, columns=["ts_code", "trade_date", "ret_1"]
            )
            self.assertTrue(report["row_count_equal"])
            self.assertTrue(report["all_partition_keys_unique"])
            self.assertEqual(set(report["years"]), {"2023", "2024"})
            with self.assertRaises(FileExistsError):
                repartition_parquet_by_year(
                    source, output, columns=["ts_code", "trade_date", "ret_1"]
                )
            overwritten = repartition_parquet_by_year(
                source, output, columns=["ts_code", "trade_date", "ret_1"], overwrite=True
            )
            self.assertTrue(overwritten["row_count_equal"])
