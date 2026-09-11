from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd

from src.data.loader import load_test, load_train, load_train_columns, load_train_tail


def make_config(root: Path) -> dict:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    cache = root / "data" / "cache"
    paths = {"root": root, "train": raw / "训练集.csv", "test": raw / "测试集_X.csv", "interim": root / "data/interim", "processed": root / "data/processed", "cache": cache, "models": root / "models", "experiments": root / "experiments", "submissions": root / "submissions", "reports": root / "reports", "logs": root / "logs"}
    return {"paths": {name: str(path) for name, path in paths.items()}, "data": {"label": "y_ret_1d", "id_cols": ["ts_code", "trade_date"], "float_columns": ["open", "high", "low", "close", "vol", "amount"], "test_required_columns": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "flag_limit_up", "flag_limit_down"], "warmup_days": 2, "parquet_cache": True, "float_dtype": "float32", "csv_chunksize": 2, "precision_sample_rows": 3}}


def sample_frame(include_label: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame({"ts_code": ["000001.SZ"] * 3, "trade_date": [20240102, 20240103, 20240104], "open": [10.0, 10.1, 10.2], "high": [10.3, 10.4, 10.5], "low": [9.9, 10.0, 10.1], "close": [10.2, 10.3, 10.4], "vol": [100.0, 110.0, 120.0], "amount": [1000.0, 1133.0, 1248.0], "flag_limit_up": [0, 0, 0], "flag_limit_down": [0, 0, 0]})
    if include_label:
        frame["y_ret_1d"] = [0.01, 0.02, None]
    return frame


class LoaderTest(TestCase):
    def test_test_cache_roundtrip_and_no_label_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            sample_frame().to_csv(config["paths"]["test"], index=False)
            first = load_test(config, use_cache=True, refresh_cache=True)
            second = load_test(config, use_cache=True)
            self.assertNotIn("y_ret_1d", first.columns)
            self.assertTrue(first.equals(second))
            self.assertTrue((Path(config["paths"]["cache"]) / "test.parquet").is_file())

    def test_train_never_falls_back_to_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            sample_frame().to_csv(config["paths"]["test"], index=False)
            with self.assertRaises(FileNotFoundError):
                load_train(config)

    def test_train_tail_uses_last_trade_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            sample_frame(include_label=True).to_csv(config["paths"]["train"], index=False)
            tail = load_train_tail(config, n_trade_dates=2, use_cache=False)
            self.assertEqual(sorted(tail["trade_date"].unique().tolist()), [20240103, 20240104])

    def test_train_columns_reads_only_requested_real_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            sample_frame(include_label=True).to_csv(config["paths"]["train"], index=False)
            selected = load_train_columns(
                config, ["ts_code", "trade_date", "y_ret_1d"], use_cache=False
            )
            self.assertEqual(list(selected.columns), ["ts_code", "trade_date", "y_ret_1d"])
            with self.assertRaises(ValueError):
                load_train_columns(config, ["made_up_label"], use_cache=False)

    def test_train_columns_applies_date_filters_on_csv_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            sample_frame(include_label=True).to_csv(config["paths"]["train"], index=False)
            selected = load_train_columns(
                config, ["ts_code", "trade_date", "y_ret_1d"], use_cache=False,
                filters=[("trade_date", ">=", 20240103), ("trade_date", "<=", 20240103)],
            )
            self.assertEqual(selected["trade_date"].tolist(), [20240103])
