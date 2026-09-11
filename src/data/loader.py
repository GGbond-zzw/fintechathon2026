"""Read-only raw-data loaders with optional versioned Parquet caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.paths import ProjectPaths

CACHE_SCHEMA_VERSION = 1


def _required_columns(config: dict[str, Any], is_train: bool) -> list[str]:
    columns = list(config["data"]["test_required_columns"])
    if is_train:
        columns.append(str(config["data"]["label"]))
    return columns


def _csv_dtypes(config: dict[str, Any], is_train: bool) -> dict[str, str]:
    float_dtype = str(config["data"].get("float_dtype", "float32"))
    dtypes: dict[str, str] = {
        "ts_code": "string",
        "trade_date": "int32",
        "flag_limit_up": "int8",
        "flag_limit_down": "int8",
    }
    for column in config["data"]["float_columns"]:
        dtypes[str(column)] = float_dtype
    if is_train:
        dtypes[str(config["data"]["label"])] = float_dtype
    return dtypes


def _raw_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_paths(config: dict[str, Any], dataset: str) -> tuple[Path, Path]:
    parquet = ProjectPaths.from_config(config).cache / f"{dataset}.parquet"
    return parquet, parquet.with_suffix(".metadata.json")


def _cache_is_fresh(raw_path: Path, metadata_path: Path) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("schema_version") == CACHE_SCHEMA_VERSION and metadata.get("raw") == _raw_identity(raw_path)


def _check_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"Missing required columns in {path}: {absent}")


def _read_csv(path: Path, config: dict[str, Any], *, is_train: bool, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        kind = "training" if is_train else "test"
        raise FileNotFoundError(f"Configured {kind} CSV not found: {path}")
    frame = pd.read_csv(path, dtype=_csv_dtypes(config, is_train), usecols=usecols, low_memory=False)
    _check_columns(frame, usecols or _required_columns(config, is_train), path)
    return frame


def write_parquet_cache(frame: pd.DataFrame, config: dict[str, Any], dataset: str, raw_path: Path) -> Path:
    """Write a new cache and sidecar without touching the raw CSV."""
    parquet_path, metadata_path = _cache_paths(config, dataset)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, engine="pyarrow", index=False, compression="zstd")
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "raw": _raw_identity(raw_path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return parquet_path


def _load_dataset(config: dict[str, Any], dataset: str, *, use_cache: bool, refresh_cache: bool) -> pd.DataFrame:
    paths = ProjectPaths.from_config(config)
    is_train = dataset == "train"
    raw_path = paths.train if is_train else paths.test
    if not raw_path.is_file():
        kind = "training" if is_train else "test"
        raise FileNotFoundError(f"Configured {kind} CSV not found: {raw_path}")
    parquet_path, metadata_path = _cache_paths(config, dataset)
    cache_enabled = bool(config["data"].get("parquet_cache", True))
    if use_cache and cache_enabled and not refresh_cache and parquet_path.is_file() and _cache_is_fresh(raw_path, metadata_path):
        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        _check_columns(frame, _required_columns(config, is_train), parquet_path)
        return frame
    frame = _read_csv(raw_path, config, is_train=is_train)
    if cache_enabled and (use_cache or refresh_cache):
        write_parquet_cache(frame, config, dataset, raw_path)
    return frame


def load_train(config: dict[str, Any], *, use_cache: bool = True, refresh_cache: bool = False) -> pd.DataFrame:
    """Load the real configured training set; never falls back to test data."""
    return _load_dataset(config, "train", use_cache=use_cache, refresh_cache=refresh_cache)


def load_train_columns(
    config: dict[str, Any], columns: Iterable[str], *, use_cache: bool = True,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Load selected real-training columns without materializing the full table."""
    requested = list(dict.fromkeys(str(column) for column in columns))
    if not requested:
        raise ValueError("At least one training column must be requested")
    allowed = _required_columns(config, True)
    unknown = [column for column in requested if column not in allowed]
    if unknown:
        raise ValueError(f"Unknown configured training columns: {unknown}")
    paths = ProjectPaths.from_config(config)
    raw_path = paths.train
    if not raw_path.is_file():
        raise FileNotFoundError(f"Configured training CSV not found: {raw_path}")
    parquet_path, metadata_path = _cache_paths(config, "train")
    if use_cache and parquet_path.is_file() and _cache_is_fresh(raw_path, metadata_path):
        frame = pd.read_parquet(
            parquet_path, columns=requested, filters=filters, engine="pyarrow"
        )
        _check_columns(frame, requested, parquet_path)
        return frame
    frame = _read_csv(raw_path, config, is_train=True, usecols=requested)
    if filters:
        for column, operator, value in filters:
            if operator == ">=":
                frame = frame.loc[frame[column] >= value]
            elif operator == "<=":
                frame = frame.loc[frame[column] <= value]
            elif operator == "==":
                frame = frame.loc[frame[column] == value]
            else:
                raise ValueError(f"Unsupported CSV fallback filter operator: {operator}")
    return frame.reset_index(drop=True)


def load_test(config: dict[str, Any], *, use_cache: bool = True, refresh_cache: bool = False) -> pd.DataFrame:
    """Load the label-free configured test set."""
    return _load_dataset(config, "test", use_cache=use_cache, refresh_cache=refresh_cache)


def load_train_tail(config: dict[str, Any], n_trade_dates: int | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    """Load the last N real training dates without ever substituting test data."""
    paths = ProjectPaths.from_config(config)
    raw_path = paths.train
    if not raw_path.is_file():
        raise FileNotFoundError(f"Configured training CSV not found: {raw_path}")
    count = int(n_trade_dates or config["data"]["warmup_days"])
    if count < 1:
        raise ValueError("n_trade_dates must be positive")
    parquet_path, metadata_path = _cache_paths(config, "train")
    if use_cache and parquet_path.is_file() and _cache_is_fresh(raw_path, metadata_path):
        dates = pd.read_parquet(parquet_path, columns=["trade_date"], engine="pyarrow")["trade_date"].drop_duplicates()
        selected = sorted(dates.tolist())[-count:]
        if not selected:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        return pd.read_parquet(parquet_path, engine="pyarrow", filters=[("trade_date", ">=", min(selected))])
    chunksize = int(config["data"].get("csv_chunksize", 250_000))
    dates: set[int] = set()
    for chunk in pd.read_csv(raw_path, usecols=["trade_date"], dtype={"trade_date": "int32"}, chunksize=chunksize):
        dates.update(int(value) for value in chunk["trade_date"].dropna().unique())
    selected = sorted(dates)[-count:]
    if not selected:
        return _read_csv(raw_path, config, is_train=True).iloc[0:0]
    cutoff = min(selected)
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_path, dtype=_csv_dtypes(config, True), chunksize=chunksize, low_memory=False):
        kept.append(chunk.loc[chunk["trade_date"] >= cutoff])
    frame = pd.concat(kept, ignore_index=True)
    _check_columns(frame, _required_columns(config, True), raw_path)
    return frame


def assess_float32_precision(csv_path: Path, config: dict[str, Any], *, is_train: bool) -> dict[str, Any]:
    """Quantify float64-to-float32 error on a deterministic CSV prefix."""
    columns = list(config["data"]["float_columns"])
    if is_train:
        columns.append(str(config["data"]["label"]))
    sample_rows = int(config["data"].get("precision_sample_rows", 100_000))
    sample = pd.read_csv(csv_path, usecols=columns, nrows=sample_rows, dtype="float64")
    report: dict[str, Any] = {"sample_rows": len(sample), "columns": {}}
    for column in columns:
        original = sample[column]
        restored = original.astype("float32").astype("float64")
        absolute = (original - restored).abs()
        denominator = original.abs().where(original != 0)
        relative = absolute / denominator
        report["columns"][column] = {
            "max_abs_error": float(absolute.max()) if absolute.notna().any() else None,
            "max_rel_error": float(relative.max()) if relative.notna().any() else None,
        }
    return report
