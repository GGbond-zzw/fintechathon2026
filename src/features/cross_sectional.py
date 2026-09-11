"""Leakage-safe same-date cross-sectional transforms."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _key_hash_sum(frame: pd.DataFrame) -> int:
    hashed = pd.util.hash_pandas_object(frame[["ts_code", "trade_date"]], index=False)
    return int(hashed.sum()) & ((1 << 64) - 1)


def transform_cross_sectional(
    frame: pd.DataFrame,
    features: list[str],
    *,
    min_group_size: int,
    zscore_ddof: int,
    zero_variance_zscore: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create percentile-rank and z-score columns within each trade date."""
    required = ["ts_code", "trade_date", *features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing cross-sectional inputs: {missing}")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Cross-sectional input keys must be unique")
    ordered = frame[required].sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    grouped = ordered.groupby("trade_date", sort=False, observed=True)[features]
    counts = grouped.transform("count")
    ranks = grouped.rank(method="average", pct=True, na_option="keep")
    means = grouped.transform("mean")
    stds = grouped.transform("std", ddof=zscore_ddof)
    valid_groups = counts >= min_group_size
    ranks = ranks.where(valid_groups)
    zscores = ((ordered[features] - means) / stds.where(stds != 0)).where(valid_groups)
    zero_variance = valid_groups & stds.eq(0) & ordered[features].notna()
    zscores = zscores.mask(zero_variance, zero_variance_zscore)
    ranks = ranks.replace([np.inf, -np.inf], np.nan).astype("float32")
    zscores = zscores.replace([np.inf, -np.inf], np.nan).astype("float32")
    ranks.columns = [f"{column}__rank" for column in features]
    zscores.columns = [f"{column}__zscore" for column in features]
    output = pd.concat([ordered[["ts_code", "trade_date"]], ranks, zscores], axis=1)

    z_daily_mean = zscores.groupby(ordered["trade_date"], sort=False).mean().abs()
    z_daily_std = zscores.groupby(ordered["trade_date"], sort=False).std(ddof=zscore_ddof)
    source_daily_std = stds.groupby(ordered["trade_date"], sort=False).first()
    standardizable = source_daily_std > 0
    standardizable.columns = [f"{column}__zscore" for column in features]
    std_deviation = (z_daily_std - 1.0).abs().where(standardizable)
    audit = {
        "rows": int(len(output)),
        "dates": int(ordered["trade_date"].nunique()),
        "source_feature_count": len(features),
        "output_feature_count": len(features) * 2,
        "key_unique": not bool(output.duplicated(["ts_code", "trade_date"]).any()),
        "sorted_by_date_code": output[["trade_date", "ts_code"]].equals(
            output[["trade_date", "ts_code"]].sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        ),
        "rank_min": float(ranks.min().min()),
        "rank_max": float(ranks.max().max()),
        "inf_count": int(np.isinf(output.drop(columns=["ts_code", "trade_date"]).to_numpy(dtype="float64", na_value=np.nan)).sum()),
        "max_abs_daily_zscore_mean": float(z_daily_mean.max().max()),
        "max_daily_zscore_std_deviation": float(std_deviation.max().max()),
        "small_group_cells": int((~valid_groups & ordered[features].notna()).sum().sum()),
        "zero_variance_cells": int(zero_variance.sum().sum()),
        "finite_ratio": {
            column: float(output[column].notna().mean())
            for column in output.columns
            if column not in ("ts_code", "trade_date")
        },
    }
    return output, audit


def build_cross_sectional_store_from_paths(
    input_dir: Path,
    output_dir: Path,
    settings: dict[str, Any],
    *,
    overwrite: bool = False,
    phase: int = 4,
) -> dict[str, Any]:
    """Build date-local transforms for any compatible yearly store."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Year-partitioned feature input not found: {input_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Cross-sectional output exists; pass overwrite explicitly: {output_dir}")
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    features = list(settings["source_features"])
    year_reports: dict[str, Any] = {}
    total_input_hash = 0
    total_output_hash = 0
    total_rows = 0
    try:
        for year_dir in sorted(input_dir.glob("year=*")):
            year = year_dir.name.split("=", 1)[1]
            source_path = year_dir / "features.parquet"
            frame = pd.read_parquet(source_path, columns=["ts_code", "trade_date", *features], engine="pyarrow")
            input_hash = _key_hash_sum(frame)
            output, audit = transform_cross_sectional(
                frame,
                features,
                min_group_size=int(settings["min_group_size"]),
                zscore_ddof=int(settings["zscore_ddof"]),
                zero_variance_zscore=float(settings["zero_variance_zscore"]),
            )
            output_hash = _key_hash_sum(output)
            target_dir = temporary / f"year={year}"
            target_dir.mkdir(parents=True)
            target_path = target_dir / "features.parquet"
            output.to_parquet(target_path, engine="pyarrow", compression="zstd", index=False)
            audit.update({
                "input_key_hash": f"{input_hash:016x}",
                "output_key_hash": f"{output_hash:016x}",
                "key_hash_equal": input_hash == output_hash,
                "file_bytes": target_path.stat().st_size,
            })
            year_reports[year] = audit
            total_rows += len(output)
            total_input_hash = (total_input_hash + input_hash) & ((1 << 64) - 1)
            total_output_hash = (total_output_hash + output_hash) & ((1 << 64) - 1)
            del frame, output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if output_dir.exists():
        backup = output_dir.with_name(f"{output_dir.name}.backup")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output_dir, backup)
        try:
            os.replace(temporary, output_dir)
        except Exception:
            os.replace(backup, output_dir)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(temporary, output_dir)
    report = {
        "phase": phase,
        "input": str(input_dir),
        "output": str(output_dir),
        "grouping_key": "trade_date",
        "source_features": features,
        "source_feature_count": len(features),
        "transforms": list(settings["transforms"]),
        "winsorize": bool(settings["winsorize"]),
        "future_dates_used": False,
        "rows": total_rows,
        "input_key_hash": f"{total_input_hash:016x}",
        "output_key_hash": f"{total_output_hash:016x}",
        "key_hash_equal": total_input_hash == total_output_hash,
        "years": year_reports,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_cross_sectional_store(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    settings = config["cross_sectional"]
    report = build_cross_sectional_store_from_paths(
        Path(config["features"]["model_base_by_year_output"]),
        Path(settings["output"]),
        settings,
        overwrite=overwrite,
        phase=4,
    )
    Path(settings["audit_output"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
