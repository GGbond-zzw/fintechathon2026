"""Assemble reusable year-partitioned model views from validated feature stores."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_train_columns


KEYS = ["ts_code", "trade_date"]


def _key_hash(frame: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(frame[KEYS], index=False).sum()) & ((1 << 64) - 1)


def model_feature_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    raw = list(config["cross_sectional"]["source_features"])
    cross_rank = [f"{name}__rank" for name in raw]
    cross_zscore = [f"{name}__zscore" for name in raw]
    market_path = Path(config["features"]["market_output"])
    market_columns = list(pd.read_parquet(market_path, engine="pyarrow").columns)
    market = [name for name in market_columns if name != "trade_date"]
    return {"raw": raw, "cross_rank": cross_rank, "cross_zscore": cross_zscore, "market": market}


def build_training_model_view(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Join features, flags and real labels one year at a time."""
    settings = config["baseline"]
    output_dir = Path(settings["model_view_output"])
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Model view exists; pass overwrite explicitly: {output_dir}")
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    base_dir = Path(config["features"]["model_base_by_year_output"])
    cross_dir = Path(config["cross_sectional"]["output"])
    groups = model_feature_groups(config)
    feature_names = groups["raw"] + groups["cross_rank"] + groups["cross_zscore"] + groups["market"]
    label = str(config["data"]["label"])
    market = pd.read_parquet(Path(config["features"]["market_output"]), engine="pyarrow")
    reports: dict[str, Any] = {}
    total_rows = 0
    total_hash = 0
    try:
        for base_year_dir in sorted(base_dir.glob("year=*")):
            year = int(base_year_dir.name.split("=", 1)[1])
            base = pd.read_parquet(base_year_dir / "features.parquet", engine="pyarrow")
            cross = pd.read_parquet(
                cross_dir / f"year={year}" / "features.parquet", engine="pyarrow"
            )
            base_hash, cross_hash = _key_hash(base), _key_hash(cross)
            if len(base) != len(cross) or base_hash != cross_hash:
                raise ValueError(f"Base/cross feature keys differ for year {year}")
            joined = base.merge(cross, on=KEYS, how="inner", validate="one_to_one")
            labels = load_train_columns(
                config,
                ["ts_code", "trade_date", "flag_limit_up", label],
                use_cache=True,
                filters=[("trade_date", ">=", year * 10000 + 101), ("trade_date", "<=", year * 10000 + 1231)],
            )
            joined = joined.merge(labels, on=KEYS, how="inner", validate="one_to_one")
            joined = joined.merge(market, on="trade_date", how="left", validate="many_to_one")
            if len(joined) != len(base):
                raise ValueError(f"Model view row loss for year {year}")
            joined = joined[[*KEYS, label, "flag_limit_up", *feature_names]].sort_values(
                ["trade_date", "ts_code"], kind="stable"
            ).reset_index(drop=True)
            if joined.duplicated(KEYS).any():
                raise ValueError(f"Duplicate model-view keys for year {year}")
            inf_count = sum(
                int(np.isinf(joined[name].to_numpy(dtype="float64", na_value=np.nan)).sum())
                for name in feature_names
            )
            if inf_count:
                raise ValueError(f"Infinite model feature values for year {year}: {inf_count}")
            target_dir = temporary / f"year={year}"
            target_dir.mkdir(parents=True)
            target = target_dir / "data.parquet"
            joined.to_parquet(target, engine="pyarrow", compression="zstd", index=False)
            key_hash = _key_hash(joined)
            reports[str(year)] = {
                "rows": int(len(joined)),
                "labeled_rows": int(joined[label].notna().sum()),
                "key_hash": f"{key_hash:016x}",
                "key_unique": True,
                "inf_count": inf_count,
                "bytes": target.stat().st_size,
            }
            total_rows += len(joined)
            total_hash = (total_hash + key_hash) & ((1 << 64) - 1)
            del base, cross, labels, joined
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    report = {
        "version": "model_view_v1",
        "rows": total_rows,
        "feature_count": len(feature_names),
        "feature_groups": groups,
        "feature_names": feature_names,
        "label": label,
        "key_hash": f"{total_hash:016x}",
        "years": reports,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    return report


def load_model_view_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["baseline"]["model_view_output"]) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model view manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
