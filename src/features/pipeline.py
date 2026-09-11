"""Memory-bounded Phase 3 feature artifact builder."""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.loader import load_train
from src.features.market import compute_market_features
from src.features.registry import build_feature_dictionary
from src.features.time_series import compute_stock_features
from src.utils.paths import ProjectPaths

FEATURE_INPUTS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "vol", "amount", "flag_limit_up", "flag_limit_down",
]


def _hash_sum(keys: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(keys, index=False).sum()) & ((1 << 64) - 1)


def _flush_batch(
    frames: list[pd.DataFrame],
    writer: pq.ParquetWriter | None,
    output_path: Path,
    compression: str,
) -> tuple[pq.ParquetWriter, pd.DataFrame]:
    batch = pd.concat(frames, ignore_index=True)
    table = pa.Table.from_pandas(batch, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
    writer.write_table(table)
    return writer, batch


def build_time_feature_store(
    base: pd.DataFrame,
    output_path: Path,
    settings: dict[str, Any],
    *,
    max_stocks: int | None = None,
    output_start_date: int | None = None,
) -> dict[str, Any]:
    """Compute per-stock features and stream batches to one Parquet file.

    Rows before ``output_start_date`` participate in causal rolling calculations
    but are not written. This is the test-set warm-up boundary; no future rows are
    consulted by ``compute_stock_features``.
    """
    source = base[FEATURE_INPUTS].sort_values(["ts_code", "trade_date"], kind="stable")
    if max_stocks is not None:
        selected = source["ts_code"].drop_duplicates().head(max_stocks)
        source = source[source["ts_code"].isin(selected)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    stock_count = 0
    null_counts: Counter[str] = Counter()
    inf_counts: Counter[str] = Counter()
    output_key_hash = 0
    feature_names: list[str] = []
    batch_size = int(settings["stocks_per_write_batch"])
    workers = int(settings.get("feature_workers", 1))
    grouped = iter(source.groupby("ts_code", sort=False, observed=True))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                stock_batch = [stock for _, stock in islice(grouped, batch_size)]
                if not stock_batch:
                    break
                featured_batch = list(executor.map(lambda stock: compute_stock_features(stock, settings), stock_batch))
                if not feature_names:
                    feature_names = [c for c in featured_batch[0].columns if c not in ("ts_code", "trade_date")]
                if output_start_date is not None:
                    featured_batch = [
                        featured.loc[featured["trade_date"] >= output_start_date].reset_index(drop=True)
                        for featured in featured_batch
                    ]
                    featured_batch = [featured for featured in featured_batch if not featured.empty]
                if not featured_batch:
                    continue
                for featured in featured_batch:
                    values = featured[feature_names]
                    null_counts.update({c: int(v) for c, v in values.isna().sum().items()})
                    inf_counts.update({c: int(np.isinf(values[c].to_numpy(dtype="float64", na_value=np.nan)).sum()) for c in feature_names})
                writer, batch = _flush_batch(featured_batch, writer, temporary, str(settings["parquet_compression"]))
                rows_written += len(batch)
                stock_count += len(featured_batch)
                output_key_hash = (output_key_hash + _hash_sum(batch[["ts_code", "trade_date"]])) & ((1 << 64) - 1)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, output_path)
    output_source = source
    if output_start_date is not None:
        output_source = output_source.loc[output_source["trade_date"] >= output_start_date]
    source_key_hash = _hash_sum(output_source[["ts_code", "trade_date"]])
    return {
        "input_rows": int(len(source)),
        "warmup_rows": int(len(source) - len(output_source)),
        "output_start_date": output_start_date,
        "rows": rows_written,
        "stocks": stock_count,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "file_bytes": output_path.stat().st_size,
        "source_key_hash": f"{source_key_hash:016x}",
        "output_key_hash": f"{output_key_hash:016x}",
        "key_hash_equal": source_key_hash == output_key_hash,
        "null_count": dict(null_counts),
        "finite_ratio": {
            name: float((rows_written - null_counts[name] - inf_counts[name]) / rows_written)
            for name in feature_names
        },
        "inf_count": dict(inf_counts),
    }


def build_hand_checks(base: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Independently calculate representative scalar formulas."""
    stock = next(group for _, group in base.groupby("ts_code", sort=False, observed=True) if group[["open", "high", "low", "close"]].notna().head(6).all(axis=1).all())
    stock = stock.sort_values("trade_date", kind="stable").head(6).reset_index(drop=True)
    features = compute_stock_features(stock, settings)
    checks = []

    def add(row: int, name: str, manual: float) -> None:
        computed = float(features.loc[row, name])
        checks.append({
            "ts_code": str(stock.loc[row, "ts_code"]),
            "trade_date": int(stock.loc[row, "trade_date"]),
            "feature": name,
            "manual": manual,
            "computed": computed,
            "abs_error": abs(manual - computed),
        })

    add(0, "intraday_return", float(stock.loc[0, "close"] / stock.loc[0, "open"] - 1.0))
    add(0, "high_low_range", float((stock.loc[0, "high"] - stock.loc[0, "low"]) / stock.loc[0, "close"]))
    add(1, "ret_1", float(stock.loc[1, "close"] / stock.loc[0, "close"] - 1.0))
    add(2, "ret_2", float(stock.loc[2, "close"] / stock.loc[0, "close"] - 1.0))
    high_5 = float(stock.loc[:4, "high"].max())
    low_5 = float(stock.loc[:4, "low"].min())
    add(4, "position_5", float((stock.loc[4, "close"] - low_5) / (high_5 - low_5)))
    return pd.DataFrame(checks)


def build_feature_artifacts(config: dict[str, Any], *, max_stocks: int | None = None) -> dict[str, Any]:
    paths = ProjectPaths.from_config(config)
    settings = config["features"]
    base = load_train(config, use_cache=True)[FEATURE_INPUTS]
    suffix = f"_smoke_{max_stocks}" if max_stocks is not None else ""
    if max_stocks is not None:
        selected = base["ts_code"].drop_duplicates().head(max_stocks)
        base = base[base["ts_code"].isin(selected)]

    def output_path(key: str) -> Path:
        path = Path(settings[key])
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")

    time_path = output_path("time_output")
    time_audit = build_time_feature_store(base, time_path, settings)
    market = compute_market_features(base)
    market_path = output_path("market_output")
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(market_path, engine="pyarrow", compression=str(settings["parquet_compression"]), index=False)
    hand_checks = build_hand_checks(base, settings)
    handcheck_path = output_path("handcheck_output")
    handcheck_path.parent.mkdir(parents=True, exist_ok=True)
    hand_checks.to_csv(handcheck_path, index=False)
    all_features = time_audit["feature_names"] + [c for c in market.columns if c != "trade_date"]
    dictionary = build_feature_dictionary(all_features)
    dictionary_path = output_path("dictionary_output")
    dictionary.to_csv(dictionary_path, index=False)
    report = {
        "phase": 3,
        "scope": "full" if max_stocks is None else f"smoke_{max_stocks}_stocks",
        "label_column_used_as_feature": False,
        "future_shift_used": False,
        "cross_sectional_rank_or_zscore_added": False,
        "time_features": time_audit,
        "market_features": {
            "rows": int(len(market)),
            "feature_count": len(market.columns) - 1,
            "feature_names": [c for c in market.columns if c != "trade_date"],
            "date_unique": bool(market["trade_date"].is_unique),
            "inf_count": int(np.isinf(market.drop(columns="trade_date").to_numpy(dtype="float64", na_value=np.nan)).sum()),
            "file_bytes": market_path.stat().st_size,
        },
        "total_feature_count": len(all_features),
        "hand_checks": {
            "rows": int(len(hand_checks)),
            "max_abs_error": float(hand_checks["abs_error"].max()),
        },
        "outputs": {
            "time_features": str(time_path), "market_features": str(market_path),
            "dictionary": str(dictionary_path), "hand_checks": str(handcheck_path),
        },
    }
    audit_path = output_path("audit_output")
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
