"""Parquet layout utilities for year-bounded downstream processing."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _key_hash_sum(frame: pd.DataFrame) -> int:
    hashed = pd.util.hash_pandas_object(frame[["ts_code", "trade_date"]], index=False)
    return int(hashed.sum()) & ((1 << 64) - 1)


def repartition_parquet_by_year(
    source_path: Path,
    output_dir: Path,
    *,
    compression: str = "zstd",
    overwrite: bool = False,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Rewrite selected stock-major columns into independently readable year files."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Feature source not found: {source_path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Partition output exists; pass overwrite explicitly: {output_dir}")
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    parquet = pq.ParquetFile(source_path)
    source_rows = parquet.metadata.num_rows
    writers: dict[int, pq.ParquetWriter] = {}
    row_counts: dict[int, int] = {}
    try:
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            years = pc.cast(pc.floor(pc.divide(table["trade_date"], 10000)), pa.int32())
            for year_value in pc.unique(years).to_pylist():
                year = int(year_value)
                subset = table.filter(pc.equal(years, year))
                year_dir = temporary / f"year={year}"
                year_dir.mkdir(parents=True, exist_ok=True)
                if year not in writers:
                    writers[year] = pq.ParquetWriter(
                        year_dir / "features.parquet", subset.schema, compression=compression
                    )
                    row_counts[year] = 0
                writers[year].write_table(subset)
                row_counts[year] += subset.num_rows
    finally:
        for writer in writers.values():
            writer.close()
        parquet.close()
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

    verified_rows = 0
    verified_hash = 0
    files: dict[str, dict[str, Any]] = {}
    for year in sorted(row_counts):
        path = output_dir / f"year={year}" / "features.parquet"
        metadata = pq.ParquetFile(path).metadata
        keys = pd.read_parquet(path, columns=["ts_code", "trade_date"], engine="pyarrow")
        key_hash = _key_hash_sum(keys)
        duplicate_rows = int(keys.duplicated(["ts_code", "trade_date"], keep=False).sum())
        verified_rows += metadata.num_rows
        verified_hash = (verified_hash + key_hash) & ((1 << 64) - 1)
        files[str(year)] = {
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
            "bytes": path.stat().st_size,
            "key_hash": f"{key_hash:016x}",
            "duplicate_key_rows": duplicate_rows,
        }
    report = {
        "source": str(source_path),
        "output": str(output_dir),
        "selected_columns": columns,
        "source_rows": source_rows,
        "verified_partition_rows": verified_rows,
        "row_count_equal": source_rows == verified_rows,
        "partition_key_hash": f"{verified_hash:016x}",
        "all_partition_keys_unique": all(item["duplicate_key_rows"] == 0 for item in files.values()),
        "years": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
