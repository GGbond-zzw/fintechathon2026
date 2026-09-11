"""Create year-bounded Phase 3 feature partitions for Phase 4 and CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.storage import repartition_parquet_by_year
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["features"]
    selected_columns = ["ts_code", "trade_date", *config["cross_sectional"]["source_features"]]
    report = repartition_parquet_by_year(
        Path(settings["time_output"]),
        Path(settings["model_base_by_year_output"]),
        compression=str(settings["parquet_compression"]),
        overwrite=args.overwrite,
        columns=selected_columns,
    )
    print(json.dumps({
        "rows": report["verified_partition_rows"],
        "row_count_equal": report["row_count_equal"],
        "keys_unique": report["all_partition_keys_unique"],
        "years": sorted(report["years"]),
        "columns": len(selected_columns),
    }, ensure_ascii=False))
    return 0 if report["row_count_equal"] and report["all_partition_keys_unique"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
