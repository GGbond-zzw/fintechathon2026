"""Build Phase 4 same-date percentile ranks and z-scores by year."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.cross_sectional import build_cross_sectional_store
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_cross_sectional_store(load_config(args.config), overwrite=args.overwrite)
    compact = {
        "rows": report["rows"],
        "source_features": report["source_feature_count"],
        "output_features": report["source_feature_count"] * 2,
        "key_hash_equal": report["key_hash_equal"],
        "years": sorted(report["years"]),
    }
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["key_hash_equal"],
        not report["future_dates_used"],
        all(year["key_unique"] for year in report["years"].values()),
        all(year["sorted_by_date_code"] for year in report["years"].values()),
        all(year["inf_count"] == 0 for year in report["years"].values()),
        all(0.0 < year["rank_min"] <= year["rank_max"] <= 1.0 for year in report["years"].values()),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
