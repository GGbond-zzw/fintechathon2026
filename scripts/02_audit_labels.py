"""Audit real training labels and their time alignment; never create features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_audit import (
    audit_label_alignment,
    label_distribution,
    load_label_audit_columns,
    load_price_audit_columns,
    sha256_file,
    time_breakdowns,
)
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = config["label_audit"]
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    label = str(config["data"]["label"])
    raw_hash_before = sha256_file(paths.train)
    frame = load_label_audit_columns(paths.train, label)
    future_prices = load_price_audit_columns(paths.test)
    distribution = label_distribution(
        frame,
        label,
        list(settings["quantiles"]),
        list(settings["extreme_abs_thresholds"]),
    )
    yearly, daily = time_breakdowns(frame, label)
    alignment, mismatch_samples = audit_label_alignment(
        frame,
        label,
        absolute_tolerance=float(settings["absolute_tolerance"]),
        relative_tolerance=float(settings["relative_tolerance"]),
        mismatch_sample_rows=int(settings["mismatch_sample_rows"]),
        future_prices=future_prices,
    )
    yearly_path = Path(settings["yearly_output_csv"])
    daily_path = Path(settings["daily_output_parquet"])
    mismatch_path = Path(settings["mismatch_output_csv"])
    for path in (yearly_path, daily_path, mismatch_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(yearly_path, index=False)
    daily.to_parquet(daily_path, engine="pyarrow", compression="zstd", index=False)
    mismatch_samples.to_csv(mismatch_path, index=False)
    raw_hash_after = sha256_file(paths.train)
    report = {
        "phase": 2,
        "source": str(paths.train),
        "source_sha256_before": raw_hash_before,
        "source_sha256_after": raw_hash_after,
        "source_unchanged": raw_hash_before == raw_hash_after,
        "future_values_used_for": "training-label boundary alignment audit only",
        "test_prices_used_for_boundary_only": True,
        "test_labels_created": False,
        "features_written": False,
        "labels_modified_or_filled": False,
        "distribution": distribution,
        "alignment": alignment,
        "outputs": {
            "yearly_csv": str(yearly_path),
            "daily_parquet": str(daily_path),
            "mismatch_samples_csv": str(mismatch_path),
        },
    }
    output_path = Path(settings["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Phase 2 label audit written to %s", output_path)
    logger.info("Best-supported alignment: %s", alignment["best_supported_definition"])
    logger.info("Training CSV unchanged: %s", report["source_unchanged"])
    immediate = alignment["methods"]["immediate_next_row"]
    return 0 if report["source_unchanged"] and immediate["match_rate_on_comparable"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
