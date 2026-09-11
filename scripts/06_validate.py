"""Build and audit Phase 6 expanding-window validation folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.validation.splitter import build_cv_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    report = build_cv_artifacts(config, overwrite=args.overwrite)
    compact = {
        "cv_type": report["cv_type"],
        "purge_trade_days": report["purge_trade_days"],
        "all_folds_valid": report["all_folds_valid"],
        "folds": [
            {
                "name": fold["name"],
                "train_end": fold["actual"]["train_end"],
                "purge_dates": fold["actual"]["purge_dates"],
                "val_start": fold["actual"]["val_start"],
            }
            for fold in report["folds"]
        ],
    }
    logger.info("Phase 6 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["cv_type"] == "expanding_window",
        report["purge_covers_label_horizon"],
        report["all_folds_valid"],
        not report["random_split_used"],
        all(fold["trade_date_counts"]["purge"] == report["purge_trade_days"] for fold in report["folds"]),
        all(fold["labeled_row_counts"]["train"] <= fold["row_counts"]["train"] for fold in report["folds"]),
        all(fold["labeled_row_counts"]["validation"] <= fold["row_counts"]["validation"] for fold in report["folds"]),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
