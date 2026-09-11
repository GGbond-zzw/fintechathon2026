"""Build Phase 3 leakage-safe V1 feature artifacts from the real training set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_feature_artifacts
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--max-stocks", type=int, default=None, help="Smoke-test a deterministic prefix of stocks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_stocks is not None and args.max_stocks < 1:
        raise ValueError("--max-stocks must be positive")
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    report = build_feature_artifacts(config, max_stocks=args.max_stocks)
    logger.info("Feature scope: %s", report["scope"])
    logger.info("Rows: %s", report["time_features"]["rows"])
    logger.info("Feature count: %s", report["total_feature_count"])
    logger.info("Key hash equal: %s", report["time_features"]["key_hash_equal"])
    logger.info("Maximum hand-check error: %.3g", report["hand_checks"]["max_abs_error"])
    checks = [
        100 <= report["total_feature_count"] <= 300,
        report["time_features"]["key_hash_equal"],
        sum(report["time_features"]["inf_count"].values()) == 0,
        report["market_features"]["date_unique"],
        report["market_features"]["inf_count"] == 0,
        report["hand_checks"]["max_abs_error"] <= 1e-6,
        not report["label_column_used_as_feature"],
        not report["future_shift_used"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
