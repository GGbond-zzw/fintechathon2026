"""Build and audit Phase 13 causal advanced feature stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.phase13 import build_phase13
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    report = build_phase13(config, overwrite=args.overwrite, max_stocks=args.max_stocks)
    compact = {
        "scope": report["scope"], "features": report["feature_count"],
        "train_rows": report["train"]["rows"], "test_rows": report["test"]["rows"],
        "warmup_sufficient": report["warmup_sufficient"],
        "train_test_schema_equal": report["train_test_schema_equal"],
        "raw_files_unchanged": report["contracts"]["raw_files_unchanged"],
        "registry": report["registry"],
    }
    logger.info("Phase 13 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["feature_count"] == 32, report["train"]["key_hash_equal"],
        report["test"]["key_hash_equal"], report["train"]["all_keys_unique"],
        report["test"]["all_keys_unique"], report["warmup_sufficient"],
        report["train_test_schema_equal"], report["hand_checks"]["max_abs_error"] <= 1e-6,
        sum(report["train"]["inf_count"].values()) == 0,
        sum(report["test"]["inf_count"].values()) == 0,
        not report["contracts"]["label_used"], not report["contracts"]["future_shift_used"],
        not report["contracts"]["test_labels_created_or_inferred"],
        not report["contracts"]["cross_sectional_screening_performed"],
        not report["contracts"]["model_training_performed"],
        report["contracts"]["raw_files_unchanged"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
