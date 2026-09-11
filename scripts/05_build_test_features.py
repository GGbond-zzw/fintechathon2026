"""Build Phase 5 test features using the real training tail as warm-up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.inference_pipeline import build_test_feature_artifacts
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None, help="Smoke-test a deterministic test-stock prefix.")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    report = build_test_feature_artifacts(
        config, overwrite=args.overwrite, max_stocks=args.max_stocks
    )
    compact = {
        "scope": report["scope"],
        "test_rows": report["test_rows"],
        "warmup_sufficient": report["warmup_sufficient"],
        "all_key_hashes_equal": report["all_key_hashes_equal"],
        "first_test_date_probes_not_all_missing": report["first_test_date_probes_not_all_missing"],
        "schema_equal_to_train": report["schema_equal_to_train"],
        "raw_files_unchanged": report["raw_files_unchanged"],
    }
    logger.info("Phase 5 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["warmup_sufficient"],
        report["all_key_hashes_equal"],
        report["first_test_date_probes_not_all_missing"],
        all(report["schema_equal_to_train"].values()),
        report["market_features"]["first_date_valid_return_count"] > 0,
        report["market_features"]["inf_count"] == 0,
        report["raw_files_unchanged"],
        not report["label_column_present_in_test_input"],
        not report["label_column_present_in_any_output"],
        not report["test_labels_created_or_inferred"],
        not report["future_shift_used"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
