"""Train, evaluate and register the Phase 12 LambdaRank baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ranking import run_lambdarank
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    report = run_lambdarank(config, overwrite=args.overwrite)
    compact = {
        name: {
            "final_score": value["summary"]["final_score"]["mean"],
            "ic_mean": value["summary"]["ic_mean"]["mean"],
            "annual_excess": value["summary"]["annual_excess"]["mean"],
            "mean_turnover": value["summary"]["mean_turnover"]["mean"],
        }
        for name, value in report["variants"].items()
    }
    logger.info("Phase 12 summary: %s", compact)
    print(json.dumps({"variants": compact, "selected_variant": report["selected_variant"],
                      "registry": report["registry"], "test_data_used": report["contracts"]["test_data_used"]},
                     ensure_ascii=False))
    checks = [
        report["contracts"]["training_groups_contiguous_by_date"],
        report["contracts"]["group_sizes_sum_to_labeled_training_rows"],
        report["contracts"]["features_match_phase8_regression"],
        report["contracts"]["boosting_rounds_match_phase8_regression"],
        not report["contracts"]["test_data_used"],
        all(len(value["folds"]) == 3 for value in report["variants"].values()),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
