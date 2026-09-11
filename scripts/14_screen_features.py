"""Run Phase 14 leakage-safe advanced feature screening and family ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.screening import run_feature_screening
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
    report = run_feature_screening(config, overwrite=args.overwrite)
    compact = {
        "screened": report["feature_count_screened"],
        "selected": report["selected_feature_count"],
        "selected_features": report["selected_features"],
        "holdout_proxy": report["all_selected_holdout_equal_weight"],
        "raw_train_unchanged": report["contracts"]["raw_train_unchanged"],
        "registry": report["registry"],
    }
    logger.info("Phase 14 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["feature_count_screened"] == 32,
        report["selected_feature_count"] > 0,
        report["daily_metric_rows"] == report["daily_metric_dates"] * 32,
        report["contracts"]["selection_uses_screen_period_only"],
        not report["contracts"]["holdout_metrics_used_for_selection"],
        not report["contracts"]["test_data_read"],
        not report["contracts"]["test_labels_created_or_inferred"],
        not report["contracts"]["predictive_model_trained"],
        report["contracts"]["raw_train_unchanged"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
