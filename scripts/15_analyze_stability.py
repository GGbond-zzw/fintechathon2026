"""Run Phase 15 ATR and training-window stability experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stability import run_stability_experiments
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    report = run_stability_experiments(config, overwrite=args.overwrite, resume=args.resume)
    compact = {
        "features": report["feature_count"],
        "advanced": report["advanced_features"],
        "selected_window": report["selected_window_for_phase16_review"],
        "incumbent": report["incumbent"],
        "raw_files_unchanged": report["contracts"]["raw_files_unchanged"],
        "registry": report["registry"],
    }
    logger.info("Phase 15 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["feature_count"] == 49,
        report["advanced_features"] == ["atr_norm_14", "atr_norm_28"],
        report["contracts"]["same_cv_validation_dates"],
        report["contracts"]["one_day_purge_preserved"],
        report["contracts"]["training_windows_use_past_dates_only"],
        not report["contracts"]["test_data_used_for_training_or_selection"],
        report["contracts"]["raw_files_unchanged"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
