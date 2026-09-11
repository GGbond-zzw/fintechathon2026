"""Build the reusable model view and train Phase 8 walk-forward baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import run_baselines
from src.models.dataset import build_training_model_view, load_model_view_manifest
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing baseline outputs.")
    parser.add_argument("--rebuild-view", action="store_true", help="Replace the year-partitioned model view.")
    parser.add_argument("--only-build-view", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    view_path = Path(config["baseline"]["model_view_output"])
    if args.rebuild_view or not view_path.is_dir():
        view = build_training_model_view(config, overwrite=args.rebuild_view)
    else:
        view = load_model_view_manifest(config)
    logger.info("Model view: rows=%s, features=%s", view["rows"], view["feature_count"])
    if args.only_build_view:
        print(json.dumps({"model_view_rows": view["rows"], "feature_count": view["feature_count"]}))
        return 0
    report = run_baselines(config, overwrite=args.overwrite)
    compact = {
        model: {
            "final_score_mean": values["summary"]["final_score"]["mean"],
            "ic_mean": values["summary"]["ic_mean"]["mean"],
            "annual_excess": values["summary"]["annual_excess"]["mean"],
            "mean_turnover": values["summary"]["mean_turnover"]["mean"],
        }
        for model, values in report["models"].items()
    }
    logger.info("Phase 8 model summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        not report["test_data_used"],
        report["cv_aggregation"] == "equal_weight_across_folds",
        all(len(model["folds"]) == 3 for model in report["models"].values()),
        all(
            fold["validation_rows"] == 1_125_300
            for model in report["models"].values() for fold in model["folds"]
        ),
        report["oof_contract"]["metrics_computed_at_persisted_precision"],
        report["oof_contract"]["keys_consistent_across_models_per_fold"],
        report["oof_contract"]["all_keys_unique"],
        report["oof_contract"]["all_predictions_finite"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
