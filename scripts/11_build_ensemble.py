"""Evaluate and register the Phase 11 daily-rank ensemble grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import run_ensemble_grid
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
    report = run_ensemble_grid(config, overwrite=args.overwrite)
    compact = {
        "selected": report["selected"], "candidate_count": report["candidate_count"],
        "grid_rows": report["grid_outputs"]["rows"], "registry": report["registry"],
        "test_data_used": report["causal_contract"]["test_data_used"],
    }
    logger.info("Phase 11 selected: %s", compact["selected"])
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["candidate_count"] == 21, report["grid_outputs"]["rows"] == 84,
        max(abs(value) for value in report["controls"]["phase10_reproduction_delta"].values()) == 0.0,
        not report["causal_contract"]["labels_used_by_transform"],
        not report["causal_contract"]["test_data_used"],
        all(item["pred_finite"] and item["keys_unique"] for item in report["oof"].values()),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
