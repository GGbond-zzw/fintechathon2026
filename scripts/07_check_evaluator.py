"""Verify local Phase 7 metrics against the untouched official evaluator."""

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
from src.validation.parity import run_official_parity_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    report = run_official_parity_check(config, overwrite=args.overwrite)
    compact = {
        "official_sha256": report["official_contract"]["sha256"],
        "maximum_absolute_difference": report["maximum_absolute_difference"],
        "parity_passed": report["parity_passed"],
        "all_branch_checks_passed": report["all_branch_checks_passed"],
    }
    logger.info("Phase 7 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    return 0 if report["parity_passed"] and report["all_branch_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
