"""Register Phase 8 baselines with immutable reproducibility manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import register_phase8_experiments
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite-audit", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    report = register_phase8_experiments(config, overwrite_audit=args.overwrite_audit)
    compact = {
        "source_tree_sha256": report["source_tree_sha256"],
        "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        "git_commit": report["git"]["commit"],
        "git_dirty": report["git"]["dirty"],
        "registry": report["registry"],
        "fold_records": report["fold_records"],
        "summary_records": report["summary_records"],
    }
    logger.info("Phase 9 summary: %s", compact)
    print(json.dumps(compact, ensure_ascii=False))
    checks = [
        report["registry_schema_exact"], report["registry_keys_unique"],
        report["fold_records"] == 9, report["summary_records"] == 3,
        report["registry"]["total"] >= 12, not report["test_data_used"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
