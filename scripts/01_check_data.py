"""Phase 1 data audit with read-only raw inputs and Parquet verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import assess_float32_precision, load_test, load_train, write_parquet_cache
from src.data.validator import compare_cache_consistency, validate_frame
from src.utils.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ProjectPaths
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild Parquet caches from raw CSV files.")
    return parser.parse_args()


def audit_dataset(config: dict[str, Any], dataset: str, *, refresh_cache: bool) -> dict[str, Any]:
    paths = ProjectPaths.from_config(config)
    raw_path = paths.train if dataset == "train" else paths.test
    is_train = dataset == "train"
    required = list(config["data"]["test_required_columns"])
    if is_train:
        required.append(str(config["data"]["label"]))
    if not raw_path.is_file():
        return {
            "dataset": dataset,
            "status": "not_available",
            "file": str(raw_path),
            "message": "Configured raw file is absent; no substitute or synthetic label was used.",
        }
    load = load_train if is_train else load_test
    raw_frame = load(config, use_cache=False)
    validation = validate_frame(
        raw_frame,
        dataset_name=dataset,
        required_columns=required,
        id_cols=list(config["data"]["id_cols"]),
    )
    precision = assess_float32_precision(raw_path, config, is_train=is_train)
    if config["data"].get("parquet_cache", True):
        write_parquet_cache(raw_frame, config, dataset, raw_path)
        cached_frame = load(config, use_cache=True, refresh_cache=False)
        cache_check = compare_cache_consistency(
            raw_frame,
            cached_frame,
            id_cols=list(config["data"]["id_cols"]),
        )
        del cached_frame
    else:
        cache_check = {"status": "disabled"}
    del raw_frame
    return {
        "dataset": dataset,
        "status": validation["status"],
        "validation": validation,
        "float32_precision": precision,
        "parquet_cache": cache_check,
        "cache_refreshed": bool(refresh_cache),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_config(config)
    paths.create_generated_directories()
    logger = configure_logging(paths.logs, config["runtime"]["log_level"])
    set_global_seed(int(config["project"]["seed"]))
    report = {
        "phase": 1,
        "raw_inputs_mutated": False,
        "test_labels_created": False,
        "datasets": {
            "test": audit_dataset(config, "test", refresh_cache=args.refresh_cache),
            "train": audit_dataset(config, "train", refresh_cache=args.refresh_cache),
        },
    }
    output_path = Path(config["runtime"]["phase1_audit_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Phase 1 audit written to %s", output_path)
    logger.info("Test status: %s", report["datasets"]["test"]["status"])
    logger.info("Train status: %s", report["datasets"]["train"]["status"])
    test_cache = report["datasets"]["test"].get("parquet_cache", {})
    if test_cache.get("status") == "mismatch":
        logger.error("Test Parquet cache is inconsistent with raw CSV")
        return 1
    if report["datasets"]["test"]["status"] == "invalid_schema":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
