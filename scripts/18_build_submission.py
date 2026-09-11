"""Build and round-trip validate the Phase 18 competition submission."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.submission.validator import build_submission
from src.utils.config import load_config

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_submission(load_config(args.config), overwrite=args.overwrite)
    print(json.dumps({"submission": report["submission"],
                      "validation": report["postwrite_validation"]}, ensure_ascii=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
