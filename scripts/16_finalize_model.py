"""Select and fit the locked Phase 16 final model configuration."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.models.finalization import run_finalization
from src.utils.config import load_config

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args(); report = run_finalization(load_config(args.config), overwrite=args.overwrite, select_only=args.select_only)
    print(json.dumps({"candidate_count": report["candidate_count"], "selected": report["selected"]}, ensure_ascii=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
