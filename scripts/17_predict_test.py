"""Generate locked Phase 17 test predictions without creating a submission CSV."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.models.inference import run_test_inference
from src.utils.config import load_config

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); report = run_test_inference(load_config(args.config), overwrite=args.overwrite)
    print(json.dumps({"prediction": report["prediction"], "key_contract": report["key_contract"],
                      "state_transfer": report["state_transfer"]}, ensure_ascii=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
