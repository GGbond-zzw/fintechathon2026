from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from src.validation.evaluator import (
    METRIC_KEYS,
    evaluate_files,
    evaluation_diagnostics,
    merge_evaluation_inputs,
    validate_official_contract,
)
from src.validation.parity import build_official_parity_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PATH = PROJECT_ROOT / "official" / "evaluate.py"
OFFICIAL_HASH = "72DC59987E92BA8AE6B506E114DFE8591B5608C9BECCD0769D5D1F778F9189C2"


def evaluator_config() -> dict:
    return {
        "paths": {"official_evaluator": str(OFFICIAL_PATH)},
        "evaluator": {
            "official_sha256": OFFICIAL_HASH,
            "constants_locked": True,
            "ic_min_samples": 30,
            "portfolio_min_samples": 100,
            "top_fraction_denominator": 10,
            "annualization_days": 252,
            "weights": {"rank_ic": 0.4, "annual_excess": 0.3, "one_minus_turnover": 0.3},
        },
    }


def load_official():
    spec = importlib.util.spec_from_file_location("official_test_evaluate", OFFICIAL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluatorTest(TestCase):
    def test_all_metrics_match_official_file_evaluator(self) -> None:
        predictions, labels, features = build_official_parity_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions.to_csv(root / "submission.csv", index=False)
            labels.to_csv(root / "测试集_Y.csv", index=False)
            features.to_csv(root / "测试集_X.csv", index=False)
            official = load_official().evaluate(str(root / "submission.csv"), str(root))
            local = evaluate_files(root / "submission.csv", root)
        self.assertEqual(tuple(local), METRIC_KEYS)
        for key in METRIC_KEYS:
            self.assertEqual(float(local[key]), float(official[key]), key)
        expected_score = (
            local["ic_mean"] * 0.4
            + local["annual_excess"] * 0.3
            + (1 - local["mean_turnover"]) * 0.3
        )
        self.assertEqual(float(local["final_score"]), float(expected_score))

    def test_thresholds_floor_top_sizes_and_turnover_reset(self) -> None:
        predictions, labels, features = build_official_parity_fixture()
        diagnostics = evaluation_diagnostics(
            merge_evaluation_inputs(predictions, labels, features)
        )
        dates = diagnostics["by_date"]
        self.assertFalse(dates[0]["rank_ic_eligible"])
        self.assertTrue(dates[1]["rank_ic_eligible"])
        self.assertFalse(dates[2]["portfolio_eligible"])
        self.assertTrue(dates[3]["portfolio_eligible"])
        self.assertEqual(dates[5]["portfolio_valid_rows"], 109)
        self.assertEqual(dates[5]["portfolio_top_n"], 10)
        self.assertEqual(dates[5]["turnover_valid_rows"], 110)
        self.assertEqual(dates[5]["turnover_top_n"], 11)
        self.assertEqual(diagnostics["turnover_comparisons"], 2)

    def test_contract_rejects_constant_weight_and_hash_drift(self) -> None:
        contract = validate_official_contract(evaluator_config())
        self.assertEqual(contract["sha256"], OFFICIAL_HASH)
        changed = evaluator_config()
        changed["evaluator"]["ic_min_samples"] = 31
        with self.assertRaises(ValueError):
            validate_official_contract(changed)
        changed = evaluator_config()
        changed["evaluator"]["weights"]["rank_ic"] = 0.5
        with self.assertRaises(ValueError):
            validate_official_contract(changed)
        changed = evaluator_config()
        changed["evaluator"]["official_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_official_contract(changed)

