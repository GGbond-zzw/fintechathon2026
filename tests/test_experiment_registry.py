from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.experiments.registry import append_registry_rows, build_source_manifest


class ExperimentRegistryTest(TestCase):
    def test_append_is_idempotent_and_rejects_conflicting_key_reuse(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            row = {
                "experiment_id": "exp1", "timestamp": "2026-01-01T00:00:00+08:00",
                "record_type": "fold", "fold": "fold_1", "model": "ridge",
                "final_score": 0.1, "notes": "same",
            }
            first = append_registry_rows(path, [row])
            second = append_registry_rows(path, [{**row, "timestamp": "later"}])
            self.assertEqual(first, {"added": 1, "skipped_identical": 0, "total": 1})
            self.assertEqual(second, {"added": 0, "skipped_identical": 1, "total": 1})
            with self.assertRaises(ValueError):
                append_registry_rows(path, [{**row, "notes": "changed"}])

    def test_source_tree_hash_is_deterministic_and_content_sensitive(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "scripts").mkdir()
            (root / "tests").mkdir()
            (root / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
            source = root / "src" / "logic.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            first = build_source_manifest(root)
            second = build_source_manifest(root)
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            source.write_text("VALUE = 2\n", encoding="utf-8")
            changed = build_source_manifest(root)
            self.assertNotEqual(first["tree_sha256"], changed["tree_sha256"])

