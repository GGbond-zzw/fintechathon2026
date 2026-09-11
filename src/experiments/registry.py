"""Register completed experiments with immutable config, source and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.validation.evaluator import METRIC_KEYS


REGISTRY_COLUMNS = [
    "experiment_id", "timestamp", "record_type", "fold", "git_commit", "git_dirty",
    "source_tree_sha256", "source_manifest_path", "config_path", "config_sha256",
    "feature_version", "feature_manifest_sha256", "model", "model_params",
    "train_period", "val_period", "seed", *METRIC_KEYS,
    "final_score_std", "final_score_worst", "artifact_manifest_path",
    "artifact_manifest_sha256", "artifact_paths", "notes", "record_sha256",
]


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest().upper()


def build_source_manifest(root: Path) -> dict[str, Any]:
    """Fingerprint executable project inputs without generated data or documentation cycles."""
    candidates = [
        root / "AGENTS.md", root / "config.yaml", root / "requirements.txt",
        root / "requirements-lock.txt", root / "official" / "evaluate.py",
    ]
    for directory in (root / "src", root / "scripts", root / "tests"):
        candidates.extend(directory.rglob("*.py"))
    files = []
    for path in sorted({path.resolve() for path in candidates if path.is_file()}):
        relative = path.relative_to(root.resolve()).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(item["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256(path_nul_file_sha256_newline)",
        "file_count": len(files),
        "tree_sha256": aggregate.hexdigest().upper(),
        "files": files,
    }


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Immutable artifact content differs: {path}")
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "UNBORN",
        "dirty": bool(status.stdout.strip()),
        "status_lines": status.stdout.splitlines(),
    }


def artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment artifact missing: {path}")
    try:
        display = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_phase8_artifact_manifest(config: dict[str, Any], source: dict[str, Any], config_snapshot: Path) -> dict[str, Any]:
    root = Path(config["paths"]["root"])
    baseline_dir = Path(config["baseline"]["output_dir"])
    model_view_dir = Path(config["baseline"]["model_view_output"])
    core = [
        Path(config["paths"]["train"]),
        Path(config["baseline"]["audit_output"]),
        Path(config["cv"]["audit_output"]),
        Path(config["cv"]["calendar_output"]),
        Path(config["evaluator"]["audit_output"]),
        root / "reports" / "tables" / "phase8_framework_review.json",
        model_view_dir / "manifest.json",
        baseline_dir / "manifest.json",
        root / "requirements-lock.txt",
        Path(config["paths"]["official_evaluator"]),
        config_snapshot,
    ]
    core.extend(sorted(model_view_dir.glob("year=*/data.parquet")))
    core.extend(sorted(path for path in baseline_dir.rglob("*") if path.is_file() and path.name != "manifest.json"))
    entries = [artifact_entry(path, root) for path in core]
    return {
        "phase": 8,
        "test_data_used": False,
        "source_tree_sha256": source["tree_sha256"],
        "feature_version": "model_view_v1",
        "artifacts": entries,
    }


def _record_digest(row: dict[str, str]) -> str:
    payload = {key: row[key] for key in REGISTRY_COLUMNS if key not in ("timestamp", "record_sha256")}
    return _json_sha256(payload)


def append_registry_rows(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Atomically append new keys, skip identical keys and reject conflicting reuse."""
    normalized = []
    for raw in rows:
        row = {column: "" if raw.get(column) is None else str(raw.get(column, "")) for column in REGISTRY_COLUMNS}
        row["record_sha256"] = _record_digest(row)
        normalized.append(row)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(existing.columns) != REGISTRY_COLUMNS:
            raise ValueError("Existing experiment registry schema differs")
    else:
        existing = pd.DataFrame(columns=REGISTRY_COLUMNS)
    key_columns = ["experiment_id", "record_type", "fold"]
    existing_keys = {
        tuple(row[column] for column in key_columns): row["record_sha256"]
        for _, row in existing.iterrows()
    }
    additions = []
    skipped = 0
    for row in normalized:
        key = tuple(row[column] for column in key_columns)
        if key in existing_keys:
            if existing_keys[key] != row["record_sha256"]:
                raise ValueError(f"Experiment registry key reused with different content: {key}")
            skipped += 1
            continue
        existing_keys[key] = row["record_sha256"]
        additions.append(row)
    combined = pd.concat([existing, pd.DataFrame(additions, columns=REGISTRY_COLUMNS)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    combined.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)
    return {"added": len(additions), "skipped_identical": skipped, "total": len(combined)}


def _model_params(config: dict[str, Any], model: str) -> dict[str, Any]:
    return dict(config["baseline"][model if model != "simple_factor" else "simple_factor"])


def register_phase8_experiments(config: dict[str, Any], *, overwrite_audit: bool = False) -> dict[str, Any]:
    root = Path(config["paths"]["root"])
    settings = config["experiments"]
    audit_path = Path(settings["audit_output"])
    if audit_path.exists() and not overwrite_audit:
        raise FileExistsError(f"Phase 9 audit exists; pass overwrite explicitly: {audit_path}")
    baseline_audit = json.loads(Path(config["baseline"]["audit_output"]).read_text(encoding="utf-8"))
    cv_audit = json.loads(Path(config["cv"]["audit_output"]).read_text(encoding="utf-8"))
    feature_manifest_path = Path(config["baseline"]["model_view_output"]) / "manifest.json"
    feature_manifest_sha = sha256_file(feature_manifest_path)
    config_path = root / "config.yaml"
    config_sha = sha256_file(config_path)
    config_snapshot = Path(settings["configs_dir"]) / f"phase8_baselines_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())

    source = build_source_manifest(root)
    source_path = Path(settings["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_manifest = build_phase8_artifact_manifest(config, source, config_snapshot)
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(settings["manifests_dir"]) / f"phase8_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)
    git = git_state(root)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    cv_by_name = {fold["name"]: fold for fold in cv_audit["folds"]}
    rows = []
    for model, payload in baseline_audit["models"].items():
        experiment_id = str(settings["phase8_experiment_ids"][model])
        params = json.dumps(_model_params(config, model), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for fold in payload["folds"]:
            cv_fold = cv_by_name[fold["name"]]
            model_path = Path(config["baseline"]["output_dir"]) / fold["name"] / (
                "ridge_model.npz" if model == "ridge" else "lightgbm_model.txt"
            )
            artifact_paths = [fold["oof"]["path"]]
            if model != "simple_factor":
                artifact_paths.append(str(model_path))
            row = {
                "experiment_id": experiment_id, "timestamp": timestamp,
                "record_type": "fold", "fold": fold["name"],
                "git_commit": git["commit"], "git_dirty": git["dirty"],
                "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
                "config_path": str(config_snapshot), "config_sha256": config_sha,
                "feature_version": baseline_audit["feature_view"]["version"],
                "feature_manifest_sha256": feature_manifest_sha,
                "model": model, "model_params": params,
                "train_period": "not_applicable" if model == "simple_factor" else (
                    f"{cv_fold['actual']['train_start']}-{cv_fold['actual']['train_end']}"
                ),
                "val_period": f"{cv_fold['actual']['val_start']}-{cv_fold['actual']['val_end']}",
                "seed": config["project"]["seed"],
                **{key: fold["metrics"][key] for key in METRIC_KEYS},
                "artifact_manifest_path": str(artifact_path),
                "artifact_manifest_sha256": artifact_sha,
                "artifact_paths": json.dumps(artifact_paths, ensure_ascii=False, separators=(",", ":")),
                "notes": "full_validation_universe_float32_oof",
            }
            rows.append(row)
        summary = payload["summary"]
        rows.append({
            "experiment_id": experiment_id, "timestamp": timestamp,
            "record_type": "summary", "fold": "all_equal_weight",
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
            "config_path": str(config_snapshot), "config_sha256": config_sha,
            "feature_version": baseline_audit["feature_view"]["version"],
            "feature_manifest_sha256": feature_manifest_sha,
            "model": model, "model_params": params,
            "train_period": "three_expanding_folds", "val_period": "2022|2023|2024",
            "seed": config["project"]["seed"],
            **{key: summary[key]["mean"] for key in METRIC_KEYS},
            "final_score_std": summary["final_score"]["std"],
            "final_score_worst": summary["final_score"]["min"],
            "artifact_manifest_path": str(artifact_path),
            "artifact_manifest_sha256": artifact_sha,
            "artifact_paths": json.dumps(
                [str(Path(config["baseline"]["output_dir"])), str(Path(config["baseline"]["audit_output"]))],
                ensure_ascii=False, separators=(",", ":"),
            ),
            "notes": "equal_weight_fold_summary_not_concatenated_turnover",
        })
    registry_path = Path(settings["registry"])
    registry_result = append_registry_rows(registry_path, rows)
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    registered = registry[registry["experiment_id"].isin(settings["phase8_experiment_ids"].values())]
    report = {
        "phase": 9,
        "source_manifest": str(source_path),
        "source_tree_sha256": source["tree_sha256"],
        "source_file_count": source["file_count"],
        "config_snapshot": str(config_snapshot),
        "config_sha256": config_sha,
        "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha,
        "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "git": git,
        "registry": {"path": str(registry_path), **registry_result},
        "registry_schema_exact": list(registry.columns) == REGISTRY_COLUMNS,
        "registry_keys_unique": not bool(registry.duplicated(["experiment_id", "record_type", "fold"]).any()),
        "registered_experiment_ids": list(settings["phase8_experiment_ids"].values()),
        "fold_records": int((registered["record_type"] == "fold").sum()),
        "summary_records": int((registered["record_type"] == "summary").sum()),
        "test_data_used": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, audit_path)
    return report
