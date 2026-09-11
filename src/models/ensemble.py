"""Daily-rank OOF ensemble and Phase 11 experiment runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.registry import (
    append_registry_rows,
    artifact_entry,
    build_source_manifest,
    git_state,
    sha256_file,
    write_immutable,
)
from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores
from src.validation.evaluator import METRIC_KEYS, evaluate_merged_frame, validate_official_contract


KEYS = ["ts_code", "trade_date"]
META = [*KEYS, "y_ret_1d", "flag_limit_up"]
OOF_COLUMNS = [*META, "pred"]


def blend_daily_rank_predictions(
    predictions: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Blend aligned model predictions after independent same-date ranking."""
    if not predictions or set(predictions) != set(weights):
        raise ValueError("Prediction and weight model names must be identical and non-empty")
    normalized = {name: float(value) for name, value in weights.items()}
    if any(value < 0.0 or value > 1.0 for value in normalized.values()):
        raise ValueError("Ensemble weights must be in [0, 1]")
    if not np.isclose(sum(normalized.values()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Ensemble weights must sum to one")

    first_name = next(iter(predictions))
    first = predictions[first_name]
    missing = sorted({*KEYS, "pred"} - set(first.columns))
    if missing:
        raise ValueError(f"Missing ensemble columns for {first_name}: {missing}")
    if first.duplicated(KEYS).any():
        raise ValueError(f"Duplicate ensemble keys for {first_name}")
    base_keys = first[KEYS].reset_index(drop=True)
    blended = np.zeros(len(first), dtype="float64")
    for name, frame in predictions.items():
        missing = sorted({*KEYS, "pred"} - set(frame.columns))
        if missing:
            raise ValueError(f"Missing ensemble columns for {name}: {missing}")
        if not frame[KEYS].reset_index(drop=True).equals(base_keys):
            raise ValueError(f"Ensemble keys or row order differ for {name}")
        values = frame["pred"].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite ensemble predictions for {name}")
        ranks = frame.groupby("trade_date", sort=False)["pred"].rank(method="average", pct=True)
        blended += normalized[name] * ranks.to_numpy(dtype="float64")
    if not np.isfinite(blended).all():
        raise ValueError("Blended predictions must all be finite")
    return base_keys.assign(pred=blended)


def _load_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=OOF_COLUMNS, engine="pyarrow")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"OOF keys are not unique: {path}")
    if not np.isfinite(frame["pred"].to_numpy(dtype="float64")).all():
        raise ValueError(f"OOF predictions are not finite: {path}")
    return frame.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _load_aligned_sources(source_dir: Path, fold: str, models: list[str]) -> dict[str, pd.DataFrame]:
    frames = {model: _load_oof(source_dir / fold / f"{model}_oof.parquet") for model in models}
    reference = frames[models[0]]
    for model in models[1:]:
        current = frames[model]
        if not current[META].equals(reference[META]):
            raise ValueError(f"OOF keys or evaluator metadata differ: {fold}/{model}")
    return frames


def _postprocess(
    frames: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    alpha: float,
    exit_fraction: float,
    denominator: int,
    minimum_pool: int,
) -> np.ndarray:
    blended = blend_daily_rank_predictions(frames, weights)
    first = next(iter(frames.values()))
    transform = blended.assign(flag_limit_up=first["flag_limit_up"].to_numpy())
    smoothed, _ = causal_rank_ema(transform, alpha)
    result, _ = hysteresis_rank_scores(
        transform, smoothed, exit_fraction,
        top_fraction_denominator=denominator, minimum_pool_size=minimum_pool,
    )
    return result


def _score(reference: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    scored = reference[META].copy()
    scored["pred"] = np.asarray(predictions, dtype="float32")
    return {key: float(value) for key, value in evaluate_merged_frame(scored).items()}


def _candidate(lightgbm_weight: float) -> str:
    return f"lightgbm_{lightgbm_weight:.2f}__ridge_{1.0 - lightgbm_weight:.2f}"


def _summaries(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, lightgbm_weight, ridge_weight), part in folds.groupby(
        ["candidate", "lightgbm_weight", "ridge_weight"], sort=False
    ):
        row: dict[str, Any] = {
            "record_type": "summary", "fold": "all_equal_weight", "candidate": candidate,
            "lightgbm_weight": float(lightgbm_weight), "ridge_weight": float(ridge_weight),
        }
        for metric in METRIC_KEYS:
            row[metric] = float(part[metric].to_numpy(dtype="float64").mean())
        scores = part["final_score"].to_numpy(dtype="float64")
        row["final_score_std"] = float(scores.std(ddof=1))
        row["final_score_worst"] = float(scores.min())
        rows.append(row)
    return pd.DataFrame(rows)


def _select(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        ["final_score", "final_score_worst", "mean_turnover", "lightgbm_weight"],
        ascending=[False, False, True, False], kind="stable",
    ).iloc[0]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_ensemble_grid(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Evaluate fixed LightGBM/Ridge rank weights on all three OOF folds."""
    validate_official_contract(config)
    settings = config["ensemble"]
    models = [str(value) for value in settings["source_models"]]
    if models != ["lightgbm", "ridge"]:
        raise ValueError("Phase 11 source order must be [lightgbm, ridge]")
    if settings["selection_metric"] != "final_score" or settings["fold_aggregation"] != "equal_weight":
        raise ValueError("Phase 11 selection must use equal-weight fold final_score")
    lightgbm_weights = [float(value) for value in settings["lightgbm_weights"]]
    if len(lightgbm_weights) != 21 or len(set(lightgbm_weights)) != 21:
        raise ValueError("Phase 11 requires 21 unique predeclared LightGBM weights")
    expected_weights = np.linspace(0.0, 1.0, 21)
    if not np.allclose(lightgbm_weights, expected_weights, rtol=0.0, atol=1e-12):
        raise ValueError("Phase 11 LightGBM weights must span 0..1 in 0.05 steps")

    root = Path(config["paths"]["root"])
    source_dir = Path(settings["source_oof_dir"])
    cv_audit = json.loads(Path(config["cv"]["audit_output"]).read_text(encoding="utf-8"))
    folds = [fold["name"] for fold in cv_audit["folds"]]
    turnover = json.loads(Path(settings["turnover_audit"]).read_text(encoding="utf-8"))
    alpha = float(turnover["selected"]["alpha"])
    exit_fraction = float(turnover["selected"]["exit_fraction"])
    denominator = int(config["evaluator"]["top_fraction_denominator"])
    minimum_pool = int(config["evaluator"]["portfolio_min_samples"])

    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        frames = _load_aligned_sources(source_dir, fold, models)
        for lightgbm_weight in lightgbm_weights:
            ridge_weight = 1.0 - lightgbm_weight
            predictions = _postprocess(
                frames, {"lightgbm": lightgbm_weight, "ridge": ridge_weight},
                alpha=alpha, exit_fraction=exit_fraction,
                denominator=denominator, minimum_pool=minimum_pool,
            )
            fold_rows.append({
                "record_type": "fold", "fold": fold,
                "candidate": _candidate(lightgbm_weight),
                "lightgbm_weight": lightgbm_weight, "ridge_weight": ridge_weight,
                **_score(frames["lightgbm"], predictions),
            })
        del frames

    fold_grid = pd.DataFrame(fold_rows)
    summary = _summaries(fold_grid)
    selected = _select(summary)
    selected_weight = float(selected["lightgbm_weight"])
    selected_candidate = str(selected["candidate"])
    grid = pd.concat([fold_grid, summary], ignore_index=True, sort=False).reindex(columns=[
        "record_type", "fold", "candidate", "lightgbm_weight", "ridge_weight", *METRIC_KEYS,
        "final_score_std", "final_score_worst",
    ])

    lightgbm_control = summary[summary["lightgbm_weight"] == 1.0].iloc[0]
    equal_weight = summary[summary["lightgbm_weight"] == 0.5].iloc[0]
    phase10_delta = {
        metric: float(lightgbm_control[metric]) - float(turnover["selected"][metric])
        for metric in METRIC_KEYS
    }
    if max(abs(value) for value in phase10_delta.values()) != 0.0:
        raise ValueError("LightGBM-only ensemble control does not exactly reproduce Phase 10")

    output_dir = Path(settings["output_dir"])
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Phase 11 output exists; pass overwrite explicitly: {output_dir}")
    temporary_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    selected_fold_metrics = fold_grid[fold_grid["candidate"] == selected_candidate].to_dict("records")
    saved_oof: dict[str, dict[str, Any]] = {}
    try:
        for fold in folds:
            frames = _load_aligned_sources(source_dir, fold, models)
            predictions = _postprocess(
                frames, {"lightgbm": selected_weight, "ridge": 1.0 - selected_weight},
                alpha=alpha, exit_fraction=exit_fraction,
                denominator=denominator, minimum_pool=minimum_pool,
            )
            metrics = _score(frames["lightgbm"], predictions)
            expected = next(item for item in selected_fold_metrics if item["fold"] == fold)
            if any(float(metrics[key]) != float(expected[key]) for key in METRIC_KEYS):
                raise ValueError(f"Persisted ensemble metric mismatch: {fold}")
            output = frames["lightgbm"][META].copy()
            output["pred"] = predictions
            fold_dir = temporary_dir / fold
            fold_dir.mkdir(parents=True)
            path = fold_dir / "ensemble_oof.parquet"
            output.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
            saved_oof[fold] = {
                "path": str(output_dir / fold / path.name), "rows": int(len(output)),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
                "pred_dtype": str(output["pred"].dtype),
                "pred_finite": bool(np.isfinite(output["pred"].to_numpy(dtype="float64")).all()),
                "keys_unique": not bool(output.duplicated(KEYS).any()),
            }

        manifest = {
            "phase": 11, "source_models": models,
            "source_selection": {
                "simple_factor": "excluded_after_negative_phase8_baseline",
                "catboost": "unavailable_deferred_under_16gb_boundary",
            },
            "selection_metric": "final_score", "cv_aggregation": "equal_weight_across_folds",
            "candidate_count": len(lightgbm_weights), "lightgbm_weights": lightgbm_weights,
            "locked_turnover_postprocess": {
                "phase10_audit": str(Path(settings["turnover_audit"])),
                "alpha": alpha, "exit_fraction": exit_fraction,
            },
            "selected": {
                "candidate": selected_candidate, "lightgbm_weight": selected_weight,
                "ridge_weight": 1.0 - selected_weight,
                **{metric: float(selected[metric]) for metric in METRIC_KEYS},
                "final_score_std": float(selected["final_score_std"]),
                "final_score_worst": float(selected["final_score_worst"]),
                "folds": selected_fold_metrics,
            },
            "controls": {
                "phase10_lightgbm_only": {metric: float(lightgbm_control[metric]) for metric in METRIC_KEYS},
                "phase10_reproduction_delta": phase10_delta,
                "equal_weight": {metric: float(equal_weight[metric]) for metric in METRIC_KEYS},
            },
            "causal_contract": {
                "each_model_ranked_within_current_date": True,
                "weights_nonnegative_and_sum_to_one": True,
                "phase10_postprocess_frozen_before_ensemble_grid": True,
                "fold_state_reset": True, "labels_used_by_transform": False,
                "test_data_used": False,
            },
            "oof": saved_oof,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if output_dir.exists():
            backup = output_dir.with_name(f"{output_dir.name}.backup")
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(output_dir, backup)
            try:
                os.replace(temporary_dir, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    grid_parquet = Path(settings["grid_output_parquet"])
    grid_csv = Path(settings["grid_output_csv"])
    grid_parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = grid_parquet.with_suffix(".tmp.parquet")
    csv_tmp = grid_csv.with_suffix(".tmp.csv")
    grid.to_parquet(parquet_tmp, engine="pyarrow", compression="zstd", index=False)
    grid.to_csv(csv_tmp, index=False, lineterminator="\n")
    os.replace(parquet_tmp, grid_parquet)
    os.replace(csv_tmp, grid_csv)

    config_path = root / "config.yaml"
    config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase11_ensemble_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root)
    source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    source_oof_paths = [source_dir / fold / f"{model}_oof.parquet" for fold in folds for model in models]
    artifact_files = [
        *source_oof_paths, Path(settings["turnover_audit"]),
        Path(config["cv"]["audit_output"]),
        Path(config["baseline"]["model_view_output"]) / "manifest.json",
        Path(config["paths"]["official_evaluator"]), config_snapshot,
        output_dir / "manifest.json", grid_parquet, grid_csv,
        *[Path(item["path"]) for item in saved_oof.values()],
    ]
    artifact_manifest = {
        "phase": 11, "test_data_used": False, "source_tree_sha256": source["tree_sha256"],
        "feature_version": "model_view_v1",
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase11_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)

    git = git_state(root)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    feature_manifest_sha = sha256_file(Path(config["baseline"]["model_view_output"]) / "manifest.json")
    cv_by_name = {fold["name"]: fold for fold in cv_audit["folds"]}
    params = json.dumps({
        "source_models": models, "lightgbm_weights": lightgbm_weights,
        "selected_lightgbm_weight": selected_weight,
        "selected_ridge_weight": 1.0 - selected_weight,
        "turnover_alpha": alpha, "turnover_exit_fraction": exit_fraction,
    }, sort_keys=True, separators=(",", ":"))
    registry_rows = []
    for item in selected_fold_metrics:
        cv_fold = cv_by_name[item["fold"]]
        registry_rows.append({
            "experiment_id": settings["experiment_id"], "timestamp": timestamp,
            "record_type": "fold", "fold": item["fold"],
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
            "config_path": str(config_snapshot), "config_sha256": config_sha,
            "feature_version": "model_view_v1", "feature_manifest_sha256": feature_manifest_sha,
            "model": "daily_rank_lightgbm_ridge_ensemble", "model_params": params,
            "train_period": f"{cv_fold['actual']['train_start']}-{cv_fold['actual']['train_end']}",
            "val_period": f"{cv_fold['actual']['val_start']}-{cv_fold['actual']['val_end']}",
            "seed": config["project"]["seed"], **{key: item[key] for key in METRIC_KEYS},
            "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
            "artifact_paths": json.dumps([saved_oof[item["fold"]]["path"]], separators=(",", ":")),
            "notes": "daily_rank_blend_then_locked_phase10_postprocess_no_test_data",
        })
    registry_rows.append({
        "experiment_id": settings["experiment_id"], "timestamp": timestamp,
        "record_type": "summary", "fold": "all_equal_weight",
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
        "config_path": str(config_snapshot), "config_sha256": config_sha,
        "feature_version": "model_view_v1", "feature_manifest_sha256": feature_manifest_sha,
        "model": "daily_rank_lightgbm_ridge_ensemble", "model_params": params,
        "train_period": "three_expanding_folds", "val_period": "2022|2023|2024",
        "seed": config["project"]["seed"], **{key: float(selected[key]) for key in METRIC_KEYS},
        "final_score_std": float(selected["final_score_std"]),
        "final_score_worst": float(selected["final_score_worst"]),
        "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(output_dir), str(grid_parquet), str(grid_csv)], separators=(",", ":")),
        "notes": "equal_weight_fold_selection_by_official_final_score_no_test_data",
    })
    registry_result = append_registry_rows(Path(config["experiments"]["registry"]), registry_rows)
    audit = {
        **manifest, "source_tree_sha256": source["tree_sha256"],
        "source_file_count": source["file_count"], "source_manifest": str(source_path),
        "config_sha256": config_sha, "config_snapshot": str(config_snapshot),
        "artifact_manifest": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "grid_outputs": {"parquet": str(grid_parquet), "csv": str(grid_csv), "rows": int(len(grid))},
        "git": git, "registry": {"path": config["experiments"]["registry"], **registry_result},
    }
    _atomic_json(Path(settings["audit_output"]), audit)
    return audit
