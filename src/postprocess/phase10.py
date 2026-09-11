"""Run and register the Phase 10 causal turnover-control grid."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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
OOF_COLUMNS = [*KEYS, "y_ret_1d", "flag_limit_up", "pred"]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_fold(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=OOF_COLUMNS, engine="pyarrow")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"OOF keys are not unique: {path}")
    if not np.isfinite(frame["pred"].to_numpy(dtype="float64")).all():
        raise ValueError(f"OOF predictions are not finite: {path}")
    return frame.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def _score(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    scored = frame[[*KEYS, "y_ret_1d", "flag_limit_up"]].copy()
    scored["pred"] = np.asarray(predictions, dtype="float32")
    return {key: float(value) for key, value in evaluate_merged_frame(scored).items()}


def _summarize(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, alpha, exit_fraction), part in grid.groupby(
        ["candidate", "alpha", "exit_fraction"], sort=False
    ):
        row: dict[str, Any] = {
            "record_type": "summary", "fold": "all_equal_weight",
            "candidate": candidate, "alpha": float(alpha), "exit_fraction": float(exit_fraction),
        }
        for metric in METRIC_KEYS:
            values = part[metric].to_numpy(dtype="float64")
            row[metric] = float(values.mean())
        scores = part["final_score"].to_numpy(dtype="float64")
        row["final_score_std"] = float(scores.std(ddof=1))
        row["final_score_worst"] = float(scores.min())
        rows.append(row)
    return pd.DataFrame(rows)


def _select(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        ["final_score", "final_score_worst", "mean_turnover", "alpha", "exit_fraction"],
        ascending=[False, False, True, False, True], kind="stable",
    ).iloc[0]


def _candidate(alpha: float, exit_fraction: float) -> str:
    return f"alpha_{alpha:.3f}__exit_{exit_fraction:.3f}"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_turnover_grid(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Evaluate the fixed OOF grid, persist the winner, and register the experiment."""
    validate_official_contract(config)
    root = Path(config["paths"]["root"])
    settings = config["turnover"]
    if settings["source_model"] != "lightgbm":
        raise ValueError("Phase 10 is locked to the Phase 8 winning LightGBM baseline")
    if settings["selection_metric"] != "final_score" or settings["fold_aggregation"] != "equal_weight":
        raise ValueError("Phase 10 selection must use equal-weight fold final_score")
    alphas = [float(value) for value in settings["alphas"]]
    exits = [float(value) for value in settings["hysteresis_exit_fractions"]]
    if len(set(alphas)) != len(alphas) or len(set(exits)) != len(exits):
        raise ValueError("Phase 10 grid values must be unique")

    baseline = json.loads(Path(config["baseline"]["audit_output"]).read_text(encoding="utf-8"))
    cv_audit = json.loads(Path(config["cv"]["audit_output"]).read_text(encoding="utf-8"))
    folds = [fold["name"] for fold in cv_audit["folds"]]
    source_dir = Path(settings["source_oof_dir"])
    source_paths = {
        fold: source_dir / fold / f"{settings['source_model']}_oof.parquet" for fold in folds
    }
    denominator = int(config["evaluator"]["top_fraction_denominator"])
    minimum_pool = int(config["evaluator"]["portfolio_min_samples"])

    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        frame = _load_fold(source_paths[fold])
        transform_input = frame[[*KEYS, "flag_limit_up", "pred"]]
        for alpha in alphas:
            smoothed, _ = causal_rank_ema(transform_input, alpha)
            for exit_fraction in exits:
                predictions, _ = hysteresis_rank_scores(
                    transform_input, smoothed, exit_fraction,
                    top_fraction_denominator=denominator, minimum_pool_size=minimum_pool,
                )
                fold_rows.append({
                    "record_type": "fold", "fold": fold,
                    "candidate": _candidate(alpha, exit_fraction),
                    "alpha": alpha, "exit_fraction": exit_fraction,
                    **_score(frame, predictions),
                })
        del frame

    fold_grid = pd.DataFrame(fold_rows)
    summaries = _summarize(fold_grid)
    selected = _select(summaries)
    selected_candidate = str(selected["candidate"])
    selected_alpha = float(selected["alpha"])
    selected_exit = float(selected["exit_fraction"])
    grid = pd.concat([fold_grid, summaries], ignore_index=True, sort=False)
    ordered_columns = [
        "record_type", "fold", "candidate", "alpha", "exit_fraction", *METRIC_KEYS,
        "final_score_std", "final_score_worst",
    ]
    grid = grid.reindex(columns=ordered_columns)

    output_dir = Path(settings["output_dir"])
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Phase 10 output exists; pass overwrite explicitly: {output_dir}")
    temporary_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    saved_oof: dict[str, dict[str, Any]] = {}
    try:
        for fold in folds:
            frame = _load_fold(source_paths[fold])
            transform_input = frame[[*KEYS, "flag_limit_up", "pred"]]
            smoothed, _ = causal_rank_ema(transform_input, selected_alpha)
            predictions, _ = hysteresis_rank_scores(
                transform_input, smoothed, selected_exit,
                top_fraction_denominator=denominator, minimum_pool_size=minimum_pool,
            )
            persisted_metrics = _score(frame, predictions)
            expected = fold_grid[
                (fold_grid["fold"] == fold) & (fold_grid["candidate"] == selected_candidate)
            ].iloc[0]
            if any(float(persisted_metrics[key]) != float(expected[key]) for key in METRIC_KEYS):
                raise ValueError(f"Persisted Phase 10 score mismatch: {fold}")
            output = frame[[*KEYS, "y_ret_1d", "flag_limit_up"]].copy()
            output["pred"] = predictions
            fold_dir = temporary_dir / fold
            fold_dir.mkdir(parents=True)
            oof_path = fold_dir / "lightgbm_turnover_oof.parquet"
            output.to_parquet(oof_path, engine="pyarrow", compression="zstd", index=False)
            saved_oof[fold] = {
                "path": str(output_dir / fold / oof_path.name),
                "rows": int(len(output)), "bytes": oof_path.stat().st_size,
                "sha256": sha256_file(oof_path), "pred_dtype": str(output["pred"].dtype),
                "pred_finite": bool(np.isfinite(output["pred"].to_numpy(dtype="float64")).all()),
                "keys_unique": not bool(output.duplicated(KEYS).any()),
            }

        rank_only_name = _candidate(1.0, 1.0 / denominator)
        rank_only = summaries[summaries["candidate"] == rank_only_name]
        if len(rank_only) != 1:
            raise ValueError("Grid must contain alpha=1 and no-buffer rank-only control")
        source_summary = baseline["models"][settings["source_model"]]["summary"]
        rank_only_delta = {
            key: float(rank_only.iloc[0][key]) - float(source_summary[key]["mean"])
            for key in METRIC_KEYS
        }
        selected_fold_metrics = fold_grid[fold_grid["candidate"] == selected_candidate].to_dict("records")
        manifest = {
            "phase": 10, "source_model": settings["source_model"],
            "selection_metric": settings["selection_metric"],
            "cv_aggregation": "equal_weight_across_folds",
            "candidate_count": len(alphas) * len(exits),
            "grid": {"alphas": alphas, "exit_fractions": exits},
            "selected": {
                "candidate": selected_candidate, "alpha": selected_alpha,
                "exit_fraction": selected_exit,
                **{key: float(selected[key]) for key in METRIC_KEYS},
                "final_score_std": float(selected["final_score_std"]),
                "final_score_worst": float(selected["final_score_worst"]),
                "folds": selected_fold_metrics,
            },
            "rank_only_metric_delta_vs_source": rank_only_delta,
            "initialization": settings["initialization"],
            "causal_contract": {
                "daily_rank_uses_same_date_only": True,
                "ema_uses_current_and_previous_stock_rows_only": True,
                "fold_state_reset": True,
                "hysteresis_uses_current_scores_and_previous_selected_set_only": True,
                "labels_used_by_transform": False,
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
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase10_turnover_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root)
    source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [
        *source_paths.values(), Path(config["baseline"]["audit_output"]),
        Path(config["cv"]["audit_output"]),
        Path(config["baseline"]["model_view_output"]) / "manifest.json",
        Path(config["paths"]["official_evaluator"]), config_snapshot,
        output_dir / "manifest.json", grid_parquet, grid_csv,
        *[Path(item["path"]) for item in saved_oof.values()],
    ]
    artifact_manifest = {
        "phase": 10, "test_data_used": False, "source_tree_sha256": source["tree_sha256"],
        "feature_version": baseline["feature_view"]["version"],
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase10_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)

    git = git_state(root)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    feature_manifest_sha = sha256_file(Path(config["baseline"]["model_view_output"]) / "manifest.json")
    cv_by_name = {fold["name"]: fold for fold in cv_audit["folds"]}
    params = json.dumps({
        "source_model": settings["source_model"], "alphas": alphas,
        "hysteresis_exit_fractions": exits, "selected_alpha": selected_alpha,
        "selected_exit_fraction": selected_exit, "initialization": settings["initialization"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    registry_rows = []
    for item in selected_fold_metrics:
        cv_fold = cv_by_name[item["fold"]]
        registry_rows.append({
            "experiment_id": settings["experiment_id"], "timestamp": timestamp,
            "record_type": "fold", "fold": item["fold"],
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
            "config_path": str(config_snapshot), "config_sha256": config_sha,
            "feature_version": baseline["feature_view"]["version"],
            "feature_manifest_sha256": feature_manifest_sha,
            "model": "lightgbm_causal_turnover_postprocess", "model_params": params,
            "train_period": f"{cv_fold['actual']['train_start']}-{cv_fold['actual']['train_end']}",
            "val_period": f"{cv_fold['actual']['val_start']}-{cv_fold['actual']['val_end']}",
            "seed": config["project"]["seed"],
            **{key: item[key] for key in METRIC_KEYS},
            "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
            "artifact_paths": json.dumps([saved_oof[item["fold"]]["path"]], separators=(",", ":")),
            "notes": "causal_rank_ema_hysteresis_float32_oof_no_test_data",
        })
    registry_rows.append({
        "experiment_id": settings["experiment_id"], "timestamp": timestamp,
        "record_type": "summary", "fold": "all_equal_weight",
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
        "config_path": str(config_snapshot), "config_sha256": config_sha,
        "feature_version": baseline["feature_view"]["version"],
        "feature_manifest_sha256": feature_manifest_sha,
        "model": "lightgbm_causal_turnover_postprocess", "model_params": params,
        "train_period": "three_expanding_folds", "val_period": "2022|2023|2024",
        "seed": config["project"]["seed"],
        **{key: float(selected[key]) for key in METRIC_KEYS},
        "final_score_std": float(selected["final_score_std"]),
        "final_score_worst": float(selected["final_score_worst"]),
        "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(output_dir), str(grid_parquet), str(grid_csv)], separators=(",", ":")),
        "notes": "equal_weight_fold_selection_by_official_final_score_no_test_data",
    })
    registry_result = append_registry_rows(Path(config["experiments"]["registry"]), registry_rows)
    audit = {
        **manifest,
        "source_tree_sha256": source["tree_sha256"], "source_file_count": source["file_count"],
        "source_manifest": str(source_path), "config_sha256": config_sha,
        "config_snapshot": str(config_snapshot), "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha,
        "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "grid_outputs": {"parquet": str(grid_parquet), "csv": str(grid_csv), "rows": int(len(grid))},
        "git": git, "registry": {"path": config["experiments"]["registry"], **registry_result},
    }
    _write_json_atomic(Path(settings["audit_output"]), audit)
    return audit
