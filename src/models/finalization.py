"""Phase 16 OOF ensemble selection and production-model fitting."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)
from src.models.baselines import (
    KEYS, META, _fit_ridge, _iter_view_batches, _write_lgb_training_memmaps,
)
from src.models.dataset import load_model_view_manifest
from src.models.ensemble import blend_daily_rank_predictions
from src.models.ranking import _write_training_memmaps
from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores
from src.validation.evaluator import METRIC_KEYS, evaluate_merged_frame, validate_official_contract


def simplex_weights(step: float) -> list[tuple[float, float, float]]:
    """Return an exact three-component simplex grid in deterministic order."""
    units = int(round(1.0 / float(step)))
    if units < 1 or not np.isclose(units * step, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("simplex step must exactly divide one")
    return [(a / units, b / units, (units - a - b) / units)
            for a in range(units + 1) for b in range(units + 1 - a)]


def select_candidate(summary: pd.DataFrame) -> pd.Series:
    """Use the locked mean/worst/turnover deterministic selection order."""
    return summary.sort_values(
        ["final_score", "final_score_worst", "mean_turnover", "candidate"],
        ascending=[False, False, True, True], kind="stable",
    ).iloc[0]


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*META, "pred"]).sort_values(
        ["trade_date", "ts_code"], kind="stable"
    ).reset_index(drop=True)
    if frame.duplicated(KEYS).any() or not np.isfinite(frame["pred"].to_numpy()).all():
        raise ValueError(f"Invalid OOF: {path}")
    return frame


def _components(config: dict[str, Any], fold: str) -> dict[str, pd.DataFrame]:
    settings = config["finalization"]
    base = _load(Path(settings["ranker_oof_dir"]) / fold / "ranker_raw_oof.parquet")
    atr = _load(Path(settings["atr_oof_dir"]) / fold / "ranker_raw_oof.parquet")
    lgb_oof = _load(Path(settings["regression_oof_dir"]) / fold / "lightgbm_oof.parquet")
    ridge_oof = _load(Path(settings["regression_oof_dir"]) / fold / "ridge_oof.parquet")
    for name, frame in {"atr_2y": atr, "lightgbm": lgb_oof, "ridge": ridge_oof}.items():
        if not frame[META].equals(base[META]):
            raise ValueError(f"OOF metadata mismatch for {fold}/{name}")
    sleeve = blend_daily_rank_predictions(
        {"lightgbm": lgb_oof, "ridge": ridge_oof},
        {"lightgbm": float(settings["regression_sleeve_weights"]["lightgbm"]),
         "ridge": float(settings["regression_sleeve_weights"]["ridge"])},
    )
    sleeve = base[META].copy().assign(pred=sleeve["pred"].to_numpy())
    components = {"ranker_base": base, "atr_2y": atr, "regression_sleeve": sleeve}
    # These ranks do not depend on candidate weights; cache them once per fold.
    for frame in components.values():
        frame["pred"] = frame.groupby("trade_date", sort=False)["pred"].rank(
            method="average", pct=True
        ).to_numpy(dtype="float32")
    return components


def _postprocess(config: dict[str, Any], frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> np.ndarray:
    turnover = json.loads(Path(config["ranking"]["turnover_audit"]).read_text(encoding="utf-8"))
    reference = frames["ranker_base"]
    values = sum(float(weights[name]) * frame["pred"].to_numpy(dtype="float64")
                 for name, frame in frames.items())
    blended = reference[KEYS].copy()
    blended["pred"] = values
    transform = blended.assign(flag_limit_up=frames["ranker_base"]["flag_limit_up"].to_numpy())
    smoothed, _ = causal_rank_ema(transform, float(turnover["selected"]["alpha"]))
    result, _ = hysteresis_rank_scores(
        transform, smoothed, float(turnover["selected"]["exit_fraction"]),
        top_fraction_denominator=int(config["evaluator"]["top_fraction_denominator"]),
        minimum_pool_size=int(config["evaluator"]["portfolio_min_samples"]),
    )
    return np.asarray(result, dtype="float32")


def _score(reference: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    frame = reference[META].copy()
    frame["pred"] = np.asarray(pred, dtype="float32")
    return {key: float(value) for key, value in evaluate_merged_frame(frame).items()}


def _candidate(weights: tuple[float, float, float]) -> str:
    return "ranker_{:.2f}__atr2y_{:.2f}__reg_{:.2f}".format(*weights)


def _summaries(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weight_cols = ["ranker_weight", "atr_2y_weight", "regression_sleeve_weight"]
    for keys, part in folds.groupby(["candidate", *weight_cols], sort=False):
        row: dict[str, Any] = {"record_type": "summary", "fold": "all_equal_weight",
                               "candidate": keys[0], **dict(zip(weight_cols, keys[1:]))}
        for metric in METRIC_KEYS:
            row[metric] = float(part[metric].mean())
        row["final_score_std"] = float(part["final_score"].std(ddof=1))
        row["final_score_worst"] = float(part["final_score"].min())
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {key: float(payload[key]["mean"]) for key in METRIC_KEYS}


def _endpoint_controls(config: dict[str, Any], summary: pd.DataFrame) -> dict[str, Any]:
    """Require all three simplex endpoints to replay their historical audits."""
    endpoints = {
        "ranker_base": (1.0, 0.0, 0.0),
        "atr_2y": (0.0, 1.0, 0.0),
        "regression_sleeve": (0.0, 0.0, 1.0),
    }
    expected = {
        "ranker_base": _mean_metrics(json.loads(Path(config["downstream"]["incumbent"]["audit"]).read_text(encoding="utf-8"))["variants"]["turnover"]["summary"]),
        "atr_2y": _mean_metrics(json.loads(Path(config["stability"]["audit_output"]).read_text(encoding="utf-8"))["models"]["trailing_2y"]["turnover"]["summary"]),
        "regression_sleeve": {key: float(value) for key, value in json.loads(Path(config["ensemble"]["audit_output"]).read_text(encoding="utf-8"))["selected"].items() if key in METRIC_KEYS},
    }
    controls: dict[str, Any] = {}
    for name, weights in endpoints.items():
        actual = summary[summary["candidate"] == _candidate(weights)].iloc[0]
        delta = {key: float(actual[key]) - expected[name][key] for key in METRIC_KEYS}
        if max(abs(value) for value in delta.values()) > 1e-12:
            raise ValueError(f"Phase 16 endpoint failed historical replay: {name} {delta}")
        controls[name] = {"candidate": _candidate(weights), "expected": expected[name], "delta": delta}
    return controls


def _all_labeled_dates(view_dir: Path) -> tuple[int, ...]:
    dates: list[int] = []
    for path in sorted(view_dir.glob("year=*/data.parquet")):
        table = pq.read_table(path, columns=["trade_date", "y_ret_1d"]).to_pandas()
        dates.extend(int(v) for v in table.loc[table["y_ret_1d"].notna(), "trade_date"].unique())
    result = tuple(sorted(set(dates)))
    if not result:
        raise ValueError("No real labeled training dates")
    return result


def _count_labeled(view_dir: Path, dates: tuple[int, ...], batch_size: int) -> int:
    return sum(int(x["y_ret_1d"].notna().sum()) for x in _iter_view_batches(
        view_dir, dates, ["trade_date", "y_ret_1d"], batch_size=batch_size
    ))


def _fit_final_models(config: dict[str, Any], selected: pd.Series, output: Path) -> dict[str, Any]:
    """Fit only components with positive selected weights, using real train labels."""
    settings, ranking, baseline = config["finalization"], config["ranking"], config["baseline"]
    base_manifest = load_model_view_manifest(config)
    base_view = Path(baseline["model_view_output"])
    atr_view = Path(config["stability"]["augmented_view_output"])
    batch_size = int(settings["batch_size"])
    all_dates = _all_labeled_dates(base_view)
    report: dict[str, Any] = {"all_labeled_dates": len(all_dates), "start": all_dates[0], "end": all_dates[-1], "models": {}}
    scratch_root = Path(config["paths"]["cache"]) / "phase16_final_training_scratch"
    shutil.rmtree(scratch_root, ignore_errors=True)
    try:
        rank_specs = [
            ("ranker_base", float(selected["ranker_weight"]), base_view,
             [f for g in ranking["feature_groups"] for f in base_manifest["feature_groups"][g]], all_dates),
            ("atr_2y", float(selected["atr_2y_weight"]), atr_view,
             json.loads((atr_view / "manifest.json").read_text(encoding="utf-8"))["feature_names"],
             tuple(d for d in all_dates if d >= 20230101)),
        ]
        for name, weight, view, features, dates in rank_specs:
            if weight <= 0:
                continue
            started = time.perf_counter()
            rows = _count_labeled(view, dates, batch_size)
            x, y, groups, group_audit = _write_training_memmaps(
                view, dates, features, "y_ret_1d", int(ranking["relevance_bins"]),
                scratch_root / name, rows, batch_size,
            )
            train_set = lgb.Dataset(x, label=y, group=groups, feature_name=features, free_raw_data=True)
            booster = lgb.train(dict(ranking["params"]), train_set, num_boost_round=int(ranking["num_boost_round"]))
            path = output / f"{name}_model.txt"
            booster.save_model(str(path))
            report["models"][name] = {"weight": weight, "path": str(path), "rows": rows,
                "feature_count": len(features), "features": features, "training_groups": group_audit,
                "iterations": booster.current_iteration(), "seconds": time.perf_counter() - started}
            del x, y, groups, train_set, booster
            gc.collect(); shutil.rmtree(scratch_root / name, ignore_errors=True)

        reg_weight = float(selected["regression_sleeve_weight"])
        if reg_weight > 0:
            lgb_features = [f for g in baseline["lightgbm"]["feature_groups"] for f in base_manifest["feature_groups"][g]]
            rows = _count_labeled(base_view, all_dates, batch_size)
            started = time.perf_counter()
            x, y = _write_lgb_training_memmaps(base_view, all_dates, lgb_features, "y_ret_1d",
                                                scratch_root / "lightgbm", rows, batch_size)
            train_set = lgb.Dataset(x, label=y, feature_name=lgb_features, free_raw_data=True)
            booster = lgb.train(dict(baseline["lightgbm"]["params"]), train_set,
                                num_boost_round=int(baseline["lightgbm"]["num_boost_round"]))
            lgb_path = output / "regression_lightgbm_model.txt"; booster.save_model(str(lgb_path))
            report["models"]["regression_lightgbm"] = {"outer_weight": reg_weight,
                "sleeve_weight": float(settings["regression_sleeve_weights"]["lightgbm"]),
                "path": str(lgb_path), "rows": rows, "feature_count": len(lgb_features),
                "features": lgb_features, "iterations": booster.current_iteration(),
                "seconds": time.perf_counter() - started}
            del x, y, train_set, booster; gc.collect(); shutil.rmtree(scratch_root / "lightgbm", ignore_errors=True)
            ridge_features = list(base_manifest["feature_groups"][baseline["ridge"]["feature_group"]])
            started = time.perf_counter()
            coef, intercept, ridge_rows = _fit_ridge(base_view, all_dates, ridge_features, "y_ret_1d",
                                                      float(baseline["ridge"]["alpha"]), batch_size)
            ridge_path = output / "regression_ridge_model.npz"
            np.savez(ridge_path, coefficients=coef, intercept=intercept,
                     features=np.asarray(ridge_features), alpha=float(baseline["ridge"]["alpha"]))
            report["models"]["regression_ridge"] = {"outer_weight": reg_weight,
                "sleeve_weight": float(settings["regression_sleeve_weights"]["ridge"]),
                "path": str(ridge_path), "rows": ridge_rows, "feature_count": len(ridge_features),
                "features": ridge_features, "seconds": time.perf_counter() - started}
        return report
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def run_finalization(config: dict[str, Any], *, overwrite: bool = False, select_only: bool = False) -> dict[str, Any]:
    """Select a locked OOF blend and fit its full-history production components."""
    started = time.perf_counter(); validate_official_contract(config)
    settings = config["finalization"]; root = Path(config["paths"]["root"])
    output = Path(settings["output_dir"]); audit_path = Path(settings["audit_output"])
    if (output.exists() or audit_path.exists()) and not overwrite and not select_only:
        raise FileExistsError("Phase 16 outputs exist; pass --overwrite")
    raw_before = {k: sha256_file(Path(config["paths"][k])) for k in ("train", "test")}
    folds = [str(x["name"]) for x in config["cv"]["folds"]]
    grid_weights = simplex_weights(float(settings["simplex_step"]))
    fold_rows: list[dict[str, Any]] = []
    saved_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for fold in folds:
        frames = _components(config, fold); saved_frames[fold] = frames
        for weights in grid_weights:
            mapping = dict(zip(("ranker_base", "atr_2y", "regression_sleeve"), weights))
            fold_rows.append({"record_type": "fold", "fold": fold, "candidate": _candidate(weights),
                "ranker_weight": weights[0], "atr_2y_weight": weights[1],
                "regression_sleeve_weight": weights[2], **_score(frames["ranker_base"], _postprocess(config, frames, mapping))})
    fold_grid = pd.DataFrame(fold_rows); summary = _summaries(fold_grid)
    controls = _endpoint_controls(config, summary)
    selected = select_candidate(summary)
    grid = pd.concat([fold_grid, summary], ignore_index=True, sort=False)
    if select_only:
        return {"phase": 16, "candidate_count": len(grid_weights), "selected": selected.to_dict()}
    temporary = output.with_name(output.name + ".tmp"); shutil.rmtree(temporary, ignore_errors=True); temporary.mkdir(parents=True)
    selected_weights = {"ranker_base": float(selected["ranker_weight"]), "atr_2y": float(selected["atr_2y_weight"]),
                        "regression_sleeve": float(selected["regression_sleeve_weight"])}
    oof: dict[str, Any] = {}
    for fold, frames in saved_frames.items():
        pred = _postprocess(config, frames, selected_weights); frame = frames["ranker_base"][META].copy(); frame["pred"] = pred
        fold_dir = temporary / fold; fold_dir.mkdir(); path = fold_dir / "final_ensemble_oof.parquet"
        frame.to_parquet(path, compression="zstd", index=False)
        oof[fold] = {"path": str(output / fold / path.name), "rows": len(frame), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path), "keys_unique": not frame.duplicated(KEYS).any(), "pred_finite": True}
    final_training = _fit_final_models(config, selected, temporary)
    for model in final_training["models"].values():
        model["path"] = str(output / Path(model["path"]).name)
    report = {"phase": 16, "candidate_count": len(grid_weights), "simplex_step": float(settings["simplex_step"]),
        "components": ["ranker_base", "atr_2y", "regression_sleeve"],
        "regression_sleeve_weights": dict(settings["regression_sleeve_weights"]),
        "selection_order": list(settings["selection_order"]), "endpoint_controls": controls,
        "selected": {k: (float(selected[k]) if k != "candidate" else str(selected[k])) for k in ["candidate", "ranker_weight", "atr_2y_weight", "regression_sleeve_weight", *METRIC_KEYS, "final_score_std", "final_score_worst"]},
        "selected_folds": fold_grid[fold_grid["candidate"] == selected["candidate"]].to_dict("records"),
        "final_training": final_training, "oof": oof,
        "locked_postprocess": {"alpha": 0.5, "exit_fraction": 0.15},
        "contracts": {"same_date_component_ranking": True, "single_postprocess_after_blend": True,
            "nonnegative_weights_sum_one": True, "selection_uses_oof_only": True,
            "final_training_uses_real_train_labels_only": True, "test_data_used_for_training_or_selection": False,
            "test_labels_created_or_inferred": False, "raw_files_unchanged": False}}
    (temporary / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output.exists(): shutil.rmtree(output)
    os.replace(temporary, output)
    grid_parquet = Path(settings["grid_output_parquet"]); grid_csv = Path(settings["grid_output_csv"])
    grid_parquet.parent.mkdir(parents=True, exist_ok=True); grid.to_parquet(grid_parquet, compression="zstd", index=False); grid.to_csv(grid_csv, index=False)
    raw_after = {k: sha256_file(Path(config["paths"][k])) for k in ("train", "test")}
    report["contracts"]["raw_files_unchanged"] = raw_before == raw_after
    report["raw_sha256_before"] = raw_before; report["raw_sha256_after"] = raw_after
    config_path = root / "config.yaml"; config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase16_finalization_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root); source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode())
    artifact_files = [config_snapshot, output / "manifest.json", grid_parquet, grid_csv,
        Path(config["paths"]["official_evaluator"]), Path(config["ranking"]["turnover_audit"]),
        *sorted(p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json")]
    artifacts = {"phase": 16, "test_data_used": False, "source_tree_sha256": source["tree_sha256"],
                 "artifacts": [artifact_entry(p, root) for p in artifact_files]}
    artifact_bytes = json.dumps(artifacts, ensure_ascii=False, indent=2).encode(); artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase16_artifacts_{artifact_sha[:12]}.json"; write_immutable(artifact_path, artifact_bytes)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds"); git = git_state(root)
    params_json = json.dumps({"weights": selected_weights, "regression_sleeve_weights": settings["regression_sleeve_weights"],
                              "alpha": 0.5, "exit_fraction": 0.15}, sort_keys=True, separators=(",", ":"))
    rows = []
    for item in report["selected_folds"]:
        rows.append({"experiment_id": settings["experiment_id"], "timestamp": timestamp, "record_type": "fold", "fold": item["fold"],
            "git_commit": git["commit"], "git_dirty": git["dirty"], "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
            "config_path": str(config_snapshot), "config_sha256": config_sha, "feature_version": "phase16_locked_blend", "feature_manifest_sha256": "MULTI_COMPONENT",
            "model": "locked_rank_ensemble", "model_params": params_json, "train_period": "component_specific_past_only", "val_period": item["fold"].replace("fold_", ""),
            "seed": config["project"]["seed"], **{k: item[k] for k in METRIC_KEYS}, "artifact_manifest_path": str(artifact_path),
            "artifact_manifest_sha256": artifact_sha, "artifact_paths": json.dumps([oof[item["fold"]]["path"]]), "notes": "phase16_oof_selection_no_test_data"})
    rows.append({"experiment_id": settings["experiment_id"], "timestamp": timestamp, "record_type": "summary", "fold": "all_equal_weight",
        "git_commit": git["commit"], "git_dirty": git["dirty"], "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
        "config_path": str(config_snapshot), "config_sha256": config_sha, "feature_version": "phase16_locked_blend", "feature_manifest_sha256": "MULTI_COMPONENT",
        "model": "locked_rank_ensemble", "model_params": params_json, "train_period": "final_models_fit_through_2024", "val_period": "2022|2023|2024",
        "seed": config["project"]["seed"], **{k: float(selected[k]) for k in METRIC_KEYS}, "final_score_std": float(selected["final_score_std"]),
        "final_score_worst": float(selected["final_score_worst"]), "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(output), str(grid_parquet)]), "notes": "final_configuration_frozen_for_phase17"})
    registry = append_registry_rows(Path(config["experiments"]["registry"]), rows)
    report.update({"run_seconds": time.perf_counter() - started, "source_tree_sha256": source["tree_sha256"], "source_manifest": str(source_path),
        "config_sha256": config_sha, "config_snapshot": str(config_snapshot), "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha, "artifact_count": len(artifacts["artifacts"]),
        "artifact_total_bytes": sum(x["bytes"] for x in artifacts["artifacts"]), "registry": registry})
    audit_path.parent.mkdir(parents=True, exist_ok=True); audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
