"""Memory-bounded LightGBM LambdaRank baseline for Phase 12."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import lightgbm as lgb
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
from src.models.baselines import (
    KEYS,
    META,
    _aggregate,
    _evaluate_predictions,
    _fold_dates,
    _iter_view_batches,
    _load_validation,
    _save_oof,
)
from src.models.dataset import load_model_view_manifest
from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores
from src.validation.evaluator import METRIC_KEYS, validate_official_contract


def relevance_labels(frame: pd.DataFrame, *, label: str, bins: int) -> np.ndarray:
    """Map non-null returns to reproducible within-date ordinal relevance bins."""
    if bins < 2 or bins > 31:
        raise ValueError("relevance bins must be between 2 and 31")
    if label not in frame or "trade_date" not in frame:
        raise ValueError("relevance input requires trade_date and label")
    if frame[label].isna().any():
        raise ValueError("relevance input must exclude missing labels")
    ranks = frame.groupby("trade_date", sort=False)[label].rank(method="average", pct=True)
    result = np.ceil(ranks.to_numpy(dtype="float64") * bins).astype("int16") - 1
    result = np.clip(result, 0, bins - 1).astype("uint8")
    return result


def complete_date_chunks(batches: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Coalesce ordered batches so a trade date is never split across yields."""
    pending: pd.DataFrame | None = None
    last_emitted: int | None = None
    for batch in batches:
        if batch.empty:
            continue
        current = batch if pending is None else pd.concat([pending, batch], ignore_index=True)
        dates = current["trade_date"].to_numpy()
        if len(dates) > 1 and np.any(dates[1:] < dates[:-1]):
            raise ValueError("Ranking input batches must be globally date ordered")
        last_date = dates[-1]
        complete = current[current["trade_date"] != last_date]
        pending = current[current["trade_date"] == last_date].copy()
        if not complete.empty:
            first_date = int(complete["trade_date"].iloc[0])
            if last_emitted is not None and first_date <= last_emitted:
                raise ValueError("A ranking date appeared after it was emitted")
            last_emitted = int(complete["trade_date"].iloc[-1])
            yield complete.reset_index(drop=True)
    if pending is not None and not pending.empty:
        first_date = int(pending["trade_date"].iloc[0])
        if last_emitted is not None and first_date <= last_emitted:
            raise ValueError("Final ranking date is not strictly after emitted dates")
        yield pending.reset_index(drop=True)


def _write_training_memmaps(
    view_dir: Path,
    dates: tuple[int, ...],
    features: list[str],
    label: str,
    bins: int,
    scratch: Path,
    expected_rows: int,
    batch_size: int,
) -> tuple[np.memmap, np.memmap, np.ndarray, dict[str, Any]]:
    scratch.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(
        scratch / "x.npy", mode="w+", dtype="float32", shape=(expected_rows, len(features))
    )
    y = np.lib.format.open_memmap(
        scratch / "y.npy", mode="w+", dtype="float32", shape=(expected_rows,)
    )
    groups: list[int] = []
    group_dates: list[int] = []
    histogram = np.zeros(bins, dtype="int64")
    offset = 0
    batches = _iter_view_batches(
        view_dir, dates, ["trade_date", label, *features], batch_size=batch_size
    )
    for complete in complete_date_chunks(batches):
        labeled = complete.loc[complete[label].notna()].reset_index(drop=True)
        if labeled.empty:
            continue
        labels = relevance_labels(labeled, label=label, bins=bins)
        sizes = labeled.groupby("trade_date", sort=False).size()
        count = len(labeled)
        x[offset:offset + count] = labeled[features].to_numpy(dtype="float32", na_value=np.nan)
        y[offset:offset + count] = labels.astype("float32")
        groups.extend(int(value) for value in sizes.to_numpy())
        group_dates.extend(int(value) for value in sizes.index)
        histogram += np.bincount(labels, minlength=bins)
        offset += count
    x.flush()
    y.flush()
    if offset != expected_rows:
        raise ValueError(f"Rank training row mismatch: expected {expected_rows}, got {offset}")
    if tuple(group_dates) != tuple(dates):
        raise ValueError("Rank groups do not exactly match the ordered training dates")
    group_array = np.asarray(groups, dtype="int32")
    if int(group_array.sum()) != expected_rows or np.any(group_array <= 0):
        raise ValueError("Rank group sizes are invalid")
    audit = {
        "rows": expected_rows, "groups": len(group_array),
        "group_size_min": int(group_array.min()), "group_size_max": int(group_array.max()),
        "group_size_sum": int(group_array.sum()),
        "first_group_date": group_dates[0], "last_group_date": group_dates[-1],
        "relevance_histogram": {str(level): int(value) for level, value in enumerate(histogram)},
    }
    return x, y, group_array, audit


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {key: float(left[key]) - float(right[key]) for key in METRIC_KEYS}


def run_lambdarank(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Train three fold-isolated rankers and score raw plus frozen turnover variants."""
    run_started = time.perf_counter()
    validate_official_contract(config)
    settings = config["ranking"]
    output_dir = Path(settings["output_dir"])
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Phase 12 output exists; pass overwrite explicitly: {output_dir}")
    manifest = load_model_view_manifest(config)
    view_dir = Path(settings["model_view_output"])
    features = [
        feature for group in settings["feature_groups"] for feature in manifest["feature_groups"][group]
    ]
    if len(features) != len(set(features)) or not set(features).issubset(manifest["feature_names"]):
        raise ValueError("Ranking feature allowlist is invalid")
    bins = int(settings["relevance_bins"])
    params = dict(settings["params"])
    if params.get("objective") != "lambdarank" or list(params.get("label_gain", [])) != list(range(bins)):
        raise ValueError("LambdaRank objective and linear label_gain must match relevance bins")
    if int(settings["num_boost_round"]) != int(config["baseline"]["lightgbm"]["num_boost_round"]):
        raise ValueError("Ranking/regression comparison requires the same boosting rounds")

    cv_audit = json.loads(Path(config["cv"]["audit_output"]).read_text(encoding="utf-8"))
    expected_rows = {fold["name"]: int(fold["labeled_row_counts"]["train"]) for fold in cv_audit["folds"]}
    folds = _fold_dates(config)
    turnover = json.loads(Path(settings["turnover_audit"]).read_text(encoding="utf-8"))
    alpha = float(turnover["selected"]["alpha"])
    exit_fraction = float(turnover["selected"]["exit_fraction"])
    denominator = int(config["evaluator"]["top_fraction_denominator"])
    minimum_pool = int(config["evaluator"]["portfolio_min_samples"])
    temporary_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    scratch_root = Path(config["paths"]["cache"]) / "phase12_rank_training_scratch"
    shutil.rmtree(scratch_root, ignore_errors=True)
    model_reports: dict[str, Any] = {"raw": {"folds": []}, "turnover": {"folds": []}}
    training_groups: dict[str, Any] = {}
    try:
        for fold in folds:
            fold_started = time.perf_counter()
            name = fold["name"]
            fold_dir = temporary_dir / name
            fold_dir.mkdir(parents=True)
            scratch = scratch_root / name
            x_train, y_train, groups, group_audit = _write_training_memmaps(
                view_dir, fold["train_dates"], features, manifest["label"], bins, scratch,
                expected_rows[name], int(settings["batch_size"]),
            )
            training_groups[name] = group_audit
            train_set = lgb.Dataset(
                x_train, label=y_train, group=groups,
                feature_name=features, free_raw_data=True,
            )
            booster = lgb.train(
                params, train_set, num_boost_round=int(settings["num_boost_round"])
            )
            validation = _load_validation(
                view_dir, fold["val_dates"], features, int(settings["batch_size"])
            )
            val_x = validation[features].to_numpy(dtype="float32", na_value=np.nan)
            raw_predictions = booster.predict(
                val_x, num_iteration=booster.best_iteration or booster.current_iteration()
            ).astype("float32")
            raw_metrics, raw_diagnostics = _evaluate_predictions(validation, raw_predictions)
            raw_oof = _save_oof(fold_dir / "ranker_raw_oof.parquet", validation, raw_predictions)

            transform = validation[[*KEYS, "flag_limit_up"]].copy()
            transform["pred"] = raw_predictions
            smoothed, _ = causal_rank_ema(transform, alpha)
            stable_predictions, _ = hysteresis_rank_scores(
                transform, smoothed, exit_fraction,
                top_fraction_denominator=denominator, minimum_pool_size=minimum_pool,
            )
            stable_metrics, stable_diagnostics = _evaluate_predictions(validation, stable_predictions)
            stable_oof = _save_oof(
                fold_dir / "ranker_turnover_oof.parquet", validation, stable_predictions
            )
            booster.save_model(str(fold_dir / "lambdarank_model.txt"))
            common = {
                "name": name, "train_rows": expected_rows[name],
                "validation_rows": int(len(validation)), "iterations": booster.current_iteration(),
                "seconds": time.perf_counter() - fold_started,
            }
            model_reports["raw"]["folds"].append({
                **common, "metrics": raw_metrics, "diagnostics": raw_diagnostics, "oof": raw_oof,
            })
            model_reports["turnover"]["folds"].append({
                **common, "metrics": stable_metrics, "diagnostics": stable_diagnostics, "oof": stable_oof,
            })
            del validation, val_x, raw_predictions, stable_predictions
            del booster, train_set, x_train, y_train, groups
            gc.collect()
            shutil.rmtree(scratch)

        for variant in model_reports.values():
            variant["summary"] = _aggregate(variant["folds"])
        for variant_name, variant in model_reports.items():
            filename = "ranker_raw_oof.parquet" if variant_name == "raw" else "ranker_turnover_oof.parquet"
            for fold in variant["folds"]:
                fold["oof"]["path"] = str(output_dir / fold["name"] / filename)

        regression = json.loads(Path(config["baseline"]["audit_output"]).read_text(encoding="utf-8"))
        ensemble = json.loads(Path(settings["ensemble_audit"]).read_text(encoding="utf-8"))
        comparison = {
            "raw_vs_phase8_regression": _metric_delta(
                {key: model_reports["raw"]["summary"][key]["mean"] for key in METRIC_KEYS},
                {key: regression["models"]["lightgbm"]["summary"][key]["mean"] for key in METRIC_KEYS},
            ),
            "turnover_vs_phase10_regression": _metric_delta(
                {key: model_reports["turnover"]["summary"][key]["mean"] for key in METRIC_KEYS},
                turnover["selected"],
            ),
            "turnover_vs_phase11_ensemble": _metric_delta(
                {key: model_reports["turnover"]["summary"][key]["mean"] for key in METRIC_KEYS},
                ensemble["selected"],
            ),
        }
        report = {
            "phase": 12, "model": "lightgbm_lambdarank",
            "feature_version": manifest["version"], "feature_count": len(features),
            "feature_groups": list(settings["feature_groups"]),
            "relevance": {
                "bins": bins, "rule": settings["relevance_rule"],
                "tie_method": "average", "missing_labels": "excluded_before_ranking",
                "scope": "within_training_trade_date_only", "label_gain": list(params["label_gain"]),
            },
            "training": {
                "objective": params["objective"], "num_boost_round": int(settings["num_boost_round"]),
                "lambdarank_truncation_level": int(params["lambdarank_truncation_level"]),
                "groups": training_groups, "run_seconds": time.perf_counter() - run_started,
            },
            "locked_turnover_postprocess": {
                "phase10_audit": str(Path(settings["turnover_audit"])),
                "alpha": alpha, "exit_fraction": exit_fraction,
            },
            "variants": model_reports, "comparison": comparison,
            "selection_metric": "equal_weight_fold_final_score",
            "selected_variant": max(
                model_reports,
                key=lambda name: model_reports[name]["summary"]["final_score"]["mean"],
            ),
            "contracts": {
                "training_groups_contiguous_by_date": True,
                "group_sizes_sum_to_labeled_training_rows": True,
                "features_match_phase8_regression": features == [
                    feature for group in config["baseline"]["lightgbm"]["feature_groups"]
                    for feature in manifest["feature_groups"][group]
                ],
                "boosting_rounds_match_phase8_regression": True,
                "fold_state_reset": True, "test_data_used": False,
            },
            "outputs": str(output_dir),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
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
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    root = Path(config["paths"]["root"])
    config_path = root / "config.yaml"
    config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase12_lambdarank_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root)
    source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [
        Path(settings["model_view_output"]) / "manifest.json",
        *sorted(Path(settings["model_view_output"]).glob("year=*/data.parquet")),
        Path(config["cv"]["audit_output"]), Path(settings["turnover_audit"]),
        Path(settings["ensemble_audit"]), Path(config["paths"]["official_evaluator"]),
        config_snapshot, output_dir / "manifest.json",
        *sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
    ]
    artifact_manifest = {
        "phase": 12, "test_data_used": False, "source_tree_sha256": source["tree_sha256"],
        "feature_version": manifest["version"],
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase12_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)

    git = git_state(root)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    feature_manifest_sha = sha256_file(Path(settings["model_view_output"]) / "manifest.json")
    cv_by_name = {fold["name"]: fold for fold in cv_audit["folds"]}
    params_json = json.dumps({
        "features": list(settings["feature_groups"]), "relevance_bins": bins,
        "relevance_rule": settings["relevance_rule"], "num_boost_round": settings["num_boost_round"],
        "params": params, "turnover_alpha": alpha, "turnover_exit_fraction": exit_fraction,
    }, sort_keys=True, separators=(",", ":"))
    registry_rows = []
    for variant_name, variant in model_reports.items():
        experiment_id = settings["experiment_ids"][variant_name]
        for item in variant["folds"]:
            cv_fold = cv_by_name[item["name"]]
            registry_rows.append({
                "experiment_id": experiment_id, "timestamp": timestamp,
                "record_type": "fold", "fold": item["name"],
                "git_commit": git["commit"], "git_dirty": git["dirty"],
                "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
                "config_path": str(config_snapshot), "config_sha256": config_sha,
                "feature_version": manifest["version"], "feature_manifest_sha256": feature_manifest_sha,
                "model": f"lightgbm_lambdarank_{variant_name}", "model_params": params_json,
                "train_period": f"{cv_fold['actual']['train_start']}-{cv_fold['actual']['train_end']}",
                "val_period": f"{cv_fold['actual']['val_start']}-{cv_fold['actual']['val_end']}",
                "seed": config["project"]["seed"], **item["metrics"],
                "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
                "artifact_paths": json.dumps([
                    item["oof"]["path"], str(output_dir / item["name"] / "lambdarank_model.txt")
                ], separators=(",", ":")),
                "notes": "date_grouped_integer_relevance_real_train_labels_no_test_data",
            })
        summary = variant["summary"]
        registry_rows.append({
            "experiment_id": experiment_id, "timestamp": timestamp,
            "record_type": "summary", "fold": "all_equal_weight",
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
            "config_path": str(config_snapshot), "config_sha256": config_sha,
            "feature_version": manifest["version"], "feature_manifest_sha256": feature_manifest_sha,
            "model": f"lightgbm_lambdarank_{variant_name}", "model_params": params_json,
            "train_period": "three_expanding_folds", "val_period": "2022|2023|2024",
            "seed": config["project"]["seed"],
            **{key: summary[key]["mean"] for key in METRIC_KEYS},
            "final_score_std": summary["final_score"]["std"],
            "final_score_worst": summary["final_score"]["min"],
            "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
            "artifact_paths": json.dumps([str(output_dir), str(Path(settings["audit_output"]))], separators=(",", ":")),
            "notes": "equal_weight_fold_summary_no_test_data",
        })
    registry_result = append_registry_rows(Path(config["experiments"]["registry"]), registry_rows)
    audit = {
        **report, "source_tree_sha256": source["tree_sha256"], "source_file_count": source["file_count"],
        "source_manifest": str(source_path), "config_sha256": config_sha,
        "config_snapshot": str(config_snapshot), "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha,
        "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "git": git, "registry": {"path": config["experiments"]["registry"], **registry_result},
    }
    _atomic_json(Path(settings["audit_output"]), audit)
    return audit
