"""Memory-bounded simple-factor, Ridge and LightGBM walk-forward baselines."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.models.dataset import load_model_view_manifest
from src.validation.evaluator import METRIC_KEYS, evaluate_merged_frame, validate_official_contract


KEYS = ["ts_code", "trade_date"]
META = [*KEYS, "y_ret_1d", "flag_limit_up"]


def _fold_dates(config: dict[str, Any]) -> list[dict[str, Any]]:
    calendar = pd.read_parquet(Path(config["cv"]["calendar_output"]), engine="pyarrow")
    result = []
    for setting in config["cv"]["folds"]:
        name = str(setting["name"])
        part = calendar[calendar["fold"].astype(str) == name]
        result.append({
            "name": name,
            "train_dates": tuple(int(v) for v in part.loc[part["role"].astype(str) == "train", "trade_date"]),
            "purge_dates": tuple(int(v) for v in part.loc[part["role"].astype(str) == "purge", "trade_date"]),
            "val_dates": tuple(int(v) for v in part.loc[part["role"].astype(str) == "validation", "trade_date"]),
        })
    return result


def _iter_view_batches(
    view_dir: Path,
    dates: tuple[int, ...],
    columns: list[str],
    *,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    date_set = set(dates)
    years = sorted({date // 10000 for date in date_set})
    for year in years:
        path = view_dir / f"year={year}" / "data.parquet"
        parquet = pq.ParquetFile(path)
        try:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                frame = batch.to_pandas()
                selected = frame["trade_date"].isin(date_set)
                if selected.any():
                    yield frame.loc[selected].reset_index(drop=True)
        finally:
            parquet.close()


def _load_validation(
    view_dir: Path,
    dates: tuple[int, ...],
    features: list[str],
    batch_size: int,
) -> pd.DataFrame:
    frames = list(_iter_view_batches(view_dir, dates, [*META, *features], batch_size=batch_size))
    if not frames:
        raise ValueError("Validation view is empty")
    return pd.concat(frames, ignore_index=True).sort_values(KEYS[::-1], kind="stable").reset_index(drop=True)


def _evaluate_predictions(validation: pd.DataFrame, predictions: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    if len(predictions) != len(validation) or not np.isfinite(predictions).all():
        raise ValueError("OOF predictions must be finite and cover every validation row")
    scored = validation[META].copy()
    # Score the exact float32 representation persisted in OOF artifacts.
    scored["pred"] = np.asarray(predictions, dtype="float32")
    metrics = {key: float(value) for key, value in evaluate_merged_frame(scored).items()}
    labeled = scored["y_ret_1d"].notna()
    error = scored.loc[labeled, "pred"] - scored.loc[labeled, "y_ret_1d"]
    diagnostics = {
        "labeled_rows": int(labeled.sum()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }
    return metrics, diagnostics


def _save_oof(path: Path, validation: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    output = validation[META].copy()
    output["pred"] = np.asarray(predictions, dtype="float32")
    output.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    key_hash = int(pd.util.hash_pandas_object(output[KEYS], index=False).sum()) & ((1 << 64) - 1)
    return {
        "path": str(path),
        "rows": int(len(output)),
        "key_hash": f"{key_hash:016x}",
        "key_unique": not bool(output.duplicated(KEYS).any()),
        "pred_finite": bool(np.isfinite(output["pred"].to_numpy(dtype="float64")).all()),
        "bytes": path.stat().st_size,
        "pred_dtype": str(output["pred"].dtype),
    }


def _fit_ridge(
    view_dir: Path,
    dates: tuple[int, ...],
    features: list[str],
    label: str,
    alpha: float,
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    size = len(features)
    xtx = np.zeros((size, size), dtype="float64")
    xty = np.zeros(size, dtype="float64")
    sum_x = np.zeros(size, dtype="float64")
    sum_y = 0.0
    rows = 0
    for frame in _iter_view_batches(view_dir, dates, ["trade_date", label, *features], batch_size=batch_size):
        frame = frame.loc[frame[label].notna()]
        if frame.empty:
            continue
        x = frame[features].to_numpy(dtype="float64", na_value=np.nan)
        x[~np.isfinite(x)] = 0.0
        y = frame[label].to_numpy(dtype="float64")
        xtx += x.T @ x
        xty += x.T @ y
        sum_x += x.sum(axis=0)
        sum_y += float(y.sum())
        rows += len(y)
    augmented = np.empty((size + 1, size + 1), dtype="float64")
    augmented[:size, :size] = xtx
    augmented[:size, size] = sum_x
    augmented[size, :size] = sum_x
    augmented[size, size] = rows
    augmented[np.arange(size), np.arange(size)] += alpha
    target = np.concatenate([xty, np.array([sum_y])])
    solution = np.linalg.solve(augmented, target)
    return solution[:size], float(solution[size]), rows


def _predict_ridge(frame: pd.DataFrame, features: list[str], coefficients: np.ndarray, intercept: float) -> np.ndarray:
    x = frame[features].to_numpy(dtype="float64", na_value=np.nan)
    x[~np.isfinite(x)] = 0.0
    return x @ coefficients + intercept


def _write_lgb_training_memmaps(
    view_dir: Path,
    dates: tuple[int, ...],
    features: list[str],
    label: str,
    scratch: Path,
    expected_rows: int,
    batch_size: int,
) -> tuple[np.memmap, np.memmap]:
    scratch.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(
        scratch / "x.npy", mode="w+", dtype="float32", shape=(expected_rows, len(features))
    )
    y = np.lib.format.open_memmap(
        scratch / "y.npy", mode="w+", dtype="float32", shape=(expected_rows,)
    )
    offset = 0
    for frame in _iter_view_batches(view_dir, dates, ["trade_date", label, *features], batch_size=batch_size):
        frame = frame.loc[frame[label].notna()]
        count = len(frame)
        if not count:
            continue
        x[offset:offset + count] = frame[features].to_numpy(dtype="float32", na_value=np.nan)
        y[offset:offset + count] = frame[label].to_numpy(dtype="float32")
        offset += count
    x.flush()
    y.flush()
    if offset != expected_rows:
        raise ValueError(f"LightGBM train row count mismatch: expected {expected_rows}, got {offset}")
    return x, y


def _aggregate(model_folds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = np.asarray([fold["metrics"][key] for fold in model_folds], dtype="float64")
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def run_baselines(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Train three real-data baselines and save full-validation OOF predictions."""
    validate_official_contract(config)
    settings = config["baseline"]
    output_dir = Path(settings["output_dir"])
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Baseline output exists; pass overwrite explicitly: {output_dir}")
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    view_dir = Path(settings["model_view_output"])
    manifest = load_model_view_manifest(config)
    folds = _fold_dates(config)
    cv_audit = json.loads(Path(config["cv"]["audit_output"]).read_text(encoding="utf-8"))
    expected_train_rows = {
        fold["name"]: int(fold["labeled_row_counts"]["train"]) for fold in cv_audit["folds"]
    }
    batch_size = int(settings["batch_size"])
    simple_feature = str(settings["simple_factor"]["feature"])
    simple_direction = float(settings["simple_factor"]["direction"])
    ridge_features = list(manifest["feature_groups"][settings["ridge"]["feature_group"]])
    lgb_features = [
        feature
        for group in settings["lightgbm"]["feature_groups"]
        for feature in manifest["feature_groups"][group]
    ]
    all_known = set(manifest["feature_names"])
    unknown = sorted({simple_feature, *ridge_features, *lgb_features} - all_known)
    if unknown:
        raise ValueError(f"Unknown baseline features: {unknown}")
    model_reports: dict[str, Any] = {name: {"folds": []} for name in ("simple_factor", "ridge", "lightgbm")}
    scratch_root = Path(config["paths"]["cache"]) / "phase8_training_scratch"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    try:
        for fold in folds:
            name = fold["name"]
            fold_dir = temporary / name
            fold_dir.mkdir(parents=True)

            started = time.perf_counter()
            simple_val = _load_validation(view_dir, fold["val_dates"], [simple_feature], batch_size)
            simple_pred = simple_direction * simple_val[simple_feature].fillna(0.5).to_numpy(dtype="float64")
            metrics, diagnostics = _evaluate_predictions(simple_val, simple_pred)
            oof = _save_oof(fold_dir / "simple_factor_oof.parquet", simple_val, simple_pred)
            model_reports["simple_factor"]["folds"].append({
                "name": name, "metrics": metrics, "diagnostics": diagnostics,
                "train_rows": 0, "validation_rows": len(simple_val),
                "seconds": time.perf_counter() - started, "oof": oof,
            })
            del simple_val, simple_pred

            started = time.perf_counter()
            coefficients, intercept, train_rows = _fit_ridge(
                view_dir, fold["train_dates"], ridge_features, manifest["label"],
                float(settings["ridge"]["alpha"]), batch_size,
            )
            ridge_val = _load_validation(view_dir, fold["val_dates"], ridge_features, batch_size)
            ridge_pred = _predict_ridge(ridge_val, ridge_features, coefficients, intercept)
            metrics, diagnostics = _evaluate_predictions(ridge_val, ridge_pred)
            oof = _save_oof(fold_dir / "ridge_oof.parquet", ridge_val, ridge_pred)
            np.savez(
                fold_dir / "ridge_model.npz", coefficients=coefficients, intercept=intercept,
                features=np.asarray(ridge_features), alpha=float(settings["ridge"]["alpha"]),
            )
            model_reports["ridge"]["folds"].append({
                "name": name, "metrics": metrics, "diagnostics": diagnostics,
                "train_rows": train_rows, "validation_rows": len(ridge_val),
                "seconds": time.perf_counter() - started, "oof": oof,
            })
            del ridge_val, ridge_pred, coefficients
            gc.collect()

            started = time.perf_counter()
            fold_scratch = scratch_root / name
            x_train, y_train = _write_lgb_training_memmaps(
                view_dir, fold["train_dates"], lgb_features, manifest["label"], fold_scratch,
                expected_train_rows[name], batch_size,
            )
            params = dict(settings["lightgbm"]["params"])
            train_set = lgb.Dataset(x_train, label=y_train, feature_name=lgb_features, free_raw_data=True)
            booster = lgb.train(
                params, train_set, num_boost_round=int(settings["lightgbm"]["num_boost_round"])
            )
            lgb_val = _load_validation(view_dir, fold["val_dates"], lgb_features, batch_size)
            val_x = lgb_val[lgb_features].to_numpy(dtype="float32", na_value=np.nan)
            lgb_pred = booster.predict(val_x, num_iteration=booster.best_iteration or booster.current_iteration())
            metrics, diagnostics = _evaluate_predictions(lgb_val, lgb_pred)
            oof = _save_oof(fold_dir / "lightgbm_oof.parquet", lgb_val, lgb_pred)
            booster.save_model(str(fold_dir / "lightgbm_model.txt"))
            model_reports["lightgbm"]["folds"].append({
                "name": name, "metrics": metrics, "diagnostics": diagnostics,
                "train_rows": expected_train_rows[name], "validation_rows": len(lgb_val),
                "seconds": time.perf_counter() - started,
                "iterations": booster.current_iteration(), "oof": oof,
            })
            del lgb_val, val_x, lgb_pred, booster, train_set, x_train, y_train
            gc.collect()
            shutil.rmtree(fold_scratch)

        for model in model_reports.values():
            model["summary"] = _aggregate(model["folds"])
        for model_name, model in model_reports.items():
            for fold in model["folds"]:
                fold["oof"]["path"] = str(
                    output_dir / fold["name"] / f"{model_name}_oof.parquet"
                )
        oof_key_consistent = all(
            len({model_reports[model]["folds"][fold_index]["oof"]["key_hash"] for model in model_reports}) == 1
            for fold_index in range(len(folds))
        )
        report = {
            "phase": 8,
            "feature_view": manifest,
            "cv_aggregation": "equal_weight_across_folds",
            "selection_metric": "final_score",
            "test_data_used": False,
            "preprocessing": {
                "simple_factor_missing": "neutral_percentile_0.5",
                "ridge_input": "same_date_zscores_missing_to_cross_sectional_mean_0",
                "lightgbm_missing": "native_nan_handling",
                "fit_scope": "each_training_fold_only",
            },
            "models": model_reports,
            "oof_contract": {
                "prediction_dtype": "float32",
                "metrics_computed_at_persisted_precision": True,
                "keys_consistent_across_models_per_fold": oof_key_consistent,
                "all_keys_unique": all(
                    fold["oof"]["key_unique"] for model in model_reports.values() for fold in model["folds"]
                ),
                "all_predictions_finite": all(
                    fold["oof"]["pred_finite"] for model in model_reports.values() for fold in model["folds"]
                ),
            },
            "catboost": {
                "status": "deferred",
                "reason": "optional_in_plan_and_deferred_under_16GB_first-baseline_resource_boundary",
            },
            "outputs": str(output_dir),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if output_dir.exists():
            backup = output_dir.with_name(f"{output_dir.name}.backup")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(output_dir, backup)
            try:
                os.replace(temporary, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output_dir)
        audit_path = Path(settings["audit_output"])
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_tmp = audit_path.with_suffix(".tmp.json")
        audit_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        audit_tmp.replace(audit_path)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
