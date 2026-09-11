"""Phase 17 locked-model inference with causal train-to-test state transfer."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.loader import load_test
from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)
from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores


KEYS = ["ts_code", "trade_date"]


def _key_hash(frame: pd.DataFrame) -> str:
    value = int(pd.util.hash_pandas_object(frame[KEYS], index=False).sum()) & ((1 << 64) - 1)
    return f"{value:016x}"


def validate_exact_keys(prediction: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    """Reject missing, extra or duplicate prediction keys."""
    if prediction.duplicated(KEYS).any() or expected.duplicated(KEYS).any():
        raise ValueError("Prediction and expected keys must each be unique")
    left = prediction[KEYS].sort_values(KEYS, kind="stable").reset_index(drop=True)
    right = expected[KEYS].sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Prediction keys do not exactly match raw test keys")
    return {"rows": int(len(left)), "key_hash": _key_hash(left), "exact_match": True}


def boundary_postprocess(
    warmup: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
    exit_fraction: float,
    denominator: int,
    minimum_pool: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run warm-up, then explicitly pass EMA and Top-set state into test."""
    if warmup.empty or test.empty or int(warmup["trade_date"].max()) >= int(test["trade_date"].min()):
        raise ValueError("Warm-up must be non-empty and strictly precede test")
    warm_smooth, ema_state = causal_rank_ema(warmup, alpha)
    _, top_state = hysteresis_rank_scores(
        warmup, warm_smooth, exit_fraction,
        top_fraction_denominator=denominator, minimum_pool_size=minimum_pool,
    )
    test_smooth, final_ema = causal_rank_ema(test, alpha, prior_state=ema_state)
    test_pred, final_top = hysteresis_rank_scores(
        test, test_smooth, exit_fraction,
        top_fraction_denominator=denominator, minimum_pool_size=minimum_pool,
        prior_top_set=top_state,
    )
    return test_pred, {
        "warmup_ema_state_size": len(ema_state), "warmup_top_state_size": len(top_state),
        "final_ema_state_size": len(final_ema), "final_top_state_size": len(final_top),
    }


def _read_partitions(root: Path, columns: list[str], dates: set[int] | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("year=*/*.parquet")):
        frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
        if dates is not None:
            frame = frame.loc[frame["trade_date"].isin(dates)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"No partition rows found: {root}")
    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "ts_code"], kind="stable"
    ).reset_index(drop=True)
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Partition keys are duplicated: {root}")
    return frame


def _last_dates(root: Path, count: int) -> tuple[int, ...]:
    dates: set[int] = set()
    for path in sorted(root.glob("year=*/*.parquet")):
        dates.update(int(x) for x in pd.read_parquet(path, columns=["trade_date"])["trade_date"].unique())
    selected = tuple(sorted(dates)[-count:])
    if len(selected) != count:
        raise ValueError(f"Expected {count} warm-up dates, found {len(selected)}")
    return selected


def _load_warmup(config: dict[str, Any], manifest: dict[str, Any]) -> pd.DataFrame:
    base_root = Path(config["baseline"]["model_view_output"])
    count = int(config["downstream"]["inference"]["postprocess_state_warmup_trade_days"])
    dates = _last_dates(base_root, count); date_set = set(dates)
    models = manifest["final_training"]["models"]
    base_features = list(models["ranker_base"]["features"])
    ridge_features = list(models["regression_ridge"]["features"])
    columns = [*KEYS, "flag_limit_up", *dict.fromkeys([*base_features, *ridge_features])]
    base = _read_partitions(base_root, columns, date_set)
    atr_names = [x for x in models["atr_2y"]["features"] if x not in base_features]
    atr = _read_partitions(Path(config["stability"]["augmented_view_output"]), [*KEYS, *atr_names], date_set)
    result = base.merge(atr, on=KEYS, how="inner", validate="one_to_one")
    if len(result) != len(base) or result["trade_date"].nunique() != count:
        raise ValueError("Warm-up ATR join lost rows or dates")
    return result


def _load_test_view(config: dict[str, Any], manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_test(config, use_cache=True)[[*KEYS, "flag_limit_up"]].copy()
    expected = raw[KEYS].copy()
    models = manifest["final_training"]["models"]
    base_features = list(models["ranker_base"]["features"])
    ridge_features = list(models["regression_ridge"]["features"])
    market = [x for x in base_features if x.startswith("market_")]
    cross = [x for x in dict.fromkeys([*base_features, *ridge_features]) if x not in market]
    view = _read_partitions(Path(config["test_features"]["cross_sectional_by_year_output"]), [*KEYS, *cross])
    market_frame = pd.read_parquet(Path(config["test_features"]["market_output"]), columns=["trade_date", *market])
    if market_frame["trade_date"].duplicated().any():
        raise ValueError("Test market features have duplicate dates")
    view = view.merge(market_frame, on="trade_date", how="left", validate="many_to_one")
    advanced_names = [x.removesuffix("__rank") for x in models["atr_2y"]["features"] if x not in base_features]
    advanced = _read_partitions(Path(config["advanced_features"]["test_output"]), [*KEYS, *advanced_names])
    advanced_rank = advanced.groupby("trade_date", sort=False)[advanced_names].rank(
        method="average", pct=True, na_option="keep"
    ).fillna(float(config["stability"]["missing_rank_fill"])).astype("float32")
    advanced_rank.columns = [f"{x}__rank" for x in advanced_names]
    advanced = pd.concat([advanced[KEYS], advanced_rank], axis=1)
    view = view.merge(advanced, on=KEYS, how="inner", validate="one_to_one")
    view = view.merge(raw, on=KEYS, how="inner", validate="one_to_one").sort_values(
        ["trade_date", "ts_code"], kind="stable"
    ).reset_index(drop=True)
    validate_exact_keys(view, expected)
    return view, expected


def _rank_by_date(dates: pd.Series, values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"trade_date": dates.to_numpy(), "value": np.asarray(values, dtype="float64")})
    if not np.isfinite(frame["value"].to_numpy()).all():
        raise ValueError("Model produced non-finite raw predictions")
    return frame.groupby("trade_date", sort=False)["value"].rank(method="average", pct=True).to_numpy()


def _predict_blend(frame: pd.DataFrame, manifest: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    models = manifest["final_training"]["models"]
    selected = manifest["selected"]; sleeve_weights = manifest["regression_sleeve_weights"]
    base_model = lgb.Booster(model_file=models["ranker_base"]["path"])
    base_features = list(models["ranker_base"]["features"])
    if base_model.feature_name() != base_features:
        raise ValueError("Base ranker feature schema differs from manifest")
    base_rank = _rank_by_date(frame["trade_date"], base_model.predict(
        frame[base_features].to_numpy(dtype="float32", na_value=np.nan)
    ))
    del base_model; gc.collect()

    atr_model = lgb.Booster(model_file=models["atr_2y"]["path"])
    atr_features = list(models["atr_2y"]["features"])
    if atr_model.feature_name() != atr_features:
        raise ValueError("ATR ranker feature schema differs from manifest")
    atr_rank = _rank_by_date(frame["trade_date"], atr_model.predict(
        frame[atr_features].to_numpy(dtype="float32", na_value=np.nan)
    ))
    del atr_model; gc.collect()

    lgb_model = lgb.Booster(model_file=models["regression_lightgbm"]["path"])
    lgb_features = list(models["regression_lightgbm"]["features"])
    if lgb_model.feature_name() != lgb_features:
        raise ValueError("(" + "Regression LightGBM feature schema differs from manifest" + ")")
    lgb_rank = _rank_by_date(frame["trade_date"], lgb_model.predict(
        frame[lgb_features].to_numpy(dtype="float32", na_value=np.nan)
    ))
    del lgb_model; gc.collect()

    ridge = np.load(models["regression_ridge"]["path"])
    ridge_features = [str(x) for x in ridge["features"]]
    if ridge_features != list(models["regression_ridge"]["features"]):
        raise ValueError("Ridge feature schema differs from manifest")
    x = frame[ridge_features].to_numpy(dtype="float64", na_value=np.nan)
    x[~np.isfinite(x)] = 0.0
    ridge_rank = _rank_by_date(frame["trade_date"], x @ ridge["coefficients"] + float(ridge["intercept"]))
    sleeve = float(sleeve_weights["lightgbm"]) * lgb_rank + float(sleeve_weights["ridge"]) * ridge_rank
    sleeve_rank = _rank_by_date(frame["trade_date"], sleeve)
    blended = (float(selected["ranker_weight"]) * base_rank
               + float(selected["atr_2y_weight"]) * atr_rank
               + float(selected["regression_sleeve_weight"]) * sleeve_rank)
    return blended, {
        "rows": len(frame), "dates": int(frame["trade_date"].nunique()),
        "component_predictions_finite": True,
        "feature_counts": {"ranker_base": len(base_features), "atr_2y": len(atr_features),
                           "regression_lightgbm": len(lgb_features), "regression_ridge": len(ridge_features)},
    }


def _schema_match(warm: pd.DataFrame, test: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, model in manifest["final_training"]["models"].items():
        features = list(model["features"])
        missing_warm = sorted(set(features) - set(warm.columns)); missing_test = sorted(set(features) - set(test.columns))
        order_ok = list(warm[features].columns) == features and list(test[features].columns) == features
        dtype_ok = [str(x) for x in warm[features].dtypes] == [str(x) for x in test[features].dtypes]
        result[name] = {"features": len(features), "missing_warmup": missing_warm, "missing_test": missing_test,
                        "order_match": order_ok, "dtype_match": dtype_ok}
        if missing_warm or missing_test or not order_ok or not dtype_ok:
            raise ValueError(f"Training-tail/test feature contract failed for {name}: {result[name]}")
    return result


def run_test_inference(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Generate Phase 17 Parquet predictions; do not create a submission CSV."""
    started = time.perf_counter(); settings = config["prediction"]; root = Path(config["paths"]["root"])
    output_dir = Path(settings["output_dir"]); audit_path = Path(settings["audit_output"])
    if (output_dir.exists() or audit_path.exists()) and not overwrite:
        raise FileExistsError("Phase 17 outputs exist; pass --overwrite")
    raw_before = {x: sha256_file(Path(config["paths"][x])) for x in ("train", "test")}
    manifest_path = Path(config["downstream"]["inference"]["locked_model_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != 16 or manifest["selected"]["candidate"] != "ranker_0.60__atr2y_0.20__reg_0.20":
        raise ValueError("Unexpected unlocked model manifest")
    warm = _load_warmup(config, manifest); test, raw_keys = _load_test_view(config, manifest)
    schemas = _schema_match(warm, test, manifest)
    warm_blend, warm_report = _predict_blend(warm, manifest)
    test_blend, test_report = _predict_blend(test, manifest)
    warm_transform = warm[[*KEYS, "flag_limit_up"]].copy(); warm_transform["pred"] = warm_blend
    test_transform = test[[*KEYS, "flag_limit_up"]].copy(); test_transform["pred"] = test_blend
    predictions, state = boundary_postprocess(
        warm_transform, test_transform, alpha=float(manifest["locked_postprocess"]["alpha"]),
        exit_fraction=float(manifest["locked_postprocess"]["exit_fraction"]),
        denominator=int(config["evaluator"]["top_fraction_denominator"]),
        minimum_pool=int(config["evaluator"]["portfolio_min_samples"]),
    )
    sorted_prediction = test[KEYS].copy(); sorted_prediction["pred"] = predictions.astype("float32")
    ordered = raw_keys.assign(_order=np.arange(len(raw_keys), dtype="int64")).merge(
        sorted_prediction, on=KEYS, how="left", validate="one_to_one"
    ).sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    key_report = validate_exact_keys(ordered, raw_keys)
    if not np.isfinite(ordered["pred"].to_numpy(dtype="float64")).all():
        raise ValueError("Final test predictions are not finite")
    temporary = output_dir.with_name(output_dir.name + ".tmp"); shutil.rmtree(temporary, ignore_errors=True); temporary.mkdir(parents=True)
    prediction_path = temporary / "test_predictions.parquet"
    ordered.to_parquet(prediction_path, compression="zstd", index=False)
    daily = ordered.groupby("trade_date", sort=True)["pred"].agg(["count", "min", "max", "mean", "std"]).reset_index()
    daily_path = temporary / "daily_diagnostics.csv"; daily.to_csv(daily_path, index=False, lineterminator="\n")
    report = {
        "phase": 17, "experiment_id": settings["experiment_id"],
        "locked_phase16_manifest": str(manifest_path), "locked_candidate": manifest["selected"]["candidate"],
        "warmup": {**warm_report, "start": int(warm["trade_date"].min()), "end": int(warm["trade_date"].max()),
                   "requested_dates": int(config["downstream"]["inference"]["postprocess_state_warmup_trade_days"])},
        "test": {**test_report, "start": int(test["trade_date"].min()), "end": int(test["trade_date"].max())},
        "feature_contract": schemas, "state_transfer": state, "key_contract": key_report,
        "prediction": {"path": str(output_dir / prediction_path.name), "rows": len(ordered), "dtype": str(ordered["pred"].dtype),
                       "finite": True, "min": float(ordered["pred"].min()), "max": float(ordered["pred"].max()),
                       "mean": float(ordered["pred"].mean()), "bytes": prediction_path.stat().st_size,
                       "sha256": sha256_file(prediction_path)},
        "contracts": {"test_labels_read_or_inferred": False, "models_or_weights_refit": False,
                      "same_date_ranking_only": True, "single_postprocess_after_blend": True,
                      "warmup_state_carried_into_test": True, "submission_csv_created": False,
                      "raw_files_unchanged": False},
    }
    (temporary / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_dir.exists(): shutil.rmtree(output_dir)
    os.replace(temporary, output_dir)
    raw_after = {x: sha256_file(Path(config["paths"][x])) for x in ("train", "test")}
    report["raw_sha256_before"] = raw_before; report["raw_sha256_after"] = raw_after
    report["contracts"]["raw_files_unchanged"] = raw_before == raw_after
    config_path = root / "config.yaml"; config_sha = sha256_file(config_path)
    snapshot = Path(config["experiments"]["configs_dir"]) / f"phase17_prediction_{config_sha[:12]}.yaml"
    write_immutable(snapshot, config_path.read_bytes())
    source = build_source_manifest(root); source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode())
    artifact_files = [snapshot, manifest_path, output_dir / "manifest.json", output_dir / prediction_path.name,
                      output_dir / daily_path.name, Path(config["paths"]["official_evaluator"]),
                      *[Path(x["path"]) for x in manifest["final_training"]["models"].values()]]
    artifacts = {"phase": 17, "test_data_used_for_inference": True, "test_labels_used": False,
                 "source_tree_sha256": source["tree_sha256"], "artifacts": [artifact_entry(x, root) for x in artifact_files]}
    artifact_bytes = json.dumps(artifacts, ensure_ascii=False, indent=2).encode(); artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase17_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds"); git = git_state(root)
    registry = append_registry_rows(Path(config["experiments"]["registry"]), [{
        "experiment_id": settings["experiment_id"], "timestamp": timestamp, "record_type": "summary", "fold": "test_inference",
        "git_commit": git["commit"], "git_dirty": git["dirty"], "source_tree_sha256": source["tree_sha256"],
        "source_manifest_path": str(source_path), "config_path": str(snapshot), "config_sha256": config_sha,
        "feature_version": "phase17_component_views", "feature_manifest_sha256": "MULTI_COMPONENT",
        "model": "locked_phase16_ensemble_inference", "model_params": json.dumps(manifest["selected"], sort_keys=True, separators=(",", ":")),
        "train_period": f"warmup_{int(warm['trade_date'].min())}_{int(warm['trade_date'].max())}",
        "val_period": f"test_{int(test['trade_date'].min())}_{int(test['trade_date'].max())}", "seed": config["project"]["seed"],
        "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(output_dir / prediction_path.name)]),
        "notes": "label_free_test_inference_state_continued_no_submission_csv",
    }])
    report.update({"run_seconds": time.perf_counter() - started, "source_tree_sha256": source["tree_sha256"],
                   "source_manifest": str(source_path), "config_sha256": config_sha, "config_snapshot": str(snapshot),
                   "artifact_manifest": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
                   "artifact_count": len(artifacts["artifacts"]), "artifact_total_bytes": sum(x["bytes"] for x in artifacts["artifacts"]),
                   "registry": registry})
    audit_path.parent.mkdir(parents=True, exist_ok=True); audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
