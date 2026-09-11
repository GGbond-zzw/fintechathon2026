"""Phase 15 ATR increment, training-window and regime stability experiments."""

from __future__ import annotations

import gc
import copy
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)
from src.models.baselines import (
    KEYS, META, _aggregate, _evaluate_predictions, _fold_dates,
    _iter_view_batches, _load_validation, _save_oof,
)
from src.models.dataset import load_model_view_manifest
from src.models.ranking import _write_training_memmaps
from src.postprocess.turnover_control import causal_rank_ema, hysteresis_rank_scores
from src.validation.evaluator import METRIC_KEYS, evaluate_merged_frame, validate_official_contract


LOGGER = logging.getLogger("fintechathon2026")


def _key_hash(frame: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(frame[KEYS], index=False).sum()) & ((1 << 64) - 1)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def trailing_train_dates(
    train_dates: tuple[int, ...], val_dates: tuple[int, ...], years: int | None,
) -> tuple[int, ...]:
    """Restrict an already-purged fold calendar to trailing calendar years."""
    if not train_dates or not val_dates:
        raise ValueError("Training and validation dates must be non-empty")
    if max(train_dates) >= min(val_dates):
        raise ValueError("Training dates must precede validation dates")
    if years is None:
        return train_dates
    if int(years) < 1:
        raise ValueError("Trailing years must be positive")
    cutoff = (min(val_dates) // 10000 - int(years)) * 10000 + 101
    selected = tuple(date for date in train_dates if date >= cutoff)
    if not selected:
        raise ValueError("Training window removed every training date")
    return selected


def build_atr_augmented_view(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Create a compact training view containing base ranker features plus ATR ranks."""
    settings = config["stability"]
    output_dir = Path(settings["augmented_view_output"])
    if output_dir.exists() and not overwrite:
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    base_dir = Path(settings["base_view_input"])
    advanced_dir = Path(settings["advanced_input"])
    base_manifest = load_model_view_manifest(config)
    base_features = [
        feature for group in config["ranking"]["feature_groups"]
        for feature in base_manifest["feature_groups"][group]
    ]
    selection = json.loads(Path(settings["phase14_selection"]).read_text(encoding="utf-8"))
    advanced_features = list(settings["advanced_features"])
    if not set(advanced_features).issubset(selection["selected_features"]):
        raise ValueError("Phase 15 ATR inputs are not in the locked Phase 14 allowlist")
    if {selection["families"][name] for name in advanced_features} != {settings["selected_family"]}:
        raise ValueError("Phase 15 advanced features do not match the configured family")
    advanced_rank = [f"{name}__rank" for name in advanced_features]
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    reports: dict[str, Any] = {}
    total_rows = 0
    total_hash = 0
    try:
        for base_year in sorted(base_dir.glob("year=*")):
            year = int(base_year.name.split("=", 1)[1])
            base = pd.read_parquet(base_year / "data.parquet", columns=[*META, *base_features])
            advanced = pd.read_parquet(
                advanced_dir / f"year={year}" / "features.parquet",
                columns=[*KEYS, *advanced_features],
            )
            if len(base) != len(advanced) or _key_hash(base) != _key_hash(advanced):
                raise ValueError(f"Base and advanced keys differ for {year}")
            ordered = advanced.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
            ranks = ordered.groupby("trade_date", sort=False, observed=True)[advanced_features].rank(
                method="average", pct=True, na_option="keep"
            ).fillna(float(settings["missing_rank_fill"])).astype("float32")
            ranks.columns = advanced_rank
            cross = pd.concat([ordered[KEYS], ranks], axis=1)
            joined = base.merge(cross, on=KEYS, how="inner", validate="one_to_one")
            joined = joined[[*META, *base_features, *advanced_rank]].sort_values(
                ["trade_date", "ts_code"], kind="stable"
            ).reset_index(drop=True)
            if len(joined) != len(base) or joined.duplicated(KEYS).any():
                raise ValueError(f"ATR augmented view lost or duplicated rows for {year}")
            if not np.isfinite(joined[advanced_rank].to_numpy(dtype="float64")).all():
                raise ValueError(f"ATR ranks are not finite for {year}")
            target_dir = temporary / f"year={year}"
            target_dir.mkdir(parents=True)
            target = target_dir / "data.parquet"
            joined.to_parquet(target, engine="pyarrow", compression=settings["parquet_compression"], index=False)
            key_hash = _key_hash(joined)
            reports[str(year)] = {
                "rows": int(len(joined)), "labeled_rows": int(joined["y_ret_1d"].notna().sum()),
                "key_hash": f"{key_hash:016x}", "key_unique": True, "bytes": target.stat().st_size,
            }
            total_rows += len(joined)
            total_hash = (total_hash + key_hash) & ((1 << 64) - 1)
            LOGGER.info("Phase 15 augmented view finished year=%s rows=%s", year, len(joined))
            del base, advanced, ordered, ranks, cross, joined
        report = {
            "version": settings["version"], "rows": total_rows,
            "label": "y_ret_1d", "feature_count": len(base_features) + len(advanced_rank),
            "feature_groups": {"base_rank_market": base_features, "advanced_rank": advanced_rank},
            "feature_names": [*base_features, *advanced_rank],
            "advanced_source_features": advanced_features,
            "missing_rank_fill": float(settings["missing_rank_fill"]),
            "key_hash": f"{total_hash:016x}", "years": reports,
        }
        (temporary / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if output_dir.exists():
            backup = output_dir.with_name(f"{output_dir.name}.backup")
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(output_dir, backup)
            try:
                os.replace(temporary, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output_dir)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _count_labeled(view_dir: Path, dates: tuple[int, ...], batch_size: int) -> int:
    return sum(
        int(frame["y_ret_1d"].notna().sum())
        for frame in _iter_view_batches(view_dir, dates, ["trade_date", "y_ret_1d"], batch_size=batch_size)
    )


def _existing_oof_report(path: Path) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    frame = pd.read_parquet(path)
    if frame.duplicated(KEYS).any() or str(frame["pred"].dtype) != "float32":
        raise ValueError(f"Recovered OOF contract failed: {path}")
    if not np.isfinite(frame["pred"].to_numpy(dtype="float64")).all():
        raise ValueError(f"Recovered OOF contains non-finite predictions: {path}")
    metrics = {key: float(value) for key, value in evaluate_merged_frame(frame).items()}
    labeled = frame["y_ret_1d"].notna()
    error = frame.loc[labeled, "pred"] - frame.loc[labeled, "y_ret_1d"]
    diagnostics = {
        "labeled_rows": int(labeled.sum()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }
    report = {
        "path": str(path), "rows": int(len(frame)),
        "key_hash": f"{_key_hash(frame):016x}", "key_unique": True,
        "pred_finite": True, "bytes": path.stat().st_size, "pred_dtype": "float32",
    }
    return report, metrics, diagnostics


def _daily_oof_diagnostics(frame: pd.DataFrame, fold: str, source: str) -> pd.DataFrame:
    rows = []
    previous: set[str] | None = None
    for date, group in frame.sort_values(["trade_date", "ts_code"], kind="stable").groupby("trade_date", sort=True):
        labeled = group[group["y_ret_1d"].notna()]
        ic = float(spearmanr(labeled["pred"], labeled["y_ret_1d"]).statistic) if len(labeled) >= 30 else np.nan
        eligible = group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()].sort_values("pred", ascending=False)
        excess = np.nan
        if len(eligible) >= 100:
            top_n = max(len(eligible) // 10, 1)
            excess = float(eligible["y_ret_1d"].iloc[:top_n].mean() - eligible["y_ret_1d"].mean())
        turnover_pool = group[group["flag_limit_up"] == 0].sort_values("pred", ascending=False)
        turnover = np.nan
        if len(turnover_pool) < 100:
            previous = None
        else:
            top_n = max(len(turnover_pool) // 10, 1)
            current = set(turnover_pool["ts_code"].astype(str).iloc[:top_n])
            if previous:
                turnover = 1.0 - len(current & previous) / len(current | previous)
            previous = current
        rows.append({
            "source": source, "fold": fold, "trade_date": int(date),
            "ic": ic, "daily_excess": excess, "turnover": turnover,
        })
    return pd.DataFrame(rows)


def _regime_table(daily: pd.DataFrame, market: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    merged = daily.merge(market, on="trade_date", how="left", validate="many_to_one")
    rows = []
    for source, group in merged.groupby("source", sort=True, observed=True):
        masks = {
            "all": pd.Series(True, index=group.index),
            "low_volatility": group["market_std_ret_1"] <= thresholds["market_std_low"],
            "high_volatility": group["market_std_ret_1"] >= thresholds["market_std_high"],
            "bearish_market": group["market_mean_ret_1"] <= thresholds["market_return_low"],
            "bullish_market": group["market_mean_ret_1"] >= thresholds["market_return_high"],
        }
        for regime, mask in masks.items():
            part = group.loc[mask]
            ic = float(part["ic"].mean())
            annual_excess = float(part["daily_excess"].mean() * 252)
            turnover = float(part["turnover"].mean())
            rows.append({
                "source": source, "regime": regime, "dates": int(len(part)),
                "ic_mean": ic, "annual_excess": annual_excess, "mean_turnover": turnover,
                "final_score_proxy": ic * 0.4 + annual_excess * 0.3 + (1.0 - turnover) * 0.3,
            })
    return pd.DataFrame(rows)


def _comparison_rows(models: dict[str, Any], incumbent: dict[str, Any]) -> pd.DataFrame:
    rows = []
    inc_summary = incumbent["variants"]["turnover"]["summary"]
    rows.append({
        "window": "phase12_incumbent", "prediction_variant": "turnover",
        **{f"{metric}_mean": inc_summary[metric]["mean"] for metric in METRIC_KEYS},
        "final_score_std": inc_summary["final_score"]["std"],
        "final_score_worst": inc_summary["final_score"]["min"],
    })
    for window, variants in models.items():
        for prediction_variant, payload in variants.items():
            summary = payload["summary"]
            rows.append({
                "window": window, "prediction_variant": prediction_variant,
                **{f"{metric}_mean": summary[metric]["mean"] for metric in METRIC_KEYS},
                "final_score_std": summary["final_score"]["std"],
                "final_score_worst": summary["final_score"]["min"],
            })
    result = pd.DataFrame(rows)
    incumbent_score = float(inc_summary["final_score"]["mean"])
    result["final_score_delta_vs_incumbent"] = result["final_score_mean"] - incumbent_score
    return result


def run_stability_experiments(
    config: dict[str, Any], *, overwrite: bool = False, resume: bool = False,
) -> dict[str, Any]:
    """Train three ATR-augmented rankers with predeclared training windows."""
    started = time.perf_counter()
    validate_official_contract(config)
    settings = config["stability"]
    output_dir = Path(settings["output_dir"])
    audit_path = Path(settings["audit_output"])
    comparison_path = Path(settings["comparison_output"])
    regime_path = Path(settings["regime_output"])
    existing = [str(path) for path in (output_dir, audit_path, comparison_path, regime_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Phase 15 outputs exist; pass --overwrite: {existing}")
    root = Path(config["paths"]["root"])
    raw_before = {
        "train": sha256_file(Path(config["paths"]["train"])),
        "test": sha256_file(Path(config["paths"]["test"])),
    }
    view_manifest = build_atr_augmented_view(config, overwrite=overwrite and not resume)
    view_dir = Path(settings["augmented_view_output"])
    features = list(view_manifest["feature_names"])
    folds = _fold_dates(config)
    ranking = config["ranking"]
    params = dict(ranking["params"])
    bins = int(ranking["relevance_bins"])
    batch_size = int(settings["batch_size"])
    turnover_audit = json.loads(Path(ranking["turnover_audit"]).read_text(encoding="utf-8"))
    alpha = float(turnover_audit["selected"]["alpha"])
    exit_fraction = float(turnover_audit["selected"]["exit_fraction"])
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    if not resume:
        shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(config["paths"]["cache"]) / "phase15_rank_training_scratch"
    if not resume:
        shutil.rmtree(scratch_root, ignore_errors=True)
    models: dict[str, Any] = {}
    train_periods: dict[str, dict[str, Any]] = {}
    signatures: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    incumbent = json.loads(Path(config["downstream"]["incumbent"]["audit"]).read_text(encoding="utf-8"))
    try:
        for window, years in settings["windows"].items():
            models[window] = {"raw": {"folds": []}, "turnover": {"folds": []}}
            train_periods[window] = {}
            for fold in folds:
                fold_started = time.perf_counter()
                name = fold["name"]
                train_dates = trailing_train_dates(fold["train_dates"], fold["val_dates"], years)
                expected_rows = _count_labeled(view_dir, train_dates, batch_size)
                scratch = scratch_root / window / name
                fold_dir = temporary / window / name
                raw_path = fold_dir / "ranker_raw_oof.parquet"
                stable_path = fold_dir / "ranker_turnover_oof.parquet"
                model_path = fold_dir / "lambdarank_model.txt"
                report_path = fold_dir / "fold_report.json"
                signature = (train_dates, fold["val_dates"])
                if resume and all(path.is_file() for path in (raw_path, stable_path, model_path)):
                    raw_oof, raw_metrics, raw_diagnostics = _existing_oof_report(raw_path)
                    stable_oof, stable_metrics, stable_diagnostics = _existing_oof_report(stable_path)
                    saved = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
                    common = saved["common"] if saved else {
                        "name": name, "train_rows": expected_rows, "train_dates": len(train_dates),
                        "train_start": train_dates[0], "train_end": train_dates[-1],
                        "validation_rows": raw_oof["rows"],
                        "iterations": lgb.Booster(model_file=str(model_path)).current_iteration(),
                        "seconds": None,
                        "group_audit": incumbent["training"]["groups"].get(name, {
                            "rows": expected_rows, "groups": len(train_dates),
                            "first_group_date": train_dates[0], "last_group_date": train_dates[-1],
                            "recovered_without_histogram": True,
                        }),
                        "resumed_existing_artifact": True,
                    }
                    raw_item = {**common, "metrics": raw_metrics, "diagnostics": raw_diagnostics, "oof": raw_oof}
                    stable_item = {**common, "metrics": stable_metrics, "diagnostics": stable_diagnostics, "oof": stable_oof}
                    models[window]["raw"]["folds"].append(raw_item)
                    models[window]["turnover"]["folds"].append(stable_item)
                    signatures[signature] = (fold_dir, raw_item, stable_item)
                    LOGGER.info("Phase 15 resumed window=%s fold=%s rows=%s", window, name, expected_rows)
                    train_periods[window][name] = {
                        "years": years, "start": train_dates[0], "end": train_dates[-1],
                        "dates": len(train_dates), "labeled_rows": expected_rows,
                    }
                    shutil.rmtree(scratch, ignore_errors=True)
                    continue
                if signature in signatures:
                    source_dir, source_raw, source_stable = signatures[signature]
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    for filename in ("ranker_raw_oof.parquet", "ranker_turnover_oof.parquet", "lambdarank_model.txt"):
                        shutil.copy2(source_dir / filename, fold_dir / filename)
                    raw_oof, raw_metrics, raw_diagnostics = _existing_oof_report(raw_path)
                    stable_oof, stable_metrics, stable_diagnostics = _existing_oof_report(stable_path)
                    common = copy.deepcopy({key: value for key, value in source_raw.items() if key not in ("metrics", "diagnostics", "oof")})
                    common.update({"name": name, "seconds": 0.0, "reused_identical_training_signature": str(source_dir)})
                    raw_item = {**common, "metrics": raw_metrics, "diagnostics": raw_diagnostics, "oof": raw_oof}
                    stable_item = {**common, "metrics": stable_metrics, "diagnostics": stable_diagnostics, "oof": stable_oof}
                    models[window]["raw"]["folds"].append(raw_item)
                    models[window]["turnover"]["folds"].append(stable_item)
                    _atomic_json(report_path, {"common": common})
                    signatures[signature] = (fold_dir, raw_item, stable_item)
                    train_periods[window][name] = {
                        "years": years, "start": train_dates[0], "end": train_dates[-1],
                        "dates": len(train_dates), "labeled_rows": expected_rows,
                    }
                    LOGGER.info("Phase 15 reused identical window=%s fold=%s from=%s", window, name, source_dir)
                    shutil.rmtree(scratch, ignore_errors=True)
                    continue
                shutil.rmtree(scratch, ignore_errors=True)
                x, y, groups, group_audit = _write_training_memmaps(
                    view_dir, train_dates, features, "y_ret_1d", bins, scratch,
                    expected_rows, batch_size,
                )
                train_set = lgb.Dataset(x, label=y, group=groups, feature_name=features, free_raw_data=True)
                booster = lgb.train(params, train_set, num_boost_round=int(ranking["num_boost_round"]))
                validation = _load_validation(view_dir, fold["val_dates"], features, batch_size)
                raw_pred = booster.predict(validation[features].to_numpy(dtype="float32", na_value=np.nan)).astype("float32")
                raw_metrics, raw_diagnostics = _evaluate_predictions(validation, raw_pred)
                fold_dir.mkdir(parents=True)
                raw_oof = _save_oof(fold_dir / "ranker_raw_oof.parquet", validation, raw_pred)
                transform = validation[[*KEYS, "flag_limit_up"]].copy()
                transform["pred"] = raw_pred
                smoothed, _ = causal_rank_ema(transform, alpha)
                stable_pred, _ = hysteresis_rank_scores(
                    transform, smoothed, exit_fraction,
                    top_fraction_denominator=int(config["evaluator"]["top_fraction_denominator"]),
                    minimum_pool_size=int(config["evaluator"]["portfolio_min_samples"]),
                )
                stable_metrics, stable_diagnostics = _evaluate_predictions(validation, stable_pred)
                stable_oof = _save_oof(fold_dir / "ranker_turnover_oof.parquet", validation, stable_pred)
                booster.save_model(str(fold_dir / "lambdarank_model.txt"))
                elapsed = time.perf_counter() - fold_started
                common = {
                    "name": name, "train_rows": expected_rows, "train_dates": len(train_dates),
                    "train_start": train_dates[0], "train_end": train_dates[-1],
                    "validation_rows": int(len(validation)), "iterations": booster.current_iteration(),
                    "seconds": elapsed, "group_audit": group_audit,
                }
                models[window]["raw"]["folds"].append({
                    **common, "metrics": raw_metrics, "diagnostics": raw_diagnostics, "oof": raw_oof,
                })
                models[window]["turnover"]["folds"].append({
                    **common, "metrics": stable_metrics, "diagnostics": stable_diagnostics, "oof": stable_oof,
                })
                _atomic_json(report_path, {"common": common})
                signatures[signature] = (
                    fold_dir, models[window]["raw"]["folds"][-1], models[window]["turnover"]["folds"][-1]
                )
                train_periods[window][name] = {
                    "years": years, "start": train_dates[0], "end": train_dates[-1],
                    "dates": len(train_dates), "labeled_rows": expected_rows,
                }
                LOGGER.info("Phase 15 trained window=%s fold=%s rows=%s seconds=%.1f", window, name, expected_rows, elapsed)
                del x, y, groups, train_set, booster, validation, raw_pred, stable_pred, smoothed
                gc.collect()
                shutil.rmtree(scratch, ignore_errors=True)
            for payload in models[window].values():
                payload["summary"] = _aggregate(payload["folds"])
        comparison = _comparison_rows(models, incumbent)
        turnover_candidates = comparison[
            (comparison["prediction_variant"] == "turnover") & (comparison["window"] != "phase12_incumbent")
        ].copy()
        selected_window = turnover_candidates.sort_values(
            ["final_score_mean", "final_score_worst", "mean_turnover_mean", "window"],
            ascending=[False, False, True, True], kind="stable",
        ).iloc[0]["window"]

        daily_parts = []
        for window in settings["windows"]:
            for fold in folds:
                path = temporary / window / fold["name"] / "ranker_turnover_oof.parquet"
                daily_parts.append(_daily_oof_diagnostics(pd.read_parquet(path), fold["name"], window))
        for fold in folds:
            path = Path(config["ranking"]["output_dir"]) / fold["name"] / "ranker_turnover_oof.parquet"
            daily_parts.append(_daily_oof_diagnostics(pd.read_parquet(path), fold["name"], "phase12_incumbent"))
        market = pd.read_parquet(
            Path(config["features"]["market_output"]),
            columns=["trade_date", "market_mean_ret_1", "market_std_ret_1"],
        )
        phase14 = json.loads(Path(config["feature_screening"]["audit_output"]).read_text(encoding="utf-8"))
        regime = _regime_table(
            pd.concat(daily_parts, ignore_index=True), market,
            phase14["regime_thresholds_fit_on_screen_only"],
        )
        _atomic_csv(comparison, comparison_path)
        _atomic_csv(regime, regime_path)
        report = {
            "phase": 15, "feature_version": view_manifest["version"],
            "feature_count": len(features), "base_feature_count": len(features) - 2,
            "advanced_features": list(settings["advanced_features"]),
            "advanced_selection_basis": "top_screen_period_family_from_phase14_not_holdout_score",
            "windows": train_periods, "models": models,
            "locked_training": {
                "objective": params["objective"], "relevance_bins": bins,
                "num_boost_round": int(ranking["num_boost_round"]),
                "alpha": alpha, "exit_fraction": exit_fraction,
            },
            "comparison_output": str(comparison_path), "regime_output": str(regime_path),
            "selected_window_for_phase16_review": str(selected_window),
            "selection_order": list(settings["selection_order"]),
            "incumbent": {
                "experiment_id": config["downstream"]["incumbent"]["experiment_id"],
                "final_score_mean": incumbent["variants"]["turnover"]["summary"]["final_score"]["mean"],
                "final_score_worst": incumbent["variants"]["turnover"]["summary"]["final_score"]["min"],
                "mean_turnover": incumbent["variants"]["turnover"]["summary"]["mean_turnover"]["mean"],
            },
            "contracts": {
                "same_cv_validation_dates": True, "one_day_purge_preserved": True,
                "relevance_and_model_params_match_phase12": True,
                "turnover_params_match_phase10": True,
                "training_windows_use_past_dates_only": True,
                "regime_thresholds_fit_on_phase14_screen_period": True,
                "test_data_used_for_training_or_selection": False,
                "raw_files_unchanged": False,
            },
            "output_dir": str(output_dir),
        }
        (temporary / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if output_dir.exists():
            backup = output_dir.with_name(f"{output_dir.name}.backup")
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(output_dir, backup)
            try:
                os.replace(temporary, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    for window, variants in models.items():
        for prediction_variant, payload in variants.items():
            filename = "ranker_raw_oof.parquet" if prediction_variant == "raw" else "ranker_turnover_oof.parquet"
            for fold in payload["folds"]:
                fold["oof"]["path"] = str(output_dir / window / fold["name"] / filename)
    raw_after = {
        "train": sha256_file(Path(config["paths"]["train"])),
        "test": sha256_file(Path(config["paths"]["test"])),
    }
    report["contracts"]["raw_files_unchanged"] = raw_before == raw_after
    report["raw_sha256_before"] = raw_before
    report["raw_sha256_after"] = raw_after
    report["run_seconds_before_provenance"] = time.perf_counter() - started
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    config_path = root / "config.yaml"
    config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase15_stability_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root)
    source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [
        Path(settings["augmented_view_output"]) / "manifest.json",
        *sorted(Path(settings["augmented_view_output"]).glob("year=*/data.parquet")),
        Path(config["cv"]["audit_output"]), Path(config["cv"]["calendar_output"]),
        Path(config["feature_screening"]["audit_output"]), Path(settings["phase14_selection"]),
        Path(config["downstream"]["incumbent"]["audit"]), Path(ranking["turnover_audit"]),
        Path(config["paths"]["official_evaluator"]), config_snapshot,
        comparison_path, regime_path, output_dir / "manifest.json",
        *sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
    ]
    artifact_manifest = {
        "phase": 15, "test_data_used": False, "source_tree_sha256": source["tree_sha256"],
        "feature_version": view_manifest["version"],
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase15_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)
    git = git_state(root)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    feature_manifest_sha = sha256_file(Path(settings["augmented_view_output"]) / "manifest.json")
    registry_rows = []
    for window, variants in models.items():
        for prediction_variant, payload in variants.items():
            experiment_id = settings["experiment_ids"][f"{window}_{prediction_variant}"]
            params_json = json.dumps({
                "window_years": settings["windows"][window], "advanced_features": settings["advanced_features"],
                "ranking": params, "num_boost_round": ranking["num_boost_round"],
                "alpha": alpha, "exit_fraction": exit_fraction,
            }, sort_keys=True, separators=(",", ":"))
            for item in payload["folds"]:
                registry_rows.append({
                    "experiment_id": experiment_id, "timestamp": timestamp,
                    "record_type": "fold", "fold": item["name"],
                    "git_commit": git["commit"], "git_dirty": git["dirty"],
                    "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
                    "config_path": str(config_snapshot), "config_sha256": config_sha,
                    "feature_version": view_manifest["version"], "feature_manifest_sha256": feature_manifest_sha,
                    "model": f"lightgbm_lambdarank_atr_{window}_{prediction_variant}", "model_params": params_json,
                    "train_period": f"{item['train_start']}-{item['train_end']}",
                    "val_period": item["name"].replace("fold_", ""), "seed": config["project"]["seed"],
                    **item["metrics"], "artifact_manifest_path": str(artifact_path),
                    "artifact_manifest_sha256": artifact_sha,
                    "artifact_paths": json.dumps([item["oof"]["path"], str(output_dir / window / item["name"] / "lambdarank_model.txt")], separators=(",", ":")),
                    "notes": "window_stability_same_validation_no_test_data",
                })
            summary = payload["summary"]
            registry_rows.append({
                "experiment_id": experiment_id, "timestamp": timestamp,
                "record_type": "summary", "fold": "all_equal_weight",
                "git_commit": git["commit"], "git_dirty": git["dirty"],
                "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
                "config_path": str(config_snapshot), "config_sha256": config_sha,
                "feature_version": view_manifest["version"], "feature_manifest_sha256": feature_manifest_sha,
                "model": f"lightgbm_lambdarank_atr_{window}_{prediction_variant}", "model_params": params_json,
                "train_period": window, "val_period": "2022|2023|2024", "seed": config["project"]["seed"],
                **{key: summary[key]["mean"] for key in METRIC_KEYS},
                "final_score_std": summary["final_score"]["std"],
                "final_score_worst": summary["final_score"]["min"],
                "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
                "artifact_paths": json.dumps([str(output_dir / window), str(comparison_path)], separators=(",", ":")),
                "notes": "equal_weight_fold_summary_no_test_data",
            })
    registry = append_registry_rows(Path(config["experiments"]["registry"]), registry_rows)
    audit = {
        **report, "run_seconds": time.perf_counter() - started,
        "source_tree_sha256": source["tree_sha256"], "source_file_count": source["file_count"],
        "source_manifest": str(source_path), "config_sha256": config_sha,
        "config_snapshot": str(config_snapshot), "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha, "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "git": git, "registry": {"path": config["experiments"]["registry"], **registry},
    }
    _atomic_json(audit_path, audit)
    return audit
