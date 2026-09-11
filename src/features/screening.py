"""Leakage-safe Phase 14 feature screening and family ablation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_train_columns
from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)
from src.validation.evaluator import METRIC_KEYS, evaluate_merged_frame, validate_official_contract
from src.utils.paths import ProjectPaths


KEYS = ["ts_code", "trade_date"]
MARKET_COLUMNS = ["market_mean_ret_1", "market_std_ret_1"]
LOGGER = logging.getLogger("fintechathon2026")


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


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, path)


def daily_rank_ic(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    label: str,
    min_samples: int,
    missing_rank_fill: float,
    collect_rank_correlation: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute date-local rank IC after a fixed neutral missing-rank policy."""
    required = [*KEYS, label, *MARKET_COLUMNS, *feature_names]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing screening columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise ValueError("Screening keys must be unique")
    rows: list[pd.DataFrame] = []
    correlation_sum = np.zeros((len(feature_names), len(feature_names)), dtype="float64")
    correlation_count = np.zeros_like(correlation_sum, dtype="int32")
    for date, group in frame.groupby("trade_date", sort=True, observed=True):
        values = group[feature_names].replace([np.inf, -np.inf], np.nan)
        ranks = values.rank(method="average", pct=True, na_option="keep").fillna(missing_rank_fill)
        valid_label = group[label].notna() & np.isfinite(group[label].to_numpy(dtype="float64", na_value=np.nan))
        label_count = int(valid_label.sum())
        ic = np.full(len(feature_names), np.nan, dtype="float64")
        if label_count >= min_samples:
            y = group.loc[valid_label, label].rank(method="average", pct=True).to_numpy(dtype="float64")
            x = ranks.loc[valid_label, feature_names].to_numpy(dtype="float64")
            y_centered = y - y.mean()
            x_centered = x - x.mean(axis=0)
            denominator = np.sqrt((x_centered * x_centered).sum(axis=0) * np.dot(y_centered, y_centered))
            np.divide(x_centered.T @ y_centered, denominator, out=ic, where=denominator > 0)
        rows.append(pd.DataFrame({
            "trade_date": np.full(len(feature_names), int(date), dtype="int32"),
            "feature": feature_names,
            "ic": ic,
            "available_rows": values.notna().sum().to_numpy(dtype="int32"),
            "universe_rows": np.full(len(feature_names), len(group), dtype="int32"),
            "labeled_rows": np.full(len(feature_names), label_count, dtype="int32"),
            "market_mean_ret_1": np.full(len(feature_names), group["market_mean_ret_1"].iloc[0]),
            "market_std_ret_1": np.full(len(feature_names), group["market_std_ret_1"].iloc[0]),
        }))
        if collect_rank_correlation:
            current = ranks.corr(min_periods=min_samples).to_numpy(dtype="float64")
            finite = np.isfinite(current)
            correlation_sum[finite] += current[finite]
            correlation_count[finite] += 1
    daily = pd.concat(rows, ignore_index=True)
    daily["year"] = (daily["trade_date"] // 10000).astype("int16")
    daily["coverage"] = (daily["available_rows"] / daily["universe_rows"]).astype("float32")
    daily["ic"] = daily["ic"].astype("float32")
    daily["market_mean_ret_1"] = daily["market_mean_ret_1"].astype("float32")
    daily["market_std_ret_1"] = daily["market_std_ret_1"].astype("float32")
    return daily, correlation_sum, correlation_count


def aggregate_daily_metrics(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    """Summarize equal-weight daily IC and row-weighted feature coverage."""
    records: list[dict[str, Any]] = []
    for feature, group in daily.groupby("feature", sort=True, observed=True):
        ic = group["ic"].dropna().astype("float64")
        mean = float(ic.mean()) if len(ic) else np.nan
        std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
        records.append({
            "period": period, "feature": feature,
            "ic_mean": mean, "ic_std": std,
            "icir": mean / std if np.isfinite(std) and std > 0 else 0.0,
            "ic_positive_ratio": float((ic > 0).mean()) if len(ic) else np.nan,
            "ic_days": int(len(ic)),
            "coverage": float(group["available_rows"].sum() / group["universe_rows"].sum()),
            "daily_coverage_min": float(group["coverage"].min()),
            "daily_coverage_mean": float(group["coverage"].mean()),
        })
    return pd.DataFrame(records)


def select_features(
    screen_summary: pd.DataFrame,
    screen_yearly: pd.DataFrame,
    family_by_feature: dict[str, str],
    mean_rank_correlation: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Select from screening-period metrics only, with family and redundancy caps."""
    yearly = screen_yearly.set_index(["feature", "year"])
    decisions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for row in screen_summary.sort_values("feature").to_dict("records"):
        feature = str(row["feature"])
        direction = 1 if float(row["ic_mean"]) >= 0 else -1
        aligned_positive = float(row["ic_positive_ratio"]) if direction == 1 else 1.0 - float(row["ic_positive_ratio"])
        annual = yearly.loc[feature] if feature in yearly.index.get_level_values(0) else pd.DataFrame()
        consistent_years = int(((annual["ic_mean"] * direction) > 0).sum()) if len(annual) else 0
        checks = {
            "coverage": float(row["coverage"]) >= float(settings["min_coverage"]),
            "abs_ic_mean": abs(float(row["ic_mean"])) >= float(settings["min_abs_ic_mean"]),
            "abs_icir": abs(float(row["icir"])) >= float(settings["min_abs_icir"]),
            "aligned_positive_ratio": aligned_positive >= float(settings["min_aligned_positive_ratio"]),
            "consistent_screen_years": consistent_years >= int(settings["min_consistent_screen_years"]),
        }
        item = {
            "feature": feature, "family": family_by_feature[feature], "direction": direction,
            "screen_ic_mean": float(row["ic_mean"]), "screen_icir": float(row["icir"]),
            "screen_coverage": float(row["coverage"]),
            "aligned_positive_ratio": aligned_positive,
            "consistent_screen_years": consistent_years, "threshold_checks": checks,
            "eligible_before_redundancy": all(checks.values()), "selected": False,
            "rejection_reason": None, "correlated_with": None,
        }
        decisions.append(item)
        if item["eligible_before_redundancy"]:
            candidates.append(item)
    candidates.sort(key=lambda item: (-abs(item["screen_ic_mean"]), item["feature"]))
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    maximum_total = int(settings["max_selected_total"])
    maximum_family = int(settings["max_selected_per_family"])
    maximum_correlation = float(settings["max_abs_rank_correlation"])
    for item in candidates:
        if len(selected) >= maximum_total:
            item["rejection_reason"] = "maximum_selected_total"
            continue
        if family_counts[item["family"]] >= maximum_family:
            item["rejection_reason"] = "maximum_selected_per_family"
            continue
        conflicts = []
        for chosen in selected:
            value = float(mean_rank_correlation.loc[item["feature"], chosen["feature"]])
            if np.isfinite(value) and abs(value) > maximum_correlation:
                conflicts.append((abs(value), chosen["feature"], value))
        if conflicts:
            _, other, value = max(conflicts)
            item["rejection_reason"] = "rank_correlation"
            item["correlated_with"] = {"feature": other, "correlation": value}
            continue
        item["selected"] = True
        selected.append(item)
        family_counts[item["family"]] += 1
    for item in decisions:
        if not item["eligible_before_redundancy"]:
            failed = [name for name, passed in item["threshold_checks"].items() if not passed]
            item["rejection_reason"] = "threshold:" + ",".join(failed)
    if not selected:
        raise ValueError("No advanced feature passed the predeclared screening rules")
    return {
        "selection_period": [int(settings["screen_start"]), int(settings["screen_end"])],
        "holdout_metrics_used_for_selection": False,
        "selection_order": "descending_abs_screen_ic_then_feature_name",
        "thresholds": {
            key: settings[key] for key in (
                "min_coverage", "min_abs_ic_mean", "min_abs_icir",
                "min_aligned_positive_ratio", "min_consistent_screen_years",
                "max_abs_rank_correlation", "max_selected_per_family", "max_selected_total",
            )
        },
        "selected_features": [item["feature"] for item in selected],
        "directions": {item["feature"]: item["direction"] for item in selected},
        "families": {item["feature"]: item["family"] for item in selected},
        "decisions": decisions,
    }


def _regime_metrics(daily: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    screen = daily[(daily["trade_date"] >= int(settings["screen_start"])) & (daily["trade_date"] <= int(settings["screen_end"]))]
    date_market = screen.drop_duplicates("trade_date")[["trade_date", *MARKET_COLUMNS]]
    low_q, high_q = (float(value) for value in settings["regime_quantiles"])
    thresholds = {
        "market_std_low": float(date_market["market_std_ret_1"].quantile(low_q)),
        "market_std_high": float(date_market["market_std_ret_1"].quantile(high_q)),
        "market_return_low": float(date_market["market_mean_ret_1"].quantile(low_q)),
        "market_return_high": float(date_market["market_mean_ret_1"].quantile(high_q)),
    }
    periods = {
        "screen": screen,
        "holdout": daily[daily["year"].isin([int(value) for value in settings["holdout_years"]])],
    }
    rows: list[pd.DataFrame] = []
    for period, subset in periods.items():
        regimes = {
            "low_volatility": subset[subset["market_std_ret_1"] <= thresholds["market_std_low"]],
            "high_volatility": subset[subset["market_std_ret_1"] >= thresholds["market_std_high"]],
            "bearish_market": subset[subset["market_mean_ret_1"] <= thresholds["market_return_low"]],
            "bullish_market": subset[subset["market_mean_ret_1"] >= thresholds["market_return_high"]],
        }
        for regime, current in regimes.items():
            summary = aggregate_daily_metrics(current, period)
            summary.insert(1, "regime", regime)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True), thresholds


def _candidate_definitions(selection: dict[str, Any]) -> dict[str, list[str]]:
    selected = list(selection["selected_features"])
    families = selection["families"]
    by_family: dict[str, list[str]] = {}
    for feature in selected:
        by_family.setdefault(families[feature], []).append(feature)
    definitions: dict[str, list[str]] = {"all_selected": selected}
    definitions.update({f"family_only:{family}": names for family, names in sorted(by_family.items())})
    for family in sorted(by_family):
        remaining = [name for name in selected if families[name] != family]
        if remaining:
            definitions[f"drop_family:{family}"] = remaining
    return definitions


def _evaluate_ablation_year(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    definitions: dict[str, list[str]],
    year: int,
    missing_rank_fill: float,
) -> list[dict[str, Any]]:
    selected = list(selection["selected_features"])
    ranks = frame.groupby("trade_date", sort=False, observed=True)[selected].rank(
        method="average", pct=True, na_option="keep"
    ).fillna(missing_rank_fill)
    for feature, direction in selection["directions"].items():
        if int(direction) < 0:
            ranks[feature] = 1.0 - ranks[feature]
    base = frame[[*KEYS, "y_ret_1d", "flag_limit_up"]].copy()
    results = []
    for candidate, features in definitions.items():
        evaluation = base.copy()
        evaluation["pred"] = ranks[features].mean(axis=1).astype("float32")
        metrics = evaluate_merged_frame(evaluation)
        results.append({
            "record_type": "year", "candidate": candidate, "year": year,
            "feature_count": len(features), "features": "|".join(features), **metrics,
        })
    return results


def _aggregate_ablation(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for candidate, group in rows.groupby("candidate", sort=True, observed=True):
        record: dict[str, Any] = {
            "record_type": "holdout_equal_weight", "candidate": candidate,
            "year": "2022|2023|2024", "feature_count": int(group["feature_count"].iloc[0]),
            "features": group["features"].iloc[0],
        }
        for metric in METRIC_KEYS:
            record[metric] = float(group[metric].mean())
        record["final_score_std"] = float(group["final_score"].std(ddof=0))
        record["final_score_worst"] = float(group["final_score"].min())
        summaries.append(record)
    return pd.concat([rows, pd.DataFrame(summaries)], ignore_index=True)


def run_feature_screening(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Run Phase 14 without reading test data or fitting a predictive model."""
    started = time.perf_counter()
    settings = config["feature_screening"]
    validate_official_contract(config)
    output_paths = [Path(settings[key]) for key in (
        "daily_output", "summary_output", "yearly_output", "regime_output",
        "redundancy_output", "ablation_output", "selection_output", "audit_output",
    )]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Phase 14 outputs exist; pass --overwrite: {existing}")
    input_dir = Path(settings["input"])
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = list(manifest["feature_names"])
    dictionary = pd.read_csv(Path(settings["dictionary_input"]))
    family_by_feature = dictionary.set_index("feature")["family"].astype(str).to_dict()
    if set(feature_names) != set(family_by_feature):
        raise ValueError("Phase 13 manifest and feature dictionary differ")
    paths = ProjectPaths.from_config(config)
    raw_before = sha256_file(paths.train)
    market = pd.read_parquet(Path(config["features"]["market_output"]), columns=["trade_date", *MARKET_COLUMNS])
    label = str(config["data"]["label"])
    daily_parts: list[pd.DataFrame] = []
    correlation_sum = np.zeros((len(feature_names), len(feature_names)), dtype="float64")
    correlation_count = np.zeros_like(correlation_sum, dtype="int32")
    year_costs: dict[str, Any] = {}
    joined_by_holdout_year: dict[int, pd.DataFrame] = {}
    for year_dir in sorted(input_dir.glob("year=*")):
        year_started = time.perf_counter()
        year = int(year_dir.name.split("=", 1)[1])
        feature_path = year_dir / "features.parquet"
        features = pd.read_parquet(feature_path, engine="pyarrow")
        labels = load_train_columns(
            config, [*KEYS, label, "flag_limit_up"], use_cache=True,
            filters=[("trade_date", ">=", year * 10000 + 101), ("trade_date", "<=", year * 10000 + 1231)],
        )
        joined = features.merge(labels, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if len(joined) != len(features) or not joined["_merge"].eq("both").all():
            raise ValueError(f"Feature/label key alignment failed for {year}")
        joined = joined.drop(columns="_merge").merge(market, on="trade_date", how="left", validate="many_to_one")
        collect = int(settings["screen_start"]) <= year * 10000 + 1231 and year * 10000 + 101 <= int(settings["screen_end"])
        daily, corr_sum, corr_count = daily_rank_ic(
            joined, feature_names, label=label,
            min_samples=int(settings["min_daily_samples"]),
            missing_rank_fill=float(settings["missing_rank_fill"]),
            collect_rank_correlation=collect,
        )
        daily_parts.append(daily)
        correlation_sum += corr_sum
        correlation_count += corr_count
        if year in [int(value) for value in settings["holdout_years"]]:
            joined_by_holdout_year[year] = joined[[*KEYS, label, "flag_limit_up", *feature_names]].copy()
        year_costs[str(year)] = {
            "rows": int(len(joined)), "input_bytes": feature_path.stat().st_size,
            "analysis_seconds": time.perf_counter() - year_started,
        }
        LOGGER.info("Phase 14 daily screening finished year=%s rows=%s", year, len(joined))
        del features, labels, joined
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(["trade_date", "feature"], kind="stable")
    screen_mask = (daily["trade_date"] >= int(settings["screen_start"])) & (daily["trade_date"] <= int(settings["screen_end"]))
    screen = daily[screen_mask]
    yearly_parts = []
    for year, group in daily.groupby("year", sort=True, observed=True):
        current = aggregate_daily_metrics(group, f"year_{int(year)}")
        current.insert(1, "year", int(year))
        yearly_parts.append(current)
    yearly = pd.concat(yearly_parts, ignore_index=True)
    summaries = [aggregate_daily_metrics(screen, "screen_2018_2021")]
    for year in [int(value) for value in settings["holdout_years"]]:
        summaries.append(aggregate_daily_metrics(daily[daily["year"] == year], f"holdout_{year}"))
    summaries.append(aggregate_daily_metrics(daily[daily["year"].isin(settings["holdout_years"])], "holdout_combined"))
    summaries.append(aggregate_daily_metrics(daily, "all_descriptive_not_for_selection"))
    summary = pd.concat(summaries, ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_correlation = np.divide(
            correlation_sum, correlation_count,
            out=np.full_like(correlation_sum, np.nan), where=correlation_count > 0,
        )
    correlation = pd.DataFrame(mean_correlation, index=feature_names, columns=feature_names)
    correlation_long = correlation.rename_axis("feature").reset_index().melt(
        id_vars="feature", var_name="other_feature", value_name="mean_daily_rank_correlation"
    )
    screen_summary = summary[summary["period"] == "screen_2018_2021"]
    screen_yearly = yearly[yearly["year"].between(2018, 2021)]
    selection = select_features(screen_summary, screen_yearly, family_by_feature, correlation, settings)
    regime, regime_thresholds = _regime_metrics(daily, settings)
    definitions = _candidate_definitions(selection)
    ablation_rows: list[dict[str, Any]] = []
    for year in [int(value) for value in settings["holdout_years"]]:
        ablation_rows.extend(_evaluate_ablation_year(
            joined_by_holdout_year[year], selection, definitions, year,
            float(settings["missing_rank_fill"]),
        ))
    ablation = _aggregate_ablation(pd.DataFrame(ablation_rows))
    _atomic_parquet(daily.reset_index(drop=True), Path(settings["daily_output"]))
    _atomic_csv(summary, Path(settings["summary_output"]))
    _atomic_csv(yearly, Path(settings["yearly_output"]))
    _atomic_csv(regime, Path(settings["regime_output"]))
    _atomic_csv(correlation_long, Path(settings["redundancy_output"]))
    _atomic_csv(ablation, Path(settings["ablation_output"]))
    _atomic_json(Path(settings["selection_output"]), selection)

    all_selected = ablation[
        (ablation["record_type"] == "holdout_equal_weight") & (ablation["candidate"] == "all_selected")
    ].iloc[0]
    raw_after = sha256_file(paths.train)
    report = {
        "phase": 14, "version": settings["version"], "feature_version": manifest["version"],
        "feature_count_screened": len(feature_names), "selected_feature_count": len(selection["selected_features"]),
        "selected_features": selection["selected_features"], "selected_directions": selection["directions"],
        "selection_period": selection["selection_period"], "holdout_years": list(settings["holdout_years"]),
        "daily_metric_rows": len(daily), "daily_metric_dates": int(daily["trade_date"].nunique()),
        "regime_thresholds_fit_on_screen_only": regime_thresholds,
        "family_ablation_candidates": len(definitions),
        "all_selected_holdout_equal_weight": {key: float(all_selected[key]) for key in METRIC_KEYS},
        "all_selected_final_score_std": float(all_selected["final_score_std"]),
        "all_selected_final_score_worst": float(all_selected["final_score_worst"]),
        "cost": {"years": year_costs, "run_seconds_before_provenance": time.perf_counter() - started},
        "outputs": {key: str(settings[key]) for key in (
            "daily_output", "summary_output", "yearly_output", "regime_output",
            "redundancy_output", "ablation_output", "selection_output",
        )},
        "contracts": {
            "selection_uses_screen_period_only": True,
            "holdout_metrics_used_for_selection": False,
            "test_data_read": False, "test_labels_created_or_inferred": False,
            "predictive_model_trained": False,
            "ablation_is_label_aware_proxy_not_final_model_performance": True,
            "same_date_rank_only": True, "future_dates_used_in_features": False,
            "raw_train_unchanged": raw_before == raw_after,
        },
        "raw_train_sha256_before": raw_before, "raw_train_sha256_after": raw_after,
    }
    root = Path(config["paths"]["root"])
    config_path = root / "config.yaml"
    config_sha = sha256_file(config_path)
    config_snapshot = Path(config["experiments"]["configs_dir"]) / f"phase14_screening_{config_sha[:12]}.yaml"
    write_immutable(config_snapshot, config_path.read_bytes())
    source = build_source_manifest(root)
    source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [
        paths.train, Path(config["paths"]["official_evaluator"]), manifest_path,
        *sorted(input_dir.glob("year=*/features.parquet")), config_snapshot,
        *[Path(settings[key]) for key in (
            "daily_output", "summary_output", "yearly_output", "regime_output",
            "redundancy_output", "ablation_output", "selection_output",
        )],
    ]
    artifact_manifest = {
        "phase": 14, "test_data_used": False, "predictive_model_trained": False,
        "source_tree_sha256": source["tree_sha256"], "feature_version": manifest["version"],
        "artifacts": [artifact_entry(path, root) for path in artifact_files],
    }
    artifact_bytes = json.dumps(artifact_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest().upper()
    artifact_path = Path(config["experiments"]["manifests_dir"]) / f"phase14_artifacts_{artifact_sha[:12]}.json"
    write_immutable(artifact_path, artifact_bytes)
    git = git_state(root)
    registry = append_registry_rows(Path(config["experiments"]["registry"]), [{
        "experiment_id": settings["experiment_id"],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "record_type": "artifact", "fold": "screen_2018_2021_holdout_2022_2024",
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "source_tree_sha256": source["tree_sha256"], "source_manifest_path": str(source_path),
        "config_path": str(config_snapshot), "config_sha256": config_sha,
        "feature_version": manifest["version"], "feature_manifest_sha256": sha256_file(manifest_path),
        "model": "feature_screening_and_rank_composite_ablation_no_predictive_model",
        "model_params": json.dumps(selection["thresholds"], sort_keys=True, separators=(",", ":")),
        "train_period": "20180102-20211231", "val_period": "2022|2023|2024",
        "seed": config["project"]["seed"],
        "artifact_manifest_path": str(artifact_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(path) for path in output_paths[:-1]], separators=(",", ":")),
        "notes": "holdout_not_used_for_selection_proxy_not_final_model_performance_no_test_data",
    }])
    audit = {
        **report, "run_seconds": time.perf_counter() - started,
        "source_tree_sha256": source["tree_sha256"], "source_file_count": source["file_count"],
        "source_manifest": str(source_path), "config_sha256": config_sha,
        "config_snapshot": str(config_snapshot), "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha, "artifact_count": len(artifact_manifest["artifacts"]),
        "artifact_total_bytes": sum(item["bytes"] for item in artifact_manifest["artifacts"]),
        "git": git, "registry": {"path": config["experiments"]["registry"], **registry},
    }
    _atomic_json(Path(settings["audit_output"]), audit)
    return audit
