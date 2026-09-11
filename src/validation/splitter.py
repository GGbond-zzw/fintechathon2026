"""Trading-calendar-aware expanding-window validation splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.loader import load_train_columns
from src.utils.paths import ProjectPaths


@dataclass(frozen=True)
class FoldSplit:
    """Exact trading dates assigned to one walk-forward fold."""

    name: str
    configured_train_start: int
    configured_train_end: int
    configured_val_start: int
    configured_val_end: int
    train_dates: tuple[int, ...]
    purge_dates: tuple[int, ...]
    val_dates: tuple[int, ...]

    def date_masks(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        if "trade_date" not in frame.columns:
            raise ValueError("Split input must contain trade_date")
        return {
            "train": frame["trade_date"].isin(self.train_dates),
            "purge": frame["trade_date"].isin(self.purge_dates),
            "validation": frame["trade_date"].isin(self.val_dates),
        }

    def labeled_masks(self, frame: pd.DataFrame, label_column: str) -> dict[str, pd.Series]:
        if label_column not in frame.columns:
            raise ValueError(f"Split input must contain label column: {label_column}")
        available = frame[label_column].notna()
        return {name: mask & available for name, mask in self.date_masks(frame).items()}


class ExpandingWindowSplitter:
    """Build deterministic folds from configured bounds and observed trade dates."""

    def __init__(
        self,
        folds: Iterable[dict[str, Any]],
        *,
        purge_trade_days: int,
        label_horizon_trade_days: int,
    ) -> None:
        self.fold_settings = list(folds)
        self.purge_trade_days = int(purge_trade_days)
        self.label_horizon_trade_days = int(label_horizon_trade_days)
        if self.label_horizon_trade_days < 1:
            raise ValueError("label_horizon_trade_days must be positive")
        if self.purge_trade_days < self.label_horizon_trade_days:
            raise ValueError(
                "purge_trade_days must be at least the label horizon to prevent boundary leakage"
            )
        if not self.fold_settings:
            raise ValueError("At least one CV fold is required")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpandingWindowSplitter":
        settings = config["cv"]
        if settings.get("type") != "expanding_window":
            raise ValueError("Only expanding_window CV is supported")
        return cls(
            settings["folds"],
            purge_trade_days=int(settings["purge_trade_days"]),
            label_horizon_trade_days=int(settings.get("label_horizon_trade_days", 1)),
        )

    def split(self, trading_dates: Iterable[int]) -> list[FoldSplit]:
        calendar = np.asarray(sorted({int(value) for value in trading_dates}), dtype="int64")
        if calendar.size == 0:
            raise ValueError("Trading calendar is empty")
        results: list[FoldSplit] = []
        seen_names: set[str] = set()
        for setting in self.fold_settings:
            name = str(setting["name"])
            if name in seen_names:
                raise ValueError(f"Duplicate fold name: {name}")
            seen_names.add(name)
            train_start, train_end = int(setting["train_start"]), int(setting["train_end"])
            val_start, val_end = int(setting["val_start"]), int(setting["val_end"])
            if train_start > train_end or val_start > val_end:
                raise ValueError(f"Invalid configured date range in {name}")
            candidate_train = calendar[(calendar >= train_start) & (calendar <= train_end)]
            validation = calendar[(calendar >= val_start) & (calendar <= val_end)]
            if len(candidate_train) <= self.purge_trade_days:
                raise ValueError(f"Not enough training dates before purge in {name}")
            if len(validation) == 0:
                raise ValueError(f"No observed validation dates in {name}")
            purge = candidate_train[-self.purge_trade_days:]
            training = candidate_train[:-self.purge_trade_days]
            if int(candidate_train[-1]) >= int(validation[0]):
                raise ValueError(f"Training/validation chronology overlaps in {name}")
            split = FoldSplit(
                name=name,
                configured_train_start=train_start,
                configured_train_end=train_end,
                configured_val_start=val_start,
                configured_val_end=val_end,
                train_dates=tuple(int(value) for value in training),
                purge_dates=tuple(int(value) for value in purge),
                val_dates=tuple(int(value) for value in validation),
            )
            if set(split.train_dates) & set(split.purge_dates):
                raise AssertionError(f"Training and purge dates overlap in {name}")
            if (set(split.train_dates) | set(split.purge_dates)) & set(split.val_dates):
                raise AssertionError(f"Training-side and validation dates overlap in {name}")
            results.append(split)
        for previous, current in zip(results, results[1:]):
            if not set(previous.train_dates).issubset(current.train_dates):
                raise ValueError("Configured folds are not expanding")
        return results


def _calendar_sha256(dates: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in dates).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def build_cv_artifacts(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Materialize the exact fold calendar and a machine-readable Phase 6 audit."""
    settings = config["cv"]
    audit_path = Path(settings["audit_output"])
    calendar_path = Path(settings["calendar_output"])
    existing = [str(path) for path in (audit_path, calendar_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"CV outputs exist; pass overwrite explicitly: {existing}")
    label = str(config["data"]["label"])
    frame = load_train_columns(config, ["ts_code", "trade_date", label], use_cache=True)
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Training keys must be unique before CV splitting")
    dates = sorted(int(value) for value in frame["trade_date"].unique())
    splitter = ExpandingWindowSplitter.from_config(config)
    folds = splitter.split(dates)
    fold_reports: list[dict[str, Any]] = []
    calendar_rows: list[dict[str, Any]] = []
    for fold in folds:
        masks = fold.date_masks(frame)
        labeled = fold.labeled_masks(frame, label)
        for role, role_dates in (
            ("train", fold.train_dates), ("purge", fold.purge_dates),
            ("validation", fold.val_dates),
        ):
            calendar_rows.extend(
                {"fold": fold.name, "trade_date": date, "role": role}
                for date in role_dates
            )
        fold_reports.append({
            "name": fold.name,
            "configured": {
                "train_start": fold.configured_train_start,
                "train_end": fold.configured_train_end,
                "val_start": fold.configured_val_start,
                "val_end": fold.configured_val_end,
            },
            "actual": {
                "train_start": fold.train_dates[0],
                "train_end": fold.train_dates[-1],
                "purge_dates": list(fold.purge_dates),
                "val_start": fold.val_dates[0],
                "val_end": fold.val_dates[-1],
            },
            "trade_date_counts": {
                "train": len(fold.train_dates),
                "purge": len(fold.purge_dates),
                "validation": len(fold.val_dates),
            },
            "row_counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "labeled_row_counts": {name: int(mask.sum()) for name, mask in labeled.items()},
            "unlabeled_row_counts": {
                name: int(masks[name].sum() - labeled[name].sum()) for name in masks
            },
            "train_validation_disjoint": not bool((masks["train"] & masks["validation"]).any()),
            "purge_excluded_from_train": not bool((masks["train"] & masks["purge"]).any()),
            "chronology_valid": fold.train_dates[-1] < fold.purge_dates[0] < fold.val_dates[0],
        })

    calendar_frame = pd.DataFrame(calendar_rows).astype({"fold": "string", "trade_date": "int32", "role": "string"})
    if calendar_frame.duplicated(["fold", "trade_date"]).any():
        raise AssertionError("A date was assigned multiple roles within one fold")
    daily_rows = frame.groupby("trade_date", sort=True, observed=True).size()
    paths = ProjectPaths.from_config(config)
    train_stat = paths.train.stat()
    report = {
        "phase": 6,
        "cv_type": settings["type"],
        "aggregation": settings.get("aggregation", "equal_weight"),
        "random_split_used": False,
        "label_column": label,
        "label_horizon_trade_days": splitter.label_horizon_trade_days,
        "purge_trade_days": splitter.purge_trade_days,
        "purge_covers_label_horizon": splitter.purge_trade_days >= splitter.label_horizon_trade_days,
        "calendar": {
            "trade_dates": len(dates),
            "start": dates[0],
            "end": dates[-1],
            "sha256": _calendar_sha256(dates),
            "rows_per_date_min": int(daily_rows.min()),
            "rows_per_date_max": int(daily_rows.max()),
        },
        "training_data": {
            "rows": int(len(frame)),
            "stocks": int(frame["ts_code"].nunique()),
            "key_unique": True,
            "raw_path": str(paths.train),
            "raw_size": train_stat.st_size,
            "raw_mtime_ns": train_stat.st_mtime_ns,
        },
        "folds": fold_reports,
        "all_folds_valid": all(
            item["train_validation_disjoint"]
            and item["purge_excluded_from_train"]
            and item["chronology_valid"]
            for item in fold_reports
        ),
        "preprocessor_fit_scope": "training_rows_within_each_fold_only",
        "calendar_output": str(calendar_path),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_tmp = calendar_path.with_suffix(".tmp.parquet")
    audit_tmp = audit_path.with_suffix(".tmp.json")
    calendar_frame.to_parquet(calendar_tmp, engine="pyarrow", compression="zstd", index=False)
    audit_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    calendar_tmp.replace(calendar_path)
    audit_tmp.replace(audit_path)
    return report
