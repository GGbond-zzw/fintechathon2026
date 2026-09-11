"""Causal score smoothing and exact-size hysteresis for turnover control."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["ts_code", "trade_date"]


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {*KEYS, "pred", "flag_limit_up"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing turnover-control columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise ValueError("Turnover-control keys must be unique")
    predictions = frame["pred"].to_numpy(dtype="float64")
    if not np.isfinite(predictions).all():
        raise ValueError("Input predictions must all be finite")


def daily_percentile_rank(frame: pd.DataFrame) -> np.ndarray:
    """Return same-date percentile ranks aligned to the input row order."""
    _validate_frame(frame)
    ranked = frame.groupby("trade_date", sort=False)["pred"].rank(method="average", pct=True)
    values = ranked.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("Daily prediction ranks must all be finite")
    return values


def causal_rank_ema(
    frame: pd.DataFrame,
    alpha: float,
    *,
    prior_state: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rank predictions by date, then smooth within stock using only current/past rows.

    When no prior state is supplied, each stock starts from its first current rank.
    A prior-state mapping lets later inference continue from an earlier date without
    changing the no-lookahead recurrence.
    """
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    ranks = daily_percentile_rank(frame)
    work = frame[KEYS].copy()
    work["_position"] = np.arange(len(work), dtype="int64")
    work["_rank"] = ranks
    work = work.sort_values(KEYS, kind="stable").reset_index(drop=True)

    if prior_state is None:
        work["_smooth"] = work.groupby("ts_code", sort=False)["_rank"].transform(
            lambda values: values.ewm(alpha=float(alpha), adjust=False).mean()
        )
    else:
        smoothed = np.empty(len(work), dtype="float64")
        for code, positions in work.groupby("ts_code", sort=False).indices.items():
            position_array = np.asarray(positions, dtype="int64")
            values = work.loc[position_array, "_rank"].to_numpy(dtype="float64")
            previous = float(prior_state[str(code)]) if str(code) in prior_state else float(values[0])
            for offset, value in enumerate(values):
                if offset == 0 and str(code) not in prior_state:
                    previous = float(value)
                else:
                    previous = float(alpha) * float(value) + (1.0 - float(alpha)) * previous
                smoothed[position_array[offset]] = previous
        work["_smooth"] = smoothed

    state_rows = work.groupby("ts_code", sort=False).tail(1)
    state = dict(zip(state_rows["ts_code"].astype(str), state_rows["_smooth"].astype(float)))
    output = np.empty(len(work), dtype="float64")
    output[work["_position"].to_numpy(dtype="int64")] = work["_smooth"].to_numpy(dtype="float64")
    if not np.isfinite(output).all():
        raise ValueError("Smoothed predictions must all be finite")
    return output, state


def hysteresis_rank_scores(
    frame: pd.DataFrame,
    smoothed: np.ndarray,
    exit_fraction: float,
    *,
    top_fraction_denominator: int = 10,
    minimum_pool_size: int = 100,
    prior_top_set: Set[str] | None = None,
) -> tuple[np.ndarray, set[str]]:
    """Map a buffered exact-size set back to unique deterministic daily scores.

    Previous members survive while they remain inside ``exit_fraction`` of the
    current eligible ranking; empty slots are filled by current score. The score
    multiset is only permuted among non-limit-up rows, so limit-up rows keep their
    original cross-sectional position. A sub-threshold day resets membership, as
    the official turnover evaluator does.
    """
    _validate_frame(frame)
    values = np.asarray(smoothed, dtype="float64")
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("smoothed must be a finite vector aligned to frame")
    if top_fraction_denominator < 1 or minimum_pool_size < 1:
        raise ValueError("Pool constants must be positive")
    top_fraction = 1.0 / float(top_fraction_denominator)
    if not top_fraction <= float(exit_fraction) <= 1.0:
        raise ValueError("exit_fraction must be between the top fraction and 1")

    work = frame[KEYS + ["flag_limit_up"]].copy()
    work["_position"] = np.arange(len(work), dtype="int64")
    work["_smooth"] = values
    work = work.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    adjusted = np.empty(len(work), dtype="float64")
    previous = {str(code) for code in (prior_top_set or set())}

    for _, positions in work.groupby("trade_date", sort=True).indices.items():
        positions = np.asarray(positions, dtype="int64")
        scores = work.loc[positions, "_smooth"].to_numpy(dtype="float64")
        codes = work.loc[positions, "ts_code"].astype(str).to_numpy()
        flags = work.loc[positions, "flag_limit_up"].to_numpy()
        order = np.lexsort((codes, -scores))
        ordinal = np.empty(len(positions), dtype="float64")
        ordinal[order] = (len(positions) - np.arange(len(positions), dtype="float64")) / len(positions)
        adjusted[positions] = ordinal

        eligible_order = order[flags[order] == 0]
        eligible_count = len(eligible_order)
        if eligible_count < minimum_pool_size:
            previous = set()
            continue
        top_n = max(eligible_count // top_fraction_denominator, 1)
        exit_n = max(top_n, int(np.floor(eligible_count * float(exit_fraction))))
        buffer_order = eligible_order[:exit_n]
        retained = [local for local in buffer_order if codes[local] in previous][:top_n]
        retained_set = set(retained)
        selected = retained + [
            local for local in eligible_order if local not in retained_set
        ][:top_n - len(retained)]
        selected_set = set(selected)
        desired_order = selected + [local for local in eligible_order if local not in selected_set]

        eligible_positions = positions[eligible_order]
        available_scores = np.sort(adjusted[eligible_positions])[::-1]
        adjusted[positions[np.asarray(desired_order, dtype="int64")]] = available_scores
        previous = {str(codes[local]) for local in selected}

    output = np.empty(len(work), dtype="float64")
    output[work["_position"].to_numpy(dtype="int64")] = adjusted
    output = output.astype("float32")
    if not np.isfinite(output).all():
        raise ValueError("Turnover-controlled scores must all be finite")
    return output, previous
