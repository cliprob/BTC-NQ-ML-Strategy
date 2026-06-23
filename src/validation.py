from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler

from .backtest import select_non_overlapping_trades, summarize_trades


LEAKAGE_TOKENS = (
    "target",
    "pnl",
    "exit",
    "future",
    "forward",
    "fwd",
    "t_plus",
    "lead",
)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_no_leakage_columns(feature_cols: Iterable[str]) -> None:
    offenders = [
        col
        for col in feature_cols
        if any(token in col.lower() for token in LEAKAGE_TOKENS)
    ]
    if offenders:
        raise ValueError(f"Potential leakage columns in feature set: {offenders}")


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    validation_frac: float = 0.2,
    time_col: str = "entry_time",
) -> ChronologicalSplit:
    if not 0 < train_frac < 1 or not 0 <= validation_frac < 1:
        raise ValueError("Split fractions must be within [0, 1)")
    if train_frac + validation_frac >= 1:
        raise ValueError("Train plus validation fractions must leave a test set")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    validation_end = int(n * (train_frac + validation_frac))
    return ChronologicalSplit(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )


def iter_walk_forward_windows(
    df: pd.DataFrame,
    train_window: int,
    test_window: int,
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
    if train_window <= 0 or test_window <= 0:
        raise ValueError("Window sizes must be positive")
    for start in range(0, len(df) - train_window - test_window + 1, test_window):
        train = df.iloc[start : start + train_window].copy()
        test = df.iloc[start + train_window : start + train_window + test_window].copy()
        yield train, test


def choose_threshold_on_train(
    train_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold_grid: Iterable[float],
    min_trades: int = 5,
) -> float:
    best_threshold = 0.5
    best_pnl = -np.inf
    scored_any = False

    for threshold in threshold_grid:
        candidate = train_df.copy()
        candidate["signal"] = probabilities >= threshold
        selected = select_non_overlapping_trades(candidate)
        if len(selected) < min_trades:
            continue
        pnl = float(selected["pnl"].sum())
        scored_any = True
        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = float(threshold)

    if scored_any:
        return best_threshold
    return 0.5


def run_walk_forward_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], ClassifierMixin],
    train_window: int,
    test_window: int,
    threshold_grid: Iterable[float] | None = None,
    min_trades: int = 5,
) -> dict[str, object]:
    validate_no_leakage_columns(feature_cols)
    if threshold_grid is None:
        threshold_grid = np.arange(0.50, 0.66, 0.01)

    selected_frames: list[pd.DataFrame] = []
    threshold_history: list[float] = []
    scaler_means: list[np.ndarray] = []

    for train_df, test_df in iter_walk_forward_windows(df, train_window, test_window):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_cols])
        x_test = scaler.transform(test_df[feature_cols])
        scaler_means.append(scaler.mean_.copy())

        model = model_factory()
        model.fit(x_train, train_df["target"])

        if hasattr(model, "predict_proba"):
            train_probs = model.predict_proba(x_train)[:, 1]
            test_probs = model.predict_proba(x_test)[:, 1]
        else:
            train_scores = model.decision_function(x_train)
            test_scores = model.decision_function(x_test)
            train_probs = 1 / (1 + np.exp(-train_scores))
            test_probs = 1 / (1 + np.exp(-test_scores))

        threshold = choose_threshold_on_train(
            train_df,
            train_probs,
            threshold_grid,
            min_trades=min_trades,
        )
        threshold_history.append(threshold)

        scored = test_df.copy()
        scored["probability"] = test_probs
        scored["signal"] = scored["probability"] >= threshold
        selected_frames.append(select_non_overlapping_trades(scored))

    selected = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else df.iloc[0:0].copy()
    )
    return {
        "selected_trades": selected,
        "metrics": summarize_trades(selected),
        "threshold_history": threshold_history,
        "scaler_means": scaler_means,
    }


def run_train_then_holdout_classifier(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], ClassifierMixin],
    threshold_grid: Iterable[float] | None = None,
    min_trades: int = 5,
) -> dict[str, object]:
    """Fit on the historical set, tune threshold there, then score holdout."""

    validate_no_leakage_columns(feature_cols)
    if threshold_grid is None:
        threshold_grid = np.arange(0.50, 0.66, 0.01)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_holdout = scaler.transform(holdout_df[feature_cols])

    model = model_factory()
    model.fit(x_train, train_df["target"])

    if hasattr(model, "predict_proba"):
        train_probs = model.predict_proba(x_train)[:, 1]
        holdout_probs = model.predict_proba(x_holdout)[:, 1]
    else:
        train_scores = model.decision_function(x_train)
        holdout_scores = model.decision_function(x_holdout)
        train_probs = 1 / (1 + np.exp(-train_scores))
        holdout_probs = 1 / (1 + np.exp(-holdout_scores))

    threshold = choose_threshold_on_train(
        train_df,
        train_probs,
        threshold_grid,
        min_trades=min_trades,
    )

    scored = holdout_df.copy()
    scored["probability"] = holdout_probs
    scored["signal"] = scored["probability"] >= threshold
    selected = select_non_overlapping_trades(scored)

    return {
        "selected_trades": selected,
        "metrics": summarize_trades(selected),
        "threshold": threshold,
        "scaler_mean": scaler.mean_.copy(),
    }
