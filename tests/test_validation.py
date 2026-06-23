import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.validation import (
    chronological_split,
    iter_walk_forward_windows,
    run_walk_forward_classifier,
    run_train_then_holdout_classifier,
    validate_no_leakage_columns,
)


def make_trade_frame(rows=120):
    idx = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    x1 = np.sin(np.arange(rows) / 4)
    x2 = np.cos(np.arange(rows) / 5)
    target = (x1 + x2 > 0).astype(int)
    pnl = np.where(target == 1, 1.5, -1.0)
    return pd.DataFrame(
        {
            "entry_time": idx,
            "entry_idx": np.arange(rows),
            "exit_idx": np.arange(rows) + 1,
            "feature_a": x1,
            "feature_b": x2,
            "target": target,
            "pnl": pnl,
        }
    )


def test_validate_no_leakage_columns_rejects_outcome_names():
    with pytest.raises(ValueError):
        validate_no_leakage_columns(["feature_a", "future_return", "pnl"])


def test_chronological_split_keeps_time_order():
    split = chronological_split(make_trade_frame(), train_frac=0.6, validation_frac=0.2)

    assert split.train["entry_time"].max() < split.validation["entry_time"].min()
    assert split.validation["entry_time"].max() < split.test["entry_time"].min()


def test_walk_forward_windows_train_before_test():
    frame = make_trade_frame()
    train, test = next(iter_walk_forward_windows(frame, train_window=40, test_window=10))

    assert train["entry_idx"].max() < test["entry_idx"].min()


def test_walk_forward_scaler_is_fit_on_train_window_only():
    frame = make_trade_frame()
    feature_cols = ["feature_a", "feature_b"]

    result = run_walk_forward_classifier(
        frame,
        feature_cols,
        model_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
        train_window=40,
        test_window=20,
        threshold_grid=[0.5],
    )

    expected_first_mean = frame.iloc[:40][feature_cols].mean().to_numpy()
    assert np.allclose(result["scaler_means"][0], expected_first_mean)
    assert result["metrics"]["trade_count"] > 0


def test_train_then_holdout_uses_train_scaler_and_holdout_scores():
    frame = make_trade_frame()
    train = frame.iloc[:80].copy()
    holdout = frame.iloc[80:].copy()
    feature_cols = ["feature_a", "feature_b"]

    result = run_train_then_holdout_classifier(
        train,
        holdout,
        feature_cols,
        model_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
        threshold_grid=[0.5],
    )

    expected_mean = train[feature_cols].mean().to_numpy()
    assert np.allclose(result["scaler_mean"], expected_mean)
    assert set(result["selected_trades"]["entry_idx"]).issubset(set(holdout["entry_idx"]))
