import numpy as np
import pandas as pd

from src.features import add_market_features, feature_columns
from src.labels import generate_labeled_trades


def make_market_frame(rows=220):
    idx = pd.date_range("2024-01-02 14:00", periods=rows, freq="min", tz="UTC")
    base_btc = 43000 + np.cumsum(np.sin(np.arange(rows) / 8) * 3 + 0.5)
    base_nq = 17000 + np.cumsum(np.sin(np.arange(rows) / 9) * 1.5 + 0.2)
    return pd.DataFrame(
        {
            "open_btc": base_btc,
            "high_btc": base_btc + 8,
            "low_btc": base_btc - 8,
            "close_btc": base_btc + np.sin(np.arange(rows) / 5),
            "volume_btc": np.full(rows, 100.0),
            "open_nq": base_nq,
            "high_nq": base_nq + 4,
            "low_nq": base_nq - 4,
            "close_nq": base_nq + np.sin(np.arange(rows) / 6),
            "volume_nq": np.full(rows, 50.0),
        },
        index=idx,
    )


def test_feature_columns_are_curated_and_exclude_outcomes():
    market = add_market_features(make_market_frame())
    cols = feature_columns(market.columns)

    assert "pnl" not in cols
    assert "target" not in cols
    assert "exit_idx" not in cols
    assert "rsi" in cols


def test_generate_labeled_trades_uses_features_and_outcomes_separately():
    market = add_market_features(make_market_frame())
    trades = generate_labeled_trades(market, max_holding_minutes=30)

    assert {"entry_idx", "exit_idx", "entry_time", "pnl", "target"}.issubset(trades.columns)
    assert "rsi" in trades.columns
    assert trades["entry_idx"].lt(trades["exit_idx"] + 1).all()
