from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "rsi",
    "atr_pct",
    "macd_hist",
    "bb_width",
    "dist_from_ema",
    "rolling_corr_60",
    "btc_ret_1m",
    "nq_ret_1m",
    "btc_nq_alignment",
    "hour_sin",
    "hour_cos",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_market_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time features from current and historical bars only."""

    df = market_data.copy()
    df["btc_ret_1m"] = np.log(df["close_btc"] / df["close_btc"].shift(1))
    df["nq_ret_1m"] = np.log(df["close_nq"] / df["close_nq"].shift(1))

    df["rsi"] = _rsi(df["close_nq"])

    high_low = df["high_nq"] - df["low_nq"]
    high_close = (df["high_nq"] - df["close_nq"].shift(1)).abs()
    low_close = (df["low_nq"] - df["close_nq"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["close_nq"]

    ema_12 = df["close_nq"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close_nq"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    sma_20 = df["close_nq"].rolling(20).mean()
    bb_std = df["close_nq"].rolling(20).std()
    df["bb_width"] = (4 * bb_std) / sma_20

    ema_50 = df["close_nq"].ewm(span=50, adjust=False).mean()
    df["dist_from_ema"] = (df["close_nq"] - ema_50) / ema_50

    df["rolling_corr_60"] = df["btc_ret_1m"].rolling(60).corr(df["nq_ret_1m"])

    btc_dir = np.sign(df["close_btc"] - df["open_btc"])
    nq_dir = np.sign(df["close_nq"] - df["open_nq"])
    df["btc_nq_alignment"] = (btc_dir == nq_dir).astype(int)

    hour = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df


def feature_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return model features available in the provided columns."""

    available = set(columns)
    return [col for col in MODEL_FEATURES if col in available]
