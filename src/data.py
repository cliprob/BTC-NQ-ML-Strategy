from __future__ import annotations

from pathlib import Path

import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlcv(path: str | Path, prefix: str) -> pd.DataFrame:
    """Load one OHLCV file and prefix market columns.

    The loader accepts either Unix-second timestamps or parseable datetime
    strings in a column named `time`.
    """

    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]

    missing = {"time", *OHLCV_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    time = df["time"]
    if pd.api.types.is_numeric_dtype(time):
        timestamp = pd.to_datetime(time.astype("int64"), unit="s", utc=True)
    else:
        timestamp = pd.to_datetime(time, utc=True)

    clean = df.loc[:, OHLCV_COLUMNS].copy()
    clean.insert(0, "timestamp", timestamp)
    clean = clean.dropna(subset=["timestamp", *OHLCV_COLUMNS])
    clean = clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    clean = clean.set_index("timestamp")
    clean = clean.rename(columns={col: f"{col}_{prefix}" for col in OHLCV_COLUMNS})
    return clean


def synchronize_markets(btc: pd.DataFrame, nq: pd.DataFrame) -> pd.DataFrame:
    """Inner-join BTC and NQ bars on their observed timestamp calendar."""

    merged = btc.join(nq, how="inner").sort_index()
    if merged.empty:
        raise ValueError("No overlapping timestamps between BTC and NQ data")
    return merged


def load_synchronized_sample(data_dir: str | Path = "data/sample") -> pd.DataFrame:
    """Load the repository sample dataset."""

    data_dir = Path(data_dir)
    btc = load_ohlcv(data_dir / "btc_1m_sample.csv", "btc")
    nq = load_ohlcv(data_dir / "nq_1m_sample.csv", "nq")
    return synchronize_markets(btc, nq)
