from __future__ import annotations

import numpy as np
import pandas as pd


def select_non_overlapping_trades(trades: pd.DataFrame, signal_col: str = "signal") -> pd.DataFrame:
    """Keep signalled trades whose entry is after the prior selected exit."""

    required = {"entry_idx", "exit_idx", signal_col}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Missing required trade columns: {sorted(missing)}")

    selected_rows = []
    last_exit = -1
    for _, row in trades.sort_values("entry_idx").iterrows():
        if not bool(row[signal_col]):
            continue
        if int(row["entry_idx"]) > last_exit:
            selected_rows.append(row)
            last_exit = int(row["exit_idx"])
    if not selected_rows:
        return trades.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def equity_curve(pnl: pd.Series | np.ndarray) -> pd.Series:
    values = pd.Series(pnl, dtype="float64")
    return values.cumsum()


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    curve = pd.Series(equity, dtype="float64")
    if curve.empty:
        return 0.0
    running_max = curve.cummax()
    return float((curve - running_max).min())


def summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
        }

    pnl = trades["pnl"].astype(float)
    curve = equity_curve(pnl)
    std = float(pnl.std(ddof=1))
    sharpe_like = float(pnl.mean() / std) if std > 0 else 0.0
    return {
        "trade_count": int(len(trades)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "max_drawdown": max_drawdown(curve),
        "sharpe_like": sharpe_like,
    }
