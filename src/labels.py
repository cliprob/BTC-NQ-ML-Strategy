from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_columns


def _trade_outcome(
    entry_idx: int,
    market_data: pd.DataFrame,
    entry_direction: float,
    max_holding_minutes: int,
    safety_close_hour: int,
    safety_close_minute: int,
) -> tuple[float, int]:
    entry_price = float(market_data.iloc[entry_idx]["open_nq"])
    subset = market_data.iloc[entry_idx : entry_idx + max_holding_minutes]

    for current_idx, row in subset.iterrows():
        current_time = current_idx
        is_eod = (
            current_time.hour == safety_close_hour
            and current_time.minute >= safety_close_minute
        ) or current_time.hour > safety_close_hour

        curr_dir_btc = np.sign(row["close_btc"] - row["open_btc"])
        curr_dir_nq = np.sign(row["close_nq"] - row["open_nq"])
        is_divergence = curr_dir_btc != curr_dir_nq
        is_reversal = curr_dir_btc == -entry_direction and curr_dir_nq == -entry_direction

        if is_eod or is_divergence or is_reversal:
            exit_pos = market_data.index.get_loc(current_idx)
            pnl = (float(row["close_nq"]) - entry_price) * entry_direction
            return pnl, int(exit_pos)

    if len(subset) == 0:
        return 0.0, entry_idx

    last_row = subset.iloc[-1]
    exit_idx = market_data.index.get_loc(subset.index[-1])
    pnl = (float(last_row["close_nq"]) - entry_price) * entry_direction
    return pnl, int(exit_idx)


def generate_labeled_trades(
    market_data: pd.DataFrame,
    max_holding_minutes: int = 600,
    safety_close_hour: int = 21,
    safety_close_minute: int = 55,
) -> pd.DataFrame:
    """Generate baseline signal candidates and their future outcomes.

    Features are copied from the signal candle at time `t`. The trade enters
    on the next bar. Future information is used only to label the outcome.
    """

    features = feature_columns(market_data.columns)
    if not features:
        raise ValueError("No model feature columns found. Run add_market_features first.")

    records: list[dict[str, object]] = []
    clean = market_data.dropna(subset=features).copy()
    if len(clean) <= max_holding_minutes + 2:
        return pd.DataFrame(columns=["entry_idx", "exit_idx", "entry_time", "pnl", "target", *features])

    for signal_pos in range(1, len(clean) - max_holding_minutes - 1):
        row = clean.iloc[signal_pos]
        dir_btc = np.sign(row["close_btc"] - row["open_btc"])
        dir_nq = np.sign(row["close_nq"] - row["open_nq"])

        if dir_btc == 0 or dir_btc != dir_nq:
            continue

        entry_pos = signal_pos + 1
        pnl, exit_pos = _trade_outcome(
            entry_pos,
            clean,
            dir_btc,
            max_holding_minutes,
            safety_close_hour,
            safety_close_minute,
        )

        record = {
            "entry_idx": int(entry_pos),
            "exit_idx": int(exit_pos),
            "entry_time": clean.index[entry_pos],
            "pnl": float(pnl),
            "target": int(pnl > 0),
        }
        record.update({feature: float(row[feature]) for feature in features})
        records.append(record)

    return pd.DataFrame.from_records(records)
