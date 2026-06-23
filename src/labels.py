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

    clean = market_data.dropna(subset=features).copy()
    if len(clean) <= max_holding_minutes + 2:
        return pd.DataFrame(
            columns=[
                "entry_idx",
                "exit_idx",
                "signal_time",
                "entry_time",
                "pnl",
                "target",
                *features,
            ]
        )

    index = clean.index
    open_btc = clean["open_btc"].to_numpy()
    close_btc = clean["close_btc"].to_numpy()
    open_nq = clean["open_nq"].to_numpy()
    close_nq = clean["close_nq"].to_numpy()
    hours = index.hour.to_numpy()
    minutes = index.minute.to_numpy()
    feature_arrays = {feature: clean[feature].to_numpy() for feature in features}

    records: list[dict[str, object]] = []
    last_signal_pos = len(clean) - max_holding_minutes - 1

    for signal_pos in range(1, last_signal_pos):
        dir_btc = np.sign(close_btc[signal_pos] - open_btc[signal_pos])
        dir_nq = np.sign(close_nq[signal_pos] - open_nq[signal_pos])

        if dir_btc == 0 or dir_btc != dir_nq:
            continue

        entry_pos = signal_pos + 1
        entry_price = float(open_nq[entry_pos])
        exit_pos = min(entry_pos + max_holding_minutes - 1, len(clean) - 1)

        for current_pos in range(entry_pos, min(entry_pos + max_holding_minutes, len(clean))):
            is_eod = (
                hours[current_pos] == safety_close_hour
                and minutes[current_pos] >= safety_close_minute
            ) or hours[current_pos] > safety_close_hour

            curr_dir_btc = np.sign(close_btc[current_pos] - open_btc[current_pos])
            curr_dir_nq = np.sign(close_nq[current_pos] - open_nq[current_pos])
            is_divergence = curr_dir_btc != curr_dir_nq
            is_reversal = curr_dir_btc == -dir_btc and curr_dir_nq == -dir_btc

            if is_eod or is_divergence or is_reversal:
                exit_pos = current_pos
                break

        pnl = (float(close_nq[exit_pos]) - entry_price) * dir_btc

        record = {
            "entry_idx": int(entry_pos),
            "exit_idx": int(exit_pos),
            "signal_time": index[signal_pos],
            "entry_time": index[entry_pos],
            "pnl": float(pnl),
            "target": int(pnl > 0),
        }
        record.update({feature: float(feature_arrays[feature][signal_pos]) for feature in features})
        records.append(record)

    return pd.DataFrame.from_records(records)
