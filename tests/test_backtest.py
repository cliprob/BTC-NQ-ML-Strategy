import pandas as pd

from src.backtest import max_drawdown, select_non_overlapping_trades, summarize_trades


def test_select_non_overlapping_trades_skips_active_position():
    trades = pd.DataFrame(
        {
            "entry_idx": [1, 2, 5, 8],
            "exit_idx": [4, 3, 7, 9],
            "pnl": [10.0, 99.0, -2.0, 4.0],
            "signal": [True, True, True, False],
        }
    )

    selected = select_non_overlapping_trades(trades)

    assert selected["entry_idx"].tolist() == [1, 5]
    assert selected["pnl"].tolist() == [10.0, -2.0]


def test_summarize_trades_is_stable_for_basic_sample():
    trades = pd.DataFrame({"pnl": [2.0, -1.0, 3.0, -2.0]})

    metrics = summarize_trades(trades)

    assert metrics["trade_count"] == 4
    assert metrics["total_pnl"] == 2.0
    assert metrics["win_rate"] == 0.5
    assert max_drawdown(pd.Series([2.0, 1.0, 4.0, 2.0])) == -2.0
