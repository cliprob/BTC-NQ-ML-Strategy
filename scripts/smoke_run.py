from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_synchronized_sample
from src.features import add_market_features, feature_columns
from src.labels import generate_labeled_trades
from src.validation import chronological_split, run_walk_forward_classifier


def main() -> None:
    market = add_market_features(load_synchronized_sample())
    trades = generate_labeled_trades(market, max_holding_minutes=60)
    if len(trades) < 80:
        raise RuntimeError(f"Sample produced too few trades for smoke run: {len(trades)}")

    features = feature_columns(trades.columns)
    split = chronological_split(trades, train_frac=0.7, validation_frac=0.0)
    result = run_walk_forward_classifier(
        split.train,
        features,
        model_factory=lambda: LogisticRegression(
            solver="liblinear",
            C=0.1,
            class_weight="balanced",
            random_state=42,
        ),
        train_window=40,
        test_window=20,
        threshold_grid=[0.50, 0.55, 0.60],
    )

    output_dir = Path("outputs/tables")
    figure_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result["metrics"]]).to_csv(output_dir / "sample_smoke_metrics.csv", index=False)
    selected = result["selected_trades"]
    selected.to_csv(output_dir / "sample_selected_trades.csv", index=False)

    if not selected.empty:
        equity = selected["pnl"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(equity.to_numpy(), color="#1f4e79", linewidth=1.8)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title("Sample Walk-Forward Equity Curve")
        ax.set_xlabel("Selected trade sequence")
        ax.set_ylabel("Cumulative NQ points")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "sample_walk_forward_equity.png", dpi=160)
        plt.close(fig)

    print("Smoke run completed")
    print(result["metrics"])


if __name__ == "__main__":
    main()
