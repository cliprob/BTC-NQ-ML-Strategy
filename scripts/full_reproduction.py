from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import equity_curve, select_non_overlapping_trades, summarize_trades
from src.data import load_ohlcv, synchronize_markets
from src.features import add_market_features, feature_columns
from src.labels import generate_labeled_trades
from src.validation import (
    chronological_split,
    iter_walk_forward_windows,
    run_train_then_holdout_classifier,
    run_walk_forward_classifier,
    validate_no_leakage_columns,
)


DEFAULT_BTC_PATH = ROOT / "2_Phase" / "working" / "ENGINE_BINANCE_BTC_2024-01-01_to_2025_12_05.csv"
DEFAULT_NQ_PATH = ROOT / "2_Phase" / "working" / "ENGINE_Nasdaq_1_minute_data_2024-01-01_to_2025-12-05.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "full_reproduction" / "latest"
LEGACY_BASE_FEATURES = ["rsi", "atr_pct", "macd_hist", "bb_width", "dist_from_ema"]
THRESHOLD_GRID = np.arange(0.50, 0.65, 0.01)

RF_CONFIGS = [
    {"name": "Scalper_Fast", "win": 2000, "step": 500, "depth": 4, "leaf": 100},
    {"name": "Investor_Long", "win": 8000, "step": 2000, "depth": 8, "leaf": 50},
    {"name": "Safe_Conservative", "win": 4000, "step": 1000, "depth": 6, "leaf": 200},
    {"name": "Genius_Complex", "win": 5000, "step": 1000, "depth": 12, "leaf": 20},
    {"name": "Balanced_Std", "win": 3500, "step": 800, "depth": 8, "leaf": 40},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local full-data BTC/NQ reproduction workflow.",
    )
    parser.add_argument("--btc-path", type=Path, default=DEFAULT_BTC_PATH)
    parser.add_argument("--nq-path", type=Path, default=DEFAULT_NQ_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default="lr,rf,svm", help="Comma-separated subset: lr,rf,svm")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional first-N synchronized rows for a quick test run")
    parser.add_argument("--feature-set", choices=["legacy", "public"], default="legacy")
    parser.add_argument("--max-holding-minutes", type=int, default=600)
    parser.add_argument("--safety-close-hour", type=int, default=21)
    parser.add_argument("--safety-close-minute", type=int, default=55)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--lr-train-window", type=int, default=3000)
    parser.add_argument("--lr-test-window", type=int, default=1000)
    parser.add_argument("--min-threshold-trades", type=int, default=10)
    parser.add_argument("--save-candidates", action="store_true", help="Write all labeled candidates to CSV")
    return parser.parse_args()


def parse_models(raw: str) -> set[str]:
    models = {item.strip().lower() for item in raw.split(",") if item.strip()}
    allowed = {"lr", "rf", "svm"}
    unknown = models - allowed
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}. Allowed: {sorted(allowed)}")
    return models


def load_full_market_data(args: argparse.Namespace) -> pd.DataFrame:
    if not args.btc_path.exists():
        raise FileNotFoundError(f"BTC file not found: {args.btc_path}")
    if not args.nq_path.exists():
        raise FileNotFoundError(f"NQ file not found: {args.nq_path}")

    btc = load_ohlcv(args.btc_path, "btc")
    nq = load_ohlcv(args.nq_path, "nq")
    market = synchronize_markets(btc, nq)
    if args.row_limit is not None:
        market = market.iloc[: args.row_limit].copy()
    return market


def add_legacy_hour_features(trades: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = trades.copy()
    hour_source = pd.to_datetime(prepared["signal_time"], utc=True)
    hour_dummies = pd.get_dummies(hour_source.dt.hour, prefix="hour", dtype=int)
    hour_dummies.index = prepared.index
    prepared = pd.concat([prepared, hour_dummies], axis=1)

    features = [
        col for col in LEGACY_BASE_FEATURES if col in prepared.columns
    ] + list(hour_dummies.columns)
    validate_no_leakage_columns(features)
    return prepared, features


def prepare_labeled_trades(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    market = load_full_market_data(args)
    featured = add_market_features(market)
    trades = generate_labeled_trades(
        featured,
        max_holding_minutes=args.max_holding_minutes,
        safety_close_hour=args.safety_close_hour,
        safety_close_minute=args.safety_close_minute,
    )
    if trades.empty:
        raise RuntimeError("No labeled trade candidates were generated")

    if args.feature_set == "legacy":
        trades, features = add_legacy_hour_features(trades)
    else:
        features = feature_columns(trades.columns)
        validate_no_leakage_columns(features)

    return trades, features, market


def baseline_result(trades: pd.DataFrame) -> dict[str, object]:
    scored = trades.copy()
    scored["signal"] = True
    selected = select_non_overlapping_trades(scored)
    return {"selected_trades": selected, "metrics": summarize_trades(selected)}


def fixed_threshold_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], object],
    train_window: int,
    test_window: int,
    threshold: float,
    score_kind: str,
    scale: bool,
) -> dict[str, object]:
    validate_no_leakage_columns(feature_cols)
    selected_frames = []

    for train_df, test_df in iter_walk_forward_windows(df, train_window, test_window):
        model = model_factory()
        if scale:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_df[feature_cols])
            x_test = scaler.transform(test_df[feature_cols])
        else:
            x_train = train_df[feature_cols]
            x_test = test_df[feature_cols]

        model.fit(x_train, train_df["target"])
        scores = score_model(model, x_test, score_kind)

        scored = test_df.copy()
        scored["score"] = scores
        scored["signal"] = scored["score"] >= threshold
        selected_frames.append(select_non_overlapping_trades(scored))

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else df.iloc[0:0].copy()
    return {"selected_trades": selected, "metrics": summarize_trades(selected)}


def fixed_threshold_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], object],
    threshold: float,
    score_kind: str,
    scale: bool,
) -> dict[str, object]:
    validate_no_leakage_columns(feature_cols)
    model = model_factory()
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_cols])
        x_holdout = scaler.transform(holdout_df[feature_cols])
    else:
        x_train = train_df[feature_cols]
        x_holdout = holdout_df[feature_cols]

    model.fit(x_train, train_df["target"])
    scores = score_model(model, x_holdout, score_kind)

    scored = holdout_df.copy()
    scored["score"] = scores
    scored["signal"] = scored["score"] >= threshold
    selected = select_non_overlapping_trades(scored)
    return {"selected_trades": selected, "metrics": summarize_trades(selected)}


def score_model(model: object, x_data: pd.DataFrame | np.ndarray, score_kind: str) -> np.ndarray:
    if score_kind == "predict_proba":
        return model.predict_proba(x_data)[:, 1]
    if score_kind == "decision_function":
        return model.decision_function(x_data)
    raise ValueError(f"Unknown score kind: {score_kind}")


def run_random_forest_battle(train_val: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], pd.DataFrame, dict[str, int | str]]:
    rows = []
    best_result: dict[str, object] | None = None
    best_config: dict[str, int | str] | None = None
    best_score = -np.inf

    for config in RF_CONFIGS:
        result = fixed_threshold_walk_forward(
            train_val,
            feature_cols,
            model_factory=lambda cfg=config: RandomForestClassifier(
                n_estimators=50,
                max_depth=int(cfg["depth"]),
                min_samples_leaf=int(cfg["leaf"]),
                random_state=42,
                n_jobs=-1,
            ),
            train_window=int(config["win"]),
            test_window=int(config["step"]),
            threshold=0.52,
            score_kind="predict_proba",
            scale=False,
        )
        metrics = result["metrics"]
        rows.append({"config": config["name"], **config, **metrics})
        score = float(metrics["sharpe_like"])
        if score > best_score:
            best_score = score
            best_result = result
            best_config = config

    if best_result is None or best_config is None:
        raise RuntimeError("Random Forest battle did not produce any result")
    return best_result, pd.DataFrame(rows), best_config


def run_svm_walk_forward(train_val: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    return fixed_threshold_walk_forward(
        train_val,
        feature_cols,
        model_factory=lambda: LinearSVC(
            C=0.01,
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            class_weight="balanced",
            random_state=42,
            max_iter=2000,
        ),
        train_window=4000,
        test_window=1000,
        threshold=0.0,
        score_kind="decision_function",
        scale=True,
    )


def write_outputs(
    args: argparse.Namespace,
    trades: pd.DataFrame,
    features: list[str],
    market: pd.DataFrame,
    results: dict[str, dict[str, object]],
    extra_tables: dict[str, pd.DataFrame],
    config: dict[str, object],
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for name, result in results.items():
        metrics_rows.append({"name": name, **result["metrics"]})
        selected = result["selected_trades"].copy()
        selected.to_csv(output_dir / f"selected_{name}.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    for name, table in extra_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    if args.save_candidates:
        trades.to_csv(output_dir / "labeled_candidates.csv", index=False)

    run_info = {
        **config,
        "market_rows": int(len(market)),
        "candidate_rows": int(len(trades)),
        "feature_columns": features,
        "results": {
            name: result["metrics"]
            for name, result in results.items()
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(to_builtin(run_info), indent=2),
        encoding="utf-8",
    )
    write_equity_plot(output_dir, results)


def write_equity_plot(output_dir: Path, results: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for name, result in results.items():
        selected = result["selected_trades"]
        if selected.empty:
            continue
        curve = equity_curve(selected["pnl"])
        ax.plot(curve.to_numpy(), linewidth=1.6, label=name)
        plotted = True

    if plotted:
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title("Full Reproduction Equity Curves")
        ax.set_xlabel("Selected trade sequence")
        ax.set_ylabel("Cumulative NQ points")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "equity_curves.png", dpi=160)
    plt.close(fig)


def to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    trades, features, market = prepare_labeled_trades(args)
    split = chronological_split(trades, train_frac=args.train_frac, validation_frac=0.0)
    train_val = split.train
    holdout = split.test

    results: dict[str, dict[str, object]] = {}
    extra_tables: dict[str, pd.DataFrame] = {}

    results["baseline_train_val"] = baseline_result(train_val)
    results["baseline_holdout"] = baseline_result(holdout)

    if "lr" in models:
        lr_factory = lambda: LogisticRegression(
            solver="liblinear",
            C=0.1,
            class_weight="balanced",
            random_state=42,
        )
        lr_walk = run_walk_forward_classifier(
            train_val,
            features,
            model_factory=lr_factory,
            train_window=args.lr_train_window,
            test_window=args.lr_test_window,
            threshold_grid=THRESHOLD_GRID,
            min_trades=args.min_threshold_trades,
        )
        results["logreg_walk_forward_train_val"] = lr_walk
        extra_tables["logreg_threshold_history"] = pd.DataFrame(
            {"threshold": lr_walk["threshold_history"]}
        )
        results["logreg_holdout"] = run_train_then_holdout_classifier(
            train_val,
            holdout,
            features,
            model_factory=lr_factory,
            threshold_grid=THRESHOLD_GRID,
            min_trades=args.min_threshold_trades,
        )

    if "rf" in models:
        rf_walk, rf_table, best_rf_config = run_random_forest_battle(train_val, features)
        results[f"random_forest_walk_forward_{best_rf_config['name']}"] = rf_walk
        extra_tables["random_forest_config_metrics"] = rf_table
        results[f"random_forest_holdout_{best_rf_config['name']}"] = fixed_threshold_holdout(
            train_val,
            holdout,
            features,
            model_factory=lambda cfg=best_rf_config: RandomForestClassifier(
                n_estimators=50,
                max_depth=int(cfg["depth"]),
                min_samples_leaf=int(cfg["leaf"]),
                random_state=42,
                n_jobs=-1,
            ),
            threshold=0.52,
            score_kind="predict_proba",
            scale=False,
        )

    if "svm" in models:
        results["svm_walk_forward_train_val"] = run_svm_walk_forward(train_val, features)
        results["svm_holdout"] = fixed_threshold_holdout(
            train_val,
            holdout,
            features,
            model_factory=lambda: LinearSVC(
                C=0.01,
                penalty="l2",
                loss="squared_hinge",
                dual=False,
                class_weight="balanced",
                random_state=42,
                max_iter=2000,
            ),
            threshold=0.0,
            score_kind="decision_function",
            scale=True,
        )

    config = {
        "btc_path": args.btc_path,
        "nq_path": args.nq_path,
        "output_dir": args.output_dir,
        "models": sorted(models),
        "row_limit": args.row_limit,
        "feature_set": args.feature_set,
        "max_holding_minutes": args.max_holding_minutes,
        "safety_close": f"{args.safety_close_hour:02d}:{args.safety_close_minute:02d}",
        "train_frac": args.train_frac,
        "lr_train_window": args.lr_train_window,
        "lr_test_window": args.lr_test_window,
        "threshold_grid": THRESHOLD_GRID.tolist(),
    }
    write_outputs(args, trades, features, market, results, extra_tables, config)

    metrics = pd.DataFrame([{"name": name, **result["metrics"]} for name, result in results.items()])
    print(metrics.to_string(index=False))
    print(f"\nWrote full reproduction outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
