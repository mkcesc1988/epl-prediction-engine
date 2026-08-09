from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import build_master, load_config
from model import build_predictions
from backtest import walk_forward_backtest


def main() -> None:
    cfg = load_config()

    print("\n=== 1. BUILD MASTER DATABASE ===")
    master = build_master(cfg)
    print(f"Master rows: {len(master)}")

    print("\n=== 2. BUILD LEAKAGE-SAFE xG PREDICTIONS ===")
    predictions = build_predictions(master, cfg)
    predictions.to_csv(cfg["paths"]["predictions"], index=False)
    valid = predictions["RawP_Over2_5_xG"].notna().sum()
    print(f"Predictions generated: {valid}")

    print("\n=== 3. WALK-FORWARD BACKTEST ===")
    evaluated, summary = walk_forward_backtest(predictions, cfg)
    summary.to_csv(cfg["paths"]["summary"], index=False)
    evaluated_path = Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions.csv"
    evaluated.to_csv(evaluated_path, index=False)

    if summary.empty:
        print("No walk-forward seasons available. Check xG coverage and season range.")
    else:
        pd.set_option("display.max_columns", None)
        print(summary.to_string(index=False))

    print("\n=== COMPLETE ===")
    print(f"Master:      {cfg['paths']['master']}")
    print(f"Predictions: {cfg['paths']['predictions']}")
    print(f"Backtest:    {cfg['paths']['summary']}")
    print(f"Validation:  {cfg['paths']['validation']}")


if __name__ == "__main__":
    main()
