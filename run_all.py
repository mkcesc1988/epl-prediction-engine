from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import build_master, load_config
from model import build_predictions
from model_v07 import build_predictions_v07
from backtest import walk_forward_backtest


def _comparison_frame(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    left = base[["Season", "Matches", "Brier", "LogLoss"]].copy()
    left = left.rename(columns={
        "Brier": "Brier_V06",
        "LogLoss": "LogLoss_V06",
        "Matches": "Matches_V06",
    })
    right = candidate[["Season", "Matches", "Brier", "LogLoss"]].copy()
    right = right.rename(columns={
        "Brier": "Brier_V07",
        "LogLoss": "LogLoss_V07",
        "Matches": "Matches_V07",
    })
    out = left.merge(right, on="Season", how="outer")
    out["BrierDelta_V07_minus_V06"] = out["Brier_V07"] - out["Brier_V06"]
    out["LogLossDelta_V07_minus_V06"] = out["LogLoss_V07"] - out["LogLoss_V06"]
    out["V07_Better_Brier"] = out["BrierDelta_V07_minus_V06"] < 0
    out["V07_Better_LogLoss"] = out["LogLossDelta_V07_minus_V06"] < 0
    return out


def main() -> None:
    cfg = load_config()

    print("\n=== 1. BUILD MASTER DATABASE ===")
    master = build_master(cfg)
    print(f"Master rows: {len(master)}")

    print("\n=== 2. BUILD V0.6 BASELINE PREDICTIONS ===")
    predictions = build_predictions(master, cfg)
    predictions.to_csv(cfg["paths"]["predictions"], index=False)
    valid = predictions["RawP_Over2_5_xG"].notna().sum()
    print(f"V0.6 predictions generated: {valid}")

    print("\n=== 3. BACKTEST V0.6 ===")
    evaluated, summary = walk_forward_backtest(predictions, cfg)
    summary.to_csv(cfg["paths"]["summary"], index=False)
    evaluated_path = Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions.csv"
    evaluated.to_csv(evaluated_path, index=False)

    print("\n=== 4. BUILD V0.7 CANDIDATE PREDICTIONS ===")
    candidate_predictions = build_predictions_v07(master, cfg)
    candidate_predictions.to_csv(cfg["paths"]["predictions_v07"], index=False)
    valid_v07 = candidate_predictions["RawP_Over2_5_xG"].notna().sum()
    print(f"V0.7 candidate predictions generated: {valid_v07}")

    print("\n=== 5. BACKTEST V0.7 CANDIDATE ===")
    evaluated_v07, summary_v07 = walk_forward_backtest(candidate_predictions, cfg)
    summary_v07.to_csv(cfg["paths"]["summary_v07"], index=False)
    evaluated_v07_path = Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions_v07.csv"
    evaluated_v07.to_csv(evaluated_v07_path, index=False)

    print("\n=== 6. COMPARE V0.6 VS V0.7 ===")
    comparison = _comparison_frame(summary, summary_v07)
    comparison.to_csv(cfg["paths"]["comparison"], index=False)

    if summary.empty:
        print("No V0.6 walk-forward seasons available.")
    else:
        pd.set_option("display.max_columns", None)
        print("\nV0.6 baseline")
        print(summary.to_string(index=False))

    if summary_v07.empty:
        print("No V0.7 walk-forward seasons available.")
    else:
        print("\nV0.7 candidate")
        print(summary_v07.to_string(index=False))

    if not comparison.empty:
        print("\nModel comparison, negative deltas favor V0.7")
        print(comparison.to_string(index=False))

    print("\n=== COMPLETE ===")
    print(f"Master:         {cfg['paths']['master']}")
    print(f"V0.6 preds:     {cfg['paths']['predictions']}")
    print(f"V0.6 backtest:  {cfg['paths']['summary']}")
    print(f"V0.7 preds:     {cfg['paths']['predictions_v07']}")
    print(f"V0.7 backtest:  {cfg['paths']['summary_v07']}")
    print(f"Comparison:     {cfg['paths']['comparison']}")
    print(f"Validation:     {cfg['paths']['validation']}")


if __name__ == "__main__":
    main()
