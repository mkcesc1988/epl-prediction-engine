from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import build_master, load_config
from model import build_predictions
from model_v07 import build_predictions_v07
from model_v11 import build_predictions_v11
from model_v12 import build_predictions_v12
from backtest import walk_forward_backtest


def _comparison_frame(base: pd.DataFrame, candidate: pd.DataFrame, candidate_label: str) -> pd.DataFrame:
    left = base[["Season", "Matches", "Brier", "LogLoss"]].copy()
    left = left.rename(columns={
        "Brier": "Brier_Base",
        "LogLoss": "LogLoss_Base",
        "Matches": "Matches_Base",
    })
    right = candidate[["Season", "Matches", "Brier", "LogLoss"]].copy()
    right = right.rename(columns={
        "Brier": f"Brier_{candidate_label}",
        "LogLoss": f"LogLoss_{candidate_label}",
        "Matches": f"Matches_{candidate_label}",
    })
    out = left.merge(right, on="Season", how="outer")
    out[f"BrierDelta_{candidate_label}_minus_Base"] = out[f"Brier_{candidate_label}"] - out["Brier_Base"]
    out[f"LogLossDelta_{candidate_label}_minus_Base"] = out[f"LogLoss_{candidate_label}"] - out["LogLoss_Base"]
    out[f"{candidate_label}_Better_Brier"] = out[f"BrierDelta_{candidate_label}_minus_Base"] < 0
    out[f"{candidate_label}_Better_LogLoss"] = out[f"LogLossDelta_{candidate_label}_minus_Base"] < 0
    return out


def main() -> None:
    cfg = load_config()

    print("\n=== 1. BUILD MASTER DATABASE ===")
    master = build_master(cfg)
    print(f"Master rows: {len(master)}")

    print("\n=== 2. BUILD V1.0 BASELINE PREDICTIONS ===")
    predictions = build_predictions(master, cfg)
    predictions.to_csv(cfg["paths"]["predictions"], index=False)
    print(f"V1.0 predictions generated: {predictions['RawP_Over2_5_xG'].notna().sum()}")

    print("\n=== 3. BACKTEST V1.0 BASELINE ===")
    evaluated, summary = walk_forward_backtest(predictions, cfg)
    summary.to_csv(cfg["paths"]["summary"], index=False)
    evaluated.to_csv(Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions.csv", index=False)

    print("\n=== 4. BUILD V0.7 RESEARCH CANDIDATE ===")
    candidate_v07 = build_predictions_v07(master, cfg)
    candidate_v07.to_csv(cfg["paths"]["predictions_v07"], index=False)
    evaluated_v07, summary_v07 = walk_forward_backtest(candidate_v07, cfg)
    summary_v07.to_csv(cfg["paths"]["summary_v07"], index=False)
    evaluated_v07.to_csv(Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions_v07.csv", index=False)
    _comparison_frame(summary, summary_v07, "V07").to_csv(cfg["paths"]["comparison"], index=False)

    print("\n=== 5. BUILD V1.1 DIXON-COLES CANDIDATE ===")
    candidate_v11 = build_predictions_v11(master, cfg)
    candidate_v11.to_csv(cfg["paths"]["predictions_v11"], index=False)
    evaluated_v11, summary_v11 = walk_forward_backtest(candidate_v11, cfg)
    summary_v11.to_csv(cfg["paths"]["summary_v11"], index=False)
    evaluated_v11.to_csv(Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions_v11.csv", index=False)
    comparison_v11 = _comparison_frame(summary, summary_v11, "V11")
    comparison_v11.to_csv(cfg["paths"]["comparison_v11"], index=False)

    print("\n=== 6. BUILD V1.2 FITTED ATTACK/DEFENSE CANDIDATE ===")
    candidate_v12 = build_predictions_v12(master, cfg)
    candidate_v12.to_csv(cfg["paths"]["predictions_v12"], index=False)
    print(f"V1.2 predictions generated: {candidate_v12['RawP_Over2_5_xG'].notna().sum()}")

    print("\n=== 7. BACKTEST V1.2 FITTED STRENGTH MODEL ===")
    evaluated_v12, summary_v12 = walk_forward_backtest(candidate_v12, cfg)
    summary_v12.to_csv(cfg["paths"]["summary_v12"], index=False)
    evaluated_v12.to_csv(Path(cfg["paths"]["processed_dir"]) / "walkforward_predictions_v12.csv", index=False)

    print("\n=== 8. COMPARE V1.0 VS V1.2 ===")
    comparison_v12 = _comparison_frame(summary, summary_v12, "V12")
    comparison_v12.to_csv(cfg["paths"]["comparison_v12"], index=False)

    pd.set_option("display.max_columns", None)
    if not summary.empty:
        print("\nV1.0 baseline")
        print(summary.to_string(index=False))
    if not summary_v12.empty:
        print("\nV1.2 fitted attack/defense candidate")
        print(summary_v12.to_string(index=False))
    if not comparison_v12.empty:
        print("\nV1.0 vs V1.2, negative deltas favor V1.2")
        print(comparison_v12.to_string(index=False))

    print("\n=== COMPLETE ===")
    print(f"Master:          {cfg['paths']['master']}")
    print(f"V1.0 backtest:   {cfg['paths']['summary']}")
    print(f"V1.2 predictions:{cfg['paths']['predictions_v12']}")
    print(f"V1.2 backtest:   {cfg['paths']['summary_v12']}")
    print(f"V1.2 comparison: {cfg['paths']['comparison_v12']}")
    print(f"Validation:      {cfg['paths']['validation']}")


if __name__ == "__main__":
    main()
