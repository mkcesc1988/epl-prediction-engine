from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_predictions import fetch_fpl_fixtures
from live_evaluation import (
    archive_predictions,
    calibration_table,
    evaluation_summary,
    rolling_evaluation,
    settle_predictions,
)
from pipeline import load_config


def _enrich_fixture_ids(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attach FPL fixture IDs to older prediction files that did not save them."""
    if predictions.empty or "FPLFixtureId" in predictions.columns:
        return predictions

    left = predictions.copy()
    right = fixtures[["Season", "Date", "HomeTeam", "AwayTeam", "FPLFixtureId"]].copy()
    left["Date"] = pd.to_datetime(left["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    right["Date"] = pd.to_datetime(right["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return left.merge(
        right,
        on=["Season", "Date", "HomeTeam", "AwayTeam"],
        how="left",
        validate="many_to_one",
    )


def _prediction_files(out_dir: Path) -> list[Path]:
    files = sorted(out_dir.glob("daily_predictions_gw*.csv"))
    latest = out_dir / "daily_predictions_latest.csv"
    if latest.exists():
        files.append(latest)
    return files


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    current_season = int(cfg.get("daily", {}).get("current_season", 2026))
    fixtures = fetch_fpl_fixtures(current_season)

    archive_path = out_dir / "prediction_history.csv"
    archive = pd.read_csv(archive_path) if archive_path.exists() else pd.DataFrame()

    files = _prediction_files(out_dir)
    if not files:
        raise FileNotFoundError("No daily prediction files found. Run daily_predictions.py first.")

    for file in files:
        predictions = pd.read_csv(file)
        predictions = _enrich_fixture_ids(predictions, fixtures)
        archive = archive_predictions(predictions, archive_path)

    settled = settle_predictions(archive, fixtures)
    settled.to_csv(archive_path, index=False)

    summary = evaluation_summary(settled)
    calibration = calibration_table(settled)
    rolling = rolling_evaluation(settled)

    summary_path = out_dir / "live_evaluation_summary.csv"
    calibration_path = out_dir / "live_calibration.csv"
    rolling_path = out_dir / "live_rolling_evaluation.csv"

    summary.to_csv(summary_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    rolling.to_csv(rolling_path, index=False)

    print("Live evaluation updated")
    print(f"Prediction archive: {archive_path}")
    print(f"Summary: {summary_path}")
    print(f"Calibration: {calibration_path}")
    print(f"Rolling metrics: {rolling_path}")
    print("\nCurrent summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
