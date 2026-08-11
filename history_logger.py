from __future__ import annotations

from pathlib import Path

import pandas as pd


HISTORY_DIR = Path("data/history")
PROCESSED_DIR = Path("data/processed")


def _append_snapshot(source: Path, destination: Path, snapshot_utc: str) -> int:
    if not source.exists():
        return 0

    current = pd.read_csv(source)
    if current.empty:
        return 0

    current.insert(0, "SnapshotUTC", snapshot_utc)

    if destination.exists():
        old = pd.read_csv(destination)
        combined = pd.concat([old, current], ignore_index=True, sort=False)
    else:
        combined = current

    # A workflow retry at the same snapshot time should not duplicate rows.
    dedupe_cols = [c for c in [
        "SnapshotUTC", "Date", "HomeTeam", "AwayTeam", "Market", "Side",
        "Bookmaker", "EventId", "Outcome", "Point"
    ] if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")

    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return len(current)


def main() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_utc = pd.Timestamp.now(tz="UTC").isoformat()

    counts = {
        "predictions": _append_snapshot(
            PROCESSED_DIR / "daily_predictions_latest.csv",
            HISTORY_DIR / "prediction_history.csv",
            snapshot_utc,
        ),
        "market_odds": _append_snapshot(
            PROCESSED_DIR / "market_odds_latest.csv",
            HISTORY_DIR / "market_odds_history.csv",
            snapshot_utc,
        ),
        "comparisons": _append_snapshot(
            PROCESSED_DIR / "market_comparison_latest.csv",
            HISTORY_DIR / "market_comparison_history.csv",
            snapshot_utc,
        ),
    }

    print(f"Snapshot UTC: {snapshot_utc}")
    for name, count in counts.items():
        print(f"Appended {count:4d} rows to {name}")


if __name__ == "__main__":
    main()
