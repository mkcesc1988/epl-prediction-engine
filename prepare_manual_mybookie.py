from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
BASE = DATA_DIR / "manual_mybookie_odds.csv"
PATTERN = "manual_mybookie_odds*.csv"
KEYS = ["Date", "HomeTeam", "AwayTeam", "Market", "Outcome", "Point"]


def main() -> None:
    files = sorted(DATA_DIR.glob(PATTERN))
    frames: list[pd.DataFrame] = []

    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        print("No manual MyBookie snapshots available")
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["Date"] = merged["Date"].astype(str)
    merged["Point"] = pd.to_numeric(merged.get("Point"), errors="coerce")
    merged["DecimalOdds"] = pd.to_numeric(merged.get("DecimalOdds"), errors="coerce")
    merged = merged.dropna(subset=["Date", "HomeTeam", "AwayTeam", "Market", "Outcome", "DecimalOdds"])

    dedupe_keys = [c for c in KEYS if c in merged.columns]
    if dedupe_keys:
        merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")

    merged = merged.drop(columns=["_source_file"], errors="ignore")
    preferred = [
        "Date", "HomeTeam", "AwayTeam", "Bookmaker", "BookmakerKey",
        "Market", "Outcome", "Point", "DecimalOdds", "ManualSource",
    ]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[cols].sort_values(["Date", "HomeTeam", "AwayTeam", "Market", "Outcome"], kind="stable")
    merged.to_csv(BASE, index=False)
    print(f"Merged {len(files)} manual snapshot files into {BASE} with {len(merged)} rows")


if __name__ == "__main__":
    main()
