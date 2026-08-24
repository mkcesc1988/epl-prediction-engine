from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY = Path("data/history")
LEDGER_PATH = HISTORY / "auto_bet_ledger.csv"


def _norm(value: object) -> str:
    s = str(value).strip().lower()
    aliases = {
        "newcastle united": "newcastle",
        "liverpool fc": "liverpool",
        "manchester city": "man city",
        "manchester united": "man united",
        "nottingham forest": "nott'm forest",
        "leeds united": "leeds",
    }
    return aliases.get(s, s)


def _slot_key(row: pd.Series) -> tuple:
    fixture = (
        str(row.get("Date", "")),
        _norm(row.get("HomeTeam")),
        _norm(row.get("AwayTeam")),
        str(row.get("ModelVersion", "V1.2")),
    )
    market = str(row.get("MarketType", ""))
    if market == "Total":
        line = pd.to_numeric(row.get("Line"), errors="coerce")
        return fixture + (market, None if pd.isna(line) else float(line))
    return fixture + (market,)


def main() -> None:
    if not LEDGER_PATH.exists():
        print("No paper ledger found")
        return

    df = pd.read_csv(LEDGER_PATH)
    if df.empty:
        print("Paper ledger empty")
        return

    entry = pd.to_datetime(df.get("EntrySnapshotUTC"), utc=True, errors="coerce")
    kickoff = pd.to_datetime(df.get("KickoffUTC"), utc=True, errors="coerce")
    result = df.get("Result", pd.Series("OPEN", index=df.index)).fillna("OPEN").astype(str)

    # Rule 1: no official bet may be admitted at or after kickoff.
    late = result.eq("OPEN") & entry.notna() & kickoff.notna() & (entry >= kickoff)
    df.loc[late, "Result"] = "EXCLUDED_LATE_ENTRY"
    df.loc[late, "ValidationStatus"] = "Excluded: entry timestamp at/after kickoff"

    # Rule 2: freeze the first official market slot for a fixture.
    # One moneyline and one spread slot per fixture. Totals are frozen per line,
    # so Over 1.5 and Under 3.5 can coexist, but opposite sides of the same line cannot.
    ordered = df.assign(_entry=entry).sort_values("_entry", na_position="last")
    claimed: dict[tuple, int] = {}
    for idx, row in ordered.iterrows():
        status = str(df.at[idx, "Result"])
        if status.startswith("EXCLUDED"):
            continue
        key = _slot_key(row)
        if key not in claimed:
            claimed[key] = idx
            continue
        first_idx = claimed[key]
        if idx == first_idx:
            continue
        # Historical settled rows are never rewritten. Only unresolved duplicate/conflicting
        # admissions are excluded from the official portfolio.
        if str(df.at[idx, "Result"]) == "OPEN":
            df.at[idx, "Result"] = "EXCLUDED_CONFLICTING_SIGNAL"
            df.at[idx, "ValidationStatus"] = (
                "Excluded: fixture/market slot already frozen by earlier official bet"
            )

    df.drop(columns=[c for c in ["_entry"] if c in df.columns], errors="ignore").to_csv(LEDGER_PATH, index=False)
    print(f"Late entries excluded: {int(late.sum())}")
    print(f"Conflicting open signals excluded: {int((df['Result'].astype(str) == 'EXCLUDED_CONFLICTING_SIGNAL').sum())}")


if __name__ == "__main__":
    main()
