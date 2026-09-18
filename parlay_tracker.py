from __future__ import annotations

from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")
HISTORY = Path("data/history")
DEFS = PROCESSED / "parlay_watch_portfolio.csv"
SINGLES = HISTORY / "auto_bet_ledger.csv"
LEDGER = HISTORY / "parlay_bet_ledger.csv"

def settle_parlays() -> None:
    if not DEFS.exists() or not SINGLES.exists():
        return
    defs = pd.read_csv(DEFS)
    singles = pd.read_csv(SINGLES)
    if defs.empty:
        return

    single_map = singles.set_index("BetID", drop=False)
    rows = []
    now = pd.Timestamp.now(tz="UTC").isoformat()

    for _, r in defs.iterrows():
        leg_ids = [x for x in str(r.get("LegBetIDs", "")).split("|") if x]
        leg_results = []
        missing = False
        for bid in leg_ids:
            if bid not in single_map.index:
                missing = True
                break
            leg_results.append(str(single_map.loc[bid].get("Result", "OPEN")))

        status = "OPEN"
        profit = pd.NA
        settled = pd.NA
        if not missing:
            if any(x == "L" for x in leg_results):
                status = "L"
            elif all(x in {"W", "PUSH"} for x in leg_results) and any(x == "W" for x in leg_results):
                status = "W"
            elif all(x == "PUSH" for x in leg_results):
                status = "PUSH"

            if status != "OPEN":
                stake = float(pd.to_numeric(r.get("StakeUnits"), errors="coerce"))
                odds = float(pd.to_numeric(r.get("EntryOdds"), errors="coerce"))
                if status == "W":
                    profit = stake * (odds - 1.0)
                elif status == "L":
                    profit = -stake
                else:
                    profit = 0.0
                settled = now

        out = r.to_dict()
        out["Status"] = status
        out["ProfitUnits"] = profit
        out["SettledUTC"] = settled
        out["LegResults"] = "|".join(leg_results) if not missing else "MISSING_LEG"
        rows.append(out)

    out = pd.DataFrame(rows)
    HISTORY.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEDGER, index=False)
    print(out[["ParlayID", "Status", "ProfitUnits"]].to_string(index=False))

if __name__ == "__main__":
    settle_parlays()
