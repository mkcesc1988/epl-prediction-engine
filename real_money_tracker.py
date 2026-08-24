from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bet_tracker import _base_market_match, _fetch_results, _norm, _settle_one

HISTORY = Path("data/history")
LEDGER_PATH = HISTORY / "real_money_bet_ledger.csv"
ODDS_HISTORY_PATH = HISTORY / "market_odds_history.csv"
SUMMARY_PATH = HISTORY / "real_money_performance_summary.csv"

NUMERIC_LEDGER_COLUMNS = [
    "EntryAmericanOdds",
    "EntryOdds",
    "StakeUSD",
    "PotentialProfitUSD",
    "ModelProbability",
    "V2Probability",
    "ClosingOdds",
    "ClosingLine",
    "LineMove",
    "SameLineClosingOdds",
    "SameLinePriceCLV",
    "HoursBeforeKickoff",
    "PriceCLV",
    "ProfitUSD",
    "BenchmarkClosingOdds",
    "BenchmarkClosingLine",
    "BenchmarkLineMove",
    "BenchmarkHoursBeforeKickoff",
    "BenchmarkPriceCLV",
    "Line",
]


def _enforce_ledger_schema(ledger: pd.DataFrame) -> pd.DataFrame:
    """Make numeric ledger fields numeric before assigning float values.

    CSV inference can load all-empty numeric columns as Arrow/string dtype. Pandas 2.x
    then raises when a float such as 0.0 is assigned into that column. Explicitly
    coercing known numeric fields makes tracker writes deterministic across pandas
    versions and regardless of how many blanks a CSV currently contains.
    """
    if ledger.empty:
        return ledger
    ledger = ledger.copy()
    for col in NUMERIC_LEDGER_COLUMNS:
        if col in ledger.columns:
            ledger[col] = pd.to_numeric(ledger[col], errors="coerce").astype("float64")
    return ledger


def _update_closing(ledger: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or odds_history.empty:
        return ledger

    ledger = _enforce_ledger_schema(ledger)
    odds = odds_history.copy()
    odds["SnapshotUTC"] = pd.to_datetime(odds["SnapshotUTC"], utc=True, errors="coerce")
    odds["DecimalOdds"] = pd.to_numeric(odds["DecimalOdds"], errors="coerce")
    odds["Point"] = pd.to_numeric(odds.get("Point"), errors="coerce")

    for idx, bet in ledger.iterrows():
        kickoff = pd.to_datetime(bet.get("KickoffUTC"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        base = _base_market_match(bet, odds)
        base = base[(base["SnapshotUTC"].notna()) & (base["SnapshotUTC"] < kickoff)].sort_values("SnapshotUTC")
        if base.empty:
            continue

        mt = str(bet.get("MarketType", ""))
        entry_odds = float(pd.to_numeric(bet.get("EntryOdds"), errors="coerce"))

        if mt == "Spread":
            latest_ts = base["SnapshotUTC"].max()
            latest = base[base["SnapshotUTC"] == latest_ts].dropna(subset=["Point", "DecimalOdds"])
            if latest.empty:
                continue
            entry_line = float(pd.to_numeric(bet.get("Line"), errors="coerce"))
            latest = latest.assign(_dist=(latest["Point"] - entry_line).abs()).sort_values(["_dist", "DecimalOdds"], ascending=[True, False])
            close = latest.iloc[0]
            close_line = float(close["Point"])
            close_odds = float(close["DecimalOdds"])
            same = base[np.isclose(base["Point"], entry_line, atol=1e-9)].sort_values("SnapshotUTC")
            same_odds = float(same.iloc[-1]["DecimalOdds"]) if not same.empty else np.nan
            ledger.at[idx, "ClosingLine"] = close_line
            ledger.at[idx, "LineMove"] = close_line - entry_line
            ledger.at[idx, "SameLineClosingOdds"] = same_odds
            ledger.at[idx, "SameLinePriceCLV"] = entry_odds / same_odds - 1.0 if np.isfinite(same_odds) and same_odds > 0 else np.nan
            ledger.at[idx, "LineAwareCLVStatus"] = "line_moved" if not np.isclose(close_line, entry_line) else "same_line"
            ledger.at[idx, "PriceCLV"] = entry_odds / close_odds - 1.0 if np.isclose(close_line, entry_line) else np.nan
        else:
            close = base.iloc[-1]
            close_odds = float(close["DecimalOdds"])
            ledger.at[idx, "PriceCLV"] = entry_odds / close_odds - 1.0
            ledger.at[idx, "LineAwareCLVStatus"] = "same_market"

        hours = (kickoff - close["SnapshotUTC"]).total_seconds() / 3600.0
        quality = "near_close" if hours <= 1.5 else "good_proxy" if hours <= 4 else "proxy" if hours <= 12 else "early_proxy"
        ledger.at[idx, "ClosingOdds"] = close_odds
        ledger.at[idx, "ClosingSnapshotUTC"] = close["SnapshotUTC"].isoformat()
        ledger.at[idx, "HoursBeforeKickoff"] = hours
        ledger.at[idx, "ClosingQuality"] = quality

    return ledger


def _settle(ledger: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or results.empty:
        return ledger
    ledger = _enforce_ledger_schema(ledger)
    for idx, bet in ledger[ledger["Result"].astype(str) == "OPEN"].iterrows():
        r = results[(results["HomeTeam"].map(_norm) == _norm(bet["HomeTeam"])) & (results["AwayTeam"].map(_norm) == _norm(bet["AwayTeam"]))]
        if r.empty:
            continue
        hg, ag = int(r.iloc[-1]["HomeGoals"]), int(r.iloc[-1]["AwayGoals"])
        result, _ = _settle_one(bet, hg, ag)
        if result == "OPEN":
            continue
        stake = float(pd.to_numeric(bet.get("StakeUSD"), errors="coerce"))
        odds = float(pd.to_numeric(bet.get("EntryOdds"), errors="coerce"))
        profit = stake * (odds - 1.0) if result == "W" else -stake if result == "L" else 0.0
        ledger.at[idx, "Result"] = result
        ledger.at[idx, "FinalScore"] = f"{hg}-{ag}"
        ledger.at[idx, "ProfitUSD"] = profit
        ledger.at[idx, "SettledUTC"] = pd.Timestamp.now(tz="UTC").isoformat()
    return ledger


def _summary(ledger: pd.DataFrame) -> pd.DataFrame:
    settled = ledger[ledger["Result"].isin(["W", "L", "PUSH"])].copy()
    if settled.empty:
        return pd.DataFrame()
    settled["StakeUSD"] = pd.to_numeric(settled["StakeUSD"], errors="coerce")
    settled["ProfitUSD"] = pd.to_numeric(settled["ProfitUSD"], errors="coerce")
    rows = []
    for label, group in [("ALL", settled)] + [(f"MarketType:{k}", g) for k, g in settled.groupby("MarketType")]:
        stake = group["StakeUSD"].sum()
        profit = group["ProfitUSD"].sum()
        rows.append({
            "Segment": label,
            "Bets": len(group),
            "Wins": int((group["Result"] == "W").sum()),
            "Losses": int((group["Result"] == "L").sum()),
            "Pushes": int((group["Result"] == "PUSH").sum()),
            "StakeUSD": stake,
            "ProfitUSD": profit,
            "ROI": profit / stake if stake else np.nan,
            "AvgPriceCLV": pd.to_numeric(group.get("PriceCLV"), errors="coerce").mean(),
            "AvgSameLinePriceCLV": pd.to_numeric(group.get("SameLinePriceCLV"), errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    ledger = _enforce_ledger_schema(ledger)
    odds_history = pd.read_csv(ODDS_HISTORY_PATH) if ODDS_HISTORY_PATH.exists() else pd.DataFrame()
    ledger = _update_closing(ledger, odds_history)
    try:
        ledger = _settle(ledger, _fetch_results())
    except Exception as exc:
        print(f"Real-money settlement skipped: {exc}")
    ledger.to_csv(LEDGER_PATH, index=False)
    summary = _summary(ledger)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Real-money bets tracked: {len(ledger)}")
    print(f"Real-money bets settled: {int(ledger['Result'].isin(['W','L','PUSH']).sum()) if not ledger.empty else 0}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
