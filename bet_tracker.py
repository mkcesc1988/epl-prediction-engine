from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline import load_config, normalize_team

PROCESSED = Path("data/processed")
HISTORY = Path("data/history")
PORTFOLIO_PATH = PROCESSED / "paper_portfolio_latest.csv"
ODDS_HISTORY_PATH = HISTORY / "market_odds_history.csv"
LEDGER_PATH = HISTORY / "auto_bet_ledger.csv"
SUMMARY_PATH = HISTORY / "bet_performance_summary.csv"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _norm(value: object) -> str:
    mapping = {
        "Manchester City": "Man City", "Manchester United": "Man United",
        "Newcastle United": "Newcastle", "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
        "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
        "Nottingham Forest": "Nott'm Forest", "Leeds United": "Leeds",
    }
    s = str(value).strip()
    return normalize_team(mapping.get(s, s))


def _bet_id(row: pd.Series) -> str:
    raw = "|".join([
        str(row.get("Date", "")), _norm(row.get("HomeTeam", "")), _norm(row.get("AwayTeam", "")),
        str(row.get("MarketType", "")), str(row.get("Selection", "")),
        str(row.get("Line", "")), str(row.get("ModelVersion", "V1.2")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _confidence_bucket(score: float) -> str:
    if score >= 80: return "80-100"
    if score >= 70: return "70-79"
    if score >= 60: return "60-69"
    return "<60"


def _edge_bucket(edge: float) -> str:
    pp = edge * 100.0
    if pp >= 10: return "10+ pp"
    if pp >= 5: return "5-9.99 pp"
    if pp >= 2: return "2-4.99 pp"
    return "<2 pp"


def _new_entries(portfolio: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    existing_ids = set(existing.get("BetID", pd.Series(dtype=str)).astype(str)) if not existing.empty else set()
    rows = []
    for _, r in portfolio.iterrows():
        bid = _bet_id(r)
        if bid in existing_ids:
            continue
        p = float(pd.to_numeric(r.get("ModelWinProbability"), errors="coerce"))
        edge = float(pd.to_numeric(r.get("ProbabilityEdge"), errors="coerce"))
        quality = float(pd.to_numeric(r.get("BetQualityScore"), errors="coerce"))
        rows.append({
            "BetID": bid,
            "EntrySnapshotUTC": now,
            "Date": r.get("Date"), "KickoffUTC": r.get("KickoffUTC"),
            "HomeTeam": _norm(r.get("HomeTeam")), "AwayTeam": _norm(r.get("AwayTeam")),
            "MarketType": r.get("MarketType"), "Selection": r.get("Selection"), "Line": r.get("Line"),
            "Bookmaker": r.get("Bookmaker"), "EntryOdds": r.get("MyBookieOdds"),
            "ModelProbability": p, "PushProbability": r.get("PushProbability", 0.0),
            "ModelFairOdds": r.get("ModelFairOdds"), "EntryImpliedProbability": r.get("MyBookieImpliedProbability"),
            "ProbabilityEdge": edge, "ExpectedReturnPerUnit": r.get("ExpectedReturnPerUnit"),
            "BetQualityScore": quality, "ProfitabilityScore": r.get("ProfitabilityScore"),
            "OverallRankScore": r.get("OverallRankScore"), "Grade": r.get("Grade"),
            "ValidationStatus": r.get("ValidationStatus"), "StakeUnits": r.get("PaperStakeUnits"),
            "StakeAmount": r.get("PaperStakeAmount"), "ModelVersion": r.get("ModelVersion", "V1.2"),
            "ConfidenceBucket": _confidence_bucket(quality), "EdgeBucket": _edge_bucket(edge),
            "ClosingOdds": np.nan, "ClosingSnapshotUTC": pd.NA, "HoursBeforeKickoff": np.nan,
            "ClosingQuality": pd.NA, "PriceCLV": np.nan, "ImpliedProbabilityCLV": np.nan,
            "Result": "OPEN", "FinalScore": pd.NA, "ProfitUnits": np.nan,
            "BrierScore": np.nan, "SettledUTC": pd.NA,
        })
    return pd.DataFrame(rows)


def _raw_market_match(bet: pd.Series, odds: pd.DataFrame) -> pd.DataFrame:
    m = odds.copy()
    m = m[(m["HomeTeam"].map(_norm) == _norm(bet["HomeTeam"])) & (m["AwayTeam"].map(_norm) == _norm(bet["AwayTeam"]))]
    book = m.get("Bookmaker", pd.Series(index=m.index, dtype=object)).astype(str).str.lower()
    key = m.get("BookmakerKey", pd.Series(index=m.index, dtype=object)).astype(str).str.lower()
    m = m[book.str.contains("mybookie", na=False) | key.str.contains("mybookie", na=False)]
    mt = str(bet["MarketType"])
    sel = str(bet["Selection"])
    if mt == "Moneyline":
        m = m[m["Market"].astype(str) == "h2h"]
        target = "Draw" if sel.lower() == "draw" else _norm(sel)
        m = m[m["Outcome"].map(lambda x: "Draw" if str(x).lower() == "draw" else _norm(x)) == target]
    elif mt == "Total":
        m = m[m["Market"].astype(str) == "totals"]
        side = sel.split()[0]
        line = float(pd.to_numeric(bet.get("Line"), errors="coerce"))
        m = m[(m["Outcome"].astype(str).str.title() == side.title()) & (pd.to_numeric(m["Point"], errors="coerce") == line)]
    elif mt == "Spread":
        m = m[m["Market"].astype(str) == "spreads"]
        line = float(pd.to_numeric(bet.get("Line"), errors="coerce"))
        team = sel.rsplit(" ", 1)[0]
        m = m[(m["Outcome"].map(_norm) == _norm(team)) & (pd.to_numeric(m["Point"], errors="coerce") == line)]
    else:
        return m.iloc[0:0]
    return m


def _update_closing(ledger: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or odds_history.empty:
        return ledger
    odds = odds_history.copy()
    odds["SnapshotUTC"] = pd.to_datetime(odds["SnapshotUTC"], utc=True, errors="coerce")
    odds["KickoffUTC"] = pd.to_datetime(odds.get("KickoffUTC"), utc=True, errors="coerce")
    odds["DecimalOdds"] = pd.to_numeric(odds["DecimalOdds"], errors="coerce")
    for idx, bet in ledger.iterrows():
        kickoff = pd.to_datetime(bet.get("KickoffUTC"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        matched = _raw_market_match(bet, odds)
        matched = matched[(matched["SnapshotUTC"].notna()) & (matched["SnapshotUTC"] < kickoff)].sort_values("SnapshotUTC")
        if matched.empty:
            continue
        close = matched.iloc[-1]
        close_odds = float(close["DecimalOdds"])
        entry_odds = float(pd.to_numeric(bet["EntryOdds"], errors="coerce"))
        hours = (kickoff - close["SnapshotUTC"]).total_seconds() / 3600.0
        quality = "near_close" if hours <= 1.5 else "good_proxy" if hours <= 4 else "proxy" if hours <= 12 else "early_proxy"
        ledger.at[idx, "ClosingOdds"] = close_odds
        ledger.at[idx, "ClosingSnapshotUTC"] = close["SnapshotUTC"].isoformat()
        ledger.at[idx, "HoursBeforeKickoff"] = hours
        ledger.at[idx, "ClosingQuality"] = quality
        ledger.at[idx, "PriceCLV"] = entry_odds / close_odds - 1.0
        ledger.at[idx, "ImpliedProbabilityCLV"] = 1.0 / close_odds - 1.0 / entry_odds
    return ledger


def _fetch_results() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 EPL prediction engine"}
    teams = requests.get(FPL_BOOTSTRAP_URL, headers=headers, timeout=45); teams.raise_for_status()
    fixtures = requests.get(FPL_FIXTURES_URL, headers=headers, timeout=45); fixtures.raise_for_status()
    team_map = {int(t["id"]): _norm(t.get("name")) for t in teams.json().get("teams", [])}
    rows = []
    for f in fixtures.json() or []:
        if not f.get("finished"):
            continue
        rows.append({"HomeTeam": team_map.get(f.get("team_h")), "AwayTeam": team_map.get(f.get("team_a")),
                     "HomeGoals": f.get("team_h_score"), "AwayGoals": f.get("team_a_score")})
    return pd.DataFrame(rows)


def _settle_one(bet: pd.Series, hg: int, ag: int) -> tuple[str, float]:
    mt, sel = str(bet["MarketType"]), str(bet["Selection"])
    if mt == "Moneyline":
        winner = _norm(bet["HomeTeam"]) if hg > ag else _norm(bet["AwayTeam"]) if ag > hg else "Draw"
        return ("W" if (_norm(sel) if sel.lower() != "draw" else "Draw") == winner else "L", 1.0 if winner == (_norm(sel) if sel.lower() != "draw" else "Draw") else 0.0)
    if mt == "Total":
        line = float(bet["Line"]); total = hg + ag; over = sel.lower().startswith("over")
        if abs(total - line) < 1e-9: return "PUSH", np.nan
        win = total > line if over else total < line
        return ("W" if win else "L", 1.0 if win else 0.0)
    if mt == "Spread":
        line = float(bet["Line"]); team = _norm(sel.rsplit(" ", 1)[0]); home = team == _norm(bet["HomeTeam"])
        margin = (hg - ag) if home else (ag - hg); adjusted = margin + line
        if abs(adjusted) < 1e-9: return "PUSH", np.nan
        win = adjusted > 0
        return ("W" if win else "L", 1.0 if win else 0.0)
    return "OPEN", np.nan


def _settle(ledger: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or results.empty:
        return ledger
    for idx, bet in ledger[ledger["Result"].astype(str) == "OPEN"].iterrows():
        r = results[(results["HomeTeam"].map(_norm) == _norm(bet["HomeTeam"])) & (results["AwayTeam"].map(_norm) == _norm(bet["AwayTeam"]))]
        if r.empty: continue
        hg, ag = int(r.iloc[-1]["HomeGoals"]), int(r.iloc[-1]["AwayGoals"])
        result, y = _settle_one(bet, hg, ag)
        if result == "OPEN": continue
        stake = float(pd.to_numeric(bet["StakeUnits"], errors="coerce")); odds = float(pd.to_numeric(bet["EntryOdds"], errors="coerce"))
        profit = stake * (odds - 1.0) if result == "W" else -stake if result == "L" else 0.0
        p = float(pd.to_numeric(bet["ModelProbability"], errors="coerce"))
        ledger.at[idx, "Result"] = result; ledger.at[idx, "FinalScore"] = f"{hg}-{ag}"; ledger.at[idx, "ProfitUnits"] = profit
        ledger.at[idx, "BrierScore"] = (p - y) ** 2 if np.isfinite(y) else np.nan
        ledger.at[idx, "SettledUTC"] = pd.Timestamp.now(tz="UTC").isoformat()
    return ledger


def _summary(ledger: pd.DataFrame) -> pd.DataFrame:
    settled = ledger[ledger["Result"].isin(["W", "L", "PUSH"])].copy()
    if settled.empty: return pd.DataFrame()
    settled["StakeUnits"] = pd.to_numeric(settled["StakeUnits"], errors="coerce")
    settled["ProfitUnits"] = pd.to_numeric(settled["ProfitUnits"], errors="coerce")
    settled["BrierScore"] = pd.to_numeric(settled["BrierScore"], errors="coerce")
    settled["PriceCLV"] = pd.to_numeric(settled["PriceCLV"], errors="coerce")
    rows = []
    groups = [("ALL", settled)]
    for col in ["MarketType", "ConfidenceBucket", "EdgeBucket", "Grade"]:
        groups += [(f"{col}:{k}", g) for k, g in settled.groupby(col, dropna=False)]
    for label, g in groups:
        stake = g["StakeUnits"].sum(); profit = g["ProfitUnits"].sum()
        rows.append({"Segment": label, "Bets": len(g), "Wins": int((g["Result"] == "W").sum()), "Losses": int((g["Result"] == "L").sum()),
                     "Pushes": int((g["Result"] == "PUSH").sum()), "StakeUnits": stake, "ProfitUnits": profit,
                     "ROI": profit / stake if stake else np.nan, "AvgCLV": g["PriceCLV"].mean(), "AvgBrier": g["BrierScore"].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    portfolio = pd.read_csv(PORTFOLIO_PATH) if PORTFOLIO_PATH.exists() else pd.DataFrame()
    ledger = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    new = _new_entries(portfolio, ledger)
    if not new.empty:
        ledger = pd.concat([ledger, new], ignore_index=True, sort=False) if not ledger.empty else new
    odds_history = pd.read_csv(ODDS_HISTORY_PATH) if ODDS_HISTORY_PATH.exists() else pd.DataFrame()
    ledger = _update_closing(ledger, odds_history)
    try:
        ledger = _settle(ledger, _fetch_results())
    except Exception as exc:
        print(f"Result settlement skipped: {exc}")
    ledger.to_csv(LEDGER_PATH, index=False)
    summary = _summary(ledger)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"New tracked bets: {len(new)}")
    print(f"Total tracked bets: {len(ledger)}")
    print(f"Settled bets: {int(ledger['Result'].isin(['W','L','PUSH']).sum()) if not ledger.empty else 0}")
    if not summary.empty:
        print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
