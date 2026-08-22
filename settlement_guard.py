from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bet_tracker import _norm, _settle_one

HISTORY = Path("data/history")
PAPER_LEDGER = HISTORY / "auto_bet_ledger.csv"
REAL_LEDGER = HISTORY / "real_money_bet_ledger.csv"
AUDIT_PATH = HISTORY / "settlement_audit_latest.csv"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
API_FOOTBALL_URL = "https://v3.football.api-sports.io/fixtures"
OVERDUE_HOURS = 3.0


def _fetch_fpl_results() -> dict[tuple[str, str], tuple[int, int]]:
    headers = {"User-Agent": "Mozilla/5.0 EPL prediction engine"}
    teams = requests.get(FPL_BOOTSTRAP_URL, headers=headers, timeout=45)
    teams.raise_for_status()
    fixtures = requests.get(FPL_FIXTURES_URL, headers=headers, timeout=45)
    fixtures.raise_for_status()
    team_map = {int(t["id"]): _norm(t.get("name")) for t in teams.json().get("teams", [])}
    out = {}
    for f in fixtures.json() or []:
        if not f.get("finished"):
            continue
        h = team_map.get(f.get("team_h")); a = team_map.get(f.get("team_a"))
        hg, ag = f.get("team_h_score"), f.get("team_a_score")
        if h and a and hg is not None and ag is not None:
            out[(h, a)] = (int(hg), int(ag))
    return out


def _fetch_api_football_results(dates: list[str]) -> dict[tuple[str, str], tuple[int, int]]:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        return {}
    headers = {"x-apisports-key": key}
    out = {}
    for date in sorted(set(dates)):
        r = requests.get(API_FOOTBALL_URL, headers=headers, params={"date": date}, timeout=45)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("response", []) or []:
            status = str(item.get("fixture", {}).get("status", {}).get("short", ""))
            if status not in {"FT", "AET", "PEN"}:
                continue
            h = _norm(item.get("teams", {}).get("home", {}).get("name"))
            a = _norm(item.get("teams", {}).get("away", {}).get("name"))
            goals = item.get("goals", {}) or {}
            hg, ag = goals.get("home"), goals.get("away")
            if h and a and hg is not None and ag is not None:
                out[(h, a)] = (int(hg), int(ag))
    return out


def _profit_expected(row: pd.Series, result: str, stake_col: str, odds_col: str) -> float:
    stake = float(pd.to_numeric(row.get(stake_col), errors="coerce"))
    odds = float(pd.to_numeric(row.get(odds_col), errors="coerce"))
    if result == "W": return stake * (odds - 1.0)
    if result == "L": return -stake
    if result == "PUSH": return 0.0
    return np.nan


def _settle_row(row: pd.Series, score: tuple[int, int], stake_col: str, odds_col: str) -> tuple[str, str, float]:
    hg, ag = score
    result, _ = _settle_one(row, hg, ag)
    profit = _profit_expected(row, result, stake_col, odds_col)
    return result, f"{hg}-{ag}", profit


def _audit_ledger(
    ledger: pd.DataFrame,
    ledger_name: str,
    id_col: str,
    stake_col: str,
    odds_col: str,
    profit_col: str,
    fpl: dict,
    api: dict,
    now: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    if ledger.empty:
        return ledger, rows

    if id_col in ledger.columns:
        dupes = ledger[ledger[id_col].astype(str).duplicated(keep=False)]
        for _, r in dupes.iterrows():
            rows.append({"Ledger": ledger_name, "BetID": r.get(id_col), "HomeTeam": r.get("HomeTeam"), "AwayTeam": r.get("AwayTeam"), "IssueType": "DUPLICATE_ID", "Severity": "ERROR", "ActualStatus": r.get("Result")})

    for idx, bet in ledger.iterrows():
        bet_id = bet.get(id_col, idx)
        h, a = _norm(bet.get("HomeTeam")), _norm(bet.get("AwayTeam"))
        key = (h, a)
        fpl_score, api_score = fpl.get(key), api.get(key)
        kickoff = pd.to_datetime(bet.get("KickoffUTC"), utc=True, errors="coerce")
        status = str(bet.get("Result", "OPEN"))
        expected_status = "OPEN"
        issue = ""
        severity = "INFO"
        verified_score = None
        source = ""

        if fpl_score is not None and api_score is not None:
            if fpl_score != api_score:
                issue, severity = "RESULT_SOURCE_CONFLICT", "ERROR"
            else:
                verified_score, source = fpl_score, "FPL+API_FOOTBALL"
        elif fpl_score is not None:
            verified_score, source = fpl_score, "FPL_ONLY"
        elif api_score is not None:
            verified_score, source = api_score, "API_FOOTBALL_ONLY"

        overdue = pd.notna(kickoff) and now > kickoff + pd.Timedelta(hours=OVERDUE_HOURS)
        excluded = status.startswith("EXCLUDED")
        if verified_score is not None and status == "OPEN" and not excluded:
            result, final_score, profit = _settle_row(bet, verified_score, stake_col, odds_col)
            ledger.at[idx, "Result"] = result
            ledger.at[idx, "FinalScore"] = final_score
            ledger.at[idx, profit_col] = profit
            ledger.at[idx, "SettledUTC"] = now.isoformat()
            status = result

        if verified_score is not None and not excluded:
            expected_status = _settle_row(bet, verified_score, stake_col, odds_col)[0]
        elif overdue and status == "OPEN" and issue == "":
            issue, severity = "OVERDUE_OPEN_NO_RESULT", "ERROR"

        if status in {"W", "L", "PUSH"}:
            final_score = str(bet.get("FinalScore", ledger.at[idx, "FinalScore"] if "FinalScore" in ledger.columns else ""))
            if not final_score or final_score.lower() in {"nan", "<na>"}:
                issue, severity = "SETTLED_MISSING_FINAL_SCORE", "ERROR"
            actual_profit = pd.to_numeric(ledger.at[idx, profit_col] if profit_col in ledger.columns else np.nan, errors="coerce")
            expected_profit = _profit_expected(ledger.loc[idx], status, stake_col, odds_col)
            if np.isfinite(expected_profit) and (pd.isna(actual_profit) or abs(float(actual_profit) - expected_profit) > 1e-6):
                issue, severity = "PROFIT_MISMATCH", "ERROR"

        if verified_score is not None and status in {"W", "L", "PUSH"} and expected_status != status:
            issue, severity = "SETTLEMENT_RESULT_MISMATCH", "ERROR"

        rows.append({
            "Ledger": ledger_name,
            "BetID": bet_id,
            "KickoffUTC": bet.get("KickoffUTC"),
            "HomeTeam": h,
            "AwayTeam": a,
            "MarketType": bet.get("MarketType"),
            "Selection": bet.get("Selection"),
            "Line": bet.get("Line"),
            "FPLScore": "" if fpl_score is None else f"{fpl_score[0]}-{fpl_score[1]}",
            "APIFootballScore": "" if api_score is None else f"{api_score[0]}-{api_score[1]}",
            "VerifiedSource": source,
            "ExpectedStatus": expected_status,
            "ActualStatus": status,
            "IssueType": issue,
            "Severity": severity,
        })
    return ledger, rows


def reconcile() -> int:
    HISTORY.mkdir(parents=True, exist_ok=True)
    paper = pd.read_csv(PAPER_LEDGER) if PAPER_LEDGER.exists() else pd.DataFrame()
    real = pd.read_csv(REAL_LEDGER) if REAL_LEDGER.exists() else pd.DataFrame()
    dates = []
    for df in [paper, real]:
        if not df.empty and "Date" in df.columns:
            dates.extend(df["Date"].dropna().astype(str).tolist())

    now = pd.Timestamp.now(tz="UTC")
    source_errors = []
    try:
        fpl = _fetch_fpl_results()
    except Exception as exc:
        fpl = {}; source_errors.append(f"FPL:{exc}")
    try:
        api = _fetch_api_football_results(dates)
    except Exception as exc:
        api = {}; source_errors.append(f"API_FOOTBALL:{exc}")

    paper, p_rows = _audit_ledger(paper, "paper", "BetID", "StakeUnits", "EntryOdds", "ProfitUnits", fpl, api, now)
    real, r_rows = _audit_ledger(real, "real_money", "RealBetID", "StakeUSD", "EntryOdds", "ProfitUSD", fpl, api, now)
    if not paper.empty: paper.to_csv(PAPER_LEDGER, index=False)
    if not real.empty: real.to_csv(REAL_LEDGER, index=False)

    audit = pd.DataFrame(p_rows + r_rows)
    if source_errors:
        extra = pd.DataFrame([{"Ledger": "SYSTEM", "BetID": "", "HomeTeam": "", "AwayTeam": "", "IssueType": "RESULT_SOURCE_ERROR", "Severity": "ERROR", "ActualStatus": " | ".join(source_errors)}])
        audit = pd.concat([audit, extra], ignore_index=True, sort=False)
    audit.to_csv(AUDIT_PATH, index=False)

    errors = int((audit.get("Severity", pd.Series(dtype=str)) == "ERROR").sum()) if not audit.empty else 0
    overdue = int((audit.get("IssueType", pd.Series(dtype=str)) == "OVERDUE_OPEN_NO_RESULT").sum()) if not audit.empty else 0
    conflicts = int((audit.get("IssueType", pd.Series(dtype=str)) == "RESULT_SOURCE_CONFLICT").sum()) if not audit.empty else 0
    print(f"Settlement audit rows: {len(audit)}")
    print(f"Overdue open bets: {overdue}")
    print(f"Settlement conflicts: {conflicts}")
    print(f"Ledger integrity errors: {errors}")
    return 0


def assert_clean() -> int:
    if not AUDIT_PATH.exists():
        print("ERROR: settlement audit file missing")
        return 1
    audit = pd.read_csv(AUDIT_PATH)
    errors = audit[audit.get("Severity", pd.Series(index=audit.index, dtype=str)).astype(str) == "ERROR"] if not audit.empty else audit
    if not errors.empty:
        print("Settlement integrity check FAILED")
        cols = [c for c in ["Ledger", "BetID", "HomeTeam", "AwayTeam", "IssueType", "ActualStatus"] if c in errors.columns]
        print(errors[cols].to_string(index=False))
        return 1
    print("Settlement integrity check PASSED")
    print("Overdue open bets: 0")
    print("Settlement conflicts: 0")
    print("Ledger integrity errors: 0")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-clean", action="store_true")
    args = parser.parse_args()
    raise SystemExit(assert_clean() if args.assert_clean else reconcile())


if __name__ == "__main__":
    main()
