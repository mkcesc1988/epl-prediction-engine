from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PREDICTIONS = Path("data/processed/daily_predictions_latest.csv")
LATEST = Path("data/processed/post_match_process_latest.csv")
HISTORY = Path("data/history/post_match_process_history.csv")
API_BASE = "https://v3.football.api-sports.io"

ALIASES = {
    "manchester united": "man united", "man utd": "man united",
    "manchester city": "man city", "newcastle united": "newcastle",
    "tottenham hotspur": "tottenham", "brighton and hove albion": "brighton",
    "brighton & hove albion": "brighton", "nottingham forest": "nott'm forest",
    "ipswich": "ipswich town", "leeds united": "leeds",
    "west ham united": "west ham", "wolverhampton wanderers": "wolves",
}


def norm_team(value: object) -> str:
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    return ALIASES.get(s, s)


def api_get(path: str, params: dict) -> dict:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is not configured")
    r = requests.get(f"{API_BASE}/{path.lstrip('/')}", headers={"x-apisports-key": key}, params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football error: {payload['errors']}")
    return payload


def _num(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _stats(block: dict) -> dict[str, float]:
    out = {}
    for item in block.get("statistics", []) or []:
        key = str(item.get("type", "")).strip().lower().replace(" ", "_")
        out[key] = _num(item.get("value"))
    return out


def _pick(stats: dict[str, float], *keys: str) -> float:
    for key in keys:
        key = key.lower().replace(" ", "_")
        if key in stats:
            return stats[key]
    return np.nan


def fetch_date(date: str) -> list[dict]:
    fixtures = api_get("fixtures", {"date": date}).get("response", []) or []
    rows = []
    for item in fixtures:
        league = item.get("league", {}) or {}
        if str(league.get("name", "")).lower() != "premier league" or str(league.get("country", "")).lower() != "england":
            continue
        if str(item.get("fixture", {}).get("status", {}).get("short", "")) not in {"FT", "AET", "PEN"}:
            continue
        fixture_id = item.get("fixture", {}).get("id")
        home = norm_team(item.get("teams", {}).get("home", {}).get("name"))
        away = norm_team(item.get("teams", {}).get("away", {}).get("name"))
        stats_payload = api_get("fixtures/statistics", {"fixture": fixture_id}).get("response", []) or []
        team_stats = {norm_team(b.get("team", {}).get("name")): _stats(b) for b in stats_payload}
        hs, aws = team_stats.get(home, {}), team_stats.get(away, {})
        goals = item.get("goals", {}) or {}
        rows.append({
            "Date": date, "FixtureID": fixture_id, "HomeKey": home, "AwayKey": away,
            "HomeGoals": _num(goals.get("home")), "AwayGoals": _num(goals.get("away")),
            "ActualHomeXG": _pick(hs, "expected_goals"), "ActualAwayXG": _pick(aws, "expected_goals"),
            "HomeShots": _pick(hs, "total_shots"), "AwayShots": _pick(aws, "total_shots"),
            "HomeShotsOnTarget": _pick(hs, "shots_on_goal"), "AwayShotsOnTarget": _pick(aws, "shots_on_goal"),
            "HomeShotsInsideBox": _pick(hs, "shots_insidebox", "shots_inside_box"),
            "AwayShotsInsideBox": _pick(aws, "shots_insidebox", "shots_inside_box"),
            "HomeBigChances": _pick(hs, "big_chances"), "AwayBigChances": _pick(aws, "big_chances"),
            "HomePossessionPct": _pick(hs, "ball_possession"), "AwayPossessionPct": _pick(aws, "ball_possession"),
        })
    return rows


def grade(row: pd.Series) -> str:
    if pd.isna(row["ActualHomeXG"]) or pd.isna(row["ActualAwayXG"]):
        return "RESULT_ONLY_NO_XG"
    total_error = abs(row["ActualTotalXG"] - row["Lambda_Total_xG"])
    split_error = max(abs(row["HomeXGError"]), abs(row["AwayXGError"]))
    if total_error <= 0.60 and split_error <= 0.75:
        return "PROCESS_ALIGNED"
    if total_error <= 0.60:
        return "TOTAL_ALIGNED_TEAM_SPLIT_MISS"
    if split_error >= 1.25:
        return "MAJOR_TEAM_XG_MISS"
    return "PROCESS_MIXED"


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    pred["HomeKey"] = pred["HomeTeam"].map(norm_team)
    pred["AwayKey"] = pred["AwayTeam"].map(norm_team)
    actual = pd.DataFrame([r for date in sorted(set(pred["Date"].astype(str))) for r in fetch_date(date)])
    if actual.empty:
        print("No finished EPL fixtures found for prediction dates")
        return
    merged = pred.merge(actual, on=["Date", "HomeKey", "AwayKey"], how="inner")
    merged["ActualTotalGoals"] = merged["HomeGoals"] + merged["AwayGoals"]
    merged["ActualTotalXG"] = merged["ActualHomeXG"] + merged["ActualAwayXG"]
    merged["HomeXGError"] = merged["ActualHomeXG"] - merged["Lambda_Home_xG"]
    merged["AwayXGError"] = merged["ActualAwayXG"] - merged["Lambda_Away_xG"]
    merged["TotalXGError"] = merged["ActualTotalXG"] - merged["Lambda_Total_xG"]
    merged["ProcessGrade"] = merged.apply(grade, axis=1)
    merged["AuditUTC"] = pd.Timestamp.now(tz="UTC").isoformat()
    cols = ["Date","FPLGameweek","Season","FixtureID","HomeTeam","AwayTeam","ModelVersion","HomeGoals","AwayGoals","ActualTotalGoals","Lambda_Home_xG","Lambda_Away_xG","Lambda_Total_xG","ActualHomeXG","ActualAwayXG","ActualTotalXG","HomeXGError","AwayXGError","TotalXGError","HomeShots","AwayShots","HomeShotsOnTarget","AwayShotsOnTarget","HomeShotsInsideBox","AwayShotsInsideBox","HomeBigChances","AwayBigChances","HomePossessionPct","AwayPossessionPct","CalP_Over2_5","P_HomeWin","P_Draw","P_AwayWin","ProcessGrade","AuditUTC"]
    out = merged[cols].copy()
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LATEST, index=False)
    if HISTORY.exists():
        hist = pd.read_csv(HISTORY)
        out_hist = pd.concat([hist, out], ignore_index=True, sort=False).drop_duplicates(subset=["Date","FixtureID","ModelVersion"], keep="last")
    else:
        out_hist = out
    out_hist.sort_values(["Date","FixtureID"]).to_csv(HISTORY, index=False)
    print(out[["Date","HomeTeam","AwayTeam","Lambda_Total_xG","ActualTotalXG","ActualTotalGoals","ProcessGrade"]].to_string(index=False))


if __name__ == "__main__":
    main()
