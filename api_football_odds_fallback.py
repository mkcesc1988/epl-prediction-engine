from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import requests

from market_comparison import _norm, compare_totals_25

BASE_URL = "https://v3.football.api-sports.io"
DATA_DIR = Path("data/processed")
PRED_PATH = DATA_DIR / "daily_predictions_latest.csv"
ODDS_PATH = DATA_DIR / "market_odds_latest.csv"
COMP_PATH = DATA_DIR / "market_comparison_latest.csv"


def _api_get(params: dict) -> dict:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is not configured")
    r = requests.get(
        f"{BASE_URL}/odds",
        headers={"x-apisports-key": key},
        params=params,
        timeout=45,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football odds error: {payload['errors']}")
    return payload


def _parse_value(value: object, bet_name: str, home: str, away: str) -> tuple[str | None, str | None, float | None]:
    raw = str(value).strip()
    low_bet = bet_name.lower()
    low = raw.lower()

    if "match winner" in low_bet or low_bet in {"1x2", "winner"}:
        if low in {"home", "1"}:
            return "h2h", home, None
        if low in {"away", "2"}:
            return "h2h", away, None
        if low in {"draw", "x"}:
            return "h2h", "Draw", None
        team = _norm(raw)
        if team in {home, away}:
            return "h2h", team, None

    if "over/under" in low_bet or "goals over" in low_bet or low_bet == "goals":
        m = re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)", raw, re.I)
        if m:
            return "totals", m.group(1).title(), float(m.group(2))

    if "asian handicap" in low_bet or "handicap" in low_bet:
        m = re.search(r"^(home|away|.+?)\s*([+-][0-9]+(?:\.[0-9]+)?)$", raw, re.I)
        if m:
            side = m.group(1).strip()
            point = float(m.group(2))
            if side.lower() == "home":
                return "spreads", home, point
            if side.lower() == "away":
                return "spreads", away, point
            team = _norm(side)
            if team in {home, away}:
                return "spreads", team, point

    return None, None, None


def fetch_api_football_odds(preds: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(preds["Date"].astype(str).unique())
    if not dates:
        return pd.DataFrame()

    rows: list[dict] = []
    page = 1
    while True:
        payload = _api_get({
            "league": 39,
            "season": 2026,
            "date": dates[0] if len(dates) == 1 else None,
            "page": page,
        })
        response = payload.get("response", []) or []
        for item in response:
            fixture = item.get("fixture", {})
            league = item.get("league", {})
            kickoff = pd.to_datetime(fixture.get("date"), utc=True, errors="coerce")
            if pd.isna(kickoff):
                continue
            date = kickoff.tz_convert(None).strftime("%Y-%m-%d")
            if date not in dates:
                continue
            home = _norm(item.get("teams", {}).get("home", {}).get("name", ""))
            away = _norm(item.get("teams", {}).get("away", {}).get("name", ""))
            if (date, home, away) not in set(zip(preds["Date"].astype(str), preds["HomeTeam"].map(_norm), preds["AwayTeam"].map(_norm))):
                continue
            for book in item.get("bookmakers", []) or []:
                book_name = str(book.get("name") or book.get("id") or "API-Football")
                book_key = f"api_football_{book.get('id', book_name)}"
                for bet in book.get("bets", []) or []:
                    bet_name = str(bet.get("name", ""))
                    for val in bet.get("values", []) or []:
                        market, outcome, point = _parse_value(val.get("value"), bet_name, home, away)
                        if not market:
                            continue
                        odd = pd.to_numeric(val.get("odd"), errors="coerce")
                        if pd.isna(odd) or float(odd) <= 1.0:
                            continue
                        rows.append({
                            "EventId": f"api-football-{fixture.get('id')}",
                            "Date": date,
                            "KickoffUTC": kickoff.isoformat(),
                            "HomeTeam": home,
                            "AwayTeam": away,
                            "Bookmaker": book_name,
                            "BookmakerKey": book_key,
                            "BookLastUpdate": item.get("update") or league.get("name") or "api_football",
                            "Market": market,
                            "Outcome": outcome,
                            "Point": point,
                            "DecimalOdds": float(odd),
                        })
        paging = payload.get("paging", {}) or {}
        current = int(paging.get("current", page) or page)
        total = int(paging.get("total", current) or current)
        if current >= total:
            break
        page += 1
    return pd.DataFrame(rows)


def main() -> None:
    if not PRED_PATH.exists():
        raise RuntimeError("daily_predictions_latest.csv is missing")
    preds = pd.read_csv(PRED_PATH)
    if preds.empty:
        print("No predictions, skipping API-Football odds fallback")
        return

    current_keys = set(zip(preds["Date"].astype(str), preds["HomeTeam"].map(_norm), preds["AwayTeam"].map(_norm)))
    existing = pd.read_csv(ODDS_PATH) if ODDS_PATH.exists() and ODDS_PATH.stat().st_size else pd.DataFrame()
    existing_current = 0
    if not existing.empty:
        existing_current = sum(
            (str(r.Date), _norm(r.HomeTeam), _norm(r.AwayTeam)) in current_keys
            for r in existing[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False)
        )
    if existing_current > 0:
        print(f"Current-gameweek market rows already available: {existing_current}; API-Football fallback not needed")
        return

    fallback = fetch_api_football_odds(preds)
    print(f"API-Football fallback market rows parsed: {len(fallback)}")
    if fallback.empty:
        return

    merged = pd.concat([existing, fallback], ignore_index=True, sort=False) if not existing.empty else fallback
    merged.to_csv(ODDS_PATH, index=False)
    comparison = compare_totals_25(preds, merged)
    comparison.to_csv(COMP_PATH, index=False)

    names = sorted(set(fallback["Bookmaker"].astype(str)))
    my = [n for n in names if "mybookie" in n.lower()]
    print(f"API-Football bookmakers parsed: {len(names)}")
    print(f"MyBookie present in API-Football feed: {bool(my)}")
    if my:
        print("MyBookie labels: " + ", ".join(my))


if __name__ == "__main__":
    main()
