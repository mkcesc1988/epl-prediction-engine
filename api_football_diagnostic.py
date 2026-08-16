from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://v3.football.api-sports.io"
OUT_DIR = Path("data/diagnostics/api_football")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPL_CLUBS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle", "Nottingham Forest", "Sunderland", "Tottenham",
    "West Ham", "Wolves", "Hull City", "Ipswich Town", "Nott'm Forest",
    "Man City", "Man United",
}

REQUEST_DELAY_SECONDS = 1.25
MAX_RETRIES = 4


def api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is not set")

    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        r = requests.get(
            url,
            headers={"x-apisports-key": key},
            params=params,
            timeout=30,
        )

        if r.status_code == 429:
            wait = 4 * (attempt + 1)
            print(f"Rate limited on {endpoint}; waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}.")
            time.sleep(wait)
            continue

        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            raise RuntimeError(f"API-Football error for {endpoint}: {payload['errors']}")
        time.sleep(REQUEST_DELAY_SECONDS)
        return payload

    raise RuntimeError(f"API-Football rate limit persisted for {endpoint} after {MAX_RETRIES} retries")


def safe_api_get(endpoint: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        return api_get(endpoint, params), ""
    except Exception as exc:
        return {"response": []}, str(exc)


def normalize_name(name: str) -> str:
    aliases = {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Nottingham Forest": "Nott'm Forest",
        "Tottenham Hotspur": "Tottenham",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton & Hove Albion": "Brighton",
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Leeds United": "Leeds",
        "Sunderland AFC": "Sunderland",
    }
    return aliases.get(name, name)


def fetch_friendlies(season: int) -> dict[str, Any]:
    return api_get(
        "fixtures",
        {
            "league": 667,
            "season": season,
            "from": f"{season}-06-15",
            "to": f"{season}-08-20",
        },
    )


def main() -> None:
    requested_season = 2026
    tested_season = requested_season
    current_season_access = True
    access_note = "2026 season accessible on current API plan."

    try:
        fixtures_payload = fetch_friendlies(requested_season)
    except RuntimeError as exc:
        message = str(exc)
        if "Free plans do not have access to this season" not in message:
            raise
        current_season_access = False
        tested_season = 2024
        access_note = (
            "Current free plan cannot access 2026. Diagnostic automatically fell back to 2024 "
            "to test Club Friendlies coverage and available fields before considering a paid plan."
        )
        print(access_note)
        fixtures_payload = fetch_friendlies(tested_season)

    fixture_rows: list[dict[str, Any]] = []
    raw_fixtures = fixtures_payload.get("response", [])
    for item in raw_fixtures:
        home_raw = item["teams"]["home"]["name"]
        away_raw = item["teams"]["away"]["name"]
        home = normalize_name(home_raw)
        away = normalize_name(away_raw)
        if home not in EPL_CLUBS and away not in EPL_CLUBS:
            continue
        fixture_rows.append(
            {
                "fixture_id": item["fixture"]["id"],
                "date": item["fixture"]["date"],
                "status": item["fixture"]["status"]["short"],
                "home_team": home,
                "away_team": away,
                "home_goals": item.get("goals", {}).get("home"),
                "away_goals": item.get("goals", {}).get("away"),
                "league_name": item["league"]["name"],
                "tested_season": tested_season,
            }
        )

    fixtures_df = pd.DataFrame(fixture_rows)
    fixtures_path = OUT_DIR / f"preseason_fixtures_{tested_season}.csv"
    fixtures_df.to_csv(fixtures_path, index=False)

    completed = fixtures_df[fixtures_df["status"].isin(["FT", "AET", "PEN"])] if not fixtures_df.empty else fixtures_df
    sample = completed.head(5)

    coverage_rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        fixture_id = int(row["fixture_id"])
        stats_payload, stats_error = safe_api_get("fixtures/statistics", {"fixture": fixture_id})
        lineups_payload, lineups_error = safe_api_get("fixtures/lineups", {"fixture": fixture_id})

        stats_resp = stats_payload.get("response", [])
        lineups_resp = lineups_payload.get("response", [])

        stat_types: set[str] = set()
        for team_block in stats_resp:
            for stat in team_block.get("statistics", []):
                stat_types.add(str(stat.get("type")))

        coverage_rows.append(
            {
                "fixture_id": fixture_id,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "statistics_available": bool(stats_resp),
                "lineups_available": bool(lineups_resp),
                "shots_on_goal": "Shots on Goal" in stat_types,
                "total_shots": "Total Shots" in stat_types,
                "possession": "Ball Possession" in stat_types,
                "corner_kicks": "Corner Kicks" in stat_types,
                "expected_goals": "expected_goals" in stat_types or "Expected Goals" in stat_types,
                "stats_error": stats_error,
                "lineups_error": lineups_error,
                "stat_fields": " | ".join(sorted(stat_types)),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = OUT_DIR / f"coverage_sample_{tested_season}.csv"
    coverage_df.to_csv(coverage_path, index=False)

    report = {
        "requested_season": requested_season,
        "tested_season": tested_season,
        "current_season_access": current_season_access,
        "access_note": access_note,
        "api_fixture_results_total": len(raw_fixtures),
        "epl_related_preseason_fixtures": len(fixtures_df),
        "completed_sample_checked": len(coverage_df),
        "statistics_coverage_pct": float(coverage_df["statistics_available"].mean() * 100) if not coverage_df.empty else 0.0,
        "lineup_coverage_pct": float(coverage_df["lineups_available"].mean() * 100) if not coverage_df.empty else 0.0,
        "shots_on_goal_coverage_pct": float(coverage_df["shots_on_goal"].mean() * 100) if not coverage_df.empty else 0.0,
        "total_shots_coverage_pct": float(coverage_df["total_shots"].mean() * 100) if not coverage_df.empty else 0.0,
        "possession_coverage_pct": float(coverage_df["possession"].mean() * 100) if not coverage_df.empty else 0.0,
        "expected_goals_coverage_pct": float(coverage_df["expected_goals"].mean() * 100) if not coverage_df.empty else 0.0,
        "rate_limit_strategy": "5-match sample, 1.25s request spacing, 429 retry backoff",
        "note": "Diagnostic only. No V1.2 production model files are modified.",
    }

    report_path = OUT_DIR / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved: {fixtures_path}")
    print(f"Saved: {coverage_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
