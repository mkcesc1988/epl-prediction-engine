"""Build historical EPL player match stats from Understat.

Uses Understat match rosters for minutes/starts and match shots for shots/SOT,
then writes the provider-neutral schema consumed by player_stats_pipeline.py.

Default range follows the main EPL database: 2021/22 through 2025/26.

Usage:
    python understat_player_ingest.py
    python understat_player_ingest.py --start-season 2021 --end-season 2025

Output:
    data/raw/player_match_stats.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import pandas as pd
from understatapi import UnderstatClient

OUT = Path("data/raw/player_match_stats.csv")


def _team_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("title") or value.get("name") or value.get("team") or "").strip()
    return str(value or "").strip()


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _match_meta(m: dict) -> tuple[str, str, str, str]:
    match_id = str(m.get("id") or "").strip()
    kickoff = str(m.get("datetime") or m.get("date") or "").strip()
    home = _team_name(m.get("h") or m.get("home") or m.get("home_team"))
    away = _team_name(m.get("a") or m.get("away") or m.get("away_team"))
    return match_id, kickoff, home, away


def _flatten_rosters(rosters: object, home: str, away: str) -> list[dict]:
    rows: list[dict] = []

    def consume(team_name: str, players: object) -> None:
        if isinstance(players, dict):
            iterable = players.values()
        elif isinstance(players, list):
            iterable = players
        else:
            return
        for p in iterable:
            if not isinstance(p, dict):
                continue
            player_id = str(p.get("id") or p.get("player_id") or "").strip()
            player = str(p.get("player") or p.get("player_name") or p.get("name") or "").strip()
            minutes = _as_float(p.get("time") if p.get("time") is not None else p.get("minutes"), 0.0)
            position = str(p.get("position") or "").strip()
            starter_raw = p.get("is_starter")
            if starter_raw is None:
                starter_raw = p.get("starter")
            if starter_raw is None:
                # Understat rosters commonly identify substitutes by position='Sub'.
                started = position.lower() not in {"sub", "substitute"} and minutes > 0
            else:
                started = str(starter_raw).strip().lower() in {"1", "true", "yes", "y"}
            rows.append({
                "player_id": player_id,
                "player": player,
                "team": team_name,
                "minutes": minutes,
                "started": started,
            })

    if isinstance(rosters, dict):
        # Understat commonly exposes h/a roster buckets.
        if "h" in rosters or "a" in rosters:
            consume(home, rosters.get("h", []))
            consume(away, rosters.get("a", []))
        else:
            for key, value in rosters.items():
                key_l = str(key).lower()
                team_name = home if key_l in {"home", "h"} else away if key_l in {"away", "a"} else str(key)
                consume(team_name, value)
    return rows


def _shot_counts(shots: object) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"shots": 0, "shots_on_target": 0})

    def consume(team_side: str, items: object) -> None:
        if not isinstance(items, list):
            return
        for s in items:
            if not isinstance(s, dict):
                continue
            pid = str(s.get("player_id") or s.get("id") or "").strip()
            pname = str(s.get("player") or s.get("player_name") or "").strip()
            key = (pid, pname)
            counts[key]["shots"] += 1
            result = str(s.get("result") or "").strip().lower()
            # Understat result labels generally include Goal, SavedShot, MissedShots,
            # BlockedShot, ShotOnPost. Goal and SavedShot are shots on target.
            if result in {"goal", "savedshot", "saved shot"}:
                counts[key]["shots_on_target"] += 1

    if isinstance(shots, dict):
        consume("h", shots.get("h", []))
        consume("a", shots.get("a", []))
    elif isinstance(shots, list):
        consume("", shots)
    return counts


def build_season(client: UnderstatClient, season: int, pause_seconds: float = 0.05) -> list[dict]:
    matches = client.league(league="EPL").get_match_data(season=str(season)) or []
    out: list[dict] = []
    for idx, m in enumerate(matches, start=1):
        match_id, kickoff, home, away = _match_meta(m)
        if not match_id:
            continue
        try:
            match_api = client.match(match=match_id)
            rosters = match_api.get_roster_data()
            shots = match_api.get_shot_data()
        except AttributeError:
            # Compatibility with alternate understatapi method names.
            match_api = client.match(match=match_id)
            rosters = match_api.get_player_data()
            shots = match_api.get_shot_data()
        except Exception as exc:
            print(f"WARNING season={season} match={match_id}: {exc}")
            continue

        roster_rows = _flatten_rosters(rosters, home, away)
        shot_map = _shot_counts(shots)

        for r in roster_rows:
            pid = r["player_id"]
            pname = r["player"]
            shot_stats = shot_map.get((pid, pname))
            if shot_stats is None and pid:
                # Some Understat responses omit player names in one endpoint.
                candidates = [v for (candidate_id, _), v in shot_map.items() if candidate_id == pid]
                shot_stats = candidates[0] if candidates else None
            if shot_stats is None and pname:
                candidates = [v for (_, candidate_name), v in shot_map.items() if candidate_name == pname]
                shot_stats = candidates[0] if candidates else None
            shot_stats = shot_stats or {"shots": 0, "shots_on_target": 0}
            team = r["team"]
            opponent = away if team == home else home if team == away else ""
            out.append({
                "fixture_id": match_id,
                "kickoff_utc": kickoff,
                "player_id": pid,
                "player": pname,
                "team": team,
                "opponent": opponent,
                "minutes": r["minutes"],
                "started": r["started"],
                "shots": shot_stats["shots"],
                "shots_on_target": shot_stats["shots_on_target"],
                "season_start": season,
            })

        if idx % 50 == 0:
            print(f"  {season}: processed {idx}/{len(matches)} matches")
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2021)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output", default=str(OUT))
    p.add_argument("--pause-seconds", type=float, default=0.05)
    args = p.parse_args()

    rows: list[dict] = []
    with UnderstatClient() as client:
        for season in range(args.start_season, args.end_season + 1):
            print(f"Fetching Understat EPL player data for {season}/{str(season + 1)[-2:]}")
            rows.extend(build_season(client, season, args.pause_seconds))

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise RuntimeError("Understat player ingestion returned no rows")
    df = df.drop_duplicates(subset=["fixture_id", "player_id", "player", "team"], keep="last")
    df = df.sort_values(["kickoff_utc", "fixture_id", "team", "player"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} player-match rows for {df['player'].nunique()} players to {out}")


if __name__ == "__main__":
    main()
