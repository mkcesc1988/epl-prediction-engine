from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yaml
from underdata.league import League
from underdata.team import Team

TEAM_MAP = {
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Norwich City": "Norwich",
}

FD_BASE = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"


def normalize_team(name: object) -> str:
    value = str(name).strip()
    return TEAM_MAP.get(value, value)


def season_label(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def season_code(start_year: int) -> str:
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def ensure_dirs(cfg: dict) -> None:
    Path(cfg["paths"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)


def fetch_football_data(start_year: int, division: str, raw_dir: str) -> pd.DataFrame:
    url = FD_BASE.format(code=season_code(start_year), division=division)
    response = requests.get(url, timeout=45)
    response.raise_for_status()

    out = Path(raw_dir) / f"football_data_{division}_{season_code(start_year)}.csv"
    out.write_bytes(response.content)

    df = pd.read_csv(out)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Football-Data missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise").dt.normalize()
    df["HomeTeam"] = df["HomeTeam"].map(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].map(normalize_team)

    wanted = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "B365>2.5", "B365<2.5", "P>2.5", "P<2.5", "PC>2.5", "PC<2.5",
    ]
    wanted = [c for c in wanted if c in df.columns]
    df = df[wanted].copy()
    df = df.rename(columns={
        "B365>2.5": "B365_Over2_5",
        "B365<2.5": "B365_Under2_5",
        "P>2.5": "Pinnacle_Over2_5",
        "P<2.5": "Pinnacle_Under2_5",
        "PC>2.5": "Pinnacle_Close_Over2_5",
        "PC<2.5": "Pinnacle_Close_Under2_5",
    })
    df.insert(0, "Season", season_label(start_year))
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["Over2_5_Result"] = (df["TotalGoals"] >= 3).astype("Int64")
    return df


def _pick_column(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    if required:
        raise RuntimeError(f"Could not find any of columns {list(aliases)}. Available: {list(df.columns)}")
    return None


def _team_names_from_league(start_year: int) -> list[str]:
    league = League(league_name="EPL", season=start_year)
    teams = league.get_teams()
    if teams is None or len(teams) == 0:
        raise RuntimeError(f"Understat returned no EPL teams for {start_year}")

    if not isinstance(teams, pd.DataFrame):
        teams = pd.DataFrame(teams)

    candidates = ["title", "team", "team_name", "equipo", "name", "Team"]
    for candidate in candidates:
        matches = [c for c in teams.columns if str(c).strip().lower() == candidate.lower()]
        if matches:
            vals = teams[matches[0]].dropna().astype(str).str.strip().tolist()
            if vals:
                return list(dict.fromkeys(vals))

    if teams.index.dtype == object:
        vals = [str(x).strip() for x in teams.index if str(x).strip()]
        if vals:
            return list(dict.fromkeys(vals))

    # Last-resort heuristic: choose the first mostly-string column.
    for c in teams.columns:
        s = teams[c].dropna()
        if len(s) and s.map(lambda x: isinstance(x, str)).mean() > 0.8:
            vals = s.astype(str).str.strip().tolist()
            return list(dict.fromkeys(vals))

    raise RuntimeError(f"Could not infer team names from Understat columns: {list(teams.columns)}")


def fetch_understat_xg(start_year: int, raw_dir: str) -> pd.DataFrame:
    team_names = _team_names_from_league(start_year)
    frames: list[pd.DataFrame] = []

    for idx, team_name in enumerate(team_names, start=1):
        print(f"  Understat {season_label(start_year)} [{idx:02d}/{len(team_names)}] {team_name}")
        history = Team(team_name=team_name, season=start_year).get_match_history()
        if history is None or len(history) == 0:
            continue
        if not isinstance(history, pd.DataFrame):
            history = pd.DataFrame(history)
        frames.append(history.copy())

    if not frames:
        raise RuntimeError(f"Understat returned no match histories for {start_year}")

    raw = pd.concat(frames, ignore_index=True)

    date_c = _pick_column(raw, ["Fecha", "Date", "date"])
    home_c = _pick_column(raw, ["Local", "HomeTeam", "Home", "home_team"])
    away_c = _pick_column(raw, ["Visitante", "AwayTeam", "Away", "away_team"])
    hg_c = _pick_column(raw, ["Goles Local", "FTHG", "Home Goals", "home_goals"])
    ag_c = _pick_column(raw, ["Goles Visitante", "FTAG", "Away Goals", "away_goals"])
    hxg_c = _pick_column(raw, ["xG Local", "Home_xG", "Home xG", "home_xg"])
    axg_c = _pick_column(raw, ["xG Visitante", "Away_xG", "Away xG", "away_xg"])

    df = pd.DataFrame({
        "Season": season_label(start_year),
        "Date": pd.to_datetime(raw[date_c], errors="coerce").dt.normalize(),
        "HomeTeam": raw[home_c].map(normalize_team),
        "AwayTeam": raw[away_c].map(normalize_team),
        "Understat_FTHG": pd.to_numeric(raw[hg_c], errors="coerce"),
        "Understat_FTAG": pd.to_numeric(raw[ag_c], errors="coerce"),
        "Home_xG": pd.to_numeric(raw[hxg_c], errors="coerce"),
        "Away_xG": pd.to_numeric(raw[axg_c], errors="coerce"),
    })

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "Home_xG", "Away_xG"])
    df = (
        df.sort_values(["Date", "HomeTeam", "AwayTeam"])
        .drop_duplicates(["Season", "Date", "HomeTeam", "AwayTeam"])
        .reset_index(drop=True)
    )

    out = Path(raw_dir) / f"understat_EPL_{season_code(start_year)}.csv"
    df.to_csv(out, index=False)
    return df


def build_master(cfg: dict) -> pd.DataFrame:
    ensure_dirs(cfg)
    fd_frames = []
    xg_frames = []

    for year in range(int(cfg["start_season"]), int(cfg["end_season"]) + 1):
        print(f"Football-Data {season_label(year)}")
        fd_frames.append(fetch_football_data(year, cfg["football_data_division"], cfg["paths"]["raw_dir"]))

        print(f"Understat {season_label(year)}")
        xg_frames.append(fetch_understat_xg(year, cfg["paths"]["raw_dir"]))

    fd = pd.concat(fd_frames, ignore_index=True)
    xg = pd.concat(xg_frames, ignore_index=True)
    key = ["Season", "Date", "HomeTeam", "AwayTeam"]

    if fd.duplicated(key).any():
        raise RuntimeError("Duplicate Football-Data match keys detected")
    if xg.duplicated(key).any():
        raise RuntimeError("Duplicate Understat match keys detected")

    master = fd.merge(xg, on=key, how="left", validate="one_to_one")
    master["score_source_mismatch"] = (
        master["Understat_FTHG"].notna()
        & master["Understat_FTAG"].notna()
        & ((master["FTHG"] != master["Understat_FTHG"]) | (master["FTAG"] != master["Understat_FTAG"]))
    ).astype(int)

    master = master.sort_values(key).reset_index(drop=True)
    master.to_csv(cfg["paths"]["master"], index=False)

    report = {
        "rows": int(len(master)),
        "duplicates": int(master.duplicated(key).sum()),
        "missing_scores": int((master["FTHG"].isna() | master["FTAG"].isna()).sum()),
        "xg_complete": int((master["Home_xG"].notna() & master["Away_xG"].notna()).sum()),
        "missing_xg": int((master["Home_xG"].isna() | master["Away_xG"].isna()).sum()),
        "score_mismatches": int(master["score_source_mismatch"].sum()),
        "by_season": {},
    }
    for season, g in master.groupby("Season"):
        report["by_season"][str(season)] = {
            "matches": int(len(g)),
            "xg_complete": int((g["Home_xG"].notna() & g["Away_xG"].notna()).sum()),
            "missing_xg": int((g["Home_xG"].isna() | g["Away_xG"].isna()).sum()),
        }

    Path(cfg["paths"]["validation"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return master


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    config = load_config()
    result = build_master(config)
    print(f"Built {len(result)} master matches -> {config['paths']['master']}")
