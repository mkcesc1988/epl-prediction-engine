from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import yaml
from understatapi import UnderstatClient

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


def _team_title(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("title") or value.get("name") or value.get("team") or "").strip()
    return str(value).strip()


def _num(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_understat_xg(start_year: int, raw_dir: str) -> pd.DataFrame:
    """Download all EPL fixtures for a season directly from Understat's league endpoint."""
    print(f"  Understat {season_label(start_year)} league match data")
    with UnderstatClient() as client:
        matches = client.league(league="EPL").get_match_data(season=str(start_year))

    if not matches:
        raise RuntimeError(f"Understat returned no EPL match data for {start_year}")

    rows = []
    for m in matches:
        home = _team_title(m.get("h"))
        away = _team_title(m.get("a"))
        date = m.get("datetime") or m.get("date")
        goals = m.get("goals") or {}
        xg = m.get("xG") or m.get("xg") or {}

        if not home or not away or not date:
            continue

        rows.append({
            "Season": season_label(start_year),
            "Date": pd.to_datetime(date, errors="coerce").normalize(),
            "HomeTeam": normalize_team(home),
            "AwayTeam": normalize_team(away),
            "Understat_FTHG": _num(goals.get("h") if isinstance(goals, dict) else None),
            "Understat_FTAG": _num(goals.get("a") if isinstance(goals, dict) else None),
            "Home_xG": _num(xg.get("h") if isinstance(xg, dict) else None),
            "Away_xG": _num(xg.get("a") if isinstance(xg, dict) else None),
            "UnderstatMatchId": m.get("id"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Understat response for {start_year} could not be parsed")

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
