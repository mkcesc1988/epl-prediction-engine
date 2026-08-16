from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from api_football_diagnostic import api_get, normalize_name
from daily_predictions import (
    _build_training_history,
    build_daily_predictions,
    fetch_current_understat_completed,
)
from model_v11 import _derived_markets, _score_matrix
from pipeline import load_config


V2_VERSION = "V2-shadow.3"


def _fetch_preseason_context(cfg: dict, season: int) -> pd.DataFrame:
    vcfg = cfg.get("v2_shadow", {})
    if not bool(vcfg.get("preseason_enabled", True)):
        return pd.DataFrame()
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("API_FOOTBALL_KEY unavailable, continuing without preseason context.")
        return pd.DataFrame()

    league_id = int(vcfg.get("preseason_league_id", 667))
    start_md = str(vcfg.get("preseason_start_month_day", "06-15"))
    end_md = str(vcfg.get("preseason_end_month_day", "08-20"))

    try:
        payload = api_get(
            "fixtures",
            {
                "league": league_id,
                "season": season,
                "from": f"{season}-{start_md}",
                "to": f"{season}-{end_md}",
            },
        )
    except Exception as exc:
        print(f"Preseason API context unavailable: {exc}")
        return pd.DataFrame()

    rows = []
    for item in payload.get("response", []):
        status = str(item.get("fixture", {}).get("status", {}).get("short", ""))
        if status not in {"FT", "AET", "PEN"}:
            continue
        home = normalize_name(str(item.get("teams", {}).get("home", {}).get("name", "")))
        away = normalize_name(str(item.get("teams", {}).get("away", {}).get("name", "")))
        date = pd.to_datetime(item.get("fixture", {}).get("date"), utc=True, errors="coerce")
        hg = pd.to_numeric(item.get("goals", {}).get("home"), errors="coerce")
        ag = pd.to_numeric(item.get("goals", {}).get("away"), errors="coerce")
        if not home or not away or pd.isna(date) or pd.isna(hg) or pd.isna(ag):
            continue
        rows.append({
            "Date": date.tz_convert(None).normalize(),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": float(hg),
            "FTAG": float(ag),
            "ContextSource": "API-Football Club Friendlies",
        })
    return pd.DataFrame(rows)


def _team_matches(history: pd.DataFrame, team: str, before: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    dates = pd.to_datetime(history.get("Date"), errors="coerce")
    mask = (dates < before) & (
        history.get("HomeTeam", pd.Series(index=history.index, dtype=str)).astype(str).eq(team)
        | history.get("AwayTeam", pd.Series(index=history.index, dtype=str)).astype(str).eq(team)
    )
    out = history.loc[mask].copy()
    out["Date"] = dates.loc[mask]
    return out.sort_values("Date")


def _context_features(history: pd.DataFrame, team: str, fixture_date: pd.Timestamp, cfg: dict) -> dict:
    vcfg = cfg.get("v2_shadow", {})
    window_days = int(vcfg.get("congestion_window_days", 14))
    matches = _team_matches(history, team, fixture_date)

    if matches.empty:
        return {"rest_days": np.nan, "matches_in_window": 0, "extra_congestion_matches": 0}

    dates = pd.to_datetime(matches["Date"], errors="coerce").dropna().sort_values()
    rest_days = max(0.0, float((fixture_date - dates.iloc[-1]).days))
    window_start = fixture_date - pd.Timedelta(days=window_days)
    matches_in_window = int(((dates >= window_start) & (dates < fixture_date)).sum())
    free_matches = int(vcfg.get("congestion_free_matches", 2))
    extra = max(0, matches_in_window - free_matches)
    return {"rest_days": rest_days, "matches_in_window": matches_in_window, "extra_congestion_matches": extra}


def _preseason_form(history: pd.DataFrame, team: str, fixture_date: pd.Timestamp, cfg: dict) -> tuple[int, float]:
    vcfg = cfg.get("v2_shadow", {})
    n = int(vcfg.get("preseason_form_matches", 5))
    matches = _team_matches(history, team, fixture_date)
    if matches.empty:
        return 0, 0.0
    if "ContextSource" in matches.columns:
        matches = matches[matches["ContextSource"].astype(str).str.contains("Friendlies", na=False)]
    else:
        return 0, 0.0
    matches = matches.tail(n)
    if matches.empty:
        return 0, 0.0

    gds = []
    for _, r in matches.iterrows():
        if str(r.get("HomeTeam")) == team:
            gd = float(r.get("FTHG", 0.0)) - float(r.get("FTAG", 0.0))
        else:
            gd = float(r.get("FTAG", 0.0)) - float(r.get("FTHG", 0.0))
        gds.append(gd)
    return len(gds), float(np.mean(gds)) if gds else 0.0


def _effective_rest(rest_days: float, cfg: dict) -> float:
    vcfg = cfg.get("v2_shadow", {})
    floor = float(vcfg.get("rest_day_floor", 3))
    cap = float(vcfg.get("rest_day_cap", 10))
    if pd.isna(rest_days):
        return cap
    return float(np.clip(rest_days, floor, cap))


def _apply_context_adjustment(row: pd.Series, history: pd.DataFrame, cfg: dict) -> dict:
    fixture_date = pd.Timestamp(row["Date"])
    home = str(row["HomeTeam"])
    away = str(row["AwayTeam"])

    home_ctx = _context_features(history, home, fixture_date, cfg)
    away_ctx = _context_features(history, away, fixture_date, cfg)
    home_rest = _effective_rest(home_ctx["rest_days"], cfg)
    away_rest = _effective_rest(away_ctx["rest_days"], cfg)

    home_form_n, home_form_gd = _preseason_form(history, home, fixture_date, cfg)
    away_form_n, away_form_gd = _preseason_form(history, away, fixture_date, cfg)

    vcfg = cfg.get("v2_shadow", {})
    rest_coeff = float(vcfg.get("rest_log_lambda_per_day", 0.015))
    congestion_coeff = float(vcfg.get("congestion_log_lambda_per_extra_match", 0.020))
    form_coeff = float(vcfg.get("preseason_goal_diff_log_lambda_per_goal", 0.018))
    max_context = float(vcfg.get("max_context_log_adjustment", 0.10))
    max_form = float(vcfg.get("max_preseason_form_log_adjustment", 0.06))

    rest_component = rest_coeff * (home_rest - away_rest)
    congestion_component = congestion_coeff * (
        away_ctx["extra_congestion_matches"] - home_ctx["extra_congestion_matches"]
    )
    form_component = float(np.clip(form_coeff * (home_form_gd - away_form_gd), -max_form, max_form))
    log_adjustment = float(np.clip(rest_component + congestion_component + form_component, -max_context, max_context))

    base_home = float(row["Lambda_Home_xG"])
    base_away = float(row["Lambda_Away_xG"])
    v2_home = base_home * math.exp(log_adjustment)
    v2_away = base_away * math.exp(-log_adjustment)

    rho = float(row.get("DixonColes_Rho", -0.08))
    max_goal = int(cfg.get("model_v12", {}).get("max_goal", 10))
    matrix = _score_matrix(v2_home, v2_away, rho, max_goal)
    markets = _derived_markets(matrix)
    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

    return {
        "HomeRestDays": home_ctx["rest_days"],
        "AwayRestDays": away_ctx["rest_days"],
        "HomeMatchesLast14d": home_ctx["matches_in_window"],
        "AwayMatchesLast14d": away_ctx["matches_in_window"],
        "HomeExtraCongestionMatches": home_ctx["extra_congestion_matches"],
        "AwayExtraCongestionMatches": away_ctx["extra_congestion_matches"],
        "HomePreseasonMatches": home_form_n,
        "AwayPreseasonMatches": away_form_n,
        "HomePreseasonAvgGoalDiff": home_form_gd,
        "AwayPreseasonAvgGoalDiff": away_form_gd,
        "RestComponentLogAdj": rest_component,
        "CongestionComponentLogAdj": congestion_component,
        "PreseasonFormLogAdj": form_component,
        "ContextLogAdjustment": log_adjustment,
        "V2_Lambda_Home_xG": v2_home,
        "V2_Lambda_Away_xG": v2_away,
        "V2_Lambda_Total_xG": v2_home + v2_away,
        "V2_P_HomeWin": markets["P_HomeWin_DC"],
        "V2_P_Draw": markets["P_Draw_DC"],
        "V2_P_AwayWin": markets["P_AwayWin_DC"],
        "V2_P_BTTS_Yes": markets["P_BTTS_Yes_DC"],
        "V2_P_Over2_5_Raw": markets["P_Over_2_5_DC"],
        "V2_P_Under2_5_Raw": 1.0 - float(markets["P_Over_2_5_DC"]),
        "V2_MostLikelyScore": f"{best_idx[0]}-{best_idx[1]}",
        "V2ContextAdjustmentApplied": abs(log_adjustment) > 1e-12,
    }


def build_v2_shadow_predictions(cfg: dict) -> pd.DataFrame:
    shadow = build_daily_predictions(cfg).copy()
    if shadow.empty:
        return shadow

    current_season = int(cfg.get("daily", {}).get("current_season", 2026))
    current_completed = fetch_current_understat_completed(current_season)
    history = _build_training_history(cfg, current_completed)

    preseason = _fetch_preseason_context(cfg, current_season)
    if not preseason.empty:
        history = pd.concat([history, preseason], ignore_index=True, sort=False)
        history = history.sort_values("Date").reset_index(drop=True)
        print(f"Loaded {len(preseason)} completed preseason/friendly matches for V2 context.")
    else:
        print("No preseason/friendly context loaded. V2 will fall back to league-only context.")

    adjustments = pd.DataFrame([_apply_context_adjustment(row, history, cfg) for _, row in shadow.iterrows()])
    shadow = pd.concat([shadow.reset_index(drop=True), adjustments], axis=1)
    shadow["BaselineModelVersion"] = shadow.get("ModelVersion", "V1.2")
    shadow["ModelVersion"] = V2_VERSION
    shadow["ShadowMode"] = True
    shadow["V2ValidationStatus"] = "Experimental preseason + rest/congestion context; not used for production bets"
    return shadow


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    shadow = build_v2_shadow_predictions(cfg)
    latest_path = out_dir / "v2_shadow_predictions_latest.csv"
    shadow.to_csv(latest_path, index=False)

    print(f"Saved V2 shadow predictions: {latest_path}")
    print("V1.2 production predictions and bet tracking were not modified.")
    if not shadow.empty:
        cols = [
            "Date", "HomeTeam", "AwayTeam", "HomeRestDays", "AwayRestDays",
            "HomeMatchesLast14d", "AwayMatchesLast14d", "HomePreseasonMatches", "AwayPreseasonMatches",
            "HomePreseasonAvgGoalDiff", "AwayPreseasonAvgGoalDiff", "ContextLogAdjustment",
            "Lambda_Home_xG", "Lambda_Away_xG", "V2_Lambda_Home_xG", "V2_Lambda_Away_xG",
            "P_HomeWin", "P_Draw", "P_AwayWin", "V2_P_HomeWin", "V2_P_Draw", "V2_P_AwayWin",
        ]
        print(shadow[[c for c in cols if c in shadow.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
