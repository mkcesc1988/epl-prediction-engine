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


V2_VERSION = "V2-shadow.4"


def _fetch_preseason_context(cfg: dict, season: int) -> pd.DataFrame:
    vcfg = cfg.get("v2_shadow", {})
    if not bool(vcfg.get("preseason_enabled", True)) or not os.environ.get("API_FOOTBALL_KEY"):
        return pd.DataFrame()

    try:
        payload = api_get(
            "fixtures",
            {
                "league": int(vcfg.get("preseason_league_id", 667)),
                "season": season,
                "from": f"{season}-{vcfg.get('preseason_start_month_day', '06-15')}",
                "to": f"{season}-{vcfg.get('preseason_end_month_day', '08-20')}",
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


def _build_power_map(shadow: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, list[float]] = {}
    for _, r in shadow.iterrows():
        home = str(r.get("HomeTeam", ""))
        away = str(r.get("AwayTeam", ""))
        if home:
            rows.setdefault(home, []).append(float(r.get("HomeAttackRating", 0.0)) + float(r.get("HomeDefenseRating", 0.0)))
        if away:
            rows.setdefault(away, []).append(float(r.get("AwayAttackRating", 0.0)) + float(r.get("AwayDefenseRating", 0.0)))
    return {team: float(np.mean(vals)) for team, vals in rows.items() if vals}


def _opponent_weight(opponent: str, power_map: dict[str, float], cfg: dict) -> tuple[float, bool]:
    vcfg = cfg.get("v2_shadow", {})
    if opponent not in power_map or not power_map:
        return float(vcfg.get("preseason_unknown_opponent_weight", 0.55)), False

    powers = np.array(list(power_map.values()), dtype=float)
    center = float(np.median(powers))
    scale = float(vcfg.get("preseason_opponent_power_scale", 0.35))
    lo = float(vcfg.get("preseason_known_opponent_weight_min", 0.75))
    hi = float(vcfg.get("preseason_known_opponent_weight_max", 1.25))
    weight = math.exp(scale * (float(power_map[opponent]) - center))
    return float(np.clip(weight, lo, hi)), True


def _context_features(history: pd.DataFrame, team: str, fixture_date: pd.Timestamp, cfg: dict) -> dict:
    window_days = int(cfg.get("v2_shadow", {}).get("congestion_window_days", 14))
    matches = _team_matches(history, team, fixture_date)
    if matches.empty:
        return {"rest_days": np.nan, "matches_in_window": 0, "extra_congestion_matches": 0}

    dates = pd.to_datetime(matches["Date"], errors="coerce").dropna().sort_values()
    rest_days = max(0.0, float((fixture_date - dates.iloc[-1]).days))
    window_start = fixture_date - pd.Timedelta(days=window_days)
    matches_in_window = int(((dates >= window_start) & (dates < fixture_date)).sum())
    free_matches = int(cfg.get("v2_shadow", {}).get("congestion_free_matches", 2))
    return {
        "rest_days": rest_days,
        "matches_in_window": matches_in_window,
        "extra_congestion_matches": max(0, matches_in_window - free_matches),
    }


def _preseason_form(
    history: pd.DataFrame,
    team: str,
    fixture_date: pd.Timestamp,
    cfg: dict,
    power_map: dict[str, float],
) -> dict:
    vcfg = cfg.get("v2_shadow", {})
    n = int(vcfg.get("preseason_form_matches", 5))
    matches = _team_matches(history, team, fixture_date)
    if matches.empty or "ContextSource" not in matches.columns:
        return {"matches": 0, "raw_avg_gd": 0.0, "weighted_avg_gd": 0.0, "avg_opp_weight": 0.0, "known_opp_share": 0.0}

    matches = matches[matches["ContextSource"].astype(str).str.contains("Friendlies", na=False)].tail(n)
    if matches.empty:
        return {"matches": 0, "raw_avg_gd": 0.0, "weighted_avg_gd": 0.0, "avg_opp_weight": 0.0, "known_opp_share": 0.0}

    gd_cap = float(vcfg.get("preseason_goal_diff_cap", 3.0))
    metric_cap = float(vcfg.get("preseason_form_metric_cap", 2.0))
    raw_gds: list[float] = []
    weighted_gds: list[float] = []
    weights: list[float] = []
    known_flags: list[float] = []

    for _, r in matches.iterrows():
        is_home = str(r.get("HomeTeam")) == team
        opponent = str(r.get("AwayTeam")) if is_home else str(r.get("HomeTeam"))
        gd = (float(r.get("FTHG", 0.0)) - float(r.get("FTAG", 0.0))) if is_home else (float(r.get("FTAG", 0.0)) - float(r.get("FTHG", 0.0)))
        capped_gd = float(np.clip(gd, -gd_cap, gd_cap))
        weight, known = _opponent_weight(opponent, power_map, cfg)
        raw_gds.append(capped_gd)
        weighted_gds.append(capped_gd * weight)
        weights.append(weight)
        known_flags.append(1.0 if known else 0.0)

    weighted_avg = float(np.sum(weighted_gds) / np.sum(weights)) if np.sum(weights) > 0 else 0.0
    return {
        "matches": len(raw_gds),
        "raw_avg_gd": float(np.mean(raw_gds)),
        "weighted_avg_gd": float(np.clip(weighted_avg, -metric_cap, metric_cap)),
        "avg_opp_weight": float(np.mean(weights)),
        "known_opp_share": float(np.mean(known_flags)),
    }


def _effective_rest(rest_days: float, cfg: dict) -> float:
    vcfg = cfg.get("v2_shadow", {})
    if pd.isna(rest_days):
        return float(vcfg.get("rest_day_cap", 10))
    return float(np.clip(rest_days, float(vcfg.get("rest_day_floor", 3)), float(vcfg.get("rest_day_cap", 10))))


def _apply_context_adjustment(row: pd.Series, history: pd.DataFrame, cfg: dict, power_map: dict[str, float]) -> dict:
    fixture_date = pd.Timestamp(row["Date"])
    home = str(row["HomeTeam"])
    away = str(row["AwayTeam"])

    home_ctx = _context_features(history, home, fixture_date, cfg)
    away_ctx = _context_features(history, away, fixture_date, cfg)
    home_form = _preseason_form(history, home, fixture_date, cfg, power_map)
    away_form = _preseason_form(history, away, fixture_date, cfg, power_map)

    vcfg = cfg.get("v2_shadow", {})
    rest_component = float(vcfg.get("rest_log_lambda_per_day", 0.015)) * (
        _effective_rest(home_ctx["rest_days"], cfg) - _effective_rest(away_ctx["rest_days"], cfg)
    )
    congestion_component = float(vcfg.get("congestion_log_lambda_per_extra_match", 0.020)) * (
        away_ctx["extra_congestion_matches"] - home_ctx["extra_congestion_matches"]
    )
    form_component = float(np.clip(
        float(vcfg.get("preseason_goal_diff_log_lambda_per_goal", 0.018))
        * (home_form["weighted_avg_gd"] - away_form["weighted_avg_gd"]),
        -float(vcfg.get("max_preseason_form_log_adjustment", 0.06)),
        float(vcfg.get("max_preseason_form_log_adjustment", 0.06)),
    ))
    log_adjustment = float(np.clip(
        rest_component + congestion_component + form_component,
        -float(vcfg.get("max_context_log_adjustment", 0.10)),
        float(vcfg.get("max_context_log_adjustment", 0.10)),
    ))

    base_home = float(row["Lambda_Home_xG"])
    base_away = float(row["Lambda_Away_xG"])
    v2_home = base_home * math.exp(log_adjustment)
    v2_away = base_away * math.exp(-log_adjustment)

    matrix = _score_matrix(v2_home, v2_away, float(row.get("DixonColes_Rho", -0.08)), int(cfg.get("model_v12", {}).get("max_goal", 10)))
    markets = _derived_markets(matrix)
    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

    return {
        "HomeRestDays": home_ctx["rest_days"],
        "AwayRestDays": away_ctx["rest_days"],
        "HomeMatchesLast14d": home_ctx["matches_in_window"],
        "AwayMatchesLast14d": away_ctx["matches_in_window"],
        "HomeExtraCongestionMatches": home_ctx["extra_congestion_matches"],
        "AwayExtraCongestionMatches": away_ctx["extra_congestion_matches"],
        "HomePreseasonMatches": home_form["matches"],
        "AwayPreseasonMatches": away_form["matches"],
        "HomePreseasonRawAvgGoalDiff": home_form["raw_avg_gd"],
        "AwayPreseasonRawAvgGoalDiff": away_form["raw_avg_gd"],
        "HomePreseasonAdjGoalDiff": home_form["weighted_avg_gd"],
        "AwayPreseasonAdjGoalDiff": away_form["weighted_avg_gd"],
        "HomePreseasonAvgOpponentWeight": home_form["avg_opp_weight"],
        "AwayPreseasonAvgOpponentWeight": away_form["avg_opp_weight"],
        "HomePreseasonKnownOpponentShare": home_form["known_opp_share"],
        "AwayPreseasonKnownOpponentShare": away_form["known_opp_share"],
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
        history = pd.concat([history, preseason], ignore_index=True, sort=False).sort_values("Date").reset_index(drop=True)
        print(f"Loaded {len(preseason)} completed preseason/friendly matches for V2 context.")
    else:
        print("No preseason/friendly context loaded. V2 will fall back to league-only context.")

    power_map = _build_power_map(shadow)
    adjustments = pd.DataFrame([_apply_context_adjustment(row, history, cfg, power_map) for _, row in shadow.iterrows()])
    shadow = pd.concat([shadow.reset_index(drop=True), adjustments], axis=1)
    shadow["BaselineModelVersion"] = shadow.get("ModelVersion", "V1.2")
    shadow["ModelVersion"] = V2_VERSION
    shadow["ShadowMode"] = True
    shadow["V2ValidationStatus"] = "Experimental opponent-adjusted preseason + rest/congestion context; not used for production bets"
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
            "HomePreseasonMatches", "AwayPreseasonMatches", "HomePreseasonRawAvgGoalDiff", "AwayPreseasonRawAvgGoalDiff",
            "HomePreseasonAdjGoalDiff", "AwayPreseasonAdjGoalDiff", "HomePreseasonAvgOpponentWeight", "AwayPreseasonAvgOpponentWeight",
            "ContextLogAdjustment", "P_HomeWin", "P_Draw", "P_AwayWin", "V2_P_HomeWin", "V2_P_Draw", "V2_P_AwayWin",
        ]
        print(shadow[[c for c in cols if c in shadow.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
