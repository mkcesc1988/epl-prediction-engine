from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest import _fit_platt, _logit, _sigmoid
from model_v11 import _derived_markets, _score_matrix
from model_v12 import _estimate_rho_from_history, _fit_strengths, build_predictions_v12
from pipeline import build_master, load_config, normalize_team, season_label


def _team_title(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("title") or value.get("name") or value.get("team") or "").strip()
    return str(value).strip()


def _num(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_current_understat(start_year: int) -> pd.DataFrame:
    """Fetch current-season EPL matches and fixtures from Understat."""
    with UnderstatClient() as client:
        matches = client.league(league="EPL").get_match_data(season=str(start_year))

    rows: list[dict] = []
    for m in matches or []:
        home = normalize_team(_team_title(m.get("h")))
        away = normalize_team(_team_title(m.get("a")))
        date = m.get("datetime") or m.get("date")
        if not home or not away or not date:
            continue

        goals = m.get("goals") or {}
        xg = m.get("xG") or m.get("xg") or {}
        rows.append({
            "Season": season_label(start_year),
            "Date": pd.to_datetime(date, errors="coerce").normalize(),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": _num(goals.get("h") if isinstance(goals, dict) else None),
            "FTAG": _num(goals.get("a") if isinstance(goals, dict) else None),
            "Home_xG": _num(xg.get("h") if isinstance(xg, dict) else None),
            "Away_xG": _num(xg.get("a") if isinstance(xg, dict) else None),
            "UnderstatMatchId": m.get("id"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No EPL {start_year}/{str(start_year + 1)[-2:]} schedule returned by Understat")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date").reset_index(drop=True)


def _build_training_history(cfg: dict, current: pd.DataFrame) -> pd.DataFrame:
    historical = build_master(cfg).copy()
    historical["Date"] = pd.to_datetime(historical["Date"])

    completed_current = current.dropna(subset=["FTHG", "FTAG", "Home_xG", "Away_xG"]).copy()
    if not completed_current.empty:
        completed_current["TotalGoals"] = completed_current["FTHG"] + completed_current["FTAG"]
        completed_current["Over2_5_Result"] = (completed_current["TotalGoals"] >= 3).astype("Int64")

    common_cols = sorted(set(historical.columns).union(completed_current.columns))
    historical = historical.reindex(columns=common_cols)
    completed_current = completed_current.reindex(columns=common_cols)

    history = pd.concat([historical, completed_current], ignore_index=True)
    history = history.drop_duplicates(["Season", "Date", "HomeTeam", "AwayTeam"], keep="last")
    return history.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def _fit_live_over25_calibrator(history: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    """Fit the live Over 2.5 calibration using only historical pre-match predictions."""
    preds = build_predictions_v12(history, cfg)
    train = preds.dropna(subset=["RawP_Over2_5_xG", "Over2_5_Result"]).copy()
    if len(train) < 200:
        return None
    return _fit_platt(train["RawP_Over2_5_xG"], train["Over2_5_Result"])


def _predict_fixture(row: pd.Series, fit, rho: float, cfg: dict, beta: np.ndarray | None) -> dict:
    mc = cfg.get("model_v12", {})
    floor = float(mc.get("lambda_floor", 0.15))
    cap = float(mc.get("lambda_cap", 4.5))
    max_goal = int(mc.get("max_goal", 10))

    h = str(row["HomeTeam"])
    a = str(row["AwayTeam"])
    ah = fit.attack.get(h, 0.0)
    dh = fit.defense.get(h, 0.0)
    aa = fit.attack.get(a, 0.0)
    da = fit.defense.get(a, 0.0)

    lam_h = math.exp(np.clip(fit.intercept + fit.home_advantage + ah - da, -4.0, 3.0))
    lam_a = math.exp(np.clip(fit.intercept + aa - dh, -4.0, 3.0))
    lam_h = max(floor, min(float(lam_h), cap))
    lam_a = max(floor, min(float(lam_a), cap))

    matrix = _score_matrix(lam_h, lam_a, rho, max_goal)
    markets = _derived_markets(matrix)
    raw_over = float(markets["P_Over_2_5_DC"])
    cal_over = raw_over if beta is None else _sigmoid(beta[0] + beta[1] * _logit(raw_over))

    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

    return {
        "Date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
        "Season": row["Season"],
        "HomeTeam": h,
        "AwayTeam": a,
        "ModelVersion": "V1.2",
        "Lambda_Home_xG": lam_h,
        "Lambda_Away_xG": lam_a,
        "Lambda_Total_xG": lam_h + lam_a,
        "DixonColes_Rho": rho,
        "P_HomeWin": markets["P_HomeWin_DC"],
        "P_Draw": markets["P_Draw_DC"],
        "P_AwayWin": markets["P_AwayWin_DC"],
        "P_BTTS_Yes": markets["P_BTTS_Yes_DC"],
        "P_BTTS_No": markets["P_BTTS_No_DC"],
        "RawP_Over0_5": markets["P_Over_0_5_DC"],
        "RawP_Over1_5": markets["P_Over_1_5_DC"],
        "RawP_Over2_5": raw_over,
        "CalP_Over2_5": cal_over,
        "CalP_Under2_5": 1.0 - cal_over,
        "FairOdds_Over2_5": 1.0 / cal_over,
        "FairOdds_Under2_5": 1.0 / (1.0 - cal_over),
        "RawP_Over3_5": markets["P_Over_3_5_DC"],
        "RawP_Over4_5": markets["P_Over_4_5_DC"],
        "MostLikelyScore": f"{best_idx[0]}-{best_idx[1]}",
        "HomeAttackRating": ah,
        "HomeDefenseRating": dh,
        "AwayAttackRating": aa,
        "AwayDefenseRating": da,
    }


def build_daily_predictions(cfg: dict) -> pd.DataFrame:
    daily = cfg.get("daily", {})
    current_season = int(daily.get("current_season", 2026))
    horizon_days = int(daily.get("horizon_days", 14))
    next_matchday_only = bool(daily.get("next_matchday_only", True))

    current = fetch_current_understat(current_season)
    history = _build_training_history(cfg, current)

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    fixtures = current[
        current["FTHG"].isna()
        & current["FTAG"].isna()
        & (current["Date"] >= today)
        & (current["Date"] <= today + pd.Timedelta(days=horizon_days))
    ].copy()

    if fixtures.empty:
        print("No upcoming EPL fixtures found inside the configured horizon.")
        return pd.DataFrame()

    if next_matchday_only:
        next_date = fixtures["Date"].min()
        fixtures = fixtures[fixtures["Date"] == next_date].copy()

    reference_date = fixtures["Date"].min()
    fit = _fit_strengths(history, reference_date, cfg)
    if fit is None:
        raise RuntimeError("V1.2 could not fit enough historical matches for daily predictions")

    rho = _estimate_rho_from_history(history, fit, cfg)
    beta = _fit_live_over25_calibrator(history, cfg)

    rows = [_predict_fixture(r, fit, rho, cfg, beta) for _, r in fixtures.iterrows()]
    return pd.DataFrame(rows).sort_values(["Date", "HomeTeam"]).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = build_daily_predictions(cfg)
    latest_path = out_dir / "daily_predictions_latest.csv"
    predictions.to_csv(latest_path, index=False)

    if predictions.empty:
        print(f"Saved empty daily prediction file: {latest_path}")
        return

    stamp = predictions["Date"].min().replace("-", "")
    dated_path = out_dir / f"daily_predictions_{stamp}.csv"
    predictions.to_csv(dated_path, index=False)

    pd.set_option("display.max_columns", None)
    print(predictions.to_string(index=False))
    print(f"\nLatest: {latest_path}")
    print(f"Dated:  {dated_path}")


if __name__ == "__main__":
    main()
