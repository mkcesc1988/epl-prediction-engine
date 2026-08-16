from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import _fit_platt, _logit, _sigmoid
from daily_predictions import fetch_current_understat_completed, _build_training_history, _fit_live_over25_calibrator
from model_v11 import _derived_markets, _score_matrix
from model_v12 import _estimate_rho_from_history, _fit_strengths
from pipeline import load_config, normalize_team


def predict_one_off(home_team: str, away_team: str, match_date: str, neutral: bool) -> pd.DataFrame:
    cfg = load_config()
    daily = cfg.get("daily", {})
    current_season = int(daily.get("current_season", 2026))

    home = normalize_team(home_team)
    away = normalize_team(away_team)
    date = pd.Timestamp(match_date)

    current_completed = fetch_current_understat_completed(current_season)
    history = _build_training_history(cfg, current_completed)

    fit = _fit_strengths(history, date, cfg)
    if fit is None:
        raise RuntimeError("V1.2 could not fit enough historical matches for one-off prediction")

    rho = _estimate_rho_from_history(history, fit, cfg)
    beta = _fit_live_over25_calibrator(history, cfg)

    mc = cfg.get("model_v12", {})
    floor = float(mc.get("lambda_floor", 0.15))
    cap = float(mc.get("lambda_cap", 4.5))
    max_goal = int(mc.get("max_goal", 10))

    ah = fit.attack.get(home, 0.0)
    dh = fit.defense.get(home, 0.0)
    aa = fit.attack.get(away, 0.0)
    da = fit.defense.get(away, 0.0)

    home_adv = 0.0 if neutral else fit.home_advantage
    lam_h = math.exp(np.clip(fit.intercept + home_adv + ah - da, -4.0, 3.0))
    lam_a = math.exp(np.clip(fit.intercept + aa - dh, -4.0, 3.0))
    lam_h = max(floor, min(float(lam_h), cap))
    lam_a = max(floor, min(float(lam_a), cap))

    matrix = _score_matrix(lam_h, lam_a, rho, max_goal)
    markets = _derived_markets(matrix)
    raw_over = float(markets["P_Over_2_5_DC"])
    cal_over = raw_over if beta is None else _sigmoid(beta[0] + beta[1] * _logit(raw_over))
    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

    row = {
        "Date": date.strftime("%Y-%m-%d"),
        "HomeTeam": home,
        "AwayTeam": away,
        "NeutralVenue": bool(neutral),
        "ModelVersion": "V1.2",
        "Lambda_Home_xG": lam_h,
        "Lambda_Away_xG": lam_a,
        "Lambda_Total_xG": lam_h + lam_a,
        "P_HomeWin": markets["P_HomeWin_DC"],
        "P_Draw": markets["P_Draw_DC"],
        "P_AwayWin": markets["P_AwayWin_DC"],
        "P_BTTS_Yes": markets["P_BTTS_Yes_DC"],
        "CalP_Over2_5": cal_over,
        "CalP_Under2_5": 1.0 - cal_over,
        "FairOdds_Over2_5": 1.0 / cal_over,
        "FairOdds_Under2_5": 1.0 / (1.0 - cal_over),
        "MostLikelyScore": f"{best_idx[0]}-{best_idx[1]}",
        "HomeAttackRating": ah,
        "HomeDefenseRating": dh,
        "AwayAttackRating": aa,
        "AwayDefenseRating": da,
    }
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--neutral", action="store_true")
    args = parser.parse_args()

    out = predict_one_off(args.home, args.away, args.date, args.neutral)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "one_off_prediction_latest.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
