from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    z = max(min(float(z), 30), -30)
    return 1 / (1 + math.exp(-z))


def _fit_platt(probabilities, outcomes) -> np.ndarray:
    X = np.array([[1.0, _logit(p)] for p in probabilities], dtype=float)
    y = np.array(outcomes, dtype=float)
    beta = np.zeros(2)

    for _ in range(50):
        z = X @ beta
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-8)
        grad = X.T @ (y - p)
        hessian = -(X.T * w) @ X
        step = np.linalg.solve(hessian, grad)
        new_beta = beta - step
        if np.max(np.abs(new_beta - beta)) < 1e-9:
            beta = new_beta
            break
        beta = new_beta
    return beta


def brier_score(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _market_odds(row):
    pairs = [
        ("Pinnacle_Close_Over2_5", "Pinnacle_Close_Under2_5", "Pinnacle Close"),
        ("Pinnacle_Over2_5", "Pinnacle_Under2_5", "Pinnacle"),
        ("B365_Over2_5", "B365_Under2_5", "Bet365"),
    ]
    for over_col, under_col, name in pairs:
        try:
            o = float(row[over_col])
            u = float(row[under_col])
            if o > 1 and u > 1:
                return o, u, name
        except Exception:
            pass
    return None, None, None


def walk_forward_backtest(predictions: pd.DataFrame, cfg: dict):
    df = predictions.dropna(subset=["RawP_Over2_5_xG", "Over2_5_Result"]).copy()
    seasons = list(dict.fromkeys(df["Season"].astype(str).tolist()))
    min_prior = int(cfg["backtest"]["min_prior_seasons_for_calibration"])
    thresholds = [float(x) for x in cfg["backtest"]["ev_thresholds"]]

    evaluated_frames = []
    summaries = []

    for idx in range(min_prior, len(seasons)):
        target = seasons[idx]
        prior = seasons[:idx]
        train = df[df["Season"].astype(str).isin(prior)]
        test = df[df["Season"].astype(str) == target].copy()
        if train.empty or test.empty:
            continue

        beta = _fit_platt(train["RawP_Over2_5_xG"], train["Over2_5_Result"])
        test["CalP_Over2_5_xG"] = [
            _sigmoid(beta[0] + beta[1] * _logit(p))
            for p in test["RawP_Over2_5_xG"]
        ]
        test["CalP_Under2_5_xG"] = 1 - test["CalP_Over2_5_xG"]
        test["TrainThrough"] = prior[-1]
        evaluated_frames.append(test)

        result = {
            "Season": target,
            "Matches": int(len(test)),
            "Brier": brier_score(test["CalP_Over2_5_xG"], test["Over2_5_Result"]),
            "LogLoss": log_loss(test["CalP_Over2_5_xG"], test["Over2_5_Result"]),
            "AveragePredictedOver": float(test["CalP_Over2_5_xG"].mean()),
            "ActualOverRate": float(test["Over2_5_Result"].mean()),
        }

        for threshold in thresholds:
            returns = []
            over_bets = 0
            under_bets = 0
            for _, row in test.iterrows():
                over_odds, under_odds, _ = _market_odds(row)
                if not over_odds:
                    continue

                p_over = float(row["CalP_Over2_5_xG"])
                p_under = 1 - p_over
                ev_over = p_over * over_odds - 1
                ev_under = p_under * under_odds - 1

                if max(ev_over, ev_under) < threshold:
                    continue

                if ev_over >= ev_under:
                    over_bets += 1
                    returns.append(over_odds - 1 if int(row["Over2_5_Result"]) == 1 else -1)
                else:
                    under_bets += 1
                    returns.append(under_odds - 1 if int(row["Over2_5_Result"]) == 0 else -1)

            key = str(threshold).replace(".", "_")
            result[f"Bets_EV_{key}"] = len(returns)
            result[f"ROI_EV_{key}"] = float(sum(returns) / len(returns)) if returns else np.nan
            result[f"ProfitUnits_EV_{key}"] = float(sum(returns)) if returns else 0.0
            result[f"OverBets_EV_{key}"] = over_bets
            result[f"UnderBets_EV_{key}"] = under_bets

        summaries.append(result)

    evaluated = pd.concat(evaluated_frames, ignore_index=True) if evaluated_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return evaluated, summary
