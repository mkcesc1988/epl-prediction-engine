from __future__ import annotations

import math

import numpy as np
import pandas as pd

from model import build_predictions


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _dc_log_likelihood(history: list[tuple[float, float, int, int]], rho: float) -> float:
    ll = 0.0
    for lam_h, lam_a, hg, ag in history:
        tau = _tau(hg, ag, lam_h, lam_a, rho)
        if tau <= 0:
            return -1e18
        p = _poisson_pmf(hg, lam_h) * _poisson_pmf(ag, lam_a) * tau
        if p <= 0:
            return -1e18
        ll += math.log(p)
    return ll


def _estimate_rho(history: list[tuple[float, float, int, int]], cfg: dict) -> float:
    dc_cfg = cfg.get("model_v11", {})
    min_matches = int(dc_cfg.get("rho_min_matches", 200))
    fallback = float(dc_cfg.get("rho_fallback", -0.08))
    if len(history) < min_matches:
        return fallback

    lo = float(dc_cfg.get("rho_grid_min", -0.20))
    hi = float(dc_cfg.get("rho_grid_max", 0.05))
    step = float(dc_cfg.get("rho_grid_step", 0.005))
    grid = np.arange(lo, hi + step / 2, step)
    scores = [(_dc_log_likelihood(history, float(rho)), float(rho)) for rho in grid]
    return max(scores, key=lambda x: x[0])[1]


def _score_matrix(lam_h: float, lam_a: float, rho: float, max_goal: int) -> np.ndarray:
    matrix = np.zeros((max_goal + 1, max_goal + 1), dtype=float)
    for h in range(max_goal + 1):
        ph = _poisson_pmf(h, lam_h)
        for a in range(max_goal + 1):
            pa = _poisson_pmf(a, lam_a)
            matrix[h, a] = ph * pa * _tau(h, a, lam_h, lam_a, rho)

    total = matrix.sum()
    if total <= 0:
        raise RuntimeError("Dixon-Coles score matrix has non-positive probability mass")
    return matrix / total


def _derived_markets(matrix: np.ndarray) -> dict[str, float]:
    n = matrix.shape[0]
    home = float(sum(matrix[h, a] for h in range(n) for a in range(n) if h > a))
    draw = float(sum(matrix[h, h] for h in range(n)))
    away = float(sum(matrix[h, a] for h in range(n) for a in range(n) if h < a))
    btts = float(sum(matrix[h, a] for h in range(1, n) for a in range(1, n)))

    out = {
        "P_HomeWin_DC": home,
        "P_Draw_DC": draw,
        "P_AwayWin_DC": away,
        "P_BTTS_Yes_DC": btts,
        "P_BTTS_No_DC": 1.0 - btts,
    }

    for line, min_total in [(0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4), (4.5, 5)]:
        p_over = float(
            sum(matrix[h, a] for h in range(n) for a in range(n) if h + a >= min_total)
        )
        suffix = str(line).replace(".", "_")
        out[f"P_Over_{suffix}_DC"] = p_over
        out[f"P_Under_{suffix}_DC"] = 1.0 - p_over

    return out


def build_predictions_v11(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Dixon-Coles candidate built on leakage-safe V0.6 expected-goal lambdas.

    The dependence parameter rho is estimated using only earlier matches that
    already had valid pre-match lambdas, so the current match outcome never
    influences its own prediction.
    """
    base = build_predictions(master, cfg).copy()
    base["Date"] = pd.to_datetime(base["Date"])
    base = base.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    dc_cfg = cfg.get("model_v11", {})
    max_goal = int(dc_cfg.get("max_goal", 10))
    rho_window = int(dc_cfg.get("rho_history_window", 760))

    history: list[tuple[float, float, int, int]] = []
    rows = []

    for _, r in base.iterrows():
        row = dict(r)
        lam_h = r.get("Lambda_Home_xG")
        lam_a = r.get("Lambda_Away_xG")

        if pd.notna(lam_h) and pd.notna(lam_a):
            rho_history = history[-rho_window:] if rho_window > 0 else history
            rho = _estimate_rho(rho_history, cfg)
            matrix = _score_matrix(float(lam_h), float(lam_a), rho, max_goal)
            markets = _derived_markets(matrix)
            row["DixonColes_Rho"] = rho
            row["RawP_Over2_5_xG"] = markets["P_Over_2_5_DC"]
            row["RawP_Under2_5_xG"] = markets["P_Under_2_5_DC"]
            row["FairOdds_Over2_5_xG"] = 1.0 / markets["P_Over_2_5_DC"]
            row["FairOdds_Under2_5_xG"] = 1.0 / markets["P_Under_2_5_DC"]
            row.update(markets)
        else:
            row["DixonColes_Rho"] = np.nan
            for key in [
                "P_HomeWin_DC", "P_Draw_DC", "P_AwayWin_DC",
                "P_BTTS_Yes_DC", "P_BTTS_No_DC",
                "P_Over_0_5_DC", "P_Under_0_5_DC",
                "P_Over_1_5_DC", "P_Under_1_5_DC",
                "P_Over_2_5_DC", "P_Under_2_5_DC",
                "P_Over_3_5_DC", "P_Under_3_5_DC",
                "P_Over_4_5_DC", "P_Under_4_5_DC",
            ]:
                row[key] = np.nan

        rows.append(row)

        if (
            pd.notna(lam_h)
            and pd.notna(lam_a)
            and pd.notna(r.get("FTHG"))
            and pd.notna(r.get("FTAG"))
        ):
            history.append((float(lam_h), float(lam_a), int(r["FTHG"]), int(r["FTAG"])))

    return pd.DataFrame(rows)
