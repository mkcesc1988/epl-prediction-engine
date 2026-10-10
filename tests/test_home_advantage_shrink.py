import numpy as np
import pandas as pd
import pytest

from model_v12 import StrengthFit, build_predictions_v12, effective_home_advantage
from home_advantage_sweep import rps, run_sweep, verdict


def _fit(ha=0.3):
    return StrengthFit(teams=["A", "B"], attack={"A": 0.0, "B": 0.0},
                       defense={"A": 0.0, "B": 0.0}, home_advantage=ha, intercept=0.3)


def test_effective_home_advantage_default_is_unchanged():
    assert effective_home_advantage(_fit(), {}) == pytest.approx(0.3)
    assert effective_home_advantage(_fit(), {"model_v12": {"home_advantage_shrink": 0.0}}) == pytest.approx(0.3)


def test_effective_home_advantage_shrinks_and_clips():
    assert effective_home_advantage(_fit(), {"model_v12": {"home_advantage_shrink": 0.5}}) == pytest.approx(0.15)
    assert effective_home_advantage(_fit(), {"model_v12": {"home_advantage_shrink": 1.0}}) == pytest.approx(0.0)
    assert effective_home_advantage(_fit(), {"model_v12": {"home_advantage_shrink": 2.0}}) == pytest.approx(0.0)
    assert effective_home_advantage(_fit(), {"model_v12": {"home_advantage_shrink": -1.0}}) == pytest.approx(0.3)


def test_rps_perfect_and_worst():
    assert rps(np.array([1.0, 0.0, 0.0]), 0) == pytest.approx(0.0)
    assert rps(np.array([0.0, 0.0, 1.0]), 0) == pytest.approx(1.0)


def _synthetic_master(seasons=3, rounds=38, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    rows = []
    start = pd.Timestamp("2020-08-01")
    day = 0
    for s in range(seasons):
        for _ in range(rounds // 2):
            order = rng.permutation(teams)
            for h, a in zip(order[:5], order[5:]):
                hx, ax = rng.gamma(4, 0.4), rng.gamma(4, 0.3)
                rows.append({
                    "Season": f"{2020 + s}/{2021 + s}",
                    "Date": start + pd.Timedelta(days=day),
                    "HomeTeam": h, "AwayTeam": a,
                    "Home_xG": hx, "Away_xG": ax,
                    "FTHG": int(rng.poisson(hx)), "FTAG": int(rng.poisson(ax)),
                    "Pinnacle_Home": 2.3, "Pinnacle_Draw": 3.4, "Pinnacle_Away": 3.2,
                })
            day += 7
        day += 60
    return pd.DataFrame(rows)


def _cfg(**extra):
    cfg = {"model_v12": {"min_history_matches": 50, "rho_min_matches": 50, "fit_window_matches": 400,
                         "rho_window_matches": 300, "rho_grid_step": 0.02},
           "home_advantage_sweep": {"shrink_grid": [0.0, 0.5, 1.0], "eval_last_n_seasons": 1}}
    cfg["model_v12"].update(extra)
    return cfg


def test_shrink_lowers_home_lambda_in_walk_forward():
    master = _synthetic_master(seasons=2)
    base = build_predictions_v12(master, _cfg()).dropna(subset=["Lambda_Home_xG"])
    shrunk = build_predictions_v12(master, _cfg(home_advantage_shrink=1.0)).dropna(subset=["Lambda_Home_xG"])
    assert (base["HomeAdvantageUsed"] == base["HomeAdvantageLog"]).all()
    assert (shrunk["HomeAdvantageUsed"] == 0.0).all()
    assert shrunk["Lambda_Home_xG"].mean() < base["Lambda_Home_xG"].mean()


def test_sweep_zero_shrink_matches_walk_forward_model():
    master = _synthetic_master(seasons=2)
    cfg = _cfg()
    detail, summary = run_sweep(master, cfg)
    assert set(summary["Shrink"]) == {0.0, 0.5, 1.0}
    assert (summary["Season"] == "ALL").sum() == 3
    assert detail["Season"].nunique() == 1  # only the last season is scored

    wf = build_predictions_v12(master, cfg)
    wf["Date"] = pd.to_datetime(wf["Date"]).dt.strftime("%Y-%m-%d")
    base = detail[detail["Shrink"] == 0.0].merge(wf, on=["Date", "HomeTeam", "AwayTeam"])
    assert len(base) == (detail["Shrink"] == 0.0).sum()
    assert np.allclose(base["P_Home"], base["P_HomeWin_DC"])
    assert np.allclose(base["P_Away"], base["P_AwayWin_DC"])
    assert isinstance(verdict(summary), str)
