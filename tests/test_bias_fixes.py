"""Tests for the underdog-bias fixes: promoted-team prior and early-season anchoring."""
import itertools
import math

import numpy as np
import pandas as pd

from gameweek_rankings import _decision_probability, _early_season_anchor_bonus
from model_v12 import _fit_strengths, promoted_team_prior


def _season(start_year: int, teams: list[str], weak: set[str], rng: np.random.Generator) -> list[dict]:
    """Double round-robin; teams in ``weak`` create less and concede more xG."""
    rows = []
    day = pd.Timestamp(f"{start_year}-08-15")
    for i, (h, a) in enumerate(itertools.permutations(teams, 2)):
        hx = 1.5 * (0.7 if h in weak else 1.0) * (1.3 if a in weak else 1.0)
        ax = 1.2 * (0.7 if a in weak else 1.0) * (1.3 if h in weak else 1.0)
        hx *= rng.uniform(0.95, 1.05)
        ax *= rng.uniform(0.95, 1.05)
        rows.append({
            "Date": day + pd.Timedelta(days=3 * i),
            "HomeTeam": h, "AwayTeam": a,
            "Home_xG": hx, "Away_xG": ax,
            "FTHG": round(hx), "FTAG": round(ax),
        })
    return rows


def _history() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = ["A", "B", "C", "D", "E", "F"]
    rows = []
    rows += _season(2022, base, set(), rng)
    rows += _season(2023, ["A", "B", "C", "D", "E", "P1"], {"P1"}, rng)   # P1 promoted, weak
    rows += _season(2024, ["A", "B", "C", "D", "E", "P2"], {"P2"}, rng)   # P2 promoted, weak
    return pd.DataFrame(rows)


def _cfg(**overrides) -> dict:
    mc = {
        "half_life_days": 365, "fit_window_matches": 1140, "min_history_matches": 20,
        "regularization": 0.20, "promoted_prior_enabled": True, "promoted_prior_auto": True,
        "promoted_prior_attack": -0.20, "promoted_prior_defense": -0.20,
        "promoted_prior_matches": 8, "promoted_prior_min_team_seasons": 1,
    }
    mc.update(overrides)
    return {"model_v12": mc}


def test_promoted_prior_identifies_promoted_and_is_negative():
    hist = _history()
    ref = pd.Timestamp("2025-03-01")  # inside the 2024/25 season
    att, dfn, promoted = promoted_team_prior(hist[hist.Date < ref], ref, _cfg())
    assert promoted == frozenset({"P2"})
    # estimated only from completed seasons (P1 in 2023/24), which was weak
    assert att < -0.15 and dfn < -0.15


def test_promoted_prior_falls_back_without_enough_history():
    hist = _history()
    ref = pd.Timestamp("2025-03-01")
    att, dfn, _ = promoted_team_prior(hist[hist.Date < ref], ref, _cfg(promoted_prior_min_team_seasons=5))
    assert (att, dfn) == (-0.20, -0.20)


def test_unknown_team_uses_prior_and_legacy_default_is_zero():
    hist = _history()
    ref = pd.Timestamp("2025-08-10")  # before a new season: nobody seen yet
    on = _fit_strengths(hist, ref, _cfg())
    off = _fit_strengths(hist, ref, _cfg(promoted_prior_enabled=False))
    assert on.team_attack("NEW") < 0 and on.team_defense("NEW") < 0
    assert off.team_attack("NEW") == 0.0 and off.team_defense("NEW") == 0.0


def test_prior_pulls_promoted_team_down_after_few_average_matches():
    hist = _history()
    # A new team plays two league-average matches at the start of 2025/26.
    start = pd.Timestamp("2025-08-16")
    extra = pd.DataFrame([
        {"Date": start, "HomeTeam": "P3", "AwayTeam": "A", "Home_xG": 1.35, "Away_xG": 1.35, "FTHG": 1, "FTAG": 1},
        {"Date": start + pd.Timedelta(days=7), "HomeTeam": "B", "AwayTeam": "P3", "Home_xG": 1.35, "Away_xG": 1.35, "FTHG": 1, "FTAG": 1},
    ])
    full = pd.concat([hist, extra], ignore_index=True)
    ref = start + pd.Timedelta(days=14)
    on = _fit_strengths(full, ref, _cfg())
    off = _fit_strengths(full, ref, _cfg(promoted_prior_enabled=False))
    assert "P3" in on.promoted_teams
    assert on.team_attack("P3") < off.team_attack("P3")
    assert on.team_defense("P3") < off.team_defense("P3")


def test_early_season_bonus_fades_out():
    cfg = {"decision": {"early_season_gameweeks": 8, "early_season_anchor_bonus": 0.16}}
    assert math.isclose(_early_season_anchor_bonus(1, cfg), 0.16)
    assert math.isclose(_early_season_anchor_bonus(8, cfg), 0.02)
    assert _early_season_anchor_bonus(9, cfg) == 0.0
    assert _early_season_anchor_bonus(None, cfg) == 0.0
    assert _early_season_anchor_bonus(float("nan"), cfg) == 0.0
    assert _early_season_anchor_bonus(1, {"decision": {}}) == 0.0


def test_early_gameweek_moves_decision_closer_to_market():
    cfg = {"decision": {"market_anchor_enabled": True, "market_anchor_weight_moneyline": 0.5,
                        "early_season_gameweeks": 8, "early_season_anchor_bonus": 0.2}}
    _, late = _decision_probability(0.40, 0.0, 0.20, "Moneyline", cfg, 20)
    _, early = _decision_probability(0.40, 0.0, 0.20, "Moneyline", cfg, 1)
    _, legacy = _decision_probability(0.40, 0.0, 0.20, "Moneyline", cfg)
    assert math.isclose(late, 0.30) and math.isclose(legacy, 0.30)
    assert abs(early - 0.20) < abs(late - 0.20)
