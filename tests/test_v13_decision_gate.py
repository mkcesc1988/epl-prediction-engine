import math

import pandas as pd
import pytest

from actionable_bets_v13 import build_decisions
from pre_kickoff_adjustment import _agreement_label
from scenario_v13 import build_scenarios, scenario_probability


NOW = pd.Timestamp("2026-10-10T18:00:00Z")


def candidate(**changes):
    row = {
        "Date": "2026-10-11", "KickoffUTC": "2026-10-11T13:00:00Z",
        "HomeTeam": "Hull City", "AwayTeam": "Everton",
        "MarketType": "Spread", "Selection": "Hull City +0.5", "Line": 0.5,
        "MyBookieOdds": 1.80, "DecisionEffectiveProbability": 0.67,
        "ModelWinProbability": 0.67, "RawModelWinProbability": 0.72,
        "PushProbability": 0.0, "ProbabilityEdge": 0.1144,
        "ExpectedReturnPerUnit": 0.206, "ConsensusBookCount": 3,
        "OverallRankScore": 74.0, "DecisionStatus": "BET_ELIGIBLE",
        "PriceTimestampUTC": "2026-10-10T17:45:00Z",
        "DixonColes_Rho": -0.075,
    }
    row.update(changes)
    return row


def test_shared_baseline_does_not_count_as_independent_agreement():
    import numpy as np
    p = np.array([0.4, 0.3, 0.3])
    assert _agreement_label(p, p.copy(), p.copy()) == "SHARED_BASELINE"


def test_lineup_confirmation_needed_before_paper_candidate():
    rankings = pd.DataFrame([candidate()])
    result = build_decisions(rankings, now_utc=NOW)
    assert result.iloc[0]["V13Status"] == "WATCH_LINEUPS_UNVERIFIED"
    assert not bool(result.iloc[0]["V13LineupsVerified"])
    verified = pd.DataFrame([{
        "Date": "2026-10-11", "HomeTeam": "Hull City",
        "AwayTeam": "Everton", "LineupsVerified": True
    }])
    result = build_decisions(rankings, lineups=verified, now_utc=NOW)
    assert result.iloc[0]["V13Status"] == "PAPER_CANDIDATE"


def test_stale_price_cannot_be_executable():
    ranked = pd.DataFrame([candidate(PriceTimestampUTC="2026-10-09T09:00:00Z")])
    lineup = pd.DataFrame([{"Date":"2026-10-11","HomeTeam":"Hull City",
        "AwayTeam":"Everton","LineupsVerified":True}])
    result = build_decisions(ranked, lineups=lineup, now_utc=NOW)
    assert result.iloc[0]["V13Status"] == "REPRICE_REQUIRED"


def test_suspends_historically_miscalibrated_longshot():
    r = candidate(MarketType="Moneyline", Selection="Hull City", Line=float("nan"),
                  MyBookieOdds=4.20, DecisionEffectiveProbability=0.30,
                  RawModelWinProbability=0.31, ExpectedReturnPerUnit=0.26)
    out = build_decisions(pd.DataFrame([r]), now_utc=NOW)
    assert out.iloc[0]["V13Status"] == "NO_BET"
    assert out.iloc[0]["V13Reason"] == "SUSPEND_ML_PLUS300_OR_LONGER"


def test_only_one_primary_selection_per_fixture_and_higher_win_prob():
    ml = candidate(MarketType="Moneyline", Selection="Hull City",
        Line=float("nan"), MyBookieOdds=3.40, DecisionEffectiveProbability=0.38,
        RawModelWinProbability=0.43, ExpectedReturnPerUnit=0.292)
    out = build_decisions(pd.DataFrame([candidate(), ml]), now_utc=NOW)
    selected = out.loc[out["V13PreferredForMatch"]]
    assert len(selected) == 1
    assert selected.iloc[0]["Selection"] == "Hull City +0.5"
    assert "ALTERNATIVE_CORRELATED" in set(out["V13Status"])


def test_hypothetical_scenario_vetoes_fragile_edge():
    pred = pd.DataFrame([{
        "Date": "2026-10-11", "HomeTeam": "Hull City", "AwayTeam": "Everton",
        "Lambda_Home_xG": 1.2497821, "Lambda_Away_xG": 1.0563435,
        "DixonColes_Rho": -0.075,
    }])
    overrides = pd.DataFrame([{
        "Date": "2026-10-11", "HomeTeam": "Hull City", "AwayTeam": "Everton",
        "HomeLambdaMultiplier": 0.928, "AwayLambdaMultiplier": 1.089,
        "AssumptionNote": "hypothetical stress",
    }])
    scen = build_scenarios(pred, overrides)
    assert len(scen) == 1
    assert scen.iloc[0]["ScenarioHomeWin"] < scen.iloc[0]["BaseHomeWin"]
    score = scenario_probability(pd.Series(candidate()), scen.iloc[0])[0]
    assert score < 0.70
    fragile = candidate(MyBookieOdds=1.71, DecisionEffectiveProbability=0.6406,
        ModelWinProbability=0.6406, RawModelWinProbability=0.6983,
        ExpectedReturnPerUnit=0.095, ProbabilityEdge=0.056)
    out = build_decisions(pd.DataFrame([fragile]), scenarios=scen, now_utc=NOW)
    assert out.iloc[0]["V13Status"] == "NO_BET"
    assert out.iloc[0]["V13Reason"] == "SCENARIO_SENSITIVE_NO_BET"


def test_scenario_limit_rejects_unjustified_multiplier():
    pred = pd.DataFrame([{
        "Date": "2026-10-11", "HomeTeam": "Hull City", "AwayTeam": "Everton",
        "Lambda_Home_xG": 1.25, "Lambda_Away_xG": 1.05, "DixonColes_Rho": -0.075,
    }])
    over = pd.DataFrame([{"Date": "2026-10-11", "HomeTeam": "Hull City",
        "AwayTeam": "Everton", "HomeLambdaMultiplier": 0.50,
        "AwayLambdaMultiplier": 1.0, "AssumptionNote": "too large"}])
    with pytest.raises(ValueError):
        build_scenarios(pred, over)
