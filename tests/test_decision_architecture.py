import math

import pandas as pd

from gameweek_rankings import (
    _consensus_no_vig_probability,
    _decision_probability,
    _decision_status,
)


CFG = {
    "decision": {
        "market_anchor_enabled": True,
        "market_anchor_weight_moneyline": 0.45,
        "market_anchor_weight_spread": 0.40,
        "market_anchor_weight_total": 0.35,
        "min_consensus_books": 2,
        "no_bet_edge": 0.03,
        "bet_eligible_edge": 0.05,
        "max_raw_model_market_gap": 0.18,
    }
}


def test_market_anchor_shrinks_toward_prior():
    p, effective = _decision_probability(0.65, 0.0, 0.55, "Spread", CFG)
    assert 0.55 < effective < 0.65
    assert math.isclose(p, effective)


def test_decision_status_no_bet_review_and_eligible():
    assert _decision_status(0.02, 4, 0.05, CFG) == "NO_BET"
    assert _decision_status(0.04, 4, 0.05, CFG) == "REVIEW"
    assert _decision_status(0.06, 1, 0.05, CFG) == "REVIEW_LOW_MARKET_SUPPORT"
    assert _decision_status(0.06, 4, 0.05, CFG) == "BET_ELIGIBLE"


def test_totals_consensus_is_devigged():
    q = pd.Series(
        {
            "Date": "2026-09-26",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "Market": "totals",
            "Outcome": "Over",
            "Point": 2.5,
        }
    )
    odds = pd.DataFrame(
        [
            {"Date": "2026-09-26", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Market": "totals", "Outcome": "Over", "Point": 2.5, "DecimalOdds": 1.91, "Bookmaker": "Book A", "BookmakerKey": "a"},
            {"Date": "2026-09-26", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Market": "totals", "Outcome": "Under", "Point": 2.5, "DecimalOdds": 1.91, "Bookmaker": "Book A", "BookmakerKey": "a"},
            {"Date": "2026-09-26", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Market": "totals", "Outcome": "Over", "Point": 2.5, "DecimalOdds": 2.00, "Bookmaker": "Book B", "BookmakerKey": "b"},
            {"Date": "2026-09-26", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Market": "totals", "Outcome": "Under", "Point": 2.5, "DecimalOdds": 1.80, "Bookmaker": "Book B", "BookmakerKey": "b"},
        ]
    )
    p, books, source = _consensus_no_vig_probability(q, odds)
    assert books == 2
    assert source == "consensus_non_mybookie"
    assert 0.47 < p < 0.50
