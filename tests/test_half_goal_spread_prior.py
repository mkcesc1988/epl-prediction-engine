import math

import numpy as np
import pandas as pd

from gameweek_rankings import _half_goal_spread_moneyline_prior


def _odds(include_consensus: bool = True) -> pd.DataFrame:
    rows = []

    def add(book, key, market, outcome, point, price):
        rows.append({
            "Date": "2026-10-17", "HomeTeam": "Troyes", "AwayTeam": "Lens",
            "Bookmaker": book, "BookmakerKey": key, "Market": market,
            "Outcome": outcome, "Point": point, "DecimalOdds": price,
        })

    # MyBookie quotes the half-goal spread; nobody else does.
    add("MyBookie.ag", "mybookieag", "spreads", "Lens", -0.5, 1.82)
    add("MyBookie.ag", "mybookieag", "spreads", "Troyes", 0.5, 1.917)
    add("MyBookie.ag", "mybookieag", "h2h", "Troyes", np.nan, 3.80)
    add("MyBookie.ag", "mybookieag", "h2h", "Draw", np.nan, 3.90)
    add("MyBookie.ag", "mybookieag", "h2h", "Lens", np.nan, 1.855)
    if include_consensus:
        for book, (h, d, a) in {"BookA": (4.0, 4.0, 2.0), "BookB": (4.2, 3.8, 1.95)}.items():
            add(book, book.lower(), "h2h", "Troyes", np.nan, h)
            add(book, book.lower(), "h2h", "Draw", np.nan, d)
            add(book, book.lower(), "h2h", "Lens", np.nan, a)
    return pd.DataFrame(rows)


def _quote(outcome: str, point: float) -> pd.Series:
    return pd.Series({
        "Date": "2026-10-17", "HomeTeam": "Troyes", "AwayTeam": "Lens",
        "Market": "spreads", "Outcome": outcome, "Point": point,
    })


def _no_vig(h, d, a, side):
    imp = {"h": 1 / h, "d": 1 / d, "a": 1 / a}
    return imp[side] / sum(imp.values())


def test_minus_half_uses_1x2_win_consensus():
    p, books, src = _half_goal_spread_moneyline_prior(_quote("Lens", -0.5), _odds())
    expected = np.median([_no_vig(4.0, 4.0, 2.0, "a"), _no_vig(4.2, 3.8, 1.95, "a")])
    assert src == "consensus_1x2_half_goal"
    assert books == 2
    assert math.isclose(p, expected)


def test_plus_half_is_win_or_draw():
    p, books, src = _half_goal_spread_moneyline_prior(_quote("Troyes", 0.5), _odds())
    win = np.median([_no_vig(4.0, 4.0, 2.0, "h"), _no_vig(4.2, 3.8, 1.95, "h")])
    draw = np.median([_no_vig(4.0, 4.0, 2.0, "d"), _no_vig(4.2, 3.8, 1.95, "d")])
    assert src == "consensus_1x2_half_goal"
    assert math.isclose(p, win + draw)


def test_falls_back_without_non_mybookie_1x2():
    p, books, src = _half_goal_spread_moneyline_prior(_quote("Lens", -0.5), _odds(include_consensus=False))
    assert src == "unavailable" and books == 0 and np.isnan(p)


def test_other_lines_untouched():
    p, _, src = _half_goal_spread_moneyline_prior(_quote("Lens", -1.5), _odds())
    assert src == "unavailable" and np.isnan(p)
