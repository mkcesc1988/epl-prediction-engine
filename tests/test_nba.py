import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nba import (HISTORY_COLUMNS, backtest, compare_odds, features,
                 market_probability, predict, validate_history)


def sample_history(n=90):
    rng = np.random.default_rng(42)
    rows = []
    teams = ["Boston Celtics", "New York Knicks", "Miami Heat", "Chicago Bulls"]
    for i in range(n):
        home, away = teams[i % 4], teams[(i+1) % 4]
        hs, aws = int(rng.integers(90, 131)), int(rng.integers(90, 131))
        if hs == aws:
            hs += 1
        rows.append([str(i), pd.Timestamp("2025-01-01", tz="UTC")+pd.Timedelta(days=i), home, away, hs, aws])
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


class NBATest(unittest.TestCase):
    def test_history_rejects_duplicate_tied_and_nonfinite_scores(self):
        h = sample_history()
        for bad in [pd.concat([h, h.iloc[:1]]), h.assign(AwayScore=h.HomeScore), h.assign(HomeScore=np.inf)]:
            with self.assertRaises(ValueError):
                validate_history(bad)

    def test_features_exclude_same_day_and_future_results(self):
        h = validate_history(sample_history())
        date = h.Date.iloc[40]
        before = features(h, "Boston Celtics", "New York Knicks", date)
        h.loc[h.Date >= date, ["HomeScore", "AwayScore"]] += 100
        np.testing.assert_array_equal(before, features(h, "Boston Celtics", "New York Knicks", date))

    def test_walkforward_earlier_predictions_do_not_change(self):
        h = sample_history()
        before, summary = backtest(h, min_train=30)
        h.loc[70:, ["HomeScore", "AwayScore"]] += 50
        after, _ = backtest(h, min_train=30)
        columns = ["HomeWinProbability", "ExpectedMargin", "ExpectedTotal"]
        pd.testing.assert_frame_equal(before.iloc[:41][columns], after.iloc[:41][columns])
        self.assertEqual(summary["Games"], 60)
        self.assertTrue(np.isfinite(list(summary.values())).all())

    def test_prediction_consistency_and_future_history_filter(self):
        h = sample_history()
        fixtures = pd.DataFrame([{"EventId": "future", "Date": "2025-04-02T23:00:00Z", "HomeTeam": "Boston Celtics", "AwayTeam": "New York Knicks"}])
        now = pd.Timestamp("2025-04-02T12:00:00Z")
        a = predict(h, fixtures, now, 30)
        future = sample_history(100).iloc[95:]
        b = predict(pd.concat([h, future]), fixtures, now, 30)
        pd.testing.assert_frame_equal(a, b)
        p = next(a.itertuples())
        self.assertAlmostEqual(p.HomeWinProbability+p.AwayWinProbability, 1)
        self.assertAlmostEqual(p.ExpectedHomeScore+p.ExpectedAwayScore, p.ExpectedTotal)
        self.assertAlmostEqual(p.ExpectedHomeScore-p.ExpectedAwayScore, p.ExpectedMargin)
        self.assertAlmostEqual(p.FairHomeSpread, -p.ExpectedMargin)

    def test_spread_sign_pushes_and_complementarity(self):
        p = SimpleNamespace(HomeTeam="A", AwayTeam="B", ExpectedMargin=5., ExpectedTotal=220., MarginSD=12., TotalSD=18.)
        win, push = market_probability(p, "spreads", "A", -5.)
        lose, other_push = market_probability(p, "spreads", "B", 5.)
        self.assertAlmostEqual(win+lose+push, 1)
        self.assertAlmostEqual(push, other_push)
        self.assertGreater(push, 0)
        self.assertGreater(market_probability(p, "spreads", "A", 5.)[0], win)
        over, push = market_probability(p, "totals", "Over", 220.5)
        under, _ = market_probability(p, "totals", "Under", 220.5)
        self.assertEqual(push, 0)
        self.assertAlmostEqual(over+under, 1)

    def test_comparison_ev_and_game_matching(self):
        h = sample_history()
        now = pd.Timestamp("2025-04-02T12:00:00Z")
        fixtures = pd.DataFrame([{"EventId": "f", "Date": "2025-04-02T23:00:00Z", "HomeTeam": "Boston Celtics", "AwayTeam": "New York Knicks"}])
        predictions = predict(h, fixtures, now, 30)
        event = {"commence_time": "2025-04-02T23:00:00Z", "home_team": "Boston Celtics", "away_team": "New York Knicks", "bookmakers": [{"key": "test", "markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 220., "price": 1.91}]}]}]}
        result = compare_odds(predictions, [event], now)
        row = result.iloc[0]
        self.assertAlmostEqual(row.ExpectedReturn, row.WinProbability*1.91+row.PushProbability-1)
        event["commence_time"] = "2025-04-05T23:00:00Z"
        self.assertTrue(compare_odds(predictions, [event], now).empty)


if __name__ == "__main__":
    unittest.main()
