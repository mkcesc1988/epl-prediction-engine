# NBA research baseline

`nba.py` runs independently of football models, portfolios and ledgers. It predicts
home winning probability, home-minus-away margin and combined final score,
including overtime. `FairHomeSpread` uses bookmaker notation: an expected home
margin of +5 gives a fair home spread of -5.

The baseline fits ridge regressions for margin and total with a joint residual
covariance. Features are each team's last 20 games of scoring and points allowed,
shrunk with five league-average pseudo-games, the evolving league scoring average,
and capped rest-day differences/sums. Home advantage is a fitted intercept.
Only results from strictly earlier UTC dates enter features or training. This
conservative boundary prevents same-day tipoff timestamps leaking final results.
Expected team scores follow from margin and total. Winner probabilities and
whole-point push probabilities use a normal approximation; residual covariance
is saved for future joint score simulation.

## Run

Install the repository requirements, then bootstrap historical results (one ESPN
scoreboard request per calendar day; regular season/playoffs only):

```bash
python nba.py --fetch-start 2024-10-22 --backtest
python nba.py --fetch-start 2026-10-20
python nba.py --odds
```

The default fixture date is today UTC interpreted as an ESPN scoreboard calendar
date. To see a different slate, use `--date YYYY-MM-DD`. Already-started games are
excluded. A minimum of 200 historical games is required (configurable with
`--min-train`, minimum 20 for development).

For offline/custom data:

```bash
python nba.py --history games.csv --fixtures fixtures.csv --backtest
```

History columns: `EventId,Date,HomeTeam,AwayTeam,HomeScore,AwayScore`.
Fixture columns: `EventId,Date,HomeTeam,AwayTeam`. Use timezone-aware scheduled
tipoff timestamps in `Date`, completed final scores, unique event IDs and consistent
full team names. Odds matching requires full names matching The Odds API and
tipoff times within three hours. Unmatched games are skipped.

## Outputs and validation

Outputs go to `data/processed/nba/`:

- `daily_predictions_latest.csv`: probabilities, expected scores, fair spread,
  residual standard deviations/covariance, training size and prediction timestamp.
- `market_comparison_latest.csv`: bookmaker prices, capture/update timestamps,
  win/push probabilities, fair decimal odds and expected return per unit stake.
  Current prices use `basketball_nba` and `h2h,spreads,totals`; set
  `THE_ODDS_API_KEY` and opt in with `--odds`. Prices are compared at their actual
  lines. Expected return includes stakes returned on a push. Missing credentials
  with `--odds` fail the run; without it the comparison file is empty.
- With `--backtest`: `walkforward_predictions.csv` and `backtest_summary.json`
  report out-of-sample Brier score, log loss and margin/total MAE. Models refit
  once per test date using earlier dates only.

This is an uncalibrated score-based baseline: no possession estimates, injuries,
lineups, player minutes, playoff-specific effects or market blending yet. The
residual scale is fitted on training errors; assess probability calibration on
walk-forward results before using the model for decisions. Backtests do not
claim cover rates, historical betting ROI or closing-line value: those require
timestamped historical lines and prices. Current odds must never be substituted
for historical quotes. No bets or bankroll allocations are generated.

ESPN is an unofficial public source whose schema/availability can change.
Fetch failures stop the run rather than quietly substituting missing history.
The workflow caches completed games and refreshes seven calendar days on each
daily run. On a cache miss it bootstraps from `history_start`; manually refresh
an earlier range if older scores need correction. Results and history are
downloadable workflow artifacts; the workflow does not commit data to main.

Run `python -m unittest discover -s tests -p test_nba.py` for leakage, validation,
spread-sign, push, score consistency and odds-matching checks. The existing CI
also discovers these tests through pytest.
