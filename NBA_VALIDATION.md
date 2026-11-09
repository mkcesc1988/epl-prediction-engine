# NBA baseline validation — 27 October 2026

Source: ESPN public NBA scoreboard. Downloaded all 735 calendar dates from
22 October 2024 through 26 October 2026 with zero fetch failures. Validated
2,683 completed games across all 30 teams. Final scores include overtime;
regular-season and playoff games are retained, preseason and exhibition teams excluded.
ESPN calendar dates can include tipoffs on the following UTC date.

## Walk-forward results

2,480 games were evaluated after a minimum 200-game training period.
Each test date uses only strictly earlier UTC dates for fitting and features.
The benchmark uses expanding historical home win rate and mean margin/total,
also restricted to earlier dates. Lower values are better for all metrics below.

| Metric | NBA model | League-average benchmark |
| --- | ---: | ---: |
| Winner Brier score | 0.2145 | 0.2483 |
| Winner log loss | 0.6174 | 0.6897 |
| Margin MAE, points | 11.65 | 13.09 |
| Total MAE, points | 15.06 | 15.95 |

Winner accuracy: 65.65% versus
54.60% for always choosing the home team. Ten-bin weighted absolute
calibration error: 1.65%. This descriptive bin statistic has no uncertainty
adjustment; extreme bins have only 6 and 15 games.

| Season | Evaluated games | Brier | Margin MAE | Total MAE |
| --- | ---: | ---: | ---: | ---: |
| 2024/25 | 1112 | 0.2150 | 11.58 | 14.83 |
| 2025/26 | 1316 | 0.2128 | 11.68 | 15.26 |
| 2026/27 | 52 | 0.2475 | 12.65 | 14.97 |

The 2026/27 sample is only 52 games and does not establish current-season quality.
No feature or parameter was tuned after inspecting these results. The evaluation
is chronological out of sample, but is not a separately reserved final holdout.

## Live fixture check

The ESPN feed returned four upcoming games on the 27 October calendar slate.
All four produced finite winner, margin and total predictions using 2,678 earlier
UTC-date games. Today's completed games were excluded conservatively. The
runtime had no Odds API credential. GitHub Actions used the repository secret
and the provider returned **HTTP 401 (`OUT_OF_USAGE_CREDITS`)** on retry.
Live bookmaker matching and line/probability comparisons remain unverified until
credits are restored or a funded key is configured. No bankroll or betting
performance conclusions follow from this run.

The real-data backtest reproduced in GitHub Actions; full code CI passed.
Seven targeted NBA tests passed, including credential-safe HTTP diagnostics.
See [validation run](https://github.com/mkcesc1988/epl-prediction-engine/actions/runs/40551648098)
for the backtest, explicit bookmaker failure and downloadable result artifacts.

## Interpretation

The model beats a simple league-average benchmark across the combined sample.
This does not establish an advantage over bookmakers. Historical bookmaker lines,
prices and decision-time availability were not present, so no spread cover rate,
over/under hit rate, ROI or closing-line value is claimed. Next evaluation should
use archived quotes and paper predictions, plus out-of-sample uncertainty checks
for spread and total probabilities. Player availability and possession features
remain absent.

Reproduce with `python nba_validate_run.py`. Machine-readable results, per-game
predictions and calibration bins are in `data/processed/nba_validation/`, and the
**NBA real-data validation** workflow publishes downloadable artifacts.
