# EPL Prediction Engine

A reproducible English Premier League prediction and backtesting engine built around match-level xG, Poisson probabilities, walk-forward calibration, and bookmaker comparison.

## What this project does

The project rebuilds its database from source, then runs the same modeling process every time:

1. Download EPL results and Over/Under odds from Football-Data.co.uk.
2. Pull match-level xG/xGA from Understat through the `underdata` Python client.
3. Normalize team names and merge the sources into one master match database.
4. Validate duplicates, missing scores, xG coverage, and cross-source score mismatches.
5. Generate leakage-safe xG Poisson probabilities using only matches played before each target match.
6. Calibrate probabilities with walk-forward Platt calibration.
7. Evaluate Brier score, log loss, and historical ROI at multiple EV thresholds.

## Current baseline model

The V0.6-style baseline uses:

- last 20 venue-specific xG matches
- minimum 5 prior venue-specific matches
- trailing 380-match EPL xG scoring baselines
- separate home and away Poisson lambdas
- Over/Under 2.5 probabilities
- walk-forward probability calibration
- EV thresholds of 0%, 2%, 5%, 7.5%, and 10%

The model intentionally does **not** use future information in historical predictions.

## Repository files

- `pipeline.py` downloads, cleans, merges, and validates source data
- `model.py` builds leakage-safe xG Poisson probabilities
- `backtest.py` calibrates probabilities and runs walk-forward betting tests
- `run_all.py` executes the full project from raw sources to backtest output
- `config.yaml` contains season range and model settings
- `COLAB.md` gives copy/paste Google Colab instructions
- `tests/test_model.py` contains model smoke tests
- `.github/workflows/ci.yml` automatically syntax-checks and tests commits

## Fastest way to run it

In Google Colab:

```python
!git clone https://github.com/mkcesc1988/epl-prediction-engine.git
%cd epl-prediction-engine
!pip -q install -r requirements.txt
!python run_all.py
```

If the repository has already been cloned in the current Colab runtime:

```python
!rm -rf /content/epl-prediction-engine
!git clone https://github.com/mkcesc1988/epl-prediction-engine.git
%cd epl-prediction-engine
!pip -q install -r requirements.txt
!python run_all.py
```

## Output files

The run creates:

- `data/processed/master_matches.csv`
- `data/processed/validation_report.json`
- `data/processed/xg_predictions.csv`
- `data/processed/walkforward_predictions.csv`
- `data/processed/backtest_summary.csv`

## Other leagues: La Liga, Bundesliga, Serie A, Ligue 1

`league_predictions.py` runs the same V1.2 xG attack/defense Dixon-Coles model on Europe's other big-five leagues. Each league is fitted only on its own history, so the EPL configuration, outputs, portfolio and bet ledgers are unchanged.

```bash
python league_predictions.py                     # all leagues in config.yaml leagues.enabled
python league_predictions.py --league SERIEA     # one league (LALIGA, BUNDESLIGA, SERIEA, LIGUE1)
python league_predictions.py --backtest          # add walk-forward backtests
python league_predictions.py --odds              # force an Odds API pull (default: only near kickoff)
```

Sources per league: Football-Data.co.uk (`SP1`, `D1`, `I1`, `F1`) for results and historical odds, Understat for xG and the fixture list, and The Odds API (`soccer_spain_la_liga`, `soccer_germany_bundesliga`, `soccer_italy_serie_a`, `soccer_france_ligue_one`) for live prices.

Outputs go to `data/processed/leagues/<CODE>/`: `daily_predictions_latest.csv` (fixtures in the next `leagues.horizon_days` days), `market_comparison_latest.csv` (O/U 2.5), `moneyline_comparison_latest.csv` (1X2 vs best price and no-vig consensus), and with `--backtest` the walk-forward summaries including `home_advantage_sweep_summary.csv` (1X2 RPS vs Pinnacle). `data/processed/leagues/all_leagues_predictions_latest.csv` combines all leagues.

Team names are canonicalised to Football-Data spellings through `leagues.py`. Names the alias map misses (for example a newly promoted club) are matched automatically by shared fixtures and listed under `reconciled_team_names` in each league's `validation_report.json`.

Per-league settings can be overridden under `leagues.overrides` in `config.yaml`. The EPL-tuned `home_advantage_shrink: 0.15` applies to every league until a league's own backtest supports a different value.

### League paper betting

When a league has a fixture within `leagues.odds_within_days` (default 2), the run pulls The Odds API prices and sends the league through the same decision stack as the EPL: `gameweek_rankings.build_rankings` blends the model toward the no-vig consensus (moneyline 0.85, spread 0.40, total 0.35, plus an early-season bonus by matchday), then the 5pp decision-edge gate, extreme-edge sanity gate and capped fractional Kelly sizing from `portfolio_manager`. League differences (`league_portfolio.py`):

- No V2 agreement gate (V2 is EPL-only), so extreme edges stay research-only.
- Each league's exposure cap is 4% of the paper bankroll, and open bets count against it.
- `portfolio.blocked_markets` keeps La Liga and Bundesliga totals ranked but out of the portfolio: in the 2023/24–2025/26 backtest, O/U bets passing the live gate lost 15% and 11% there.
- Ligue 1 totals are in goal units (`model_v12.totals_goal_scale`), correcting a goals-below-xG drift that biased its totals high.

Paper bets are recorded in `data/history/leagues/paper_bet_ledger.csv`, settled from Understat results, and summarised by league in `paper_performance_summary.csv`. Nothing here touches the EPL ledgers or real-money tracking.

The **League predictions** workflow runs daily at 12:30 UTC and can be run by hand with a league list, backtests and odds turned on.

## How to judge model improvements

A new feature should not be accepted just because historical ROI increases. The primary development checks are:

- out-of-sample Brier score
- out-of-sample log loss
- calibration stability
- ROI stability across seasons
- sample size at each EV threshold

Changes should be tested one at a time so we know what actually improved the model.

## Next upgrades after the baseline is verified

Potential upgrades include Dixon-Coles correction, promoted-team priors, shrinkage between venue-specific and overall xG, rest days, lineup/injury information, BTTS, 1X2 probabilities, and league simulation. These should be added only after the current pipeline runs cleanly and the baseline backtest is reproduced.
