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
