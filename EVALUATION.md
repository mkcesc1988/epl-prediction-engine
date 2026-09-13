# Live evaluation layer

This branch adds leakage-safe live tracking around the existing V1.2 EPL model. It does not change model parameters or prediction logic.

## What it records

`evaluate_live.py` collects the first saved prediction for each `(Season, FPLFixtureId, ModelVersion)` and writes it to:

- `data/processed/prediction_history.csv`

Once FPL marks a fixture finished, the same row is settled with:

- actual home and away goals
- H/D/A result
- Over 2.5 outcome
- BTTS outcome
- settlement timestamp

The first prediction is preserved even if later daily runs produce a different probability. This is intentional so the evaluation cannot rewrite history after kickoff.

## Reports

Running:

```bash
python evaluate_live.py
```

writes:

- `data/processed/live_evaluation_summary.csv`
- `data/processed/live_calibration.csv`
- `data/processed/live_rolling_evaluation.csv`

The summary includes:

- 1X2 accuracy
- multiclass Brier score
- multiclass log loss
- Ranked Probability Score
- BTTS Brier score and log loss
- Over 2.5 Brier score and log loss
- home, away and total-goal MAE

Calibration is reported by probability bucket for Home, Draw, Away, BTTS and Over 2.5. Rolling reports are created once at least 20, 50 or 100 settled matches exist.

## Existing prediction files

Older daily prediction files do not contain `FPLFixtureId`. `evaluate_live.py` enriches them by joining to the official FPL fixture schedule using season, date, home team and away team before archiving.

## Interpretation

Do not tune V1.2 from one or two surprising results. Use the live ledger to look for persistent calibration or scoring-rule weaknesses over a meaningful sample. A 60% forecast is expected to fail frequently. The goal is calibrated probabilities, not perfect exact-score prediction.

## Tests

Pure evaluation behavior is covered in `tests/test_live_evaluation.py` and can be run with:

```bash
python -m unittest discover -s tests
```
