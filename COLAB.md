# Run the EPL Prediction Engine in Google Colab

Open a new Colab notebook and run these cells in order.

## 1. Clone the repository

```python
!git clone https://github.com/mkcesc1988/epl-prediction-engine.git
%cd epl-prediction-engine
```

If Colab says the folder already exists, restart the Colab runtime or run `!rm -rf /content/epl-prediction-engine` before cloning again.

## 2. Install dependencies

```python
!pip -q install -r requirements.txt
```

## 3. Run the entire project

```python
!python run_all.py
```

The pipeline will download Football-Data results and odds, pull Understat xG/xGA, build the master database, create leakage-safe Poisson probabilities, calibrate them walk-forward, and run the historical backtest.

## 4. Inspect the results

```python
import pandas as pd
pd.read_csv('data/processed/backtest_summary.csv')
```

## Output files

- `data/processed/master_matches.csv`
- `data/processed/validation_report.json`
- `data/processed/xg_predictions.csv`
- `data/processed/walkforward_predictions.csv`
- `data/processed/backtest_summary.csv`

If `run_all.py` fails, copy the full error output into ChatGPT. Do not manually change the model code until the data-source error is identified.
