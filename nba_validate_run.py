"""Reproducible real-data validation runner; outputs stay under data/processed/nba_validation."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from nba import HISTORY_COLUMNS, backtest, predict, read_json, scoreboard, validate_history

ROOT = Path('data/processed/nba_validation')
CACHE = ROOT / 'scoreboards'
CACHE.mkdir(parents=True, exist_ok=True)


def fetch(day):
    path = CACHE / (day + '.json')
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            rows = scoreboard(day)
            path.write_text(json.dumps(rows, allow_nan=True))
            return rows
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    now = pd.Timestamp.now(tz='UTC')
    days = [str(d.date()) for d in pd.date_range('2024-10-22', (now-pd.Timedelta(days=1)).date())]
    failures, rows = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {pool.submit(fetch, d): d for d in days}
        for n, future in enumerate(as_completed(tasks), 1):
            try:
                rows.extend(future.result())
            except Exception as e:
                failures.append({'date':tasks[future], 'error':type(e).__name__})
            if n % 50 == 0:
                print(f'Fetched {n}/{len(days)} dates; errors={len(failures)}', flush=True)
    (ROOT / 'fetch_failures.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError('History fetch incomplete; see fetch_failures.json')
    history = validate_history(pd.DataFrame([r for r in rows if r['Completed']], columns=HISTORY_COLUMNS).drop_duplicates('EventId'))
    history.to_csv(ROOT / 'games.csv', index=False)
    print(f'Validated {len(history)} real games; backtesting', flush=True)
    detail, summary = backtest(history)
    detail.to_csv(ROOT / 'walkforward_predictions.csv', index=False)
    h = history.set_index('EventId')
    benchmark = []
    for row in detail.itertuples():
        past = history[history.Date.dt.normalize() < row.Date.normalize()]
        benchmark.append({'EventId':row.EventId, 'HomeRate':float((past.HomeScore>past.AwayScore).mean()), 'MeanMargin':float((past.HomeScore-past.AwayScore).mean()), 'MeanTotal':float((past.HomeScore+past.AwayScore).mean())})
    b = pd.DataFrame(benchmark)
    p = b.HomeRate.clip(1e-9, 1-1e-9)
    y = detail.HomeWon
    baseline = {'BrierScore':float(((p-y)**2).mean()), 'LogLoss':float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()), 'MarginMAE':float((b.MeanMargin-detail.ActualMargin).abs().mean()), 'TotalMAE':float((b.MeanTotal-detail.ActualTotal).abs().mean())}
    bins = pd.cut(detail.HomeWinProbability, bins=np.linspace(0,1,11), include_lowest=True)
    calibration = detail.groupby(bins, observed=True).agg(Games=('HomeWon','size'), MeanProbability=('HomeWinProbability','mean'), ActualWinRate=('HomeWon','mean'))
    calibration.to_csv(ROOT / 'winner_calibration.csv')
    seasons = []
    for season, group in detail.groupby(np.where(detail.Date.dt.month>=10, detail.Date.dt.year, detail.Date.dt.year-1)):
        p = group.HomeWinProbability.clip(1e-9, 1-1e-9)
        y = group.HomeWon
        seasons.append({'Season':f'{season}/{str(season+1)[-2:]}', 'Games':len(group), 'BrierScore':float(((p-y)**2).mean()), 'LogLoss':float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()), 'MarginMAE':float((group.ExpectedMargin-group.ActualMargin).abs().mean()), 'TotalMAE':float((group.ExpectedTotal-group.ActualTotal).abs().mean())})
    live = scoreboard(now.strftime('%Y-%m-%d'))
    (ROOT / 'live_scoreboard.json').write_text(json.dumps(live, indent=2, allow_nan=True))
    predictions = predict(history, pd.DataFrame(live, columns=HISTORY_COLUMNS+['Completed']), now)
    predictions.to_csv(ROOT / 'live_predictions.csv', index=False)
    live_status = {'FetchedAt':now.isoformat(), 'FeedGames':len(live), 'FuturePredictions':len(predictions), 'OddsKeyConfigured':bool(os.getenv('THE_ODDS_API_KEY'))}
    result = {'FetchedAt':now.isoformat(), 'Source':'ESPN scoreboard', 'HistoryGames':len(history), 'FirstTipoff':history.Date.min().isoformat(), 'LastTipoff':history.Date.max().isoformat(), 'Backtest':summary, 'ExpandingLeagueMeanBaseline':baseline, 'Seasons':seasons, 'Live':live_status}
    (ROOT / 'validation_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
