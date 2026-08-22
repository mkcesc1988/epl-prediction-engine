from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from bet_tracker import _norm, _settle_one

HISTORY = Path('data/history')
MANUAL = Path('data/manual_results.csv')
PAPER = HISTORY / 'auto_bet_ledger.csv'
REAL = HISTORY / 'real_money_bet_ledger.csv'


def _apply(path: Path, stake_col: str, odds_col: str, profit_col: str) -> int:
    if not path.exists() or not MANUAL.exists():
        return 0
    ledger = pd.read_csv(path)
    manual = pd.read_csv(MANUAL)
    changed = 0
    for idx, bet in ledger.iterrows():
        if str(bet.get('Result', 'OPEN')) != 'OPEN':
            continue
        m = manual[(manual['Date'].astype(str) == str(bet.get('Date'))) &
                   (manual['HomeTeam'].map(_norm) == _norm(bet.get('HomeTeam'))) &
                   (manual['AwayTeam'].map(_norm) == _norm(bet.get('AwayTeam')))]
        if m.empty:
            continue
        row = m.iloc[-1]
        hg, ag = int(row['HomeGoals']), int(row['AwayGoals'])
        result, y = _settle_one(bet, hg, ag)
        if result == 'OPEN':
            continue
        stake = float(pd.to_numeric(bet.get(stake_col), errors='coerce'))
        odds = float(pd.to_numeric(bet.get(odds_col), errors='coerce'))
        profit = stake * (odds - 1.0) if result == 'W' else -stake if result == 'L' else 0.0
        ledger.at[idx, 'Result'] = result
        ledger.at[idx, 'FinalScore'] = f'{hg}-{ag}'
        ledger.at[idx, profit_col] = profit
        if 'BrierScore' in ledger.columns and y == y:
            p = float(pd.to_numeric(bet.get('ModelProbability'), errors='coerce'))
            ledger.at[idx, 'BrierScore'] = (p - y) ** 2
        ledger.at[idx, 'SettledUTC'] = pd.Timestamp.now(tz='UTC').isoformat()
        changed += 1
    ledger.to_csv(path, index=False)
    return changed


def main() -> None:
    paper = _apply(PAPER, 'StakeUnits', 'EntryOdds', 'ProfitUnits')
    real = _apply(REAL, 'StakeUSD', 'EntryOdds', 'ProfitUSD')
    print(f'Manual verified settlements applied, paper={paper}, real_money={real}')


if __name__ == '__main__':
    main()
