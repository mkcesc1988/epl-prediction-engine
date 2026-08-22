from pathlib import Path

import pandas as pd

from bet_tracker import _summary as paper_summary
from real_money_tracker import _summary as real_summary

HISTORY = Path("data/history")
PAPER_LEDGER = HISTORY / "auto_bet_ledger.csv"
REAL_LEDGER = HISTORY / "real_money_bet_ledger.csv"
PAPER_SUMMARY = HISTORY / "bet_performance_summary.csv"
REAL_SUMMARY = HISTORY / "real_money_performance_summary.csv"


def main() -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)

    paper = pd.read_csv(PAPER_LEDGER) if PAPER_LEDGER.exists() and PAPER_LEDGER.stat().st_size else pd.DataFrame()
    real = pd.read_csv(REAL_LEDGER) if REAL_LEDGER.exists() and REAL_LEDGER.stat().st_size else pd.DataFrame()

    psummary = paper_summary(paper) if not paper.empty else pd.DataFrame()
    rsummary = real_summary(real) if not real.empty else pd.DataFrame()

    psummary.to_csv(PAPER_SUMMARY, index=False)
    rsummary.to_csv(REAL_SUMMARY, index=False)

    print(f"Paper settled: {int(paper['Result'].isin(['W','L','PUSH']).sum()) if not paper.empty else 0}")
    print(f"Real-money settled: {int(real['Result'].isin(['W','L','PUSH']).sum()) if not real.empty else 0}")


if __name__ == "__main__":
    main()
