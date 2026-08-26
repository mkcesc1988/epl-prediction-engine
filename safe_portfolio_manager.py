from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

import portfolio_manager

DATA_DIR = Path("data/processed")
RANKINGS = DATA_DIR / "gameweek_rankings_latest.csv"
PORTFOLIO = DATA_DIR / "paper_portfolio_latest.csv"
V2_AUDIT = DATA_DIR / "v1_v2_agreement_audit_latest.csv"
EXTREME_AUDIT = DATA_DIR / "extreme_edge_audit_latest.csv"

BASE_COLUMNS = [
    "Date", "KickoffUTC", "HomeTeam", "AwayTeam", "MarketType", "Selection", "Line",
    "MyBookieOdds", "ModelWinProbability", "PushProbability", "ExpectedReturnPerUnit",
    "BetQualityScore", "ProfitabilityScore", "OverallRankScore",
]
PORTFOLIO_COLUMNS = BASE_COLUMNS + [
    "V2ModelVersion", "V2WinProbability", "V2PushProbability", "V2ExpectedReturnPerUnit",
    "V1V2ProbabilityGap", "V2Agreement", "V2AdmissionStatus", "ExtremeEdgeFlag",
    "ExtremeEdgeThreshold", "ExtremeLinePairValid", "ConsensusBookCount", "ConsensusMedianOdds",
    "ConsensusMedianImpliedProbability", "ModelConsensusProbabilityGap",
    "MyBookieConsensusPriceDeviationPct", "ExtremeEdgeSanityPass", "ExtremeEdgeSanityStatus",
    "FullKellyFraction", "PaperStakeFraction", "PaperStakeUnits", "PaperStakeAmount",
    "ExpectedPaperProfit", "PortfolioExposurePct", "PortfolioRank", "SizingNote",
]
V2_AUDIT_COLUMNS = BASE_COLUMNS + [
    "V2ModelVersion", "V2WinProbability", "V2PushProbability", "V2ExpectedReturnPerUnit",
    "V1V2ProbabilityGap", "V2Agreement", "V2AdmissionStatus",
]
EXTREME_AUDIT_COLUMNS = V2_AUDIT_COLUMNS + [
    "ExtremeEdgeFlag", "ExtremeEdgeThreshold", "ExtremeLinePairValid", "ConsensusBookCount",
    "ConsensusMedianOdds", "ConsensusMedianImpliedProbability", "ModelConsensusProbabilityGap",
    "MyBookieConsensusPriceDeviationPct", "ExtremeEdgeSanityPass", "ExtremeEdgeSanityStatus",
]


def _write_empty_outputs(reason: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=PORTFOLIO_COLUMNS).to_csv(PORTFOLIO, index=False)
    pd.DataFrame(columns=V2_AUDIT_COLUMNS).to_csv(V2_AUDIT, index=False)
    pd.DataFrame(columns=EXTREME_AUDIT_COLUMNS).to_csv(EXTREME_AUDIT, index=False)
    print(f"No portfolio built: {reason}")
    print("Portfolio selections: 0")


def main() -> None:
    if not RANKINGS.exists() or RANKINGS.stat().st_size == 0:
        _write_empty_outputs("gameweek rankings file is missing or empty")
        return

    try:
        rankings = pd.read_csv(RANKINGS)
    except EmptyDataError:
        _write_empty_outputs("gameweek rankings file has no parseable columns")
        return

    if rankings.empty:
        _write_empty_outputs("gameweek rankings contain zero rows")
        return

    portfolio_manager.main()


if __name__ == "__main__":
    main()
