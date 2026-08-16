from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import load_config

DATA_DIR = Path("data/processed")


def _kelly_fraction(p: float, odds: float, push: float = 0.0) -> float:
    """Kelly fraction for decimal odds, adjusted for push probability.

    This is used only as a paper-sizing reference. Hard exposure caps are
    applied later by build_portfolio.
    """
    if odds <= 1.0 or p <= 0.0:
        return 0.0
    b = odds - 1.0
    q = max(0.0, 1.0 - p - push)
    raw = (b * p - q) / b
    return max(0.0, float(raw))


def build_portfolio(rankings: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()

    pcfg = cfg.get("portfolio", {})
    paper_bankroll = float(pcfg.get("paper_bankroll", 100.0))
    unit_size_pct = float(pcfg.get("unit_size_pct", 0.01))
    fractional_kelly = float(pcfg.get("fractional_kelly", 0.25))
    max_stake_pct = float(pcfg.get("max_stake_pct", 0.02))
    max_gameweek_exposure_pct = float(pcfg.get("max_gameweek_exposure_pct", 0.08))
    min_quality = float(pcfg.get("min_bet_quality", 60.0))
    min_ev = float(pcfg.get("min_expected_return", 0.02))
    max_same_match_exposure_pct = float(pcfg.get("max_same_match_exposure_pct", 0.03))
    min_stake_units = float(pcfg.get("min_stake_units", 0.25))

    df = rankings.copy()
    for col in [
        "ModelWinProbability", "PushProbability", "MyBookieOdds",
        "ExpectedReturnPerUnit", "BetQualityScore", "ProfitabilityScore",
        "OverallRankScore",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    eligible = df[
        (df["ExpectedReturnPerUnit"] >= min_ev)
        & (df["BetQualityScore"] >= min_quality)
        & (df["MyBookieOdds"] > 1.0)
    ].copy()

    if eligible.empty:
        return pd.DataFrame(columns=list(df.columns) + [
            "FullKellyFraction", "PaperStakeFraction", "PaperStakeUnits",
            "PaperStakeAmount", "ExpectedPaperProfit", "PortfolioExposurePct",
            "PortfolioRank", "SizingNote",
        ])

    eligible = eligible.sort_values(
        ["OverallRankScore", "ExpectedReturnPerUnit"],
        ascending=[False, False],
    ).reset_index(drop=True)

    max_total = paper_bankroll * max_gameweek_exposure_pct
    max_per_bet = paper_bankroll * max_stake_pct
    max_per_match = paper_bankroll * max_same_match_exposure_pct
    unit_amount = paper_bankroll * unit_size_pct
    min_stake_amount = min_stake_units * unit_amount

    used_total = 0.0
    used_by_match: dict[tuple[str, str, str], float] = {}
    rows: list[dict] = []

    for _, r in eligible.iterrows():
        p = float(r["ModelWinProbability"])
        push = float(r.get("PushProbability", 0.0) or 0.0)
        odds = float(r["MyBookieOdds"])
        full_kelly = _kelly_fraction(p, odds, push)
        target = paper_bankroll * full_kelly * fractional_kelly
        target = min(target, max_per_bet)

        match_key = (str(r.get("Date", "")), str(r["HomeTeam"]), str(r["AwayTeam"]))
        remaining_match = max(0.0, max_per_match - used_by_match.get(match_key, 0.0))
        remaining_total = max(0.0, max_total - used_total)
        stake = min(target, remaining_match, remaining_total)

        # Exclude dust positions. The threshold is expressed in units so it
        # scales naturally with the configured bankroll and unit size.
        if stake < min_stake_amount - 1e-12:
            continue

        used_total += stake
        used_by_match[match_key] = used_by_match.get(match_key, 0.0) + stake

        out = r.to_dict()
        out.update({
            "FullKellyFraction": full_kelly,
            "PaperStakeFraction": stake / paper_bankroll if paper_bankroll > 0 else 0.0,
            "PaperStakeUnits": stake / unit_amount if unit_amount > 0 else 0.0,
            "PaperStakeAmount": stake,
            "ExpectedPaperProfit": stake * float(r["ExpectedReturnPerUnit"]),
            "PortfolioExposurePct": used_total / paper_bankroll if paper_bankroll > 0 else 0.0,
            "SizingNote": f"paper only; capped fractional Kelly; minimum {min_stake_units:g}u",
        })
        rows.append(out)

        if used_total >= max_total - 1e-9:
            break

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(0, "PortfolioRank", np.arange(1, len(out) + 1))
    return out


def main() -> None:
    cfg = load_config()
    rankings_path = DATA_DIR / "gameweek_rankings_latest.csv"
    if not rankings_path.exists():
        raise RuntimeError("Run gameweek rankings before portfolio manager")

    rankings = pd.read_csv(rankings_path)
    portfolio = build_portfolio(rankings, cfg)

    out_path = DATA_DIR / "paper_portfolio_latest.csv"
    portfolio.to_csv(out_path, index=False)

    pcfg = cfg.get("portfolio", {})
    bankroll = float(pcfg.get("paper_bankroll", 100.0))
    exposure = float(portfolio["PaperStakeAmount"].sum()) if not portfolio.empty else 0.0
    expected_profit = float(portfolio["ExpectedPaperProfit"].sum()) if not portfolio.empty else 0.0

    print(f"Paper bankroll:         {bankroll:.2f}")
    print(f"Portfolio selections:  {len(portfolio)}")
    print(f"Paper exposure:        {exposure:.2f} ({(exposure / bankroll if bankroll else 0):.1%})")
    print(f"Expected paper profit: {expected_profit:.2f}")
    if not portfolio.empty:
        cols = [
            "PortfolioRank", "HomeTeam", "AwayTeam", "MarketType", "Selection",
            "MyBookieOdds", "BetQualityScore", "ExpectedReturnPerUnit",
            "PaperStakeUnits", "PaperStakeAmount", "ExpectedPaperProfit",
        ]
        print(portfolio[[c for c in cols if c in portfolio.columns]].to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
