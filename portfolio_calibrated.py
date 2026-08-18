from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import load_config
from portfolio_manager import apply_extreme_edge_gate, apply_v2_agreement_gate, build_portfolio

DATA_DIR = Path("data/processed")


def _moneyline_calibration(row: pd.Series) -> tuple[bool, float, str]:
    if str(row.get("MarketType", "")) != "Moneyline":
        return True, 1.0, "Not a moneyline, calibration gate not applicable"

    odds = float(pd.to_numeric(row.get("MyBookieOdds"), errors="coerce"))
    if odds < 2.00:
        return True, 1.0, "Favorite/short price, underdog calibration gate not applicable"
    if odds < 2.50:
        return False, 0.0, "BLOCKED: +100 to +149 bucket had materially negative historical V1.2 ROI"
    if odds < 3.00:
        return True, 1.0, "FULL: +150 to +199 was V1.2's strongest validated underdog bucket"
    if odds < 4.00:
        return True, 0.50, "HALF: +200 to +299 was near break-even and overconfident historically"
    return False, 0.0, "BLOCKED: +300 and longer had materially negative historical V1.2 ROI"


def apply_moneyline_calibration_gate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), pd.DataFrame()

    kept = []
    audit = []
    for _, row in df.iterrows():
        admitted, multiplier, status = _moneyline_calibration(row)
        out = row.to_dict()
        out.update({
            "MoneylineCalibrationPass": admitted,
            "MoneylineStakeMultiplier": multiplier,
            "MoneylineCalibrationStatus": status,
        })
        audit.append(out)
        if admitted:
            kept.append(out)
    return pd.DataFrame(kept), pd.DataFrame(audit)


def apply_post_sizing_multipliers(portfolio: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty:
        return portfolio

    out = portfolio.copy()
    multiplier = pd.to_numeric(out.get("MoneylineStakeMultiplier", 1.0), errors="coerce").fillna(1.0)
    out["PaperStakeAmount"] = pd.to_numeric(out["PaperStakeAmount"], errors="coerce") * multiplier
    out["PaperStakeUnits"] = pd.to_numeric(out["PaperStakeUnits"], errors="coerce") * multiplier
    out["PaperStakeFraction"] = pd.to_numeric(out["PaperStakeFraction"], errors="coerce") * multiplier
    out["ExpectedPaperProfit"] = out["PaperStakeAmount"] * pd.to_numeric(out["ExpectedReturnPerUnit"], errors="coerce")

    bankroll = 100.0
    if len(out):
        bankroll_guess = out["PaperStakeAmount"].sum() / max(out["PaperStakeFraction"].sum(), 1e-12)
        if bankroll_guess > 0:
            bankroll = bankroll_guess
    out["PortfolioExposurePct"] = out["PaperStakeAmount"].cumsum() / bankroll
    out["SizingNote"] = out.get("SizingNote", "").astype(str) + "; historical moneyline calibration multiplier applied"
    return out


def main() -> None:
    cfg = load_config()
    rankings_path = DATA_DIR / "gameweek_rankings_latest.csv"
    v2_path = DATA_DIR / "v2_shadow_predictions_latest.csv"
    odds_path = DATA_DIR / "market_odds_latest.csv"

    rankings = pd.read_csv(rankings_path)
    v2 = pd.read_csv(v2_path) if v2_path.exists() and v2_path.stat().st_size > 0 else pd.DataFrame()
    odds = pd.read_csv(odds_path) if odds_path.exists() and odds_path.stat().st_size > 0 else pd.DataFrame()

    v2_admitted, v2_audit = apply_v2_agreement_gate(rankings, v2, cfg)
    sanity_admitted, extreme_audit = apply_extreme_edge_gate(v2_admitted, odds, cfg)
    calibrated, calibration_audit = apply_moneyline_calibration_gate(sanity_admitted)
    portfolio = build_portfolio(calibrated, cfg)
    portfolio = apply_post_sizing_multipliers(portfolio)

    portfolio.to_csv(DATA_DIR / "paper_portfolio_latest.csv", index=False)
    v2_audit.to_csv(DATA_DIR / "v1_v2_agreement_audit_latest.csv", index=False)
    extreme_audit.to_csv(DATA_DIR / "extreme_edge_audit_latest.csv", index=False)
    calibration_audit.to_csv(DATA_DIR / "moneyline_calibration_audit_latest.csv", index=False)

    exposure = float(pd.to_numeric(portfolio.get("PaperStakeAmount"), errors="coerce").sum()) if not portfolio.empty else 0.0
    expected_profit = float(pd.to_numeric(portfolio.get("ExpectedPaperProfit"), errors="coerce").sum()) if not portfolio.empty else 0.0
    print(f"Final calibrated selections: {len(portfolio)}")
    print(f"Final calibrated exposure: {exposure:.2f}u")
    print(f"Expected model profit: {expected_profit:.2f}u")
    if not portfolio.empty:
        cols = ["PortfolioRank", "HomeTeam", "AwayTeam", "MarketType", "Selection", "MyBookieOdds", "ModelWinProbability", "V2WinProbability", "ExpectedReturnPerUnit", "MoneylineStakeMultiplier", "MoneylineCalibrationStatus", "PaperStakeUnits"]
        print(portfolio[[c for c in cols if c in portfolio.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
