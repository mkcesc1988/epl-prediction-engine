from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import load_config

DATA_DIR = Path("data/processed")


def moneyline_bucket(odds: float) -> str:
    if 2.00 <= odds < 2.50:
        return "+100 to +149"
    if 2.50 <= odds < 3.00:
        return "+150 to +199"
    if 3.00 <= odds < 4.00:
        return "+200 to +299"
    if odds >= 4.00:
        return "+300 and longer"
    return "favorite / shorter than +100"


def multiplier_for(row: pd.Series, cfg: dict) -> tuple[float, str]:
    if str(row.get("MarketType", "")) != "Moneyline":
        return 1.0, "not a moneyline, historical moneyline gate not applied"

    odds = float(pd.to_numeric(row.get("MyBookieOdds"), errors="coerce"))
    bucket = moneyline_bucket(odds)
    pcfg = cfg.get("portfolio", {})

    if bucket == "+100 to +149":
        mult = float(pcfg.get("ml_bucket_mult_100_149", 0.0))
    elif bucket == "+150 to +199":
        mult = float(pcfg.get("ml_bucket_mult_150_199", 1.0))
    elif bucket == "+200 to +299":
        mult = float(pcfg.get("ml_bucket_mult_200_299", 0.5))
    elif bucket == "+300 and longer":
        mult = float(pcfg.get("ml_bucket_mult_300_plus", 0.0))
    else:
        mult = 1.0

    return mult, f"historical V1.2 moneyline bucket {bucket}, stake multiplier {mult:.2f}"


def apply_gate(portfolio: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if portfolio.empty:
        return portfolio.copy(), pd.DataFrame()

    pcfg = cfg.get("portfolio", {})
    min_stake_units = float(pcfg.get("min_stake_units", 0.25))
    paper_bankroll = float(pcfg.get("paper_bankroll", 100.0))
    unit_size_pct = float(pcfg.get("unit_size_pct", 0.01))
    unit_amount = paper_bankroll * unit_size_pct

    rows = []
    audit = []
    for _, r in portfolio.iterrows():
        out = r.to_dict()
        original_units = float(pd.to_numeric(r.get("PaperStakeUnits"), errors="coerce"))
        mult, note = multiplier_for(r, cfg)
        adjusted_units = original_units * mult
        admitted = adjusted_units >= min_stake_units - 1e-12

        out.update({
            "PreCalibrationStakeUnits": original_units,
            "MoneylineCalibrationMultiplier": mult,
            "MoneylineCalibrationStatus": note,
            "CalibrationGateAdmitted": admitted,
        })
        audit.append(out.copy())
        if not admitted:
            continue

        adjusted_amount = adjusted_units * unit_amount
        out["PaperStakeUnits"] = adjusted_units
        out["PaperStakeAmount"] = adjusted_amount
        out["PaperStakeFraction"] = adjusted_amount / paper_bankroll if paper_bankroll else 0.0
        out["ExpectedPaperProfit"] = adjusted_amount * float(pd.to_numeric(r.get("ExpectedReturnPerUnit"), errors="coerce"))
        out["SizingNote"] = str(r.get("SizingNote", "")) + "; historical moneyline calibration gate"
        rows.append(out)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.reset_index(drop=True)
        result["PortfolioRank"] = range(1, len(result) + 1)
        result["PortfolioExposurePct"] = result["PaperStakeAmount"].cumsum() / paper_bankroll if paper_bankroll else 0.0
    return result, pd.DataFrame(audit)


def main() -> None:
    cfg = load_config()
    path = DATA_DIR / "paper_portfolio_latest.csv"
    if not path.exists() or path.stat().st_size == 0:
        print("No paper portfolio to calibrate")
        return

    portfolio = pd.read_csv(path)
    adjusted, audit = apply_gate(portfolio, cfg)
    adjusted.to_csv(path, index=False)
    audit_path = DATA_DIR / "moneyline_calibration_gate_latest.csv"
    audit.to_csv(audit_path, index=False)

    print(f"Pre-gate selections: {len(portfolio)}")
    print(f"Post-gate selections: {len(adjusted)}")
    if not adjusted.empty:
        cols = ["Selection", "MyBookieOdds", "PreCalibrationStakeUnits", "MoneylineCalibrationMultiplier", "PaperStakeUnits", "MoneylineCalibrationStatus"]
        print(adjusted[[c for c in cols if c in adjusted.columns]].to_string(index=False))
    print(f"Saved: {path}")
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()
