from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gameweek_rankings import _ev_with_push, _moneyline_prob, _norm, _spread_prob, _totals_prob
from model_v11 import _score_matrix
from pipeline import load_config

DATA_DIR = Path("data/processed")


def _kelly_fraction(p: float, odds: float, push: float = 0.0) -> float:
    if odds <= 1.0 or p <= 0.0:
        return 0.0
    b = odds - 1.0
    q = max(0.0, 1.0 - p - push)
    raw = (b * p - q) / b
    return max(0.0, float(raw))


def _v2_probability(row: pd.Series, v2: pd.Series, cfg: dict) -> tuple[float, float]:
    market = str(row.get("MarketType", ""))
    selection = str(row.get("Selection", ""))
    line_raw = pd.to_numeric(row.get("Line"), errors="coerce")
    line = float(line_raw) if pd.notna(line_raw) else np.nan

    pred = pd.Series({
        "HomeTeam": v2.get("HomeTeam"),
        "AwayTeam": v2.get("AwayTeam"),
        "P_HomeWin": v2.get("V2_P_HomeWin"),
        "P_Draw": v2.get("V2_P_Draw"),
        "P_AwayWin": v2.get("V2_P_AwayWin"),
    })

    if market == "Moneyline":
        return _moneyline_prob(pred, selection)

    lam_h = pd.to_numeric(v2.get("V2_Lambda_Home_xG"), errors="coerce")
    lam_a = pd.to_numeric(v2.get("V2_Lambda_Away_xG"), errors="coerce")
    if pd.isna(lam_h) or pd.isna(lam_a):
        return np.nan, np.nan
    rho = float(pd.to_numeric(v2.get("DixonColes_Rho", -0.08), errors="coerce"))
    max_goal = int(cfg.get("model_v12", {}).get("max_goal", 10))
    matrix = _score_matrix(float(lam_h), float(lam_a), rho, max_goal)

    if market == "Spread":
        if pd.isna(line):
            return np.nan, np.nan
        team = selection
        # Selection is stored as e.g. "Hull City +1.5". Strip the signed line.
        suffix = f" {line:+g}"
        if team.endswith(suffix):
            team = team[: -len(suffix)]
        return _spread_prob(matrix, pred["HomeTeam"], pred["AwayTeam"], team, line)

    if market == "Total":
        if pd.isna(line):
            return np.nan, np.nan
        side = "Over" if selection.lower().startswith("over") else "Under" if selection.lower().startswith("under") else ""
        if not side:
            return np.nan, np.nan
        return _totals_prob(matrix, side, line)

    return np.nan, np.nan


def apply_v2_agreement_gate(rankings: pd.DataFrame, v2_shadow: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pcfg = cfg.get("portfolio", {})
    require = bool(pcfg.get("require_v2_agreement", True))
    min_v2_ev = float(pcfg.get("v2_min_expected_return", 0.0))
    max_gap = float(pcfg.get("v2_max_probability_gap", 0.10))

    if rankings.empty:
        return rankings.copy(), pd.DataFrame()

    if not require:
        out = rankings.copy()
        out["V2Agreement"] = True
        out["V2AdmissionStatus"] = "V2 gate disabled"
        return out, pd.DataFrame()

    if v2_shadow.empty:
        audit = rankings.copy()
        audit["V2Agreement"] = False
        audit["V2AdmissionStatus"] = "RESEARCH_ONLY: missing V2 shadow output"
        return rankings.iloc[0:0].copy(), audit

    v2 = v2_shadow.copy()
    for col in ["HomeTeam", "AwayTeam"]:
        v2[col] = v2[col].map(_norm)
    v2["Date"] = v2["Date"].astype(str)
    v2_map = {(r["Date"], r["HomeTeam"], r["AwayTeam"]): r for _, r in v2.iterrows()}

    admitted: list[dict] = []
    audit_rows: list[dict] = []

    for _, r in rankings.iterrows():
        out = r.to_dict()
        key = (str(r.get("Date", "")), _norm(r.get("HomeTeam")), _norm(r.get("AwayTeam")))
        shadow = v2_map.get(key)

        if shadow is None:
            out.update({
                "V2ModelVersion": pd.NA,
                "V2WinProbability": np.nan,
                "V2PushProbability": np.nan,
                "V2ExpectedReturnPerUnit": np.nan,
                "V1V2ProbabilityGap": np.nan,
                "V2Agreement": False,
                "V2AdmissionStatus": "RESEARCH_ONLY: fixture missing from V2 shadow",
            })
            audit_rows.append(out)
            continue

        p2, push2 = _v2_probability(r, shadow, cfg)
        odds = float(pd.to_numeric(r.get("MyBookieOdds"), errors="coerce"))
        p1 = float(pd.to_numeric(r.get("ModelWinProbability"), errors="coerce"))
        ev2 = _ev_with_push(float(p2), float(push2), odds) if pd.notna(p2) else np.nan
        gap = abs(float(p2) - p1) if pd.notna(p2) else np.nan

        reasons = []
        if pd.isna(p2):
            reasons.append("V2 probability unavailable")
        elif ev2 < min_v2_ev:
            reasons.append(f"V2 EV {ev2:+.1%} below {min_v2_ev:+.1%}")
        if pd.notna(gap) and gap > max_gap:
            reasons.append(f"V1/V2 probability gap {gap:.1%} exceeds {max_gap:.1%}")

        agreed = len(reasons) == 0
        status = "ADMITTED: V1.2 and V2.3 agree" if agreed else "RESEARCH_ONLY: " + "; ".join(reasons)
        out.update({
            "V2ModelVersion": shadow.get("ModelVersion", "V2-shadow"),
            "V2WinProbability": p2,
            "V2PushProbability": push2,
            "V2ExpectedReturnPerUnit": ev2,
            "V1V2ProbabilityGap": gap,
            "V2Agreement": agreed,
            "V2AdmissionStatus": status,
        })
        audit_rows.append(out)
        if agreed:
            admitted.append(out)

    return pd.DataFrame(admitted), pd.DataFrame(audit_rows)


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
        "OverallRankScore", "V2WinProbability", "V2ExpectedReturnPerUnit",
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
            "SizingNote": f"paper only; V1.2 sizing; V2.3 agreement gate; capped fractional Kelly; minimum {min_stake_units:g}u",
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
    v2_path = DATA_DIR / "v2_shadow_predictions_latest.csv"
    if not rankings_path.exists():
        raise RuntimeError("Run gameweek rankings before portfolio manager")

    rankings = pd.read_csv(rankings_path)
    v2 = pd.read_csv(v2_path) if v2_path.exists() and v2_path.stat().st_size > 0 else pd.DataFrame()
    admitted, audit = apply_v2_agreement_gate(rankings, v2, cfg)
    portfolio = build_portfolio(admitted, cfg)

    out_path = DATA_DIR / "paper_portfolio_latest.csv"
    audit_path = DATA_DIR / "v1_v2_agreement_audit_latest.csv"
    portfolio.to_csv(out_path, index=False)
    audit.to_csv(audit_path, index=False)

    pcfg = cfg.get("portfolio", {})
    bankroll = float(pcfg.get("paper_bankroll", 100.0))
    exposure = float(portfolio["PaperStakeAmount"].sum()) if not portfolio.empty else 0.0
    expected_profit = float(portfolio["ExpectedPaperProfit"].sum()) if not portfolio.empty else 0.0
    agreements = int(audit["V2Agreement"].fillna(False).sum()) if not audit.empty and "V2Agreement" in audit.columns else 0

    print(f"V1.2 ranked selections: {len(rankings)}")
    print(f"V1.2/V2.3 agreements:  {agreements}")
    print(f"Research-only rows:    {len(audit) - agreements if not audit.empty else 0}")
    print(f"Paper bankroll:         {bankroll:.2f}")
    print(f"Portfolio selections:  {len(portfolio)}")
    print(f"Paper exposure:        {exposure:.2f} ({(exposure / bankroll if bankroll else 0):.1%})")
    print(f"Expected paper profit: {expected_profit:.2f}")
    if not portfolio.empty:
        cols = [
            "PortfolioRank", "HomeTeam", "AwayTeam", "MarketType", "Selection",
            "MyBookieOdds", "ModelWinProbability", "V2WinProbability",
            "ExpectedReturnPerUnit", "V2ExpectedReturnPerUnit", "V1V2ProbabilityGap",
            "PaperStakeUnits", "ExpectedPaperProfit", "V2AdmissionStatus",
        ]
        print(portfolio[[c for c in cols if c in portfolio.columns]].to_string(index=False))
    print(f"Saved: {out_path}")
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()
