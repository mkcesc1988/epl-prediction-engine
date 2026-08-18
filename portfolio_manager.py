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


def _is_mybookie_odds(df: pd.DataFrame) -> pd.Series:
    title = df.get("Bookmaker", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    key = df.get("BookmakerKey", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    return title.str.contains("mybookie", na=False) | key.str.contains("mybookie", na=False)


def _market_key(row: pd.Series) -> str:
    return {"Moneyline": "h2h", "Spread": "spreads", "Total": "totals"}.get(str(row.get("MarketType", "")), "")


def _selection_outcome(row: pd.Series) -> str:
    market = str(row.get("MarketType", ""))
    selection = str(row.get("Selection", ""))
    line_raw = pd.to_numeric(row.get("Line"), errors="coerce")
    if market == "Spread" and pd.notna(line_raw):
        suffix = f" {float(line_raw):+g}"
        if selection.endswith(suffix):
            return selection[: -len(suffix)]
    if market == "Total":
        if selection.lower().startswith("over"):
            return "Over"
        if selection.lower().startswith("under"):
            return "Under"
    return selection


def _matching_market_rows(row: pd.Series, odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds.copy()
    work = odds.copy()
    work["Date"] = work["Date"].astype(str)
    work["HomeTeam"] = work["HomeTeam"].map(_norm)
    work["AwayTeam"] = work["AwayTeam"].map(_norm)
    market = _market_key(row)
    if not market:
        return work.iloc[0:0].copy()
    mask = (
        work["Date"].eq(str(row.get("Date", "")))
        & work["HomeTeam"].eq(_norm(row.get("HomeTeam")))
        & work["AwayTeam"].eq(_norm(row.get("AwayTeam")))
        & work["Market"].astype(str).eq(market)
    )
    return work.loc[mask].copy()


def _line_pair_valid(row: pd.Series, odds: pd.DataFrame) -> bool:
    rows = _matching_market_rows(row, odds)
    if rows.empty:
        return False
    rows = rows[_is_mybookie_odds(rows)].copy()
    if rows.empty:
        return False

    market = str(row.get("MarketType", ""))
    line_raw = pd.to_numeric(row.get("Line"), errors="coerce")
    line = float(line_raw) if pd.notna(line_raw) else np.nan
    outcome = _selection_outcome(row)

    if market == "Moneyline":
        outs = rows["Outcome"].astype(str).str.lower()
        needed = {_norm(row.get("HomeTeam")).lower(), _norm(row.get("AwayTeam")).lower(), "draw"}
        observed = {_norm(x).lower() if str(x).lower() != "draw" else "draw" for x in rows["Outcome"].astype(str)}
        return needed.issubset(observed)

    points = pd.to_numeric(rows.get("Point"), errors="coerce")
    if market == "Spread" and pd.notna(line):
        other_team = _norm(row.get("AwayTeam")) if _norm(outcome) == _norm(row.get("HomeTeam")) else _norm(row.get("HomeTeam"))
        counterpart = rows[
            rows["Outcome"].map(_norm).eq(other_team)
            & np.isclose(points, -line, atol=1e-9, equal_nan=False)
        ]
        return not counterpart.empty

    if market == "Total" and pd.notna(line):
        opposite = "Under" if outcome == "Over" else "Over"
        counterpart = rows[
            rows["Outcome"].astype(str).str.title().eq(opposite)
            & np.isclose(points, line, atol=1e-9, equal_nan=False)
        ]
        return not counterpart.empty

    return False


def _consensus_metrics(row: pd.Series, odds: pd.DataFrame) -> dict:
    rows = _matching_market_rows(row, odds)
    if rows.empty:
        return {"books": 0, "median_odds": np.nan, "median_implied": np.nan, "price_deviation": np.nan}

    rows = rows[~_is_mybookie_odds(rows)].copy()
    if rows.empty:
        return {"books": 0, "median_odds": np.nan, "median_implied": np.nan, "price_deviation": np.nan}

    outcome = _selection_outcome(row)
    market = str(row.get("MarketType", ""))
    line_raw = pd.to_numeric(row.get("Line"), errors="coerce")
    line = float(line_raw) if pd.notna(line_raw) else np.nan

    if market == "Moneyline":
        if outcome.lower() == "draw":
            rows = rows[rows["Outcome"].astype(str).str.lower().eq("draw")]
        else:
            rows = rows[rows["Outcome"].map(_norm).eq(_norm(outcome))]
    elif market == "Spread" and pd.notna(line):
        points = pd.to_numeric(rows.get("Point"), errors="coerce")
        rows = rows[rows["Outcome"].map(_norm).eq(_norm(outcome)) & np.isclose(points, line, atol=1e-9, equal_nan=False)]
    elif market == "Total" and pd.notna(line):
        points = pd.to_numeric(rows.get("Point"), errors="coerce")
        rows = rows[rows["Outcome"].astype(str).str.title().eq(outcome) & np.isclose(points, line, atol=1e-9, equal_nan=False)]
    else:
        rows = rows.iloc[0:0]

    prices = pd.to_numeric(rows.get("DecimalOdds"), errors="coerce")
    rows = rows[(prices > 1.0) & prices.notna()].copy()
    if rows.empty:
        return {"books": 0, "median_odds": np.nan, "median_implied": np.nan, "price_deviation": np.nan}

    prices = pd.to_numeric(rows["DecimalOdds"], errors="coerce")
    median_odds = float(prices.median())
    my_odds = float(pd.to_numeric(row.get("MyBookieOdds"), errors="coerce"))
    book_id = rows.get("BookmakerKey", rows.get("Bookmaker", pd.Series(index=rows.index, dtype=object))).astype(str)
    books = int(book_id.nunique())
    return {
        "books": books,
        "median_odds": median_odds,
        "median_implied": 1.0 / median_odds,
        "price_deviation": abs(my_odds - median_odds) / median_odds,
    }


def apply_extreme_edge_gate(admitted: pd.DataFrame, odds: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if admitted.empty:
        return admitted.copy(), pd.DataFrame()

    pcfg = cfg.get("portfolio", {})
    threshold = float(pcfg.get("extreme_edge_threshold", 0.25))
    max_vgap = float(pcfg.get("extreme_max_v1_v2_probability_gap", 0.03))
    min_books = int(pcfg.get("extreme_min_consensus_books", 3))
    max_market_gap = float(pcfg.get("extreme_max_model_consensus_probability_gap", 0.18))
    max_price_dev = float(pcfg.get("extreme_max_mybookie_price_deviation_pct", 0.20))

    kept: list[dict] = []
    audit_rows: list[dict] = []

    for _, r in admitted.iterrows():
        out = r.to_dict()
        ev = float(pd.to_numeric(r.get("ExpectedReturnPerUnit"), errors="coerce"))
        extreme = ev >= threshold
        line_ok = _line_pair_valid(r, odds) if extreme else True
        consensus = _consensus_metrics(r, odds) if extreme else {
            "books": 0, "median_odds": np.nan, "median_implied": np.nan, "price_deviation": np.nan
        }
        p1 = float(pd.to_numeric(r.get("ModelWinProbability"), errors="coerce"))
        vgap = float(pd.to_numeric(r.get("V1V2ProbabilityGap"), errors="coerce")) if pd.notna(r.get("V1V2ProbabilityGap")) else np.nan
        market_gap = abs(p1 - consensus["median_implied"]) if pd.notna(consensus["median_implied"]) else np.nan

        reasons: list[str] = []
        if extreme:
            if pd.isna(vgap) or vgap > max_vgap:
                reasons.append(f"extreme-edge V1/V2 gap {vgap:.1%} exceeds {max_vgap:.1%}" if pd.notna(vgap) else "extreme-edge V1/V2 gap unavailable")
            if not line_ok:
                reasons.append("market line/counterpart validation failed")
            if consensus["books"] < min_books:
                reasons.append(f"only {consensus['books']} non-MyBookie consensus books, need {min_books}")
            if pd.isna(market_gap) or market_gap > max_market_gap:
                reasons.append(f"model/consensus probability gap {market_gap:.1%} exceeds {max_market_gap:.1%}" if pd.notna(market_gap) else "consensus probability unavailable")
            if pd.isna(consensus["price_deviation"]) or consensus["price_deviation"] > max_price_dev:
                reasons.append(f"MyBookie price deviation {consensus['price_deviation']:.1%} exceeds {max_price_dev:.1%}" if pd.notna(consensus["price_deviation"]) else "consensus price unavailable")

        passed = len(reasons) == 0
        if not extreme:
            status = "PASS: normal edge, extreme audit not required"
        elif passed:
            status = "PASS: extreme edge validated by V2, line pairing, and broader market"
        else:
            status = "RESEARCH_ONLY: " + "; ".join(reasons)

        out.update({
            "ExtremeEdgeFlag": extreme,
            "ExtremeEdgeThreshold": threshold,
            "ExtremeLinePairValid": line_ok,
            "ConsensusBookCount": consensus["books"],
            "ConsensusMedianOdds": consensus["median_odds"],
            "ConsensusMedianImpliedProbability": consensus["median_implied"],
            "ModelConsensusProbabilityGap": market_gap,
            "MyBookieConsensusPriceDeviationPct": consensus["price_deviation"],
            "ExtremeEdgeSanityPass": passed,
            "ExtremeEdgeSanityStatus": status,
        })
        audit_rows.append(out)
        if passed:
            kept.append(out)

    return pd.DataFrame(kept), pd.DataFrame(audit_rows)


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
            "SizingNote": f"paper only; V1.2 sizing; V2.3 agreement + extreme-edge sanity gates; capped fractional Kelly; minimum {min_stake_units:g}u",
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
    odds_path = DATA_DIR / "market_odds_latest.csv"
    if not rankings_path.exists():
        raise RuntimeError("Run gameweek rankings before portfolio manager")

    rankings = pd.read_csv(rankings_path)
    v2 = pd.read_csv(v2_path) if v2_path.exists() and v2_path.stat().st_size > 0 else pd.DataFrame()
    odds = pd.read_csv(odds_path) if odds_path.exists() and odds_path.stat().st_size > 0 else pd.DataFrame()

    v2_admitted, v2_audit = apply_v2_agreement_gate(rankings, v2, cfg)
    sanity_admitted, extreme_audit = apply_extreme_edge_gate(v2_admitted, odds, cfg)
    portfolio = build_portfolio(sanity_admitted, cfg)

    out_path = DATA_DIR / "paper_portfolio_latest.csv"
    v2_audit_path = DATA_DIR / "v1_v2_agreement_audit_latest.csv"
    extreme_audit_path = DATA_DIR / "extreme_edge_audit_latest.csv"
    portfolio.to_csv(out_path, index=False)
    v2_audit.to_csv(v2_audit_path, index=False)
    extreme_audit.to_csv(extreme_audit_path, index=False)

    pcfg = cfg.get("portfolio", {})
    bankroll = float(pcfg.get("paper_bankroll", 100.0))
    exposure = float(portfolio["PaperStakeAmount"].sum()) if not portfolio.empty else 0.0
    expected_profit = float(portfolio["ExpectedPaperProfit"].sum()) if not portfolio.empty else 0.0
    agreements = int(v2_audit["V2Agreement"].fillna(False).sum()) if not v2_audit.empty and "V2Agreement" in v2_audit.columns else 0
    extreme_flags = int(extreme_audit["ExtremeEdgeFlag"].fillna(False).sum()) if not extreme_audit.empty else 0
    extreme_passes = int((extreme_audit.get("ExtremeEdgeFlag", False) & extreme_audit.get("ExtremeEdgeSanityPass", False)).sum()) if not extreme_audit.empty else 0

    print(f"V1.2 ranked selections: {len(rankings)}")
    print(f"V1.2/V2.3 agreements:  {agreements}")
    print(f"Extreme edges audited: {extreme_flags}")
    print(f"Extreme edges passed:  {extreme_passes}")
    print(f"Paper bankroll:         {bankroll:.2f}")
    print(f"Portfolio selections:  {len(portfolio)}")
    print(f"Paper exposure:        {exposure:.2f} ({(exposure / bankroll if bankroll else 0):.1%})")
    print(f"Expected paper profit: {expected_profit:.2f}")
    if not portfolio.empty:
        cols = [
            "PortfolioRank", "HomeTeam", "AwayTeam", "MarketType", "Selection",
            "MyBookieOdds", "ModelWinProbability", "V2WinProbability",
            "ExpectedReturnPerUnit", "V2ExpectedReturnPerUnit", "V1V2ProbabilityGap",
            "ExtremeEdgeFlag", "ConsensusBookCount", "ConsensusMedianOdds",
            "ModelConsensusProbabilityGap", "MyBookieConsensusPriceDeviationPct",
            "PaperStakeUnits", "ExpectedPaperProfit", "ExtremeEdgeSanityStatus",
        ]
        print(portfolio[[c for c in cols if c in portfolio.columns]].to_string(index=False))
    print(f"Saved: {out_path}")
    print(f"Saved: {v2_audit_path}")
    print(f"Saved: {extreme_audit_path}")


if __name__ == "__main__":
    main()
