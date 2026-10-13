"""Market-anchored decisions, paper portfolio and paper ledger for non-EPL leagues.

Reuses the EPL decision stack unchanged:

* ``gameweek_rankings.build_rankings`` blends the model toward the no-vig
  market consensus with the ``decision`` weights (moneyline 0.85, spread 0.40,
  total 0.35, plus the early-season bonus by matchday) and grades every fresh
  MyBookie quote.
* ``portfolio_manager.apply_extreme_edge_gate`` and ``build_portfolio`` apply
  the >=5pp edge gate, quality/EV floors and capped fractional Kelly sizing.

Differences from the EPL, set in ``leagues.league_config``:

* No V2 agreement gate (the V2 shadow model is EPL-only), so extreme edges
  (EV >= ``portfolio.extreme_edge_threshold``) stay research-only.
* ``portfolio.blocked_markets`` keeps markets that lost money in a league's
  backtest (e.g. La Liga and Bundesliga totals) ranked but out of the portfolio.
* Exposure cap per league (``leagues.defaults.portfolio``), with bets already
  open in the league ledger counting against it so daily reruns over a 7-day
  horizon cannot stack exposure on the same weekend.

Bets are paper only and go to ``data/history/leagues/paper_bet_ledger.csv``,
settled from Understat results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bet_tracker import _bet_id, _ensure_columns, _new_entries, _settle_one, _summary
from gameweek_rankings import build_rankings
from leagues import harmonize_team_names
from portfolio_manager import apply_extreme_edge_gate, apply_v2_agreement_gate, build_portfolio

HISTORY_DIR = Path("data/history/leagues")
LEDGER_PATH = HISTORY_DIR / "paper_bet_ledger.csv"
SUMMARY_PATH = HISTORY_DIR / "paper_performance_summary.csv"


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _open_bets(ledger: pd.DataFrame, league: str) -> pd.DataFrame:
    if ledger.empty or "League" not in ledger.columns:
        return ledger.iloc[0:0]
    return ledger[(ledger["League"].astype(str) == league) & (ledger["Result"].astype(str) == "OPEN")]


def build_league_portfolio(
    league: str,
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    cfg: dict,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (rankings, portfolio, extreme_audit) for one league's fixtures."""
    rankings = build_rankings(predictions, odds, cfg)
    if rankings.empty:
        return rankings, pd.DataFrame(), pd.DataFrame()
    rankings.insert(0, "League", league)
    rankings["BetID"] = rankings.apply(_bet_id, axis=1)

    blocked = {str(m) for m in (cfg.get("portfolio", {}).get("blocked_markets") or [])}
    rankings["PortfolioBlocked"] = rankings["MarketType"].astype(str).isin(blocked)
    candidates = rankings[~rankings["PortfolioBlocked"]]
    if candidates.empty:
        return rankings, pd.DataFrame(), pd.DataFrame()

    admitted, _ = apply_v2_agreement_gate(candidates, pd.DataFrame(), cfg)
    sanity, extreme_audit = apply_extreme_edge_gate(admitted, odds, cfg)
    if sanity.empty:
        return rankings, pd.DataFrame(), extreme_audit

    # Bets already open for this league hold their exposure; skip their
    # fixtures and size new bets only from what is left of the cap.
    open_bets = _open_bets(ledger, league)
    held_matches = (
        set(zip(open_bets["Date"].astype(str), open_bets["HomeTeam"], open_bets["AwayTeam"]))
        if not open_bets.empty else set()
    )
    if held_matches:
        keys = list(zip(sanity["Date"].astype(str), sanity["HomeTeam"], sanity["AwayTeam"]))
        sanity = sanity[[k not in held_matches for k in keys]]

    pcfg = dict(cfg.get("portfolio", {}))
    bankroll = float(pcfg.get("paper_bankroll", 100.0))
    cap_pct = float(pcfg.get("max_gameweek_exposure_pct", 0.08))
    held = float(pd.to_numeric(open_bets["StakeAmount"], errors="coerce").sum()) if not open_bets.empty else 0.0
    remaining_pct = max(0.0, cap_pct - (held / bankroll if bankroll > 0 else 0.0))
    sized_cfg = {**cfg, "portfolio": {**pcfg, "max_gameweek_exposure_pct": remaining_pct}}

    portfolio = build_portfolio(sanity, sized_cfg) if remaining_pct > 0 and not sanity.empty else pd.DataFrame()
    if not portfolio.empty:
        portfolio["SizingNote"] = (
            f"paper only; {league}; market-anchored probability; >=5pp decision edge gate; "
            "no V2 gate (EPL-only), extreme edges research-only; capped fractional Kelly; "
            f"league cap {cap_pct:.0%} incl. {held:.2f} already open"
        )
    return rankings, portfolio, extreme_audit


def record_bets(ledger: pd.DataFrame, portfolio: pd.DataFrame, league: str) -> tuple[pd.DataFrame, int]:
    """Append portfolio bets not yet in the ledger; return (ledger, new count)."""
    ledger = _ensure_columns(ledger.copy()) if not ledger.empty else ledger
    if portfolio.empty:
        return ledger, 0
    new = _new_entries(portfolio, ledger)
    if new.empty:
        return ledger, 0
    new.insert(0, "League", league)
    new["TrackingSource"] = "LeaguePortfolio"
    ledger = pd.concat([ledger, new], ignore_index=True, sort=False) if not ledger.empty else new
    return ledger, len(new)


def settle_bets(ledger: pd.DataFrame, results: pd.DataFrame, league: str) -> tuple[pd.DataFrame, int]:
    """Settle this league's open bets from completed results (Date, HomeTeam, AwayTeam, FTHG, FTAG)."""
    if ledger.empty or results.empty:
        return ledger, 0
    open_bets = _open_bets(ledger, league)
    if open_bets.empty:
        return ledger, 0
    res = results.dropna(subset=["FTHG", "FTAG"]).copy()
    res["Date"] = pd.to_datetime(res["Date"]).dt.normalize()
    # Results come straight from Understat; line any odd spellings up with the ledger.
    ref = open_bets[["Date", "HomeTeam", "AwayTeam"]].assign(Date=lambda d: pd.to_datetime(d["Date"]))
    res, _ = harmonize_team_names(ref, res, allow_fuzzy=False)

    # Columns read back from CSV as all-NaN floats must accept text.
    for col in ("Result", "FinalScore", "SettledUTC"):
        ledger[col] = ledger[col].astype(object)

    settled = 0
    for idx, bet in open_bets.iterrows():
        bet_date = pd.to_datetime(bet.get("Date"), errors="coerce")
        match = res[(res["HomeTeam"] == bet["HomeTeam"]) & (res["AwayTeam"] == bet["AwayTeam"])]
        if pd.notna(bet_date):
            match = match[(match["Date"] - bet_date).abs() <= pd.Timedelta(days=1)]
        if match.empty:
            continue
        hg, ag = int(match.iloc[-1]["FTHG"]), int(match.iloc[-1]["FTAG"])
        result, y = _settle_one(bet, hg, ag)
        if result == "OPEN":
            continue
        stake = float(pd.to_numeric(bet.get("StakeUnits"), errors="coerce"))
        odds = float(pd.to_numeric(bet.get("EntryOdds"), errors="coerce"))
        p = float(pd.to_numeric(bet.get("ModelProbability"), errors="coerce"))
        ledger.at[idx, "Result"] = result
        ledger.at[idx, "FinalScore"] = f"{hg}-{ag}"
        ledger.at[idx, "ProfitUnits"] = stake * (odds - 1.0) if result == "W" else -stake if result == "L" else 0.0
        ledger.at[idx, "BrierScore"] = (p - y) ** 2 if np.isfinite(y) else np.nan
        ledger.at[idx, "SettledUTC"] = pd.Timestamp.now(tz="UTC").isoformat()
        settled += 1
    return ledger, settled


def performance_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "League" not in ledger.columns:
        return pd.DataFrame()
    frames = []
    for league, g in [("ALL", ledger)] + list(ledger.groupby("League")):
        s = _summary(g)
        if not s.empty:
            s.insert(0, "League", league)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_ledger(ledger: pd.DataFrame) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    performance_summary(ledger).to_csv(SUMMARY_PATH, index=False)
