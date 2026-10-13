"""Predictions, odds comparison and backtests for La Liga, Bundesliga, Serie A and Ligue 1.

Runs the same V1.2 xG attack/defense Dixon-Coles model as the EPL engine, fitted
separately on each league's own history. EPL files are never touched; every
league writes to data/processed/leagues/<CODE>/:

  daily_predictions_latest.csv      fixtures in the next ``leagues.horizon_days``
  daily_predictions_<YYYYMMDD>.csv  dated copy
  market_odds_latest.csv            The Odds API prices (when THE_ODDS_API_KEY is set)
  market_comparison_latest.csv      O/U 2.5 model vs market
  moneyline_comparison_latest.csv   1X2 model vs best price and no-vig consensus
  rankings_latest.csv               MyBookie quotes blended toward the market and graded
  paper_portfolio_latest.csv        new paper bets sized by capped fractional Kelly
  extreme_edge_audit_latest.csv     extreme-edge sanity checks
  home_advantage_sweep_summary.csv  (--backtest) 1X2 RPS vs Pinnacle by season

plus combined files in data/processed/leagues/ and a paper ledger with results
in data/history/leagues/ (see league_portfolio.py).

Everything here is paper only: nothing feeds the EPL portfolio, EPL ledgers or
real-money tracking.

Usage:
  python league_predictions.py                    # all enabled leagues
  python league_predictions.py --league SERIEA    # one league
  python league_predictions.py --backtest         # also run walk-forward backtests
  python league_predictions.py --odds             # pull odds now (default: only near kickoff)
  python league_predictions.py --no-odds          # never pull odds
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from daily_predictions import build_daily_predictions, fetch_current_understat_completed
from league_portfolio import (
    LEDGER_PATH,
    build_league_portfolio,
    load_ledger,
    performance_summary,
    record_bets,
    save_ledger,
    settle_bets,
)
from leagues import LEAGUES, enabled_leagues, harmonize_team_names, league_config
from market_comparison import _norm, compare_totals_25, fetch_market_odds
from pipeline import load_config


def align_odds_to_predictions(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Put Odds API rows on the prediction frame's team names and dates.

    Bookmaker spellings that the alias map misses are matched by fixture, and
    kickoff dates are snapped to the prediction date (Understat dates are local
    time, Odds API dates are UTC, so late kickoffs can differ by a day).
    """
    if predictions.empty or odds.empty:
        return odds

    pred_keys = predictions[["Date", "HomeTeam", "AwayTeam"]].copy()
    pred_keys["Date"] = pd.to_datetime(pred_keys["Date"]).dt.normalize()

    out = odds.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out["HomeTeam"] = out["HomeTeam"].map(_norm)
    out["AwayTeam"] = out["AwayTeam"].map(_norm)
    # Moneyline outcomes are team names; put them on the same spelling as the teams.
    is_h2h = out["Market"].eq("h2h") & out["Outcome"].astype(str).str.lower().ne("draw")
    out.loc[is_h2h, "Outcome"] = out.loc[is_h2h, "Outcome"].map(_norm)

    # Only events in the prediction window can be matched; anything later would
    # just invite false name matches.
    lo = pred_keys["Date"].min() - pd.Timedelta(days=1)
    hi = pred_keys["Date"].max() + pd.Timedelta(days=1)
    events = out.loc[out["Date"].between(lo, hi), ["Date", "HomeTeam", "AwayTeam"]].drop_duplicates()
    _, learned = harmonize_team_names(pred_keys, events, allow_fuzzy=False)
    if learned:
        print(f"  Reconciled bookmaker team names: {learned}")
        for col in ("HomeTeam", "AwayTeam"):
            out[col] = out[col].replace(learned)
        out.loc[is_h2h, "Outcome"] = out.loc[is_h2h, "Outcome"].replace(learned)

    date_for_pair = {}
    for d, h, a in pred_keys.itertuples(index=False):
        date_for_pair[(h, a)] = d
    snapped = []
    for d, h, a in out[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False):
        pd_date = date_for_pair.get((h, a))
        snapped.append(pd_date if pd_date is not None and abs((pd_date - d).days) <= 1 else d)
    out["Date"] = pd.to_datetime(pd.Series(snapped, index=out.index)).dt.strftime("%Y-%m-%d")

    unmatched = (
        out[["HomeTeam", "AwayTeam"]]
        .drop_duplicates()
        .merge(pred_keys[["HomeTeam", "AwayTeam"]], how="left", indicator=True)
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    if not unmatched.empty:
        pairs = ", ".join(f"{h} v {a}" for h, a in unmatched[["HomeTeam", "AwayTeam"]].itertuples(index=False))
        print(f"  Odds events with no matching prediction (outside horizon or unmapped name): {pairs}")
    return out


def compare_moneyline(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Model 1X2 probabilities vs the best available price and the no-vig consensus."""
    if predictions.empty or odds.empty:
        return pd.DataFrame()
    h2h = odds[odds["Market"] == "h2h"].copy()
    if h2h.empty:
        return pd.DataFrame()

    def side(row: pd.Series) -> str | None:
        outcome = _norm(row["Outcome"])
        if str(row["Outcome"]).strip().lower() == "draw":
            return "Draw"
        if outcome == row["HomeTeam"]:
            return "Home"
        if outcome == row["AwayTeam"]:
            return "Away"
        return None

    h2h["Side"] = h2h.apply(side, axis=1)
    h2h = h2h.dropna(subset=["Side", "DecimalOdds"])
    h2h = h2h[h2h["DecimalOdds"] > 1.0]
    if h2h.empty:
        return pd.DataFrame()

    keys = ["Date", "HomeTeam", "AwayTeam"]
    # No-vig consensus: de-vig each complete bookmaker book, then take the median.
    books = h2h.pivot_table(index=keys + ["BookmakerKey"], columns="Side", values="DecimalOdds", aggfunc="max")
    books = books.dropna(subset=[c for c in ["Home", "Draw", "Away"] if c in books.columns])
    consensus = pd.DataFrame()
    if not books.empty and {"Home", "Draw", "Away"}.issubset(books.columns):
        implied = 1.0 / books[["Home", "Draw", "Away"]]
        novig = implied.div(implied.sum(axis=1), axis=0)
        grouped = novig.groupby(level=keys)
        consensus = grouped.median().add_prefix("Consensus_")
        consensus["ConsensusBooks"] = grouped.size()
        consensus = consensus.reset_index()

    best_idx = h2h.groupby(keys + ["Side"])["DecimalOdds"].idxmax()
    best = h2h.loc[best_idx, keys + ["Side", "Bookmaker", "DecimalOdds"]]

    pred = predictions.copy()
    pred["Date"] = pred["Date"].astype(str)
    merged = best.merge(pred, on=keys, how="inner")
    if not consensus.empty:
        merged = merged.merge(consensus, on=keys, how="left")

    prob_col = {"Home": "P_HomeWin", "Draw": "P_Draw", "Away": "P_AwayWin"}
    rows = []
    for _, r in merged.iterrows():
        p = float(r[prob_col[r["Side"]]])
        odds_best = float(r["DecimalOdds"])
        cons = r.get(f"Consensus_{r['Side']}")
        cons = float(cons) if cons is not None and pd.notna(cons) else np.nan
        rows.append({
            "League": r.get("League"),
            "Date": r["Date"],
            "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            "Market": "1X2",
            "Side": r["Side"],
            "ModelProbability": p,
            "ModelFairOdds": 1.0 / p if p > 0 else np.nan,
            "ConsensusNoVigProbability": cons,
            "ModelMinusConsensus": p - cons if pd.notna(cons) else np.nan,
            "ConsensusBooks": r.get("ConsensusBooks"),
            "BestBookmaker": r["Bookmaker"],
            "BestMarketOdds": odds_best,
            "BestMarketExpectedReturn": p * odds_best - 1.0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("BestMarketExpectedReturn", ascending=False).reset_index(drop=True)


def run_backtest(lcfg: dict, out_dir: Path) -> str:
    """Walk-forward 1X2 RPS (vs Pinnacle) and O/U 2.5 Brier backtests for one league."""
    from backtest import walk_forward_backtest
    from home_advantage_sweep import run_sweep, verdict
    from model_v12 import build_predictions_v12
    from pipeline import build_master

    master = build_master(lcfg)
    print(f"  Master rows: {len(master)}")

    preds = build_predictions_v12(master, lcfg)
    preds.to_csv(lcfg["paths"]["predictions_v12"], index=False)
    _, summary_v12 = walk_forward_backtest(preds, lcfg)
    summary_v12.to_csv(lcfg["paths"]["summary_v12"], index=False)

    detail, sweep = run_sweep(master, lcfg)
    if sweep.empty:
        return "No matches scored."
    sweep.to_csv(out_dir / "home_advantage_sweep_summary.csv", index=False)
    detail.to_csv(out_dir / "home_advantage_sweep_detail.csv", index=False)

    shrink = float(lcfg.get("model_v12", {}).get("home_advantage_shrink", 0.0))
    overall = sweep[sweep["Season"] == "ALL"].set_index("Shrink")
    row = overall.loc[shrink] if shrink in overall.index else overall.iloc[0]
    gap = float(row["RPS"]) - float(row["Market_RPS"])
    return (
        f"1X2 RPS {row['RPS']:.4f} vs Pinnacle {row['Market_RPS']:.4f} (gap {gap:+.4f}) "
        f"over {int(row['Matches'])} matches. {verdict(sweep)}"
    )


def _odds_due(predictions: pd.DataFrame, cfg: dict) -> bool:
    """True when a fixture kicks off within ``leagues.odds_within_days``."""
    if predictions.empty:
        return False
    within = int((cfg.get("leagues") or {}).get("odds_within_days", 2))
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return bool((pd.to_datetime(predictions["Date"]) <= today + pd.Timedelta(days=within)).any())


def _fill_kickoffs(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Use the bookmakers' UTC kickoff so started matches are never ranked."""
    if predictions.empty or odds.empty or "KickoffUTC" not in odds.columns:
        return predictions
    kick = (
        odds.dropna(subset=["KickoffUTC"])
        .drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
        .set_index(["Date", "HomeTeam", "AwayTeam"])["KickoffUTC"]
    )
    out = predictions.copy()
    keys = list(zip(out["Date"].astype(str), out["HomeTeam"], out["AwayTeam"]))
    found = [kick.get(k) for k in keys]
    out["KickoffUTC"] = [f if f is not None else k for f, k in zip(found, out["KickoffUTC"])]
    return out


def run_league(
    cfg: dict,
    code: str,
    odds_mode: str,
    backtest: bool,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Predict, price, rank and paper-bet one league.

    ``odds_mode`` is "force", "auto" (only near kickoff) or "off".
    Returns (predictions, portfolio, ledger, backtest verdict).
    """
    lcfg = league_config(cfg, code)
    out_dir = Path(lcfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(lcfg["paths"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"\n=== {LEAGUES[code]['name']} ({code}) ===")

    # Settle anything already played before placing new paper bets.
    season = int(lcfg.get("daily", {}).get("current_season", 2026))
    results = fetch_current_understat_completed(season, lcfg["league"])
    ledger, n_settled = settle_bets(ledger, results, code)
    if n_settled:
        print(f"  Settled paper bets: {n_settled}")

    predictions = build_daily_predictions(lcfg)
    if not predictions.empty:
        stamp = str(predictions["Date"].min()).replace("-", "")
        cols = ["Date", "HomeTeam", "AwayTeam", "P_HomeWin", "P_Draw", "P_AwayWin", "CalP_Over2_5", "MostLikelyScore"]
        print(predictions[cols].round(3).to_string(index=False))

    portfolio = pd.DataFrame()
    pull = odds_mode == "force" or (odds_mode == "auto" and _odds_due(predictions, lcfg))
    if pull and not predictions.empty:
        odds = fetch_market_odds(lcfg)
        odds = align_odds_to_predictions(predictions, odds)
        predictions = _fill_kickoffs(predictions, odds)
        odds.to_csv(out_dir / "market_odds_latest.csv", index=False)
        totals = compare_totals_25(predictions, odds)
        if not totals.empty:
            totals.insert(0, "League", code)
        totals.to_csv(out_dir / "market_comparison_latest.csv", index=False)
        moneyline = compare_moneyline(predictions, odds)
        moneyline.to_csv(out_dir / "moneyline_comparison_latest.csv", index=False)

        rankings, portfolio, extreme_audit = build_league_portfolio(code, predictions, odds, lcfg, ledger)
        rankings.to_csv(out_dir / "rankings_latest.csv", index=False)
        portfolio.to_csv(out_dir / "paper_portfolio_latest.csv", index=False)
        extreme_audit.to_csv(out_dir / "extreme_edge_audit_latest.csv", index=False)
        ledger, n_new = record_bets(ledger, portfolio, code)
        eligible = int(rankings["DecisionStatus"].eq("BET_ELIGIBLE").sum()) if not rankings.empty else 0
        print(
            f"  Odds rows: {len(odds)} | O/U 2.5: {len(totals)} | 1X2: {len(moneyline)} | "
            f"MyBookie selections ranked: {len(rankings)} | bet-eligible: {eligible} | "
            f"paper bets: {len(portfolio)} ({n_new} new)"
        )
        if not portfolio.empty:
            show = ["Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "MyBookieOdds",
                    "RawModelWinProbability", "ModelWinProbability", "ConsensusFairProbability",
                    "ExpectedReturnPerUnit", "PaperStakeUnits"]
            print(portfolio[[c for c in show if c in portfolio.columns]].round(3).to_string(index=False))
    elif odds_mode == "auto":
        print("  Odds skipped: no fixture inside leagues.odds_within_days.")

    predictions.to_csv(out_dir / "daily_predictions_latest.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(out_dir / f"daily_predictions_{stamp}.csv", index=False)

    verdict_text = None
    if backtest:
        verdict_text = run_backtest(lcfg, out_dir)
        print(f"  Backtest: {verdict_text}")
    return predictions, portfolio, ledger, verdict_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", action="append", help="League code (repeatable); default: leagues.enabled")
    parser.add_argument("--backtest", action="store_true", help="Also run walk-forward backtests")
    parser.add_argument("--odds", action="store_true", help="Fetch The Odds API prices for this run")
    parser.add_argument("--no-odds", action="store_true", help="Skip The Odds API")
    args = parser.parse_args(argv)

    cfg = load_config()
    codes = [c.upper() for c in args.league] if args.league else enabled_leagues(cfg)
    codes = [c for c in codes if c != "EPL"]
    unknown = [c for c in codes if c not in LEAGUES]
    if unknown:
        parser.error(f"Unknown league(s) {unknown}. Known: {sorted(c for c in LEAGUES if c != 'EPL')}")
    if not codes:
        print("No non-EPL leagues enabled (config.yaml leagues.enabled).")
        return 0

    if args.no_odds:
        odds_mode = "off"
    elif args.odds:
        odds_mode = "force"
    else:
        odds_mode = "auto" if bool((cfg.get("leagues") or {}).get("fetch_odds", True)) else "off"

    ledger = load_ledger()
    frames, portfolios, failures, verdicts = [], [], [], {}
    for code in codes:
        try:
            preds, portfolio, ledger, verdict_text = run_league(cfg, code, odds_mode, args.backtest, ledger)
            if not preds.empty:
                frames.append(preds)
            if not portfolio.empty:
                portfolios.append(portfolio)
            if verdict_text:
                verdicts[code] = verdict_text
        except Exception as exc:  # one league failing must not block the others
            failures.append(code)
            print(f"ERROR: {code} failed: {exc}")
            traceback.print_exc()

    if not ledger.empty:
        save_ledger(ledger)
        summary = performance_summary(ledger)
        open_n = int(ledger["Result"].astype(str).eq("OPEN").sum())
        print(f"\nLeague paper ledger: {len(ledger)} bets, {open_n} open -> {LEDGER_PATH}")
        if not summary.empty:
            print(summary[summary["Segment"] == "ALL"].round(3).to_string(index=False))
    portfolio_path = Path(cfg["paths"]["processed_dir"]) / "leagues" / "all_leagues_paper_portfolio_latest.csv"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    (pd.concat(portfolios, ignore_index=True) if portfolios else pd.DataFrame()).to_csv(portfolio_path, index=False)

    combined_path = Path(cfg["paths"]["processed_dir"]) / "leagues" / "all_leagues_predictions_latest.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined predictions: {len(combined)} fixtures -> {combined_path}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("## League predictions\n\n")
            fh.write(f"Fixtures predicted: {len(combined)}\n\n")
            fh.write(f"New paper bets this run: {sum(len(p) for p in portfolios)}\n\n")
            for code, text in verdicts.items():
                fh.write(f"- **{LEAGUES[code]['name']}**: {text}\n")
            if failures:
                fh.write(f"\nFailed: {', '.join(failures)}\n")

    if failures and len(failures) == len(codes):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
