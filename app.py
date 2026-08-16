from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

DATA_DIR = Path("data/processed")
HISTORY_DIR = Path("data/history")

st.set_page_config(page_title="EPL Prediction Engine", page_icon="⚽", layout="wide")


def load_csv_from(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not read {name}: {exc}")
        return pd.DataFrame()


def load_csv(name: str) -> pd.DataFrame:
    return load_csv_from(DATA_DIR, name)


def load_history(name: str) -> pd.DataFrame:
    return load_csv_from(HISTORY_DIR, name)


def pct(value: object, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "–"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "–"


def dec(value: object, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "–"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "–"


def num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def pick_explanation(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    home = str(row.get("HomeTeam", "Home"))
    away = str(row.get("AwayTeam", "Away"))
    selection = str(row.get("Selection", ""))
    market = str(row.get("MarketType", ""))
    model_p = pd.to_numeric(row.get("ModelWinProbability"), errors="coerce")
    implied = pd.to_numeric(row.get("MyBookieImpliedProbability"), errors="coerce")
    fair = pd.to_numeric(row.get("ModelFairOdds"), errors="coerce")
    odds = pd.to_numeric(row.get("MyBookieOdds"), errors="coerce")
    ev = pd.to_numeric(row.get("ExpectedReturnPerUnit"), errors="coerce")
    lam_h = pd.to_numeric(row.get("Lambda_Home_xG"), errors="coerce")
    lam_a = pd.to_numeric(row.get("Lambda_Away_xG"), errors="coerce")

    if pd.notna(model_p):
        reasons.append(f"Model win probability: {model_p * 100:.1f}% for {selection}.")
    if pd.notna(implied):
        reasons.append(f"MyBookie implied probability: {implied * 100:.1f}%.")
    if pd.notna(model_p) and pd.notna(implied):
        reasons.append(f"Probability edge: {(model_p - implied) * 100:+.1f} percentage points.")
    if pd.notna(fair) and pd.notna(odds):
        reasons.append(f"Model fair odds: {fair:.2f}, MyBookie price: {odds:.2f}.")
    if pd.notna(ev):
        reasons.append(f"Expected return at this price: {ev * 100:+.1f}% per unit risked.")
    if pd.notna(lam_h) and pd.notna(lam_a):
        reasons.append(f"Projected xG: {home} {lam_h:.2f}, {away} {lam_a:.2f}.")
        if market == "Total":
            reasons.append(f"Projected total xG: {lam_h + lam_a:.2f}.")
    validation = str(row.get("ValidationStatus", "") or "")
    if validation:
        reasons.append(f"Validation: {validation}.")
    return reasons


def performance_breakdown(settled: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if settled.empty or group_col not in settled.columns:
        return pd.DataFrame()
    rows = []
    for key, grp in settled.groupby(group_col, dropna=False):
        stake = num_series(grp, "StakeUnits").sum()
        profit = num_series(grp, "ProfitUnits").sum()
        wins = (grp.get("Result", pd.Series(dtype=str)).astype(str) == "W").sum()
        brier = num_series(grp, "BrierScore").mean()
        clv = num_series(grp, "PriceCLV").mean()
        rows.append({
            group_col: key,
            "Bets": len(grp),
            "Wins": int(wins),
            "WinRate": wins / len(grp) if len(grp) else 0.0,
            "StakeUnits": stake,
            "ProfitUnits": profit,
            "ROI": profit / stake if stake > 0 else 0.0,
            "AvgPriceCLV": clv,
            "BrierScore": brier,
        })
    return pd.DataFrame(rows).sort_values("ProfitUnits", ascending=False)


st.title("⚽ EPL Prediction Engine")
st.caption("Production model V1.2 · xG strength ratings · Dixon-Coles · MyBookie market research")

predictions = load_csv("daily_predictions_latest.csv")
rankings = load_csv("gameweek_rankings_latest.csv")
portfolio = load_csv("paper_portfolio_latest.csv")
market = load_csv("market_comparison_latest.csv")
pred_history = load_history("prediction_history.csv")
market_history = load_history("market_comparison_history.csv")
clv = load_history("clv_report.csv")
ledger = load_history("auto_bet_ledger.csv")
perf_summary = load_history("bet_performance_summary.csv")
backtest_summary = load_csv("backtest_summary_v12.csv")
model_comparison = load_csv("model_comparison_v12.csv")

pred_tab, rank_tab, portfolio_tab, market_tab, live_tab, clv_tab, perf_tab = st.tabs([
    "Today's Predictions", "Gameweek Rankings", "Paper Portfolio", "Market Comparison",
    "Live History", "Closing Line", "Model Performance",
])

with pred_tab:
    st.subheader("Upcoming EPL fixtures")
    if predictions.empty:
        st.info("No daily prediction file is available yet.")
    else:
        for _, row in predictions.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}")
                st.caption(str(row.get("KickoffUTC", row.get("Date", ""))))
                a, b, c = st.columns(3)
                a.metric("Home xG", dec(row.get("Lambda_Home_xG")))
                b.metric("Away xG", dec(row.get("Lambda_Away_xG")))
                c.metric("Total xG", dec(row.get("Lambda_Total_xG")))
                a, b, c = st.columns(3)
                a.metric("Home win", pct(row.get("P_HomeWin")))
                b.metric("Draw", pct(row.get("P_Draw")))
                c.metric("Away win", pct(row.get("P_AwayWin")))
                a, b, c = st.columns(3)
                a.metric("BTTS Yes", pct(row.get("P_BTTS_Yes")))
                b.metric("Over 2.5", pct(row.get("CalP_Over2_5", row.get("RawP_Over2_5"))))
                c.metric("Most likely", str(row.get("MostLikelyScore", "–")))

with rank_tab:
    st.subheader("Gameweek MyBookie rankings")
    st.caption("Bet Quality Score is a ranking heuristic, not a literal probability.")
    if rankings.empty:
        st.info("No gameweek ranking file is available yet.")
    else:
        view = rankings.copy()
        if "ExpectedReturnPerUnit" in view.columns:
            positive = st.toggle("Positive expected return only", value=True)
            if positive:
                view = view[num_series(view, "ExpectedReturnPerUnit") > 0]
        preferred = ["GameweekRank", "Grade", "Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "MyBookieOdds", "ModelWinProbability", "ProbabilityEdge", "ExpectedReturnPerUnit", "BetQualityScore", "ProfitabilityScore", "OverallRankScore", "ValidationStatus"]
        st.dataframe(view[[c for c in preferred if c in view.columns]], use_container_width=True, hide_index=True)
        st.markdown("### Top picks")
        for _, row in view.head(5).iterrows():
            with st.container(border=True):
                st.markdown(f"**{row.get('MarketType', '')}: {row.get('Selection', '')}**")
                st.write(f"{row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}")
                a, b, c, d = st.columns(4)
                a.metric("MyBookie", dec(row.get("MyBookieOdds")))
                b.metric("Model P", pct(row.get("ModelWinProbability")))
                c.metric("Bet quality", f"{float(row.get('BetQualityScore', 0)):.1f}/100")
                d.metric("EV", pct(row.get("ExpectedReturnPerUnit")))
                with st.expander("Why this pick?"):
                    for reason in pick_explanation(row): st.write(f"• {reason}")

with portfolio_tab:
    st.subheader("Conservative paper portfolio")
    if portfolio.empty:
        st.info("No paper portfolio is available yet.")
    else:
        exposure = num_series(portfolio, "PaperStakeUnits").sum()
        exp_profit = num_series(portfolio, "ExpectedPaperProfit").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Selections", len(portfolio)); c2.metric("Exposure", f"{exposure:.2f}u"); c3.metric("Expected profit", f"{exp_profit:.2f}u")
        preferred = ["PortfolioRank", "Grade", "Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "MyBookieOdds", "ModelWinProbability", "BetQualityScore", "ExpectedReturnPerUnit", "PaperStakeUnits", "ExpectedPaperProfit"]
        st.dataframe(portfolio[[c for c in preferred if c in portfolio.columns]], use_container_width=True, hide_index=True)

with market_tab:
    st.subheader("Model vs MyBookie and broader market")
    if market.empty: st.info("No market comparison file is available yet.")
    else: st.dataframe(market, use_container_width=True, hide_index=True)

with live_tab:
    st.subheader("Permanent live record")
    if pred_history.empty and market_history.empty: st.info("No permanent history has been stored yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction rows", len(pred_history)); c2.metric("Market rows", len(market_history))
        snapshots = market_history["SnapshotUTC"].nunique() if "SnapshotUTC" in market_history.columns else 0
        c3.metric("Snapshots", snapshots)
        if not market_history.empty: st.dataframe(market_history.tail(200), use_container_width=True, hide_index=True)

with clv_tab:
    st.subheader("Closing-line tracking")
    if clv.empty: st.info("CLV needs multiple pre-kickoff market snapshots.")
    else:
        price_clv = num_series(clv, "PriceCLV")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tracked sides", len(clv)); c2.metric("Mean price CLV", pct(price_clv.mean())); c3.metric("Positive CLV", int((price_clv > 0).sum()))
        st.dataframe(clv, use_container_width=True, hide_index=True)

with perf_tab:
    st.subheader("Live V1.2 betting performance")
    st.caption("This is the permanent out-of-sample tracker for bets admitted by the paper portfolio.")
    if ledger.empty:
        st.info("No automatic tracked bets are available yet. Run EPL market comparison first.")
    else:
        work = ledger.copy()
        result = work.get("Result", pd.Series("OPEN", index=work.index)).fillna("OPEN").astype(str)
        open_bets = work[result == "OPEN"].copy(); settled = work[result.isin(["W", "L", "PUSH"])].copy()
        tracked = len(work); open_count = len(open_bets); settled_count = len(settled)
        open_exposure = num_series(open_bets, "StakeUnits").sum(); profit_units = num_series(settled, "ProfitUnits").sum(); settled_stake = num_series(settled, "StakeUnits").sum()
        roi = profit_units / settled_stake if settled_stake > 0 else None
        win_rate = ((settled.get("Result", pd.Series(dtype=str)).astype(str) == "W").mean() if settled_count else None)
        avg_clv = num_series(work, "PriceCLV").mean() if "PriceCLV" in work.columns else None
        brier = num_series(settled, "BrierScore").mean() if settled_count else None
        r1 = st.columns(4); r1[0].metric("Tracked bets", tracked); r1[1].metric("Open", open_count); r1[2].metric("Settled", settled_count); r1[3].metric("Open exposure", f"{open_exposure:.2f}u")
        r2 = st.columns(4); r2[0].metric("Realized units", f"{profit_units:+.2f}u"); r2[1].metric("ROI", pct(roi) if roi is not None else "Waiting"); r2[2].metric("Win rate", pct(win_rate) if win_rate is not None else "Waiting"); r2[3].metric("Avg price CLV", pct(avg_clv) if avg_clv is not None and pd.notna(avg_clv) else "Waiting")
        r3 = st.columns(2); r3[0].metric("Brier score", dec(brier, 3) if brier is not None and pd.notna(brier) else "Waiting"); r3[1].metric("Model version", str(work.get("ModelVersion", pd.Series(["V1.2"])).iloc[0]))
        st.markdown("### Open tracked bets")
        if open_bets.empty: st.info("No open bets.")
        else:
            cols = ["Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "EntryOdds", "ModelProbability", "ProbabilityEdge", "StakeUnits", "ClosingOdds", "PriceCLV", "ConfidenceBucket", "EdgeBucket", "Grade"]
            st.dataframe(open_bets[[c for c in cols if c in open_bets.columns]], use_container_width=True, hide_index=True)
        if settled.empty:
            st.info("No EPL tracked bets have settled yet. ROI, win rate and Brier score will populate automatically after results are available.")
        else:
            settled = settled.copy(); settled["SettledOrder"] = pd.to_datetime(settled.get("SettledUTC"), errors="coerce"); settled = settled.sort_values(["SettledOrder", "Date"], na_position="last"); settled["CumulativeUnits"] = num_series(settled, "ProfitUnits").fillna(0).cumsum()
            st.markdown("### Cumulative units"); chart = settled[["CumulativeUnits"]].copy(); chart.index = range(1, len(chart) + 1); st.line_chart(chart)
            st.markdown("### Settled bets"); cols = ["Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "EntryOdds", "ClosingOdds", "StakeUnits", "Result", "FinalScore", "ProfitUnits", "PriceCLV", "BrierScore", "ConfidenceBucket", "EdgeBucket"]
            st.dataframe(settled[[c for c in cols if c in settled.columns]], use_container_width=True, hide_index=True)
            for group_col, title in [("ConfidenceBucket", "Performance by confidence"), ("EdgeBucket", "Performance by edge"), ("MarketType", "Performance by market")]:
                breakdown = performance_breakdown(settled, group_col)
                if not breakdown.empty:
                    st.markdown(f"### {title}"); display = breakdown.copy(); display["WinRate"] = display["WinRate"].map(lambda x: f"{x:.1%}"); display["ROI"] = display["ROI"].map(lambda x: f"{x:.1%}"); display["AvgPriceCLV"] = display["AvgPriceCLV"].map(lambda x: "–" if pd.isna(x) else f"{x:.1%}"); st.dataframe(display, use_container_width=True, hide_index=True)

    if not perf_summary.empty:
        st.markdown("### Latest tracker summary")
        st.dataframe(perf_summary, use_container_width=True, hide_index=True)

    with st.expander("Historical V1.2 backtest (research only)"):
        if backtest_summary.empty:
            st.info("No V1.2 backtest summary is available in data/processed.")
        else:
            summary = backtest_summary.iloc[0]
            c1, c2, c3 = st.columns(3); c1.metric("Fixtures", int(summary.get("Fixtures", 0))); c2.metric("Log loss", dec(summary.get("LogLoss"), 3)); c3.metric("Brier score", dec(summary.get("BrierScore"), 3))
            c1, c2, c3 = st.columns(3); c1.metric("Home calibration MAE", pct(summary.get("HomeCalibrationMAE"))); c2.metric("Draw calibration MAE", pct(summary.get("DrawCalibrationMAE"))); c3.metric("Away calibration MAE", pct(summary.get("AwayCalibrationMAE")))
            if not model_comparison.empty:
                st.markdown("### Baseline vs V1.2"); st.dataframe(model_comparison, use_container_width=True, hide_index=True)

st.divider()
st.caption("Research and decision-support only. Bet Quality Score is a heuristic ranking metric, not a probability. Paper sizing and model-implied expected profit do not guarantee future profit.")