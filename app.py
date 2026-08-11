from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data/processed")
HISTORY_DIR = Path("data/history")

st.set_page_config(page_title="EPL Prediction Engine", page_icon="⚽", layout="wide")


def load_csv_from(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {name}: {exc}")
        return pd.DataFrame()


def load_csv(name: str) -> pd.DataFrame:
    return load_csv_from(DATA_DIR, name)


def load_history(name: str) -> pd.DataFrame:
    return load_csv_from(HISTORY_DIR, name)


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "–"


def dec(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "–"


st.title("⚽ EPL Prediction Engine")
st.caption("Production model V1.2 · xG strength ratings · Dixon-Coles · live market research")

predictions = load_csv("daily_predictions_latest.csv")
market = load_csv("market_comparison_latest.csv")
summary = load_csv("backtest_summary_v12.csv")
comparison = load_csv("model_comparison_v12.csv")
pred_history = load_history("prediction_history.csv")
market_history = load_history("market_comparison_history.csv")
clv = load_history("clv_report.csv")

pred_tab, market_tab, live_tab, clv_tab, perf_tab = st.tabs([
    "Today's Predictions", "Market Comparison", "Live History", "Closing Line", "Model Performance"
])

with pred_tab:
    if predictions.empty:
        st.info("No daily prediction file is available yet. Run the Daily EPL predictions workflow first.")
    else:
        st.subheader("Upcoming EPL fixtures")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fixtures", len(predictions))
        if "Date" in predictions.columns:
            c2.metric("Next match date", str(predictions["Date"].min()))
        if "ModelVersion" in predictions.columns:
            c3.metric("Model", str(predictions["ModelVersion"].iloc[0]))

        for _, row in predictions.iterrows():
            home, away = str(row.get("HomeTeam", "Home")), str(row.get("AwayTeam", "Away"))
            with st.container(border=True):
                st.markdown(f"### {home} vs {away}")
                st.caption(str(row.get("KickoffUTC", row.get("Date", ""))))
                a, b, c = st.columns(3)
                a.metric("Home xG", dec(row.get("Lambda_Home_xG")))
                b.metric("Away xG", dec(row.get("Lambda_Away_xG")))
                c.metric("Expected total", dec(row.get("Lambda_Total_xG")))
                a, b, c = st.columns(3)
                a.metric("Home win", pct(row.get("P_HomeWin")))
                b.metric("Draw", pct(row.get("P_Draw")))
                c.metric("Away win", pct(row.get("P_AwayWin")))
                a, b, c, d = st.columns(4)
                a.metric("BTTS Yes", pct(row.get("P_BTTS_Yes")))
                b.metric("Over 2.5", pct(row.get("CalP_Over2_5", row.get("RawP_Over2_5"))))
                c.metric("Fair O2.5", dec(row.get("FairOdds_Over2_5")))
                d.metric("Fair U2.5", dec(row.get("FairOdds_Under2_5")))
                st.caption(f"Most likely score: {row.get('MostLikelyScore', '–')}")

with market_tab:
    if market.empty:
        st.info("No market comparison file is available yet.")
    else:
        st.subheader("Model vs bookmaker market")
        st.caption("Research view only. No staking or automated betting is performed.")
        edge = pd.to_numeric(market.get("ExpectedReturnPerUnit"), errors="coerce") if "ExpectedReturnPerUnit" in market.columns else pd.Series(dtype=float)
        c1, c2, c3 = st.columns(3)
        c1.metric("Market rows", len(market))
        c2.metric("Positive discrepancies", int((edge > 0).sum()) if not edge.empty else 0)
        if not edge.empty and edge.notna().any():
            c3.metric("Largest expected return", pct(edge.max()))
        preferred = ["Date", "HomeTeam", "AwayTeam", "Market", "Side", "Bookmaker", "ModelProbability", "ModelFairOdds", "MarketOdds", "MarketImpliedProbability", "ProbabilityDifference", "ExpectedReturnPerUnit"]
        cols = [c for c in preferred if c in market.columns]
        st.dataframe(market[cols] if cols else market, use_container_width=True, hide_index=True)

with live_tab:
    st.subheader("Permanent live record")
    if pred_history.empty and market_history.empty:
        st.info("No permanent live-history snapshot has been committed yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction rows", len(pred_history))
        c2.metric("Market comparison rows", len(market_history))
        snapshots = market_history["SnapshotUTC"].nunique() if (not market_history.empty and "SnapshotUTC" in market_history.columns) else 0
        c3.metric("Snapshots", snapshots)
        if not market_history.empty:
            hist = market_history.sort_values("SnapshotUTC", ascending=False) if "SnapshotUTC" in market_history.columns else market_history
            st.dataframe(hist, use_container_width=True, hide_index=True)

with clv_tab:
    st.subheader("Closing-line tracking")
    st.caption("The latest stored pre-kickoff quote is used as a closing proxy. The quality label shows how close that snapshot was to kickoff.")
    if clv.empty:
        st.info("CLV needs multiple market snapshots. The scheduled workflow will build this automatically as prices accumulate.")
    else:
        price_clv = pd.to_numeric(clv.get("PriceCLV"), errors="coerce") if "PriceCLV" in clv.columns else pd.Series(dtype=float)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tracked sides", len(clv))
        c2.metric("Mean price CLV", pct(price_clv.mean()) if not price_clv.empty else "–")
        c3.metric("Positive CLV", int((price_clv > 0).sum()) if not price_clv.empty else 0)
        if "SnapshotsObserved" in clv.columns:
            c4.metric("Max snapshots", int(pd.to_numeric(clv["SnapshotsObserved"], errors="coerce").max()))
        preferred = ["Date", "HomeTeam", "AwayTeam", "Side", "FirstObservedOdds", "ClosingProxyOdds", "PriceCLV", "ModelEdgeAtFirst", "ModelEdgeAtClose", "HoursBeforeKickoff", "ClosingQuality", "SnapshotsObserved"]
        cols = [c for c in preferred if c in clv.columns]
        st.dataframe(clv[cols] if cols else clv, use_container_width=True, hide_index=True)

with perf_tab:
    st.subheader("Historical V1.2 performance")
    if summary.empty:
        st.info("No V1.2 backtest summary is available in data/processed.")
    else:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    if not comparison.empty:
        st.markdown("### V1.2 vs baseline")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

st.divider()
st.caption("Research and decision-support only. Historical backtests and model probabilities do not guarantee future profitability.")
