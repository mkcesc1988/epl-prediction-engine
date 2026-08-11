from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data/processed")

st.set_page_config(
    page_title="EPL Prediction Engine",
    page_icon="⚽",
    layout="wide",
)


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {name}: {exc}")
        return pd.DataFrame()


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


def probability_bar(value: object) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


st.title("⚽ EPL Prediction Engine")
st.caption("Production model V1.2 · xG strength ratings · Dixon-Coles · live market research")

predictions = load_csv("daily_predictions_latest.csv")
market = load_csv("market_comparison_latest.csv")
summary = load_csv("backtest_summary_v12.csv")
comparison = load_csv("model_comparison_v12.csv")

pred_tab, market_tab, perf_tab = st.tabs([
    "Today's Predictions",
    "Market Comparison",
    "Model Performance",
])

with pred_tab:
    if predictions.empty:
        st.info("No daily prediction file is available yet. Run the Daily EPL predictions workflow first.")
    else:
        st.subheader("Upcoming EPL fixtures")
        fixture_count = len(predictions)
        c1, c2, c3 = st.columns(3)
        c1.metric("Fixtures", fixture_count)
        if "Date" in predictions.columns:
            c2.metric("Next match date", str(predictions["Date"].min()))
        if "ModelVersion" in predictions.columns:
            c3.metric("Model", str(predictions["ModelVersion"].iloc[0]))

        for _, row in predictions.iterrows():
            home = str(row.get("HomeTeam", "Home"))
            away = str(row.get("AwayTeam", "Away"))
            kickoff = row.get("KickoffUTC", row.get("Date", ""))
            with st.container(border=True):
                st.markdown(f"### {home} vs {away}")
                st.caption(str(kickoff))

                a, b, c = st.columns(3)
                a.metric("Home xG", dec(row.get("Lambda_Home_xG")))
                b.metric("Away xG", dec(row.get("Lambda_Away_xG")))
                c.metric("Expected total", dec(row.get("Lambda_Total_xG")))

                st.markdown("**1X2**")
                a, b, c = st.columns(3)
                a.metric("Home win", pct(row.get("P_HomeWin")))
                b.metric("Draw", pct(row.get("P_Draw")))
                c.metric("Away win", pct(row.get("P_AwayWin")))

                st.markdown("**Goals markets**")
                a, b, c, d = st.columns(4)
                a.metric("BTTS Yes", pct(row.get("P_BTTS_Yes")))
                b.metric("Over 2.5", pct(row.get("CalP_Over2_5", row.get("RawP_Over2_5"))))
                c.metric("Fair O2.5", dec(row.get("FairOdds_Over2_5")))
                d.metric("Fair U2.5", dec(row.get("FairOdds_Under2_5")))

                if "MostLikelyScore" in row.index:
                    st.caption(f"Most likely score: {row.get('MostLikelyScore', '–')}")

with market_tab:
    if market.empty:
        st.info("No market comparison file is available yet. Run the EPL market comparison workflow first.")
    else:
        st.subheader("Model vs bookmaker market")
        st.caption("Research view only. No staking or automated betting is performed.")

        edge_col = None
        for candidate in ["ExpectedReturnPerUnit", "ExpectedValue", "EV", "ProbabilityDifference"]:
            if candidate in market.columns:
                edge_col = candidate
                break

        if edge_col is not None:
            clean_edge = pd.to_numeric(market[edge_col], errors="coerce")
            positive = int((clean_edge > 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Market rows", len(market))
            c2.metric("Positive discrepancies", positive)
            if clean_edge.notna().any():
                c3.metric("Largest discrepancy", pct(clean_edge.max()))

        display = market.copy()
        preferred = [
            "Date", "HomeTeam", "AwayTeam", "Market", "Selection", "Bookmaker",
            "ModelProbability", "ModelFairOdds", "BookmakerOdds", "ImpliedProbability",
            "ProbabilityDifference", "ExpectedReturnPerUnit",
        ]
        columns = [c for c in preferred if c in display.columns]
        if not columns:
            columns = list(display.columns)
        st.dataframe(display[columns], use_container_width=True, hide_index=True)

with perf_tab:
    st.subheader("Historical V1.2 performance")

    if summary.empty:
        st.info("No V1.2 backtest summary is available in data/processed.")
    else:
        metric_cols = st.columns(4)
        if "Brier" in summary.columns:
            metric_cols[0].metric("Mean Brier", dec(pd.to_numeric(summary["Brier"], errors="coerce").mean(), 4))
        if "LogLoss" in summary.columns:
            metric_cols[1].metric("Mean log loss", dec(pd.to_numeric(summary["LogLoss"], errors="coerce").mean(), 4))
        if "Matches" in summary.columns:
            metric_cols[2].metric("Backtest matches", int(pd.to_numeric(summary["Matches"], errors="coerce").sum()))
        if "Season" in summary.columns:
            metric_cols[3].metric("Seasons tested", summary["Season"].nunique())

        st.dataframe(summary, use_container_width=True, hide_index=True)

    if not comparison.empty:
        st.markdown("### V1.2 vs baseline")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "This dashboard is a research and decision-support interface. Historical backtests and model probabilities do not guarantee future profitability."
)
