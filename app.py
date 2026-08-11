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
st.caption("Production model V1.2 · xG strength ratings · Dixon-Coles · MyBookie market research")

predictions = load_csv("daily_predictions_latest.csv")
market = load_csv("market_comparison_latest.csv")
rankings = load_csv("gameweek_rankings_latest.csv")
portfolio = load_csv("paper_portfolio_latest.csv")
summary = load_csv("backtest_summary_v12.csv")
comparison = load_csv("model_comparison_v12.csv")
pred_history = load_history("prediction_history.csv")
market_history = load_history("market_comparison_history.csv")
clv = load_history("clv_report.csv")

pred_tab, rank_tab, portfolio_tab, market_tab, live_tab, clv_tab, perf_tab = st.tabs([
    "Today's Predictions",
    "Gameweek Rankings",
    "Paper Portfolio",
    "Market Comparison",
    "Live History",
    "Closing Line",
    "Model Performance",
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

with rank_tab:
    st.subheader("Gameweek MyBookie rankings")
    st.caption(
        "Model Win Probability is the model's estimated chance of the selection winning. "
        "Bet Quality Score is a 0–100 ranking heuristic combining probability, price edge, and validation depth. "
        "It is not a literal percentage chance of success."
    )

    if rankings.empty:
        st.info("No gameweek ranking file is available yet. Run EPL market comparison after the latest code update.")
    else:
        markets = ["All"] + sorted(rankings["MarketType"].dropna().astype(str).unique().tolist()) if "MarketType" in rankings.columns else ["All"]
        selected_market = st.selectbox("Market", markets, index=0)
        positive_only = st.toggle("Show positive expected return only", value=True)

        view = rankings.copy()
        if selected_market != "All":
            view = view[view["MarketType"].astype(str) == selected_market]
        if positive_only and "ExpectedReturnPerUnit" in view.columns:
            view = view[pd.to_numeric(view["ExpectedReturnPerUnit"], errors="coerce") > 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ranked selections", len(view))
        if not view.empty and "BetQualityScore" in view.columns:
            c2.metric("Top bet quality", f"{pd.to_numeric(view['BetQualityScore'], errors='coerce').max():.1f}/100")
        if not view.empty and "ExpectedProfitPer100" in view.columns:
            c3.metric("Top expected profit / $100", f"${pd.to_numeric(view['ExpectedProfitPer100'], errors='coerce').max():.2f}")
        if not view.empty and "OverallRankScore" in view.columns:
            c4.metric("Top overall score", f"{pd.to_numeric(view['OverallRankScore'], errors='coerce').max():.1f}/100")

        if view.empty:
            st.info("No selections match the current filters.")
        else:
            sort_options = [c for c in ["OverallRankScore", "BetQualityScore", "ExpectedReturnPerUnit", "ExpectedProfitPer100", "ModelWinProbability"] if c in view.columns]
            sort_choice = st.selectbox("Rank by", sort_options, index=0)
            view = view.sort_values(sort_choice, ascending=False).reset_index(drop=True)
            view["DisplayRank"] = range(1, len(view) + 1)

            preferred = [
                "DisplayRank", "Grade", "Date", "HomeTeam", "AwayTeam", "MarketType", "Selection",
                "MyBookieOdds", "ModelWinProbability", "ModelFairOdds", "MyBookieImpliedProbability",
                "ProbabilityEdge", "ExpectedReturnPerUnit", "ExpectedProfitPer100",
                "BetQualityScore", "ProfitabilityScore", "OverallRankScore", "ValidationStatus",
            ]
            cols = [c for c in preferred if c in view.columns]
            st.dataframe(view[cols] if cols else view, use_container_width=True, hide_index=True)

            st.markdown("### Top 5 this gameweek")
            for _, row in view.head(5).iterrows():
                ev = float(row.get("ExpectedReturnPerUnit", 0.0))
                with st.container(border=True):
                    st.markdown(
                        f"**#{int(row['DisplayRank'])} · {row.get('Grade', '')} · "
                        f"{row.get('MarketType', '')}: {row.get('Selection', '')}**"
                    )
                    st.write(f"{row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}")
                    a, b, c, d = st.columns(4)
                    a.metric("MyBookie", dec(row.get("MyBookieOdds")))
                    b.metric("Model win probability", pct(row.get("ModelWinProbability")))
                    c.metric("Bet quality", f"{float(row.get('BetQualityScore', 0)):.1f}/100")
                    d.metric("Expected profit / $100", f"${ev * 100:.2f}")
                    if pd.notna(row.get("ValidationStatus")):
                        st.caption(f"Validation: {row.get('ValidationStatus')}")

with portfolio_tab:
    st.subheader("Conservative paper portfolio")
    st.caption(
        "Paper-testing only. Sizing uses capped fractional Kelly with hard per-selection, per-match, "
        "and total gameweek exposure limits. It does not place wagers."
    )
    if portfolio.empty:
        st.info("No paper portfolio is available yet. The next EPL market comparison run will generate it automatically.")
    else:
        exposure = pd.to_numeric(portfolio.get("PaperStakeAmount"), errors="coerce").sum() if "PaperStakeAmount" in portfolio.columns else 0.0
        expected_profit = pd.to_numeric(portfolio.get("ExpectedPaperProfit"), errors="coerce").sum() if "ExpectedPaperProfit" in portfolio.columns else 0.0
        max_exposure = pd.to_numeric(portfolio.get("PortfolioExposurePct"), errors="coerce").max() if "PortfolioExposurePct" in portfolio.columns else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paper selections", len(portfolio))
        c2.metric("Paper exposure", f"{exposure:.2f} units")
        c3.metric("Expected paper profit", f"{expected_profit:.2f} units")
        c4.metric("Gameweek exposure", pct(max_exposure))

        preferred = [
            "PortfolioRank", "Grade", "Date", "HomeTeam", "AwayTeam", "MarketType", "Selection",
            "MyBookieOdds", "ModelWinProbability", "BetQualityScore", "ExpectedReturnPerUnit",
            "PaperStakeUnits", "PaperStakeAmount", "ExpectedPaperProfit", "SizingNote",
        ]
        cols = [c for c in preferred if c in portfolio.columns]
        st.dataframe(portfolio[cols] if cols else portfolio, use_container_width=True, hide_index=True)

        st.markdown("### Paper allocation")
        for _, row in portfolio.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**#{int(row.get('PortfolioRank', 0))} · {row.get('MarketType', '')}: {row.get('Selection', '')}**"
                )
                st.write(f"{row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}")
                a, b, c, d = st.columns(4)
                a.metric("MyBookie", dec(row.get("MyBookieOdds")))
                b.metric("Bet quality", f"{float(row.get('BetQualityScore', 0)):.1f}/100")
                c.metric("Paper units", f"{float(row.get('PaperStakeUnits', 0)):.2f}")
                d.metric("Expected paper profit", f"{float(row.get('ExpectedPaperProfit', 0)):.2f}")

with market_tab:
    if market.empty:
        st.info("No market comparison file is available yet.")
    else:
        st.subheader("Model vs MyBookie and broader market")
        st.caption("Research view only. No staking or automated betting is performed.")
        edge_col = "MyBookieExpectedReturn" if "MyBookieExpectedReturn" in market.columns else "ExpectedReturnPerUnit"
        edge = pd.to_numeric(market.get(edge_col), errors="coerce") if edge_col in market.columns else pd.Series(dtype=float)
        c1, c2, c3 = st.columns(3)
        c1.metric("Market rows", len(market))
        c2.metric("Positive MyBookie discrepancies", int((edge > 0).sum()) if not edge.empty else 0)
        if not edge.empty and edge.notna().any():
            c3.metric("Largest expected return", pct(edge.max()))
        preferred = [
            "Date", "HomeTeam", "AwayTeam", "Market", "Side", "ModelProbability", "ModelFairOdds",
            "MyBookieOdds", "MyBookieImpliedProbability", "MyBookieProbabilityDifference", "MyBookieExpectedReturn",
            "BestBookmaker", "BestMarketOdds", "BestMarketExpectedReturn", "MyBookiePriceGapVsBest",
        ]
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
st.caption(
    "Research and decision-support only. Bet Quality Score is a heuristic ranking metric, not a probability. "
    "Paper sizing and model-implied expected profit do not guarantee future profit."
)
