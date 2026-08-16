from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

HISTORY_DIR = Path("data/history")
LEDGER_PATH = HISTORY_DIR / "auto_bet_ledger.csv"

st.set_page_config(page_title="Bet Performance | EPL Prediction Engine", page_icon="📈", layout="wide")


def load_ledger() -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LEDGER_PATH)
    except Exception as exc:
        st.error(f"Could not read automatic bet ledger: {exc}")
        return pd.DataFrame()

    numeric_cols = [
        "EntryOdds", "ClosingOdds", "ModelProbability", "EntryImpliedProbability",
        "ProbabilityEdge", "ExpectedReturnPerUnit", "BetQualityScore",
        "ProfitabilityScore", "OverallRankScore", "StakeUnits", "StakeAmount",
        "PriceCLV", "ImpliedProbabilityCLV", "ProfitUnits", "BrierScore",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["EntrySnapshotUTC", "KickoffUTC", "ClosingSnapshotUTC", "SettledUTC"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{float(value) * 100:.{digits}f}%"


def units(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{float(value):+.2f}u"


def summary_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    rows = []
    for label, group in df.groupby(column, dropna=False):
        stake = group["StakeUnits"].sum(min_count=1) if "StakeUnits" in group.columns else float("nan")
        profit = group["ProfitUnits"].sum(min_count=1) if "ProfitUnits" in group.columns else float("nan")
        wins = int((group.get("Result", pd.Series(index=group.index, dtype=str)) == "W").sum())
        losses = int((group.get("Result", pd.Series(index=group.index, dtype=str)) == "L").sum())
        pushes = int((group.get("Result", pd.Series(index=group.index, dtype=str)) == "P").sum())
        roi = profit / stake if pd.notna(stake) and stake > 0 and pd.notna(profit) else float("nan")
        clv = group["PriceCLV"].mean() if "PriceCLV" in group.columns else float("nan")
        brier = group["BrierScore"].mean() if "BrierScore" in group.columns else float("nan")
        rows.append({
            column: "Unknown" if pd.isna(label) else label,
            "Bets": len(group),
            "W": wins,
            "L": losses,
            "P": pushes,
            "StakeUnits": stake,
            "ProfitUnits": profit,
            "ROI": roi,
            "AvgPriceCLV": clv,
            "AvgBrier": brier,
        })
    return pd.DataFrame(rows).sort_values(["ProfitUnits", "Bets"], ascending=[False, False])


st.title("📈 V1.2 Bet Performance")
st.caption(
    "Live out-of-sample tracking from the automatic paper-bet ledger. "
    "Open bets are shown separately and do not affect settled ROI or Brier score."
)

ledger = load_ledger()

if ledger.empty:
    st.info("No automatic tracked bets are available yet. Run EPL market comparison to populate the ledger.")
    st.stop()

result_series = ledger.get("Result", pd.Series("OPEN", index=ledger.index)).fillna("OPEN").astype(str)
open_bets = ledger[result_series.eq("OPEN")].copy()
settled = ledger[result_series.isin(["W", "L", "P"])].copy()

st.markdown("### Portfolio status")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tracked bets", len(ledger))
c2.metric("Open", len(open_bets))
c3.metric("Settled", len(settled))
open_units = open_bets["StakeUnits"].sum() if "StakeUnits" in open_bets.columns else 0.0
c4.metric("Open exposure", f"{open_units:.2f}u")
if "ModelVersion" in ledger.columns and ledger["ModelVersion"].notna().any():
    c5.metric("Model", str(ledger["ModelVersion"].dropna().iloc[-1]))
else:
    c5.metric("Model", "V1.2")

if settled.empty:
    st.info(
        "No tracked EPL bets have settled yet. ROI, realized units, win rate, and Brier score will appear automatically after results are recorded."
    )
else:
    total_stake = settled["StakeUnits"].sum() if "StakeUnits" in settled.columns else float("nan")
    total_profit = settled["ProfitUnits"].sum() if "ProfitUnits" in settled.columns else float("nan")
    roi = total_profit / total_stake if pd.notna(total_stake) and total_stake > 0 else float("nan")
    wins = int((settled["Result"] == "W").sum())
    losses = int((settled["Result"] == "L").sum())
    decisions = wins + losses
    win_rate = wins / decisions if decisions else float("nan")
    avg_clv = settled["PriceCLV"].mean() if "PriceCLV" in settled.columns else float("nan")
    avg_brier = settled["BrierScore"].mean() if "BrierScore" in settled.columns else float("nan")

    st.markdown("### Settled performance")
    a, b, c, d, e = st.columns(5)
    a.metric("Net units", units(total_profit))
    b.metric("ROI", pct(roi))
    c.metric("Win rate", pct(win_rate))
    d.metric("Average CLV", pct(avg_clv))
    e.metric("Brier score", f"{avg_brier:.3f}" if pd.notna(avg_brier) else "–")

    curve = settled.copy()
    sort_col = "SettledUTC" if "SettledUTC" in curve.columns and curve["SettledUTC"].notna().any() else "KickoffUTC"
    if sort_col in curve.columns:
        curve = curve.sort_values(sort_col)
    curve["CumulativeUnits"] = curve["ProfitUnits"].fillna(0).cumsum()
    chart = curve[[sort_col, "CumulativeUnits"]].dropna() if sort_col in curve.columns else curve[["CumulativeUnits"]]
    if not chart.empty:
        st.markdown("### Cumulative units")
        if sort_col in chart.columns:
            st.line_chart(chart.set_index(sort_col)["CumulativeUnits"])
        else:
            st.line_chart(chart["CumulativeUnits"])

st.markdown("### Open tracked bets")
if open_bets.empty:
    st.caption("No open bets.")
else:
    preferred_open = [
        "Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "EntryOdds",
        "ClosingOdds", "ModelProbability", "ProbabilityEdge", "ExpectedReturnPerUnit",
        "BetQualityScore", "Grade", "StakeUnits", "PriceCLV", "ClosingQuality",
    ]
    cols = [c for c in preferred_open if c in open_bets.columns]
    st.dataframe(open_bets[cols] if cols else open_bets, use_container_width=True, hide_index=True)

if not settled.empty:
    st.markdown("### Performance breakdowns")
    dimensions = [c for c in ["MarketType", "ConfidenceBucket", "EdgeBucket", "Grade"] if c in settled.columns]
    if dimensions:
        dimension = st.selectbox("Break down by", dimensions)
        breakdown = summary_by(settled, dimension)
        if not breakdown.empty:
            format_df = breakdown.copy()
            for col in ["ROI", "AvgPriceCLV"]:
                if col in format_df.columns:
                    format_df[col] = format_df[col].map(lambda x: pct(x) if pd.notna(x) else "–")
            for col in ["StakeUnits", "ProfitUnits", "AvgBrier"]:
                if col in format_df.columns:
                    format_df[col] = pd.to_numeric(format_df[col], errors="coerce").round(3)
            st.dataframe(format_df, use_container_width=True, hide_index=True)

    st.markdown("### Settled bet ledger")
    preferred_settled = [
        "Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "EntryOdds", "ClosingOdds",
        "ModelProbability", "ProbabilityEdge", "StakeUnits", "Result", "FinalScore",
        "ProfitUnits", "PriceCLV", "BrierScore", "Grade", "ConfidenceBucket", "EdgeBucket",
    ]
    cols = [c for c in preferred_settled if c in settled.columns]
    st.dataframe(settled[cols] if cols else settled, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Interpretation: ROI measures realized return on staked units. Positive CLV means the entry price was better than the stored pre-kickoff closing proxy. "
    "Brier score measures probability calibration, lower is better. Small samples should not be treated as proof of predictive edge."
)
