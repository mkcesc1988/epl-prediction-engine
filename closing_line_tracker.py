from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_PATH = Path("data/history/market_comparison_history.csv")
OUTPUT_PATH = Path("data/history/clv_report.csv")


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def build_clv_report(history: pd.DataFrame) -> pd.DataFrame:
    """Build a fixture-level closing-line report from accumulated market snapshots.

    The earliest stored quote is treated as the first observed market price.
    The latest snapshot strictly before kickoff is treated as the closing proxy.
    `ClosingQuality` indicates how close that proxy was to kickoff.
    """
    if history.empty:
        return pd.DataFrame()

    required = {
        "SnapshotUTC", "KickoffUTC", "HomeTeam", "AwayTeam", "Market", "Side",
        "MarketOdds", "MarketImpliedProbability", "ModelProbability", "ModelFairOdds",
        "ProbabilityDifference", "ExpectedReturnPerUnit",
    }
    missing = required.difference(history.columns)
    if missing:
        raise RuntimeError(f"CLV history missing required columns: {sorted(missing)}")

    df = history.copy()
    df["SnapshotUTC"] = _to_utc(df["SnapshotUTC"])
    df["KickoffUTC"] = _to_utc(df["KickoffUTC"])
    df["MarketOdds"] = pd.to_numeric(df["MarketOdds"], errors="coerce")
    df["MarketImpliedProbability"] = pd.to_numeric(df["MarketImpliedProbability"], errors="coerce")
    df["ModelProbability"] = pd.to_numeric(df["ModelProbability"], errors="coerce")
    df["ModelFairOdds"] = pd.to_numeric(df["ModelFairOdds"], errors="coerce")
    df["ProbabilityDifference"] = pd.to_numeric(df["ProbabilityDifference"], errors="coerce")
    df["ExpectedReturnPerUnit"] = pd.to_numeric(df["ExpectedReturnPerUnit"], errors="coerce")
    df = df.dropna(subset=["SnapshotUTC", "KickoffUTC", "MarketOdds", "HomeTeam", "AwayTeam", "Market", "Side"])

    # Never use an in-play or post-kickoff quote as the closing line.
    df = df[df["SnapshotUTC"] < df["KickoffUTC"]].copy()
    if df.empty:
        return pd.DataFrame()

    keys = ["Date", "HomeTeam", "AwayTeam", "Market", "Side"]
    rows: list[dict] = []

    for key, group in df.groupby(keys, dropna=False, sort=True):
        group = group.sort_values("SnapshotUTC").reset_index(drop=True)
        first = group.iloc[0]
        close = group.iloc[-1]

        hours_to_close = (close["KickoffUTC"] - close["SnapshotUTC"]).total_seconds() / 3600.0
        if hours_to_close <= 1.5:
            quality = "near_close"
        elif hours_to_close <= 4.0:
            quality = "good_proxy"
        elif hours_to_close <= 12.0:
            quality = "proxy"
        else:
            quality = "early_proxy"

        first_odds = float(first["MarketOdds"])
        close_odds = float(close["MarketOdds"])
        first_imp = 1.0 / first_odds
        close_imp = 1.0 / close_odds

        # Positive PriceCLV means the first observed price was better than the
        # market's latest available pre-kickoff price.
        price_clv = first_odds / close_odds - 1.0
        implied_prob_clv = close_imp - first_imp

        model_p = float(first["ModelProbability"]) if pd.notna(first["ModelProbability"]) else np.nan
        model_edge_first = model_p - first_imp if np.isfinite(model_p) else np.nan
        model_edge_close = model_p - close_imp if np.isfinite(model_p) else np.nan
        ev_first = model_p * first_odds - 1.0 if np.isfinite(model_p) else np.nan
        ev_close = model_p * close_odds - 1.0 if np.isfinite(model_p) else np.nan

        rows.append({
            "Date": key[0],
            "KickoffUTC": close["KickoffUTC"].isoformat(),
            "HomeTeam": key[1],
            "AwayTeam": key[2],
            "Market": key[3],
            "Side": key[4],
            "ModelVersion": first.get("ModelVersion", "V1.2"),
            "ModelProbability": model_p,
            "ModelFairOdds": first.get("ModelFairOdds"),
            "FirstSnapshotUTC": first["SnapshotUTC"].isoformat(),
            "FirstBookmaker": first.get("Bookmaker"),
            "FirstObservedOdds": first_odds,
            "CloseSnapshotUTC": close["SnapshotUTC"].isoformat(),
            "CloseBookmaker": close.get("Bookmaker"),
            "ClosingProxyOdds": close_odds,
            "HoursBeforeKickoff": hours_to_close,
            "ClosingQuality": quality,
            "PriceCLV": price_clv,
            "ImpliedProbabilityCLV": implied_prob_clv,
            "ModelEdgeAtFirst": model_edge_first,
            "ModelEdgeAtClose": model_edge_close,
            "ExpectedReturnAtFirst": ev_first,
            "ExpectedReturnAtClose": ev_close,
            "SnapshotsObserved": int(len(group)),
        })

    return pd.DataFrame(rows).sort_values(
        ["KickoffUTC", "HomeTeam", "AwayTeam", "Market", "Side"]
    ).reset_index(drop=True)


def main() -> None:
    if not HISTORY_PATH.exists():
        print("No market comparison history exists yet; CLV report not created.")
        return

    history = pd.read_csv(HISTORY_PATH)
    report = build_clv_report(history)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(f"CLV rows: {len(report)}")
    if not report.empty:
        print(report[[
            "Date", "HomeTeam", "AwayTeam", "Side", "FirstObservedOdds",
            "ClosingProxyOdds", "PriceCLV", "HoursBeforeKickoff",
            "ClosingQuality", "SnapshotsObserved",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
