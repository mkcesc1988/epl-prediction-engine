from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from daily_predictions import build_daily_predictions
from pipeline import load_config, normalize_team

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _norm(name: object) -> str:
    value = str(name).strip()
    mapping = {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton and Hove Albion": "Brighton",
        "Nottingham Forest": "Nott'm Forest",
        "Leeds United": "Leeds",
    }
    return normalize_team(mapping.get(value, value))


def fetch_market_odds(cfg: dict) -> pd.DataFrame:
    """Fetch current EPL market prices for paper-testing and model evaluation.

    Requires THE_ODDS_API_KEY in the environment. This module records market
    prices and model-market discrepancies only. It does not place wagers or
    calculate stakes.
    """
    key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "THE_ODDS_API_KEY is not configured. Add it as a GitHub Actions secret to enable live market comparison."
        )

    mc = cfg.get("market_comparison", {})
    sport_key = str(mc.get("odds_api_sport", "soccer_epl"))
    regions = str(mc.get("regions", "us"))
    markets = str(mc.get("markets", "h2h,totals"))

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    response = requests.get(
        url,
        params={
            "apiKey": key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=45,
    )
    response.raise_for_status()

    rows: list[dict] = []
    for event in response.json() or []:
        kickoff = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        home = _norm(event.get("home_team"))
        away = _norm(event.get("away_team"))

        for book in event.get("bookmakers", []) or []:
            for market in book.get("markets", []) or []:
                for outcome in market.get("outcomes", []) or []:
                    price = outcome.get("price")
                    if price is None:
                        continue
                    rows.append({
                        "EventId": event.get("id"),
                        "Date": kickoff.tz_convert(None).strftime("%Y-%m-%d"),
                        "KickoffUTC": kickoff.isoformat(),
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "Bookmaker": book.get("title") or book.get("key"),
                        "BookmakerKey": book.get("key"),
                        "BookLastUpdate": book.get("last_update"),
                        "Market": market.get("key"),
                        "Outcome": outcome.get("name"),
                        "Point": outcome.get("point"),
                        "DecimalOdds": float(price),
                    })

    return pd.DataFrame(rows)


def compare_totals_25(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Compare V1.2 calibrated O/U 2.5 probabilities with observed market prices."""
    if predictions.empty or odds.empty:
        return pd.DataFrame()

    totals = odds[(odds["Market"] == "totals") & (pd.to_numeric(odds["Point"], errors="coerce") == 2.5)].copy()
    if totals.empty:
        return pd.DataFrame()

    totals["Side"] = totals["Outcome"].astype(str).str.lower().map({"over": "Over", "under": "Under"})
    totals = totals.dropna(subset=["Side", "DecimalOdds"])

    idx = totals.groupby(["Date", "HomeTeam", "AwayTeam", "Side"])["DecimalOdds"].idxmax()
    best = totals.loc[idx].reset_index(drop=True)

    pred = predictions.copy()
    pred["Date"] = pred["Date"].astype(str)
    merged = best.merge(
        pred,
        on=["Date", "HomeTeam", "AwayTeam"],
        how="inner",
        validate="many_to_one",
        suffixes=("_Market", "_Model"),
    )

    rows: list[dict] = []
    for _, r in merged.iterrows():
        side = r["Side"]
        model_p = float(r["CalP_Over2_5"] if side == "Over" else r["CalP_Under2_5"])
        market_odds = float(r["DecimalOdds"])
        implied = 1.0 / market_odds

        rows.append({
            "Date": r["Date"],
            "KickoffUTC": r.get("KickoffUTC_Model", r.get("KickoffUTC_Market")),
            "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            "Market": "Total Goals 2.5",
            "Side": side,
            "ModelVersion": r.get("ModelVersion", "V1.2"),
            "ModelProbability": model_p,
            "ModelFairOdds": 1.0 / model_p,
            "Bookmaker": r["Bookmaker"],
            "MarketOdds": market_odds,
            "MarketImpliedProbability": implied,
            "ProbabilityDifference": model_p - implied,
            "ExpectedReturnPerUnit": model_p * market_odds - 1.0,
            "Lambda_Home_xG": r.get("Lambda_Home_xG"),
            "Lambda_Away_xG": r.get("Lambda_Away_xG"),
            "Lambda_Total_xG": r.get("Lambda_Total_xG"),
            "MostLikelyScore": r.get("MostLikelyScore"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("ProbabilityDifference", ascending=False).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = build_daily_predictions(cfg)
    predictions.to_csv(out_dir / "daily_predictions_latest.csv", index=False)

    odds = fetch_market_odds(cfg)
    odds.to_csv(out_dir / "market_odds_latest.csv", index=False)

    comparison = compare_totals_25(predictions, odds)
    comparison.to_csv(out_dir / "market_comparison_latest.csv", index=False)

    print(f"Daily fixtures predicted: {len(predictions)}")
    print(f"Market outcome rows:      {len(odds)}")
    print(f"O/U 2.5 comparisons:     {len(comparison)}")
    if not comparison.empty:
        pd.set_option("display.max_columns", None)
        print(comparison[[
            "Date", "HomeTeam", "AwayTeam", "Side", "ModelProbability",
            "ModelFairOdds", "Bookmaker", "MarketOdds",
            "ProbabilityDifference", "ExpectedReturnPerUnit"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
