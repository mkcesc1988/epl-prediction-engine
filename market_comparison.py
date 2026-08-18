from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from daily_predictions import build_daily_predictions
from pipeline import load_config, normalize_team

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MANUAL_MYBOOKIE_PATH = Path("data/manual_mybookie_odds.csv")


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


def _is_mybookie(row: pd.Series) -> bool:
    title = str(row.get("Bookmaker", "")).lower()
    key = str(row.get("BookmakerKey", "")).lower()
    return "mybookie" in title or "mybookie" in key


def _load_manual_mybookie() -> pd.DataFrame:
    if not MANUAL_MYBOOKIE_PATH.exists() or MANUAL_MYBOOKIE_PATH.stat().st_size == 0:
        return pd.DataFrame()
    manual = pd.read_csv(MANUAL_MYBOOKIE_PATH)
    if manual.empty:
        return manual
    manual["HomeTeam"] = manual["HomeTeam"].map(_norm)
    manual["AwayTeam"] = manual["AwayTeam"].map(_norm)
    manual["Date"] = manual["Date"].astype(str)
    manual["Bookmaker"] = manual.get("Bookmaker", "MyBookie.ag")
    manual["BookmakerKey"] = manual.get("BookmakerKey", "mybookie_manual")
    manual["BookLastUpdate"] = manual.get("ManualSource", "manual_screenshot")
    manual["KickoffUTC"] = pd.NA
    manual["EventId"] = "manual-" + manual.index.astype(str)
    return manual[[
        "EventId", "Date", "KickoffUTC", "HomeTeam", "AwayTeam", "Bookmaker",
        "BookmakerKey", "BookLastUpdate", "Market", "Outcome", "Point", "DecimalOdds"
    ]]


def fetch_market_odds(cfg: dict) -> pd.DataFrame:
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

    live = pd.DataFrame(rows)
    manual = _load_manual_mybookie()
    if manual.empty:
        return live

    if live.empty:
        return manual

    # Keep the broader live market, but replace matching MyBookie quotes with the
    # user's current screenshot prices. This lets rankings use the actual book the
    # user can access while preserving other books for market benchmarking.
    live = live.copy()
    live["Date"] = live["Date"].astype(str)
    live["HomeTeam"] = live["HomeTeam"].map(_norm)
    live["AwayTeam"] = live["AwayTeam"].map(_norm)

    manual_keys = set()
    for _, r in manual.iterrows():
        point = pd.to_numeric(r.get("Point"), errors="coerce")
        point_key = None if pd.isna(point) else round(float(point), 4)
        manual_keys.add((r["Date"], r["HomeTeam"], r["AwayTeam"], str(r["Market"]), str(r["Outcome"]), point_key))

    keep_mask = []
    for _, r in live.iterrows():
        if not _is_mybookie(r):
            keep_mask.append(True)
            continue
        point = pd.to_numeric(r.get("Point"), errors="coerce")
        point_key = None if pd.isna(point) else round(float(point), 4)
        key_tuple = (r["Date"], r["HomeTeam"], r["AwayTeam"], str(r["Market"]), str(r["Outcome"]), point_key)
        keep_mask.append(key_tuple not in manual_keys)

    return pd.concat([live.loc[keep_mask], manual], ignore_index=True, sort=False)


def compare_totals_25(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or odds.empty:
        return pd.DataFrame()

    totals = odds[(odds["Market"] == "totals") & (pd.to_numeric(odds["Point"], errors="coerce") == 2.5)].copy()
    if totals.empty:
        return pd.DataFrame()

    totals["Side"] = totals["Outcome"].astype(str).str.lower().map({"over": "Over", "under": "Under"})
    totals = totals.dropna(subset=["Side", "DecimalOdds"])
    totals["IsMyBookie"] = totals.apply(_is_mybookie, axis=1)

    group_cols = ["Date", "HomeTeam", "AwayTeam", "Side"]
    best_idx = totals.groupby(group_cols)["DecimalOdds"].idxmax()
    best = totals.loc[best_idx].copy().reset_index(drop=True)
    best = best.rename(columns={
        "Bookmaker": "BestBookmaker",
        "BookmakerKey": "BestBookmakerKey",
        "DecimalOdds": "BestMarketOdds",
        "BookLastUpdate": "BestBookLastUpdate",
    })

    mybookie = totals[totals["IsMyBookie"]].copy()
    if not mybookie.empty:
        my_idx = mybookie.groupby(group_cols)["DecimalOdds"].idxmax()
        mybookie = mybookie.loc[my_idx].copy().reset_index(drop=True)
        mybookie = mybookie[group_cols + ["Bookmaker", "BookmakerKey", "BookLastUpdate", "DecimalOdds"]]
        mybookie = mybookie.rename(columns={
            "Bookmaker": "MyBookieBookmaker",
            "BookmakerKey": "MyBookieKey",
            "BookLastUpdate": "MyBookieLastUpdate",
            "DecimalOdds": "MyBookieOdds",
        })
        quotes = best.merge(mybookie, on=group_cols, how="left", validate="one_to_one")
    else:
        quotes = best.copy()
        quotes["MyBookieBookmaker"] = pd.NA
        quotes["MyBookieKey"] = pd.NA
        quotes["MyBookieLastUpdate"] = pd.NA
        quotes["MyBookieOdds"] = pd.NA

    pred = predictions.copy()
    pred["Date"] = pred["Date"].astype(str)
    merged = quotes.merge(pred, on=["Date", "HomeTeam", "AwayTeam"], how="inner", validate="many_to_one", suffixes=("_Market", "_Model"))

    rows: list[dict] = []
    for _, r in merged.iterrows():
        side = r["Side"]
        model_p = float(r["CalP_Over2_5"] if side == "Over" else r["CalP_Under2_5"])
        fair_odds = 1.0 / model_p
        best_odds = float(r["BestMarketOdds"])
        best_implied = 1.0 / best_odds
        best_ev = model_p * best_odds - 1.0
        my_odds_raw = r.get("MyBookieOdds")
        my_odds = float(my_odds_raw) if pd.notna(my_odds_raw) else None
        my_implied = (1.0 / my_odds) if my_odds else None
        my_ev = (model_p * my_odds - 1.0) if my_odds else None
        rows.append({
            "Date": r["Date"], "KickoffUTC": r.get("KickoffUTC_Model", r.get("KickoffUTC_Market")),
            "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"], "Market": "Total Goals 2.5", "Side": side,
            "ModelVersion": r.get("ModelVersion", "V1.2"), "ModelProbability": model_p, "ModelFairOdds": fair_odds,
            "Bookmaker": r["BestBookmaker"], "MarketOdds": best_odds, "MarketImpliedProbability": best_implied,
            "ProbabilityDifference": model_p - best_implied, "ExpectedReturnPerUnit": best_ev,
            "BestBookmaker": r["BestBookmaker"], "BestMarketOdds": best_odds, "BestMarketImpliedProbability": best_implied,
            "BestMarketExpectedReturn": best_ev, "MyBookieAvailable": my_odds is not None,
            "MyBookieBookmaker": r.get("MyBookieBookmaker"), "MyBookieOdds": my_odds,
            "MyBookieImpliedProbability": my_implied, "MyBookieProbabilityDifference": (model_p - my_implied) if my_implied else None,
            "MyBookieExpectedReturn": my_ev, "MyBookiePriceGapVsBest": (my_odds - best_odds) if my_odds else None,
            "Lambda_Home_xG": r.get("Lambda_Home_xG"), "Lambda_Away_xG": r.get("Lambda_Away_xG"),
            "Lambda_Total_xG": r.get("Lambda_Total_xG"), "MostLikelyScore": r.get("MostLikelyScore"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_col = "MyBookieExpectedReturn" if out["MyBookieExpectedReturn"].notna().any() else "BestMarketExpectedReturn"
    return out.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


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
        my_count = int(comparison["MyBookieAvailable"].fillna(False).sum()) if "MyBookieAvailable" in comparison.columns else 0
        print(f"MyBookie comparisons:    {my_count}")
        pd.set_option("display.max_columns", None)
        print(comparison[["Date", "HomeTeam", "AwayTeam", "Side", "ModelProbability", "ModelFairOdds", "MyBookieOdds", "MyBookieExpectedReturn", "BestBookmaker", "BestMarketOdds", "BestMarketExpectedReturn"]].to_string(index=False))


if __name__ == "__main__":
    main()
