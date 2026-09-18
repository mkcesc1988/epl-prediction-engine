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

    captured_raw = manual.get("CapturedAt", pd.Series(index=manual.index, dtype=object))
    captured = pd.to_datetime(captured_raw, utc=True, errors="coerce")
    manual["QuoteCapturedAt"] = captured
    manual["BookLastUpdate"] = captured
    manual["PriceSource"] = "manual_screenshot"
    manual["KickoffUTC"] = pd.NA
    manual["EventId"] = "manual-" + manual.index.astype(str)

    return manual[[
        "EventId", "Date", "KickoffUTC", "HomeTeam", "AwayTeam", "Bookmaker",
        "BookmakerKey", "BookLastUpdate", "QuoteCapturedAt", "PriceSource",
        "Market", "Outcome", "Point", "DecimalOdds"
    ]]


def _live_odds_allowed() -> bool:
    value = os.getenv("THE_ODDS_API_LIVE", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _quote_timestamp(df: pd.DataFrame) -> pd.Series:
    last_update = pd.to_datetime(
        df.get("BookLastUpdate", pd.Series(index=df.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    captured = pd.to_datetime(
        df.get("QuoteCapturedAt", pd.Series(index=df.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    return last_update.fillna(captured)


def _fetch_odds_payload(url: str, params: dict, label: str) -> list[dict]:
    try:
        response = requests.get(url, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json() or []
        if not isinstance(payload, list):
            raise ValueError("odds payload is not a list")
        return payload
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        remaining = None
        if getattr(exc, "response", None) is not None:
            remaining = exc.response.headers.get("x-requests-remaining")
        detail = f"HTTP {status}" if status is not None else exc.__class__.__name__
        if remaining is not None:
            detail += f", requests remaining={remaining}"
        print(f"WARNING: {label} unavailable ({detail})")
        return []
    except ValueError as exc:
        print(f"WARNING: {label} returned invalid JSON ({exc})")
        return []


def fetch_market_odds(cfg: dict) -> pd.DataFrame:
    manual = _load_manual_mybookie()
    key = os.getenv("THE_ODDS_API_KEY", "").strip()

    if not _live_odds_allowed():
        print("Odds API live pull disabled for this run; using manual MyBookie prices only.")
        return manual

    if not key:
        print("WARNING: THE_ODDS_API_KEY is not configured; using manual MyBookie prices only.")
        return manual

    mc = cfg.get("market_comparison", {})
    sport_key = str(mc.get("odds_api_sport", "soccer_epl"))
    regions = str(mc.get("regions", "us"))
    markets = str(mc.get("markets", "h2h,spreads,totals"))
    mybookie_key = str(mc.get("mybookie_bookmaker_key", "mybookieag")).strip() or "mybookieag"

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    common = {
        "apiKey": key,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    broad_payload = _fetch_odds_payload(
        url,
        {**common, "regions": regions},
        "Odds API regional market feed",
    )
    mybookie_payload = _fetch_odds_payload(
        url,
        {**common, "bookmakers": mybookie_key},
        f"Odds API explicit MyBookie feed ({mybookie_key})",
    )

    if not broad_payload and not mybookie_payload:
        print("WARNING: all live odds requests failed or returned no events; using manual MyBookie prices only.")
        return manual

    captured_at = pd.Timestamp.now(tz="UTC").isoformat()
    rows: list[dict] = []

    for source, payload in [
        ("odds_api_region", broad_payload),
        ("odds_api_mybookie_direct", mybookie_payload),
    ]:
        for event in payload:
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
                            "QuoteCapturedAt": captured_at,
                            "PriceSource": source,
                            "Market": market.get("key"),
                            "Outcome": outcome.get("name"),
                            "Point": outcome.get("point"),
                            "DecimalOdds": float(price),
                        })

    live = pd.DataFrame(rows)
    if live.empty:
        return manual

    live["Date"] = live["Date"].astype(str)
    live["HomeTeam"] = live["HomeTeam"].map(_norm)
    live["AwayTeam"] = live["AwayTeam"].map(_norm)
    live["_QuoteTimestamp"] = _quote_timestamp(live)
    live["_SourcePriority"] = live["PriceSource"].eq("odds_api_mybookie_direct").astype(int)

    dedupe = [
        "Date", "HomeTeam", "AwayTeam", "BookmakerKey",
        "Market", "Outcome", "Point",
    ]
    live = (
        live.sort_values(
            ["_QuoteTimestamp", "_SourcePriority"],
            ascending=[True, True],
            na_position="first",
        )
        .drop_duplicates(dedupe, keep="last")
        .drop(columns=["_QuoteTimestamp", "_SourcePriority"])
        .reset_index(drop=True)
    )

    direct_mybookie_rows = int(
        live["PriceSource"].eq("odds_api_mybookie_direct").sum()
    )
    print(f"Explicit MyBookie live rows fetched: {direct_mybookie_rows}")

    if manual.empty:
        return live

    # Keep both live and manual quotes. Downstream executable ranking chooses the
    # freshest MyBookie quote, so an old screenshot can never overwrite a newer
    # live API price. Manual screenshots remain a fallback and an audit trail.
    return pd.concat([live, manual], ignore_index=True, sort=False)


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
        mybookie["_QuoteTimestamp"] = _quote_timestamp(mybookie)
        mybookie = (
            mybookie.sort_values(
                ["_QuoteTimestamp", "DecimalOdds"],
                ascending=[False, False],
                na_position="last",
            )
            .drop_duplicates(group_cols, keep="first")
            .reset_index(drop=True)
        )
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
