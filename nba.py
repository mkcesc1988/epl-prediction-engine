"""Isolated NBA research baseline. All scores include overtime.

CSV history: EventId,Date,HomeTeam,AwayTeam,HomeScore,AwayScore.
Date is the scheduled tipoff in UTC. Only completed games belong in history.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.special import ndtr

HISTORY_COLUMNS = ["EventId", "Date", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore"]
PREDICTION_COLUMNS = ["EventId", "Date", "HomeTeam", "AwayTeam", "PredictedAt", "TrainingGames", "HomeWinProbability", "AwayWinProbability", "ExpectedMargin", "FairHomeSpread", "ExpectedTotal", "ExpectedHomeScore", "ExpectedAwayScore", "MarginSD", "TotalSD", "MarginTotalCovariance"]
COMPARISON_COLUMNS = ["EventId", "Date", "HomeTeam", "AwayTeam", "CapturedAt", "Bookmaker", "BookLastUpdate", "Market", "Outcome", "Point", "DecimalOdds", "WinProbability", "PushProbability", "FairDecimalOdds", "ExpectedReturn"]


def read_json(url: str):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def scoreboard(day: str) -> list[dict]:
    """ESPN's public scoreboard; dates follow the feed's calendar convention."""
    payload = read_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?" + urlencode({"dates": pd.Timestamp(day).strftime("%Y%m%d"), "limit": 100}))
    rows = []
    for event in payload.get("events", []):
        competition = event["competitions"][0]
        # Regular season and playoffs only; preseason/exhibition is a different population.
        if int(event.get("season", {}).get("type", 0)) not in {2, 3}:
            continue
        sides = {c["homeAway"]: c for c in competition["competitors"]}
        if set(sides) != {"home", "away"}:
            continue
        if any(int(c["team"]["id"]) not in range(1, 31) for c in sides.values()):
            continue  # exclude All-Star/exhibition teams
        completed = bool(competition["status"]["type"].get("completed"))
        rows.append({"EventId": str(event["id"]), "Date": event["date"],
                     "HomeTeam": sides["home"]["team"]["displayName"],
                     "AwayTeam": sides["away"]["team"]["displayName"],
                     "HomeScore": float(sides["home"]["score"]) if completed else np.nan,
                     "AwayScore": float(sides["away"]["score"]) if completed else np.nan,
                     "Completed": completed})
    return rows


def validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(HISTORY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"History missing columns: {sorted(missing)}")
    h = frame[HISTORY_COLUMNS].copy()
    if h.isna().any().any():
        raise ValueError("History contains missing values; include completed games only")
    h["Date"] = pd.to_datetime(h["Date"], utc=True, errors="raise")
    h["EventId"] = h["EventId"].astype(str)
    for col in ["HomeTeam", "AwayTeam"]:
        h[col] = h[col].astype(str).str.strip()
        if h[col].eq("").any():
            raise ValueError("Empty team name")
    scores = h[["HomeScore", "AwayScore"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(scores.to_numpy()).all() or (scores <= 0).any().any() or (scores % 1 != 0).any().any():
        raise ValueError("Scores must be positive finite integers")
    h[["HomeScore", "AwayScore"]] = scores
    if h["EventId"].duplicated().any() or h.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate games in history")
    if h["HomeTeam"].eq(h["AwayTeam"]).any() or h["HomeScore"].eq(h["AwayScore"]).any():
        raise ValueError("Invalid teams or tied final NBA score")
    return h.sort_values(["Date", "EventId"]).reset_index(drop=True)


def features(history: pd.DataFrame, home: str, away: str, date) -> np.ndarray:
    # Exclude the entire UTC date: tipoff is not a trustworthy result-availability timestamp.
    date = pd.Timestamp(date)
    past = history[history.Date.dt.normalize() < date.normalize()]
    baseline = float(past[["HomeScore", "AwayScore"]].to_numpy().mean()) if len(past) else 110.0

    def team(name):
        games = past[(past.HomeTeam == name) | (past.AwayTeam == name)].tail(20)
        is_home = games.HomeTeam.eq(name)
        scored = np.where(is_home, games.HomeScore, games.AwayScore)
        allowed = np.where(is_home, games.AwayScore, games.HomeScore)
        # Five pseudo-games shrink small samples toward the contemporaneous league mean.
        offense = (scored.sum() - len(games) * baseline) / (len(games) + 5)
        defense = (allowed.sum() - len(games) * baseline) / (len(games) + 5)
        rest = min(max((date.normalize() - games.Date.iloc[-1].normalize()).days - 1, 0), 5) if len(games) else 2
        return offense, defense, rest

    ho, hd, hr = team(home)
    ao, ad, ar = team(away)
    return np.array([1., baseline - 110., ho, hd, ao, ad, hr - ar, hr + ar - 4.])


def design(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([features(history, r.HomeTeam, r.AwayTeam, r.Date) for r in history.itertuples()])
    y = np.column_stack([history.HomeScore - history.AwayScore, history.HomeScore + history.AwayScore])
    # Predict deviations from a fixed intercept; fit the intercept freely below.
    return x, y - np.array([0., 220.])


def fit(x: np.ndarray, y: np.ndarray, ridge: float = 20.) -> tuple[np.ndarray, np.ndarray]:
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    residuals = y - x @ beta
    # Provisional residual scale, measured independently by walk-forward evaluation.
    covariance = residuals.T @ residuals / max(len(y) - x.shape[1], 1)
    covariance += np.eye(2)  # avoid degenerate probabilities on unusually uniform samples
    return beta, covariance


def predict(history: pd.DataFrame, fixtures: pd.DataFrame, as_of, min_train: int = 200) -> pd.DataFrame:
    history = validate_history(history)
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    history = history[history.Date.dt.normalize() < as_of.tz_convert("UTC").normalize()]
    if len(history) < min_train:
        raise ValueError(f"Need at least {min_train} earlier completed games; found {len(history)}")
    x, y = design(history)
    beta, covariance = fit(x, y)
    rows = []
    for r in fixtures.itertuples():
        date = pd.to_datetime(r.Date, utc=True)
        if date <= as_of:
            continue
        margin, total = features(history, r.HomeTeam, r.AwayTeam, date) @ beta + [0., 220.]
        msd, tsd = np.sqrt(np.diag(covariance))
        ph = float(ndtr(margin / msd))
        rows.append(dict(zip(PREDICTION_COLUMNS, [str(r.EventId), date.isoformat(), r.HomeTeam, r.AwayTeam, as_of.isoformat(), len(history), ph, 1-ph, margin, -margin, total, (total+margin)/2, (total-margin)/2, msd, tsd, covariance[0, 1]])))
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def market_probability(prediction, market: str, outcome: str, point: float = 0.) -> tuple[float, float]:
    """Return unconditional win and push probabilities; spread point belongs to outcome."""
    if market == "h2h":
        if outcome not in {prediction.HomeTeam, prediction.AwayTeam}:
            raise ValueError("Unknown moneyline outcome")
        return (prediction.HomeWinProbability if outcome == prediction.HomeTeam else prediction.AwayWinProbability), 0.
    if market == "spreads":
        if outcome not in {prediction.HomeTeam, prediction.AwayTeam}:
            raise ValueError("Unknown spread outcome")
        mean = prediction.ExpectedMargin * (1 if outcome == prediction.HomeTeam else -1)
        sd, threshold, above = prediction.MarginSD, -point, True
    elif market == "totals" and outcome in {"Over", "Under"}:
        mean, sd, threshold, above = prediction.ExpectedTotal, prediction.TotalSD, point, outcome == "Over"
    else:
        raise ValueError("Unsupported market/outcome")
    integer = float(threshold).is_integer()
    # Discretized normal scores handle whole-point pushes and half-point lines.
    low = threshold - .5 if integer else np.floor(threshold) + .5
    high = threshold + .5 if integer else low
    lower, upper = float(ndtr((low-mean)/sd)), float(ndtr((high-mean)/sd))
    return (1-upper if above else lower), max(upper-lower, 0.)


def compare_odds(predictions: pd.DataFrame, payload: list[dict], captured_at) -> pd.DataFrame:
    rows = []
    captured_at = pd.to_datetime(captured_at, utc=True)
    for event in payload:
        date = pd.to_datetime(event["commence_time"], utc=True)
        matched = predictions[(predictions.HomeTeam == event["home_team"]) & (predictions.AwayTeam == event["away_team"])]
        matched = matched[(pd.to_datetime(matched.Date, utc=True) - date).abs() <= pd.Timedelta(hours=3)]
        if date <= captured_at or len(matched) != 1:
            continue
        prediction = next(matched.itertuples())
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    point = float(outcome.get("point", 0.))
                    price = float(outcome["price"])
                    if not np.isfinite([point, price]).all() or price <= 1:
                        continue
                    win, push = market_probability(prediction, market["key"], outcome["name"], point)
                    fair = (1-push)/win if win > 0 else np.inf
                    rows.append(dict(zip(COMPARISON_COLUMNS, [prediction.EventId, date.isoformat(), prediction.HomeTeam, prediction.AwayTeam, captured_at.isoformat(), book["key"], market.get("last_update", book.get("last_update")), market["key"], outcome["name"], point, price, win, push, fair, win*price+push-1])))
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def backtest(history: pd.DataFrame, min_train: int = 200) -> tuple[pd.DataFrame, dict]:
    history = validate_history(history)
    x, y = design(history)
    rows = []
    # One fit per date: no same-day score can enter another game's training data.
    for date, group in history.groupby(history.Date.dt.normalize(), sort=True):
        mask = (history.Date.dt.normalize() < date).to_numpy()
        if mask.sum() < min_train:
            continue
        beta, covariance = fit(x[mask], y[mask])
        for i in group.index:
            margin, total = x[i] @ beta + [0., 220.]
            prob = float(ndtr(margin / np.sqrt(covariance[0, 0])))
            actual_margin, actual_total = y[i] + [0., 220.]
            rows.append({"EventId": history.at[i, "EventId"], "Date": history.at[i, "Date"], "HomeWinProbability": prob, "HomeWon": int(actual_margin > 0), "ExpectedMargin": margin, "ActualMargin": actual_margin, "ExpectedTotal": total, "ActualTotal": actual_total})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Not enough history for an out-of-sample backtest")
    p = result.HomeWinProbability.clip(1e-9, 1-1e-9)
    actual = result.HomeWon
    summary = {"Games": len(result), "BrierScore": float(((p-actual)**2).mean()), "LogLoss": float(-(actual*np.log(p)+(1-actual)*np.log(1-p)).mean()), "MarginMAE": float((result.ExpectedMargin-result.ActualMargin).abs().mean()), "TotalMAE": float((result.ExpectedTotal-result.ActualTotal).abs().mean())}
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/raw/nba/games.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/nba"))
    parser.add_argument("--fetch-start", help="Fetch completed ESPN games from YYYY-MM-DD")
    parser.add_argument("--fetch-end", help="Inclusive fetch end; defaults to yesterday UTC")
    parser.add_argument("--date", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), help="ESPN fixture calendar date")
    parser.add_argument("--fixtures", type=Path, help="Offline CSV: EventId,Date,HomeTeam,AwayTeam")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--odds", action="store_true", help="Fetch current odds (consumes API credits)")
    parser.add_argument("--min-train", type=int, default=200)
    args = parser.parse_args()
    if args.min_train < 20:
        parser.error("--min-train must be at least 20")
    if args.fetch_start:
        end = args.fetch_end or (pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        dates = pd.date_range(args.fetch_start, end)
        if not len(dates):
            parser.error("fetch end precedes fetch start")
        rows = []
        for day in dates:
            rows.extend(r for r in scoreboard(str(day.date())) if r["Completed"])
        new = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
        old = pd.read_csv(args.history, dtype={"EventId": str}) if args.history.exists() else new.iloc[:0]
        history = validate_history(pd.concat([old, new]).drop_duplicates("EventId", keep="last"))
        args.history.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.history, index=False)
    if not args.history.exists():
        parser.error("History not found. Use --fetch-start or supply --history (see NBA.md).")
    history = validate_history(pd.read_csv(args.history, dtype={"EventId": str}))
    now = pd.Timestamp.now(tz="UTC")
    history = history[history.Date.dt.normalize() < now.normalize()]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backtest:
        detail, summary = backtest(history, args.min_train)
        detail.to_csv(args.output / "walkforward_predictions.csv", index=False)
        (args.output / "backtest_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2))
    fixtures = pd.read_csv(args.fixtures, dtype={"EventId": str}) if args.fixtures else pd.DataFrame(scoreboard(args.date), columns=HISTORY_COLUMNS+["Completed"])
    predictions = predict(history, fixtures, now, args.min_train)
    predictions.to_csv(args.output / "daily_predictions_latest.csv", index=False)
    # Always replace comparisons, so absent credentials cannot expose stale edges.
    comparisons = pd.DataFrame(columns=COMPARISON_COLUMNS)
    comparisons.to_csv(args.output / "market_comparison_latest.csv", index=False)
    if args.odds:
        key = os.getenv("THE_ODDS_API_KEY", "").strip()
        if not key:
            parser.error("--odds requires THE_ODDS_API_KEY")
        try:
            payload = read_json("https://api.the-odds-api.com/v4/sports/basketball_nba/odds?" + urlencode({"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "decimal", "dateFormat": "iso"}))
        except Exception:
            # urllib exceptions may embed the credential-bearing URL.
            (args.output / "market_comparison_latest.csv").write_text(",".join(COMPARISON_COLUMNS)+"\n")
            raise RuntimeError("NBA odds request failed; verify provider availability and API credentials") from None
        comparisons = compare_odds(predictions, payload, pd.Timestamp.now(tz="UTC"))
    comparisons.to_csv(args.output / "market_comparison_latest.csv", index=False)
    print(f"Wrote {len(predictions)} NBA predictions and {len(comparisons)} market comparisons to {args.output}")


if __name__ == "__main__":
    main()
