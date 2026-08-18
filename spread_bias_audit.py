from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gameweek_rankings import _ev_with_push, _spread_prob
from model_v11 import _score_matrix
from pipeline import load_config, normalize_team, season_code, season_label

FD_BASE = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"


def _line_bucket(point: float) -> str:
    if point >= 1.5:
        return "+1.5 or more"
    if point >= 0.5:
        return "+0.5 to +1.0"
    if point > -0.5:
        return "near pick'em"
    if point > -1.5:
        return "-0.5 to -1.0"
    return "-1.5 or more"


def _supported_line(point: float) -> bool:
    doubled = round(point * 2)
    return abs(point * 2 - doubled) < 1e-9


def fetch_handicap_history(cfg: dict) -> pd.DataFrame:
    frames = []
    for year in range(int(cfg["start_season"]), int(cfg["end_season"]) + 1):
        url = FD_BASE.format(code=season_code(year), division=cfg["football_data_division"])
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        raw = pd.read_csv(BytesIO(resp.content))
        if "AHh" not in raw.columns:
            continue
        cols = [c for c in [
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AHh",
            "PCAHH", "PCAHA", "PAHH", "PAHA", "B365AHH", "B365AHA",
        ] if c in raw.columns]
        df = raw[cols].copy()
        df["Season"] = season_label(year)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
        df["HomeTeam"] = df["HomeTeam"].map(normalize_team)
        df["AwayTeam"] = df["AwayTeam"].map(normalize_team)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prices(row: pd.Series) -> tuple[float | None, float | None, str | None]:
    for hc, ac, label in [
        ("PCAHH", "PCAHA", "Pinnacle Close"),
        ("PAHH", "PAHA", "Pinnacle"),
        ("B365AHH", "B365AHA", "Bet365"),
    ]:
        try:
            h, a = float(row[hc]), float(row[ac])
            if h > 1 and a > 1:
                return h, a, label
        except Exception:
            pass
    return None, None, None


def build_spread_audit(predictions: pd.DataFrame, handicap: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty or handicap.empty:
        return pd.DataFrame(), pd.DataFrame()

    pred = predictions.copy()
    pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.normalize()
    keys = ["Season", "Date", "HomeTeam", "AwayTeam"]
    merged = pred.merge(handicap, on=keys, how="inner", suffixes=("", "_odds"))
    max_goal = int(cfg.get("model_v12", {}).get("max_goal", 10))
    rows = []

    for _, r in merged.iterrows():
        line = pd.to_numeric(r.get("AHh"), errors="coerce")
        lh = pd.to_numeric(r.get("Lambda_Home_xG"), errors="coerce")
        la = pd.to_numeric(r.get("Lambda_Away_xG"), errors="coerce")
        rho = pd.to_numeric(r.get("DixonColes_Rho"), errors="coerce")
        if pd.isna(line) or pd.isna(lh) or pd.isna(la) or pd.isna(rho) or not _supported_line(float(line)):
            continue
        h_odds, a_odds, source = _prices(r)
        if h_odds is None:
            continue

        raw = np.array([1 / h_odds, 1 / a_odds], dtype=float)
        market_p = raw / raw.sum()
        matrix = _score_matrix(float(lh), float(la), float(rho), max_goal)
        hg, ag = int(r["FTHG"]), int(r["FTAG"])

        for side, team, point, odds, mp in [
            ("Home", r["HomeTeam"], float(line), h_odds, float(market_p[0])),
            ("Away", r["AwayTeam"], -float(line), a_odds, float(market_p[1])),
        ]:
            pwin, ppush = _spread_prob(matrix, r["HomeTeam"], r["AwayTeam"], team, point)
            margin = (hg - ag) if side == "Home" else (ag - hg)
            adjusted = margin + point
            if adjusted > 1e-9:
                result, realized = "W", odds - 1.0
            elif abs(adjusted) <= 1e-9:
                result, realized = "PUSH", 0.0
            else:
                result, realized = "L", -1.0
            ev = _ev_with_push(float(pwin), float(ppush), float(odds))
            rows.append({
                "Season": r["Season"], "Date": r["Date"], "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
                "Side": side, "Team": team, "Point": point, "LineBucket": _line_bucket(point),
                "OddsSource": source, "DecimalOdds": odds, "ModelWinProbability": pwin,
                "ModelPushProbability": ppush, "MarketNoVigProbability": mp,
                "ModelMinusMarketProbability": pwin - mp, "ModelExpectedReturn": ev,
                "Result": result, "FlatBetReturn": realized, "PositiveModelEV": ev > 0,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summaries = []
    for bucket, b in detail.groupby("LineBucket"):
        for subset_name, subset in [("All quoted sides", b), ("Positive model EV", b[b["PositiveModelEV"]])]:
            if subset.empty:
                continue
            nonpush = subset[subset["Result"] != "PUSH"]
            actual_win = float((nonpush["Result"] == "W").mean()) if not nonpush.empty else np.nan
            summaries.append({
                "LineBucket": bucket, "Subset": subset_name, "Bets": len(subset),
                "AveragePoint": float(subset["Point"].mean()), "AverageDecimalOdds": float(subset["DecimalOdds"].mean()),
                "AverageModelWinProbability": float(subset["ModelWinProbability"].mean()),
                "AverageMarketNoVigProbability": float(subset["MarketNoVigProbability"].mean()),
                "ActualWinRateExcludingPush": actual_win, "PushRate": float((subset["Result"] == "PUSH").mean()),
                "AverageModelExpectedReturn": float(subset["ModelExpectedReturn"].mean()),
                "FlatBetROI": float(subset["FlatBetReturn"].mean()),
                "FlatBetProfitUnits": float(subset["FlatBetReturn"].sum()),
            })
    return detail, pd.DataFrame(summaries)


def main() -> None:
    cfg = load_config()
    pred_path = Path(cfg["paths"]["predictions_v12"])
    if not pred_path.exists():
        raise RuntimeError("Run V1.2 historical predictions first")
    predictions = pd.read_csv(pred_path)
    handicap = fetch_handicap_history(cfg)
    detail, summary = build_spread_audit(predictions, handicap, cfg)
    out = Path(cfg["paths"]["processed_dir"])
    detail.to_csv(out / "spread_bias_detail_v12.csv", index=False)
    summary.to_csv(out / "spread_bias_summary_v12.csv", index=False)
    print(summary.to_string(index=False) if not summary.empty else "No supported historical spread rows")


if __name__ == "__main__":
    main()
