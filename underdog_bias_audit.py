from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ODDS_BUCKETS = [
    (2.00, 2.50, "+100 to +149"),
    (2.50, 3.00, "+150 to +199"),
    (3.00, 4.00, "+200 to +299"),
    (4.00, np.inf, "+300 and longer"),
]


def _valid_1x2(row: pd.Series) -> tuple[float | None, float | None, float | None, str | None]:
    sources = [
        ("Pinnacle_Home", "Pinnacle_Draw", "Pinnacle_Away", "Pinnacle"),
        ("B365_Home", "B365_Draw", "B365_Away", "Bet365"),
    ]
    for hc, dc, ac, label in sources:
        try:
            h, d, a = float(row[hc]), float(row[dc]), float(row[ac])
            if h > 1 and d > 1 and a > 1:
                return h, d, a, label
        except Exception:
            pass
    return None, None, None, None


def _bucket(decimal_odds: float) -> str | None:
    for lo, hi, label in ODDS_BUCKETS:
        if decimal_odds >= lo and decimal_odds < hi:
            return label
    return None


def build_underdog_audit(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    for _, r in predictions.iterrows():
        h_odds, d_odds, a_odds, source = _valid_1x2(r)
        if h_odds is None:
            continue

        probs_raw = np.array([1 / h_odds, 1 / d_odds, 1 / a_odds], dtype=float)
        overround = probs_raw.sum()
        market_probs = probs_raw / overround

        candidates = [
            ("Home", str(r.get("HomeTeam", "")), h_odds, float(r.get("P_HomeWin_DC", np.nan)), int(float(r.get("FTHG", np.nan)) > float(r.get("FTAG", np.nan))), market_probs[0]),
            ("Away", str(r.get("AwayTeam", "")), a_odds, float(r.get("P_AwayWin_DC", np.nan)), int(float(r.get("FTAG", np.nan)) > float(r.get("FTHG", np.nan))), market_probs[2]),
        ]

        for side, team, odds, model_p, outcome, market_p in candidates:
            if pd.isna(model_p) or pd.isna(r.get("FTHG")) or pd.isna(r.get("FTAG")):
                continue
            bucket = _bucket(float(odds))
            if bucket is None:
                continue
            model_edge = float(model_p - market_p)
            model_ev = float(model_p * odds - 1.0)
            realized_return = float(odds - 1.0 if outcome == 1 else -1.0)
            rows.append({
                "Season": r.get("Season"),
                "Date": r.get("Date"),
                "HomeTeam": r.get("HomeTeam"),
                "AwayTeam": r.get("AwayTeam"),
                "Side": side,
                "Team": team,
                "OddsSource": source,
                "DecimalOdds": odds,
                "OddsBucket": bucket,
                "ModelProbability": model_p,
                "MarketNoVigProbability": market_p,
                "ModelMinusMarketProbability": model_edge,
                "ModelExpectedReturn": model_ev,
                "Won": outcome,
                "FlatBetReturn": realized_return,
                "PositiveModelEdge": model_edge > 0,
                "PositiveModelEV": model_ev > 0,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summaries: list[dict] = []
    for bucket in [x[2] for x in ODDS_BUCKETS]:
        b = detail[detail["OddsBucket"] == bucket]
        for subset_name, subset in [
            ("All quoted sides", b),
            ("Positive model edge", b[b["PositiveModelEdge"]]),
            ("Positive model EV", b[b["PositiveModelEV"]]),
        ]:
            if subset.empty:
                continue
            n = len(subset)
            actual = float(subset["Won"].mean())
            model = float(subset["ModelProbability"].mean())
            market = float(subset["MarketNoVigProbability"].mean())
            flat_roi = float(subset["FlatBetReturn"].mean())
            summaries.append({
                "OddsBucket": bucket,
                "Subset": subset_name,
                "Bets": n,
                "AverageDecimalOdds": float(subset["DecimalOdds"].mean()),
                "AverageModelProbability": model,
                "AverageMarketNoVigProbability": market,
                "ActualWinRate": actual,
                "ModelCalibrationGap": model - actual,
                "MarketCalibrationGap": market - actual,
                "AverageModelMinusMarketProbability": float(subset["ModelMinusMarketProbability"].mean()),
                "AverageModelExpectedReturn": float(subset["ModelExpectedReturn"].mean()),
                "FlatBetROI": flat_roi,
                "FlatBetProfitUnits": float(subset["FlatBetReturn"].sum()),
            })

    summary = pd.DataFrame(summaries)
    return detail, summary


def save_underdog_audit(predictions: pd.DataFrame, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail, summary = build_underdog_audit(predictions)
    detail.to_csv(out / "underdog_bias_detail_v12.csv", index=False)
    summary.to_csv(out / "underdog_bias_summary_v12.csv", index=False)
    return detail, summary
