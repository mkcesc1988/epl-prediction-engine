from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd


def _weighted_mean(records: list[dict], key: str, now: pd.Timestamp, half_life_days: float) -> float | None:
    if not records:
        return None
    values = []
    weights = []
    for rec in records:
        age_days = max((now - rec["Date"]).days, 0)
        weight = 0.5 ** (age_days / half_life_days)
        values.append(float(rec[key]))
        weights.append(weight)
    if sum(weights) <= 0:
        return None
    return float(np.average(values, weights=weights))


def _poisson_over_25(total_lambda: float) -> float:
    p0 = math.exp(-total_lambda)
    p1 = p0 * total_lambda
    p2 = p1 * total_lambda / 2
    return 1 - (p0 + p1 + p2)


def build_predictions_v07(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Experimental V0.7 candidate.

    Improvements over V0.6:
    - exponential recency weighting by calendar days
    - shrinkage of venue-specific form toward overall team form
    - leakage-safe rolling league baselines

    The candidate is intentionally kept separate from V0.6 until the
    walk-forward comparison proves it is better out of sample.
    """
    model_cfg = cfg.get("model_v07", {})
    half_life = float(model_cfg.get("half_life_days", 180))
    venue_window = int(model_cfg.get("venue_window", 30))
    overall_window = int(model_cfg.get("overall_window", 60))
    league_window = int(model_cfg.get("league_window", 760))
    min_matches = int(model_cfg.get("min_matches", 5))
    shrinkage_matches = float(model_cfg.get("shrinkage_matches", 10))
    floor = float(cfg["model"].get("lambda_floor", 0.15))
    cap = float(cfg["model"].get("lambda_cap", 4.5))

    df = master.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    home_hist: dict[str, list[dict]] = defaultdict(list)
    away_hist: dict[str, list[dict]] = defaultdict(list)
    overall_hist: dict[str, list[dict]] = defaultdict(list)
    league_home_hist: list[dict] = []
    league_away_hist: list[dict] = []
    output = []

    for _, r in df.iterrows():
        now = r["Date"]
        home = r["HomeTeam"]
        away = r["AwayTeam"]

        hh = home_hist[home][-venue_window:]
        aa = away_hist[away][-venue_window:]
        ho = overall_hist[home][-overall_window:]
        ao = overall_hist[away][-overall_window:]
        lh = league_home_hist[-league_window:]
        la = league_away_hist[-league_window:]

        row = dict(r)
        valid = (
            len(hh) >= min_matches
            and len(aa) >= min_matches
            and len(ho) >= min_matches
            and len(ao) >= min_matches
            and len(lh) > 0
            and len(la) > 0
        )

        if valid:
            league_home = _weighted_mean(lh, "xG", now, half_life)
            league_away = _weighted_mean(la, "xG", now, half_life)

            h_home_for = _weighted_mean(hh, "xGF", now, half_life)
            h_home_against = _weighted_mean(hh, "xGA", now, half_life)
            a_away_for = _weighted_mean(aa, "xGF", now, half_life)
            a_away_against = _weighted_mean(aa, "xGA", now, half_life)

            h_overall_for = _weighted_mean(ho, "xGF", now, half_life)
            h_overall_against = _weighted_mean(ho, "xGA", now, half_life)
            a_overall_for = _weighted_mean(ao, "xGF", now, half_life)
            a_overall_against = _weighted_mean(ao, "xGA", now, half_life)

            wh = len(hh) / (len(hh) + shrinkage_matches)
            wa = len(aa) / (len(aa) + shrinkage_matches)

            home_xgf = wh * h_home_for + (1 - wh) * h_overall_for
            home_xga = wh * h_home_against + (1 - wh) * h_overall_against
            away_xgf = wa * a_away_for + (1 - wa) * a_overall_for
            away_xga = wa * a_away_against + (1 - wa) * a_overall_against

            lam_home = max(floor, min(home_xgf * away_xga / league_home, cap))
            lam_away = max(floor, min(away_xgf * home_xga / league_away, cap))
            total_lambda = lam_home + lam_away
            p_over = _poisson_over_25(total_lambda)

            row.update({
                "Lambda_Home_xG": lam_home,
                "Lambda_Away_xG": lam_away,
                "Lambda_Total_xG": total_lambda,
                "RawP_Over2_5_xG": p_over,
                "RawP_Under2_5_xG": 1 - p_over,
                "FairOdds_Over2_5_xG": 1 / p_over,
                "FairOdds_Under2_5_xG": 1 / (1 - p_over),
                "ModelVersion": "V0.7-candidate",
            })
        else:
            row.update({
                "Lambda_Home_xG": np.nan,
                "Lambda_Away_xG": np.nan,
                "Lambda_Total_xG": np.nan,
                "RawP_Over2_5_xG": np.nan,
                "RawP_Under2_5_xG": np.nan,
                "FairOdds_Over2_5_xG": np.nan,
                "FairOdds_Under2_5_xG": np.nan,
                "ModelVersion": "V0.7-candidate",
            })

        output.append(row)

        if pd.notna(r.get("Home_xG")) and pd.notna(r.get("Away_xG")):
            hxg = float(r["Home_xG"])
            axg = float(r["Away_xG"])
            home_record = {"Date": now, "xGF": hxg, "xGA": axg}
            away_record = {"Date": now, "xGF": axg, "xGA": hxg}
            home_hist[home].append(home_record)
            away_hist[away].append(away_record)
            overall_hist[home].append(home_record)
            overall_hist[away].append(away_record)
            league_home_hist.append({"Date": now, "xG": hxg})
            league_away_hist.append({"Date": now, "xG": axg})

    return pd.DataFrame(output)
