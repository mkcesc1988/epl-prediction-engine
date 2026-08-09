from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd


def poisson_over_25(lmbda: float) -> float:
    p0 = math.exp(-lmbda)
    p1 = p0 * lmbda
    p2 = p1 * lmbda / 2
    return 1 - (p0 + p1 + p2)


def build_predictions(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Leakage-safe xG Poisson predictions using only prior matches."""
    window = int(cfg["model"]["venue_window"])
    min_matches = int(cfg["model"]["min_venue_matches"])
    floor = float(cfg["model"]["lambda_floor"])
    cap = float(cfg["model"]["lambda_cap"])

    df = master.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    home_hist = defaultdict(list)
    away_hist = defaultdict(list)
    league_home_xg: list[float] = []
    league_away_xg: list[float] = []
    output = []

    for _, r in df.iterrows():
        h = r["HomeTeam"]
        a = r["AwayTeam"]
        hh = home_hist[h][-window:]
        aa = away_hist[a][-window:]

        hxgf = np.mean([x["xGF"] for x in hh]) if hh else None
        hxga = np.mean([x["xGA"] for x in hh]) if hh else None
        axgf = np.mean([x["xGF"] for x in aa]) if aa else None
        axga = np.mean([x["xGA"] for x in aa]) if aa else None

        base_h = np.mean(league_home_xg[-380:]) if league_home_xg else None
        base_a = np.mean(league_away_xg[-380:]) if league_away_xg else None

        row = dict(r)
        valid = (
            pd.notna(r.get("Home_xG"))
            and pd.notna(r.get("Away_xG"))
            and base_h is not None
            and base_a is not None
            and len(hh) >= min_matches
            and len(aa) >= min_matches
            and None not in [hxgf, hxga, axgf, axga]
        )

        if valid:
            lam_h = max(floor, min(hxgf * axga / base_h, cap))
            lam_a = max(floor, min(axgf * hxga / base_a, cap))
            p_over = poisson_over_25(lam_h + lam_a)
            row.update({
                "Lambda_Home_xG": lam_h,
                "Lambda_Away_xG": lam_a,
                "Lambda_Total_xG": lam_h + lam_a,
                "RawP_Over2_5_xG": p_over,
                "RawP_Under2_5_xG": 1 - p_over,
                "FairOdds_Over2_5_xG": 1 / p_over,
                "FairOdds_Under2_5_xG": 1 / (1 - p_over),
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
            })

        output.append(row)

        # Update histories only after calculating the current prediction.
        if pd.notna(r.get("Home_xG")) and pd.notna(r.get("Away_xG")):
            hxg = float(r["Home_xG"])
            axg = float(r["Away_xG"])
            home_hist[h].append({"xGF": hxg, "xGA": axg})
            away_hist[a].append({"xGF": axg, "xGA": hxg})
            league_home_xg.append(hxg)
            league_away_xg.append(axg)

    return pd.DataFrame(output)
