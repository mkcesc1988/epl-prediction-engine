"""Multi-season walk-forward test of shrinking the V1.2 home advantage.

The strength fit does not depend on the shrink (it is applied after fitting),
so each match date is fitted once and every shrink value in the grid is scored
from that same fit. Only seasons in the evaluation window are predicted; all
earlier matches are still used as training history.

Outputs (data/processed/):
  home_advantage_sweep_summary.csv  one row per season x shrink, plus ALL rows
  home_advantage_sweep_detail.csv   one row per match x shrink

Run: python home_advantage_sweep.py
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from model_v11 import _derived_markets, _score_matrix
from model_v12 import _estimate_rho_from_history, _fit_strengths, effective_home_advantage
from pipeline import build_master, load_config


def _outcome(hg: float, ag: float) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def rps(p: np.ndarray, outcome: int) -> float:
    y = np.zeros(3)
    y[outcome] = 1.0
    cp, cy = np.cumsum(p), np.cumsum(y)
    return float(((cp[:2] - cy[:2]) ** 2).sum() / 2.0)


def _devig(home: float, draw: float, away: float) -> np.ndarray | None:
    odds = np.array([home, draw, away], dtype=float)
    if not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
        return None
    implied = 1.0 / odds
    return implied / implied.sum()


def _predict_probs(fit, cfg: dict, home: str, away: str, rho: float) -> tuple[np.ndarray, float, float]:
    mc = cfg.get("model_v12", {})
    floor = float(mc.get("lambda_floor", 0.15))
    cap = float(mc.get("lambda_cap", 4.5))
    max_goal = int(mc.get("max_goal", 10))
    ah, dh = fit.attack.get(home, 0.0), fit.defense.get(home, 0.0)
    aa, da = fit.attack.get(away, 0.0), fit.defense.get(away, 0.0)
    home_adv = effective_home_advantage(fit, cfg)
    lam_h = max(floor, min(math.exp(np.clip(fit.intercept + home_adv + ah - da, -4.0, 3.0)), cap))
    lam_a = max(floor, min(math.exp(np.clip(fit.intercept + aa - dh, -4.0, 3.0)), cap))
    m = _derived_markets(_score_matrix(lam_h, lam_a, rho, max_goal))
    return np.array([m["P_HomeWin_DC"], m["P_Draw_DC"], m["P_AwayWin_DC"]]), lam_h, lam_a


def run_sweep(master: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = cfg.get("home_advantage_sweep", {})
    grid = [float(x) for x in sc.get("shrink_grid", [0.0, 0.15, 0.3, 0.45, 0.6])]
    if 0.0 not in grid:
        grid = [0.0] + grid
    n_seasons = int(sc.get("eval_last_n_seasons", 3))

    df = master.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    seasons = list(dict.fromkeys(df["Season"].astype(str)))
    eval_seasons = set(seasons[-n_seasons:])

    cfgs = {}
    for s in grid:
        c = copy.deepcopy(cfg)
        c.setdefault("model_v12", {})["home_advantage_shrink"] = s
        cfgs[s] = c

    rows: list[dict] = []
    dates = list(df.groupby("Date", sort=True))
    for i, (match_date, day_games) in enumerate(dates):
        history = df[df["Date"] < match_date]
        targets = day_games[
            day_games["Season"].astype(str).isin(eval_seasons)
            & day_games["FTHG"].notna()
            & day_games["FTAG"].notna()
        ]
        if targets.empty:
            continue
        fit = _fit_strengths(history, pd.Timestamp(match_date), cfg)
        if fit is None:
            continue
        rhos = {s: _estimate_rho_from_history(history, fit, cfgs[s]) for s in grid}
        for _, r in targets.iterrows():
            o = _outcome(r["FTHG"], r["FTAG"])
            mkt = _devig(r.get("Pinnacle_Home", np.nan), r.get("Pinnacle_Draw", np.nan), r.get("Pinnacle_Away", np.nan))
            for s in grid:
                p, lam_h, lam_a = _predict_probs(fit, cfgs[s], str(r["HomeTeam"]), str(r["AwayTeam"]), rhos[s])
                rows.append({
                    "Season": str(r["Season"]),
                    "Date": pd.Timestamp(match_date).strftime("%Y-%m-%d"),
                    "HomeTeam": r["HomeTeam"],
                    "AwayTeam": r["AwayTeam"],
                    "FTHG": int(r["FTHG"]),
                    "FTAG": int(r["FTAG"]),
                    "Shrink": s,
                    "HomeAdvantageLog": fit.home_advantage,
                    "HomeAdvantageUsed": effective_home_advantage(fit, cfgs[s]),
                    "Rho": rhos[s],
                    "Lambda_Home": lam_h,
                    "Lambda_Away": lam_a,
                    "P_Home": p[0], "P_Draw": p[1], "P_Away": p[2],
                    "Outcome": "HDA"[o],
                    "RPS": rps(p, o),
                    "LogLoss": -math.log(max(p[o], 1e-12)),
                    "Correct": int(int(np.argmax(p)) == o),
                    "Mkt_RPS": rps(mkt, o) if mkt is not None else np.nan,
                })
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(dates)} dates, {len(rows) // len(grid)} matches scored")

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    def _summ(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "Matches": len(g),
            "RPS": g["RPS"].mean(),
            "LogLoss": g["LogLoss"].mean(),
            "Accuracy": g["Correct"].mean(),
            "AvgP_Home": g["P_Home"].mean(),
            "ActualHomeRate": (g["Outcome"] == "H").mean(),
            "AvgP_Draw": g["P_Draw"].mean(),
            "ActualDrawRate": (g["Outcome"] == "D").mean(),
            "AvgLambdaHome": g["Lambda_Home"].mean(),
            "ActualHomeGoals": g["FTHG"].mean(),
            "AvgHomeAdvantageLog": g["HomeAdvantageLog"].mean(),
            "Market_RPS": g["Mkt_RPS"].mean(),
            "MarketMatches": int(g["Mkt_RPS"].notna().sum()),
        })

    per_season = detail.groupby(["Season", "Shrink"]).apply(_summ, include_groups=False).reset_index()
    overall = detail.groupby("Shrink").apply(_summ, include_groups=False).reset_index()
    overall.insert(0, "Season", "ALL")
    summary = pd.concat([per_season, overall], ignore_index=True)

    base = summary[summary["Shrink"] == 0.0].set_index("Season")["RPS"]
    summary["RPS_Delta_vs_Base"] = summary["RPS"] - summary["Season"].map(base)
    return detail, summary


def verdict(summary: pd.DataFrame) -> str:
    seasons = summary[summary["Season"] != "ALL"]
    overall = summary[summary["Season"] == "ALL"].set_index("Shrink")
    best = float(overall["RPS"].idxmin())
    if best == 0.0:
        return "No shrink beats the current model overall. Keep home_advantage_shrink: 0.0."
    wins = seasons[seasons["Shrink"] == best]["RPS_Delta_vs_Base"]
    n_better = int((wins < 0).sum())
    delta = float(overall.loc[best, "RPS_Delta_vs_Base"])
    msg = (f"Best shrink {best:.2f}: overall RPS change {delta:+.4f}, "
           f"better in {n_better} of {len(wins)} seasons.")
    if n_better == len(wins) and len(wins) >= 2:
        msg += f" Consistent gain: consider home_advantage_shrink: {best}."
    else:
        msg += " Not consistent across seasons: keep 0.0."
    return msg


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building master database...")
    master = build_master(cfg)
    print(f"Master rows: {len(master)}")

    detail, summary = run_sweep(master, cfg)
    if summary.empty:
        print("No matches scored; nothing to report.")
        return

    detail.to_csv(out_dir / "home_advantage_sweep_detail.csv", index=False)
    summary.to_csv(out_dir / "home_advantage_sweep_summary.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["Season", "Shrink", "Matches", "RPS", "RPS_Delta_vs_Base", "LogLoss",
            "AvgP_Home", "ActualHomeRate", "AvgLambdaHome", "ActualHomeGoals", "Market_RPS"]
    table = summary[cols].round(4)
    print(table.to_string(index=False))
    v = verdict(summary)
    print("\n" + v)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("## Home advantage sweep\n\n")
            fh.write(f"**{v}**\n\n")
            fh.write(table.to_markdown(index=False) if _has_tabulate() else "```\n" + table.to_string(index=False) + "\n```")
            fh.write("\n")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
