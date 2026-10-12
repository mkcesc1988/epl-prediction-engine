"""Walk-forward A/B test of the underdog-bias fixes against the previous model.

Variants (each refit walk-forward on matches strictly before each match date):
  baseline  promoted prior OFF, home_advantage_shrink 0.0   (model before the fix)
  fixed     promoted prior ON,  home_advantage_shrink from config (0.15)

Scored on the last ``eval_last_n_seasons`` seasons against Pinnacle closing
1X2 prices (de-vigged):
  * RPS / log loss overall, for promoted-team matches and for Aug-Oct matches
  * plus-money calibration: mean model probability vs actual win rate
  * flat 1-unit moneyline bets where decision EV > 5%, using the decision
    layer's market anchor (old 0.45 vs new 0.85 + early-season bonus)

Outputs (data/processed/):
  bias_fix_backtest_summary.csv
  bias_fix_backtest_detail.csv

Run: python bias_fix_backtest.py
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gameweek_rankings import _decision_probability
from home_advantage_sweep import _devig, _outcome, _predict_probs, rps
from model_v12 import _estimate_rho_from_history, _fit_strengths, promoted_team_prior, season_start_year
from pipeline import build_master, load_config

EV_THRESHOLD = 0.05


def _variants(cfg: dict) -> dict[str, dict]:
    base = copy.deepcopy(cfg)
    base.setdefault("model_v12", {})["promoted_prior_enabled"] = False
    base["model_v12"]["home_advantage_shrink"] = 0.0
    base.setdefault("decision", {})["market_anchor_weight_moneyline"] = 0.45
    base["decision"]["early_season_anchor_bonus"] = 0.0
    return {"baseline": base, "fixed": copy.deepcopy(cfg)}


def _approx_gameweek(date: pd.Timestamp) -> float:
    """Rough gameweek from the date (season starts mid-August, ~1 GW/week)."""
    start = pd.Timestamp(year=int(season_start_year([date])[0]), month=8, day=12)
    return max(1.0, math.floor((date - start).days / 7.0) + 1.0)


def run_backtest(master: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_seasons = int(cfg.get("home_advantage_sweep", {}).get("eval_last_n_seasons", 3))
    variants = _variants(cfg)

    df = master.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    seasons = list(dict.fromkeys(df["Season"].astype(str)))
    eval_seasons = set(seasons[-n_seasons:])

    rows: list[dict] = []
    dates = list(df.groupby("Date", sort=True))
    for i, (match_date, day_games) in enumerate(dates):
        targets = day_games[
            day_games["Season"].astype(str).isin(eval_seasons)
            & day_games["FTHG"].notna() & day_games["FTAG"].notna()
        ]
        if targets.empty:
            continue
        history = df[df["Date"] < match_date]
        ref = pd.Timestamp(match_date)
        _, _, promoted = promoted_team_prior(history, ref, cfg)
        gw = _approx_gameweek(ref)

        for name, vcfg in variants.items():
            fit = _fit_strengths(history, ref, vcfg)
            if fit is None:
                continue
            rho = _estimate_rho_from_history(history, fit, vcfg)
            for _, r in targets.iterrows():
                h, a = str(r["HomeTeam"]), str(r["AwayTeam"])
                o = _outcome(r["FTHG"], r["FTAG"])
                p, _, _ = _predict_probs(fit, vcfg, h, a, rho)
                odds = np.array([r.get("Pinnacle_Home", np.nan), r.get("Pinnacle_Draw", np.nan),
                                 r.get("Pinnacle_Away", np.nan)], dtype=float)
                mkt = _devig(*odds)
                row = {
                    "Variant": name, "Season": str(r["Season"]), "Date": ref.strftime("%Y-%m-%d"),
                    "HomeTeam": h, "AwayTeam": a, "Outcome": "HDA"[o],
                    "PromotedMatch": int(h in promoted or a in promoted),
                    "EarlySeason": int(ref.month in (8, 9, 10)),
                    "P_Home": p[0], "P_Draw": p[1], "P_Away": p[2],
                    "RPS": rps(p, o), "LogLoss": -math.log(max(p[o], 1e-12)),
                    "Mkt_RPS": rps(mkt, o) if mkt is not None else np.nan,
                    "Mkt_LogLoss": -math.log(max(mkt[o], 1e-12)) if mkt is not None else np.nan,
                }
                # Moneyline sides (home/away) at Pinnacle prices
                for k, side in ((0, "Home"), (2, "Away")):
                    if mkt is None:
                        row.update({f"{side}_Odds": np.nan, f"{side}_DecisionP": np.nan})
                        continue
                    dp, _ = _decision_probability(float(p[k]), 0.0, float(mkt[k]), "Moneyline", vcfg, gw)
                    row.update({f"{side}_Odds": odds[k], f"{side}_MktP": mkt[k], f"{side}_DecisionP": dp,
                                f"{side}_Promoted": int((h if side == "Home" else a) in promoted)})
                rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(dates)} dates")

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    return detail, summarize(detail)


def _sides(detail: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side, k, letter in (("Home", "P_Home", "H"), ("Away", "P_Away", "A")):
        s = detail[["Variant", "Season", "EarlySeason", k, f"{side}_Odds", f"{side}_DecisionP", f"{side}_Promoted", "Outcome"]].copy()
        s.columns = ["Variant", "Season", "EarlySeason", "ModelP", "Odds", "DecisionP", "Promoted", "Outcome"]
        s["Side"] = side
        s["Won"] = (s["Outcome"] == letter).astype(int)
        parts.append(s)
    s = pd.concat(parts, ignore_index=True).dropna(subset=["Odds"])
    s["DecisionEV"] = s["DecisionP"] * s["Odds"] - 1.0
    s["Return"] = np.where(s["Won"] == 1, s["Odds"] - 1.0, -1.0)
    return s


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    sides = _sides(detail)
    plus = sides[sides["Odds"] >= 2.0]
    out = []
    for variant in detail["Variant"].unique():
        d = detail[detail["Variant"] == variant]
        sv = sides[sides["Variant"] == variant]
        pv = plus[plus["Variant"] == variant]
        for label, mask_d, mask_p in (
            ("ALL", d.index == d.index, pv.index == pv.index),
            ("PromotedTeams", d["PromotedMatch"] == 1, pv["Promoted"] == 1),
            ("EarlySeason", d["EarlySeason"] == 1, pv["EarlySeason"] == 1),
            ("HomeSides", d.index == d.index, pv["Side"] == "Home"),
        ):
            dd, pp = d[mask_d], pv[mask_p]
            bets = sv[sv["DecisionEV"] > EV_THRESHOLD]
            if label == "PromotedTeams":
                bets = bets[bets["Promoted"] == 1]
            elif label == "EarlySeason":
                bets = bets[bets["EarlySeason"] == 1]
            elif label == "HomeSides":
                bets = bets[bets["Side"] == "Home"]
            out.append({
                "Variant": variant, "Segment": label, "Matches": len(dd),
                "RPS": dd["RPS"].mean(), "Market_RPS": dd["Mkt_RPS"].mean(),
                "LogLoss": dd["LogLoss"].mean(), "Market_LogLoss": dd["Mkt_LogLoss"].mean(),
                "PlusMoneySides": len(pp), "PlusMoney_AvgModelP": pp["ModelP"].mean(),
                "PlusMoney_ActualWinRate": pp["Won"].mean(),
                "PlusMoney_Overrate": pp["ModelP"].mean() - pp["Won"].mean(),
                "Bets_EVgt5": len(bets), "Bets_ROI": bets["Return"].mean() if len(bets) else np.nan,
                "Longshot_Bets": int((bets["Odds"] >= 4.0).sum()),
                "Longshot_ROI": bets.loc[bets["Odds"] >= 4.0, "Return"].mean() if (bets["Odds"] >= 4.0).any() else np.nan,
            })
    return pd.DataFrame(out)


def verdict(summary: pd.DataFrame) -> list[str]:
    s = summary.set_index(["Variant", "Segment"])
    lines = []
    for seg in ("ALL", "PromotedTeams", "EarlySeason", "HomeSides"):
        if ("baseline", seg) not in s.index or ("fixed", seg) not in s.index:
            continue
        b, f = s.loc[("baseline", seg)], s.loc[("fixed", seg)]
        lines.append(
            f"{seg}: RPS {b.RPS:.4f} -> {f.RPS:.4f} (market {b.Market_RPS:.4f}); "
            f"plus-money overrate {b.PlusMoney_Overrate:+.3f} -> {f.PlusMoney_Overrate:+.3f}; "
            f"bets {int(b.Bets_EVgt5)} @ {b.Bets_ROI:+.3f} -> {int(f.Bets_EVgt5)} @ {f.Bets_ROI:+.3f}"
        )
    b, f = s.loc[("baseline", "ALL")], s.loc[("fixed", "ALL")]
    lines.append("VERDICT: " + ("fixed model improves overall RPS." if f.RPS < b.RPS
                                else "fixed model does NOT improve overall RPS - review before relying on it."))
    return lines


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building master database...")
    master = build_master(cfg)
    print(f"Master rows: {len(master)}")
    detail, summary = run_backtest(master, cfg)
    if summary.empty:
        print("No matches scored.")
        return
    detail.to_csv(out_dir / "bias_fix_backtest_detail.csv", index=False)
    summary.to_csv(out_dir / "bias_fix_backtest_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    lines = verdict(summary)
    print("\n" + "\n".join(lines))
    step = os.environ.get("GITHUB_STEP_SUMMARY")
    if step:
        with open(step, "a", encoding="utf-8") as fh:
            fh.write("## Underdog-bias fix backtest\n\n")
            fh.write("\n".join(f"- {x}" for x in lines) + "\n\n")
            fh.write(summary.round(4).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
