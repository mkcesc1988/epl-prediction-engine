"""FPL expected-points projections built on the V1.2 team model.

The team model already predicts goals for and against for every fixture.
This script turns that into Fantasy Premier League expected points per
player, then plans the user's squad:

1. Team layer   - V1.2 attack/defence ratings give each team's expected goals
                  for and against in every upcoming fixture (next N gameweeks),
                  plus clean-sheet and goals-conceded distributions.
2. Player layer - each player's share of his team's chances (season xG/xA per
                  90 relative to the team's xG per match), how often he starts,
                  his availability flag, and a shrunk per-90 rate for
                  everything else (bonus, defensive contributions, saves, cards).
3. Squad layer  - for the configured FPL team ID: best XI and captain for the
                  next gameweek, estimated free transfers, and the best single
                  and double transfers over the horizon, within budget and the
                  three-per-club limit.

Outputs (data/processed/):
- fpl_player_projections_latest.csv  every player, xPts per gameweek
- fpl_transfer_options_latest.csv    ranked single/double transfers
- fpl_squad_report_latest.md         readable plan for the configured team

Everything is fetched from the public FPL API (no key needed). Selling prices
are approximated with current prices because the public picks endpoint does
not expose them.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from model_v11 import _score_matrix
from pipeline import load_config, normalize_team

FPL_BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 EPL prediction engine"}

POS_NAME = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_PTS = 3
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
HIT_COST = 4.0


# --------------------------------------------------------------------------
# Fetching (public FPL API)
# --------------------------------------------------------------------------

def _get_json(path: str) -> object:
    from daily_predictions import _get_json_with_retry

    return _get_json_with_retry(f"{FPL_BASE}/{path}", HEADERS, f"FPL {path}")


def fetch_fpl() -> tuple[dict, list]:
    return _get_json("bootstrap-static/"), _get_json("fixtures/")


def fetch_entry(team_id: int, event: int) -> tuple[dict, dict]:
    picks = _get_json(f"entry/{team_id}/event/{event}/picks/")
    history = _get_json(f"entry/{team_id}/history/")
    return picks, history


# --------------------------------------------------------------------------
# Team layer
# --------------------------------------------------------------------------

def fixture_frame(bootstrap: dict, fixtures: list) -> pd.DataFrame:
    team_name = {int(t["id"]): normalize_team(t["name"]) for t in bootstrap["teams"]}
    rows = []
    for f in fixtures:
        if f.get("event") is None:
            continue
        rows.append({
            "FixtureId": f.get("id"),
            "GW": int(f["event"]),
            "HomeId": int(f["team_h"]),
            "AwayId": int(f["team_a"]),
            "HomeTeam": team_name[int(f["team_h"])],
            "AwayTeam": team_name[int(f["team_a"])],
            "Finished": bool(f.get("finished") or f.get("finished_provisional")),
        })
    return pd.DataFrame(rows)


def team_games_played(fx: pd.DataFrame) -> dict[int, int]:
    done = fx[fx["Finished"]]
    counts = pd.concat([done["HomeId"], done["AwayId"]]).value_counts()
    return {int(k): int(v) for k, v in counts.items()}


def _expected_floor_half(dist: np.ndarray) -> float:
    goals = np.arange(len(dist))
    return float((dist * (goals // 2)).sum())


def team_fixture_projections(fx: pd.DataFrame, lambdas, rho: float, max_goal: int = 10) -> pd.DataFrame:
    """One row per team per fixture with goals for/against and clean-sheet odds.

    ``lambdas(home, away) -> (lam_home, lam_away)`` comes from the team model.
    """
    rows = []
    for _, f in fx.iterrows():
        lh, la = lambdas(f["HomeTeam"], f["AwayTeam"])
        m = np.asarray(_score_matrix(float(lh), float(la), float(rho), max_goal))
        home_conc = m.sum(axis=0)  # distribution of away goals
        away_conc = m.sum(axis=1)  # distribution of home goals
        base = {"FixtureId": f["FixtureId"], "GW": f["GW"]}
        rows.append({**base, "TeamId": f["HomeId"], "OppId": f["AwayId"], "Home": True,
                     "xGF": lh, "xGA": la, "P_CS": float(home_conc[0]),
                     "E_ConcededPairs": _expected_floor_half(home_conc)})
        rows.append({**base, "TeamId": f["AwayId"], "OppId": f["HomeId"], "Home": False,
                     "xGF": la, "xGA": lh, "P_CS": float(away_conc[0]),
                     "E_ConcededPairs": _expected_floor_half(away_conc)})
    return pd.DataFrame(rows)


def model_lambdas(cfg: dict, reference_date: pd.Timestamp):
    """Fit V1.2 exactly as daily_predictions does and return a lambda function."""
    from daily_predictions import _build_training_history, fetch_current_understat_completed
    from model_v12 import _estimate_rho_from_history, _fit_strengths, effective_home_advantage

    season = int(cfg.get("daily", {}).get("current_season", 2026))
    history = _build_training_history(cfg, fetch_current_understat_completed(season))
    fit = _fit_strengths(history, reference_date, cfg)
    if fit is None:
        raise RuntimeError("V1.2 could not fit enough history for FPL projections")
    rho = _estimate_rho_from_history(history, fit, cfg)
    ha = effective_home_advantage(fit, cfg)
    mc = cfg.get("model_v12", {})
    floor, cap = float(mc.get("lambda_floor", 0.15)), float(mc.get("lambda_cap", 4.5))

    def lam(home: str, away: str) -> tuple[float, float]:
        lh = math.exp(np.clip(fit.intercept + ha + fit.team_attack(home) - fit.team_defense(away), -4, 3))
        la = math.exp(np.clip(fit.intercept + fit.team_attack(away) - fit.team_defense(home), -4, 3))
        return max(floor, min(lh, cap)), max(floor, min(la, cap))

    return lam, float(rho)


# --------------------------------------------------------------------------
# Player layer
# --------------------------------------------------------------------------

def player_rates(bootstrap: dict, games: dict[int, int], prior_minutes: float = 270.0) -> pd.DataFrame:
    """Per-player involvement rates, shrunk toward position averages."""
    p = pd.DataFrame(bootstrap["elements"])
    num = ["expected_goals", "expected_assists", "minutes", "starts", "goals_scored",
           "assists", "clean_sheets", "goals_conceded", "total_points", "now_cost"]
    for c in num:
        p[c] = pd.to_numeric(p.get(c, 0), errors="coerce").fillna(0.0)
    p["element_type"] = p["element_type"].astype(int)
    p["team"] = p["team"].astype(int)
    p["TeamGames"] = p["team"].map(games).fillna(0).clip(lower=1)

    team_xg_pm = p.groupby("team")["expected_goals"].sum() / p.groupby("team")["TeamGames"].first()
    league_xg_pm = float(team_xg_pm.mean()) if len(team_xg_pm) else 1.3
    p["TeamXGpm"] = p["team"].map(team_xg_pm).clip(lower=0.3).fillna(league_xg_pm)

    m90 = p["minutes"] / 90.0
    p["XGShareRaw"] = (p["expected_goals"] / m90.where(m90 > 0)) / p["TeamXGpm"]
    p["XAShareRaw"] = (p["expected_assists"] / m90.where(m90 > 0)) / p["TeamXGpm"]

    # Points not explained by appearances/goals/assists/clean sheets/conceded:
    # bonus, defensive contributions, saves, cards, penalties saved/missed.
    gpts = p["element_type"].map(GOAL_PTS)
    cspts = p["element_type"].map(CS_PTS)
    sub_apps = ((p["minutes"] - 85 * p["starts"]) / 25).clip(lower=0).round()
    app_pts = 2 * p["starts"] + sub_apps
    conc_pen = np.where(p["element_type"] <= 2, (p["goals_conceded"] // 2), 0)
    p["OtherPts"] = (p["total_points"] - app_pts - gpts * p["goals_scored"]
                     - ASSIST_PTS * p["assists"] - cspts * p["clean_sheets"] + conc_pen)
    p["OtherRawP90"] = p["OtherPts"] / m90.where(m90 > 0)

    k = prior_minutes / 90.0
    regulars = p[p["minutes"] >= prior_minutes]
    for raw, out, total in [("XGShareRaw", "XGShare", "expected_goals"),
                            ("XAShareRaw", "XAShare", "expected_assists")]:
        prior = (regulars.groupby("element_type")[total].sum()
                 / (regulars.groupby("element_type")["minutes"].sum() / 90.0)
                 / league_xg_pm)
        pri = p["element_type"].map(prior).fillna(0.0)
        p[out] = (p[raw].fillna(0) * m90 + pri * k) / (m90 + k)
    prior_other = regulars.groupby("element_type")["OtherPts"].sum() / (regulars.groupby("element_type")["minutes"].sum() / 90.0)
    pri = p["element_type"].map(prior_other).fillna(0.0)
    p["OtherP90"] = (p["OtherRawP90"].fillna(0) * m90 + pri * k) / (m90 + k)

    p["StartRate"] = (p["starts"] / p["TeamGames"]).clip(0, 1)
    p["MinutesRate"] = (p["minutes"] / (90 * p["TeamGames"])).clip(0, 1)
    p["SubRate"] = (sub_apps / p["TeamGames"]).clip(lower=0)
    p["SubRate"] = np.minimum(p["SubRate"], 1 - p["StartRate"])
    return p


def availability(row: pd.Series, first_gw: bool) -> float:
    status = str(row.get("status", "a"))
    chance = row.get("chance_of_playing_next_round")
    if status == "u":
        return 0.0
    if first_gw:
        if chance is not None and not (isinstance(chance, float) and math.isnan(chance)):
            return float(chance) / 100.0
        return 0.0 if status in {"i", "s", "n"} else 1.0
    if status in {"i", "s", "n"}:
        return 0.5
    return 1.0


def project_players(players: pd.DataFrame, team_fx: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    out = players[["id", "web_name", "team", "element_type", "now_cost", "status",
                   "total_points", "selected_by_percent"]].copy() if "selected_by_percent" in players else \
        players[["id", "web_name", "team", "element_type", "now_cost", "status", "total_points"]].copy()
    out["Pos"] = out["element_type"].map(POS_NAME)
    out["Price"] = out["now_cost"] / 10.0
    pos = players["element_type"].to_numpy()
    for i, gw in enumerate(gws):
        tf = team_fx[team_fx["GW"] == gw]
        avail = players.apply(availability, axis=1, first_gw=(i == 0)).to_numpy()
        p60 = avail * players["StartRate"].to_numpy() * 0.92
        p_play = np.minimum(1.0, avail * (players["StartRate"].to_numpy() + players["SubRate"].to_numpy()))
        mins = avail * players["MinutesRate"].to_numpy()
        total = np.zeros(len(players))
        for _, fxr in tf.iterrows():  # handles blanks (no row) and doubles (two rows)
            mask = (players["team"].to_numpy() == fxr["TeamId"]).astype(float)
            goals = mins * players["XGShare"].to_numpy() * fxr["xGF"]
            assists = mins * players["XAShare"].to_numpy() * fxr["xGF"]
            cs = p60 * fxr["P_CS"] * np.vectorize(CS_PTS.get)(pos)
            conc = np.where(pos <= 2, p60 * fxr["E_ConcededPairs"], 0.0)
            pts = (2 * p60 + 1 * np.clip(p_play - p60, 0, None)
                   + goals * np.vectorize(GOAL_PTS.get)(pos) + assists * ASSIST_PTS
                   + cs - conc + mins * players["OtherP90"].to_numpy())
            total += mask * pts
        out[f"xPts_GW{gw}"] = total
    out["xPts_Total"] = out[[f"xPts_GW{g}" for g in gws]].sum(axis=1)
    return out.sort_values("xPts_Total", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Squad layer
# --------------------------------------------------------------------------

def best_xi(squad: pd.DataFrame, col: str) -> tuple[list[int], int]:
    """Best legal XI by ``col`` (greedy is optimal for FPL's min/max shape)."""
    chosen: list[int] = []
    count = {k: 0 for k in XI_MIN}
    ranked = squad.sort_values(col, ascending=False)
    for pos_id, need in XI_MIN.items():
        for pid in ranked[ranked["element_type"] == pos_id]["id"].head(need):
            chosen.append(int(pid))
            count[pos_id] += 1
    for _, r in ranked.iterrows():
        if len(chosen) == 11:
            break
        if int(r["id"]) in chosen or count[int(r["element_type"])] >= XI_MAX[int(r["element_type"])]:
            continue
        chosen.append(int(r["id"]))
        count[int(r["element_type"])] += 1
    xi = squad[squad["id"].isin(chosen)]
    captain = int(xi.sort_values(col, ascending=False)["id"].iloc[0])
    return chosen, captain


def squad_value(squad: pd.DataFrame, gws: list[int], decay: float) -> float:
    total = 0.0
    for i, gw in enumerate(gws):
        col = f"xPts_GW{gw}"
        xi, cap = best_xi(squad, col)
        pts = squad.set_index("id").loc[xi, col]
        total += (decay ** i) * (pts.sum() + pts.loc[cap])
    return float(total)


def estimate_free_transfers(history: dict, next_gw: int) -> int:
    chips = {int(c["event"]) for c in history.get("chips", []) if c.get("name") in {"wildcard", "freehit"}}
    made = {int(h["event"]): int(h.get("event_transfers", 0)) for h in history.get("current", [])}
    ft = 0
    for gw in range(1, next_gw):
        if gw == 1:
            ft = 1
            continue
        if gw not in chips:
            ft = max(ft - made.get(gw, 0), 0)
        ft = min(ft + 1, 5)
    return ft


def transfer_options(proj: pd.DataFrame, squad_ids: list[int], bank: float, gws: list[int],
                     decay: float, free_transfers: int, top_per_slot: int = 8) -> pd.DataFrame:
    proj = proj.copy()
    proj["Horizon"] = sum((decay ** i) * proj[f"xPts_GW{g}"] for i, g in enumerate(gws))
    squad = proj[proj["id"].isin(squad_ids)].copy()
    base = squad_value(squad, gws, decay)
    pool = proj[~proj["id"].isin(squad_ids) & (proj["status"] != "u")]
    shortlist = {pos: pool[pool["element_type"] == pos].nlargest(60, "Horizon") for pos in POS_NAME}

    def legal(new_squad: pd.DataFrame) -> bool:
        return int(new_squad["team"].value_counts().max()) <= MAX_PER_CLUB

    singles = []
    for _, out_p in squad.iterrows():
        budget = bank + out_p["Price"]
        cands = shortlist[int(out_p["element_type"])]
        cands = cands[cands["Price"] <= budget + 1e-9]
        for _, in_p in cands.iterrows():
            new = pd.concat([squad[squad["id"] != out_p["id"]], in_p.to_frame().T]).astype(squad.dtypes.to_dict())
            if not legal(new):
                continue
            gain = squad_value(new, gws, decay) - base
            singles.append({"Type": "single", "Out": out_p["web_name"], "In": in_p["web_name"],
                            "OutIds": (int(out_p["id"]),), "InIds": (int(in_p["id"]),),
                            "Cost": float(in_p["Price"] - out_p["Price"]), "Gain": gain})
    singles_df = pd.DataFrame(singles)
    if singles_df.empty:
        return singles_df

    doubles = []
    best_ins = (singles_df.sort_values("Gain", ascending=False)
                .groupby(singles_df["OutIds"].map(lambda t: t[0])).head(top_per_slot))
    by_out = {k: g for k, g in best_ins.groupby(best_ins["OutIds"].map(lambda t: t[0]))}
    for o1, o2 in itertools.combinations(by_out, 2):
        for _, a in by_out[o1].iterrows():
            for _, b in by_out[o2].iterrows():
                if a["InIds"][0] == b["InIds"][0] or a["Cost"] + b["Cost"] > bank + 1e-9:
                    continue
                ids = [i for i in squad_ids if i not in (o1, o2)] + [a["InIds"][0], b["InIds"][0]]
                new = proj[proj["id"].isin(ids)]
                if len(new) != 15 or not legal(new):
                    continue
                doubles.append({"Type": "double", "Out": f"{a['Out']} + {b['Out']}",
                                "In": f"{a['In']} + {b['In']}", "OutIds": (o1, o2),
                                "InIds": (a["InIds"][0], b["InIds"][0]),
                                "Cost": a["Cost"] + b["Cost"],
                                "Gain": squad_value(new, gws, decay) - base})
    allopts = pd.concat([singles_df, pd.DataFrame(doubles)], ignore_index=True)
    n = allopts["Type"].map({"single": 1, "double": 2})
    allopts["HitCost"] = np.clip(n - free_transfers, 0, None) * HIT_COST
    allopts["NetGain"] = allopts["Gain"] - allopts["HitCost"]
    allopts["BankAfter"] = bank - allopts["Cost"]
    return allopts.sort_values("NetGain", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def squad_report(proj: pd.DataFrame, squad_ids: list[int], opts: pd.DataFrame, bank: float,
                 free_transfers: int, gws: list[int], team_name: str = "") -> str:
    team_short = proj.attrs.get("team_short", {})
    nxt = f"xPts_GW{gws[0]}"
    squad = proj[proj["id"].isin(squad_ids)].copy()
    xi, cap = best_xi(squad, nxt)
    squad["XI"] = squad["id"].isin(xi)
    lines = [f"# FPL plan {team_name} - GW{gws[0]} (horizon GW{gws[0]}-{gws[-1]})", "",
             f"Bank £{bank:.1f}m · estimated free transfers: {free_transfers} "
             "(selling prices approximated by current prices)", "",
             f"**Captain:** {squad.set_index('id').loc[cap, 'web_name']}", "",
             "| Player | Club | Pos | £ | " + " | ".join(f"GW{g}" for g in gws) + " | Start |",
             "|---|---|---|---|" + "---|" * len(gws) + "---|"]
    for _, r in squad.sort_values(["XI", "element_type", nxt], ascending=[False, True, False]).iterrows():
        vals = " | ".join(f"{r[f'xPts_GW{g}']:.1f}" for g in gws)
        lines.append(f"| {r['web_name']} | {team_short.get(int(r['team']), r['team'])} | {r['Pos']} | "
                     f"{r['Price']:.1f} | {vals} | {'XI' if r['XI'] else 'bench'} |")
    lines += ["", "## Best transfers", "", "| Out | In | Gain | Hit | Net | Bank after |", "|---|---|---|---|---|---|"]
    for _, o in opts.head(10).iterrows():
        lines.append(f"| {o['Out']} | {o['In']} | {o['Gain']:.1f} | -{o['HitCost']:.0f} | "
                     f"{o['NetGain']:.1f} | £{o['BankAfter']:.1f}m |")
    if not opts.empty and opts["NetGain"].iloc[0] <= 0.5:
        lines += ["", "No transfer clears the bar this week - consider rolling the free transfer."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--team-id", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    fcfg = cfg.get("fpl", {})
    team_id = args.team_id or fcfg.get("team_id")
    horizon = int(args.horizon or fcfg.get("horizon_gameweeks", 3))
    decay = float(fcfg.get("horizon_decay", 0.9))
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bootstrap, fixtures = fetch_fpl()
    next_ev = next(e for e in bootstrap["events"] if e.get("is_next"))
    current_ev = next((e for e in bootstrap["events"] if e.get("is_current")), None)
    gws = list(range(int(next_ev["id"]), min(int(next_ev["id"]) + horizon, 39)))

    fx = fixture_frame(bootstrap, fixtures)
    upcoming = fx[fx["GW"].isin(gws)]
    lam, rho = model_lambdas(cfg, pd.Timestamp(next_ev["deadline_time"]).tz_convert(None).normalize())
    team_fx = team_fixture_projections(upcoming, lam, rho)

    players = player_rates(bootstrap, team_games_played(fx))
    proj = project_players(players, team_fx, gws)
    proj.attrs["team_short"] = {int(t["id"]): t["short_name"] for t in bootstrap["teams"]}
    proj.insert(3, "Club", proj["team"].map(proj.attrs["team_short"]))
    proj.to_csv(out_dir / "fpl_player_projections_latest.csv", index=False)
    print(proj.head(25)[["web_name", "Club", "Pos", "Price"] + [c for c in proj if c.startswith("xPts")]]
          .round(2).to_string(index=False))

    if not team_id or current_ev is None:
        return
    picks, history = fetch_entry(int(team_id), int(current_ev["id"]))
    squad_ids = [int(p["element"]) for p in picks["picks"]]
    bank = float(picks["entry_history"]["bank"]) / 10.0
    if picks.get("active_chip") == "freehit":
        print("Free Hit was active last gameweek; squad shown is the Free Hit team.")
    fts = estimate_free_transfers(history, int(next_ev["id"]))
    opts = transfer_options(proj, squad_ids, bank, gws, decay, fts)
    opts.drop(columns=["OutIds", "InIds"]).to_csv(out_dir / "fpl_transfer_options_latest.csv", index=False)
    report = squad_report(proj, squad_ids, opts, bank, fts, gws, team_name=str(team_id))
    (out_dir / "fpl_squad_report_latest.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
