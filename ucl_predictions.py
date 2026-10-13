"""Champions League match predictions and MyBookie bet screening.

UCL games are cross-league, so the per-league V1.2 ratings can't be compared
directly. This module puts every club on one scale and reuses the engine's
goal model and decision rules:

1. Strength: ClubElo ratings (all competitions, opponent-adjusted, so domestic
   and European form are both in it) plus a UCL home advantage.
2. Form: Elo change over the last ``ucl.form_days`` (shown, already inside the
   rating), recent domestic results/xG, and this season's UCL results. For
   big-five clubs a small "results vs xG" correction pulls a team that has
   out- or under-performed its xG over the last 6 league games back toward it.
3. Goals: the Elo win expectancy sets the goal supremacy, and the total comes
   from the bookmaker consensus total nudged by both teams' domestic xG tempo.
   A Dixon-Coles score matrix then prices 1X2, spreads and totals.
4. Decision: the same market anchoring, edge gate and fractional Kelly as the
   EPL engine (``decision`` and ``portfolio`` in config.yaml), priced at the
   MyBookie odds in ``data/manual_ucl_mybookie_odds.csv``.

Every source is optional except ClubElo; failures are reported and skipped.
Outputs go to data/processed/ucl/.
"""
from __future__ import annotations

import difflib
import io
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from leagues import strip_accents
from model_v11 import _score_matrix
from pipeline import load_config, normalize_team

OUT_DIR = Path("data/processed/ucl")
ODDS_PATH = Path("data/manual_ucl_mybookie_odds.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 EPL prediction engine"}

# Our name -> (ClubElo country, ClubElo name candidates, domestic source, source key)
CLUBS: dict[str, dict] = {
    "Feyenoord": {"country": "NED", "elo": ["Feyenoord"], "fd": "N1"},
    "Como": {"country": "ITA", "elo": ["Como"], "understat": "Serie_A"},
    "LASK": {"country": "AUT", "elo": ["LASK", "Linz"], "fd_new": "AUT"},
    "Liverpool": {"country": "ENG", "elo": ["Liverpool"], "understat": "EPL"},
    "Roma": {"country": "ITA", "elo": ["Roma"], "understat": "Serie_A"},
    "Real Madrid": {"country": "ESP", "elo": ["Real Madrid"], "understat": "La_Liga"},
    "Aston Villa": {"country": "ENG", "elo": ["Aston Villa"], "understat": "EPL"},
    "Fenerbahce": {"country": "TUR", "elo": ["Fenerbahce"], "fd": "T1"},
    "Bodo/Glimt": {"country": "NOR", "elo": ["Bodoe/Glimt", "Bodo/Glimt", "Bodoe Glimt", "Glimt"], "fd_new": "NOR"},
    "Dortmund": {"country": "GER", "elo": ["Dortmund"], "understat": "Bundesliga"},
    "Shakhtar": {"country": "UKR", "elo": ["Shakhtar"]},
    "AEK": {"country": "GRE", "elo": ["AEK"], "fd": "G1"},
    "Man City": {"country": "ENG", "elo": ["Man City"], "understat": "EPL"},
    "Paris SG": {"country": "FRA", "elo": ["Paris SG"], "understat": "Ligue_1"},
    "Betis": {"country": "ESP", "elo": ["Betis"], "understat": "La_Liga"},
    "Porto": {"country": "POR", "elo": ["Porto"], "fd": "P1"},
    "Slovan Bratislava": {"country": "SVK", "elo": ["Slovan Bratislava", "Slovan"]},
    "Stuttgart": {"country": "GER", "elo": ["Stuttgart"], "understat": "Bundesliga"},
}


# ---------------------------------------------------------------- helpers
def american_to_decimal(american: float) -> float:
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, strip_accents(a).lower(), strip_accents(b).lower()).ratio()


def elo_expectancy(elo_home: float, elo_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_home + hfa - elo_away) / 400.0))


def lambdas_for(expectancy: float, total: float, rho: float, max_goal: int = 10) -> tuple[float, float]:
    """Split ``total`` expected goals so that P(home) + P(draw)/2 = expectancy."""
    total = float(np.clip(total, 0.8, 6.0))

    def score(sup: float) -> float:
        lh = max(0.05, (total + sup) / 2.0)
        la = max(0.05, (total - sup) / 2.0)
        m = _score_matrix(lh, la, rho, max_goal)
        return float(np.tril(m, -1).sum() + 0.5 * np.trace(m))

    lo, hi = -total + 0.1, total - 0.1
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if score(mid) < expectancy:
            lo = mid
        else:
            hi = mid
    sup = (lo + hi) / 2.0
    return max(0.05, (total + sup) / 2.0), max(0.05, (total - sup) / 2.0)


def total_from_over_prob(p_over: float, line: float, rho: float) -> float:
    """Expected total goals whose Poisson/DC split gives ``p_over`` at ``line``."""
    lo, hi = 0.8, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        m = _score_matrix(mid / 2, mid / 2, rho, 10)
        g = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        if float(m[g > line].sum()) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def matrix_markets(m: np.ndarray) -> dict:
    g = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    diff = np.subtract.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return {"home": float(np.tril(m, -1).sum()), "draw": float(np.trace(m)), "away": float(np.triu(m, 1).sum()),
            "goals": g, "diff": diff}


def selection_probability(m: np.ndarray, market: str, outcome: str, point: float, home: str, away: str) -> tuple[float, float]:
    """(win probability, push probability) for one MyBookie selection."""
    mk = matrix_markets(m)
    if market == "h2h":
        return {home: mk["home"], away: mk["away"], "Draw": mk["draw"]}[outcome], 0.0
    if market == "totals":
        over_p = float(m[mk["goals"] > point].sum())
        push = float(m[mk["goals"] == point].sum())
        return (over_p if outcome == "Over" else 1.0 - over_p - push), push
    if market == "spreads":
        margin = mk["diff"] if outcome == home else -mk["diff"]
        return float(m[margin + point > 0].sum()), float(m[margin + point == 0].sum())
    raise ValueError(market)


# ---------------------------------------------------------------- sources
def fetch_clubelo(day: pd.Timestamp) -> pd.DataFrame:
    last: Exception | None = None
    text = None
    for attempt in range(3):
        for scheme in ("http", "https"):
            try:
                r = requests.get(f"{scheme}://api.clubelo.com/{day:%Y-%m-%d}", headers=HEADERS, timeout=25)
                r.raise_for_status()
                text = r.text
                break
            except requests.RequestException as exc:
                last = exc
        if text is not None:
            break
        import time
        time.sleep(10 * (attempt + 1))
    if text is None:
        raise RuntimeError(f"ClubElo unreachable: {last}")
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "Elo" not in df.columns:
        raise RuntimeError(f"ClubElo returned no ratings for {day:%Y-%m-%d}")
    return df


def match_elo(elo: pd.DataFrame, team: str) -> tuple[str, float] | None:
    info = CLUBS[team]
    pool = elo[elo["Country"].astype(str) == info["country"]]
    if pool.empty:
        pool = elo
    names = pool["Club"].astype(str)
    for cand in info["elo"]:
        exact = pool[names.str.lower() == cand.lower()]
        if not exact.empty:
            return str(exact.iloc[0]["Club"]), float(exact.iloc[0]["Elo"])
    best = max(names, key=lambda n: max(_similar(n, c) for c in info["elo"] + [team]))
    if max(_similar(best, c) for c in info["elo"] + [team]) >= 0.6:
        row = pool[names == best].iloc[0]
        return str(row["Club"]), float(row["Elo"])
    return None


def understat_form(league: str, season: int, n: int = 6) -> tuple[pd.DataFrame, float]:
    """Per-team last-n league games (xG and goals) and the league's xG per team-game."""
    from understatapi import UnderstatClient

    with UnderstatClient() as client:
        matches = client.league(league=league).get_match_data(season=str(season))
    rows = []
    for m in matches or []:
        if str(m.get("isResult")).strip().lower() not in {"true", "1"}:
            continue
        h, a = normalize_team(m["h"]["title"]), normalize_team(m["a"]["title"])
        hg, ag = float(m["goals"]["h"]), float(m["goals"]["a"])
        hx, ax = float(m["xG"]["h"]), float(m["xG"]["a"])
        d = pd.to_datetime(m["datetime"])
        rows.append((d, h, hg, ag, hx, ax))
        rows.append((d, a, ag, hg, ax, hx))
    df = pd.DataFrame(rows, columns=["Date", "Team", "GF", "GA", "xGF", "xGA"]).sort_values("Date")
    league_xg = float(df["xGF"].mean()) if not df.empty else np.nan
    last = df.groupby("Team").tail(n)
    form = last.groupby("Team").agg(
        Games=("GF", "size"), GF=("GF", "mean"), GA=("GA", "mean"), xGF=("xGF", "mean"), xGA=("xGA", "mean"),
        Pts=("GF", lambda s: 0.0),
    )
    pts = last.assign(P=np.select([last.GF > last.GA, last.GF == last.GA], [3, 1], 0)).groupby("Team")["P"].mean()
    form["PPG"] = pts
    form["Results"] = last.groupby("Team").apply(
        lambda g: "".join("W" if f > a else "D" if f == a else "L" for f, a in zip(g.GF, g.GA)), include_groups=False
    )
    return form.drop(columns=["Pts"]), league_xg


def football_data_form(team: str, info: dict, season: int, n: int = 6) -> dict | None:
    """Last-n domestic results from Football-Data (no xG) for non-big-five clubs."""
    if "fd" in info:
        code = f"{str(season)[-2:]}{str(season + 1)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/{info['fd']}.csv"
        cols = ("HomeTeam", "AwayTeam", "FTHG", "FTAG")
    elif "fd_new" in info:
        url = f"https://www.football-data.co.uk/new/{info['fd_new']}.csv"
        cols = ("Home", "Away", "HG", "AG")
    else:
        return None
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="ignore")))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[cols[2], cols[3]])
    df = df[df["Date"] >= pd.Timestamp.now() - pd.Timedelta(days=400)]
    teams = set(df[cols[0]]).union(df[cols[1]])
    name = max(teams, key=lambda t: _similar(str(t), team))
    if _similar(str(name), team) < 0.6:
        return None
    games = df[(df[cols[0]] == name) | (df[cols[1]] == name)].sort_values("Date").tail(n)
    res, gf, ga = [], [], []
    for _, g in games.iterrows():
        home = g[cols[0]] == name
        f, a = (g[cols[2]], g[cols[3]]) if home else (g[cols[3]], g[cols[2]])
        gf.append(f); ga.append(a); res.append("W" if f > a else "D" if f == a else "L")
    pts = [3 if r == "W" else 1 if r == "D" else 0 for r in res]
    return {"Games": len(res), "GF": float(np.mean(gf)), "GA": float(np.mean(ga)), "PPG": float(np.mean(pts)),
            "Results": "".join(res)}


def ucl_results(season: int) -> pd.DataFrame:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY not set")
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        params={"league": 2, "season": season},
        headers={"x-apisports-key": key}, timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    rows = []
    for f in payload.get("response", []):
        if f["fixture"]["status"]["short"] not in {"FT", "AET", "PEN"}:
            continue
        rows.append({
            "Date": f["fixture"]["date"][:10], "Round": f["league"].get("round"),
            "Home": f["teams"]["home"]["name"], "Away": f["teams"]["away"]["name"],
            "HG": f["goals"]["home"], "AG": f["goals"]["away"],
        })
    return pd.DataFrame(rows)


def team_ucl_form(results: pd.DataFrame, team: str) -> str:
    if results.empty:
        return ""
    names = set(results["Home"]).union(results["Away"])
    best = max(names, key=lambda n: _similar(n, team))
    if _similar(best, team) < 0.55:
        return ""
    out = []
    for _, r in results[(results.Home == best) | (results.Away == best)].sort_values("Date").iterrows():
        home = r.Home == best
        f, a = (r.HG, r.AG) if home else (r.AG, r.HG)
        opp = r.Away if home else r.Home
        out.append(f"{'W' if f > a else 'D' if f == a else 'L'} {f}-{a} {'v' if home else '@'} {opp}")
    return "; ".join(out)


def odds_api_consensus(fixtures: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Median no-vig probabilities from non-MyBookie books for each fixture's markets."""
    key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("THE_ODDS_API_KEY not set")
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds",
        params={"apiKey": key, "regions": "us,eu", "markets": "h2h,totals,spreads", "oddsFormat": "decimal"},
        timeout=60,
    )
    r.raise_for_status()
    print(f"Odds API credits remaining: {r.headers.get('x-requests-remaining')}")
    events = r.json() or []
    rows = []
    for _, fx in fixtures.iterrows():
        if not events:
            break
        ev = max(events, key=lambda e: _similar(e["home_team"], fx.HomeTeam) + _similar(e["away_team"], fx.AwayTeam))
        if _similar(ev["home_team"], fx.HomeTeam) + _similar(ev["away_team"], fx.AwayTeam) < 0.9:
            continue
        for book in ev.get("bookmakers", []):
            is_my = "mybookie" in book["key"]
            for mk in book.get("markets", []):
                for o in mk.get("outcomes", []):
                    name = o["name"]
                    if mk["key"] in {"h2h", "spreads"} and name != "Draw":
                        name = fx.HomeTeam if _similar(name, ev["home_team"]) >= _similar(name, ev["away_team"]) else fx.AwayTeam
                    rows.append({"HomeTeam": fx.HomeTeam, "AwayTeam": fx.AwayTeam, "Book": book["key"], "MyBookie": is_my,
                                 "Market": mk["key"], "Outcome": name, "Point": o.get("point"), "Decimal": float(o["price"])})
    return pd.DataFrame(rows)


def consensus_prob(cons: pd.DataFrame, home: str, away: str, market: str, outcome: str, point: float | None) -> tuple[float, int]:
    if cons.empty:
        return np.nan, 0
    g = cons[(cons.HomeTeam == home) & (cons.AwayTeam == away) & (cons.Market == market) & (~cons.MyBookie)]
    probs = []
    for _, b in g.groupby("Book"):
        if market == "h2h":
            p = {r.Outcome: 1 / r.Decimal for r in b.itertuples()}
            if {home, away, "Draw"} <= set(p):
                probs.append(p[outcome] / (p[home] + p[away] + p["Draw"]))
        elif market == "totals":
            same = b[np.isclose(b.Point.astype(float), point)]
            p = {r.Outcome: 1 / r.Decimal for r in same.itertuples()}
            if {"Over", "Under"} <= set(p):
                probs.append(p[outcome] / (p["Over"] + p["Under"]))
        elif market == "spreads":
            other = away if outcome == home else home
            sel = b[(b.Outcome == outcome) & np.isclose(b.Point.astype(float), point)]
            opp = b[(b.Outcome == other) & np.isclose(b.Point.astype(float), -point)]
            if not sel.empty and not opp.empty:
                p1, p2 = 1 / sel.Decimal.iloc[0], 1 / opp.Decimal.iloc[0]
                probs.append(p1 / (p1 + p2))
    return (float(np.median(probs)), len(probs)) if probs else (np.nan, 0)


# ---------------------------------------------------------------- model
def build(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    uc = cfg.get("ucl", {})
    hfa = float(uc.get("home_advantage_elo", 65.0))
    form_days = int(uc.get("form_days", 42))
    xg_luck_elo = float(uc.get("xg_luck_elo_per_goal", 30.0))
    xg_luck_cap = float(uc.get("xg_luck_cap_elo", 25.0))
    tempo_weight = float(uc.get("tempo_weight", 0.5))
    tempo_cap = float(uc.get("tempo_cap", 0.08))
    rho = float(cfg.get("model_v12", {}).get("rho_fallback", -0.08))
    season = int(cfg.get("daily", {}).get("current_season", 2026))
    notes: list[str] = []

    lines = pd.read_csv(ODDS_PATH)
    lines["Decimal"] = lines["AmericanOdds"].map(american_to_decimal)
    lines["Point"] = pd.to_numeric(lines["Point"], errors="coerce")
    fixtures = lines[["Date", "KickoffPT", "HomeTeam", "AwayTeam"]].drop_duplicates().reset_index(drop=True)
    teams = sorted(set(fixtures.HomeTeam).union(fixtures.AwayTeam))

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    elo_now = fetch_clubelo(today)
    elo_then = fetch_clubelo(today - pd.Timedelta(days=form_days))

    team_rows = {}
    for t in teams:
        now = match_elo(elo_now, t)
        then = match_elo(elo_then, t)
        if now is None:
            raise RuntimeError(f"No ClubElo rating for {t}")
        team_rows[t] = {"Team": t, "EloClub": now[0], "Elo": now[1],
                        f"EloChange{form_days}d": now[1] - then[1] if then else np.nan}

    # Domestic form.
    understat_cache: dict[str, tuple[pd.DataFrame, float]] = {}
    for t in teams:
        info = CLUBS[t]
        row = team_rows[t]
        try:
            if "understat" in info:
                lg = info["understat"]
                if lg not in understat_cache:
                    understat_cache[lg] = understat_form(lg, season)
                form, league_xg = understat_cache[lg]
                f = form.loc[t]
                row.update({"DomGames": int(f.Games), "DomResults": f.Results, "DomPPG": f.PPG,
                            "DomGF": f.GF, "DomGA": f.GA, "DomxGF": f.xGF, "DomxGA": f.xGA, "LeagueXG": league_xg})
                # Results vs xG: positive = scoring/defending better than xG suggests.
                luck = (f.GF - f.GA) - (f.xGF - f.xGA)
                row["XgLuckPerGame"] = luck
                row["EloXgAdj"] = float(np.clip(-xg_luck_elo * luck, -xg_luck_cap, xg_luck_cap))
            else:
                f = football_data_form(t, info, season)
                if f:
                    row.update({"DomGames": f["Games"], "DomResults": f["Results"], "DomPPG": f["PPG"],
                                "DomGF": f["GF"], "DomGA": f["GA"]})
        except Exception as exc:  # noqa: BLE001 - form is optional context
            notes.append(f"Domestic form unavailable for {t}: {exc}")
        row.setdefault("EloXgAdj", 0.0)

    try:
        ucl = ucl_results(season)
        for t in teams:
            team_rows[t]["UCLForm"] = team_ucl_form(ucl, t)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"UCL results unavailable from API-Football: {exc}")

    try:
        cons = odds_api_consensus(fixtures, cfg)
        if cons.empty:
            notes.append("Odds API returned no UCL events; using MyBookie de-vigged prices as the market.")
    except Exception as exc:  # noqa: BLE001
        cons = pd.DataFrame()
        notes.append(f"Odds API consensus unavailable ({exc}); using MyBookie de-vigged prices as the market.")

    team_df = pd.DataFrame(team_rows.values())

    dc = cfg.get("decision", {})
    weights = {"h2h": float(dc.get("market_anchor_weight_moneyline", 0.85)),
               "spreads": float(dc.get("market_anchor_weight_spread", 0.40)),
               "totals": float(dc.get("market_anchor_weight_total", 0.35))}
    no_bet, eligible = float(dc.get("no_bet_edge", 0.03)), float(dc.get("bet_eligible_edge", 0.05))
    max_gap = float(dc.get("max_raw_model_market_gap", 0.18))
    pc = cfg.get("portfolio", {})
    kelly_frac, max_stake = float(pc.get("fractional_kelly", 0.25)), float(pc.get("max_stake_pct", 0.02))

    games, bets = [], []
    for _, fx in fixtures.iterrows():
        h, a = fx.HomeTeam, fx.AwayTeam
        th, ta = team_rows[h], team_rows[a]
        elo_h, elo_a = th["Elo"] + th["EloXgAdj"], ta["Elo"] + ta["EloXgAdj"]
        exp = elo_expectancy(elo_h, elo_a, hfa)
        fl = lines[(lines.HomeTeam == h) & (lines.AwayTeam == a)]

        # Market total: consensus at MyBookie's line, else MyBookie de-vig, else 2.8.
        tot = fl[fl.Market == "totals"]
        market_total, total_src = 2.8, "default"
        if not tot.empty:
            line = float(tot.Point.iloc[0])
            p_over, n = consensus_prob(cons, h, a, "totals", "Over", line)
            if n == 0 and set(tot.Outcome) >= {"Over", "Under"}:
                io_, iu_ = 1 / tot[tot.Outcome == "Over"].Decimal.iloc[0], 1 / tot[tot.Outcome == "Under"].Decimal.iloc[0]
                p_over, total_src = io_ / (io_ + iu_), "mybookie_devig"
            elif n == 0 and len(tot) == 1:
                # One side visible: strip a typical ~4.5% two-way margin.
                p_side = (1 / tot.Decimal.iloc[0]) / 1.045
                p_over = p_side if tot.Outcome.iloc[0] == "Over" else 1 - p_side
                total_src = "mybookie_one_side"
            elif n:
                total_src = f"consensus_{n}_books"
            if np.isfinite(p_over):
                market_total = total_from_over_prob(p_over, line, rho)
        # Tempo: both clubs' domestic xG per game vs their league, shrunk and capped.
        tempo = []
        for t in (th, ta):
            if pd.notna(t.get("DomxGF")) and pd.notna(t.get("LeagueXG")):
                tempo.append((t["DomxGF"] + t["DomxGA"]) / (2 * t["LeagueXG"]))
        tempo_factor = float(np.clip(1 + tempo_weight * (np.mean(tempo) - 1), 1 - tempo_cap, 1 + tempo_cap)) if tempo else 1.0
        model_total = market_total * tempo_factor

        lam_h, lam_a = lambdas_for(exp, model_total, rho)
        m = _score_matrix(lam_h, lam_a, rho, 10)
        mk = matrix_markets(m)
        best = np.unravel_index(np.argmax(m), m.shape)
        mb_h2h = {r.Outcome: 1 / r.Decimal for r in fl[fl.Market == "h2h"].itertuples()}
        mb_sum = sum(mb_h2h.values())
        cons_h, _ = consensus_prob(cons, h, a, "h2h", h, None)
        cons_d, _ = consensus_prob(cons, h, a, "h2h", "Draw", None)
        cons_a, nbooks = consensus_prob(cons, h, a, "h2h", a, None)
        games.append({
            "Date": fx.Date, "KickoffPT": fx.KickoffPT, "HomeTeam": h, "AwayTeam": a,
            "EloHome": th["Elo"], "EloAway": ta["Elo"], "EloXgAdjHome": th["EloXgAdj"], "EloXgAdjAway": ta["EloXgAdj"],
            "EloExpectancyHome": exp, "MarketTotal": market_total, "TotalSource": total_src, "TempoFactor": tempo_factor,
            "Lambda_Home": lam_h, "Lambda_Away": lam_a, "P_Home": mk["home"], "P_Draw": mk["draw"], "P_Away": mk["away"],
            "MostLikelyScore": f"{best[0]}-{best[1]}",
            "MyBookieNoVig_Home": mb_h2h.get(h, np.nan) / mb_sum, "MyBookieNoVig_Draw": mb_h2h.get("Draw", np.nan) / mb_sum,
            "MyBookieNoVig_Away": mb_h2h.get(a, np.nan) / mb_sum,
            "Consensus_Home": cons_h, "Consensus_Draw": cons_d, "Consensus_Away": cons_a, "ConsensusBooks": nbooks,
        })

        for sel in fl.itertuples():
            point = float(sel.Point) if pd.notna(sel.Point) else 0.0
            p_win, p_push = selection_probability(m, sel.Market, sel.Outcome, point, h, a)
            prior, n = consensus_prob(cons, h, a, sel.Market, sel.Outcome, point if sel.Market != "h2h" else None)
            prior_src = f"consensus_{n}_books"
            if n == 0:
                # De-vig MyBookie's own market as the prior.
                if sel.Market == "h2h":
                    prior = (1 / sel.Decimal) / mb_sum
                else:
                    pair = fl[(fl.Market == sel.Market) & (fl.Outcome != sel.Outcome)]
                    if sel.Market == "spreads":
                        pair = pair[np.isclose(pair.Point.astype(float), -point)]
                    prior = (1 / sel.Decimal) / ((1 / sel.Decimal) + (1 / pair.Decimal.iloc[0])) if not pair.empty else 1 / sel.Decimal
                prior_src = "mybookie_devig"
            nonpush = max(1e-9, 1 - p_push)
            raw_eff = p_win / nonpush
            w = weights[sel.Market]
            dec_eff = (1 - w) * raw_eff + w * prior
            dec_win = dec_eff * nonpush
            ev = dec_win * (sel.Decimal - 1) - (1 - dec_win - p_push)
            edge = dec_eff - prior
            gap = abs(raw_eff - prior)
            if edge < no_bet:
                status = "NO_BET"
            elif edge < eligible:
                status = "REVIEW"
            elif gap > max_gap:
                status = "REVIEW_MODEL_MARKET_DISAGREEMENT"
            else:
                status = "BET_ELIGIBLE"
            b = sel.Decimal - 1
            kelly = max(0.0, (b * dec_win - (1 - dec_win - p_push)) / b) if b > 0 else 0.0
            label = sel.Outcome if sel.Market == "h2h" else (
                f"{sel.Outcome} {point:+g}" if sel.Market == "spreads" else f"{sel.Outcome} {point:g}")
            bets.append({
                "Game": f"{h} v {a}", "KickoffPT": fx.KickoffPT, "Market": sel.Market, "Selection": label,
                "AmericanOdds": int(sel.AmericanOdds), "Decimal": round(sel.Decimal, 3),
                "RawModelProb": raw_eff, "MarketProb": prior, "MarketSource": prior_src,
                "DecisionProb": dec_eff, "Edge": edge, "EV": ev, "Status": status,
                "StakeUnits": round(100 * min(max_stake, kelly * kelly_frac), 2) if status == "BET_ELIGIBLE" else 0.0,
            })

    return pd.DataFrame(games), pd.DataFrame(bets), notes + [f"Teams: {len(team_df)}"], team_df


def main() -> None:
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games, bets, notes, teams = build(cfg)
    games.to_csv(OUT_DIR / "ucl_predictions_latest.csv", index=False)
    bets.sort_values(["Status", "EV"], ascending=[True, False]).to_csv(OUT_DIR / "ucl_bet_screen_latest.csv", index=False)
    teams.to_csv(OUT_DIR / "ucl_team_form_latest.csv", index=False)
    (OUT_DIR / "ucl_notes_latest.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(teams.round(2).to_string(index=False))
    print(games[["HomeTeam", "AwayTeam", "P_Home", "P_Draw", "P_Away", "Lambda_Home", "Lambda_Away", "MostLikelyScore"]].round(3).to_string(index=False))
    top = bets[bets.Status != "NO_BET"].sort_values("EV", ascending=False)
    print(top.round(3).to_string(index=False))
    print("\n".join(notes))


if __name__ == "__main__":
    main()
