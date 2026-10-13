"""League registry for running the V1.2 engine on Europe's other big-five leagues.

Every league is fitted on its own match history (teams never meet across these
leagues outside Europe), so the EPL configuration and outputs are untouched and
each league writes to its own folders under ``data/raw/leagues/<code>`` and
``data/processed/leagues/<code>``.

Canonical team names follow Football-Data.co.uk spellings, matching how the EPL
pipeline already canonicalises names. The alias maps below translate Understat
and The Odds API spellings to those canonical names. Anything a map misses is
reconciled automatically by fixture overlap (see ``reconcile_team_names``), and
leftovers are reported in each league's validation report.
"""
from __future__ import annotations

import copy
import difflib
import unicodedata
from pathlib import Path

import pandas as pd

LEAGUES: dict[str, dict] = {
    "EPL": {
        "name": "Premier League",
        "understat": "EPL",
        "football_data_division": "E0",
        "odds_api_sport": "soccer_epl",
    },
    "LALIGA": {
        "name": "La Liga",
        "understat": "La_Liga",
        "football_data_division": "SP1",
        "odds_api_sport": "soccer_spain_la_liga",
    },
    "BUNDESLIGA": {
        "name": "Bundesliga",
        "understat": "Bundesliga",
        "football_data_division": "D1",
        "odds_api_sport": "soccer_germany_bundesliga",
    },
    "SERIEA": {
        "name": "Serie A",
        "understat": "Serie_A",
        "football_data_division": "I1",
        "odds_api_sport": "soccer_italy_serie_a",
    },
    "LIGUE1": {
        "name": "Ligue 1",
        "understat": "Ligue_1",
        "football_data_division": "F1",
        "odds_api_sport": "soccer_france_ligue_one",
    },
}

# Understat / Odds API / common spellings -> Football-Data canonical names.
# Names are disjoint across leagues, so one combined map is safe.
TEAM_ALIASES: dict[str, str] = {
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "RCD Espanyol": "Espanol",
    "Real Sociedad": "Sociedad",
    "Rayo Vallecano": "Vallecano",
    "Real Valladolid": "Valladolid",
    "Cádiz CF": "Cadiz",
    "Cadiz CF": "Cadiz",
    "Alavés": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Leganés": "Leganes",
    "CD Leganes": "Leganes",
    "Almería": "Almeria",
    "UD Almeria": "Almeria",
    "Elche CF": "Elche",
    "Real Oviedo": "Oviedo",
    "CA Osasuna": "Osasuna",
    "Girona FC": "Girona",
    "Granada CF": "Granada",
    "RCD Mallorca": "Mallorca",
    "UD Las Palmas": "Las Palmas",
    "Levante UD": "Levante",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Getafe CF": "Getafe",
    "FC Barcelona": "Barcelona",
    # Bundesliga
    "Arminia Bielefeld": "Bielefeld",
    "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "1. FC Köln": "FC Koln",
    "1. FC Koln": "FC Koln",
    "Greuther Fuerth": "Greuther Furth",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Hertha Berlin": "Hertha",
    "Hertha BSC": "Hertha",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia M.Gladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "Mainz 05": "Mainz",
    "FSV Mainz 05": "Mainz",
    "1. FSV Mainz 05": "Mainz",
    "RasenBallsport Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum": "Bochum",
    "SV Werder Bremen": "Werder Bremen",
    "FC Heidenheim": "Heidenheim",
    "1. FC Heidenheim": "Heidenheim",
    "St. Pauli": "St Pauli",
    "FC St. Pauli": "St Pauli",
    "Hamburger SV": "Hamburg",
    "SC Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "FC Augsburg": "Augsburg",
    "SV Darmstadt 98": "Darmstadt",
    "FC Bayern Munich": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "FC Schalke 04": "Schalke 04",
    # Serie A
    "AC Milan": "Milan",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    "Hellas Verona": "Verona",
    "Hellas Verona FC": "Verona",
    "Parma Calcio 1913": "Parma",
    "SSC Napoli": "Napoli",
    "Juventus FC": "Juventus",
    "ACF Fiorentina": "Fiorentina",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Genoa CFC": "Genoa",
    "US Lecce": "Lecce",
    "Como 1907": "Como",
    "US Sassuolo": "Sassuolo",
    "US Cremonese": "Cremonese",
    "Pisa SC": "Pisa",
    "AC Monza": "Monza",
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG",
    "Saint-Etienne": "St Etienne",
    "Saint Etienne": "St Etienne",
    "AS Saint-Étienne": "St Etienne",
    "Clermont Foot": "Clermont",
    "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "LOSC Lille": "Lille",
    "RC Lens": "Lens",
    "Stade Rennais": "Rennes",
    "Stade Brestois 29": "Brest",
    "OGC Nice": "Nice",
    "FC Nantes": "Nantes",
    "RC Strasbourg": "Strasbourg",
    "Toulouse FC": "Toulouse",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Le Havre AC": "Le Havre",
    "FC Lorient": "Lorient",
    "FC Metz": "Metz",
    "Montpellier HSC": "Montpellier",
    "Stade de Reims": "Reims",
}

# Per-league output paths, relative to the league folders.
_LEAGUE_PATHS = {
    "master": "master_matches.csv",
    "predictions": "xg_predictions.csv",
    "predictions_v07": "xg_predictions_v07.csv",
    "predictions_v11": "xg_predictions_v11.csv",
    "predictions_v12": "xg_predictions_v12.csv",
    "summary": "backtest_summary.csv",
    "summary_v07": "backtest_summary_v07.csv",
    "summary_v11": "backtest_summary_v11.csv",
    "summary_v12": "backtest_summary_v12.csv",
    "comparison": "model_comparison.csv",
    "comparison_v11": "model_comparison_v11.csv",
    "comparison_v12": "model_comparison_v12.csv",
    "validation": "validation_report.json",
}


def strip_accents(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def alias_team(name: str) -> str:
    """Map a known alias (with or without accents) to its canonical name."""
    return TEAM_ALIASES.get(name) or TEAM_ALIASES.get(strip_accents(name)) or name


def enabled_leagues(cfg: dict) -> list[str]:
    """Non-EPL leagues switched on in ``config.yaml`` (``leagues.enabled``)."""
    codes = (cfg.get("leagues") or {}).get("enabled") or []
    out = []
    for code in codes:
        key = str(code).strip().upper()
        if key not in LEAGUES:
            raise ValueError(f"Unknown league '{code}'. Known: {sorted(LEAGUES)}")
        if key != "EPL":
            out.append(key)
    return out


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def league_config(cfg: dict, code: str) -> dict:
    """Return a copy of ``cfg`` pointed at one league's sources and output folders.

    ``leagues.defaults`` applies to every non-EPL league and
    ``leagues.overrides.<CODE>`` to one league (e.g.
    ``model_v12.home_advantage_shrink``); neither touches the EPL.
    """
    key = code.upper()
    meta = LEAGUES[key]
    out = copy.deepcopy(cfg)
    if key == "EPL":
        return out

    out["league"] = meta["understat"]
    out["league_code"] = key
    out["league_name"] = meta["name"]
    out["football_data_division"] = meta["football_data_division"]
    out.setdefault("market_comparison", {})["odds_api_sport"] = meta["odds_api_sport"]
    # Manual MyBookie screenshots are EPL-only.
    out["market_comparison"]["use_manual_mybookie"] = False
    out["market_comparison"]["mybookie_direct_only_if_missing"] = True
    # The V2 shadow model (rest, congestion, preseason context) is EPL-only, so
    # the V1/V2 agreement gate cannot run. Extreme edges still need V2, so they
    # stay research-only for these leagues.
    out.setdefault("portfolio", {})["require_v2_agreement"] = False

    league_cfg = cfg.get("leagues") or {}
    daily = out.setdefault("daily", {})
    daily["next_gameweek_only"] = False
    daily["horizon_days"] = int(league_cfg.get("horizon_days", 7))

    raw_root = Path(cfg["paths"]["raw_dir"]) / "leagues" / key
    proc_root = Path(cfg["paths"]["processed_dir"]) / "leagues" / key
    paths = out["paths"]
    paths["raw_dir"] = str(raw_root)
    paths["processed_dir"] = str(proc_root)
    for name, filename in _LEAGUE_PATHS.items():
        paths[name] = str(proc_root / filename)

    _deep_update(out, copy.deepcopy(league_cfg.get("defaults") or {}))
    _deep_update(out, copy.deepcopy((league_cfg.get("overrides") or {}).get(key, {})))
    return out


def reconcile_team_names(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    date_tolerance_days: int = 1,
    allow_fuzzy: bool = True,
) -> dict[str, str]:
    """Map team names in ``other`` that don't appear in ``reference`` onto reference names.

    Both frames need Date, HomeTeam and AwayTeam. Matching uses fixture overlap:
    two names are the same club when they play the same opponent on (about) the
    same date. A name-similarity tiebreak resolves the rare ambiguous case, and
    with ``allow_fuzzy`` a name with no shared fixture falls back to similarity
    alone (off for small reference sets, where it could pair e.g. Paris FC with
    Paris SG).
    Returns ``{other_name: reference_name}`` for the names it could resolve.
    """
    if reference.empty or other.empty:
        return {}

    ref_teams = set(reference["HomeTeam"]).union(reference["AwayTeam"])
    oth_teams = set(other["HomeTeam"]).union(other["AwayTeam"])
    unknown = sorted(t for t in oth_teams - ref_teams if isinstance(t, str) and t)
    if not unknown:
        return {}
    candidates = sorted(t for t in ref_teams - oth_teams if isinstance(t, str) and t)
    if not candidates:
        candidates = sorted(t for t in ref_teams if isinstance(t, str) and t)

    ref = reference[["Date", "HomeTeam", "AwayTeam"]].copy()
    ref["Date"] = pd.to_datetime(ref["Date"]).dt.normalize()
    oth = other[["Date", "HomeTeam", "AwayTeam"]].copy()
    oth["Date"] = pd.to_datetime(oth["Date"]).dt.normalize()

    # (date, opponent, venue) for every reference appearance, keyed by team.
    ref_index: dict[str, set] = {}
    for d, h, a in ref.itertuples(index=False):
        for shift in range(-date_tolerance_days, date_tolerance_days + 1):
            day = d + pd.Timedelta(days=shift)
            ref_index.setdefault(h, set()).add((day, a, "H"))
            ref_index.setdefault(a, set()).add((day, h, "A"))

    mapping: dict[str, str] = {}
    strength: dict[str, tuple] = {}
    for name in unknown:
        appearances = []
        for d, h, a in oth.itertuples(index=False):
            if h == name:
                appearances.append((d, a, "H"))
            elif a == name:
                appearances.append((d, h, "A"))
        if not appearances:
            continue
        scores = {}
        for cand in candidates:
            idx = ref_index.get(cand, set())
            overlap = sum(1 for app in appearances if app in idx)
            if overlap:
                sim = difflib.SequenceMatcher(None, strip_accents(name).lower(), cand.lower()).ratio()
                scores[cand] = (overlap, sim)
        if scores:
            best = max(scores, key=lambda c: scores[c])
            mapping[name] = best
            strength[name] = scores[best]
        elif allow_fuzzy:
            # No shared fixtures (e.g. odds for a fixture we don't predict): name similarity only.
            close = difflib.get_close_matches(strip_accents(name), candidates, n=1, cutoff=0.6)
            if close:
                mapping[name] = close[0]
                strength[name] = (0, 0.0)

    # Never send two different names to the same club: keep the best-supported one.
    by_target: dict[str, str] = {}
    for name, target in mapping.items():
        keep = by_target.get(target)
        if keep is None or strength[name] > strength[keep]:
            by_target[target] = name
    return {name: target for target, name in by_target.items()}


def harmonize_team_names(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    max_rounds: int = 3,
    allow_fuzzy: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename teams in ``other`` to ``reference`` spellings, repeating so that
    fixtures between two unknown names resolve once one side is learned."""
    out = other.copy()
    learned: dict[str, str] = {}
    for _ in range(max_rounds):
        mapping = reconcile_team_names(reference, out, allow_fuzzy=allow_fuzzy)
        mapping = {k: v for k, v in mapping.items() if k not in learned}
        if not mapping:
            break
        learned.update(mapping)
        out["HomeTeam"] = out["HomeTeam"].replace(mapping)
        out["AwayTeam"] = out["AwayTeam"].replace(mapping)
    return out, learned
