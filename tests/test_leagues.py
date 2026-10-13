import pandas as pd

from leagues import (
    LEAGUES,
    alias_team,
    enabled_leagues,
    harmonize_team_names,
    league_config,
    reconcile_team_names,
)
from pipeline import load_config, normalize_team


def _fixtures(rows):
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam"]).assign(Date=lambda d: pd.to_datetime(d["Date"]))


def test_aliases_map_to_football_data_names():
    assert alias_team("Atletico Madrid") == "Ath Madrid"
    assert alias_team("Atlético Madrid") == "Ath Madrid"
    assert alias_team("RasenBallsport Leipzig") == "RB Leipzig"
    assert alias_team("Paris Saint Germain") == "Paris SG"
    assert alias_team("Inter Milan") == "Inter"
    assert normalize_team("AC Milan") == "Milan"


def test_epl_names_unchanged():
    assert normalize_team("Manchester United") == "Man United"
    assert normalize_team("Arsenal") == "Arsenal"
    assert normalize_team("Nottingham Forest") == "Nott'm Forest"


def test_league_config_isolates_outputs_and_keeps_epl_untouched():
    cfg = load_config()
    lcfg = league_config(cfg, "SERIEA")
    assert lcfg["league"] == "Serie_A"
    assert lcfg["football_data_division"] == "I1"
    assert lcfg["market_comparison"]["odds_api_sport"] == "soccer_italy_serie_a"
    assert lcfg["market_comparison"]["use_manual_mybookie"] is False
    assert "leagues" in lcfg["paths"]["master"] and "SERIEA" in lcfg["paths"]["master"]
    assert lcfg["daily"]["next_gameweek_only"] is False
    # Original config is not mutated.
    assert cfg["league"] == "EPL"
    assert cfg["paths"]["master"] == "data/processed/master_matches.csv"
    assert league_config(cfg, "EPL") == cfg


def test_league_overrides_apply_to_one_league_only():
    cfg = load_config()
    cfg["leagues"]["overrides"] = {"BUNDESLIGA": {"model_v12": {"home_advantage_shrink": 0.3}}}
    assert league_config(cfg, "BUNDESLIGA")["model_v12"]["home_advantage_shrink"] == 0.3
    assert league_config(cfg, "LIGUE1")["model_v12"]["home_advantage_shrink"] == cfg["model_v12"]["home_advantage_shrink"]


def test_enabled_leagues_from_config():
    cfg = load_config()
    assert enabled_leagues(cfg) == ["LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1"]
    assert all(code in LEAGUES for code in enabled_leagues(cfg))


def test_reconcile_by_fixture_overlap():
    ref = _fixtures([
        ("2025-08-16", "Club X", "Barcelona"),
        ("2025-08-23", "Getafe", "Club X"),
        ("2025-08-30", "Club X", "Sevilla"),
        ("2025-08-30", "Barcelona", "Getafe"),
    ])
    other = _fixtures([
        ("2025-08-16", "Real Club X CF", "Barcelona"),
        ("2025-08-24", "Getafe", "Real Club X CF"),  # one day off: still matches
        ("2025-08-30", "Real Club X CF", "Sevilla"),
        ("2025-08-30", "Barcelona", "Getafe"),
    ])
    assert reconcile_team_names(ref, other) == {"Real Club X CF": "Club X"}


def test_harmonize_resolves_fixture_between_two_unknown_names():
    ref = _fixtures([
        ("2025-08-16", "Alpha", "Beta"),
        ("2025-08-23", "Alpha", "Gamma"),
        ("2025-08-30", "Delta", "Beta"),
    ])
    other = _fixtures([
        ("2025-08-16", "Alpha FC", "Beta SC"),
        ("2025-08-23", "Alpha FC", "Gamma"),
        ("2025-08-30", "Delta", "Beta SC"),
    ])
    fixed, learned = harmonize_team_names(ref, other)
    assert learned == {"Alpha FC": "Alpha", "Beta SC": "Beta"}
    assert set(fixed["HomeTeam"]).union(fixed["AwayTeam"]) == {"Alpha", "Beta", "Gamma", "Delta"}


def test_reconcile_never_merges_two_names_into_one_club():
    ref = _fixtures([("2025-08-16", "Club X", "Barcelona")])
    other = _fixtures([
        ("2025-08-16", "Club X One", "Barcelona"),
        ("2025-09-16", "Club X Two", "Somebody"),
    ])
    mapping = reconcile_team_names(ref, other)
    assert list(mapping.values()).count("Club X") <= 1
    assert mapping.get("Club X One") == "Club X"


def test_parse_understat_fixtures_splits_played_and_upcoming():
    from daily_predictions import parse_understat_fixtures

    matches = [
        {"id": "1", "isResult": True, "h": {"title": "Atletico Madrid"}, "a": {"title": "Real Betis"},
         "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.9", "a": "0.7"}, "datetime": "2026-10-04 19:00:00"},
        {"id": "2", "isResult": False, "h": {"title": "Athletic Club"}, "a": {"title": "Barcelona"},
         "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}, "datetime": "2026-10-18 21:00:00"},
    ]
    df = parse_understat_fixtures(matches, 2026)
    assert list(df["HomeTeam"]) == ["Ath Madrid", "Ath Bilbao"]
    assert list(df["Finished"]) == [True, False]
    assert df.loc[1, "Date"] == pd.Timestamp("2026-10-18")
    assert df["Season"].iloc[0] == "2026/27"


def test_moneyline_comparison_uses_model_and_no_vig_consensus():
    from league_predictions import align_odds_to_predictions, compare_moneyline

    preds = pd.DataFrame([{
        "League": "LALIGA", "Date": "2026-10-18", "HomeTeam": "Ath Bilbao", "AwayTeam": "Barcelona",
        "P_HomeWin": 0.30, "P_Draw": 0.25, "P_AwayWin": 0.45,
    }])
    rows = []
    for book, (h, d, a) in {"b1": (3.2, 3.8, 2.1), "b2": (3.4, 3.6, 2.05)}.items():
        for outcome, price in [("Athletic Bilbao", h), ("Draw", d), ("Barcelona", a)]:
            rows.append({
                "Date": "2026-10-18", "HomeTeam": "Athletic Bilbao", "AwayTeam": "Barcelona",
                "Bookmaker": book, "BookmakerKey": book, "Market": "h2h",
                "Outcome": outcome, "Point": None, "DecimalOdds": price,
            })
    odds = align_odds_to_predictions(preds, pd.DataFrame(rows))
    assert set(odds["HomeTeam"]) == {"Ath Bilbao"}
    ml = compare_moneyline(preds, odds).set_index("Side")
    assert set(ml.index) == {"Home", "Draw", "Away"}
    assert ml.loc["Home", "BestMarketOdds"] == 3.4
    assert abs(ml.loc["Home", "BestMarketExpectedReturn"] - (0.30 * 3.4 - 1)) < 1e-12
    assert abs(ml["ConsensusNoVigProbability"].sum() - 1.0) < 1e-9
    assert (ml["ConsensusBooks"] == 2).all()


def test_odds_dates_snap_to_prediction_date():
    from league_predictions import align_odds_to_predictions

    preds = pd.DataFrame([{"Date": "2026-10-18", "HomeTeam": "Paris SG", "AwayTeam": "Lyon"}])
    odds = pd.DataFrame([{
        "Date": "2026-10-19", "HomeTeam": "Paris SG", "AwayTeam": "Lyon", "Bookmaker": "b",
        "BookmakerKey": "b", "Market": "totals", "Outcome": "Over", "Point": 2.5, "DecimalOdds": 1.7,
    }])
    assert align_odds_to_predictions(preds, odds)["Date"].tolist() == ["2026-10-18"]
