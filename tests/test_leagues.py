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


def _portfolio_row(**kw):
    row = {
        "League": "SERIEA", "Date": "2026-10-18", "KickoffUTC": "2026-10-18T18:45:00+00:00",
        "HomeTeam": "Milan", "AwayTeam": "Atalanta", "MarketType": "Moneyline", "Selection": "Milan",
        "Line": None, "Bookmaker": "MyBookie", "MyBookieOdds": 2.3, "ModelWinProbability": 0.48,
        "MyBookieImpliedProbability": 1 / 2.3, "ProbabilityEdge": 0.05, "ExpectedReturnPerUnit": 0.104,
        "BetQualityScore": 65.0, "PaperStakeUnits": 1.5, "PaperStakeAmount": 1.5, "ModelVersion": "V1.2",
    }
    row.update(kw)
    return row


def test_league_ledger_records_once_and_settles():
    from league_portfolio import performance_summary, record_bets, settle_bets

    portfolio = pd.DataFrame([_portfolio_row()])
    ledger, n = record_bets(pd.DataFrame(), portfolio, "SERIEA")
    assert n == 1 and ledger.loc[0, "League"] == "SERIEA" and ledger.loc[0, "Result"] == "OPEN"
    ledger, n = record_bets(ledger, portfolio, "SERIEA")
    assert n == 0 and len(ledger) == 1

    results = pd.DataFrame([{"Date": "2026-10-18", "HomeTeam": "Milan", "AwayTeam": "Atalanta", "FTHG": 2, "FTAG": 1}])
    # Another league's results never settle this bet.
    _, other = settle_bets(ledger.copy(), results, "LIGUE1")
    assert other == 0
    ledger, n = settle_bets(ledger, results, "SERIEA")
    assert n == 1
    assert ledger.loc[0, "Result"] == "W" and ledger.loc[0, "FinalScore"] == "2-1"
    assert abs(ledger.loc[0, "ProfitUnits"] - 1.5 * 1.3) < 1e-9
    summary = performance_summary(ledger)
    assert set(summary["League"]) == {"ALL", "SERIEA"}


def test_settlement_ignores_same_pairing_on_other_dates():
    from league_portfolio import record_bets, settle_bets

    ledger, _ = record_bets(pd.DataFrame(), pd.DataFrame([_portfolio_row()]), "SERIEA")
    old_meeting = pd.DataFrame([{"Date": "2026-02-01", "HomeTeam": "Milan", "AwayTeam": "Atalanta", "FTHG": 0, "FTAG": 3}])
    ledger, n = settle_bets(ledger, old_meeting, "SERIEA")
    assert n == 0 and ledger.loc[0, "Result"] == "OPEN"


def test_open_bets_count_against_league_exposure_cap(monkeypatch):
    import league_portfolio as lp

    cfg = league_config(load_config(), "SERIEA")
    ranked = pd.DataFrame([
        _portfolio_row(HomeTeam="Milan", AwayTeam="Atalanta", DecisionStatus="BET_ELIGIBLE", OverallRankScore=70.0),
        _portfolio_row(HomeTeam="Roma", AwayTeam="Genoa", Selection="Roma", DecisionStatus="BET_ELIGIBLE", OverallRankScore=69.0),
    ])
    ranked["RawModelWinProbability"] = ranked["ModelWinProbability"]
    ranked["PushProbability"] = 0.0
    monkeypatch.setattr(lp, "build_rankings", lambda p, o, c: ranked.drop(columns=["League"]).copy())

    captured = {}
    def fake_build_portfolio(rows, sized_cfg):
        captured["rows"] = rows
        captured["cap"] = sized_cfg["portfolio"]["max_gameweek_exposure_pct"]
        return pd.DataFrame()
    monkeypatch.setattr(lp, "build_portfolio", fake_build_portfolio)

    ledger, _ = lp.record_bets(pd.DataFrame(), pd.DataFrame([_portfolio_row(PaperStakeAmount=1.5)]), "SERIEA")
    lp.build_league_portfolio("SERIEA", pd.DataFrame(), pd.DataFrame(), cfg, ledger)
    # The open Milan bet holds 1.5 of the 4.0 cap and its fixture is skipped.
    assert abs(captured["cap"] - (0.04 - 0.015)) < 1e-12
    assert list(captured["rows"]["HomeTeam"]) == ["Roma"]


def test_league_config_disables_v2_gate_and_sets_league_cap():
    cfg = load_config()
    lcfg = league_config(cfg, "LALIGA")
    assert lcfg["portfolio"]["require_v2_agreement"] is False
    assert lcfg["portfolio"]["max_gameweek_exposure_pct"] == 0.04
    assert lcfg["market_comparison"]["mybookie_direct_only_if_missing"] is True
    assert cfg["portfolio"]["require_v2_agreement"] is True  # EPL untouched


def test_odds_only_pulled_near_kickoff():
    from league_predictions import _odds_due

    cfg = load_config()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    soon = pd.DataFrame({"Date": [(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]})
    later = pd.DataFrame({"Date": [(today + pd.Timedelta(days=5)).strftime("%Y-%m-%d")]})
    assert _odds_due(soon, cfg) is True
    assert _odds_due(later, cfg) is False


def test_matchday_from_completed_fixtures():
    from daily_predictions import league_matchday

    all_fx = pd.DataFrame({
        "HomeTeam": ["A", "C", "A", "B"], "AwayTeam": ["B", "D", "C", "D"], "Finished": [True, True, True, False],
    })
    upcoming = all_fx[~all_fx["Finished"]]
    assert league_matchday(all_fx, upcoming).tolist() == [2]  # B and D each played once


def test_goal_scale_off_by_default_and_tracks_goals_per_xg():
    from model_v12 import goal_conversion_scale

    cfg = load_config()
    days = pd.date_range("2025-08-01", periods=300, freq="D")
    hist = pd.DataFrame({"Date": days, "FTHG": 1, "FTAG": 1, "Home_xG": 1.25, "Away_xG": 1.25})
    ref = pd.Timestamp("2026-06-01")
    assert goal_conversion_scale(hist, ref, cfg) == 1.0  # EPL default: off
    lcfg = league_config(cfg, "LIGUE1")
    assert abs(goal_conversion_scale(hist, ref, lcfg) - 0.8) < 1e-9  # 2 goals / 2.5 xG
    # Only matches before the reference date count (no leakage).
    assert goal_conversion_scale(hist, days[100], lcfg) == 1.0  # < 200 prior matches


def test_goal_scale_changes_totals_but_not_1x2():
    from model_v11 import _derived_markets, _score_matrix
    from model_v12 import totals_in_goal_units

    base = _derived_markets(_score_matrix(1.6, 1.2, -0.08, 10))
    scaled = totals_in_goal_units(base, 1.6, 1.2, -0.08, 0.9, 10)
    assert scaled["P_HomeWin_DC"] == base["P_HomeWin_DC"]
    assert scaled["P_Draw_DC"] == base["P_Draw_DC"]
    assert scaled["P_Over_2_5_DC"] < base["P_Over_2_5_DC"]
    assert scaled["P_BTTS_Yes_DC"] < base["P_BTTS_Yes_DC"]
    assert totals_in_goal_units(base, 1.6, 1.2, -0.08, 1.0, 10) is base


def test_blocked_markets_stay_ranked_but_out_of_portfolio(monkeypatch):
    import league_portfolio as lp

    cfg = league_config(load_config(), "LALIGA")
    ranked = pd.DataFrame([
        _portfolio_row(League=None, HomeTeam="Betis", AwayTeam="Celta", MarketType="Total", Selection="Over 2.5",
                       DecisionStatus="BET_ELIGIBLE", OverallRankScore=70.0),
        _portfolio_row(League=None, HomeTeam="Betis", AwayTeam="Celta", MarketType="Spread", Selection="Betis -0.5",
                       DecisionStatus="BET_ELIGIBLE", OverallRankScore=60.0),
    ]).drop(columns=["League"])
    ranked["RawModelWinProbability"] = ranked["ModelWinProbability"]
    ranked["PushProbability"] = 0.0
    monkeypatch.setattr(lp, "build_rankings", lambda p, o, c: ranked.copy())
    seen = {}

    def fake_build_portfolio(rows, sized_cfg):
        seen["rows"] = rows
        return pd.DataFrame()

    monkeypatch.setattr(lp, "build_portfolio", fake_build_portfolio)

    rankings, _, _ = lp.build_league_portfolio("LALIGA", pd.DataFrame(), pd.DataFrame(), cfg, pd.DataFrame())
    assert rankings.set_index("MarketType")["PortfolioBlocked"].to_dict() == {"Total": True, "Spread": False}
    assert list(seen["rows"]["MarketType"]) == ["Spread"]
