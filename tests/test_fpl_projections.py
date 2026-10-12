import pandas as pd

import fpl_projections as fp


def _league():
    teams = [{"id": i, "name": n, "short_name": n[:3].upper()} for i, n in
             [(1, "Arsenal"), (2, "Spurs"), (3, "Chelsea"), (4, "Leeds"), (5, "Brighton"), (6, "Fulham")]]
    elements = []
    pid = 1
    for t in teams:
        tid = t["id"]
        strength = {1: 1.8, 2: 1.2, 3: 1.5, 4: 0.9, 5: 1.3, 6: 1.1}[tid]
        shape = [(1, 0.0, 0.0, 540, 6)] + [(2, 0.1, 0.1, 540, 6)] * 5 + \
                [(3, 0.4, 0.3, 540, 6)] * 5 + [(4, 0.8, 0.2, 540, 6)] * 3
        shape += [(1, 0.0, 0.0, 0, 0)]  # backup keeper
        for pos, xg, xa, mins, starts in shape:
            elements.append({
                "id": pid, "web_name": f"P{pid}", "team": tid, "element_type": pos,
                "now_cost": 50 + 10 * pos, "status": "a", "chance_of_playing_next_round": None,
                "expected_goals": xg * strength * 6 / 1.5, "expected_assists": xa * 6,
                "minutes": mins, "starts": starts, "goals_scored": 0, "assists": 0,
                "clean_sheets": 1 if pos <= 2 else 0, "goals_conceded": 6, "total_points": 20,
                "selected_by_percent": "5.0",
            })
            pid += 1
    fixtures = []
    fid = 1
    for gw in range(1, 7):
        for h, a in [(1, 2), (3, 4), (5, 6)] if gw % 2 else [(2, 1), (4, 3), (6, 5)]:
            fixtures.append({"id": fid, "event": gw, "team_h": h, "team_a": a, "finished": True})
            fid += 1
    for gw in (7, 8):
        fixtures += [{"id": fid, "event": gw, "team_h": 1, "team_a": 4, "finished": False},
                     {"id": fid + 1, "event": gw, "team_h": 3, "team_a": 2, "finished": False},
                     {"id": fid + 2, "event": gw, "team_h": 5, "team_a": 6, "finished": False}]
        fid += 3
    return {"teams": teams, "elements": elements}, fixtures


def _lam(home, away):
    attack = {"Arsenal": 1.9, "Tottenham": 1.3, "Chelsea": 1.6, "Leeds": 0.9, "Brighton": 1.4, "Fulham": 1.2}
    return attack[home], attack[away] * 0.8


def _projection():
    boot, fixtures = _league()
    fx = fp.fixture_frame(boot, fixtures)
    tfx = fp.team_fixture_projections(fx[fx["GW"].isin([7, 8])], _lam, -0.05)
    players = fp.player_rates(boot, fp.team_games_played(fx))
    return boot, fx, tfx, players, fp.project_players(players, tfx, [7, 8])


def test_fixture_names_use_model_naming():
    boot, fixtures = _league()
    fx = fp.fixture_frame(boot, fixtures)
    assert "Tottenham" in set(fx["HomeTeam"]) | set(fx["AwayTeam"])
    assert "Spurs" not in set(fx["HomeTeam"]) | set(fx["AwayTeam"])


def test_clean_sheet_probabilities_are_sane():
    _, _, tfx, _, _ = _projection()
    assert tfx["P_CS"].between(0, 1).all()
    ars = tfx[(tfx["TeamId"] == 1) & (tfx["GW"] == 7)].iloc[0]
    lee = tfx[(tfx["TeamId"] == 4) & (tfx["GW"] == 7)].iloc[0]
    assert ars["P_CS"] > lee["P_CS"]


def test_unused_backup_keeper_projects_zero():
    _, _, _, players, proj = _projection()
    backups = players[(players["minutes"] == 0)]["id"]
    assert proj[proj["id"].isin(backups)]["xPts_Total"].abs().max() < 1e-9


def test_injured_player_scores_zero_next_gw():
    boot, fixtures = _league()
    boot["elements"][5]["status"] = "i"
    boot["elements"][5]["chance_of_playing_next_round"] = 0
    fx = fp.fixture_frame(boot, fixtures)
    tfx = fp.team_fixture_projections(fx[fx["GW"].isin([7, 8])], _lam, -0.05)
    proj = fp.project_players(fp.player_rates(boot, fp.team_games_played(fx)), tfx, [7, 8])
    row = proj[proj["id"] == boot["elements"][5]["id"]].iloc[0]
    assert row["xPts_GW7"] == 0
    assert row["xPts_GW8"] > 0


def test_best_xi_is_legal_and_captain_is_top_scorer():
    _, _, _, _, proj = _projection()
    squad_ids = list(proj[proj["team"] == 1]["id"])  # 15-man club squad
    squad = proj[proj["id"].isin(squad_ids)]
    xi, cap = fp.best_xi(squad, "xPts_GW7")
    picked = squad[squad["id"].isin(xi)]
    assert len(xi) == 11
    counts = picked["element_type"].value_counts()
    assert counts.get(1, 0) == 1 and counts.get(2, 0) >= 3 and counts.get(4, 0) >= 1
    assert picked.set_index("id").loc[cap, "xPts_GW7"] == picked["xPts_GW7"].max()


def test_free_transfer_estimate_respects_chips():
    history = {"current": [{"event": 2, "event_transfers": 0}, {"event": 3, "event_transfers": 9},
                           {"event": 4, "event_transfers": 2}, {"event": 5, "event_transfers": 15},
                           {"event": 6, "event_transfers": 1}],
               "chips": [{"name": "freehit", "event": 3}, {"name": "wildcard", "event": 5}]}
    assert fp.estimate_free_transfers(history, 7) == 3


def test_transfer_options_respect_budget_and_club_limit():
    _, _, _, _, proj = _projection()
    # Legal squad: 2 GK, 5 DEF, 5 MID, 3 FWD, at most 3 per club (round robin).
    squad_ids, per_club = [], {}
    for pos, n in fp.SQUAD_SHAPE.items():
        cands = proj[proj["element_type"] == pos].sort_values(["team", "xPts_Total"])
        picked = 0
        rounds = [g.reset_index(drop=True) for _, g in cands.groupby("team")]
        i = 0
        while picked < n and i < 10:
            for g in rounds:
                if picked >= n:
                    break
                if i < len(g) and per_club.get(int(g.loc[i, "team"]), 0) < 3:
                    squad_ids.append(int(g.loc[i, "id"]))
                    per_club[int(g.loc[i, "team"])] = per_club.get(int(g.loc[i, "team"]), 0) + 1
                    picked += 1
            i += 1
    assert len(squad_ids) == 15
    opts = fp.transfer_options(proj, squad_ids, bank=0.0, gws=[7, 8], decay=0.9, free_transfers=1, top_per_slot=4)
    assert not opts.empty
    assert (opts["BankAfter"] >= -1e-9).all()
    assert (opts.loc[opts["Type"] == "double", "HitCost"] == 4).all()
    best = opts.iloc[0]
    assert best["NetGain"] >= opts["NetGain"].iloc[-1]
