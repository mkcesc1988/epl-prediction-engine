import pandas as pd

import api_football_odds_fallback as fallback
import safe_market_comparison as smc


def test_odds_fallback_plan_error_does_not_crash(monkeypatch, tmp_path):
    preds = tmp_path / "preds.csv"
    pd.DataFrame([{"Date": "2026-10-24", "HomeTeam": "Arsenal", "AwayTeam": "Everton"}]).to_csv(preds, index=False)
    monkeypatch.setattr(fallback, "PRED_PATH", preds)
    monkeypatch.setattr(fallback, "ODDS_PATH", tmp_path / "odds.csv")

    def boom(_):
        raise RuntimeError("API-Football odds error: {'plan': 'Free plans do not have access to this season'}")

    monkeypatch.setattr(fallback, "fetch_api_football_odds", boom)
    fallback.main()  # must not raise


def _preds(kickoffs):
    return pd.DataFrame({"Date": [k[:10] for k in kickoffs], "KickoffUTC": kickoffs})


def test_live_odds_due_window():
    cfg = {"market_comparison": {"odds_within_days": 2}}
    now = pd.Timestamp("2026-10-20T17:00:00Z")
    assert not smc.live_odds_due(_preds(["2026-10-23T19:00:00+00:00"]), cfg, now)
    assert smc.live_odds_due(_preds(["2026-10-22T12:00:00+00:00"]), cfg, now)
    # already-kicked-off fixtures do not count
    assert not smc.live_odds_due(_preds(["2026-10-20T12:00:00+00:00"]), cfg, now)
    assert not smc.live_odds_due(pd.DataFrame(), cfg, now)
