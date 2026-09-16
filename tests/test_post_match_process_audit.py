import post_match_process_audit as audit


def test_fetch_date_safe_skips_free_plan_date_error(monkeypatch):
    def fake_fetch_date(date):
        raise RuntimeError(
            "API-Football error: {'plan': 'Free plans do not have access to this date, try from 2026-09-15 to 2026-09-17.'}"
        )

    monkeypatch.setattr(audit, "fetch_date", fake_fetch_date)

    rows, skipped = audit.fetch_date_safe("2026-09-14")

    assert rows == []
    assert skipped is not None
    assert skipped["Date"] == "2026-09-14"
    assert skipped["AuditStatus"] == "AUDIT_SKIPPED"
    assert skipped["Reason"] == "DATA_UNAVAILABLE"
    assert "Free plans do not have access to this date" in skipped["APIError"]


def test_fetch_date_safe_reraises_other_runtime_errors(monkeypatch):
    def fake_fetch_date(date):
        raise RuntimeError("API_FOOTBALL_KEY is not configured")

    monkeypatch.setattr(audit, "fetch_date", fake_fetch_date)

    try:
        audit.fetch_date_safe("2026-09-16")
    except RuntimeError as exc:
        assert "API_FOOTBALL_KEY is not configured" in str(exc)
    else:
        raise AssertionError("Expected non-plan RuntimeError to be re-raised")
