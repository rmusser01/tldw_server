import json
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Watchlists import pipeline as wl_pipeline


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    # Isolate per-user DBs for this test file
    base_dir = tmp_path / "test_user_dbs_backoff"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    # Force real pipeline fetch path even if other suites enabled TEST_MODE
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    # Lower the threshold to speed up the test and keep defaults small
    monkeypatch.setenv("WATCHLISTS_304_BACKOFF_THRESHOLD", "2")
    monkeypatch.setenv("WATCHLISTS_304_BACKOFF_BASE_SEC", "60")
    monkeypatch.setenv("WATCHLISTS_304_BACKOFF_MAX_SEC", "300")
    monkeypatch.setenv("WATCHLISTS_304_BACKOFF_JITTER_PCT", "0.0")
    yield
    for db_path in base_dir.glob("*/Media_DB_v2.db"):
        reset_managed_sqlite_backends(sqlite_targets=[str(db_path), str(db_path.resolve())])


@pytest.mark.asyncio
async def test_backoff_defer_until_after_consecutive_304s(monkeypatch):
    user_id = 840
    # Ensure a clean slate DB for this user
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    p = DatabasePaths.get_media_db_path(user_id)
    reset_managed_sqlite_backends(sqlite_targets=[str(p), str(p.resolve())])
    try:
        if p.exists():
            p.unlink()
    except Exception:
        _ = None

    db = WatchlistsDatabase.for_user(user_id)

    src = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 5}),
        tags=["news"],
        group_ids=[],
    )

    # Stub RSS fetcher to always return 304 Not Modified
    calls = {"n": 0}

    async def _stub_fetch(url, **kwargs):
        calls["n"] += 1
        return {"status": 304}

    monkeypatch.setattr(wl_pipeline, "fetch_rss_feed", _stub_fetch)
    # Pipeline may use history-aware fetcher by default; stub it too
    monkeypatch.setattr(wl_pipeline, "fetch_rss_feed_history", _stub_fetch)

    job = db.create_job(
        name="Job",
        description=None,
        scope_json=json.dumps({"sources": [src.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    # First run -> 304 (no backoff yet because threshold=2)
    await wl_pipeline.run_watchlist_job(user_id, job.id)
    assert calls["n"] == 1
    s1 = db.get_source(src.id)
    assert s1.defer_until is None
    assert s1.consec_not_modified == 1
    assert (s1.status or "").startswith("not_modified")

    # Second run -> 304 again; threshold reached -> defer_until should be set
    await wl_pipeline.run_watchlist_job(user_id, job.id)
    assert calls["n"] == 2
    s2 = db.get_source(src.id)
    assert s2.consec_not_modified == 2
    assert s2.defer_until is not None
    assert (s2.status or "").startswith("not_modified_backoff:")
    # Defer timestamp should be in the future
    try:
        defer_dt = datetime.fromisoformat(s2.defer_until)
        assert defer_dt.tzinfo is not None
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        assert defer_dt > now_utc
    except Exception:
        pytest.fail("defer_until is not a valid ISO datetime")

    # Immediate third run should skip calling fetch due to deferral
    await wl_pipeline.run_watchlist_job(user_id, job.id)
    assert calls["n"] == 2  # unchanged
