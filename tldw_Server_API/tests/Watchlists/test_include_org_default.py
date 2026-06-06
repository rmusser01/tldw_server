import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job
from tldw_Server_API.app.core.DB_Management.scope_context import set_scope, reset_scope
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _test_env(monkeypatch, tmp_path):
     # Offline behavior and isolated DBs
    monkeypatch.setenv("TEST_MODE", "1")
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_include_org_default"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    # Isolate AuthNZ DB as well
    asyncio.run(reset_db_pool())
    auth_db_path = tmp_path / "authnz_org_default.db"
    # Ensure a fresh DB per run to avoid leftover rows across test sessions
    try:
        auth_db_path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except TypeError:
        # Python <3.8: Path.unlink() has no missing_ok
        try:
            auth_db_path.unlink()
        except FileNotFoundError:
            # File already absent; acceptable for cleanup
            pass
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{auth_db_path}")
    asyncio.run(reset_db_pool())

    # Ensure AuthNZ tables exist
    ensure_authnz_tables(auth_db_path)
    # Remove any pre-existing test org slug to avoid UNIQUE violations across runs
    try:
        loop = asyncio.get_event_loop()
        async def _cleanup():
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool as _get
            p = await _get()
            try:
                await p.execute("DELETE FROM organizations WHERE slug = ?", ("orga",))
            except Exception:
                _ = None
        if loop.is_running():
            # In pytest-asyncio, we might already be in an event loop; best-effort
            loop.run_until_complete(_cleanup())
        else:
            loop.run_until_complete(_cleanup())
    except Exception:
        _ = None
    try:
        yield
    finally:
        asyncio.run(reset_db_pool())


@pytest.mark.asyncio
async def test_include_only_gating_enabled_by_org_default():
    # Create organization with metadata enabling require_include_default
    pool = await get_db_pool()
    metadata = {"watchlists": {"require_include_default": True}}
    await pool.execute(
        "INSERT INTO organizations (name, slug, metadata) VALUES (?, ?, ?)",
        ("OrgA", "orga", json.dumps(metadata)),
    )
    row = await pool.fetchone("SELECT id FROM organizations WHERE slug = ?", "orga")
    org_id = int(row["id"] if isinstance(row, dict) else row[0])

    # Establish scope with this organization so pipeline picks org default
    user_id = 904
    scope_token = set_scope(user_id=user_id, org_ids=[org_id], team_ids=[], is_admin=False)
    try:
        db = WatchlistsDatabase.for_user(user_id)

        # RSS source (test mode yields 1 item with title/summary containing 'Test')
        src = db.create_source(
            name="Feed",
            url="https://example.com/feed.xml",
            source_type="rss",
            active=True,
            settings_json=json.dumps({"limit": 1}),
            tags=["gated"],
            group_ids=[],
        )

        # Job with an include rule that does NOT match the test item → expect filtered, not ingested
        job = db.create_job(
            name="IncludeOnlyByOrg",
            description=None,
            scope_json=json.dumps({"sources": [src.id]}),
            schedule_expr=None,
            schedule_timezone="UTC",
            active=True,
            max_concurrency=None,
            per_host_delay_ms=None,
            retry_policy_json=None,
            output_prefs_json=None,
            job_filters_json=json.dumps({
                # no require_include in payload; org default should enforce include-only gating
                "filters": [
                    {"type": "keyword", "action": "include", "value": {"keywords": ["NoMatch"], "match": "any"}},
                ]
            }),
        )

        res = await run_watchlist_job(user_id, job.id)
        assert res.get("items_found", 0) >= 1
        assert res.get("items_ingested", 0) == 0  # include-only gating blocked ingestion

        items, total = db.list_items(run_id=None, job_id=job.id, status=None, limit=50, offset=0)
        assert total >= 1
        # All should be filtered since include didn't match
        assert all(i.status == "filtered" for i in items)
    finally:
        reset_scope(scope_token)


@pytest.mark.asyncio
async def test_include_only_gating_enabled_by_env_default_y(monkeypatch):
    monkeypatch.setenv("WATCHLISTS_REQUIRE_INCLUDE_DEFAULT", "y")

    user_id = 905
    db = WatchlistsDatabase.for_user(user_id)
    feed_url = f"https://example.com/feed-{uuid4().hex}.xml"

    src = db.create_source(
        name="Feed",
        url=feed_url,
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["gated"],
        group_ids=[],
    )

    job = db.create_job(
        name="IncludeOnlyByEnv",
        description=None,
        scope_json=json.dumps({"sources": [src.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
        job_filters_json=json.dumps(
            {
                "filters": [
                    {
                        "type": "keyword",
                        "action": "include",
                        "value": {"keywords": ["NoMatch"], "match": "any"},
                    },
                ]
            }
        ),
    )

    res = await run_watchlist_job(user_id, job.id)
    assert res.get("items_found", 0) >= 1
    assert res.get("items_ingested", 0) == 0
