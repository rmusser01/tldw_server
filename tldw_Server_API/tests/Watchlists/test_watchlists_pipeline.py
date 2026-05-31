import json
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from tldw_Server_API.app.core.Watchlists import pipeline
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import AudioBriefingTriggerResult
from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _test_mode_env(monkeypatch, tmp_path):
     # Force offline behavior
    monkeypatch.setenv("TEST_MODE", "1")
    # Route per-user DBs into an isolated temp directory per test.
    base_dir = tmp_path / "test_user_dbs_pipeline"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    yield


@pytest.mark.asyncio
async def test_pipeline_happy_path_test_mode():
    user_id = 777
    db = WatchlistsDatabase.for_user(user_id)

    # Group
    grp = db.create_group(name="News", description="", parent_group_id=None)

    # RSS source (1 item)
    rss = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["news"],
        group_ids=[grp.id],
    )

    # Site source with top_n=2 (frontpage)
    site = db.create_source(
        name="Site",
        url="https://example.com/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 2, "discover_method": "frontpage"}),
        tags=["news"],
        group_ids=[grp.id],
    )

    # Job with tag scope
    job = db.create_job(
        name="Brief",
        description=None,
        scope_json=json.dumps({"tags": ["news"]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    res = await run_watchlist_job(user_id, job.id)
    # items_found: 1 (rss) + 2 (site top links)
    assert res.get("items_found", 0) >= 3
    # items_ingested: expect at least 2 (rss once, site first link; duplicates skipped by URL uniqueness)
    assert res.get("items_ingested", 0) >= 2

    collections_db = CollectionsDatabase.for_user(user_id)
    coll_items, coll_total = collections_db.list_content_items(page=1, size=10)
    assert coll_total >= 1
    assert any(item.origin == "watchlist" for item in coll_items)
    assert any("news" in item.tags for item in coll_items)

    # Validate run row updated
    runs, total = db.list_runs_for_job(job.id, limit=10, offset=0)
    assert total >= 1
    run = runs[0]
    assert run.status == "succeeded"
    stats = json.loads(run.stats_json or "{}") if run.stats_json else {}
    assert stats.get("items_found", 0) >= 3


@pytest.mark.asyncio
async def test_pipeline_uses_managed_media_database_when_persisting(monkeypatch):
    user_id = 782
    db = WatchlistsDatabase.for_user(user_id)

    rss = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["news"],
        group_ids=[],
    )

    job = db.create_job(
        name="Persisted Feed Job",
        description=None,
        scope_json=json.dumps({"sources": [rss.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=json.dumps({"persist_to_media_db": True}),
    )

    class _MediaDb:
        backend = object()

        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 82, "watchlist-media-uuid", "stored"

    media_db = _MediaDb()
    fake_repo = _FakeRepo()
    managed_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield media_db
        finally:
            media_db.close_connection()

    monkeypatch.setattr(pipeline, "managed_media_database", _fake_managed_media_database, raising=False)
    monkeypatch.setattr(
        pipeline,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(pipeline, "get_media_repository", lambda db: fake_repo, raising=False)

    result = await run_watchlist_job(user_id, job.id)

    assert result["items_ingested"] >= 1
    assert media_db.closed is True
    assert len(fake_repo.calls) >= 1
    assert len(managed_calls) == 1
    assert managed_calls[0]["client_id"] == str(user_id)
    assert managed_calls[0]["initialize"] is True
    assert managed_calls[0]["kwargs"]["db_path"] == str(DatabasePaths.get_media_db_path(user_id))
    assert "suppress_close_exceptions" in managed_calls[0]["kwargs"]


@pytest.mark.asyncio
async def test_pipeline_sets_next_run_for_utc_timezone():
    user_id = 779
    db = WatchlistsDatabase.for_user(user_id)

    rss = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["news"],
        group_ids=[],
    )

    job = db.create_job(
        name="Scheduled Job",
        description=None,
        scope_json=json.dumps({"sources": [rss.id]}),
        schedule_expr="0 8 * * *",
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    await run_watchlist_job(user_id, job.id)
    updated = db.get_job(job.id)
    assert updated.next_run_at is not None


@pytest.mark.asyncio
async def test_pipeline_origin_override_feed():
    user_id = 780
    db = WatchlistsDatabase.for_user(user_id)

    rss = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1, "collections_origin": "feed"}),
        tags=["news"],
        group_ids=[],
    )

    job = db.create_job(
        name="Feed Job",
        description=None,
        scope_json=json.dumps({"sources": [rss.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    res = await run_watchlist_job(user_id, job.id)
    assert res.get("items_ingested", 0) >= 1

    collections_db = CollectionsDatabase.for_user(user_id)
    items, total = collections_db.list_content_items(origin="feed", page=1, size=10)
    assert total >= 1
    assert all(item.origin == "feed" for item in items)


@pytest.mark.asyncio
async def test_pipeline_batches_companion_activity_writes(monkeypatch):
    user_id = 781
    db = WatchlistsDatabase.for_user(user_id)
    personalization_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    personalization_db.update_profile(str(user_id), enabled=1)

    rss = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["news"],
        group_ids=[],
    )

    job = db.create_job(
        name="Feed Job",
        description=None,
        scope_json=json.dumps({"sources": [rss.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    bulk_calls: list[int] = []

    def _fake_bulk_insert(self, *, user_id, events):
        bulk_calls.append(len(events))
        return [f"evt-{index}" for index, _event in enumerate(events, start=1)]

    monkeypatch.setattr(
        PersonalizationDB,
        "insert_companion_activity_events_bulk",
        _fake_bulk_insert,
        raising=False,
    )

    await run_watchlist_job(
        user_id,
        job.id,
        capture_companion_activity=True,
        companion_route=f"/api/v1/watchlists/jobs/{job.id}/run",
    )

    assert bulk_calls
    assert sum(bulk_calls) >= 1


@pytest.mark.asyncio
async def test_scope_resolution_groups_and_tags():
    user_id = 778
    db = WatchlistsDatabase.for_user(user_id)

    g1 = db.create_group(name="G1", description=None, parent_group_id=None)
    g2 = db.create_group(name="G2", description=None, parent_group_id=None)

    s_a = db.create_source(
        name="A",
        url="https://a.example.com/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 1}),
        tags=["alpha"],
        group_ids=[g1.id],
    )
    s_b = db.create_source(
        name="B",
        url="https://b.example.com/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 1}),
        tags=["alpha", "beta"],
        group_ids=[g2.id],
    )
    s_c = db.create_source(
        name="C",
        url="https://c.example.com/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 1}),
        tags=["beta"],
        group_ids=[g1.id],
    )

    job = db.create_job(
        name="Selector",
        description=None,
        scope_json=json.dumps({"tags": ["alpha"], "groups": [g2.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    res = await run_watchlist_job(user_id, job.id)
    assert res.get("items_found", 0) >= 2

    # Validate only A and B updated (C not selected)
    r_a = db.get_source(s_a.id)
    r_b = db.get_source(s_b.id)
    r_c = db.get_source(s_c.id)
    assert r_a.last_scraped_at is not None
    assert r_b.last_scraped_at is not None
    assert r_c.last_scraped_at is None


@pytest.mark.asyncio
async def test_scope_resolution_groups_only():
    user_id = 781
    db = WatchlistsDatabase.for_user(user_id)

    g1 = db.create_group(name="OnlyGroup", description=None, parent_group_id=None)
    g2 = db.create_group(name="OtherGroup", description=None, parent_group_id=None)

    s_in = db.create_source(
        name="InScope",
        url="https://in.example.com/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 1}),
        tags=["inside"],
        group_ids=[g1.id],
    )
    s_out = db.create_source(
        name="OutScope",
        url="https://out.example.com/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 1}),
        tags=["outside"],
        group_ids=[g2.id],
    )

    job = db.create_job(
        name="GroupOnly",
        description=None,
        scope_json=json.dumps({"groups": [g1.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    res = await run_watchlist_job(user_id, job.id)
    assert res.get("items_found", 0) >= 1
    assert res.get("items_ingested", 0) >= 1

    in_row = db.get_source(s_in.id)
    out_row = db.get_source(s_out.id)
    assert in_row.last_scraped_at is not None
    assert out_row.last_scraped_at is None


@pytest.mark.asyncio
async def test_rss_dedup_and_meta_and_stats():
    user_id = 779
    db = WatchlistsDatabase.for_user(user_id)

    src = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["news"],
        group_ids=[],
    )
    job = db.create_job(
        name="RSSRun",
        description=None,
        scope_json=json.dumps({"tags": ["news"]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    one = await run_watchlist_job(user_id, job.id)
    assert one.get("items_found", 0) >= 1
    assert one.get("items_ingested", 0) >= 1

    # Second run should be deduped
    two = await run_watchlist_job(user_id, job.id)
    assert two.get("items_found", 0) >= 1
    assert two.get("items_ingested", 0) == 0

    # Source meta updated
    srow = db.get_source(src.id)
    assert srow.last_scraped_at is not None


@pytest.mark.skip(reason="Full pipeline enqueue check requires live ingestion; covered by unit tests")
@pytest.mark.asyncio
async def test_watchlist_run_enqueues_embeddings(monkeypatch):
    user_id = 780
    db = WatchlistsDatabase.for_user(user_id)

    src = db.create_source(
        name="Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["embed"],
        group_ids=[],
    )
    job = db.create_job(
        name="EmbeddingRun",
        description=None,
        scope_json=json.dumps({"tags": ["embed"]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    # Allow full pipeline path (not TEST_MODE); stub network fetchers.
    monkeypatch.setenv("TEST_MODE", "0")

    async def fake_fetch_rss_feed(url, etag=None, last_modified=None, timeout=8.0, tenant_id="default"):
        return {
            "status": 200,
            "items": [
                {
                    "title": "Stub Item",
                    "url": "https://example.com/article",
                    "summary": "Stub summary",
                    "published": None,
                }
            ],
            "etag": None,
            "last_modified": None,
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.pipeline.fetch_rss_feed",
        fake_fetch_rss_feed,
    )

    def fake_fetch_site_article(link: str):
        return {"title": "Stub Item", "url": link, "content": "Body", "author": None}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.pipeline.fetch_site_article",
        fake_fetch_site_article,
    )

    def fake_add_media_with_keywords(self, **kwargs):

        return 123, "uuid-123", "created"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.pipeline.MediaDatabase.add_media_with_keywords",
        fake_add_media_with_keywords,
    )

    captured = []

    async def fake_enqueue_embeddings_job_for_item(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.pipeline.enqueue_embeddings_job_for_item",
        fake_enqueue_embeddings_job_for_item,
    )

    res = await run_watchlist_job(user_id, job.id)
    assert res.get("items_ingested", 0) >= 1
    assert captured, "expected embeddings enqueue call"
    assert captured[0]["metadata"]["origin"] == "watchlist"


@pytest.mark.asyncio
async def test_site_scrape_rules_integration(monkeypatch):
    user_id = 881
    db = WatchlistsDatabase.for_user(user_id)

    source = db.create_source(
        name="Rules Site",
        url="https://example.com/blog",
        source_type="site",
        active=True,
        settings_json=json.dumps(
            {
                "scrape_rules": {
                    "list_url": "https://example.com/blog",
                    "limit": 2,
                    "skip_article_fetch": True,
                }
            }
        ),
        tags=["scrape"],
        group_ids=[],
    )

    job = db.create_job(
        name="ScrapeJob",
        description=None,
        scope_json=json.dumps({"sources": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    first = await run_watchlist_job(user_id, job.id)
    assert first.get("items_found") == 2
    assert first.get("items_ingested") >= 2
    assert db.has_seen_item(source.id, "https://example.com/blog/test-scrape-1")

    second = await run_watchlist_job(user_id, job.id)
    assert second.get("items_found") == 2
    assert second.get("items_ingested") == 0


@pytest.mark.asyncio
async def test_pipeline_persists_post_run_audio_and_output_stats():
    user_id = 882
    db = WatchlistsDatabase.for_user(user_id)
    unique_suffix = time.time_ns()

    source = db.create_source(
        name="Audio Feed",
        url=f"https://example.com/audio-feed-{unique_suffix}.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=["audio"],
        group_ids=[],
    )

    job = db.create_job(
        name="AudioOutputJob",
        description=None,
        scope_json=json.dumps({"sources": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=json.dumps(
            {
                "auto_output": {"enabled": True, "type": "briefing_markdown"},
                "generate_audio": True,
            }
        ),
    )

    def create_auto_output_artifact(**kwargs):
        return kwargs["collections_db"].create_output_artifact(
            type_="briefing_markdown",
            title="Auto Output",
            format_="md",
            storage_path="auto-output.md",
            metadata_json=json.dumps(
                {
                    "origin": "watchlists",
                    "generation_mode": "auto_output",
                    "run_id": kwargs["run"].id,
                    "job_id": kwargs["job"].id,
                }
            ),
            job_id=kwargs["job"].id,
            run_id=kwargs["run"].id,
        ).id

    with (
        patch(
            "tldw_Server_API.app.core.Watchlists.pipeline._maybe_auto_generate_output",
            new=AsyncMock(side_effect=create_auto_output_artifact),
        ),
        patch(
            "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
            new=AsyncMock(return_value=AudioBriefingTriggerResult(status="submitted", task_id="task_stage2")),
        ),
    ):
        result = await run_watchlist_job(user_id, job.id)

    auto_output_id = int(result["auto_output_id"])
    assert result.get("audio_briefing_task_id") == "task_stage2"
    assert result.get("audio_briefing_status") == "queued"

    persisted_run = db.get_run(int(result["run_id"]))
    persisted_stats = json.loads(persisted_run.stats_json or "{}")
    assert persisted_stats.get("auto_output_id") == auto_output_id
    assert persisted_stats.get("audio_briefing_task_id") == "task_stage2"
    assert persisted_stats.get("audio_briefing_status") == "queued"

    output_row = CollectionsDatabase.for_user(user_id).get_output_artifact(auto_output_id)
    output_metadata = json.loads(output_row.metadata_json or "{}")
    assert output_metadata["origin"] == "watchlists"
    assert output_metadata["generation_mode"] == "auto_output"
    assert output_metadata["audio_briefing_requested"] is True
    assert output_metadata["audio_briefing_status"] == "queued"
    assert output_metadata["audio_briefing_task_id"] == "task_stage2"
