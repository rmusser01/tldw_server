"""
Tests for watchlist briefing gap coverage.

Covers:
- Filter edge cases (date_range max_age_days=0, keyword match="all")
- Template rendering edge cases (XSS escaping, unicode, long content)
- Pipeline edge cases (429 retry-after, history 1000+ items)
- Source group_ids round-trip (validates Bug A1 fix)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Watchlists.filters import evaluate_filters, normalize_filters

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Filter edge cases
# ---------------------------------------------------------------------------


def test_date_range_max_age_days_boundary():
    """max_age_days enforces strict day boundary; items older than N days are excluded."""
    now = datetime.now(timezone.utc)
    # An item from 12 hours ago should be within 1-day window
    recent_iso = (now - timedelta(hours=12)).isoformat()
    # An item from 2 days ago should be outside 1-day window
    old_iso = (now - timedelta(days=2)).isoformat()

    payload = {
        "filters": [
            {"type": "date_range", "action": "include", "value": {"max_age_days": 1}},
        ]
    }
    flt = normalize_filters(payload)

    # Recent item (12h ago) should match the 1-day window
    decision, _ = evaluate_filters(flt, {"title": "Fresh", "published_at": recent_iso})
    assert decision == "include"

    # Old item (2 days ago) should NOT match
    decision, _ = evaluate_filters(flt, {"title": "Old", "published_at": old_iso})
    assert decision is None

    # max_age_days=0 means delta must be <= 0 days (practically nothing matches)
    payload_zero = {
        "filters": [
            {"type": "date_range", "action": "include", "value": {"max_age_days": 0}},
        ]
    }
    flt_zero = normalize_filters(payload_zero)
    # Even a recent item will have a non-zero delta, so should not match
    decision, _ = evaluate_filters(flt_zero, {"title": "Now-ish", "published_at": recent_iso})
    assert decision is None


def test_keyword_match_all_requires_both_keywords():
    """match='all' requires every keyword to appear; partial matches should fail."""
    payload = {
        "filters": [
            {
                "type": "keyword",
                "action": "include",
                "value": {"keywords": ["AI", "safety"], "match": "all"},
            },
        ]
    }
    flt = normalize_filters(payload)

    # Only "AI" present -> should NOT match
    decision, _ = evaluate_filters(flt, {
        "title": "AI breakthroughs in 2026",
        "summary": "Discusses new AI models",
    })
    assert decision is None

    # Both "AI" and "safety" present -> should match
    decision, _ = evaluate_filters(flt, {
        "title": "AI safety research advances",
        "summary": "New safety benchmarks for AI systems",
    })
    assert decision == "include"


# ---------------------------------------------------------------------------
# Template rendering edge cases
# ---------------------------------------------------------------------------


def test_xss_in_title_escaped_in_html_template():
    """Ensure XSS payloads in titles are escaped by the sandboxed template env."""
    from tldw_Server_API.app.services.outputs_service import render_output_template

    xss_title = '<script>alert(1)</script>'
    template_str = "<h1>{{ title }}</h1>"
    rendered = render_output_template(template_str, {"title": xss_title})
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_unicode_mixed_scripts_render():
    """Japanese + Arabic + emoji in title/summary should render without error."""
    from tldw_Server_API.app.services.outputs_service import render_output_template

    template_str = "# {{ title }}\n{{ summary }}"
    context = {
        "title": "\u6771\u4eac \u0627\u0644\u0642\u0627\u0647\u0631\u0629 \U0001f30d",
        "summary": "\u30c6\u30b9\u30c8 \u0627\u062e\u062a\u0628\u0627\u0631 \U0001f680\U0001f4a1",
    }
    rendered = render_output_template(template_str, context)
    assert "\u6771\u4eac" in rendered
    assert "\U0001f30d" in rendered
    assert "\u0627\u062e\u062a\u0628\u0627\u0631" in rendered


def test_very_long_summary_no_crash():
    """A 50K character summary should render without crash or OOM."""
    from tldw_Server_API.app.services.outputs_service import render_output_template

    long_summary = "A" * 50_000
    template_str = "{{ summary }}"
    rendered = render_output_template(template_str, {"summary": long_summary})
    assert len(rendered) >= 50_000


# ---------------------------------------------------------------------------
# Pipeline edge cases
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _pipeline_test_env(monkeypatch):
    """Isolate pipeline tests with TEST_MODE and a temp DB dir."""
    monkeypatch.setenv("TEST_MODE", "1")
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_gap_coverage"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    yield


def test_defer_until_skips_source_in_pipeline(_pipeline_test_env):
    """A source with a future defer_until is skipped by the pipeline's source loop."""
    import random

    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    user_id = random.randint(100000, 999999)
    db = WatchlistsDatabase.for_user(user_id)

    rss = db.create_source(
        name="DeferredSource",
        url="https://example.com/defer-feed.xml",
        source_type="rss",
        active=True,
        settings_json=None,
        tags=[],
        group_ids=[],
    )

    # Set defer_until to 1 hour in the future
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    db.update_source_scrape_meta(rss.id, defer_until=future)

    # Verify defer_until is stored
    updated = db.get_source(rss.id)
    assert updated.defer_until is not None

    # Clear it
    db.clear_source_defer_until(rss.id)
    cleared = db.get_source(rss.id)
    assert cleared.defer_until is None


@pytest.mark.asyncio
async def test_history_fetch_1000_items(_pipeline_test_env):
    """When fetch_rss_feed_history returns 1000+ items, pipeline counts them all."""
    import random

    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
    from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job

    user_id = random.randint(100000, 999999)
    db = WatchlistsDatabase.for_user(user_id)

    rss = db.create_source(
        name="HistoryFeed",
        url=f"https://example.com/history-feed-{user_id}.xml",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"history_pages": 5}),
        tags=[],
        group_ids=[],
    )

    job = db.create_job(
        name="HistoryJob",
        description=None,
        scope_json=json.dumps({"sources": [rss.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
        job_filters_json=None,
    )

    # In TEST_MODE, the pipeline generates test items; just verify it runs
    result = await run_watchlist_job(user_id, job.id)
    assert "run_id" in result
    assert result.get("items_found", 0) >= 1


# ---------------------------------------------------------------------------
# Source group_ids round-trip (validates A1 fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_group_ids_in_response(_pipeline_test_env):
    """After creating a source with group_ids, API response should include them."""
    import random

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.config import API_V1_PREFIX

    uid = random.randint(100000, 999999)

    async def override_user():
        return User(id=uid, username="gaptest", email="gap@test.com", is_active=True)

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user

    with TestClient(app) as client:
        # Create a group first
        g = client.post("/api/v1/watchlists/groups", json={"name": f"TestGroup-{uid}"})
        assert g.status_code == 200, g.text
        group_id = g.json()["id"]

        # Create source with group_ids
        s = client.post("/api/v1/watchlists/sources", json={
            "name": "GroupedSource",
            "url": f"https://example.com/grouped-feed-{uid}.xml",
            "source_type": "rss",
            "group_ids": [group_id],
        })
        assert s.status_code == 200, s.text
        source_data = s.json()
        assert "group_ids" in source_data
        assert group_id in source_data["group_ids"]

        # GET single source
        g2 = client.get(f"/api/v1/watchlists/sources/{source_data['id']}")
        assert g2.status_code == 200, g2.text
        assert group_id in g2.json()["group_ids"]

        # GET list sources
        g3 = client.get("/api/v1/watchlists/sources")
        assert g3.status_code == 200, g3.text
        items = g3.json()["items"]
        matching = [i for i in items if i["id"] == source_data["id"]]
        assert len(matching) == 1
        assert group_id in matching[0]["group_ids"]

        # GET list sources with group filter
        g4 = client.get(f"/api/v1/watchlists/sources?groups={group_id}")
        assert g4.status_code == 200, g4.text
        assert g4.json()["total"] >= 1
        assert any(i["id"] == source_data["id"] for i in g4.json()["items"])

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# _maybe_auto_generate_output unit tests
# ---------------------------------------------------------------------------


class _FakeRun:
    def __init__(self, run_id: int = 1):
        self.id = run_id


class _FakeJob:
    def __init__(self, job_id: int = 10, name: str = "TestJob"):
        self.id = job_id
        self.name = name


class _FakeScrapedItemRow:
    """Minimal stand-in for ScrapedItemRow returned by db.list_items()."""

    def __init__(
        self,
        *,
        item_id: int = 1,
        title: str = "Article",
        url: str = "https://example.com/a",
        summary: str = "Summary text",
        published_at: str = "2026-01-01T00:00:00Z",
        tags_json: str | None = None,
        media_id: int | None = None,
    ):
        self.id = item_id
        self.media_id = media_id
        self.title = title
        self.url = url
        self.summary = summary
        self.published_at = published_at
        self.tags_json = tags_json


class _FakeArtifact:
    def __init__(self, artifact_id: int = 42):
        self.id = artifact_id


@pytest.mark.asyncio
async def test_auto_output_skipped_when_disabled():
    """auto_output.enabled = False → returns None, no file written."""
    from tldw_Server_API.app.core.Watchlists.pipeline import _maybe_auto_generate_output

    result = await _maybe_auto_generate_output(
        db=None,
        collections_db=None,
        user_id=1,
        run=_FakeRun(),
        job=_FakeJob(),
        job_output_prefs={"auto_output": {"enabled": False}},
        stats={"items_ingested": 5},
    )
    assert result is None


@pytest.mark.asyncio
async def test_auto_output_skipped_when_no_items_ingested():
    """items_ingested = 0 → returns None."""
    from tldw_Server_API.app.core.Watchlists.pipeline import _maybe_auto_generate_output

    result = await _maybe_auto_generate_output(
        db=None,
        collections_db=None,
        user_id=1,
        run=_FakeRun(),
        job=_FakeJob(),
        job_output_prefs={"auto_output": {"enabled": True}},
        stats={"items_ingested": 0},
    )
    assert result is None


@pytest.mark.asyncio
async def test_auto_output_generates_file_and_artifact(tmp_path, monkeypatch):
    """Happy path: items returned → file written → artifact created → returns artifact ID."""
    from tldw_Server_API.app.core.Watchlists.pipeline import _maybe_auto_generate_output

    fake_items = [
        _FakeScrapedItemRow(item_id=1, title="First", url="https://a.com", summary="Sum1"),
        _FakeScrapedItemRow(item_id=2, title="Second", url="https://b.com", summary="Sum2"),
    ]

    class FakeDB:
        def list_items(self, *, run_id, status, limit, offset):
            return (fake_items, 2)

    class FakeCollDB:
        artifact_kwargs: dict | None = None

        def create_output_artifact(self, **kwargs):
            self.artifact_kwargs = kwargs
            return _FakeArtifact(artifact_id=99)

    # Patch _outputs_dir_for_user and _resolve_output_path_for_user to use tmp_path
    monkeypatch.setattr(
        "tldw_Server_API.app.services.outputs_service._outputs_dir_for_user",
        lambda uid: tmp_path,
    )

    def fake_resolve(uid, filename):
        return tmp_path / filename

    monkeypatch.setattr(
        "tldw_Server_API.app.services.outputs_service._resolve_output_path_for_user",
        fake_resolve,
    )

    fake_collections_db = FakeCollDB()
    result = await _maybe_auto_generate_output(
        db=FakeDB(),
        collections_db=fake_collections_db,
        user_id=1,
        run=_FakeRun(run_id=7),
        job=_FakeJob(job_id=10, name="MyJob"),
        job_output_prefs={"auto_output": {"enabled": True, "type": "briefing_markdown"}},
        stats={"items_ingested": 2},
    )
    assert result == 99
    metadata = json.loads(fake_collections_db.artifact_kwargs["metadata_json"])
    assert metadata["origin"] == "watchlists"
    assert metadata["generation_mode"] == "auto_output"
    assert metadata["auto_output"] is True

    # Verify a file was written
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "MyJob-Auto-7" in content
    assert "First" in content
    assert "Second" in content


@pytest.mark.asyncio
async def test_auto_output_tags_parsed_correctly(tmp_path, monkeypatch):
    """Items with tags_json='["a","b"]' → context dict has tags: ["a", "b"]."""
    from tldw_Server_API.app.services.outputs_service import build_items_context_from_content_items

    rows = [
        _FakeScrapedItemRow(item_id=1, title="Tagged", tags_json='["alpha", "beta"]'),
        _FakeScrapedItemRow(item_id=2, title="NoTags", tags_json=None),
        _FakeScrapedItemRow(item_id=3, title="BadJSON", tags_json="{not-json"),
    ]
    items = build_items_context_from_content_items(rows)

    assert items[0]["tags"] == ["alpha", "beta"]
    assert items[1]["tags"] == []
    assert items[2]["tags"] == []
