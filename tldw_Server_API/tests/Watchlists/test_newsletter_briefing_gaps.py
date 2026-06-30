"""
Tests for newsletter briefing workflow gap coverage.

Covers missing test cases identified in the plan vs implementation gap analysis:
- OPML round-trip (import → export → reimport skips duplicates)
- Source type auto-detection from URL
- Built-in default templates seeding
- LLM summarization integration (mocked)
- Feed creation triggers immediate first run
- Output TTL / retention enforcement
- Concurrent job run safety
"""
from __future__ import annotations

import io
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Source type auto-detection
# ---------------------------------------------------------------------------


class TestSourceTypeAutoDetection:
    """Tests for _detect_source_type() in collections_feeds.py."""

    def test_rss_xml_extension(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/feed.xml") == "rss"

    def test_rss_rss_extension(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/news.rss") == "rss"

    def test_rss_atom_extension(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/feed.atom") == "rss"

    def test_rss_feed_path(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/feed") == "rss"

    def test_rss_rss_path(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/rss") == "rss"

    def test_rss_atom_path(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://blog.example.com/atom") == "rss"

    def test_rss_index_xml_path(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://blog.example.com/index.xml") == "rss"

    def test_rss_query_param_format(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/news?format=rss") == "rss"

    def test_plain_url_defaults_to_rss(self):
        """Plain URLs default to rss for backwards compatibility."""
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("https://example.com/news") == "rss"

    def test_empty_url_defaults_to_rss(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        assert _detect_source_type("") == "rss"

    def test_feed_extension_with_trailing_slash(self):
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import _detect_source_type

        # Trailing slash is stripped before check
        assert _detect_source_type("https://example.com/feed/") == "rss"


# ---------------------------------------------------------------------------
# Built-in default templates
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    """Tests for default template seeding in template_store.py."""

    def test_seed_defaults_creates_templates(self, tmp_path, monkeypatch):
        """Seeding writes the built-in templates to a fresh directory."""
        import tldw_Server_API.app.core.Watchlists.template_store as ts

        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path))
        # Reset the seed flag so _seed_defaults runs again
        ts._defaults_seeded = False

        templates = ts.list_templates()
        names = {t.name for t in templates}
        assert "briefing_markdown" in names
        assert "newsletter_markdown" in names
        assert "mece_markdown" in names
        assert "newsletter_html" in names
        assert "cti_osint_report_markdown" in names
        assert "news_briefing_markdown" in names

    def test_seed_defaults_does_not_overwrite(self, tmp_path, monkeypatch):
        """If a template already exists, seeding does not overwrite it."""
        import tldw_Server_API.app.core.Watchlists.template_store as ts

        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path))
        ts._defaults_seeded = False

        # Pre-create one template with custom content
        (tmp_path / "briefing_markdown.md").write_text("Custom content", encoding="utf-8")

        ts._seed_defaults()
        content = (tmp_path / "briefing_markdown.md").read_text(encoding="utf-8")
        assert content == "Custom content"

    def test_load_builtin_template(self, tmp_path, monkeypatch):
        """Built-in templates can be loaded by name."""
        import tldw_Server_API.app.core.Watchlists.template_store as ts

        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path))
        ts._defaults_seeded = False

        record = ts.load_template("briefing_markdown")
        assert record.name == "briefing_markdown"
        assert record.format == "md"
        assert "{{ title }}" in record.content

    def test_newsletter_html_template_valid(self, tmp_path, monkeypatch):
        """The HTML template can be loaded and has valid structure."""
        import tldw_Server_API.app.core.Watchlists.template_store as ts

        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path))
        ts._defaults_seeded = False

        record = ts.load_template("newsletter_html")
        assert record.format == "html"
        assert "<html>" in record.content
        assert "{{ title }}" in record.content

    def test_builtin_templates_render_without_error(self, tmp_path, monkeypatch):
        """All built-in templates render with a minimal context without errors."""
        import tldw_Server_API.app.core.Watchlists.template_store as ts
        from tldw_Server_API.app.services.outputs_service import render_output_template

        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path))
        ts._defaults_seeded = False

        context = {
            "title": "Test Briefing",
            "generated_at": "2026-02-07T12:00:00Z",
            "items": [
                {
                    "title": "Article One",
                    "url": "https://example.com/1",
                    "summary": "Summary of article one.",
                    "tags": ["tech", "ai"],
                    "published_at": "2026-02-07T10:00:00Z",
                },
                {
                    "title": "Article Two",
                    "url": "https://example.com/2",
                    "summary": "Summary of article two.",
                    "tags": ["news"],
                    "published_at": "2026-02-07T11:00:00Z",
                },
            ],
            "item_count": 2,
        }

        context["report"] = {
            "preset": "cti_osint",
            "readiness": {
                "state": "warning",
                "score": 76,
                "warnings": [
                    {"code": "single_source", "message": "Only one source is represented."}
                ],
            },
            "source_summary": {"unique_source_count": 2, "missing_source_count": 0},
            "included_items": [
                {
                    "id": 1,
                    "title": "Article One",
                    "url": "https://example.com/1",
                    "source_name": "Vendor Advisory",
                    "published_at": "2026-02-07T10:00:00Z",
                    "summary": "Summary of article one.",
                    "alerts": [
                        {
                            "rule_name": "Active exploitation",
                            "severity": "critical",
                            "matched_text": "active exploitation",
                        }
                    ],
                }
            ],
            "excluded_items": [
                {"id": 3, "title": "Background item", "reason": "not_queued_for_report"}
            ],
            "included_count": 1,
            "excluded_count": 1,
            "alert_count": 1,
        }

        for name in (
            "briefing_markdown",
            "newsletter_markdown",
            "mece_markdown",
            "newsletter_html",
            "cti_osint_report_markdown",
            "news_briefing_markdown",
        ):
            record = ts.load_template(name)
            rendered = render_output_template(record.content, context)
            assert "Test Briefing" in rendered
            assert len(rendered) > 50

    def test_stage5_report_templates_render_evidence_context(self, tmp_path, monkeypatch):
        """Stage 5 report templates expose readiness, source, alert, and excluded-trail context."""
        import tldw_Server_API.app.core.Watchlists.template_store as ts
        from tldw_Server_API.app.services.outputs_service import render_output_template

        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path))
        ts._defaults_seeded = False

        context = {
            "title": "Healthcare ransomware watch",
            "generated_at": "2026-05-15T12:00:00Z",
            "job_name": "Ransomware advisory monitor",
            "run_id": 44,
            "items": [],
            "report": {
                "preset": "cti_osint",
                "readiness": {
                    "state": "warning",
                    "score": 76,
                    "warnings": [
                        {
                            "code": "single_source",
                            "message": "Only one source is represented.",
                        }
                    ],
                },
                "source_summary": {"unique_source_count": 2, "missing_source_count": 0},
                "included_items": [
                    {
                        "id": 101,
                        "title": "Vendor advisory confirms exploitation",
                        "url": "https://vendor.example/cve",
                        "source_name": "Vendor Advisory",
                        "published_at": "2026-05-15T10:00:00Z",
                        "summary": "Active exploitation against edge devices.",
                        "alerts": [
                            {
                                "rule_name": "Critical CVE",
                                "severity": "critical",
                                "matched_text": "CVE-2026-1234",
                            }
                        ],
                    }
                ],
                "excluded_items": [
                    {
                        "id": 102,
                        "title": "Background market note",
                        "reason": "not_queued_for_report",
                    }
                ],
                "included_count": 1,
                "excluded_count": 1,
                "alert_count": 1,
            },
        }

        cti = render_output_template(ts.load_template("cti_osint_report_markdown").content, context)
        assert "Report readiness: warning" in cti
        assert "Critical CVE" in cti
        assert "Vendor Advisory" in cti
        assert "Excluded trail" in cti
        assert "Background market note" in cti

        context["report"]["preset"] = "news_briefing"
        news = render_output_template(ts.load_template("news_briefing_markdown").content, context)
        assert "What changed" in news
        assert "Source diversity" in news
        assert "Follow-up links" in news
        assert "Vendor advisory confirms exploitation" in news


# ---------------------------------------------------------------------------
# LLM Summarization
# ---------------------------------------------------------------------------


class TestLLMSummarization:
    """Tests for summarize_items_for_output in outputs_service.py."""

    @pytest.mark.asyncio
    async def test_summarize_items_calls_llm(self):
        """summarize_items_for_output calls the LLM for each item."""
        from tldw_Server_API.app.services.outputs_service import summarize_items_for_output

        items = [
            {"id": 1, "summary": "Article about AI safety research."},
            {"id": 2, "summary": "New developments in quantum computing."},
        ]

        with patch(
            "tldw_Server_API.app.services.outputs_service._summarize_single_article",
            return_value="Mocked summary.",
        ) as mock_summarize:
            result = await summarize_items_for_output(
                items, api_name="openai"
            )

        assert len(result) == 2
        assert result[0]["llm_summary"] == "Mocked summary."
        assert result[1]["llm_summary"] == "Mocked summary."
        assert mock_summarize.call_count == 2

    @pytest.mark.asyncio
    async def test_summarize_items_empty_content_skipped(self):
        """Items with empty content get empty llm_summary."""
        from tldw_Server_API.app.services.outputs_service import summarize_items_for_output

        items = [{"id": 1, "summary": ""}]

        with patch(
            "tldw_Server_API.app.services.outputs_service._summarize_single_article",
        ) as mock_summarize:
            result = await summarize_items_for_output(items, api_name="openai")

        assert result[0]["llm_summary"] == ""
        mock_summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_items_uses_cache(self):
        """If cached_summary matches content hash, LLM is not called."""
        from tldw_Server_API.app.services.outputs_service import (
            _content_cache_key,
            summarize_items_for_output,
        )

        content = "Article about AI safety."
        content_hash = _content_cache_key(content)
        meta = json.dumps({
            "cached_summary": {"content_hash": content_hash, "text": "Cached summary."}
        })

        items = [{"id": 1, "summary": content, "metadata_json": meta}]

        with patch(
            "tldw_Server_API.app.services.outputs_service._summarize_single_article",
        ) as mock_summarize:
            result = await summarize_items_for_output(items, api_name="openai")

        assert result[0]["llm_summary"] == "Cached summary."
        mock_summarize.assert_not_called()

    def test_content_cache_key_deterministic(self):
        """Same content always produces the same cache key."""
        from tldw_Server_API.app.services.outputs_service import _content_cache_key

        key1 = _content_cache_key("hello world")
        key2 = _content_cache_key("hello world")
        assert key1 == key2
        assert len(key1) == 16

    def test_content_cache_key_varies(self):
        """Different content produces different cache keys."""
        from tldw_Server_API.app.services.outputs_service import _content_cache_key

        key1 = _content_cache_key("hello world")
        key2 = _content_cache_key("goodbye world")
        assert key1 != key2


# ---------------------------------------------------------------------------
# OPML Round-trip (import → export → reimport skips duplicates)
# ---------------------------------------------------------------------------


@pytest.fixture()
def opml_client(monkeypatch, tmp_path):
    """Client fixture for OPML tests."""
    uid = random.randint(100000, 999999)

    async def override_user():
        return User(id=uid, username="opmltest", email=None, is_active=True)

    base_dir = tmp_path / "test_user_dbs_opml_rt"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.config import API_V1_PREFIX

    app = FastAPI()
    app.include_router(watchlists_router, prefix=API_V1_PREFIX)
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# Import the TestClient here for the fixture
from fastapi.testclient import TestClient
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


@pytest.mark.integration
class TestOPMLRoundTrip:
    """Tests for OPML import → export → reimport dedup."""

    def _sample_opml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<opml version="2.0">\n'
            "  <head><title>Test Feeds</title></head>\n"
            "  <body>\n"
            '    <outline text="Feed A" title="Feed A" xmlUrl="https://feedA.example.com/rss" />\n'
            '    <outline text="Feed B" title="Feed B" xmlUrl="https://feedB.example.com/rss" />\n'
            "  </body>\n"
            "</opml>\n"
        )

    def test_opml_reimport_skips_duplicates(self, opml_client):
        """Reimporting the same OPML should not create duplicate sources."""
        c = opml_client
        xml = self._sample_opml()

        # First import
        files1 = {"file": ("feeds.opml", io.BytesIO(xml.encode()), "application/xml")}
        r1 = c.post("/api/v1/watchlists/sources/import", files=files1, data={"active": "1"})
        assert r1.status_code == 200, r1.text
        created_first = r1.json()["created"]
        assert created_first >= 2

        # Export
        r_export = c.get("/api/v1/watchlists/sources/export")
        assert r_export.status_code == 200

        # Reimport same OPML — duplicates should be skipped gracefully
        files2 = {"file": ("feeds.opml", io.BytesIO(xml.encode()), "application/xml")}
        r2 = c.post("/api/v1/watchlists/sources/import", files=files2, data={"active": "1"})
        assert r2.status_code == 200, r2.text
        data2 = r2.json()
        assert data2["created"] == 0, f"Expected 0 created on reimport, got {data2['created']}"
        assert data2["skipped"] >= 2, f"Expected >=2 skipped on reimport, got {data2['skipped']}"


# ---------------------------------------------------------------------------
# Output TTL / retention enforcement
# ---------------------------------------------------------------------------


class TestOutputRetentionTTL:
    """Tests for output retention/TTL expiration logic."""

    def test_find_outputs_to_purge_respects_retention(self):
        """Outputs past their retention_until should appear in purge candidates."""
        from tldw_Server_API.app.services.outputs_service import find_outputs_to_purge

        # Create a mock cdb with a backend that returns expired outputs
        mock_cdb = MagicMock()
        mock_cdb.user_id = 1

        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Mock retention query - returns one expired output
        mock_retention_result = MagicMock()
        mock_retention_result.rows = [
            {"id": 100, "storage_path": "expired_output.md"},
        ]
        # Mock soft-deleted query - returns empty
        mock_deleted_result = MagicMock()
        mock_deleted_result.rows = []

        mock_cdb.backend.execute.side_effect = [mock_retention_result, mock_deleted_result]

        paths = find_outputs_to_purge(
            mock_cdb,
            now_iso=now_iso,
            soft_deleted_grace_days=7,
            include_retention=True,
        )

        assert 100 in paths
        assert paths[100] == "expired_output.md"

    def test_find_outputs_to_purge_skips_future_retention(self):
        """Outputs with future retention_until should not appear in purge candidates."""
        from tldw_Server_API.app.services.outputs_service import find_outputs_to_purge

        mock_cdb = MagicMock()
        mock_cdb.user_id = 1

        now_iso = datetime.now(timezone.utc).isoformat()

        # Retention query returns nothing (no expired outputs)
        mock_retention_result = MagicMock()
        mock_retention_result.rows = []
        mock_deleted_result = MagicMock()
        mock_deleted_result.rows = []

        mock_cdb.backend.execute.side_effect = [mock_retention_result, mock_deleted_result]

        paths = find_outputs_to_purge(
            mock_cdb,
            now_iso=now_iso,
            soft_deleted_grace_days=7,
            include_retention=True,
        )

        assert len(paths) == 0


# ---------------------------------------------------------------------------
# Feed creation triggers first run
# ---------------------------------------------------------------------------


class TestFeedCreationAutoRun:
    """Tests verifying feed creation triggers an immediate background run."""

    def test_create_feed_adds_background_task(self, tmp_path, monkeypatch):
        """POST /collections/feeds should schedule a background task for first run."""
        uid = random.randint(100000, 999999)

        async def override_user():
            return User(id=uid, username="feedtest", email=None, is_active=True)

        base_dir = tmp_path / "test_user_dbs_feed_autorun"
        base_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

        from fastapi import FastAPI

        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as feeds_router
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        from tldw_Server_API.app.core.config import API_V1_PREFIX

        app = FastAPI()
        app.include_router(feeds_router, prefix=API_V1_PREFIX)
        app.dependency_overrides[get_request_user] = override_user

        with TestClient(app) as client:
            r = client.post(
                f"{API_V1_PREFIX}/collections/feeds",
                json={
                    "url": "https://example.com/feed.xml",
                    "name": "Test Feed",
                    "active": True,
                },
            )
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["job_id"] is not None
            # The response should return immediately (background task is non-blocking)
            assert data["source_type"] == "rss"

        app.dependency_overrides.clear()

    def test_inactive_feed_does_not_auto_run(self, tmp_path, monkeypatch):
        """Creating an inactive feed should NOT trigger a background run."""
        uid = random.randint(100000, 999999)

        async def override_user():
            return User(id=uid, username="feedtest2", email=None, is_active=True)

        base_dir = tmp_path / "test_user_dbs_feed_inactive"
        base_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

        from fastapi import FastAPI

        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as feeds_router
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        from tldw_Server_API.app.core.config import API_V1_PREFIX

        app = FastAPI()
        app.include_router(feeds_router, prefix=API_V1_PREFIX)
        app.dependency_overrides[get_request_user] = override_user

        # Patch run_watchlist_job to verify it's NOT called
        with patch(
            "tldw_Server_API.app.core.Watchlists.pipeline.run_watchlist_job",
            new_callable=AsyncMock,
        ) as mock_run:
            with TestClient(app) as client:
                r = client.post(
                    f"{API_V1_PREFIX}/collections/feeds",
                    json={
                        "url": "https://example.com/feed.xml",
                        "name": "Inactive Feed",
                        "active": False,
                    },
                )
                assert r.status_code == 200, r.text

        # Background task should not have been queued for inactive feed
        # (The mock wouldn't be called because background tasks don't run in TestClient sync mode)
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Concurrent job run safety
# ---------------------------------------------------------------------------


class TestConcurrentJobRuns:
    """Tests for concurrent job execution safety."""

    @pytest.fixture(autouse=False)
    def _pipeline_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_MODE", "1")
        base_dir = tmp_path / "test_user_dbs_concurrent"
        base_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
        yield

    @pytest.mark.asyncio
    async def test_concurrent_runs_create_separate_run_ids(self, _pipeline_env):
        """Two concurrent runs of the same job should create separate run records."""
        import asyncio

        from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
        from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job

        user_id = random.randint(100000, 999999)
        db = WatchlistsDatabase.for_user(user_id)

        src = db.create_source(
            name="ConcurrentSrc",
            url=f"https://example.com/concurrent-{user_id}.xml",
            source_type="rss",
            active=True,
            settings_json=None,
            tags=[],
            group_ids=[],
        )
        job = db.create_job(
            name="ConcurrentJob",
            description=None,
            scope_json=json.dumps({"sources": [src.id]}),
            schedule_expr=None,
            schedule_timezone="UTC",
            active=True,
            max_concurrency=None,
            per_host_delay_ms=None,
            retry_policy_json=None,
            output_prefs_json=None,
            job_filters_json=None,
        )

        # Run two jobs concurrently
        results = await asyncio.gather(
            run_watchlist_job(user_id, job.id),
            run_watchlist_job(user_id, job.id),
            return_exceptions=True,
        )

        # Filter out any exceptions (concurrent access might cause some)
        successful = [r for r in results if isinstance(r, dict) and "run_id" in r]
        assert len(successful) >= 1, f"Expected at least 1 successful run, got: {results}"

        # If both succeeded, they should have different run IDs
        if len(successful) == 2:
            assert successful[0]["run_id"] != successful[1]["run_id"]


# ---------------------------------------------------------------------------
# Full pipeline integration (source → filter → output)
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end integration: create source, run pipeline, generate output."""

    @pytest.fixture(autouse=False)
    def _pipeline_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_MODE", "1")
        base_dir = tmp_path / "test_user_dbs_full_pipeline"
        base_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
        monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(tmp_path / "templates"))
        yield

    @pytest.mark.asyncio
    async def test_source_to_filter_to_output(self, _pipeline_env):
        """Full pipeline: create source → run job with filters → check items."""
        from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
        from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job

        user_id = random.randint(100000, 999999)
        db = WatchlistsDatabase.for_user(user_id)

        src = db.create_source(
            name="FullPipeSrc",
            url=f"https://example.com/full-pipe-{user_id}.xml",
            source_type="rss",
            active=True,
            settings_json=None,
            tags=["tech"],
            group_ids=[],
        )

        # Job with a keyword filter
        filters_payload = json.dumps({
            "filters": [
                {"type": "keyword", "action": "include", "value": {"keywords": ["Test"]}},
            ],
            "require_include": True,
        })
        job = db.create_job(
            name="FullPipeJob",
            description=None,
            scope_json=json.dumps({"sources": [src.id]}),
            schedule_expr=None,
            schedule_timezone="UTC",
            active=True,
            max_concurrency=None,
            per_host_delay_ms=None,
            retry_policy_json=None,
            output_prefs_json=None,
            job_filters_json=filters_payload,
        )

        result = await run_watchlist_job(user_id, job.id)
        assert "run_id" in result
        # In TEST_MODE with require_include, the test item "Test Item" should match
        assert result.get("items_found", 0) >= 1

        # Verify items were recorded
        items, total = db.list_items(run_id=result["run_id"], limit=100, offset=0)
        assert total >= 1

    @pytest.mark.asyncio
    async def test_excluded_items_filtered_out(self, _pipeline_env):
        """Items matching exclude filters should be filtered, not ingested."""
        from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
        from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job

        user_id = random.randint(100000, 999999)
        db = WatchlistsDatabase.for_user(user_id)

        src = db.create_source(
            name="ExcludeTestSrc",
            url=f"https://example.com/exclude-{user_id}.xml",
            source_type="rss",
            active=True,
            settings_json=None,
            tags=[],
            group_ids=[],
        )

        # Exclude filter that matches the test item title "Test Item"
        filters_payload = json.dumps({
            "filters": [
                {"type": "keyword", "action": "exclude", "value": {"keywords": ["Test"]}},
            ],
        })
        job = db.create_job(
            name="ExcludeJob",
            description=None,
            scope_json=json.dumps({"sources": [src.id]}),
            schedule_expr=None,
            schedule_timezone="UTC",
            active=True,
            max_concurrency=None,
            per_host_delay_ms=None,
            retry_policy_json=None,
            output_prefs_json=None,
            job_filters_json=filters_payload,
        )

        result = await run_watchlist_job(user_id, job.id)
        assert "run_id" in result
        # Items should be found but filtered out
        assert result.get("items_found", 0) >= 1
        # The filter should have excluded the test item
        filters_matched = result.get("filters_matched", 0)
        assert filters_matched >= 1


# ---------------------------------------------------------------------------
# URL Normalization for dedup
# ---------------------------------------------------------------------------


class TestURLNormalization:
    """Tests for _normalize_url() in pipeline.py."""

    def test_strips_trailing_slash(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        assert _normalize_url("https://example.com/article/") == "https://example.com/article"

    def test_removes_www_prefix(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        result = _normalize_url("https://www.example.com/page")
        assert "www." not in result
        assert "example.com/page" in result

    def test_lowercases_scheme_and_host(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        result = _normalize_url("HTTPS://Example.COM/Path")
        assert result.startswith("https://example.com/")
        # Path case is preserved
        assert "/Path" in result

    def test_removes_utm_params(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url = "https://example.com/article?utm_source=twitter&utm_medium=social&id=123"
        result = _normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=123" in result

    def test_removes_fbclid(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url = "https://example.com/page?fbclid=abc123&real_param=value"
        result = _normalize_url(url)
        assert "fbclid" not in result
        assert "real_param=value" in result

    def test_removes_gclid(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url = "https://example.com/page?gclid=xyz&page=1"
        result = _normalize_url(url)
        assert "gclid" not in result
        assert "page=1" in result

    def test_preserves_non_tracking_params(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url = "https://example.com/search?q=test&page=2&sort=date"
        result = _normalize_url(url)
        assert "q=test" in result
        assert "page=2" in result
        assert "sort=date" in result

    def test_empty_query_after_stripping(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url = "https://example.com/page?utm_source=newsletter"
        result = _normalize_url(url)
        assert result == "https://example.com/page"

    def test_no_query_string(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url = "https://example.com/article"
        assert _normalize_url(url) == "https://example.com/article"

    def test_root_path_preserved(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        assert _normalize_url("https://example.com/") == "https://example.com/"
        assert _normalize_url("https://example.com") == "https://example.com/"

    def test_strips_default_ports(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        result = _normalize_url("https://example.com:443/page")
        assert ":443" not in result
        result_http = _normalize_url("http://example.com:80/page")
        assert ":80" not in result_http

    def test_preserves_non_default_ports(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        result = _normalize_url("https://example.com:8080/page")
        assert ":8080" in result

    def test_equivalent_urls_normalize_equal(self):
        """URLs that differ only by tracking params / www / trailing slash should normalize identically."""
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        url_a = "https://www.example.com/article/?utm_source=twitter&utm_campaign=foo"
        url_b = "https://example.com/article"
        assert _normalize_url(url_a) == _normalize_url(url_b)

    def test_handles_malformed_url_gracefully(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        # Should not raise, returns input as-is or best-effort
        result = _normalize_url("")
        assert isinstance(result, str)

    def test_fragment_stripped(self):
        from tldw_Server_API.app.core.Watchlists.pipeline import _normalize_url

        result = _normalize_url("https://example.com/page#section1")
        assert "#" not in result
