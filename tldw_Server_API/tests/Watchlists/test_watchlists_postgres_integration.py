from __future__ import annotations

import os
import json

import pytest

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def _pg_backend(db_name: str):
    return DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.POSTGRESQL,
            pg_host=os.getenv("TEST_DB_HOST", "localhost"),
            pg_port=int(os.getenv("TEST_DB_PORT", "5432")),
            pg_database=db_name,
            pg_user=os.getenv("TEST_DB_USER", "tldw_user"),
            pg_password=os.getenv("TEST_DB_PASSWORD", "TestPassword123!"),
            pg_sslmode=os.getenv("TEST_DB_SSLMODE", "prefer"),
        )
    )


def test_watchlists_postgres_round_trip(request: pytest.FixtureRequest):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]

    backend = _pg_backend(db_name)
    db = WatchlistsDatabase(user_id="1", backend=backend)

    source = db.create_source(
        name="Example Feed",
        url="https://example.com/rss",
        source_type="rss",
        active=True,
        settings_json=None,
        tags=["news"],
    )
    assert source.id > 0

    group = db.create_group(name="Default", description=None, parent_group_id=None)
    assert group.id > 0

    job = db.create_job(
        name="Daily Watch",
        description=None,
        scope_json=json.dumps({"source_ids": [source.id]}),
        schedule_expr="0 * * * *",
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=1000,
        retry_policy_json=None,
        output_prefs_json=None,
        job_filters_json=None,
    )
    assert job.id > 0

    run = db.create_run(job.id, status="queued")
    assert run.id > 0

    item = db.record_scraped_item(
        run_id=run.id,
        job_id=job.id,
        source_id=source.id,
        media_id=None,
        media_uuid=None,
        url="https://example.com/post",
        title="Example Post",
        summary="Summary",
        published_at=None,
        tags=["news"],
        status="new",
    )
    assert item.id > 0


def test_watchlists_postgres_output_prefs_round_trip(request: pytest.FixtureRequest):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]

    backend = _pg_backend(db_name)
    db = WatchlistsDatabase(user_id="1", backend=backend)

    source = db.create_source(
        name="Prefs Feed",
        url="https://example.com/prefs-rss",
        source_type="rss",
        active=True,
        settings_json=None,
        tags=["prefs"],
    )

    output_prefs = {
        "template": {
            "default_name": "newsletter_html",
            "default_format": "html",
            "default_version": 2,
        },
        "retention": {
            "default_seconds": 3600,
            "temporary_seconds": 600,
        },
        "deliveries": {
            "email": {
                "enabled": True,
                "body_format": "html",
            }
        },
    }

    job = db.create_job(
        name="Prefs Job",
        description=None,
        scope_json=json.dumps({"source_ids": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=500,
        retry_policy_json=None,
        output_prefs_json=json.dumps(output_prefs),
        job_filters_json=None,
    )

    fetched = db.get_job(job.id)
    assert fetched.id == job.id
    parsed = json.loads(fetched.output_prefs_json or "{}")
    assert parsed.get("template", {}).get("default_name") == "newsletter_html"
    assert parsed.get("template", {}).get("default_format") == "html"
    assert parsed.get("template", {}).get("default_version") == 2
    assert parsed.get("retention", {}).get("default_seconds") == 3600
    assert parsed.get("retention", {}).get("temporary_seconds") == 600
    assert parsed.get("deliveries", {}).get("email", {}).get("enabled") is True


def test_watchlists_postgres_job_filters_round_trip(request: pytest.FixtureRequest):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]

    backend = _pg_backend(db_name)
    db = WatchlistsDatabase(user_id="1", backend=backend)

    source = db.create_source(
        name="Filters Feed",
        url="https://example.com/filters-rss",
        source_type="rss",
        active=True,
        settings_json=None,
        tags=["filters"],
    )

    job = db.create_job(
        name="Filters Job",
        description=None,
        scope_json=json.dumps({"source_ids": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=750,
        retry_policy_json=None,
        output_prefs_json=None,
        job_filters_json=None,
    )

    filters_payload = {
        "filters": [
            {
                "type": "keyword",
                "action": "include",
                "priority": 100,
                "is_active": True,
                "value": {"keywords": ["ai", "ml"], "match": "any", "field": "title"},
            },
            {
                "type": "regex",
                "action": "exclude",
                "priority": 90,
                "is_active": True,
                "value": {"pattern": "(?i)rumor", "field": "title", "flags": "i"},
            },
        ]
    }

    updated = db.set_job_filters(job.id, filters_payload)
    assert updated.id == job.id
    parsed = db.get_job_filters(job.id)
    assert len(parsed.get("filters", [])) == 2
    assert parsed["filters"][0]["type"] == "keyword"
    assert parsed["filters"][0]["priority"] == 100
    assert parsed["filters"][1]["type"] == "regex"
    assert parsed["filters"][1]["action"] == "exclude"


def test_watchlists_postgres_seen_state_round_trip(request: pytest.FixtureRequest):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]

    backend = _pg_backend(db_name)
    db = WatchlistsDatabase(user_id="1", backend=backend)

    source = db.create_source(
        name="Seen Feed",
        url="https://example.com/seen-rss",
        source_type="rss",
        active=True,
        settings_json=None,
        tags=["seen"],
    )

    key = "seen-key-1"
    assert db.has_seen_item(source.id, key) is False
    db.mark_seen_item(source.id, key, etag="etag-1")
    assert db.has_seen_item(source.id, key) is True

    # Upsert should keep row cardinality stable while refreshing latest_seen_at.
    db.mark_seen_item(source.id, key, last_modified="Mon, 01 Jan 2024 00:00:00 GMT")
    stats = db.get_seen_item_stats(source.id)
    assert int(stats.get("seen_count") or 0) == 1
    keys = db.list_seen_item_keys(source.id, limit=10)
    assert key in keys

    cleared = db.clear_seen_items(source.id)
    assert cleared == 1
    stats_after = db.get_seen_item_stats(source.id)
    assert int(stats_after.get("seen_count") or 0) == 0
