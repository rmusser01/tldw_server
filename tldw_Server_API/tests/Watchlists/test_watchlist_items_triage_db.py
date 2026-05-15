from __future__ import annotations

import json
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


pytestmark = pytest.mark.unit


def _make_db(tmp_path, *, user_id: int = 124) -> WatchlistsDatabase:
    db_path = tmp_path / f"watchlists_items_triage_{user_id}.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    return WatchlistsDatabase(user_id=user_id, backend=backend)


def _seed_review_queue(db: WatchlistsDatabase) -> dict[str, Any]:
    watchlist = db.create_watchlist(
        name="Stage 4 CTI review",
        objective="Track exploitable vulnerabilities",
        domain="cti_osint",
        priority="high",
        tags=["cti"],
    )
    first_source = db.create_source(
        name="Alpha Advisory Feed",
        url="https://example.com/alpha.xml",
        source_type="rss",
        tags=["advisory"],
        watchlist_id=int(watchlist.id),
    )
    second_source = db.create_source(
        name="Beta News Feed",
        url="https://example.com/beta.xml",
        source_type="rss",
        tags=["news"],
        watchlist_id=int(watchlist.id),
    )
    job = db.create_job(
        name="Stage 4 monitor",
        description=None,
        scope_json=json.dumps({"sources": [int(first_source.id), int(second_source.id)]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=0,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
        job_filters_json=None,
        watchlist_id=int(watchlist.id),
    )
    run = db.create_run(int(job.id), status="finished")
    older = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job.id),
        source_id=int(second_source.id),
        media_id=None,
        media_uuid=None,
        url="https://example.com/beta/cve-old",
        title="Older ransomware advisory",
        summary="Older advisory with critical ransomware context.",
        content="Older content",
        published_at="2026-05-13T08:00:00+00:00",
        tags=["ransomware"],
        status="ingested",
    )
    newer = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job.id),
        source_id=int(first_source.id),
        media_id=None,
        media_uuid=None,
        url="https://example.com/alpha/cve-new",
        title="Newer CISA advisory",
        summary="Newer advisory with active exploitation context.",
        content="Newer content",
        published_at="2026-05-15T08:00:00+00:00",
        tags=["cve"],
        status="ingested",
    )
    db.backend.execute(
        "UPDATE scraped_items SET created_at = ? WHERE id = ?",
        ("2026-05-13T09:00:00+00:00", int(older.id)),
    )
    db.backend.execute(
        "UPDATE scraped_items SET created_at = ? WHERE id = ?",
        ("2026-05-15T09:00:00+00:00", int(newer.id)),
    )
    db.update_item_flags(int(newer.id), reviewed=True)
    return {
        "watchlist": watchlist,
        "sources": [first_source, second_source],
        "job": job,
        "run": run,
        "older": older,
        "newer": newer,
    }


def test_list_items_supports_server_authoritative_sort_modes(tmp_path):
    db = _make_db(tmp_path)
    seeded = _seed_review_queue(db)
    watchlist_id = int(seeded["watchlist"].id)
    older_id = int(seeded["older"].id)
    newer_id = int(seeded["newer"].id)

    default_rows, default_total = db.list_items(watchlist_id=watchlist_id, limit=10, offset=0)
    published_asc, _ = db.list_items(
        watchlist_id=watchlist_id,
        sort="published_asc",
        limit=10,
        offset=0,
    )
    unread_first, _ = db.list_items(
        watchlist_id=watchlist_id,
        sort="unread_first",
        limit=10,
        offset=0,
    )
    source_asc, _ = db.list_items(
        watchlist_id=watchlist_id,
        sort="source_asc",
        limit=10,
        offset=0,
    )

    assert default_total == 2
    assert [int(row.id) for row in default_rows] == [newer_id, older_id]
    assert [int(row.id) for row in published_asc] == [older_id, newer_id]
    assert [int(row.id) for row in unread_first] == [older_id, newer_id]
    assert [int(row.id) for row in source_asc] == [newer_id, older_id]

    with pytest.raises(ValueError, match="invalid_item_sort"):
        db.list_items(watchlist_id=watchlist_id, sort="novelty_desc", limit=10, offset=0)


def test_list_items_filters_by_content_alert_context_and_summarizes_alerts(tmp_path):
    db = _make_db(tmp_path)
    seeded = _seed_review_queue(db)
    watchlist_id = int(seeded["watchlist"].id)
    source_id = int(seeded["sources"][0].id)
    job_id = int(seeded["job"].id)
    run_id = int(seeded["run"].id)
    alerted_item_id = int(seeded["newer"].id)
    quiet_item_id = int(seeded["older"].id)

    critical_rule = db.create_content_alert_rule(
        watchlist_id=watchlist_id,
        name="Critical exploitation",
        rule_kind="keyword",
        pattern="active exploitation",
        severity="critical",
    )
    low_rule = db.create_content_alert_rule(
        watchlist_id=watchlist_id,
        name="General ransomware",
        rule_kind="keyword",
        pattern="ransomware",
        severity="low",
    )
    critical_alert = db.create_content_alert(
        watchlist_id=watchlist_id,
        rule_id=int(critical_rule.id),
        item_id=alerted_item_id,
        run_id=run_id,
        job_id=job_id,
        source_id=source_id,
        severity="critical",
        title="Newer CISA advisory",
        snippet="active exploitation context",
        matched_text="active exploitation",
        evidence={"field": "summary"},
        dedupe_key=f"{watchlist_id}:{critical_rule.id}:{alerted_item_id}",
    )
    dismissed_alert = db.create_content_alert(
        watchlist_id=watchlist_id,
        rule_id=int(low_rule.id),
        item_id=alerted_item_id,
        run_id=run_id,
        job_id=job_id,
        source_id=source_id,
        severity="low",
        title="Newer CISA advisory",
        snippet="ransomware context",
        matched_text="ransomware",
        evidence={"field": "summary"},
        dedupe_key=f"{watchlist_id}:{low_rule.id}:{alerted_item_id}",
        status="dismissed",
    )

    alert_rows, alert_total = db.list_items(
        watchlist_id=watchlist_id,
        has_alert=True,
        include_alert_summary=True,
        limit=10,
        offset=0,
    )
    quiet_rows, quiet_total = db.list_items(
        watchlist_id=watchlist_id,
        has_alert=False,
        limit=10,
        offset=0,
    )
    critical_rows, critical_total = db.list_items(
        watchlist_id=watchlist_id,
        alert_severity="critical",
        include_alert_summary=True,
        limit=10,
        offset=0,
    )
    dismissed_rows, dismissed_total = db.list_items(
        watchlist_id=watchlist_id,
        alert_status="dismissed",
        limit=10,
        offset=0,
    )
    rule_rows, rule_total = db.list_items(
        watchlist_id=watchlist_id,
        alert_rule_id=int(critical_rule.id),
        limit=10,
        offset=0,
    )
    severity_sorted, _ = db.list_items(
        watchlist_id=watchlist_id,
        sort="alert_severity_desc",
        include_alert_summary=True,
        limit=10,
        offset=0,
    )
    alert_counts = db.get_item_smart_counts(
        watchlist_id=watchlist_id,
        alert_severity="critical",
    )

    assert alert_total == 1
    assert [int(row.id) for row in alert_rows] == [alerted_item_id]
    assert quiet_total == 1
    assert [int(row.id) for row in quiet_rows] == [quiet_item_id]
    assert critical_total == 1
    assert [int(row.id) for row in critical_rows] == [alerted_item_id]
    assert dismissed_total == 1
    assert [int(row.id) for row in dismissed_rows] == [alerted_item_id]
    assert rule_total == 1
    assert [int(row.id) for row in rule_rows] == [alerted_item_id]
    assert [int(row.id) for row in severity_sorted] == [alerted_item_id, quiet_item_id]
    assert alert_counts["all"] == 1
    assert alert_counts["unread"] == 0

    summary = alert_rows[0].alert_summary
    assert summary == {
        "total": 2,
        "unread": 1,
        "read": 0,
        "dismissed": 1,
        "highest_severity": "critical",
        "latest_alert_id": int(dismissed_alert.id),
        "latest_alert_status": "dismissed",
        "latest_alert_created_at": dismissed_alert.created_at,
        "latest_matched_text": "ransomware",
        "rule_ids": [int(critical_rule.id), int(low_rule.id)],
        "severities": ["critical", "low"],
    }
    assert critical_rows[0].alert_summary["latest_alert_id"] in {
        int(critical_alert.id),
        int(dismissed_alert.id),
    }
