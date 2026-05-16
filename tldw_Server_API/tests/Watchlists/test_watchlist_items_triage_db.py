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


def test_list_items_search_treats_like_wildcards_literally(tmp_path):
    db = _make_db(tmp_path)
    seeded = _seed_review_queue(db)
    watchlist_id = int(seeded["watchlist"].id)
    job_id = int(seeded["job"].id)
    run_id = int(seeded["run"].id)
    source_id = int(seeded["sources"][0].id)

    literal = db.record_scraped_item(
        run_id=run_id,
        job_id=job_id,
        source_id=source_id,
        media_id=None,
        media_uuid=None,
        url="https://example.com/literal-wildcards",
        title="Literal 100%_match advisory",
        summary="Contains literal SQL wildcard characters.",
        content="literal wildcard content",
        published_at="2026-05-16T08:00:00+00:00",
        tags=["wildcard"],
        status="ingested",
    )
    db.record_scraped_item(
        run_id=run_id,
        job_id=job_id,
        source_id=source_id,
        media_id=None,
        media_uuid=None,
        url="https://example.com/wildcard-near-miss",
        title="Literal 100XXmatch advisory",
        summary="Should not match escaped percent underscore query.",
        content="near miss content",
        published_at="2026-05-16T09:00:00+00:00",
        tags=["wildcard"],
        status="ingested",
    )

    rows, total = db.list_items(watchlist_id=watchlist_id, search="100%_match", limit=10, offset=0)

    assert total == 1
    assert [int(row.id) for row in rows] == [int(literal.id)]


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


def _add_item(
    db: WatchlistsDatabase,
    *,
    watchlist_id: int,
    source_label: str,
    title: str,
    reviewed: bool = False,
) -> int:
    source = db.create_source(
        name=f"{source_label} Feed",
        url=f"https://example.com/{source_label}-{title.lower().replace(' ', '-')}.xml",
        source_type="rss",
        tags=[source_label],
        watchlist_id=watchlist_id,
    )
    job = db.create_job(
        name=f"{source_label} monitor",
        description=None,
        scope_json=json.dumps({"sources": [int(source.id)]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=0,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
        job_filters_json=None,
        watchlist_id=watchlist_id,
    )
    run = db.create_run(int(job.id), status="finished")
    item = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job.id),
        source_id=int(source.id),
        media_id=None,
        media_uuid=None,
        url=f"https://example.com/{source_label}/{title.lower().replace(' ', '-')}",
        title=title,
        summary=f"{title} summary",
        content=f"{title} content",
        published_at="2026-05-15T10:00:00+00:00",
        tags=[source_label],
        status="ingested",
    )
    if reviewed:
        db.update_item_flags(int(item.id), reviewed=True)
    return int(item.id)


def test_batch_update_items_by_ids_stays_watchlist_scoped(tmp_path):
    db = _make_db(tmp_path)
    seeded = _seed_review_queue(db)
    watchlist_id = int(seeded["watchlist"].id)
    older_id = int(seeded["older"].id)
    newer_id = int(seeded["newer"].id)
    other_watchlist = db.create_watchlist(
        name="Other Watchlist",
        objective="Track unrelated news",
        domain="news",
    )
    outsider_id = _add_item(
        db,
        watchlist_id=int(other_watchlist.id),
        source_label="outside",
        title="Outside update",
    )

    result = db.batch_update_items(
        watchlist_id=watchlist_id,
        item_ids=[older_id, newer_id, outsider_id, 999999],
        reviewed=True,
        queued_for_briefing=True,
        limit=10,
    )

    assert result["matched"] == 2
    assert result["changed"] == 2
    assert result["unchanged"] == 0
    assert result["failed"] == 2
    assert result["matched_ids"] == [older_id, newer_id]
    assert result["changed_ids"] == [older_id, newer_id]
    assert result["failed_ids"] == [outsider_id, 999999]
    assert result["capped"] is False
    assert result["exhausted"] is True
    assert bool(db.get_item(older_id).reviewed) is True
    assert bool(db.get_item(newer_id).queued_for_briefing) is True
    assert bool(db.get_item(outsider_id).queued_for_briefing) is False

    no_op = db.batch_update_items(
        watchlist_id=watchlist_id,
        item_ids=[older_id, newer_id],
        reviewed=True,
        queued_for_briefing=True,
        limit=10,
    )

    assert no_op["matched"] == 2
    assert no_op["changed"] == 0
    assert no_op["unchanged"] == 2
    assert no_op["unchanged_ids"] == [older_id, newer_id]


def test_batch_update_items_by_filter_scope_reports_caps(tmp_path):
    db = _make_db(tmp_path)
    watchlist = db.create_watchlist(
        name="Scope batch Watchlist",
        objective="Track unread items",
        domain="general",
    )
    first_id = _add_item(db, watchlist_id=int(watchlist.id), source_label="scope", title="First")
    second_id = _add_item(db, watchlist_id=int(watchlist.id), source_label="scope", title="Second")
    third_id = _add_item(db, watchlist_id=int(watchlist.id), source_label="scope", title="Third")

    result = db.batch_update_items(
        watchlist_id=int(watchlist.id),
        scope={"status": "ingested", "reviewed": False},
        reviewed=True,
        limit=2,
    )

    assert result["matched"] == 2
    assert result["changed"] == 2
    assert result["failed"] == 0
    assert result["capped"] is True
    assert result["exhausted"] is False
    assert result["matched_ids"] == [third_id, second_id]
    assert bool(db.get_item(first_id).reviewed) is False
    assert bool(db.get_item(second_id).reviewed) is True
    assert bool(db.get_item(third_id).reviewed) is True


def test_item_saved_views_are_watchlist_scoped_and_validated(tmp_path):
    db = _make_db(tmp_path)
    seeded = _seed_review_queue(db)
    watchlist_id = int(seeded["watchlist"].id)
    source_id = int(seeded["sources"][0].id)
    other_watchlist = db.create_watchlist(
        name="Other Saved Views",
        objective="Separate saved views",
        domain="news",
    )
    outside_source = db.create_source(
        name="Outside source",
        url="https://example.com/outside-saved-view.xml",
        source_type="rss",
        watchlist_id=int(other_watchlist.id),
    )

    view = db.create_item_saved_view(
        watchlist_id=watchlist_id,
        name="Critical unread",
        filters={"source_id": source_id, "smart_filter": "unread", "alert_severity": "critical"},
        sort="unread_first",
        is_default=True,
    )

    assert view.name == "Critical unread"
    assert view.filters_json == json.dumps(
        {"source_id": source_id, "smart_filter": "unread", "alert_severity": "critical"},
        sort_keys=True,
    )
    assert view.sort == "unread_first"
    assert int(view.is_default) == 1
    assert [int(row.id) for row in db.list_item_saved_views(watchlist_id=watchlist_id)] == [int(view.id)]
    assert db.list_item_saved_views(watchlist_id=int(other_watchlist.id)) == []

    updated = db.update_item_saved_view(
        view_id=int(view.id),
        watchlist_id=watchlist_id,
        fields={"name": "Critical queue", "filters": {"source_id": source_id}, "sort": "created_asc"},
    )

    assert updated.name == "Critical queue"
    assert json.loads(updated.filters_json) == {"source_id": source_id}
    assert updated.sort == "created_asc"

    with pytest.raises(ValueError, match="invalid_item_sort"):
        db.create_item_saved_view(
            watchlist_id=watchlist_id,
            name="Bad sort",
            filters={},
            sort="novelty_desc",
        )
    with pytest.raises(ValueError, match="source_not_in_watchlist"):
        db.create_item_saved_view(
            watchlist_id=watchlist_id,
            name="Bad source",
            filters={"source_id": int(outside_source.id)},
            sort="created_desc",
        )
    with pytest.raises(ValueError, match="invalid_saved_view_filter"):
        db.create_item_saved_view(
            watchlist_id=watchlist_id,
            name="Bad filter",
            filters={"confidence": "high"},
            sort="created_desc",
        )

    assert db.delete_item_saved_view(view_id=int(view.id), watchlist_id=watchlist_id) is True
    assert db.delete_item_saved_view(view_id=int(view.id), watchlist_id=watchlist_id) is False
