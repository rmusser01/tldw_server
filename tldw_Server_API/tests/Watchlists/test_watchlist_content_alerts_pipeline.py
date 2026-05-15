from __future__ import annotations

import json
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.Watchlists.content_alerts import evaluate_content_alert_rules_for_item


pytestmark = pytest.mark.unit


def _make_db(tmp_path, *, user_id: int = 123) -> WatchlistsDatabase:
    db_path = tmp_path / f"watchlists_content_alerts_pipeline_{user_id}.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    return WatchlistsDatabase(user_id=user_id, backend=backend)


def _seed_item(
    db: WatchlistsDatabase,
    *,
    watchlist_name: str = "Healthcare CTI",
    source_tags: list[str] | None = None,
    title: str = "CVE-2026-1234 exploitation observed in hospitals",
    summary: str = "Ransomware operators are exploiting CVE-2026-1234.",
    content: str = "The advisory describes active exploitation and emergency mitigation steps.",
):
    watchlist = db.create_watchlist(
        name=watchlist_name,
        objective="Track active exploitation against hospitals",
        domain="cti_osint",
        priority="critical",
        tags=["cti", "healthcare"],
    )
    source = db.create_source(
        name="Advisory feed",
        url="https://example.com/advisories.xml",
        source_type="rss",
        tags=source_tags or ["advisory", "cti"],
        watchlist_id=int(watchlist.id),
    )
    job = db.create_job(
        name="Daily advisory monitor",
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
        watchlist_id=int(watchlist.id),
    )
    run = db.create_run(int(job.id), status="finished")
    item = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job.id),
        source_id=int(source.id),
        media_id=None,
        media_uuid=None,
        url="https://example.com/advisory/cve-2026-1234",
        title=title,
        summary=summary,
        content=content,
        published_at="2026-05-15T10:00:00+00:00",
        tags=["cve", "ransomware"],
        status="ingested",
    )
    return watchlist, source, job, run, item


class _FailingNotifier:
    def notify_or_batch(self, payload: dict[str, Any]) -> str:
        raise RuntimeError("notification sink unavailable")


def test_evaluate_content_alert_rules_creates_evidence_alert_and_notification(tmp_path):
    db = _make_db(tmp_path)
    watchlist, source, job, run, item = _seed_item(db)
    rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist.id),
        name="Active exploitation",
        rule_kind="descriptor",
        match_mode="contains",
        pattern="active exploitation",
        severity="critical",
        source_constraints={"source_tags": ["advisory"]},
    )
    notifications: list[dict[str, Any]] = []

    class _CaptureNotifier:
        def notify_or_batch(self, payload: dict[str, Any]) -> str:
            notifications.append(payload)
            return "logged"

    created = evaluate_content_alert_rules_for_item(
        db,
        watchlist_id=int(watchlist.id),
        item=item,
        notifier=_CaptureNotifier(),
    )
    duplicate = evaluate_content_alert_rules_for_item(
        db,
        watchlist_id=int(watchlist.id),
        item=item,
        notifier=_CaptureNotifier(),
    )
    alerts, total = db.list_content_alerts(int(watchlist.id), limit=50, offset=0)

    assert len(created) == 1
    assert duplicate == []
    assert total == 1
    assert alerts[0].rule_id == rule.id
    assert alerts[0].item_id == item.id
    assert alerts[0].run_id == run.id
    assert alerts[0].job_id == job.id
    assert alerts[0].source_id == source.id
    assert "active exploitation" in alerts[0].snippet.lower()
    assert json.loads(alerts[0].evidence_json or "{}")["url"] == item.url
    assert notifications[0]["type"] == "watchlist_content_alert"
    assert notifications[0]["watchlist_id"] == int(watchlist.id)
    assert notifications[0]["rule_id"] == int(rule.id)
    assert notifications[0]["item_id"] == int(item.id)


def test_evaluate_content_alert_rules_respects_source_constraints_and_disabled_rules(tmp_path):
    db = _make_db(tmp_path)
    watchlist, source, *_rest, item = _seed_item(db, source_tags=["news"])
    matching_source_rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist.id),
        name="Advisory only",
        rule_kind="keyword",
        pattern="ransomware",
        severity="high",
        source_constraints={"source_tags": ["advisory"]},
    )
    disabled_rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist.id),
        name="Disabled ransomware",
        rule_kind="keyword",
        pattern="ransomware",
        severity="high",
    )
    db.update_content_alert_rule(
        int(disabled_rule.id),
        watchlist_id=int(watchlist.id),
        fields={"enabled": False},
    )

    created = evaluate_content_alert_rules_for_item(db, watchlist_id=int(watchlist.id), item=item)
    alerts, total = db.list_content_alerts(int(watchlist.id), limit=50, offset=0)

    assert matching_source_rule.source_constraints_json
    assert created == []
    assert alerts == []
    assert total == 0
    assert source.tags == ["news"]


def test_evaluate_content_alert_rules_ignores_notification_failures(tmp_path):
    db = _make_db(tmp_path)
    watchlist, *_rest, item = _seed_item(db)
    db.create_content_alert_rule(
        watchlist_id=int(watchlist.id),
        name="CVE keyword",
        rule_kind="keyword",
        pattern="CVE-2026-1234",
        severity="critical",
    )

    created = evaluate_content_alert_rules_for_item(
        db,
        watchlist_id=int(watchlist.id),
        item=item,
        notifier=_FailingNotifier(),
    )
    alerts, total = db.list_content_alerts(int(watchlist.id), limit=50, offset=0)

    assert len(created) == 1
    assert total == 1
    assert alerts[0].status == "unread"
