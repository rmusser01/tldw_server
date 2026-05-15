from __future__ import annotations

import json
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


pytestmark = pytest.mark.unit


def _make_db(tmp_path, *, user_id: int = 123) -> WatchlistsDatabase:
    db_path = tmp_path / f"watchlists_content_alerts_{user_id}.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    return WatchlistsDatabase(user_id=user_id, backend=backend)


def _seed_item(
    db: WatchlistsDatabase,
    *,
    label: str = "cti",
    tags: list[str] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    watchlist = db.create_watchlist(
        name=f"{label} Watchlist",
        objective="Track exploitation reports",
        domain="cti_osint",
        priority="high",
        tags=["cti", "exploitation"],
    )
    source = db.create_source(
        name=f"{label} advisory feed",
        url=f"https://example.com/{label}/advisories.xml",
        source_type="rss",
        tags=tags or ["advisory", "cti"],
        watchlist_id=int(watchlist.id),
    )
    job = db.create_job(
        name=f"{label} monitor",
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
        url="https://example.com/cti/cve-2026-1234",
        title="Critical CVE-2026-1234 exploitation observed",
        summary="Multiple sources report active exploitation of CVE-2026-1234.",
        content="Observed exploitation overlaps with ransomware targeting hospitals.",
        published_at="2026-05-15T10:00:00+00:00",
        tags=["cve", "ransomware"],
        status="ingested",
    )
    return watchlist, source, job, run, item


def test_content_alert_rule_crud_and_alert_review_state_round_trip(tmp_path):
    db = _make_db(tmp_path)
    watchlist, source, job, run, item = _seed_item(db)

    rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist.id),
        name="Active CVE exploitation",
        rule_kind="cve",
        match_mode="contains",
        pattern="CVE-2026-1234",
        severity="critical",
        source_constraints={"source_ids": [int(source.id)], "source_tags": ["advisory"]},
        metadata={"descriptor": "active exploitation"},
    )

    listed_rules, rules_total = db.list_content_alert_rules(int(watchlist.id), limit=50, offset=0)
    updated_rule = db.update_content_alert_rule(
        int(rule.id),
        watchlist_id=int(watchlist.id),
        fields={"enabled": False, "severity": "high"},
    )
    enabled_rules, enabled_total = db.list_content_alert_rules(
        int(watchlist.id),
        enabled=True,
        limit=50,
        offset=0,
    )

    assert rules_total == 1
    assert [int(row.id) for row in listed_rules] == [int(rule.id)]
    assert json.loads(rule.source_constraints_json or "{}") == {
        "source_ids": [int(source.id)],
        "source_tags": ["advisory"],
    }
    assert updated_rule.enabled == 0
    assert updated_rule.severity == "high"
    assert enabled_total == 0
    assert enabled_rules == []

    alert = db.create_content_alert(
        watchlist_id=int(watchlist.id),
        rule_id=int(rule.id),
        item_id=int(item.id),
        run_id=int(run.id),
        job_id=int(job.id),
        source_id=int(source.id),
        severity="critical",
        title="Critical CVE-2026-1234 exploitation observed",
        snippet="active exploitation of CVE-2026-1234",
        matched_text="CVE-2026-1234",
        evidence={
            "url": item.url,
            "source_url": source.url,
            "published_at": item.published_at,
        },
        dedupe_key=f"{watchlist.id}:{rule.id}:{item.id}",
    )
    duplicate = db.create_content_alert(
        watchlist_id=int(watchlist.id),
        rule_id=int(rule.id),
        item_id=int(item.id),
        run_id=int(run.id),
        job_id=int(job.id),
        source_id=int(source.id),
        severity="critical",
        title="Critical CVE-2026-1234 exploitation observed",
        snippet="active exploitation of CVE-2026-1234",
        matched_text="CVE-2026-1234",
        evidence={"url": item.url},
        dedupe_key=f"{watchlist.id}:{rule.id}:{item.id}",
    )
    listed_alerts, alerts_total = db.list_content_alerts(int(watchlist.id), limit=50, offset=0)
    read_alert = db.update_content_alert(
        int(alert.id),
        watchlist_id=int(watchlist.id),
        fields={"status": "read"},
    )

    assert duplicate.id == alert.id
    assert alerts_total == 1
    assert [int(row.id) for row in listed_alerts] == [int(alert.id)]
    assert read_alert.status == "read"
    assert read_alert.read_at is not None
    assert json.loads(alert.evidence_json or "{}")["source_url"] == source.url


def test_content_alert_rule_validation_and_watchlist_scoping(tmp_path):
    db = _make_db(tmp_path)
    first, *_ = _seed_item(db, label="first")
    second, *_ = _seed_item(db, label="second")

    with pytest.raises(ValueError, match="content_alert_pattern_required"):
        db.create_content_alert_rule(
            watchlist_id=int(first.id),
            name="Missing pattern",
            rule_kind="keyword",
            pattern=" ",
        )

    with pytest.raises(ValueError, match="invalid_content_alert_regex"):
        db.create_content_alert_rule(
            watchlist_id=int(first.id),
            name="Bad regex",
            rule_kind="regex",
            match_mode="regex",
            pattern="[",
        )

    rule = db.create_content_alert_rule(
        watchlist_id=int(first.id),
        name="Ransomware",
        rule_kind="keyword",
        pattern="ransomware",
        severity="high",
    )

    other_rules, other_total = db.list_content_alert_rules(int(second.id), limit=50, offset=0)

    assert rule.watchlist_id == first.id
    assert other_total == 0
    assert other_rules == []


def test_schema_contains_content_alert_tables(tmp_path):
    db = _make_db(tmp_path)

    rule_columns = {row["name"] for row in db.backend.get_table_info("watchlist_content_alert_rules")}
    alert_columns = {row["name"] for row in db.backend.get_table_info("watchlist_content_alerts")}

    assert {
        "id",
        "user_id",
        "watchlist_id",
        "name",
        "enabled",
        "rule_kind",
        "match_mode",
        "pattern",
        "severity",
        "source_constraints_json",
    }.issubset(rule_columns)
    assert {
        "id",
        "user_id",
        "watchlist_id",
        "rule_id",
        "item_id",
        "run_id",
        "job_id",
        "source_id",
        "status",
        "snippet",
        "matched_text",
        "evidence_json",
        "dedupe_key",
    }.issubset(alert_columns)


class _CapturingPostgresBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self) -> None:
        self.ddl = ""
        self.executed: list[str] = []

    def create_tables(self, ddl: str) -> None:
        self.ddl = ddl

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append(query)

    def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        if table_name == "scrape_jobs":
            return [
                {"name": "wf_schedule_id"},
                {"name": "job_filters_json"},
                {"name": "watchlist_id"},
            ]
        if table_name == "sources":
            return [
                {"name": "defer_until"},
                {"name": "consec_not_modified"},
                {"name": "consec_errors"},
            ]
        if table_name == "scrape_run_items":
            return [{"name": "source_id"}]
        if table_name == "scraped_items":
            return [{"name": "content"}, {"name": "queued_for_briefing"}]
        return []


def test_postgres_schema_text_includes_content_alert_contract():
    backend = _CapturingPostgresBackend()

    WatchlistsDatabase(user_id=123, backend=backend)  # type: ignore[arg-type]

    assert "CREATE TABLE IF NOT EXISTS watchlist_content_alert_rules" in backend.ddl
    assert "CREATE TABLE IF NOT EXISTS watchlist_content_alerts" in backend.ddl
    assert "id BIGSERIAL PRIMARY KEY" in backend.ddl
    assert "watchlist_id BIGINT NOT NULL" in backend.ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_content_alerts_dedupe_key" in backend.ddl
