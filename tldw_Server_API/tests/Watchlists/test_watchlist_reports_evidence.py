from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import (
    WatchlistOutputEvidenceResponse,
    WatchlistReportEvidenceSnapshot,
    WatchlistReportReadiness,
)
from tldw_Server_API.app.core.Watchlists.report_evidence import (
    build_legacy_live_only_readiness,
    build_report_evidence_snapshot,
    evaluate_report_readiness,
)


pytestmark = pytest.mark.unit


def _row(**fields):
    return SimpleNamespace(**fields)


def _job_row(job_id: int, name: str):
    return _row(id=job_id, name=name, watchlist_id=1)


def _run_row(run_id: int, status: str = "finished"):
    return _row(id=run_id, job_id=1, status=status)


def _item_row(
    item_id: int,
    *,
    source_id: int | None = 11,
    title: str = "Update",
    url: str | None = "https://source.example/update",
    published_at: str | None = "2026-05-15T10:00:00+00:00",
    reviewed: bool = True,
    queued_for_briefing: bool = True,
    tags: list[str] | None = None,
):
    return _row(
        id=item_id,
        run_id=10,
        job_id=1,
        source_id=source_id,
        url=url,
        title=title,
        summary=f"{title} summary",
        published_at=published_at,
        tags=tags or ["watchlist"],
        reviewed=1 if reviewed else 0,
        queued_for_briefing=1 if queued_for_briefing else 0,
        created_at="2026-05-15T10:30:00+00:00",
    )


def _source_row(source_id: int, name: str, url: str | None = None):
    return _row(
        id=source_id,
        name=name,
        url=url or f"https://source{source_id}.example/rss.xml",
        source_type="rss",
    )


def _alert_row(
    alert_id: int,
    severity: str,
    status: str = "unread",
    *,
    rule_id: int = 301,
    rule_name: str = "CVE exploit",
):
    return _row(
        id=alert_id,
        rule_id=rule_id,
        rule_name=rule_name,
        severity=severity,
        status=status,
        title="Critical CVE exploit observed",
        snippet="Active exploitation of CVE-2026-1234",
        matched_text="CVE-2026-1234",
        evidence={"source_url": "https://source11.example/rss.xml"},
        created_at="2026-05-15T11:00:00+00:00",
    )


def _warning_codes(snapshot: dict) -> set[str]:
    return {warning["code"] for warning in snapshot["readiness"]["warnings"]}


def test_report_evidence_snapshot_marks_ready_with_diverse_sources_and_alerts():
    snapshot = build_report_evidence_snapshot(
        watchlist_id=1,
        job=_job_row(1, "Hospital ransomware monitor"),
        run=_run_row(10),
        included_items=[
            _item_row(101, source_id=11, title="Vendor advisory", url="https://a.example/cve"),
            _item_row(102, source_id=12, title="Local report", url="https://b.example/news"),
        ],
        excluded_items=[
            _item_row(103, source_id=11, title="Ignored update", queued_for_briefing=False)
        ],
        sources={
            11: _source_row(11, "Vendor advisory", "https://a.example/rss.xml"),
            12: _source_row(12, "Local news", "https://b.example/rss.xml"),
        },
        alerts={101: [_alert_row(201, "critical")]},
        preset="cti_osint",
        generated_at="2026-05-15T12:00:00+00:00",
    )

    assert snapshot["readiness"]["state"] == "ready"
    assert snapshot["readiness"]["score"] == 100
    assert snapshot["source_summary"]["unique_source_count"] == 2
    assert snapshot["source_summary"]["missing_source_count"] == 0
    assert snapshot["included_count"] == 2
    assert snapshot["excluded_count"] == 1
    assert snapshot["alert_count"] == 1
    assert snapshot["critical_alert_count"] == 1
    assert snapshot["included_items"][0]["alerts"][0]["severity"] == "critical"
    assert snapshot["excluded_items"][0]["reason"] == "not_queued_for_report"
    assert WatchlistReportEvidenceSnapshot.model_validate(snapshot).readiness.state == "ready"
    json.dumps(snapshot)


def test_report_readiness_blocks_empty_report():
    snapshot = build_report_evidence_snapshot(
        watchlist_id=1,
        job=_job_row(1, "Empty monitor"),
        run=_run_row(10),
        included_items=[],
        excluded_items=[],
        sources={},
        alerts={},
        preset="general_research",
        generated_at="2026-05-15T12:00:00+00:00",
    )

    assert snapshot["readiness"]["state"] == "blocked"
    assert snapshot["readiness"]["score"] == 0
    assert _warning_codes(snapshot) == {"no_included_items"}


def test_report_readiness_warns_for_single_source_missing_provenance_and_unreviewed_queue():
    snapshot = build_report_evidence_snapshot(
        watchlist_id=1,
        job=_job_row(1, "Weak evidence monitor"),
        run=_run_row(10),
        included_items=[
            _item_row(
                101,
                source_id=11,
                title="Unreviewed update",
                url=None,
                reviewed=False,
                queued_for_briefing=True,
            )
        ],
        excluded_items=[],
        sources={11: _source_row(11, "Only source")},
        alerts={101: [_alert_row(201, "medium")]},
        preset="general_research",
        generated_at="2026-05-15T12:00:00+00:00",
    )

    assert snapshot["readiness"]["state"] == "warning"
    assert {
        "single_source",
        "missing_source_provenance",
        "unreviewed_queued_items",
    }.issubset(_warning_codes(snapshot))
    readiness = WatchlistReportReadiness.model_validate(evaluate_report_readiness(snapshot))
    assert readiness.state == "warning"
    assert readiness.score < 100


def test_report_readiness_warns_for_cti_without_alert_evidence():
    snapshot = build_report_evidence_snapshot(
        watchlist_id=1,
        job=_job_row(1, "CTI no alert monitor"),
        run=_run_row(10),
        included_items=[
            _item_row(101, source_id=11, title="Advisory", url="https://a.example/cve"),
            _item_row(102, source_id=12, title="Report", url="https://b.example/news"),
        ],
        excluded_items=[],
        sources={
            11: _source_row(11, "Vendor advisory", "https://a.example/rss.xml"),
            12: _source_row(12, "Local news", "https://b.example/rss.xml"),
        },
        alerts={},
        preset="cti_osint",
        generated_at="2026-05-15T12:00:00+00:00",
    )

    assert snapshot["readiness"]["state"] == "warning"
    assert "no_alert_evidence" in _warning_codes(snapshot)


def test_report_readiness_warns_for_stale_news_updates():
    snapshot = build_report_evidence_snapshot(
        watchlist_id=1,
        job=_job_row(1, "News monitor"),
        run=_run_row(10),
        included_items=[
            _item_row(
                101,
                source_id=11,
                title="Old news",
                url="https://news.example/story",
                published_at="2026-04-01T12:00:00+00:00",
            ),
            _item_row(
                102,
                source_id=12,
                title="Older wire",
                url="https://wire.example/story",
                published_at="2026-04-02T12:00:00+00:00",
            ),
        ],
        excluded_items=[],
        sources={
            11: _source_row(11, "News", "https://news.example/rss.xml"),
            12: _source_row(12, "Wire", "https://wire.example/rss.xml"),
        },
        alerts={},
        preset="news_briefing",
        generated_at="2026-05-15T12:00:00+00:00",
    )

    assert snapshot["readiness"]["state"] == "warning"
    assert "stale_updates" in _warning_codes(snapshot)


def test_legacy_live_only_readiness_is_schema_compatible():
    readiness = build_legacy_live_only_readiness()
    response = WatchlistOutputEvidenceResponse.model_validate(
        {
            "output_id": 42,
            "immutable_snapshot": False,
            "snapshot": None,
            "readiness": readiness,
        }
    )

    assert response.readiness.state == "legacy_live_only"
    assert response.readiness.warnings[0].code == "legacy_live_only"
