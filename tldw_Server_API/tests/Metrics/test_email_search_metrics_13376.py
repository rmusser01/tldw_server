"""Real SQLite search metrics must remain registered across telemetry resets."""

import json
import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Metrics import metrics_manager

pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch):
    instance = metrics_manager.MetricsRegistry()
    monkeypatch.setattr(metrics_manager, "_metrics_registry", instance)
    return instance


@pytest.fixture
def db(tmp_path):
    instance = MediaDatabase(db_path=str(tmp_path / "search.db"), client_id="synthetic")
    metadata = {"source_key": "synthetic", "email": {"message_id": "<synthetic@example.test>", "subject": "Synthetic"}}
    media_id = instance.add_media_with_keywords(
        media_type="email",
        content="Synthetic body",
        safe_metadata=json.dumps(metadata),
    )[0]
    instance.upsert_email_message_graph(
        media_id=media_id,
        metadata=metadata,
        body_text="Synthetic body",
        source_key="synthetic",
    )
    yield instance
    instance.close_connection()


def test_search_families_remain_available_after_real_search_and_reset(db, registry):
    db.search_email_messages(query="subject:Synthetic")
    registry.reset()
    assert {
        "email_native_search_requests_total",
        "email_native_search_parse_failures_total",
        "email_native_search_duration_seconds",
        "email_native_search_results_total",
    } <= set(registry.metrics)
    assert registry.metrics["email_native_search_duration_seconds"].unit == "s"
    assert registry.metrics["email_native_search_results_total"].buckets


@pytest.mark.parametrize("query", [None, "subject:Synthetic"])
@pytest.mark.parametrize("include_deleted", [False, True])
def test_real_search_emits_requests_results_and_latency_without_query_data(db, registry, query, include_deleted):
    rows, total = db.search_email_messages(query=query, include_deleted=include_deleted)
    assert len(rows) == total == 1
    labels = {"query_present": "true" if query else "false", "include_deleted": "true" if include_deleted else "false"}
    assert registry.get_cumulative_counter("email_native_search_requests_total", {**labels, "phase": "attempt"}) == 1
    assert registry.get_cumulative_counter("email_native_search_requests_total", {**labels, "phase": "success"}) == 1
    assert registry.get_metric_stats("email_native_search_results_total", labels)["sum"] == 1
    assert registry.get_metric_stats("email_native_search_duration_seconds", labels)["sum"] > 0
    exported = registry.export_prometheus_format()
    assert "subject:Synthetic" not in exported
    assert "synthetic@example.test" not in exported
    assert "Synthetic body" not in exported


def test_real_invalid_query_is_visible_without_error_or_query_labels(db, registry):
    with pytest.raises(InputError):
        db.search_email_messages(query="(private@example.test)")
    labels = {"query_present": "true", "include_deleted": "false"}
    assert registry.get_cumulative_counter("email_native_search_parse_failures_total", labels) == 1
    assert (
        registry.get_cumulative_counter("email_native_search_requests_total", {**labels, "phase": "parse_error"}) == 1
    )
    assert registry.get_metric_stats("email_native_search_duration_seconds", labels)["count"] == 1
    assert "private@example.test" not in registry.export_prometheus_format()


def test_real_database_failure_uses_the_registered_request_labels(db, registry):
    db.execute_query("DROP INDEX idx_email_media_visibility", commit=True)
    with pytest.raises(sqlite3.OperationalError):
        db.search_email_messages(query="subject:Synthetic")
    labels = {"query_present": "true", "include_deleted": "false"}
    assert registry.get_cumulative_counter("email_native_search_requests_total", {**labels, "phase": "error"}) == 1
    assert registry.get_metric_stats("email_native_search_duration_seconds", labels)["count"] == 1
    assert "error_type=" not in registry.export_prometheus_format()
