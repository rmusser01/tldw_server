"""Synthetic real-registry and database checks for the email monitoring contract."""

import json

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
    instance = MediaDatabase(db_path=str(tmp_path / "metrics.db"), client_id="synthetic")
    yield instance
    instance.close_connection()


def add_email(db, message_id="<synthetic@example.test>", **kwargs):
    """Write a complete synthetic identity through the real repository."""
    return db.add_media_with_keywords(
        url="synthetic.eml",
        title="Synthetic subject",
        media_type="email",
        content="Synthetic body",
        safe_metadata=json.dumps({"source_key": "synthetic", "email": {"message_id": message_id}}),
        **kwargs,
    )


def test_email_families_are_registered_and_survive_registry_reset(registry):
    required = {
        "email_ingestion_parse_total",
        "email_ingestion_parse_seconds",
        "email_ingestion_persist_total",
        "email_ingestion_persist_seconds",
        "email_ingestion_dedupe_total",
        "email_native_persist_seconds",
    }
    registry.reset()
    assert required <= set(registry.metrics)


def test_real_email_identity_match_emits_once_and_keeps_distinct_body_identity(db, registry):
    first = add_email(db)[0]
    duplicate = add_email(db)[0]
    distinct = add_email(db, "<distinct@example.test>")[0]
    assert duplicate == first != distinct
    assert (
        registry.get_cumulative_counter("email_ingestion_persist_total", {"backend": "sqlite", "outcome": "success"})
        == 3
    )
    assert registry.get_cumulative_counter("email_ingestion_dedupe_total", {"backend": "sqlite"}) == 1
    assert (
        registry.get_metric_stats("email_ingestion_persist_seconds", {"backend": "sqlite", "outcome": "success"})[
            "count"
        ]
        == 3
    )
    assert registry.get_metric_stats("email_ingestion_persist_seconds")["sum"] > 0
    exported = registry.export_prometheus_format()
    assert 'email_ingestion_persist_seconds_count{backend="sqlite",outcome="success"} 3' in exported
    assert "synthetic@example.test" not in exported
    assert "Synthetic body" not in exported
    assert "Synthetic subject" not in exported


def test_validation_failure_is_counted_without_a_committed_message(db, registry):
    with pytest.raises(InputError):
        db.add_media_with_keywords(media_type="email", content=None)
    assert (
        registry.get_cumulative_counter("email_ingestion_persist_total", {"backend": "sqlite", "outcome": "error"}) == 1
    )
    assert (
        registry.get_metric_stats("email_ingestion_persist_seconds", {"backend": "sqlite", "outcome": "error"})["count"]
        == 1
    )
    assert db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 0


def test_transaction_rollback_records_failure_and_no_success(db, registry, monkeypatch):
    def fail_sync(*args, **kwargs):
        raise InputError("Synthetic write failure")

    monkeypatch.setattr(db, "_log_sync_event", fail_sync)
    with pytest.raises(InputError):
        add_email(db)
    assert db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 0
    assert (
        registry.get_cumulative_counter("email_ingestion_persist_total", {"backend": "sqlite", "outcome": "error"}) == 1
    )
    assert registry.get_cumulative_counter_total("email_ingestion_persist_total") == 1


def test_non_email_writes_do_not_emit_email_metrics(db, registry):
    db.add_media_with_keywords(media_type="document", content="Synthetic body")
    assert registry.get_cumulative_counter_total("email_ingestion_persist_total") == 0
    assert registry.get_cumulative_counter_total("email_ingestion_dedupe_total") == 0


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_parse_metrics_collapse_unbounded_labels_and_invalid_durations(registry, duration):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email.email_ingestion_metrics import record_email_parse

    record_email_parse(source_format="private@example.test", outcome="secret header", duration_seconds=duration)
    assert registry.get_cumulative_counter("email_ingestion_parse_total", {"format": "other", "outcome": "error"}) == 1
    assert registry.get_metric_stats("email_ingestion_parse_seconds")["sum"] == 0
    assert "private@example.test" not in registry.export_prometheus_format()
    assert "secret header" not in registry.export_prometheus_format()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_outcome", ["success", "error", "skipped_flag"])
async def test_real_primary_persistence_exports_media_and_native_outcomes(
    tmp_path, registry, monkeypatch, native_outcome
):
    import asyncio
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: native_outcome != "skipped_flag")
    if native_outcome == "error":

        def fail_graph(*args, **kwargs):
            raise DatabaseError("Synthetic graph failure")

        monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", fail_graph)
    result = {
        "status": "Success",
        "content": "Synthetic body",
        "metadata": {"title": "Synthetic subject", "email": {"message_id": "<synthetic@example.test>"}},
    }
    await persistence.persist_doc_item_and_children(
        final_result=result,
        form_data=SimpleNamespace(keywords=[]),
        media_type="email",
        item_input_ref="synthetic.eml",
        processing_filename="synthetic.eml",
        chunk_options=None,
        path_kind="upload",
        db_path=str(tmp_path / "native.db"),
        client_id="synthetic",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )
    assert result["db_id"]
    labels = {"path_kind": "primary", "outcome": native_outcome}
    assert registry.get_cumulative_counter("email_native_persist_total", labels) == 1
    assert registry.get_metric_stats("email_native_persist_seconds", labels)["count"] == 1
    assert (
        registry.get_cumulative_counter("email_ingestion_persist_total", {"backend": "sqlite", "outcome": "success"})
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path_kind", ["archive_child", "attachment_child"])
@pytest.mark.parametrize("graph_fails", [False, True])
async def test_real_child_persistence_records_each_attempt_once(
    tmp_path, registry, monkeypatch, path_kind, graph_fails
):
    import asyncio
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    if graph_fails:

        def fail_graph(*args, **kwargs):
            raise DatabaseError("Synthetic graph failure")

        monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", fail_graph)
    is_archive = path_kind == "archive_child"
    result = {
        "status": "Success",
        "content": "" if is_archive else "Synthetic parent body",
        "metadata": {"email": {"message_id": "<parent@example.test>"}},
        "children": [
            {
                "status": "Success",
                "content": "Identical synthetic child body",
                "metadata": {"filename": f"child-{i}.eml", "email": {"message_id": f"<child-{i}@example.test>"}},
            }
            for i in range(2)
        ],
    }
    await persistence.persist_doc_item_and_children(
        final_result=result,
        form_data=SimpleNamespace(keywords=[], accept_mbox=is_archive, ingest_attachments=not is_archive),
        media_type="email",
        item_input_ref="synthetic.mbox" if is_archive else "synthetic.eml",
        processing_filename="synthetic.mbox" if is_archive else "synthetic.eml",
        chunk_options=None,
        path_kind="upload",
        db_path=str(tmp_path / "children.db"),
        client_id="synthetic",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )
    assert len({child["db_id"] for child in result["child_db_results"]}) == 2
    labels = {"path_kind": path_kind, "outcome": "error" if graph_fails else "success"}
    assert registry.get_cumulative_counter("email_native_persist_total", labels) == 2
    assert registry.get_metric_stats("email_native_persist_seconds", labels)["count"] == 2
    assert registry.get_cumulative_counter_total("email_ingestion_persist_total") == (2 if is_archive else 3)
    assert registry.get_cumulative_counter_total("email_ingestion_dedupe_total") == 0


def test_concurrent_insert_recheck_records_one_match(db, registry, monkeypatch):
    existing = add_email(db)[0]
    registry.reset()
    real_fetch = db._fetchone_with_connection
    lookup_count = 0

    def first_lookup_misses(connection, query, params=None):
        nonlocal lookup_count
        if "FROM Media m WHERE m.url = ?" in query:
            lookup_count += 1
            if lookup_count == 1:
                return None
        return real_fetch(connection, query, params)

    monkeypatch.setattr(db, "_fetchone_with_connection", first_lookup_misses)
    assert add_email(db)[0] == existing
    assert lookup_count == 2
    assert registry.get_cumulative_counter_total("email_ingestion_dedupe_total") == 1
    assert registry.get_cumulative_counter_total("email_ingestion_persist_total") == 1


def test_counter_failure_keeps_real_write_and_duration_observation(db, registry, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("secret exception content must not be logged")

    monkeypatch.setattr(registry, "increment", unavailable)
    media_id = add_email(db)[0]
    assert db.get_media_by_id(media_id)["content"] == "Synthetic body"
    assert registry.get_metric_stats("email_ingestion_persist_seconds")["count"] == 1


def test_registry_failure_preserves_real_write_and_original_operation_error(db, monkeypatch):
    def unavailable():
        raise RuntimeError("Synthetic registry unavailable")

    monkeypatch.setattr(metrics_manager, "get_metrics_registry", unavailable)
    media_id = add_email(db)[0]
    assert db.get_media_by_id(media_id)["content"] == "Synthetic body"
    with pytest.raises(InputError, match="Content cannot be None"):
        db.add_media_with_keywords(media_type="email", content=None)


@pytest.mark.parametrize("outcome", ["parsed", "error"])
def test_parse_outcome_preserves_elapsed_seconds_in_monitoring(registry, outcome):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email.email_ingestion_metrics import record_email_parse

    record_email_parse(source_format="eml", outcome=outcome, duration_seconds=0.25)
    labels = {"format": "eml", "outcome": outcome}
    assert registry.get_cumulative_counter("email_ingestion_parse_total", labels) == 1
    assert registry.get_metric_stats("email_ingestion_parse_seconds", labels)["sum"] == 0.25
    exported = registry.export_prometheus_format()
    assert f'email_ingestion_parse_seconds_sum{{format="eml",outcome="{outcome}"}} 0.25' in exported
