"""Operational incidents contain metadata, never failed content."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation import execution_incidents as incidents
from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata, execution_scope

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "review.db"
    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(path))
    monkeypatch.setenv("MODERATION_REVIEW_CAPTURE_ENABLED", "false")
    incidents.prepare_moderation_incident_store()
    return ModerationReviewStore(path)


def report(phase=None, reason="deadline_exceeded"):
    with execution_scope() as scope:
        incidents.report_execution_failure(
            ModerationExecutionError(reason),
            operation="evaluate",
            phase=phase,
            scope=scope,
            input_chars=12,
        )


def test_failure_excerpt_is_constant():
    assert incidents.ENGINE_FAILURE_EXCERPT == "[Moderation execution failed; content withheld]"


def test_capture_disabled_still_records_incident(store):
    report()
    [item] = store.list_items()["items"]
    assert (item["category"], item["source_type"], item["status"], item["severity"], item["recommended_action"]) == (
        "moderation_engine_failure",
        "moderation_engine",
        "needs_review",
        "high",
        "escalate",
    )
    assert item["matches"] == []
    assert item["effective_policy"] == {}
    assert item["excerpt"] == incidents.ENGINE_FAILURE_EXCERPT


def test_unknown_phase_creates_one_review_item_and_one_decision_history(store):
    report("legacy-unknown")
    [item] = store.list_items()["items"]
    assert item["phase"] == "unspecified"
    decision = store.record_decision(item["id"], action="escalate", decided_by="ops")
    assert len(store.get_item(item["id"], include_history=True)["decision_history"]) == 1
    assert {event["item_id"] for event in store.list_audit()["events"]} == {item["id"]}
    store.undo_decision(item["id"], undo_token=decision["undo_token"], actor_id="ops")
    assert store.get_item(item["id"])["status"] == "needs_review"


def test_concurrent_failures_deduplicate_in_same_five_minute_bucket(store, monkeypatch):
    monkeypatch.setattr(incidents.time, "time", lambda: 600)
    metrics = []
    monkeypatch.setattr(incidents, "log_counter", lambda name, **kw: metrics.append((name, kw)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: report(), range(16)))
    assert store.list_items()["total"] == 1
    assert len([name for name, _ in metrics if name == "moderation_execution_failures_total"]) == 16
    assert len(store.list_audit()["events"]) == 1


def test_new_bucket_records_new_incident(store, monkeypatch):
    monkeypatch.setattr(incidents.time, "time", lambda: 899)
    report()
    monkeypatch.setattr(incidents.time, "time", lambda: 900)
    report()
    assert store.list_items()["total"] == 2


def test_outer_and_service_failure_report_once(store):
    with execution_scope() as scope:
        error = ModerationExecutionError("worker_crashed")
        for _ in range(2):
            incidents.report_execution_failure(error, operation="evaluate", phase=None, scope=scope, input_chars=10)
    assert store.list_items()["total"] == 1


def test_store_locked_or_unavailable_preserves_original_block(store, monkeypatch):
    import sqlite3

    with sqlite3.connect(store.db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        report()
        assert time.monotonic() - started < 0.8
    assert store.list_items()["total"] == 0
    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(store.db_path / "unavailable"))
    report()


def test_failure_sentinels_never_enter_logs_metrics_or_review(store, monkeypatch):
    messages, metrics = [], []
    sink = logger.add(lambda message: messages.append(str(message)))
    monkeypatch.setattr(incidents, "log_counter", lambda name, **kw: metrics.append((name, kw)))
    try:
        report("SECRET_CONTENT_PATTERN_REPLACEMENT_STDERR")
        with execution_scope() as scope:
            scope.correlation_id = "SECRET_EXCEPTION"
            incidents.report_execution_failure(
                ModerationExecutionError("invalid_response"),
                operation="redact",
                phase=None,
                scope=scope,
                input_chars=10,
            )
    finally:
        logger.remove(sink)
    serialized = json.dumps([messages, metrics, store.list_items()])
    assert "SECRET" not in serialized
    assert all("correlation_id" not in labels["labels"] for _, labels in metrics)


@pytest.mark.asyncio
async def test_saturated_reporting_never_blocks_event_loop_or_queues_work(store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = incidents._persist_failure

    def stalled(*args, **kwargs):
        entered.set()
        release.wait(3)
        original(*args, **kwargs)

    monkeypatch.setattr(incidents, "_persist_failure", stalled)
    metadata = FailureMetadata("evaluate", None, 1)
    with execution_scope() as first:
        task = asyncio.create_task(
            incidents.report_execution_failure_async(
                ModerationExecutionError("capacity_exhausted"), metadata=metadata, scope=first
            )
        )
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        started = time.monotonic()
        with execution_scope() as second:
            await incidents.report_execution_failure_async(
                ModerationExecutionError("capacity_exhausted"), metadata=metadata, scope=second
            )
        assert time.monotonic() - started < 0.1
        beats = 0
        while not task.done():
            beats += 1
            await asyncio.sleep(0.01)
        assert beats > 5
        assert incidents._report_thread is not None and incidents._report_thread.is_alive()
        assert time.monotonic() - started < 0.8
    finally:
        release.set()
        await task
        while incidents._report_thread is not None and incidents._report_thread.is_alive():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_reporting_cancel_and_shutdown_retain_live_slot(store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(incidents, "_closed", False)
    calls = []

    def stalled(*args, **kwargs):
        calls.append(True)
        entered.set()
        release.wait(3)

    monkeypatch.setattr(incidents, "_persist_failure", stalled)
    metadata = FailureMetadata("redact", None, 1)
    with execution_scope() as scope:
        task = asyncio.create_task(
            incidents.report_execution_failure_async(
                ModerationExecutionError("size_exceeded"), metadata=metadata, scope=scope
            )
        )
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        owner = incidents._report_thread
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        started = time.monotonic()
        await incidents.close_moderation_incident_reporting()
        assert time.monotonic() - started < 0.8
        assert incidents._report_thread is owner and owner.is_alive()
        with execution_scope() as scope:
            await incidents.report_execution_failure_async(
                ModerationExecutionError("size_exceeded"), metadata=metadata, scope=scope
            )
        assert calls == [True]
    finally:
        release.set()
        while incidents._report_thread is not None and incidents._report_thread.is_alive():
            await asyncio.sleep(0.001)


def test_unknown_phase_has_no_false_attribution(store):
    report(None)
    assert [item["phase"] for item in store.list_items()["items"]] == ["unspecified"]


@pytest.mark.asyncio
async def test_reporting_slot_is_not_released_before_thread_exits(store, monkeypatch):
    tail, in_tail = threading.Event(), threading.Event()
    original_thread = threading.Thread
    calls = []

    class TailThread(original_thread):
        def run(self):
            super().run()
            in_tail.set()
            tail.wait(3)

    monkeypatch.setattr(incidents.threading, "Thread", TailThread)
    monkeypatch.setattr(incidents, "_persist_failure", lambda *a, **k: calls.append(True))
    metadata = FailureMetadata("evaluate", None, 1)
    with execution_scope() as first:
        task = asyncio.create_task(
            incidents.report_execution_failure_async(
                ModerationExecutionError("worker_crashed"), metadata=metadata, scope=first
            )
        )
    try:
        while not in_tail.is_set():
            await asyncio.sleep(0.001)
        with execution_scope() as second:
            await incidents.report_execution_failure_async(
                ModerationExecutionError("worker_crashed"), metadata=metadata, scope=second
            )
        assert calls == [True]
    finally:
        tail.set()
        await task


def test_lifecycle_prepares_isolated_schema_before_traffic(tmp_path, monkeypatch):
    from tldw_Server_API.app.services.startup_moderation_workers import provide_moderation_worker_specs

    path = tmp_path / "lifecycle.db"
    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(path))
    specs = provide_moderation_worker_specs(None)
    assert path.exists()
    assert specs[0].name == "moderation_execution"
    report()
    assert ModerationReviewStore(path).list_items()["total"] == 1


def test_reporter_failure_signal_is_sanitized_rate_limited_even_if_metrics_fail(store, monkeypatch):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="ERROR")
    monkeypatch.setattr(incidents, "_last_error", float("-inf"))

    def fail(*args, **kwargs):
        raise OSError("SECRET_STORE_STDERR")

    monkeypatch.setattr(incidents, "_persist_failure", fail)
    monkeypatch.setattr(incidents, "log_counter", fail)
    try:
        report()
        report()
    finally:
        logger.remove(sink)
    assert len(messages) == 1
    assert "enforcement remains blocked" in messages[0]
    assert "SECRET" not in messages[0]


@pytest.mark.asyncio
async def test_unavailable_reporter_preserves_original_async_block(store, monkeypatch):
    from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async

    error = ModerationExecutionError("invalid_response")

    def call():
        raise error

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(store.db_path / "not-a-directory"))
    with pytest.raises(ModerationExecutionError) as caught:
        await run_moderation_async(call, metadata=FailureMetadata("evaluate", None, 1))
    assert caught.value is error
