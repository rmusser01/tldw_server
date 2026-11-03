"""Transport-only offload contract, independent of future incident reporting."""

import asyncio
import contextvars
import threading
import time

import pytest

from tldw_Server_API.app.core.Moderation.execution_models import ExecutionScope

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def scope():
    return ExecutionScope(time.monotonic() + 2, threading.Event(), "async")


async def test_callable_timeout_error_is_not_misreported_as_transport_deadline():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    shared = scope()

    def call():
        raise TimeoutError("trusted override timeout")

    with pytest.raises(TimeoutError, match="trusted override timeout"):
        await _run_moderation_offload(call, scope=shared)
    assert not shared.cancelled.is_set()


async def test_async_heartbeat_runs_while_worker_stalls():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    release = threading.Event()
    task = asyncio.create_task(_run_moderation_offload(lambda: release.wait(1), scope=scope()))
    try:
        for _ in range(5):
            await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
        assert await task


async def test_cancelled_offload_signals_owner():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    shared, entered, finished = scope(), threading.Event(), threading.Event()

    def call():
        entered.set()
        shared.cancelled.wait(2)
        finished.set()

    task = asyncio.create_task(_run_moderation_offload(call, scope=shared))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert shared.cancelled.is_set()
    assert await asyncio.to_thread(finished.wait, 1)


async def test_context_does_not_leak_between_concurrent_calls():
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    current = contextvars.ContextVar("independent")

    async def one(value):
        token = current.set(value)
        try:
            return await _run_moderation_offload(current.get, scope=scope())
        finally:
            current.reset(token)

    assert await asyncio.gather(one("first"), one("second")) == ["first", "second"]


async def test_cancelled_offloads_retain_four_thread_slots_until_finished():
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload

    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]

    def call(index):
        entered[index].set()
        release.wait(2)

    tasks = [asyncio.create_task(_run_moderation_offload(lambda i=i: call(i), scope=scope())) for i in range(4)]
    try:
        while not all(event.is_set() for event in entered):
            await asyncio.sleep(0.001)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            await _run_moderation_offload(lambda: None, scope=scope())
    finally:
        release.set()
        await asyncio.sleep(0.05)


async def test_quarantined_injected_runner_keeps_offload_admission_owned(incident_store):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import _run_moderation_offload
    from tldw_Server_API.tests.unit.test_moderation_execution_runner import Child, request, runner

    children = [Child(stalled=True, unreapable=True) for _ in range(4)]
    owners = [runner(lambda *a, child=child, **k: child) for child in children]
    scopes = [scope() for _ in range(4)]
    for shared in scopes:
        shared.deadline = time.monotonic() + 0.05
    tasks = [
        asyncio.create_task(
            _run_moderation_offload(
                lambda owner=owner, shared=shared: owner.execute(request(), shared),
                scope=shared,
            )
        )
        for owner, shared in zip(owners, scopes, strict=True)
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0.65)
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            await _run_moderation_offload(lambda: None, scope=scope())
    finally:
        for child in children:
            child.returncode = -9
        for owner in owners:
            owner.close()
        await asyncio.sleep(0.1)


@pytest.mark.timeout(30)
async def test_real_worker_heartbeat_in_independent_containment(tmp_path):
    import sys
    from pathlib import Path

    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe
    from tldw_Server_API.tests.unit.test_moderation_process_harness import (
        test_outer_harness_timeout_kills_probe_and_grandchild,
        test_probe_hard_exit_and_invoker_disconnect_still_clean_tree,
    )

    test_outer_harness_timeout_kills_probe_and_grandchild(tmp_path)
    test_probe_hard_exit_and_invoker_disconnect_still_clean_tree(tmp_path)
    probe = Path(__file__).resolve().parents[1] / "helpers" / "moderation_supervisor_probe.py"
    result = run_isolated_probe([sys.executable, "-I", "-S", str(probe), "heartbeat"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(b"OK heartbeat")


@pytest.fixture
def incident_store(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store

    path = tmp_path / "review.db"
    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(path))
    prepare_moderation_incident_store()
    return ModerationReviewStore(path)


async def test_async_admission_rejection_records_incident_without_invoking_callable(incident_store, monkeypatch):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation import async_execution as execution
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata

    monkeypatch.setattr(execution, "_active", 4)
    entered = []
    with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
        await execution.run_moderation_async(
            lambda: entered.append(True), metadata=FailureMetadata("evaluate", None, 10)
        )
    assert not entered
    assert incident_store.list_items()["total"] == 1


async def test_oversized_input_is_reported_inside_admission_boundary(incident_store):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata

    entered = []
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        await run_moderation_async(
            lambda: entered.append(True), metadata=FailureMetadata("redact", "output", 1_000_001)
        )
    assert not entered
    assert incident_store.list_items()["items"][0]["context"]["reason"] == "size_exceeded"


async def test_outer_and_sync_reporter_share_claim_and_occurrence_metric(incident_store, monkeypatch):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation import execution_incidents as incidents
    from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata, execution_scope

    metrics = []
    monkeypatch.setattr(incidents, "log_counter", lambda name, **kw: metrics.append(name))

    def service():
        with execution_scope() as shared:
            error = ModerationExecutionError("worker_crashed")
            incidents.report_execution_failure(error, operation="evaluate", phase=None, scope=shared, input_chars=1)
            raise error

    with pytest.raises(ModerationExecutionError):
        await run_moderation_async(service, metadata=FailureMetadata("evaluate", None, 1))
    assert metrics.count("moderation_execution_failures_total") == 1
    assert incident_store.list_items()["total"] == 1


async def test_transport_timeout_reports_even_with_scope_cancelled(incident_store):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata, execution_scope

    release = threading.Event()
    try:
        with execution_scope(timeout_s=0.03) as shared:
            with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
                await run_moderation_async(lambda: release.wait(1), metadata=FailureMetadata("evaluate", None, 1))
            assert shared.cancelled.is_set()
        assert incident_store.list_items()["total"] == 1
    finally:
        release.set()
        await asyncio.sleep(0.03)


async def test_normal_disconnect_is_not_reported(incident_store):
    from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata, execution_scope

    entered = threading.Event()

    def service():
        with execution_scope() as shared:
            entered.set()
            shared.cancelled.wait(1)

    task = asyncio.create_task(run_moderation_async(service, metadata=FailureMetadata("evaluate", None, 1)))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert incident_store.list_items()["total"] == 0
    await asyncio.sleep(0.03)


@pytest.mark.parametrize("kill_fails", [False, True])
@pytest.mark.parametrize("batch", [False, True])
async def test_cancelled_unconfirmed_cleanup_reports_once_and_keeps_ownership(incident_store, monkeypatch, kill_fails, batch):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation import async_execution as execution
    from tldw_Server_API.app.core.Moderation import execution_incidents as incidents
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata, execution_scope
    from tldw_Server_API.tests.unit.test_moderation_execution_runner import Child, request, runner

    entered, publish = threading.Event(), threading.Event()
    child = Child(stalled=True, unreapable=True)
    metrics, io_threads = [], []
    persist = incidents._persist_failure

    def record(*args, **kwargs):
        io_threads.append(threading.current_thread().name)
        persist(*args, **kwargs)

    def kill():
        child.killed.set()
        if kill_fails:
            raise PermissionError("SECRET_KILL_EXCEPTION")

    def factory(*args, **kwargs):
        entered.set()
        assert publish.wait(2)
        return child

    child.kill = kill
    owner = runner(factory)
    submitted = request()
    operation, chars = "custom_substitute", 5
    if batch:
        from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_redaction_batch_request
        from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
        from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits

        submitted = snapshot_redaction_batch_request(
            ["hello", "world"], ModerationPolicy(), None, EvaluationLimits(100, 0, 1000, None), submitted.bounds
        )
        operation, chars = "redact_many", 10
    monkeypatch.setattr(incidents, "_persist_failure", record)
    monkeypatch.setattr(incidents, "log_counter", lambda name, **kw: metrics.append((name, kw)))
    with execution_scope() as shared:
        task = asyncio.create_task(execution.run_moderation_async(
            lambda: owner.execute(submitted, shared), metadata=FailureMetadata(operation, None, chars)
        ))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        publish.set()
        async with asyncio.timeout(1.5):
            while incident_store.list_items()["total"] == 0:
                await asyncio.sleep(0.005)
        [item] = incident_store.list_items()["items"]
        assert item["context"]["reason"] == "cleanup_failed"
        assert item["context"]["operation"] == operation
        assert item["context"]["input_chars"] == chars
        assert item["excerpt"] == incidents.ENGINE_FAILURE_EXCERPT
        assert "SECRET" not in str(item) + str(metrics)
        assert len(owner._entries) == 1
        assert execution._active == 1
        assert io_threads == ["moderation-incident"]
        assert len([m for m in metrics if m[0] == "moderation_execution_failures_total"]) == 1
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            owner.execute(request(), scope())
        await asyncio.sleep(0.1)
        assert incident_store.list_items()["total"] == 1
    finally:
        publish.set()
        child.returncode = -9
        owner.close()
        async with asyncio.timeout(2):
            while execution._active or (incidents._report_thread and incidents._report_thread.is_alive()):
                await asyncio.sleep(0.005)
    assert not owner._entries


async def test_cancelled_runner_confirmed_cleanup_is_quiet(incident_store):
    from tldw_Server_API.app.core.Moderation import async_execution as execution
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata, execution_scope
    from tldw_Server_API.tests.unit.test_moderation_execution_runner import Child, request, runner

    entered = threading.Event()
    child = Child(stalled=True)

    def factory(*args, **kwargs):
        entered.set()
        return child

    owner = runner(factory)
    with execution_scope() as shared:
        task = asyncio.create_task(execution.run_moderation_async(
            lambda: owner.execute(request(), shared), metadata=FailureMetadata("custom_substitute", None, 5)
        ))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(2):
            while execution._active:
                await asyncio.sleep(0.005)
        assert not owner._entries
        assert child.killed.is_set()
        assert incident_store.list_items()["total"] == 0
    finally:
        owner.close()


@pytest.mark.parametrize("cancel_at", ["cleanup_started", "quarantined", None], ids=["during-cleanup", "after-quarantine", "not-cancelled"])
@pytest.mark.parametrize("primary_claimed", [False, True], ids=["unclaimed", "primary-claimed"])
async def test_cleanup_owner_reports_independently_of_cancellation_order(
    incident_store, monkeypatch, cancel_at, primary_claimed,
):
    import io
    import json

    from loguru import logger

    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Moderation import async_execution as execution
    from tldw_Server_API.app.core.Moderation import execution_incidents as incidents
    from tldw_Server_API.app.core.Moderation.execution_models import ExecutionBounds, FailureMetadata, execution_scope
    from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_redaction_batch_request
    from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
    from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits
    from tldw_Server_API.tests.unit.test_moderation_execution_runner import Child, request, runner

    cleanup_started, runner_returned = threading.Event(), threading.Event()
    child = Child(stalled=True, unreapable=True)
    child.stdout = io.BytesIO(b"")  # EOF raises a typed error before cleanup/cancellation.
    submitted = snapshot_redaction_batch_request(
        ["PRIVATE_ALPHA", "PRIVATE_BETA"], ModerationPolicy(), "output",
        EvaluationLimits(100, 0, 1000, None), ExecutionBounds(),
    )
    metadata = FailureMetadata("redact_many", "output", 25)
    messages, metrics, io_threads, underlying_errors = [], [], [], []
    persist = incidents._persist_failure

    def record(*args, **kwargs):
        io_threads.append(threading.current_thread().name)
        persist(*args, **kwargs)

    def kill():
        cleanup_started.set()
        raise PermissionError("PRIVATE_KILL_EXCEPTION")

    child.kill = kill
    owner = runner(lambda *a, **k: child)
    monkeypatch.setattr(incidents, "_persist_failure", record)
    monkeypatch.setattr(incidents, "log_counter", lambda name, **kw: metrics.append((name, kw)))
    sink = logger.add(lambda message: messages.append(str(message)))

    def run():
        if primary_claimed:
            incidents.report_execution_failure(
                ModerationExecutionError("invalid_response"), operation="redact_many", phase="output",
                scope=shared, input_chars=25,
            )
        try:
            return owner.execute(submitted, shared)
        except ModerationExecutionError as error:
            underlying_errors.append(error.reason)
            raise
        finally:
            runner_returned.set()

    with execution_scope() as shared:
        task = asyncio.create_task(execution.run_moderation_async(run, metadata=metadata))
    try:
        async with asyncio.timeout(1.5):
            while not cleanup_started.is_set():
                await asyncio.sleep(0.001)
        with owner._lock:
            [entry] = owner._entries
            assert entry.reason == "invalid_response"
        if cancel_at == "quarantined":
            async with asyncio.timeout(1.5):
                while not runner_returned.is_set():
                    await asyncio.sleep(0.001)
            assert entry.quarantined and entry.owner_done
            assert execution._active == 1 and not task.done()
        if cancel_at is not None:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        async with asyncio.timeout(1.25):
            while not runner_returned.is_set() or incident_store.list_items()["total"] == 0:
                await asyncio.sleep(0.005)
        [item] = incident_store.list_items()["items"]
        assert item["context"]["reason"] == ("invalid_response" if primary_claimed else "cleanup_failed")
        assert item["context"]["operation"] == "redact_many"
        assert item["context"]["input_chars"] == 25
        assert item["phase"] == "output"
        assert item["excerpt"] == incidents.ENGINE_FAILURE_EXCERPT
        assert io_threads == (["moderation-offload"] if primary_claimed else ["moderation-incident"])
        assert len([m for m in metrics if m[0] == "moderation_execution_failures_total"]) == 1
        assert entry.quarantined and len(owner._entries) == 1 and execution._active == 1
        assert shared.cancelled.is_set() is (cancel_at is not None)
        with pytest.raises(ModerationExecutionError, match="capacity_exhausted"):
            owner.execute(request(), scope())
        assert "PRIVATE" not in json.dumps([item, metrics, messages])
    finally:
        child.returncode = -9
        owner.close()
        outcomes = await asyncio.gather(task, return_exceptions=True)
        async with asyncio.timeout(2):
            while execution._active or (incidents._report_thread and incidents._report_thread.is_alive()):
                await asyncio.sleep(0.005)
        logger.remove(sink)
    assert not owner._entries
    assert underlying_errors == ["invalid_response"]
    if cancel_at is None:
        assert isinstance(outcomes[0], ModerationExecutionError) and outcomes[0].reason == "invalid_response"
    else:
        assert isinstance(outcomes[0], asyncio.CancelledError)
    assert incident_store.list_items()["total"] == 1


async def test_closed_loop_delivery_retires_offload_without_warning(monkeypatch):
    import warnings
    from pathlib import Path

    from tldw_Server_API.app.core.Moderation import async_execution as execution

    entered, release, delivered = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.new_event_loop()
    schedule = loop.call_soon_threadsafe

    def delivery(*args):
        delivered.set()
        return schedule(*args)

    monkeypatch.setattr(loop, "call_soon_threadsafe", delivery)

    def call():
        entered.set()
        release.wait(2)

    async def start():
        task = asyncio.create_task(execution._run_moderation_offload(call, scope=scope()))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        await asyncio.to_thread(loop.run_until_complete, start())
        loop.close()
        release.set()
        assert await asyncio.to_thread(delivered.wait, 1)
        async with asyncio.timeout(2):
            while execution._active:
                await asyncio.sleep(0.001)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            compile(Path(execution.__file__).read_text(), execution.__file__, "exec")
    finally:
        release.set()
        if not loop.is_closed():
            loop.close()
