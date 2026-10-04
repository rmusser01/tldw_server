"""Real worker transitions/publication remain responsive and drain on cancel."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetGenerationRequest, VNAssetPackCreate
from tldw_Server_API.app.core.DB_Management import VNAssetPacks_DB
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.jobs import vn_asset_generation_jobs_queue
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["transaction", "write_lock"])
@pytest.mark.parametrize("cancel_count", [0, 2])
async def test_legacy_setup_failure_before_acquisition_preserves_same_key_v0_sibling(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    boundary: str, cancel_count: int,
) -> None:
    """A delivery that never acquired display cannot consume an executing V0 sibling."""
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    legacy = repo.create_batch(
        pack_id=handoff.payload["pack_id"], requested_by_user_id=1, status="enqueued",
        total_slots=1, total_variants=2, planned_count=2,
        options={"slot_ids": [handoff.payload["slot_id"]], "variant_count": 2},
    )
    payload = {**handoff.payload, "batch_id": legacy["id"]}
    sibling_entered, sibling_release, setup_entered, setup_release = (threading.Event() for _ in range(4))
    provider_calls: list[Any] = []

    class BlockingAdapter(FakeImageAdapter):
        """Keep the first real legacy worker inside its native provider call."""

        def generate(self, request: Any) -> Any:
            """Record admission before blocking, then return the normal image result."""
            provider_calls.append(request)
            sibling_entered.set()
            assert sibling_release.wait(8), "legacy sibling provider release timed out"
            return super().generate(request)

    adapter, rejected_adapter = BlockingAdapter(), FakeImageAdapter()
    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    sibling_task = asyncio.create_task(worker.handle_generate_variant(payload))
    rejected_task: asyncio.Task[Any] | None = None
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    cleanup_calls: list[dict[str, Any]] = []
    begin, finish = repo.begin_inline_legacy_display, repo.finish_legacy_display
    transaction, lock_variant = repo.db.transaction, VNAssetPacks_DB._lock_variant

    def observe_begin(*args: Any, **kwargs: Any) -> None:
        """Identify the rejecting setup thread without replacing repository acquisition."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        begin(*args, **kwargs)

    def fail_before_acquisition() -> None:
        """Block before marker acquisition so cancellation must drain the native failure."""
        setup_entered.set()
        assert setup_release.wait(4), "pre-acquisition setup release timed out"
        raise OSError("native pre-acquisition failure")

    @contextmanager
    def checked_transaction(*args: Any, **kwargs: Any) -> Any:
        """Reject only the second delivery before entering its real transaction."""
        if boundary == "transaction" and observed and threading.current_thread() is observed[0][0]:
            fail_before_acquisition()
        with transaction(*args, **kwargs) as connection:
            yield connection

    def checked_lock(*args: Any, **kwargs: Any) -> None:
        """Reject only the second delivery before taking its real write lock."""
        if boundary == "write_lock" and observed and threading.current_thread() is observed[0][0]:
            fail_before_acquisition()
        lock_variant(*args, **kwargs)

    def observe_finish(*args: Any, **kwargs: Any) -> None:
        """Retain real reconciliation while detecting cleanup by a non-owning delivery."""
        cleanup_calls.append(kwargs)
        finish(*args, **kwargs)

    try:
        assert await asyncio.to_thread(sibling_entered.wait, 2)
        before_batch = repo.get_batch(legacy["id"])
        before_slot = repo.get_slot(payload["slot_id"])
        before_v1 = repo.get_variant_outcome(
            handoff.payload["batch_id"], payload["slot_id"], 0,
        )
        assert before_slot["status"] == "generating"
        worker.image_registry = FakeImageRegistry(rejected_adapter)
        monkeypatch.setattr(repo, "begin_inline_legacy_display", observe_begin)
        monkeypatch.setattr(repo.db, "transaction", checked_transaction)
        monkeypatch.setattr(VNAssetPacks_DB, "_lock_variant", checked_lock)
        monkeypatch.setattr(repo, "finish_legacy_display", observe_finish)
        rejected_task = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 1}))
        assert await asyncio.to_thread(setup_entered.wait, 2)
        for _ in range(cancel_count):
            rejected_task.cancel()
            await asyncio.sleep(0)
            assert not rejected_task.done()
        setup_release.set()
        expected_error = asyncio.CancelledError if cancel_count else OSError
        with pytest.raises(expected_error):
            await rejected_task
        assert repo.get_slot(payload["slot_id"])["status"] == "generating"
        assert cleanup_calls == []
        assert repo.get_batch(legacy["id"]) == before_batch
        assert repo.get_variant_outcome(legacy["id"], payload["slot_id"], 1) is None
        assert repo.get_variant_outcome(handoff.payload["batch_id"], payload["slot_id"], 0) == before_v1
        assert len(provider_calls) == 1
        assert rejected_adapter.requests == []
        assert storage.records == {}
        assert not sibling_task.done()
        assert len(observed) == 1
        thread, connection = observed[0]
        assert thread is not threading.current_thread()
        assert not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        sibling_release.set()
        assert (await sibling_task)["item_id"] > 0
        assert len(cleanup_calls) == 1
        current = repo.get_batch(legacy["id"])
        assert tuple(current[key] for key in ("completed_count", "failed_count", "cancelled_count")) == (1, 0, 0)
    finally:
        setup_release.set()
        sibling_release.set()
        await asyncio.gather(*(task for task in (rejected_task, sibling_task) if task is not None), return_exceptions=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_setup_native_failure_after_acquisition_cleans_display(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An acquired marker still needs cleanup when native setup fails without cancellation."""
    repo = handoff.service.repo
    legacy = repo.create_batch(
        pack_id=handoff.payload["pack_id"], requested_by_user_id=1, status="enqueued",
        total_slots=1, total_variants=1, planned_count=1,
    )
    payload = {**handoff.payload, "batch_id": legacy["id"]}
    adapter, storage = FakeImageAdapter(), ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    begin, finish = repo.begin_inline_legacy_display, repo.finish_legacy_display
    cleanup_calls: list[dict[str, Any]] = []

    def acquired_then_failed(*args: Any, **kwargs: Any) -> None:
        """Raise only after the repository has incremented this delivery's marker."""
        begin(*args, **kwargs)
        raise OSError("native post-acquisition failure")

    def observe_finish(*args: Any, **kwargs: Any) -> None:
        """Keep real cleanup and observe its ownership flag."""
        cleanup_calls.append(kwargs)
        finish(*args, **kwargs)

    monkeypatch.setattr(repo, "begin_inline_legacy_display", acquired_then_failed)
    monkeypatch.setattr(repo, "finish_legacy_display", observe_finish)
    with pytest.raises(OSError, match="native post-acquisition failure"):
        await worker.handle_generate_variant(payload)
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["inline"] is True
    assert repo.get_slot(payload["slot_id"])["status"] != "generating"
    assert repo.get_batch(legacy["id"]) == legacy
    assert adapter.requests == []
    assert storage.records == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_setup_cancel_then_display_cancel_preserves_sibling_inline_fence(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain both cancellation stages without leaking a marker or releasing a V1 sibling."""
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    sibling = handoff.payload
    identity = {
        "batch_id": sibling["batch_id"], "slot_id": sibling["slot_id"], "variant_index": 0,
    }
    sibling_token = uuid.uuid4().hex
    repo.claim_variant(
        **identity, lease_id="inline", attempt_token=sibling_token,
        item_fields={"pack_id": sibling["pack_id"]},
    )
    repo.start_variant_generation(**identity, attempt_token=sibling_token)
    before_sibling = repo.get_variant_outcome(**identity)
    legacy = repo.create_batch(
        pack_id=sibling["pack_id"], requested_by_user_id=1, status="enqueued",
        total_slots=1, total_variants=1, planned_count=1,
    )
    payload = {**sibling, "batch_id": legacy["id"]}
    adapter, storage = FakeImageAdapter(), ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    setup_entered, setup_release, cleanup_entered, cleanup_release = (threading.Event() for _ in range(4))
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    begin, finish = repo.begin_inline_legacy_display, repo.finish_legacy_display

    def held_begin(*args: Any, **kwargs: Any) -> None:
        """Hold the actual marker acquisition in its owning transaction."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction():
            begin(*args, **kwargs)
            setup_entered.set()
            assert setup_release.wait(4), "legacy setup release timed out"

    def held_finish(*args: Any, **kwargs: Any) -> None:
        """Hold actual marker cleanup after cancellation has drained setup."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction():
            finish(*args, **kwargs)
            cleanup_entered.set()
            assert cleanup_release.wait(4), "legacy display release timed out"

    monkeypatch.setattr(repo, "begin_inline_legacy_display", held_begin)
    monkeypatch.setattr(repo, "finish_legacy_display", held_finish)
    task = asyncio.create_task(worker.handle_generate_variant(payload))
    try:
        assert await asyncio.to_thread(setup_entered.wait, 2)
        assert not setup_release.is_set()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        setup_release.set()
        assert await asyncio.to_thread(cleanup_entered.wait, 2)
        assert not cleanup_release.is_set()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(observed) == 2
        for thread, connection in observed:
            assert thread is not threading.current_thread()
            assert not thread.is_alive()
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        assert repo.get_variant_outcome(**identity) == before_sibling
        assert repo.get_slot(sibling["slot_id"])["status"] == "generating"
        assert repo.get_batch(legacy["id"]) == legacy
        assert adapter.requests == []
        assert storage.records == {}
        repo.release_variant_claim(**identity, attempt_token=sibling_token)
        outcome = repo.get_variant_outcome(**identity)
        assert (outcome["claim_token"], outcome["claim_lease_id"]) == (None, None)
        assert repo.get_slot(sibling["slot_id"])["status"] != "generating"
    finally:
        setup_release.set()
        cleanup_release.set()
        if not task.done():
            await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["legacy_setup", "inline_setup", "jobs_setup", "release", "setup_release"])
@pytest.mark.parametrize("cancel_count,native_failure", [(0, False), (1, False), (2, False), (2, True)])
async def test_worker_setup_and_inline_release_drain_on_owning_thread(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    operation: str, cancel_count: int, native_failure: bool,
) -> None:
    """Block real setup/release writes and verify cancellation, fences and ownership."""
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    payload = dict(handoff.payload)
    if operation == "legacy_setup":
        batch = repo.create_batch(
            pack_id=payload["pack_id"], requested_by_user_id=1, status="enqueued",
            total_slots=1, total_variants=1, planned_count=1,
        )
        payload["batch_id"] = batch["id"]
    before = repo.get_batch(payload["batch_id"])
    adapter, storage = FakeImageAdapter(), ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    if operation == "release":
        def retryable_generation(_request: Any) -> None:
            """Leave the real inline claim planned so release must clear its fence."""
            raise VNAssetGenerationError("test_provider_busy", retryable=True)

        monkeypatch.setattr(adapter, "generate", retryable_generation)
    name = (
        "begin_inline_legacy_display" if operation == "legacy_setup" else
        "release_variant_claim" if operation in {"release", "setup_release"} else "claim_variant"
    )
    original = getattr(repo, name)
    entered, release, pulse, finished = (threading.Event() for _ in range(4))
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    cleanup_observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    arguments: list[dict[str, Any]] = []
    responsive: list[bool] = []
    pending_at_cancel: list[bool] = []
    observer_errors: list[Exception] = []
    loop = asyncio.get_running_loop()

    def held_operation(*args: Any, **kwargs: Any) -> Any:
        """Hold the complete native operation until the independent observer releases it."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        arguments.append(kwargs)
        try:
            with repo.db.transaction():
                result = original(*args, **kwargs)
                entered.set()
                assert release.wait(8), "setup/release observer did not release transaction"
                if native_failure:
                    raise OSError("native setup/release failure")
            return result
        finally:
            finished.set()

    monkeypatch.setattr(repo, name, held_operation)
    cleanup_name = "finish_legacy_display" if operation == "legacy_setup" else "release_variant_claim"
    if cleanup_name != name:
        original_cleanup = getattr(repo, cleanup_name)

        def observe_cleanup(*args: Any, **kwargs: Any) -> None:
            """Observe, but do not replace, actual cleanup after a drained setup."""
            cleanup_observed.append((threading.current_thread(), repo.db.get_connection()))
            original_cleanup(*args, **kwargs)

        monkeypatch.setattr(repo, cleanup_name, observe_cleanup)
    if operation == "setup_release":
        original_claim = repo.claim_variant

        def cancel_after_claim(**kwargs: Any) -> dict[str, Any]:
            """Request real cancellation after acquiring an inline setup fence."""
            result = original_claim(**kwargs)
            acknowledged = threading.Event()

            def cancel_setup() -> None:
                task.cancel()
                acknowledged.set()

            loop.call_soon_threadsafe(cancel_setup)
            assert acknowledged.wait(2), "setup cancellation was not acknowledged"
            return result

        monkeypatch.setattr(repo, "claim_variant", cancel_after_claim)
    task = asyncio.create_task(worker.handle_generate_variant(
        payload, job=handoff.job if operation == "jobs_setup" else None,
    ))

    async def unrelated_progress() -> None:
        responsive.append(not release.is_set())
        pulse.set()

    def observe_loop() -> None:
        """Bound a stalled loop and acknowledge each cancellation before native release."""
        try:
            assert entered.wait(5), "setup/release was not reached"
            loop.call_soon_threadsafe(lambda: asyncio.create_task(unrelated_progress()))
            if not pulse.wait(1):
                return
            for _request in range(cancel_count):
                acknowledged = threading.Event()

                def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                    task.cancel()
                    pending_at_cancel.append(not task.done() and not finished.is_set())
                    acknowledged.set()

                loop.call_soon_threadsafe(cancel_once)
                assert acknowledged.wait(1), "setup/release cancellation was not acknowledged"
            if cancel_count:
                drained = threading.Event()

                def observe_drain() -> None:
                    pending_at_cancel.append(not task.done() and not finished.is_set())
                    drained.set()

                loop.call_soon_threadsafe(observe_drain)
                assert drained.wait(1), "setup/release drain was not observed"
        except (AssertionError, RuntimeError) as exc:
            observer_errors.append(exc)
        finally:
            release.set()

    observer = threading.Thread(target=observe_loop, name="setup-release-observer")
    observer.start()
    error, result = None, None
    try:
        try:
            result = await task
        except (asyncio.CancelledError, VNAssetGenerationError, OSError) as exc:
            error = exc
        await asyncio.sleep(0)
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert observer_errors == []
        assert responsive == [True], "unrelated coroutine ran only after native setup/release released"
        assert pending_at_cancel == ([True] * (cancel_count + 1) if cancel_count else [])
        assert finished.is_set()
        for thread, connection in observed + cleanup_observed:
            assert thread is not threading.current_thread()
            assert not thread.is_alive()
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        cancelled = bool(cancel_count) or operation == "setup_release"
        if cancelled:
            assert isinstance(error, asyncio.CancelledError)
        elif operation == "release":
            assert isinstance(error, VNAssetGenerationError)
            assert str(error) == "test_provider_busy"
            assert error.retryable
        else:
            assert error is None
            assert result["item_id"] > 0
        current = repo.get_batch(payload["batch_id"])
        counters = ("completed_count", "failed_count", "cancelled_count")
        assert tuple(current[key] for key in counters) == (
            tuple(before[key] for key in counters) if cancelled or operation == "release" else (1, 0, 0)
        )
        if operation == "legacy_setup":
            assert cleanup_observed
            assert repo.get_slot(payload["slot_id"])["status"] != "generating"
        else:
            outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
            assert outcome["outcome_status"] == ("planned" if cancelled or operation == "release" else "completed")
            if operation == "jobs_setup":
                assert cleanup_observed == []
                if not native_failure:
                    assert outcome["claim_lease_id"] == handoff.job["lease_id"]
                    assert outcome["claim_token"] == arguments[0]["attempt_token"]
            elif cancelled or operation == "release":
                if native_failure and operation in {"release", "setup_release"}:
                    assert (outcome["claim_token"], outcome["claim_lease_id"]) == (
                        arguments[0]["attempt_token"], "inline",
                    )
                    original(**arguments[0])
                    recovered = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
                    assert (recovered["claim_token"], recovered["claim_lease_id"]) == (None, None)
                else:
                    assert (outcome["claim_token"], outcome["claim_lease_id"]) == (None, None)
            assert handoff.jobs.get_job(handoff.job["id"])["lease_id"] == handoff.job["lease_id"]
        assert len(adapter.requests) == len(storage.records) == (0 if cancelled or operation == "release" else 1)
        if operation == "legacy_setup" and cancelled:
            await worker.handle_generate_variant(handoff.payload)
            assert repo.get_slot(payload["slot_id"])["status"] == "reviewing"
    finally:
        release.set()
        observer.join(timeout=8)
        if not task.done():
            await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation,cancel_count,fail,generation_fail", [
    ("start", 0, False, False), ("start", 0, True, False),
    ("start", 1, False, False), ("start", 2, False, False), ("start", 2, True, False),
    ("display", 0, False, False), ("display", 0, True, False), ("display", 0, True, True),
    ("display", 1, False, False), ("display", 2, False, False), ("display", 2, True, False),
])
async def test_public_worker_transition_drains_off_loop(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    operation: str, cancel_count: int, fail: bool, generation_fail: bool,
) -> None:
    """Hold native start/display writes and observe scheduling and disposition.

    Args:
        handoff: File-backed VN/Jobs rows and storage with a valid V1 delivery.
        monkeypatch: Hold the complete public operation, not its SQL algorithm.
        operation: V1 start or V0 final display reconciliation.
        cancel_count: Zero observes native success/error, one/two call Task.cancel.
        fail: Inject a native write error before the held transaction commits.
        generation_fail: Fail the V0 adapter too, proving cleanup error isolation.

    Returns:
        None: Assert loop responsiveness, drained rollback/close, state and effects.
    """
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    payload = dict(handoff.payload)
    if operation == "display":
        batch = repo.create_batch(
            pack_id=payload["pack_id"], requested_by_user_id=1, status="enqueued",
            total_slots=1, total_variants=1, planned_count=1,
        )
        payload["batch_id"] = batch["id"]
    adapter = FakeImageAdapter()
    if generation_fail:
        def fail_generation(_request: Any) -> None:
            """Raise the original generation error before final display cleanup."""
            raise RuntimeError("original generation failure")

        monkeypatch.setattr(adapter, "generate", fail_generation)
    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage,
    )
    name = "start_variant_generation" if operation == "start" else "finish_legacy_display"
    original = getattr(repo, name)
    started, release, pulse = threading.Event(), threading.Event(), threading.Event()
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    responsive: list[bool] = []

    def held_transition(*args: Any, **kwargs: Any) -> None:
        """Keep the real operation inside an owning connection's transaction."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction():
            original(*args, **kwargs)
            started.set()
            assert release.wait(4), "transition observer did not release transaction"
            if fail:
                raise OSError("native transition failure")

    monkeypatch.setattr(repo, name, held_transition)
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(worker.handle_generate_variant(
        payload, job=handoff.job if operation == "start" else None,
    ))

    def observe_loop() -> None:
        """Request scheduling and real cancellation before releasing the write."""
        try:
            if not started.wait(3):
                return
            loop.call_soon_threadsafe(pulse.set)
            responsive.append(pulse.wait(1))
            if responsive[-1]:
                for _request in range(cancel_count):
                    acknowledged = threading.Event()

                    def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                        """Request caller cancellation on the actual event loop."""
                        task.cancel()
                        acknowledged.set()

                    loop.call_soon_threadsafe(cancel_once)
                    assert acknowledged.wait(1), "cancellation callback did not run"
        finally:
            release.set()

    observer = threading.Thread(target=observe_loop, name="transition-loop-observer")
    observer.start()
    try:
        if cancel_count:
            with pytest.raises(asyncio.CancelledError):
                await task
        elif generation_fail:
            with pytest.raises(RuntimeError, match="original generation failure"):
                await task
        elif fail and operation == "start":
            with pytest.raises(OSError, match="native transition failure"):
                await task
        else:
            assert (await task)["item_id"] > 0
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert responsive == [True]
        thread, connection = observed[0]
        assert thread is not threading.current_thread()
        assert not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        current = repo.get_batch(payload["batch_id"])
        assert current is not None
        if operation == "start":
            outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
            assert outcome is not None
            assert outcome["outcome_status"] == (
                "planned" if cancel_count else "failed" if fail else "completed"
            )
            assert len(adapter.requests) == (0 if cancel_count or fail else 1)
            assert len(storage.records) == (0 if cancel_count or fail else 1)
        else:
            assert (current["completed_count"], current["failed_count"]) == (
                (0, 1) if generation_fail else (1, 0)
            )
            assert len(storage.records) == (0 if generation_fail else 1)
    finally:
        release.set()
        observer.join(timeout=5)
        if not task.done():
            await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["jobs", "inline"])
@pytest.mark.parametrize("cancel_count,native_failure", [
    (0, False), (0, True), (1, False), (2, False), (2, True),
])
async def test_fresh_publication_drains_off_loop_and_preserves_recovery(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    delivery: str, cancel_count: int, native_failure: bool,
) -> None:
    """Fresh publication yields, drains native work and preserves approved bytes/replay."""
    repo, payload = handoff.service.repo, handoff.payload
    owner = repo.db.get_connection()
    approved_storage = ChargedStorage(handoff.outputs)
    approved = repo.create_item(pack_id=payload["pack_id"], slot_id=payload["slot_id"], source="imported")
    registration = await approved_storage(
        user_id=1, item_id=approved["id"], image_bytes=b"existing-approved-png",
    )
    repo.update_item_storage(
        approved["id"], generated_file_id=registration["id"], storage_ref=registration["storage_path"],
        mime_type="image/png", width=None, height=None, bytes=len(b"existing-approved-png"),
    )
    approved = repo.update_item_review(approved["id"], review_status="approved", preferred=True)
    approved_path = handoff.outputs / registration["storage_path"]
    approved_digest = hashlib.sha256(approved_path.read_bytes()).hexdigest()
    initial_batch = repo.get_batch(payload["batch_id"])
    adapter, storage = FakeImageAdapter(), ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    job = handoff.job if delivery == "jobs" else None
    original = repo.complete_variant
    entered, release, pulse, finished = (threading.Event() for _ in range(4))
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    arguments: list[dict[str, Any]] = []
    responsive: list[bool] = []
    pending_at_cancel: list[bool] = []
    observer_errors: list[Exception] = []
    failure = OSError("native publication failure")

    def held_publication(**kwargs: Any) -> dict[str, Any]:
        """Hold actual publication writes until the native observer permits commit/rollback."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        arguments.append(kwargs)
        try:
            with repo.db.transaction():
                result = original(**kwargs)
                entered.set()
                assert release.wait(8), "publication observer did not release transaction"
                if native_failure:
                    raise failure
            return result
        finally:
            finished.set()

    monkeypatch.setattr(repo, "complete_variant", held_publication)
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(worker.handle_generate_variant(payload, job=job))

    async def unrelated_progress() -> None:
        """Record unrelated coroutine execution while the real write is still held."""
        responsive.append(not release.is_set())
        pulse.set()

    def schedule_progress() -> None:
        """Schedule an actual unrelated coroutine on the worker's event loop."""
        asyncio.create_task(unrelated_progress())

    def observe_publication() -> None:
        """Bound even a blocked loop, acknowledging cancellations before native release."""
        try:
            assert entered.wait(5), "fresh publication was not reached"
            loop.call_soon_threadsafe(schedule_progress)
            if not pulse.wait(1):
                return
            for _request in range(cancel_count):
                acknowledged = threading.Event()

                def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                    """Cancel the real task and observe its still-undrained native operation."""
                    task.cancel()
                    pending_at_cancel.append(not task.done() and not finished.is_set())
                    acknowledged.set()

                loop.call_soon_threadsafe(cancel_once)
                assert acknowledged.wait(1), "publication cancellation was not acknowledged"
            if cancel_count:
                acknowledged_drain = threading.Event()

                def observe_drain() -> None:
                    """Observe cancellation after the cancelled task has had a loop turn."""
                    pending_at_cancel.append(not task.done() and not finished.is_set())
                    acknowledged_drain.set()

                loop.call_soon_threadsafe(observe_drain)
                assert acknowledged_drain.wait(1), "publication drain was not observed"
        except (AssertionError, RuntimeError) as exc:
            observer_errors.append(exc)
        finally:
            release.set()

    observer = threading.Thread(target=observe_publication, name="fresh-publication-observer")
    observer.start()
    result, error = None, None
    try:
        try:
            result = await task
        except (VNAssetGenerationError, asyncio.CancelledError) as exc:
            error = exc
        await asyncio.sleep(0)
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert observer_errors == []
        assert responsive == [True], "unrelated coroutine ran only after native publication released"
        assert pending_at_cancel == ([True] * (cancel_count + 1) if cancel_count else [])
        assert finished.is_set()
        thread, connection = observed[0]
        assert thread is not threading.current_thread()
        assert not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        if cancel_count:
            assert isinstance(error, asyncio.CancelledError)
        elif native_failure:
            assert isinstance(error, VNAssetGenerationError)
            assert str(error) == "vn_asset_publication_retryable"
            assert error.retryable
            assert error.__cause__ is failure
        else:
            assert error is None
            assert result["item_id"] == arguments[0]["item_id"]
        outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
        item = repo.get_item(outcome["item_id"])
        assert arguments[0]["batch_id"] == payload["batch_id"]
        assert arguments[0]["slot_id"] == payload["slot_id"]
        assert arguments[0]["variant_index"] == 0
        assert (arguments[0]["validate_authority"] is not None) is (delivery == "jobs")
        assert outcome["outcome_status"] == ("planned" if native_failure else "completed")
        assert item["review_status"] == ("hidden" if native_failure else "draft")
        if native_failure and delivery == "inline":
            assert (outcome["claim_token"], outcome["claim_lease_id"]) == (None, None)
        else:
            assert outcome["claim_token"] == arguments[0]["attempt_token"]
        batch = repo.get_batch(payload["batch_id"])
        counters = ("completed_count", "failed_count", "cancelled_count")
        assert tuple(batch[key] for key in counters) == (
            tuple(initial_batch[key] for key in counters) if native_failure else (1, 0, 0)
        )
        record = storage.records[item["generated_file_id"]]
        assert record["source_ref"] == f"vn_asset_item:{item['id']}"
        assert (handoff.outputs / item["storage_ref"]).read_bytes() == b"fake-png"
        assert storage.usage == len(b"fake-png")
        assert len(adapter.requests) == len(storage.records) == 1
        assert repo.get_item(approved["id"]) == approved
        assert hashlib.sha256(approved_path.read_bytes()).hexdigest() == approved_digest

        monkeypatch.setattr(repo, "complete_variant", original)
        if native_failure and delivery == "jobs":
            assert handoff.jobs.release_job(
                job["id"], worker_id=job["worker_id"], lease_id=job["lease_id"], enforce=True,
            )
            job = handoff.jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="publication-recovery",
                lease_seconds=120, job_type="vn_asset_generate_variant",
            )
            assert job is not None and job["id"] == handoff.job["id"]
            assert job["lease_id"] != handoff.job["lease_id"]
        recovered = await worker.handle_generate_variant(payload, job=job)
        assert recovered["item_id"] == item["id"]
        completed = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
        completed_batch = repo.get_batch(payload["batch_id"])
        completed_item = repo.get_item(item["id"])
        assert completed["outcome_status"] == "completed"
        assert tuple(completed_batch[key] for key in counters) == (1, 0, 0)
        assert (await worker.handle_generate_variant(payload, job=job))["item_id"] == item["id"]
        assert repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0) == completed
        assert repo.get_batch(payload["batch_id"]) == completed_batch
        assert repo.get_item(item["id"]) == completed_item
        assert len(adapter.requests) == len(storage.records) == 1
        assert storage.usage == len(b"fake-png")
        assert repo.get_item(approved["id"]) == approved
        assert hashlib.sha256(approved_path.read_bytes()).hexdigest() == approved_digest
    finally:
        release.set()
        observer.join(timeout=8)
        if not task.done():
            await task
        assert not observer.is_alive()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ["memory", "transaction"])
async def test_fresh_publication_retains_native_owner_fallback(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fallback: str,
) -> None:
    """Actual private-memory/caller-transaction publication retains owner work and handle."""
    db = CharactersRAGDB(":memory:", client_id="fresh-publication-memory") if fallback == "memory" else handoff.service.repo.db
    try:
        if fallback == "memory":
            jobs = JobManager(db_path=tmp_path / "fallback-jobs.db")
            service = VNAssetPackService(db, owner_user_id=1, jobs_manager=jobs)
            character = db.add_character_card({"name": "Mira", "description": "Archivist"})
            pack = service.create_pack(VNAssetPackCreate(title="Memory publication", primary_character_id=character))
            slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
            batch = service.start_generation(pack.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id]))
            payload = {"user_id": 1, "pack_id": pack.id, "slot_id": slot.id, "batch_id": batch.batch_id, "variant_index": 0}
        else:
            service, jobs, payload = handoff.service, handoff.jobs, handoff.payload
        repo = service.repo
        owner, owner_thread = db.get_connection(), threading.current_thread()
        before_batch = repo.get_batch(payload["batch_id"])
        before_recipes = repo.list_batch_recipes(payload["batch_id"])
        before_outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
        before_items = repo.list_items(payload["pack_id"])
        before_slot = repo.get_slot(payload["slot_id"])
        adapter, storage = FakeImageAdapter(), ChargedStorage(handoff.outputs)
        worker = VNAssetGenerationWorker(
            repo=repo, jobs_manager=jobs, image_registry=FakeImageRegistry(adapter),
            backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
        )
        observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
        original = repo.complete_variant

        def observe_publication(**kwargs: Any) -> dict[str, Any]:
            """Observe the actual publication connection without replacing its native writes."""
            observed.append((threading.current_thread(), db.get_connection()))
            return original(**kwargs)

        monkeypatch.setattr(repo, "complete_variant", observe_publication)
        if fallback == "transaction":
            with pytest.raises(RuntimeError, match="caller rollback"):
                with db.transaction():
                    result = await worker.handle_generate_variant(payload)
                    assert owner.in_transaction
                    assert repo.get_item(result["item_id"])["review_status"] == "draft"
                    assert repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)["outcome_status"] == "completed"
                    assert repo.get_batch(payload["batch_id"])["completed_count"] == 1
                    raise RuntimeError("caller rollback")
            assert repo.get_batch(payload["batch_id"]) == before_batch
            assert repo.list_batch_recipes(payload["batch_id"]) == before_recipes
            assert repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0) == before_outcome
            assert repo.list_items(payload["pack_id"]) == before_items
            assert repo.get_slot(payload["slot_id"]) == before_slot
        else:
            result = await worker.handle_generate_variant(payload)
            assert repo.get_item(result["item_id"])["review_status"] == "draft"
            assert repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)["outcome_status"] == "completed"
            assert repo.get_batch(payload["batch_id"])["completed_count"] == 1
        assert observed == [(owner_thread, owner)]
        assert not owner.in_transaction
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        assert len(adapter.requests) == len(storage.records) == 1
        assert (handoff.outputs / storage.records[result["item_id"]]["storage_path"]).read_bytes() == b"fake-png"
    finally:
        if fallback == "memory":
            db.close_connection()
