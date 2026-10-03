"""Real worker start/display transactions remain responsive and drain on cancel."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


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
