"""Post-model reads and definitive failure writes drain without blocking the loop."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["batch_read", "failure"])
@pytest.mark.parametrize("cancel_count,native_failure", [
    (0, False), (0, True), (1, False), (2, False), (2, True),
])
async def test_public_delivery_bookkeeping_drains_off_loop(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    operation: str, cancel_count: int, native_failure: bool,
) -> None:
    """Hold a complete native operation after the real adapter result or failure.

    Args:
        handoff: Independent native VN/Jobs state with a current owned delivery.
        monkeypatch: Hold only a public repository operation around its real transaction.
        operation: Post-model batch read or definitive variant failure persistence.
        cancel_count: Zero preserves native disposition; one/two invoke actual Task.cancel.
        native_failure: Raise a native OSError after the real held work, rolling it back.

    Returns:
        None: Observe loop progress, drained owned connection/thread, native error
        or caller cancellation, unchanged authoritative identity and exact effects.
    """
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    generated, started, release, pulse = (threading.Event() for _ in range(4))
    responsive: list[bool] = []
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []

    class DeliveryAdapter(FakeImageAdapter):
        """Deliver real deterministic bytes, optionally raising a definitive failure."""

        def generate(self, request: ImageGenRequest) -> ImageGenResult:
            """Record the real call before opening the post-model observation gate."""
            result = super().generate(request)
            generated.set()
            if operation == "failure":
                raise ValueError("adapter rejected image")
            return result

    name = "get_batch" if operation == "batch_read" else "fail_variant"
    original = getattr(repo, name)

    def held_operation(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Hold the first public operation after the model at the owning transaction."""
        if not generated.is_set() or started.is_set():
            return original(*args, **kwargs)
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction():
            result = original(*args, **kwargs)
            started.set()
            assert release.wait(5), "bookkeeping observer did not release"
            if native_failure:
                raise OSError("native bookkeeping failure")
            return result

    monkeypatch.setattr(repo, name, held_operation)
    adapter = DeliveryAdapter()
    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage,
    )
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(worker.handle_generate_variant(handoff.payload, job=handoff.job))

    def observe_loop() -> None:
        """Request a real loop callback and cancellation before releasing native work."""
        try:
            if not started.wait(4):
                return
            loop.call_soon_threadsafe(pulse.set)
            responsive.append(pulse.wait(1))
            if responsive[-1]:
                for _request in range(cancel_count):
                    acknowledged = threading.Event()

                    def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                        """Request caller cancellation on the actual event-loop thread."""
                        task.cancel()
                        acknowledged.set()

                    loop.call_soon_threadsafe(cancel_once)
                    assert acknowledged.wait(1), "caller cancellation callback did not run"
        finally:
            release.set()

    observer = threading.Thread(target=observe_loop, name="delivery-bookkeeping-observer")
    observer.start()
    try:
        if cancel_count:
            with pytest.raises(asyncio.CancelledError):
                await task
        elif native_failure:
            with pytest.raises(OSError, match="native bookkeeping failure"):
                await task
        elif operation == "failure":
            with pytest.raises(ValueError, match="adapter rejected image"):
                await task
        else:
            assert (await task)["item_id"] > 0
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert responsive == [True]
        thread, connection = observed[0]
        assert thread is not threading.current_thread() and not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        outcome = repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0)
        batch = repo.get_batch(handoff.payload["batch_id"])
        assert outcome is not None and batch is not None
        assert len(adapter.requests) == 1
        if operation == "failure":
            assert outcome["outcome_status"] == ("planned" if native_failure else "failed")
            assert (batch["completed_count"], batch["failed_count"]) == (0, 0 if native_failure else 1)
            assert storage.records == {}
        elif cancel_count:
            assert outcome["outcome_status"] == "planned"
            assert (batch["completed_count"], batch["failed_count"]) == (0, 0)
            assert storage.records == {}
        elif native_failure:
            assert outcome["outcome_status"] == "failed"
            assert (batch["completed_count"], batch["failed_count"]) == (0, 1)
            assert storage.records == {}
        else:
            assert outcome["outcome_status"] == "completed"
            assert (batch["completed_count"], batch["failed_count"]) == (1, 0)
            assert len(storage.records) == 1
    finally:
        release.set()
        observer.join(timeout=6)
        if not task.done():
            await task
