"""Native variant preflight reads and terminal reconciliation drain off-loop."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["recipe", "lease", "cancel"])
@pytest.mark.parametrize("cancel_count,native_failure", [
    (0, False), (0, True), (1, False), (2, False), (2, True),
])
async def test_native_variant_preflight_drains_off_loop(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    operation: str, cancel_count: int, native_failure: bool,
) -> None:
    """Hold one complete public native operation before model execution.

    Args:
        handoff: Real independent file-backed VN metadata and owned Jobs delivery.
        monkeypatch: Surround a public method, retaining its native read/transaction.
        operation: Decoded recipe read, authoritative lease read or terminal cancel.
        cancel_count: Zero preserves native disposition; one/two request Task.cancel.
        native_failure: Raise OSError after the held work, forcing transaction rollback.

    Returns:
        None: Assert loop progress, original native/cancelled disposition, owning
        thread exit/connection close, caller handle survival and exact public effects.
    """
    repo = handoff.service.repo
    payload = handoff.payload
    owner = repo.db.get_connection()
    if operation == "cancel":
        repo.cancel_batch(payload["batch_id"])
    before = repo.get_batch(payload["batch_id"])
    started, release, pulse = (threading.Event() for _ in range(3))
    responsive: list[bool] = []
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    materialized: list[dict[str, Any] | None] = []
    target = handoff.jobs if operation == "lease" else repo
    name = {"recipe": "get_batch_recipe", "lease": "get_job", "cancel": "cancel_batch"}[operation]
    original = getattr(target, name)

    def held_operation(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Run the real public operation inside its owned VN observation lifetime."""
        if started.is_set():
            return original(*args, **kwargs)
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction():
            result = original(*args, **kwargs)
            materialized.append(result)
            started.set()
            assert release.wait(5), "preflight observer did not release"
            if native_failure:
                raise OSError("native preflight failure")
            return result

    monkeypatch.setattr(target, name, held_operation)
    adapter = FakeImageAdapter()
    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage,
    )
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(worker.handle_generate_variant(payload, job=handoff.job))

    def observe_loop() -> None:
        """Schedule real loop progress and cancellation before releasing native work."""
        try:
            if not started.wait(4):
                return
            loop.call_soon_threadsafe(pulse.set)
            responsive.append(pulse.wait(1))
            if responsive[-1]:
                for _request in range(cancel_count):
                    acknowledged = threading.Event()

                    def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                        """Request actual caller cancellation on the event-loop thread."""
                        task.cancel()
                        acknowledged.set()

                    loop.call_soon_threadsafe(cancel_once)
                    assert acknowledged.wait(1), "caller cancellation was not acknowledged"
        finally:
            release.set()

    observer = threading.Thread(target=observe_loop, name="variant-preflight-observer")
    observer.start()
    try:
        if cancel_count:
            with pytest.raises(asyncio.CancelledError):
                await task
        elif native_failure:
            with pytest.raises(OSError, match="native preflight failure"):
                await task
        elif operation == "cancel":
            with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
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
        assert type(materialized[0]) is dict
        if operation == "recipe":
            assert type(materialized[0]["labels"]) is dict
        outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
        batch = repo.get_batch(payload["batch_id"])
        assert before is not None and outcome is not None and batch is not None
        if operation == "cancel":
            assert (batch["completed_count"], batch["failed_count"], batch["cancelled_count"]) == (
                before["completed_count"], before["failed_count"], before["cancelled_count"],
            )
            assert outcome["outcome_status"] == "cancelled"
        elif cancel_count or native_failure:
            assert outcome["outcome_status"] == "planned"
            assert (batch["completed_count"], batch["failed_count"]) == (0, 0)
        else:
            assert outcome["outcome_status"] == "completed"
            assert (batch["completed_count"], batch["failed_count"]) == (1, 0)
        successful = operation != "cancel" and not cancel_count and not native_failure
        assert len(adapter.requests) == int(successful)
        assert len(storage.records) == int(successful)
    finally:
        release.set()
        observer.join(timeout=6)
        if not task.done():
            await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_model", "after_model", "after_save", "before_publication"])
async def test_standalone_delivery_lease_reads_remain_responsive(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """Observe a standalone native Jobs read through supported generation seams.

    Args:
        handoff: Actual file-backed VN/Jobs state and current owned variant delivery.
        monkeypatch: Surround public Jobs lookup and storage attachment, never a lock.
        phase: Adapter acquisition, returned model bytes, saved file or attached item.

    Returns:
        None: Assert a responsive loop, disposed owned handle, sole native model,
        stored byte registration and published outcome with unchanged counters.
    """
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    armed, started, release, pulse = (threading.Event() for _ in range(4))
    responsive: list[bool] = []
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    storage = ChargedStorage(handoff.outputs)
    original_lookup = handoff.jobs.get_job
    original_attach = repo.update_item_storage

    def held_lookup(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Hold the first standalone lookup after the chosen supported boundary."""
        result = original_lookup(*args, **kwargs)
        connection = repo.db.get_connection()
        if armed.is_set() and not started.is_set() and not connection.in_transaction:
            observed.append((threading.current_thread(), connection))
            started.set()
            assert release.wait(5), "lease observer did not release"
        return result

    def attach_then_arm(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Keep native attachment and then expose the pre-publication observation."""
        result = original_attach(*args, **kwargs)
        if phase == "before_publication":
            armed.set()
        return result

    class BoundaryAdapter(FakeImageAdapter):
        """Arm after the real injected model result has materialized."""

        def generate(self, request: ImageGenRequest) -> ImageGenResult:
            """Generate the same actual bytes and identify the post-model boundary."""
            result = super().generate(request)
            if phase == "after_model":
                armed.set()
            return result

    class BoundaryRegistry(FakeImageRegistry):
        """Expose the supported adapter-acquisition boundary before model execution."""

        def get_adapter(self, backend: str) -> FakeImageAdapter | None:
            """Preserve adapter selection before arming the lease-read observation."""
            result = super().get_adapter(backend)
            if phase == "before_model":
                armed.set()
            return result

    async def save_then_arm(**kwargs: Any) -> dict[str, Any]:
        """Preserve real saved bytes/registration and identify the post-save boundary."""
        result = await storage(**kwargs)
        if phase == "after_save":
            armed.set()
        return result

    monkeypatch.setattr(handoff.jobs, "get_job", held_lookup)
    monkeypatch.setattr(repo, "update_item_storage", attach_then_arm)
    adapter = BoundaryAdapter()
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=BoundaryRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=save_then_arm,
        generated_files_repo=storage,
    )
    loop = asyncio.get_running_loop()

    def observe_loop() -> None:
        """Prove the loop can advance while the real authority observation is held."""
        try:
            if started.wait(4):
                loop.call_soon_threadsafe(pulse.set)
                responsive.append(pulse.wait(1))
        finally:
            release.set()

    observer = threading.Thread(target=observe_loop, name="standalone-lease-observer")
    observer.start()
    try:
        result = await worker.handle_generate_variant(handoff.payload, job=handoff.job)
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert responsive == [True]
        thread, connection = observed[0]
        assert thread is not threading.current_thread() and not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        assert len(adapter.requests) == len(storage.records) == 1
        item = repo.get_item(result["item_id"])
        assert item is not None and item["review_status"] == "draft"
        batch = repo.get_batch(handoff.payload["batch_id"])
        assert batch is not None and (batch["completed_count"], batch["failed_count"]) == (1, 0)
    finally:
        release.set()
        observer.join(timeout=6)
