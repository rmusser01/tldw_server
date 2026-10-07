"""Public replay responsiveness and connection ownership regression contracts."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.VN_Assets.jobs import vn_asset_generation_jobs_queue
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import (
    ChargedStorage,
)
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import (
    handoff as handoff,
)
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get_item", "get_batch_recipe", "update_item_storage", "complete_variant"])
async def test_public_replay_repository_wait_does_not_stall_loop(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """Replay yields during each repository boundary and disposes only its own connection."""
    repo = handoff.service.repo
    storage = ChargedStorage(handoff.outputs)
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
        unregister_generated_file=storage.unregister,
    )
    original_attach = repo.update_item_storage
    if operation == "get_item":
        first = await worker.handle_generate_variant(handoff.payload, job=handoff.job)
    else:
        def interrupt(*_args: Any, **_kwargs: Any) -> None:
            """Leave a genuine registered file for a replacement delivery to reconcile."""
            raise RuntimeError("attachment interrupted")

        monkeypatch.setattr(repo, "update_item_storage", interrupt)
        with pytest.raises(VNAssetGenerationError, match="vn_asset_storage_handoff_retryable"):
            await worker.handle_generate_variant(handoff.payload, job=handoff.job)
        monkeypatch.setattr(repo, "update_item_storage", original_attach)
        first = {"item_id": repo.get_variant_outcome(
            handoff.payload["batch_id"], handoff.payload["slot_id"], 0,
        )["item_id"]}
        assert handoff.jobs.release_job(
            handoff.job["id"], worker_id="handoff", lease_id=handoff.job["lease_id"], enforce=True,
        )
        handoff.job = handoff.jobs.acquire_next_job(
            domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="replay", lease_seconds=120,
            job_type="vn_asset_generate_variant",
        )
        assert handoff.job is not None
    owner = repo.db.get_connection()
    owner_thread = threading.get_ident()
    blocked, release, responsive = threading.Event(), threading.Event(), threading.Event()
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    original = getattr(repo, operation)
    reads = 0

    def delayed(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Delay the real replay operation, leaving any earlier admission read untouched."""
        nonlocal reads
        reads += 1
        if operation != "get_batch_recipe" or reads > 1:
            observed.append((threading.current_thread(), repo.db.get_connection()))
            blocked.set()
            assert release.wait(3), "replay query safety release timed out"
        return original(*args, **kwargs)

    def safety_release() -> None:
        """Bound the original loop stall without requiring a responsive event loop."""
        if blocked.wait(3):
            responsive.wait(0.5)
        release.set()

    monkeypatch.setattr(repo, operation, delayed)
    watchdog = threading.Thread(target=safety_release)
    watchdog.start()
    pending = asyncio.create_task(worker.handle_generate_variant(handoff.payload, job=handoff.job))
    try:
        assert await asyncio.to_thread(blocked.wait, 3)
        resumed_before_release = not release.is_set()
        responsive.set()
        result = await pending
        assert resumed_before_release, "event loop resumed only after synchronous replay released"
        assert result["item_id"] == first["item_id"]
        assert len(adapter.requests) == 1
        assert repo.get_batch(handoff.payload["batch_id"])["completed_count"] == 1
        assert observed
        for thread, connection in observed:
            assert thread.ident != owner_thread
            assert not thread.is_alive()
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        release.set()
        watchdog.join(3)
        if not pending.done():
            await pending


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ["memory", "transaction"])
async def test_replay_operation_retains_owner_connection_fallback(handoff: SimpleNamespace, fallback: str) -> None:
    """Private memory and caller transactions never move or close their owner handle."""
    db = CharactersRAGDB(":memory:", client_id="replay-memory") if fallback == "memory" else handoff.service.repo.db
    repo = VNAssetPacksRepository(db)
    repo.initialize_schema()
    owner = db.get_connection()
    owner_thread = threading.get_ident()
    observed: list[tuple[int, Any]] = []

    def read() -> dict[str, Any] | None:
        """Read a real item while observing the caller's public connection identity."""
        observed.append((threading.get_ident(), db.get_connection()))
        return repo.get_item(-1)

    try:
        if fallback == "memory":
            assert await repo.run_worker_replay_operation(read) is None
            assert observed == [(owner_thread, owner)]
            assert owner.execute("SELECT 1").fetchone()[0] == 1
        else:
            with db.transaction():
                assert await repo.run_worker_replay_operation(read) is None
                assert observed == [(owner_thread, owner)]
                assert owner.in_transaction
                assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if fallback == "memory":
            db.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replay_operation_error_rolls_back_and_closes_owned_connection(handoff: SimpleNamespace) -> None:
    """A failed complete thread-owned operation rolls back and leaves caller state usable."""
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    observed: list[sqlite3.Connection] = []

    def failing_transition() -> dict[str, Any] | None:
        """Fail after a real write inside the operation-owned transaction."""
        observed.append(repo.db.get_connection())
        with repo.db.transaction() as connection:
            connection.execute("UPDATE vn_asset_batches SET completed_count=99 WHERE id=?", (handoff.payload["batch_id"],))
            raise OSError("replay transition unavailable")

    with pytest.raises(OSError, match="replay transition unavailable"):
        await repo.run_worker_replay_operation(failing_transition)
    assert repo.get_batch(handoff.payload["batch_id"])["completed_count"] == 0
    assert owner.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        observed[0].execute("SELECT 1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replay_cancellation_drains_operation_before_closing_handle(handoff: SimpleNamespace) -> None:
    """Coroutine cancellation waits for thread completion/commit and disposes its handle."""
    repo = handoff.service.repo
    started, release = threading.Event(), threading.Event()
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []

    def delayed_transition() -> dict[str, Any] | None:
        """Hold a complete real transition on its own connection until permitted to commit."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction() as connection:
            started.set()
            assert release.wait(3), "cancellation drain release timed out"
            connection.execute("UPDATE vn_asset_batches SET enqueue_error='drained' WHERE id=?", (handoff.payload["batch_id"],))
        return repo.get_batch(handoff.payload["batch_id"])

    pending = asyncio.create_task(repo.run_worker_replay_operation(delayed_transition))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert repo.get_batch(handoff.payload["batch_id"])["enqueue_error"] == "drained"
        thread, connection = observed[0]
        assert not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    finally:
        release.set()
        if not pending.done():
            await pending
