"""Public claim responsiveness and owned transaction/cancellation regressions."""

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
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    FakeGenerationGate,
    FakeImageAdapter,
    FakeImageRegistry,
    StoredVNSaver,
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count,fail", [(0, False), (0, True), (1, False), (2, False), (2, True)])
async def test_public_claim_is_responsive_and_drains_owned_transaction(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, cancel_count: int, fail: bool,
) -> None:
    """An actual claim drains its transaction off-loop before error/cancel return.

    Args:
        handoff: Real file-backed VN/Jobs authority and valid versioned delivery.
        monkeypatch: Wrap the public claim operation without replacing its SQL.
        cancel_count: Zero observes success/native failure; one/two cancel the task.
        fail: Raise a native OSError inside the owned transaction after admission.

    Returns:
        None: Assert responsiveness, commit/rollback, closure and generation effects.
    """
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    original_claim = repo.claim_variant
    started, release, pulse = threading.Event(), threading.Event(), threading.Event()
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    responsive: list[bool] = []
    adapter = FakeImageAdapter()
    storage = StoredVNSaver(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage,
    )

    def held_claim(**kwargs: Any) -> dict[str, Any]:
        """Keep the real nested claim write open until the independent observer releases it."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction():
            item = original_claim(**kwargs)
            started.set()
            assert release.wait(4), "claim observer failed to release transaction"
            if fail:
                raise OSError("claim transaction failed")
            return item

    monkeypatch.setattr(repo, "claim_variant", held_claim)
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(worker.handle_generate_variant(handoff.payload, job=handoff.job))

    def observe_loop() -> None:
        """Probe the event loop while the claim still holds its real write transaction."""
        try:
            if not started.wait(3):
                return
            loop.call_soon_threadsafe(pulse.set)
            responsive.append(pulse.wait(1))
            if responsive[-1]:
                for _request in range(cancel_count):
                    acknowledged = threading.Event()

                    def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                        """Request actual caller cancellation and acknowledge loop execution."""
                        task.cancel()
                        acknowledged.set()

                    loop.call_soon_threadsafe(cancel_once)
                    assert acknowledged.wait(1), "cancellation callback did not run"
        finally:
            release.set()

    observer = threading.Thread(target=observe_loop, name="claim-loop-observer")
    observer.start()
    try:
        if cancel_count:
            with pytest.raises(asyncio.CancelledError):
                await task
        elif fail:
            with pytest.raises(OSError, match="claim transaction failed"):
                await task
        else:
            result = await task
            assert result["item_id"] > 0
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert responsive == [True]
        thread, connection = observed[0]
        assert thread is not threading.current_thread()
        assert not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        outcome = repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0)
        assert outcome is not None
        assert outcome["outcome_status"] == ("completed" if not cancel_count and not fail else "planned")
        assert (outcome["item_id"] is None) is fail
        assert len(adapter.requests) == (0 if cancel_count or fail else 1)
    finally:
        release.set()
        observer.join(timeout=5)
        if not task.done():
            await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_cancelled_inline_claim_releases_fence_for_public_retry(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, cancel_count: int,
) -> None:
    """A drained inline claim cancellation releases only its token and reuses its item.

    Args:
        handoff: Real file-backed VN authority with an unclaimed versioned recipe.
        monkeypatch: Hold the real public claim transaction for cancellation.
        cancel_count: One or two actual caller Task.cancel requests before commit.

    Returns:
        None: Verify released fence, preserved reservation and successful inline retry.
    """
    repo = handoff.service.repo
    started, release = threading.Event(), threading.Event()
    original_claim = repo.claim_variant
    adapter = FakeImageAdapter()
    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage,
    )

    def held_inline_claim(**kwargs: Any) -> dict[str, Any]:
        """Complete the real inline claim in a transaction held until cancellation."""
        with repo.db.transaction():
            item = original_claim(**kwargs)
            started.set()
            assert release.wait(3), "inline claim release timed out"
            return item

    monkeypatch.setattr(repo, "claim_variant", held_inline_claim)
    task = asyncio.create_task(worker.handle_generate_variant(handoff.payload))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        for _request in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        outcome = repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0)
        assert outcome is not None
        assert (outcome["outcome_status"], outcome["claim_lease_id"], outcome["claim_token"]) == (
            "planned", None, None,
        )
        reserved_item = repo.get_item(outcome["item_id"])
        assert reserved_item is not None
        assert (reserved_item["review_status"], reserved_item["generated_file_id"]) == ("hidden", None)
        assert len(adapter.requests) == 0
        monkeypatch.setattr(repo, "claim_variant", original_claim)
        result = await worker.handle_generate_variant(handoff.payload)
        assert result["item_id"] == reserved_item["id"]
        completed = repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0)
        assert completed is not None and completed["outcome_status"] == "completed"
        assert len(adapter.requests) == 1
        batch = repo.get_batch(handoff.payload["batch_id"])
        assert batch is not None
        assert (batch["completed_count"], batch["failed_count"], batch["cancelled_count"]) == (1, 0, 0)
    finally:
        release.set()
        if not task.done():
            await task
