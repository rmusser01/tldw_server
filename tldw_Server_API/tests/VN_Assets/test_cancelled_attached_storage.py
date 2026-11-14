"""Terminal cancellation reclaims only its exact unpublished attached file."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    FakeGenerationGate,
    FakeImageAdapter,
    FakeImageRegistry,
)


async def _cancel_after_attachment(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Interrupt the caller after real attachment and terminal cancellation commit.

    Args:
        handoff: Real VN/Jobs databases and output directory from the shared fixture.
        monkeypatch: Restore the public boundary after simulating lost acknowledgement.

    Returns:
        The public worker, real saved bytes/registration, hidden item and cancelled
        batch/recipe snapshots, ready for terminal redelivery.
    """
    repo = handoff.service.repo
    storage = ChargedStorage(handoff.outputs)
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage, unregister_generated_file=storage.unregister,
    )
    attach = repo.update_item_storage

    def attach_then_cancel(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Interrupt acknowledgement only after real attachment/cancellation commit."""
        attach(*args, **kwargs)
        repo.cancel_batch(handoff.payload["batch_id"])
        handoff.jobs.cancel_job(handoff.job["id"])
        raise asyncio.CancelledError("attachment acknowledgement interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(repo, "update_item_storage", attach_then_cancel)
        with pytest.raises(asyncio.CancelledError):
            await worker.handle_generate_variant(handoff.payload, job=handoff.job)
    outcome = repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0)
    item = repo.get_item(outcome["item_id"])
    record = storage.records[item["generated_file_id"]]
    assert item["review_status"] == "hidden"
    assert outcome["outcome_status"] == "cancelled"
    assert (handoff.outputs / record["storage_path"]).read_bytes() == b"fake-png"
    return SimpleNamespace(
        worker=worker, storage=storage, adapter=adapter, item=item, record=record,
        outcome=outcome, batch=repo.get_batch(handoff.payload["batch_id"]),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejected_publication_reclaims_terminal_attached_file(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean a terminal attachment immediately even if Jobs never redelivers.

    Args:
        handoff: Real VN/Jobs rows and actual output bytes for one owned delivery.
        monkeypatch: Commit public cancellation only after native attachment returns.

    Returns:
        None: Reject publication while releasing only the hidden attachment's
        registration/bytes/callback charge and preserving terminal counts.
    """
    repo = handoff.service.repo
    storage = ChargedStorage(handoff.outputs)
    adapter = FakeImageAdapter()
    attach = repo.update_item_storage

    def attach_then_cancel(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Attach real bytes before cancellation races the native publication guard."""
        item = attach(*args, **kwargs)
        repo.cancel_batch(handoff.payload["batch_id"])
        return item

    monkeypatch.setattr(repo, "update_item_storage", attach_then_cancel)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage, unregister_generated_file=storage.unregister,
    )
    with pytest.raises(VNAssetGenerationError):
        await worker.handle_generate_variant(handoff.payload, job=handoff.job)
    outcome = repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0)
    batch = repo.get_batch(handoff.payload["batch_id"])
    assert outcome["outcome_status"] == "cancelled"
    assert (batch["completed_count"], batch["failed_count"], batch["cancelled_count"]) == (0, 0, 1)
    assert repo.get_item(outcome["item_id"])["generated_file_id"] is None
    assert repo.list_items(handoff.payload["pack_id"]) == []
    assert storage.records == {} and storage.usage == 0
    assert len(storage.removed) == 1
    assert list(handoff.outputs.glob("generated-*.png")) == []
    assert len(adapter.requests) == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("outage", ["none", "unlink", "unregister"])
async def test_terminal_redelivery_reclaims_attached_hidden_file(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, outage: str,
) -> None:
    """Drain exact attached cancellation; preserve discoverability through outages.

    Args:
        handoff: Real database and output fixture; quota is the supported callback double.
        monkeypatch: Inject only a native physical deletion outage when requested.
        outage: No failure, physical unlink failure, or quota-aware removal failure.

    Returns:
        None: Assert detachment, once-only physical/accounting cleanup, unchanged
        recipe/batch outcomes, no model rerun and survival of the caller handle.
    """
    state = await _cancel_after_attachment(handoff, monkeypatch)
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    path = handoff.outputs / state.record["storage_path"]
    if outage == "unlink":
        unlink = Path.unlink

        def unavailable_unlink(target: Path, missing_ok: bool = False) -> None:
            """Fail native deletion of only the exact owned generated file."""
            if target == path:
                raise OSError("native unlink unavailable")
            unlink(target, missing_ok=missing_ok)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", unavailable_unlink)
            with pytest.raises(OSError, match="native unlink unavailable"):
                await state.worker.handle_generate_variant(handoff.payload, job=handoff.job)
        assert path.read_bytes() == b"fake-png"
        assert state.storage.records and state.storage.usage == len(b"fake-png")
        assert repo.get_item(state.item["id"])["generated_file_id"] is None
    elif outage == "unregister":
        async def unavailable_unregister(_file_id: int, hard_delete: bool = False) -> bool:
            """Retain discoverable accounting when the supported removal API fails."""
            assert hard_delete
            return False

        state.worker.unregister_generated_file = unavailable_unregister
        with pytest.raises(VNAssetGenerationError, match="vn_asset_cancelled_storage_cleanup_retryable"):
            await state.worker.handle_generate_variant(handoff.payload, job=handoff.job)
        assert not path.exists()
        assert state.storage.records and state.storage.usage == len(b"fake-png")
        assert repo.get_item(state.item["id"])["generated_file_id"] is None
        state.worker.unregister_generated_file = state.storage.unregister
    for _delivery in range(2):
        with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
            await state.worker.handle_generate_variant(handoff.payload, job=handoff.job)
        assert not path.exists()
        assert state.storage.records == {} and state.storage.usage == 0
        assert state.storage.removed == [state.record["id"]]
        assert repo.get_item(state.item["id"])["generated_file_id"] is None
        assert repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0) == state.outcome
        assert repo.get_batch(handoff.payload["batch_id"]) == state.batch
    assert len(state.adapter.requests) == 1
    assert owner.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["foreign", "referenced", "mismatched", "draft", "approved"])
async def test_attached_terminal_cleanup_preserves_guarded_files(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    """Never detach/delete foreign, other-referenced, mismatched or visible files.

    Args:
        handoff: Native VN/Jobs/output state for a cancelled unpublished attachment.
        monkeypatch: Install the setup cancellation at the public attachment seam.
        boundary: The current registration/item guard that must refuse cleanup.

    Returns:
        None: Preserve bytes/registration/charge, exact item, recipe and counters.
    """
    state = await _cancel_after_attachment(handoff, monkeypatch)
    repo = handoff.service.repo
    if boundary == "foreign":
        state.record["user_id"] = 2
    elif boundary == "referenced":
        repo.create_item(
            slot_id=handoff.payload["slot_id"], pack_id=handoff.payload["pack_id"],
            generated_file_id=state.record["id"],
        )
    elif boundary == "mismatched":
        repo.update_item_storage(
            state.item["id"], generated_file_id=state.record["id"] + 100,
            storage_ref=state.item["storage_ref"], mime_type=state.item["mime_type"],
            width=state.item["width"], height=state.item["height"], bytes=state.item["bytes"],
        )
    else:
        repo.update_item_review(state.item["id"], review_status=boundary)
    item = repo.get_item(state.item["id"])
    for _delivery in range(2):
        with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
            await state.worker.handle_generate_variant(handoff.payload, job=handoff.job)
    assert repo.get_item(state.item["id"]) == item
    assert repo.get_variant_outcome(handoff.payload["batch_id"], handoff.payload["slot_id"], 0) == state.outcome
    assert repo.get_batch(handoff.payload["batch_id"]) == state.batch
    assert state.storage.records and state.storage.usage == len(b"fake-png")
    assert state.storage.removed == []
    assert (handoff.outputs / state.record["storage_path"]).read_bytes() == b"fake-png"
    assert len(state.adapter.requests) == 1
