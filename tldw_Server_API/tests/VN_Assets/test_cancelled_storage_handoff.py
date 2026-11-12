"""Public cancellation handoff regressions with real VN/Jobs rows and bytes."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetGenerationRequest, VNAssetPackCreate
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.jobs import vn_asset_generation_jobs_queue
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    FakeGenerationGate,
    FakeImageAdapter,
    FakeImageRegistry,
    StoredVNSaver,
)


@pytest.fixture
def handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[SimpleNamespace, None, None]:
    """Provision independent file-backed VN and Jobs authorities and output storage."""
    db = CharactersRAGDB(str(tmp_path / "handoff-vn.db"), client_id="vn-handoff-test")
    jobs_path = tmp_path / "handoff-jobs.db"
    jobs = JobManager(db_path=jobs_path)
    service = VNAssetPackService(db, owner_user_id=1, jobs_manager=jobs)
    character = db.add_character_card({"name": "Mira", "description": "Archivist"})
    pack = service.create_pack(VNAssetPackCreate(title="Handoff", primary_character_id=character))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    batch = service.start_generation(pack.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id]))
    # The public worker owns fanout; acquire the real variant delivery afterwards.
    VNAssetGenerationWorker(repo=service.repo, jobs_manager=jobs).handle_enqueue_batch({
        "user_id": 1, "pack_id": pack.id, "batch_id": batch.batch_id,
    })
    job = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="handoff", lease_seconds=120,
        job_type="vn_asset_generate_variant",
    )
    assert job is not None
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))
    yield SimpleNamespace(
        service=service, jobs=jobs, jobs_path=jobs_path, job=job, outputs=tmp_path,
        payload={"user_id": 1, "pack_id": pack.id, "batch_id": batch.batch_id,
                 "slot_id": slot.id, "variant_index": 0},
    )
    db.close_connection()


class ChargedStorage(StoredVNSaver):
    """Record real bytes and expose the quota-aware unregistration contract."""

    def __init__(self, outputs: Path) -> None:
        """Keep registration usage and successful quota releases for assertions."""
        super().__init__(outputs)
        self.usage = 0
        self.removed: list[int] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Register bytes and charge the owner once."""
        record = await super().__call__(**kwargs)
        self.usage += int(record["file_size_bytes"])
        return record

    async def get_file_by_source_ref(self, **identity: Any) -> dict[str, Any] | None:
        """Find the exact owned source registration for a terminal redelivery."""
        return next((record for record in self.records.values() if all(
            record.get(key) == value for key, value in identity.items()
        )), None)

    async def unregister(self, file_id: int, hard_delete: bool = False) -> bool:
        """Remove the registration and release its usage through the supported callback."""
        assert hard_delete
        record = self.records.pop(file_id, None)
        if record is None:
            return False
        self.usage -= int(record["file_size_bytes"])
        self.removed.append(file_id)
        return True


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["cancelled", "referenced", "foreign", "takeover", "requested"])
async def test_public_handoff_reclaims_only_terminal_unreferenced_storage(handoff: SimpleNamespace, boundary: str) -> None:
    """Terminal cancellation reclaims orphan bytes, never a foreign/referenced/recoverable file."""
    storage = ChargedStorage(handoff.outputs)
    repo = handoff.service.repo
    payload = handoff.payload
    original_batch = repo.get_batch(payload["batch_id"])

    async def save_then_revoke(**kwargs: Any) -> dict[str, Any]:
        """Revoke real authority immediately after registration but before attachment."""
        record = await storage(**kwargs)
        if boundary == "takeover":
            assert handoff.jobs.release_job(
                handoff.job["id"], worker_id="handoff", lease_id=handoff.job["lease_id"], enforce=True,
            )
            assert handoff.jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="replacement",
                lease_seconds=120, job_type="vn_asset_generate_variant",
            ) is not None
        elif boundary == "requested":
            with sqlite3.connect(handoff.jobs_path) as connection:
                connection.execute("UPDATE jobs SET cancel_requested_at=CURRENT_TIMESTAMP WHERE id=?", (handoff.job["id"],))
        else:
            repo.cancel_batch(payload["batch_id"])
            handoff.jobs.cancel_job(handoff.job["id"])
            if boundary == "foreign":
                record["user_id"] = 2
            if boundary == "referenced":
                repo.create_item(slot_id=payload["slot_id"], pack_id=payload["pack_id"], generated_file_id=record["id"])
        return record

    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=save_then_revoke,
        generated_files_repo=storage, unregister_generated_file=storage.unregister,
    )
    with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
        await worker.handle_generate_variant(payload, job=handoff.job)
    outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
    item = repo.get_item(outcome["item_id"])
    assert item["generated_file_id"] is None
    current_batch = repo.get_batch(payload["batch_id"])
    assert current_batch["completed_count"] == original_batch["completed_count"] == 0
    assert current_batch["failed_count"] == original_batch["failed_count"] == 0
    assert current_batch["cancelled_count"] == (0 if boundary in {"takeover", "requested"} else 1)
    assert storage.usage == (0 if boundary == "cancelled" else len(b"fake-png"))
    assert (handoff.outputs / f"generated-{item['id']}.png").exists() is (boundary != "cancelled")
    assert len(storage.removed) == (1 if boundary == "cancelled" else 0)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("temporary_removal_failure", [False, True])
async def test_cancelled_redelivery_reclaims_interrupted_registration(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, temporary_removal_failure: bool,
) -> None:
    """A fresh public delivery can reclaim a registration left before cancellation."""
    storage = ChargedStorage(handoff.outputs)
    adapter = FakeImageAdapter()
    repo = handoff.service.repo
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage,
        generated_files_repo=storage, unregister_generated_file=storage.unregister,
    )
    original_attach = repo.update_item_storage

    def interrupted_attachment(*_args: Any, **_kwargs: Any) -> None:
        """Simulate process handoff failure after actual storage registration."""
        raise RuntimeError("interrupted attachment")

    monkeypatch.setattr(repo, "update_item_storage", interrupted_attachment)
    with pytest.raises(VNAssetGenerationError, match="vn_asset_storage_handoff_retryable"):
        await worker.handle_generate_variant(handoff.payload, job=handoff.job)
    monkeypatch.setattr(repo, "update_item_storage", original_attach)
    repo.cancel_batch(handoff.payload["batch_id"])
    handoff.jobs.cancel_job(handoff.job["id"])
    if temporary_removal_failure:
        async def unavailable_unregister(_file_id: int, hard_delete: bool = False) -> bool:
            """Retain the discoverable registration and charge while removal is unavailable."""
            assert hard_delete
            return False

        worker.unregister_generated_file = unavailable_unregister
        with pytest.raises(VNAssetGenerationError, match="vn_asset_cancelled_storage_cleanup_retryable"):
            await worker.handle_generate_variant(handoff.payload, job=handoff.job)
        assert storage.usage == len(b"fake-png")
        assert storage.records
        assert list(handoff.outputs.glob("generated-*.png")) == []
        worker.unregister_generated_file = storage.unregister
    with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
        await worker.handle_generate_variant(handoff.payload, job=handoff.job)
    assert len(adapter.requests) == 1
    assert storage.usage == 0
    assert storage.records == {}
    assert len(storage.removed) == 1
    assert list(handoff.outputs.glob("generated-*.png")) == []
    with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
        await worker.handle_generate_variant(handoff.payload, job=handoff.job)
    assert len(storage.removed) == 1
