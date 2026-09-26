from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import vn_assets as vn_assets_endpoint
from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetBulkReviewRequest,
    VNAssetGenerationRequest,
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenResult
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.concurrency import BackendGenerationGate, BackendGenerationLease
from tldw_Server_API.app.core.VN_Assets.constants import ERROR_ITEM_LIMIT_EXCEEDED
from tldw_Server_API.app.core.VN_Assets.jobs import (
    enqueue_batch_idempotency_key,
    generate_variant_idempotency_key,
    vn_asset_generation_jobs_queue,
)
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService


class FakeJobs:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self._by_idempotency_key: dict[str, dict[str, Any]] = {}
        self.cancelled_ids: list[int] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        idempotency_key = str(kwargs.get("idempotency_key") or "")
        if idempotency_key and idempotency_key in self._by_idempotency_key:
            return self._by_idempotency_key[idempotency_key]

        job = {
            "id": len(self.created) + 1,
            "status": "queued",
            **kwargs,
        }
        self.created.append(job)
        if idempotency_key:
            self._by_idempotency_key[idempotency_key] = job
        return job

    def list_jobs(self, **filters: Any) -> list[dict[str, Any]]:
        jobs = self.created
        for key, value in filters.items():
            if key in {"limit", "sort_by", "sort_order"} or value is None:
                continue
            jobs = [job for job in jobs if job.get(key) == value]
        return jobs[: int(filters.get("limit") or len(jobs))]

    def get_job(self, job_id: int, **_kwargs: Any) -> dict[str, Any] | None:
        return next((job for job in self.created if job["id"] == job_id), None)

    def cancel_job(self, job_id: int, *, reason: str | None = None) -> bool:
        self.cancelled_ids.append(job_id)
        for job in self.created:
            if int(job["id"]) == job_id:
                job["status"] = "cancelled"
                job["cancellation_reason"] = reason
                return True
        return False


class RejectingJobs:
    def create_job(self, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("queued job quota exceeded")


class FailingChildJobs(FakeJobs):
    def __init__(self, *, fail_after_children: int) -> None:
        super().__init__()
        self.fail_after_children = fail_after_children

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        if (
            kwargs.get("job_type") == "vn_asset_generate_variant"
            and len(self.created) >= self.fail_after_children
        ):
            raise ValueError("child job quota exceeded")
        return super().create_job(**kwargs)


class FakeImageAdapter:
    def __init__(self, content: bytes = b"fake-png") -> None:
        self.content = content
        self.requests: list[Any] = []

    def generate(self, request: Any) -> ImageGenResult:
        self.requests.append(request)
        return ImageGenResult(
            content=self.content,
            content_type="image/png",
            bytes_len=len(self.content),
        )


class FakeImageRegistry:
    def __init__(self, adapter: FakeImageAdapter) -> None:
        self.adapter = adapter
        self.resolved_backends: list[str | None] = []
        self.adapter_names: list[str] = []

    def resolve_backend(self, requested: str | None) -> str | None:
        self.resolved_backends.append(requested)
        return requested or "stable_diffusion_cpp"

    def get_adapter(self, name: str) -> FakeImageAdapter | None:
        self.adapter_names.append(name)
        return self.adapter


class FakeGenerationGate:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str | None]] = []

    def try_acquire(self, backend: str, *, model: str | None = None) -> BackendGenerationLease:
        self.requests.append((backend, model))
        return BackendGenerationLease(acquired=True, backend=backend, model=model)


class RecordingVNSaver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "id": 77,
            "storage_path": "vn_assets/2026/04/24/generated.png",
            "mime_type": "image/png",
        }


class FailingVNSaver:
    async def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("storage failed")


class StoredVNSaver:
    """Disk-backed generated-file double for replay integration with the VN DB."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self.records: dict[int, dict[str, Any]] = {}

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        item_id = int(kwargs["item_id"])
        storage_path = f"generated-{item_id}.png"
        (self.outputs_dir / storage_path).write_bytes(kwargs["image_bytes"])
        record = {
            "id": item_id, "user_id": kwargs["user_id"], "source_feature": "vn_assets",
            "source_ref": f"vn_asset_item:{item_id}", "is_deleted": False,
            "storage_path": storage_path, "file_size_bytes": len(kwargs["image_bytes"]),
            "mime_type": "image/png",
        }
        self.records[item_id] = record
        return record

    async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        return self.records.get(file_id)


class EmptyGeneratedFiles:
    async def get_file_by_source_ref(self, **_kwargs: Any) -> None:
        return None


class BlockingFirstImageAdapter(FakeImageAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self._calls = 0
        self._lock = threading.Lock()

    def generate(self, request: Any) -> ImageGenResult:
        with self._lock:
            self._calls += 1
            first = self._calls == 1
        if first:
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError("test adapter was not released")
        return super().generate(request)


@pytest.mark.asyncio
async def test_duplicate_delivery_with_same_lease_does_not_call_adapter_twice(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    child = fake_jobs.create_job(job_type="vn_asset_generate_variant", owner_user_id="1")
    child.update(status="processing", lease_id="lease-1", leased_until="2099-01-01 00:00:00")
    adapter = BlockingFirstImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
        generated_files_repo=EmptyGeneratedFiles(),
    )
    payload = {
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }
    first = asyncio.create_task(worker.handle_generate_variant(payload, job=dict(child)))
    assert await asyncio.to_thread(adapter.started.wait, 5)
    try:
        with pytest.raises(ValueError, match="vn_asset_variant_in_progress"):
            await worker.handle_generate_variant(payload, job=dict(child))
    finally:
        adapter.release.set()
    await first

    assert len(adapter.requests) == 1
    assert len(saver.calls) == 1
    assert service.repo.count_items_for_generation(pack_with_slots.id) == 1


@pytest.mark.asyncio
async def test_stale_lease_cannot_publish_after_new_lease_takes_claim(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    child = fake_jobs.create_job(job_type="vn_asset_generate_variant", owner_user_id="1")
    child.update(status="processing", lease_id="lease-1", leased_until="2099-01-01 00:00:00")
    old_job = dict(child)
    adapter = BlockingFirstImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
        generated_files_repo=EmptyGeneratedFiles(),
    )
    payload = {
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }
    first = asyncio.create_task(worker.handle_generate_variant(payload, job=old_job))
    assert await asyncio.to_thread(adapter.started.wait, 5)
    child["lease_id"] = "lease-2"
    try:
        result = await worker.handle_generate_variant(payload, job=dict(child))
    finally:
        adapter.release.set()
    with pytest.raises(ValueError, match="vn_asset_job_lease_lost"):
        await first

    assert result["generated_file_id"] == 77
    assert len(saver.calls) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1


@pytest.mark.asyncio
async def test_cancellation_during_adapter_does_not_register_asset(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    adapter = BlockingFirstImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
    )
    pending = asyncio.create_task(worker.handle_generate_variant({
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }))
    assert await asyncio.to_thread(adapter.started.wait, 5)
    service.cancel_generation(pack_with_slots.id)
    adapter.release.set()
    with pytest.raises(ValueError, match="vn_asset_batch_terminal"):
        await pending

    assert saver.calls == []
    assert service.repo.get_batch(batch.batch_id)["cancelled_count"] == 1
    assert service.repo.count_items_for_generation(pack_with_slots.id) == 0


@pytest.mark.asyncio
async def test_takeover_backend_contention_is_retryable_until_stale_adapter_exits(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace, tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    jobs = JobManager(db_path=tmp_path / "busy-takeover-jobs.db")
    jobs.create_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant", owner_user_id="1", payload={},
    )
    acquisition = {
        "domain": "vn_assets", "queue": vn_asset_generation_jobs_queue(), "lease_seconds": 120,
    }
    old_job = jobs.acquire_next_job(**acquisition, worker_id="old-worker")
    assert old_job is not None
    adapter = BlockingFirstImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=BackendGenerationGate(default_local_limit=1),
        save_vn_asset_image=saver, generated_files_repo=EmptyGeneratedFiles(),
    )
    payload = {
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }
    pending = asyncio.create_task(worker.handle_generate_variant(payload, job=old_job))
    assert await asyncio.to_thread(adapter.started.wait, 5)
    try:
        assert jobs.release_job(
            int(old_job["id"]), worker_id="old-worker",
            lease_id=str(old_job["lease_id"]), enforce=True,
        )
        replacement = jobs.acquire_next_job(**acquisition, worker_id="replacement-worker")
        assert replacement is not None
        with pytest.raises(VNAssetGenerationError, match="vn_asset_backend_busy") as caught:
            await worker.handle_generate_variant(payload, job=replacement)
        assert caught.value.retryable is True
        assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "planned"
        assert service.repo.get_batch(batch.batch_id)["failed_count"] == 0
    finally:
        adapter.release.set()
        with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
            await pending
    assert jobs.release_job(
        int(replacement["id"]), worker_id="replacement-worker",
        lease_id=str(replacement["lease_id"]), enforce=True,
    )
    retry = jobs.acquire_next_job(**acquisition, worker_id="retry-worker")
    assert retry is not None
    result = await worker.handle_generate_variant(payload, job=retry)
    assert result["generated_file_id"] == 77
    assert len(saver.calls) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1


@pytest.mark.asyncio
async def test_legacy_backend_contention_keeps_failure_behavior(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.repo.create_batch(
        pack_id=pack_with_slots.id, requested_by_user_id=1, status="enqueued", total_variants=1,
    )
    gate = BackendGenerationGate(default_local_limit=1)
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, backend_gate=gate,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
    )
    with gate.try_acquire("stable_diffusion_cpp"):
        with pytest.raises(ValueError, match="vn_asset_backend_busy"):
            await worker.handle_generate_variant({
                "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
                "batch_id": batch["id"], "user_id": 1,
            })
    assert service.repo.get_batch(batch["id"])["status"] == "failed"
    assert service.repo.get_batch(batch["id"])["failed_count"] == 1


@pytest.mark.asyncio
async def test_expired_job_lease_cannot_claim_a_variant(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    child = fake_jobs.create_job(job_type="vn_asset_generate_variant", owner_user_id="1")
    child.update(status="processing", lease_id="expired", leased_until="2000-01-01 00:00:00")
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    with pytest.raises(ValueError, match="vn_asset_job_lease_lost"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        }, job=dict(child))
    assert adapter.requests == []
    assert service.repo.count_items_for_generation(pack_with_slots.id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_loss", ["cancelled", "replaced", "expired"])
async def test_lease_loss_after_storage_attachment_blocks_publication(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_loss: str,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    jobs_path = tmp_path / "publication-jobs.db"
    jobs = JobManager(db_path=jobs_path)
    child = jobs.create_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant", owner_user_id="1", payload={},
    )
    job = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        worker_id="vn-worker", lease_seconds=120,
    )
    assert job is not None
    assert job["id"] == child["id"]
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    original_update = service.repo.update_item_storage

    def attach_then_lose_lease(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        item = original_update(*args, **kwargs)
        if lease_loss == "cancelled":
            jobs.cancel_job(int(job["id"]))
        elif lease_loss == "replaced":
            assert jobs.release_job(
                int(job["id"]), worker_id="vn-worker", lease_id=str(job["lease_id"]), enforce=True,
            )
            replacement = jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
                worker_id="replacement-worker", lease_seconds=120,
            )
            assert replacement is not None
            assert replacement["lease_id"] != job["lease_id"]
        else:
            with sqlite3.connect(jobs_path) as conn:
                conn.execute("UPDATE jobs SET leased_until = '2000-01-01 00:00:00' WHERE id = ?", (job["id"],))
        return item

    monkeypatch.setattr(service.repo, "update_item_storage", attach_then_lose_lease)
    with pytest.raises(ValueError, match="vn_asset_job_lease_lost"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        }, job=job)

    outcome = service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)
    assert outcome["outcome_status"] == "planned"
    assert service.repo.get_item(outcome["item_id"])["review_status"] == "hidden"
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 0


@pytest.mark.asyncio
async def test_delayed_jobs_validation_cannot_replace_new_claim(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    jobs = JobManager(db_path=tmp_path / "delayed-validation-jobs.db")
    jobs.create_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant", owner_user_id="1", payload={},
    )
    old_job = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        worker_id="old-worker", lease_seconds=120,
    )
    assert old_job is not None
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    original_get = jobs.get_job
    delayed = False

    def delayed_get(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        nonlocal delayed
        snapshot = original_get(*args, **kwargs)
        if not delayed:
            delayed = True
            assert jobs.release_job(
                int(old_job["id"]), worker_id="old-worker",
                lease_id=str(old_job["lease_id"]), enforce=True,
            )
            new_job = jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
                worker_id="new-worker", lease_seconds=120,
            )
            assert new_job is not None
            service.repo.claim_variant(
                batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
                lease_id=str(new_job["lease_id"]), attempt_token="new-claim",
                item_fields={"pack_id": pack_with_slots.id}, allow_takeover=True,
                expected_claim_token=None,
                validate_authority=lambda: worker._require_current_job_lease(new_job, user_id=1),
            )
        return snapshot

    monkeypatch.setattr(jobs, "get_job", delayed_get)
    with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        }, job=old_job)
    outcome = service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)
    assert outcome["claim_token"] == "new-claim"
    assert outcome["outcome_status"] == "planned"
    assert service.repo.count_items_for_generation(pack_with_slots.id) == 1
    assert adapter.requests == []


@pytest.mark.parametrize("transition", ["claim", "attach", "complete", "fail"])
def test_repository_checks_jobs_authority_after_variant_write_lock(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str,
) -> None:
    from tldw_Server_API.app.core.DB_Management import VNAssetPacks_DB as db_module
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    jobs = JobManager(db_path=tmp_path / "lock-validation-jobs.db")
    jobs.create_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant", owner_user_id="1", payload={},
    )
    job = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        worker_id="vn-worker", lease_seconds=120,
    )
    assert job is not None
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=jobs)
    identity = {"batch_id": batch.batch_id, "slot_id": slot.id, "variant_index": 0}
    item = None
    if transition != "claim":
        item = service.repo.claim_variant(
            **identity, lease_id=str(job["lease_id"]), attempt_token="claim",
            item_fields={"pack_id": pack_with_slots.id, "generated_file_id": 17},
        )
    worker._require_current_job_lease(job, user_id=1)
    original_lock = db_module._lock_variant

    def lock_then_cancel(*args: Any) -> None:
        original_lock(*args)
        assert jobs.cancel_job(int(job["id"]))

    monkeypatch.setattr(db_module, "_lock_variant", lock_then_cancel)
    authority = {"validate_authority": lambda: worker._require_current_job_lease(job, user_id=1)}
    with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
        if transition == "claim":
            service.repo.claim_variant(
                **identity, **authority, lease_id=str(job["lease_id"]), attempt_token="claim",
                item_fields={"pack_id": pack_with_slots.id},
            )
        elif transition == "attach":
            assert item is not None
            service.repo.update_item_storage(
                item["id"], **identity, **authority, attempt_token="claim",
                generated_file_id=88, storage_ref="stale.png", mime_type="image/png",
                width=10, height=10, bytes=3,
            )
        elif transition == "complete":
            assert item is not None
            service.repo.complete_variant(**identity, **authority, item_id=item["id"], attempt_token="claim")
        else:
            service.repo.fail_variant(**identity, **authority, error="adapter failed", attempt_token="claim")
    outcome = service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)
    assert outcome["outcome_status"] == "planned"
    if item is None:
        assert outcome["item_id"] is None
    else:
        assert service.repo.get_item(item["id"])["generated_file_id"] == 17
        assert service.repo.get_item(item["id"])["review_status"] == "hidden"


@pytest.fixture
def chacha_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-assets-jobs-test-client")
    yield database
    database.close_connection()


@pytest.mark.asyncio
async def test_cancelled_legacy_reservation_is_reconciled_on_redelivery(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
        item_fields={"pack_id": pack_with_slots.id},
    )
    service.repo.update_batch(batch.batch_id, {"status": "cancelled"})
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)
    with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        })
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "cancelled"
    assert service.repo.get_batch(batch.batch_id)["cancelled_count"] == 1
    assert service.repo.get_item(item["id"])["review_status"] == "hidden"
    assert service.repo.count_items_for_generation(pack_with_slots.id) == 0


@pytest.mark.asyncio
async def test_adapter_failure_after_jobs_cancellation_keeps_variant_planned(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace, tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    jobs = JobManager(db_path=tmp_path / "adapter-failure-jobs.db")
    jobs.create_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant", owner_user_id="1", payload={},
    )
    job = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        worker_id="vn-worker", lease_seconds=120,
    )
    assert job is not None

    class CancelledFailingAdapter(FakeImageAdapter):
        def generate(self, request: Any) -> ImageGenResult:
            assert jobs.cancel_job(int(job["id"]))
            raise RuntimeError("adapter failed after cancellation")

    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(CancelledFailingAdapter()),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=saver,
    )
    with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        }, job=job)
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "planned"
    assert service.repo.get_batch(batch.batch_id)["failed_count"] == 0
    assert saver.calls == []


@pytest.fixture
def character_id(chacha_db: CharactersRAGDB) -> int:
    return chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
        }
    )


@pytest.fixture
def fake_jobs() -> FakeJobs:
    return FakeJobs()


@pytest.fixture
def service(chacha_db: CharactersRAGDB, fake_jobs: FakeJobs) -> VNAssetPackService:
    return VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)


@pytest.fixture
def pack_with_slots(
    service: VNAssetPackService,
    character_id: int,
) -> SimpleNamespace:
    pack = service.create_pack(VNAssetPackCreate(title="Generated Pack", primary_character_id=character_id))
    slots = service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    return SimpleNamespace(id=pack.id, slots=slots)


@pytest.fixture
def batch_with_slots(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> SimpleNamespace:
    result = service.start_generation(pack_with_slots.id, user_id=1)
    parent_job = fake_jobs.created[-1]
    fake_jobs.created.clear()
    return SimpleNamespace(
        id=result.batch_id,
        pack_id=pack_with_slots.id,
        slots=pack_with_slots.slots,
        job_payload=parent_job["payload"],
    )


def test_generation_endpoint_enqueues_single_parent_job(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    result = service.start_generation(pack_with_slots.id, user_id=1)

    assert result.batch_id
    assert result.status == "queued"
    assert result.planned_count == sum(slot.variant_count for slot in pack_with_slots.slots)
    assert result.enqueued_count == 0
    assert result.enqueue_error is None
    assert len(fake_jobs.created) == 1
    job = fake_jobs.created[0]
    assert job["domain"] == "vn_assets"
    assert job["queue"] == "default"
    assert job["job_type"] == "vn_asset_enqueue_batch"
    assert job["batch_group"] == f"vn_assets:user:1:pack:{pack_with_slots.id}:batch:{result.batch_id}"
    assert job["idempotency_key"] == (
        f"vn_assets:user:1:pack:{pack_with_slots.id}:batch:{result.batch_id}:enqueue"
    )
    assert job["payload"] == {
        "pack_id": pack_with_slots.id,
        "batch_id": result.batch_id,
        "user_id": 1,
    }


def test_generation_job_idempotency_keys_are_scoped_by_owner() -> None:
    parent_one = enqueue_batch_idempotency_key(user_id=1, pack_id=1, batch_id=1)
    parent_two = enqueue_batch_idempotency_key(user_id=2, pack_id=1, batch_id=1)
    child_one = generate_variant_idempotency_key(
        user_id=1,
        pack_id=1,
        batch_id=1,
        slot_id=1,
        variant_index=0,
    )
    child_two = generate_variant_idempotency_key(
        user_id=2,
        pack_id=1,
        batch_id=1,
        slot_id=1,
        variant_index=0,
    )

    assert parent_one != parent_two
    assert child_one != child_two


def test_vn_asset_generation_queue_is_allowed_by_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES", raising=False)
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES_VN_ASSETS", raising=False)

    jobs = JobManager(db_path=tmp_path / "jobs.db")

    assert vn_asset_generation_jobs_queue() in jobs._get_allowed_queues("vn_assets")


def test_generation_retries_original_batch_when_parent_enqueue_is_rejected(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_with_slots.id}",
        "idempotency_key": "retry-parent-after-quota",
        "payload_hash": "same-request",
    }
    service.repo.claim_idempotency_record(owner_user_id=1, **receipt)
    with pytest.raises(ValueError, match="queued job quota exceeded"):
        service.start_generation(
            pack_with_slots.id,
            user_id=1,
            jobs_manager=RejectingJobs(),
            idempotency_receipt=receipt,
        )

    batches = service.repo.list_batches(pack_with_slots.id)
    assert len(batches) == 1
    assert batches[0]["status"] == "queued"
    assert batches[0]["enqueue_error"] == "queued job quota exceeded"
    record = service.repo.get_idempotency_record(
        owner_user_id=1, scope=receipt["scope"],
        resource_id=receipt["resource_id"], idempotency_key=receipt["idempotency_key"],
    )
    jobs = FakeJobs()
    recovered = service.recover_generation_receipt(
        record, pack_id=pack_with_slots.id, jobs_manager=jobs,
    )
    replay = service.recover_generation_receipt(
        record, pack_id=pack_with_slots.id, jobs_manager=jobs,
    )
    assert recovered.batch_id == replay.batch_id == batches[0]["id"]
    assert len(jobs.created) == 1
    assert service.repo.get_batch(batches[0]["id"])["enqueue_error"] is None


def test_start_generation_enforces_item_limit_against_existing_items(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    limited_service = VNAssetPackService(
        chacha_db,
        owner_user_id=1,
        jobs_manager=fake_jobs,
        item_limit=1,
    )
    pack = limited_service.create_pack(
        VNAssetPackCreate(title="Limited Pack", primary_character_id=character_id)
    )
    slot = limited_service.create_slot(
        pack.id,
        VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary", variant_count=1),
    )
    limited_service.repo.create_item(
        pack_id=pack.id,
        slot_id=slot.id,
        variant_index=0,
    )

    with pytest.raises(ValueError, match=ERROR_ITEM_LIMIT_EXCEEDED):
        limited_service.start_generation(pack.id, user_id=1)


def test_unpublished_reserved_item_cannot_be_read_or_reviewed(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
        item_fields={"pack_id": pack_with_slots.id, "generated_file_id": 77},
    )

    with pytest.raises(ValueError, match="item_not_found"):
        service.get_item_for_pack(pack_with_slots.id, item["id"])
    with pytest.raises(ValueError, match="item_not_found"):
        service.review_item_for_pack(
            pack_with_slots.id, item["id"], VNAssetReviewRequest(review_status="approved")
        )
    with pytest.raises(ValueError, match="item_not_found"):
        service.bulk_review_items_for_pack(
            pack_with_slots.id,
            VNAssetBulkReviewRequest(item_ids=[item["id"]], review_status="approved"),
        )


def test_active_batch_prevents_deleting_frozen_slot(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Deletion Pack", primary_character_id=character_id)
    )
    slot = service.create_slot(
        pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="primary", variant_count=1)
    )
    batch = service.start_generation(
        pack.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id])
    )

    with pytest.raises(ValueError, match="slot_has_active_generation"):
        service.delete_slot(pack.id, slot.id)

    assert service.repo.get_slot(slot.id) is not None
    assert service.repo.get_batch_recipe(batch.batch_id, slot.id, 0) is not None
    service.cancel_generation(pack.id)
    service.delete_slot(pack.id, slot.id)
    assert service.repo.get_slot(slot.id) is None


def test_fanout_uses_deterministic_child_idempotency(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    worker.handle_enqueue_batch(batch_with_slots.job_payload)
    worker.handle_enqueue_batch(batch_with_slots.job_payload)

    assert fake_jobs.created
    first_job = fake_jobs.created[0]
    assert first_job["queue"] == "generation"
    assert first_job["job_type"] == "vn_asset_generate_variant"
    assert first_job["idempotency_key"].startswith(
        f"vn_assets:user:1:pack:{batch_with_slots.pack_id}:batch:{batch_with_slots.id}:slot:"
    )
    assert first_job["batch_group"] == (
        f"vn_assets:user:1:pack:{batch_with_slots.pack_id}:batch:{batch_with_slots.id}"
    )
    assert first_job["payload"] == {
        "pack_id": batch_with_slots.pack_id,
        "slot_id": batch_with_slots.slots[0].id,
        "variant_index": 0,
        "batch_id": batch_with_slots.id,
        "user_id": 1,
    }
    assert len(fake_jobs.created) == sum(slot.variant_count for slot in batch_with_slots.slots)


def test_fanout_uses_original_variants_after_slot_edit(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    first_slot = batch_with_slots.slots[0]
    service.repo.update_slot(first_slot.id, {"variant_count": 3})
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    worker.handle_enqueue_batch(batch_with_slots.job_payload)

    assert len(fake_jobs.created) == sum(slot.variant_count for slot in batch_with_slots.slots)
    assert all(job["payload"]["variant_index"] == 0 for job in fake_jobs.created)


def test_fanout_does_not_regress_a_batch_completed_by_fast_children(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    original_create = fake_jobs.create_job

    def complete_during_fanout(**kwargs: Any) -> dict[str, Any]:
        job = original_create(**kwargs)
        service.repo.update_batch(batch_with_slots.id, {"status": "completed"})
        return job

    monkeypatch.setattr(fake_jobs, "create_job", complete_during_fanout)
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    worker.handle_enqueue_batch(batch_with_slots.job_payload)

    assert service.repo.get_batch(batch_with_slots.id)["status"] == "completed"


def test_fanout_rejects_payload_owner_mismatch(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)
    bad_payload = dict(batch_with_slots.job_payload)
    bad_payload["user_id"] = 2

    with pytest.raises(ValueError, match="vn_asset_job_owner_mismatch"):
        worker.handle_enqueue_batch(bad_payload)

    assert fake_jobs.created == []


@pytest.mark.asyncio
async def test_generate_variant_creates_draft_item_with_generated_file(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    adapter = FakeImageAdapter()
    registry = FakeImageRegistry(adapter)
    gate = FakeGenerationGate()
    saver = RecordingVNSaver()
    slot = pack_with_slots.slots[0]
    batch = service.repo.create_batch(
        pack_id=pack_with_slots.id,
        requested_by_user_id=1,
        status="enqueued",
        total_slots=1,
        total_variants=1,
        planned_count=1,
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=registry,
        backend_gate=gate,
        save_vn_asset_image=saver,
    )

    result = await worker.handle_generate_variant(
        {
            "pack_id": pack_with_slots.id,
            "slot_id": slot.id,
            "variant_index": 0,
            "batch_id": batch["id"],
            "user_id": 1,
        }
    )

    items = service.repo.list_items(pack_id=pack_with_slots.id)
    assert result["status"] == "draft_created"
    assert result["item_id"] == items[0]["id"]
    assert len(items) == 1
    assert items[0]["review_status"] == "draft"
    assert items[0]["generated_file_id"] == 77
    assert items[0]["storage_ref"] == "vn_assets/2026/04/24/generated.png"
    assert items[0]["mime_type"] == "image/png"
    assert items[0]["bytes"] == len(adapter.content)
    assert saver.calls[0]["item_id"] == items[0]["id"]
    assert saver.calls[0]["pack_id"] == pack_with_slots.id
    assert saver.calls[0]["asset_type"] == slot.asset_type
    assert adapter.requests[0].backend == "stable_diffusion_cpp"
    assert "Labels:" in adapter.requests[0].prompt
    assert gate.requests == [("stable_diffusion_cpp", None)]
    assert service.repo.get_batch(batch["id"])["completed_count"] == 1
    assert service.repo.get_batch(batch["id"])["status"] == "completed"
    assert service.repo.get_slot(slot.id)["status"] == "reviewing"


@pytest.mark.asyncio
async def test_generation_uses_original_recipe_after_source_edits(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    chacha_db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    service.repo.update_pack(pack_with_slots.id, {
        "style_prompt": "Original watercolor style",
        "default_backend": "stable_diffusion_cpp",
    })
    service.repo.update_slot(slot.id, {
        "prompt_template": "Original lantern scene",
        "labels": {"expression": "thoughtful"},
    })
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )

    service.repo.update_pack(pack_with_slots.id, {
        "style_prompt": "Changed oil style",
        "default_backend": "openrouter",
    })
    service.repo.update_slot(slot.id, {
        "prompt_template": "Changed ocean scene",
        "labels": {"expression": "angry"},
    })
    chacha_db.execute_query(
        "UPDATE character_cards SET description = ? WHERE id = ?",
        ("Changed character description", service.repo.get_pack(pack_with_slots.id)["primary_character_id"]),
    )
    adapter = FakeImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
    )
    await worker.handle_generate_variant({
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    })

    assert "Original lantern scene" in adapter.requests[0].prompt
    assert "Original watercolor style" in adapter.requests[0].prompt
    assert "Changed" not in adapter.requests[0].prompt
    assert adapter.requests[0].backend == "stable_diffusion_cpp"
    assert saver.calls[0]["labels"] == {"expression": "thoughtful"}


@pytest.mark.asyncio
async def test_versioned_batch_with_missing_recipe_fails_closed(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    service.repo.db.execute_query(
        "DELETE FROM vn_asset_generation_recipes WHERE batch_id = ?",
        (batch.batch_id,),
    )
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )

    with pytest.raises(ValueError, match="vn_asset_recipe_not_found") as error:
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id,
            "slot_id": slot.id,
            "variant_index": 0,
            "batch_id": batch.batch_id,
            "user_id": 1,
        })
    assert error.value.code == "vn_asset_recipe_not_found"
    assert error.value.context["batch_id"] == batch.batch_id
    translated = vn_assets_endpoint._handle_value_error(error.value)
    assert translated.status_code == 404
    assert translated.detail == "vn_asset_recipe_not_found"
    assert adapter.requests == []
    assert service.repo.get_batch(batch.batch_id)["status"] == "failed"


@pytest.mark.asyncio
async def test_unknown_recipe_version_never_uses_mutable_sources(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    service.repo.db.execute_query(
        "UPDATE vn_asset_batches SET recipe_version = 99 WHERE id = ?",
        (batch.batch_id,),
    )
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
    )

    with pytest.raises(ValueError, match="vn_asset_recipe_version_unsupported"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id,
            "slot_id": slot.id,
            "variant_index": 0,
            "batch_id": batch.batch_id,
            "user_id": 1,
        })
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_completed_variant_redelivery_reuses_item_without_generating_again(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    adapter = FakeImageAdapter()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))

    storage = StoredVNSaver(tmp_path)
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=storage,
        generated_files_repo=storage,
    )
    payload = {
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    }

    first = await worker.handle_generate_variant(payload)
    service.repo.update_item_review(first["item_id"], review_status="approved", preferred=True)
    second = await worker.handle_generate_variant(payload)

    assert second == first
    assert len(adapter.requests) == 1
    assert len(service.repo.list_items(pack_with_slots.id)) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1
    assert service.repo.get_item(first["item_id"])["review_status"] == "approved"
    assert service.repo.get_item(first["item_id"])["preferred"] == 1


@pytest.mark.asyncio
async def test_cancelled_batch_does_not_publish_reserved_variant(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
        item_fields={"pack_id": pack_with_slots.id, "generated_file_id": 77},
    )
    service.repo.update_batch(batch.batch_id, {"status": "cancelled"})
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    with pytest.raises(ValueError, match="vn_asset_batch_terminal"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        })

    assert service.repo.get_item(item["id"])["review_status"] == "hidden"
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 0


def test_cancellation_before_publication_rejects_reserved_variant(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
        item_fields={"pack_id": pack_with_slots.id, "generated_file_id": 77},
    )
    service.repo.update_batch(batch.batch_id, {"status": "cancelled"})

    with pytest.raises(ValueError, match="vn_asset_batch_terminal"):
        service.repo.complete_variant(
            batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
            item_id=item["id"],
        )

    assert service.repo.get_item(item["id"])["review_status"] == "hidden"
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 0


@pytest.mark.asyncio
async def test_failed_variant_does_not_strand_queued_sibling(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FailOnceAdapter(FakeImageAdapter):
        def generate(self, request: Any) -> ImageGenResult:
            if not self.requests:
                self.requests.append(request)
                raise RuntimeError("first variant failed")
            return super().generate(request)

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id], variant_count=2),
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FailOnceAdapter()),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {
        "pack_id": pack_with_slots.id, "slot_id": slot.id,
        "batch_id": batch.batch_id, "user_id": 1,
    }

    with pytest.raises(RuntimeError, match="first variant failed"):
        await worker.handle_generate_variant({**payload, "variant_index": 0})
    after_failure = service.repo.get_batch(batch.batch_id)
    sibling = await worker.handle_generate_variant({**payload, "variant_index": 1})

    assert after_failure["status"] == "processing"
    assert sibling["status"] == "draft_created"
    assert service.repo.get_batch(batch.batch_id)["status"] == "failed"
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1
    assert service.repo.get_batch(batch.batch_id)["failed_count"] == 1


@pytest.mark.asyncio
async def test_late_failure_does_not_regress_completed_variant(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    await worker.handle_generate_variant({
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    })

    service.repo.fail_variant(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0, error="late failure"
    )

    assert service.repo.get_slot(slot.id)["status"] == "reviewing"
    assert service.repo.get_batch(batch.batch_id)["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", [KeyboardInterrupt, RuntimeError])
async def test_retry_recovers_registered_file_after_worker_interruption(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interruption: type[BaseException],
) -> None:
    from tldw_Server_API.app.core.VN_Assets import worker as worker_module
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    adapter = FakeImageAdapter()
    saver = RecordingVNSaver()
    saved_file = tmp_path / "recovered.png"
    saved_file.write_bytes(adapter.content)
    monkeypatch.setattr(
        worker_module, "resolve_vn_asset_storage_path", lambda **_kwargs: saved_file
    )

    class RegisteredFiles:
        async def get_file_by_source_ref(
            self, *, user_id: int, source_feature: str, source_ref: str
        ) -> dict[str, Any] | None:
            if not saver.calls:
                return None
            return {
                "id": 77,
                "user_id": user_id,
                "source_feature": source_feature,
                "source_ref": source_ref,
                "storage_path": "vn_assets/2026/04/24/generated.png",
                "mime_type": "image/png",
                "file_size_bytes": len(adapter.content),
                "is_deleted": False,
            }

    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
        generated_files_repo=RegisteredFiles(),
    )
    original_update = service.repo.update_item_storage

    def interrupted_update(*args: Any, **kwargs: Any) -> None:
        raise interruption("worker interrupted after file registration")

    monkeypatch.setattr(service.repo, "update_item_storage", interrupted_update)
    payload = {
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    }
    with pytest.raises((KeyboardInterrupt, ValueError)):
        await worker.handle_generate_variant(payload)
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "planned"
    monkeypatch.setattr(service.repo, "update_item_storage", original_update)
    event_loop_thread = threading.get_ident()
    file_check_threads: list[int] = []
    original_is_file = Path.is_file

    def checked_is_file(path: Path) -> bool:
        if path == saved_file:
            file_check_threads.append(threading.get_ident())
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", checked_is_file)

    result = await worker.handle_generate_variant(payload)

    assert len(file_check_threads) == 1
    assert file_check_threads[0] != event_loop_thread
    assert result["generated_file_id"] == 77
    assert len(adapter.requests) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1
    assert service.repo.list_items(pack_with_slots.id)[0]["review_status"] == "draft"


@pytest.mark.asyncio
async def test_replay_rejects_registered_file_owned_by_another_user(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id,
        slot_id=slot.id,
        variant_index=0,
        item_fields={"pack_id": pack_with_slots.id},
    )

    class ForeignFiles:
        async def get_file_by_source_ref(
            self, *, user_id: int, source_feature: str, source_ref: str
        ) -> dict[str, Any]:
            return {
                "id": 77,
                "user_id": 2,
                "source_feature": source_feature,
                "source_ref": source_ref,
                "is_deleted": False,
            }

    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        generated_files_repo=ForeignFiles(),
    )
    with pytest.raises(VNAssetGenerationError) as raised:
        await worker._replay_variant(
            batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
            user_id=1, pack_id=pack_with_slots.id,
        )
    assert raised.value.retryable is True
    assert service.repo.get_item(item["id"])["generated_file_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("attached,invalid", [
    (attached, invalid) for attached in (False, True)
    for invalid in ("missing", "zero", "truncated", "owner", "deleted", "id", "feature", "source", "outside")
] + [(True, "attached_id"), (True, "attached_path")])
async def test_planned_replay_rejects_invalid_storage_without_generation_or_publication(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attached: bool, invalid: str,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0, item_fields={"pack_id": pack_with_slots.id},
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: outputs))
    image = outputs / "replay.png"
    image.write_bytes(b"12345678")
    record = {
        "id": 77, "user_id": 1, "source_feature": "vn_assets",
        "source_ref": f"vn_asset_item:{item['id']}", "is_deleted": False,
        "storage_path": "replay.png", "file_size_bytes": 8, "mime_type": "image/png",
    }
    if attached:
        service.repo.update_item_storage(
            item["id"], generated_file_id=77, storage_ref="replay.png", mime_type="image/png",
            width=512, height=512, bytes=8,
        )
    if invalid == "missing":
        image.unlink()
    elif invalid in {"zero", "truncated"}:
        image.write_bytes(b"" if invalid == "zero" else b"1234")
    elif invalid == "outside":
        (tmp_path / "outside.png").write_bytes(b"12345678")
        record["storage_path"] = "../outside.png"
    else:
        record.update({
            "owner": {"user_id": 2}, "deleted": {"is_deleted": True}, "id": {"id": 0},
            "feature": {"source_feature": "image_generation"}, "source": {"source_ref": "vn_asset_item:999"},
            "attached_id": {"id": 78}, "attached_path": {"storage_path": "other.png"},
        }[invalid])

    class Files:
        async def get_file_by_source_ref(self, **_kwargs: Any) -> dict[str, Any]:
            return record

        async def get_file_by_id(self, _file_id: int) -> dict[str, Any]:
            return record

    adapter = FakeImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, generated_files_repo=Files(),
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(), save_vn_asset_image=saver,
    )
    with pytest.raises(VNAssetGenerationError) as raised:
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
            "batch_id": batch.batch_id, "user_id": 1,
        })
    assert raised.value.retryable is True
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "planned"
    assert service.repo.get_item(item["id"])["review_status"] == "hidden"
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 0
    assert service.repo.list_items(pack_with_slots.id) == []
    assert adapter.requests == []
    assert saver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_completed_replay_validates_bytes_without_changing_approved_review(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: bool,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0, item_fields={"pack_id": pack_with_slots.id},
    )
    image = tmp_path / "completed.png"
    image.write_bytes(b"12345678")
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))
    service.repo.update_item_storage(
        item["id"], generated_file_id=77, storage_ref="completed.png", mime_type="image/png",
        width=512, height=512, bytes=8,
    )
    service.repo.complete_variant(batch_id=batch.batch_id, slot_id=slot.id, variant_index=0, item_id=item["id"])
    service.repo.update_item_review(item["id"], review_status="approved", preferred=True)
    before = service.repo.get_item(item["id"])
    if missing:
        image.unlink()

    class Files:
        async def get_file_by_id(self, _file_id: int) -> dict[str, Any]:
            return {
                "id": 77, "user_id": 1, "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{item['id']}", "is_deleted": False,
                "storage_path": "completed.png", "file_size_bytes": 8,
            }

    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs, generated_files_repo=Files())
    payload = {"pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
               "batch_id": batch.batch_id, "user_id": 1}
    if missing:
        with pytest.raises(VNAssetGenerationError) as raised:
            await worker.handle_generate_variant(payload)
        assert raised.value.retryable is True
    else:
        assert (await worker.handle_generate_variant(payload))["item_id"] == item["id"]
    assert service.repo.get_item(item["id"]) == before
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1


@pytest.mark.asyncio
async def test_failed_variant_redelivery_does_not_increment_failure_count(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FailingAdapter:
        def generate(self, _request: Any) -> None:
            raise RuntimeError("provider failed")

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FailingAdapter()),
        backend_gate=FakeGenerationGate(),
    )
    payload = {
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    }

    with pytest.raises(RuntimeError, match="provider failed"):
        await worker.handle_generate_variant(payload)
    with pytest.raises(ValueError, match="vn_asset_batch_terminal"):
        await worker.handle_generate_variant(payload)

    stored = service.repo.get_batch(batch.batch_id)
    assert stored["failed_count"] == 1
    assert stored["status"] == "failed"


@pytest.mark.asyncio
async def test_versioned_batch_counts_only_committed_variants(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id], variant_count=2),
    )
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))
    storage = StoredVNSaver(tmp_path)
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=storage,
        generated_files_repo=storage,
    )
    payload = {
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "batch_id": batch.batch_id,
        "user_id": 1,
    }

    await worker.handle_generate_variant({**payload, "variant_index": 0})
    midpoint = service.repo.get_batch(batch.batch_id)
    await worker.handle_generate_variant({**payload, "variant_index": 1})
    await worker.handle_generate_variant({**payload, "variant_index": 0})

    assert midpoint["completed_count"] == 1
    assert midpoint["status"] == "processing"
    stored = service.repo.get_batch(batch.batch_id)
    assert stored["completed_count"] == 2
    assert stored["status"] == "completed"


@pytest.mark.asyncio
async def test_generate_variant_rolls_back_item_when_file_persistence_fails(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.repo.create_batch(
        pack_id=pack_with_slots.id,
        requested_by_user_id=1,
        status="enqueued",
        total_slots=1,
        total_variants=1,
        planned_count=1,
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=FailingVNSaver(),
    )

    with pytest.raises(RuntimeError, match="storage failed"):
        await worker.handle_generate_variant(
            {
                "pack_id": pack_with_slots.id,
                "slot_id": slot.id,
                "variant_index": 0,
                "batch_id": batch["id"],
                "user_id": 1,
            }
        )

    assert service.repo.list_items(pack_with_slots.id) == []


@pytest.mark.asyncio
async def test_generate_variant_includes_pack_world_book_context(
    chacha_db: CharactersRAGDB,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    world_books = WorldBookService(chacha_db)
    world_book_id = world_books.create_world_book("Archive Lore")
    world_books.add_entry(
        world_book_id=world_book_id,
        keywords=["archive"],
        content="Orbital archive doors glow blue.",
        priority=10,
    )
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(
        VNAssetPackCreate(
            title="Lore Pack",
            primary_character_id=character_id,
            source_world_book_ids=[world_book_id],
        )
    )
    slot = service.create_slot(
        pack.id,
        VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary", variant_count=1),
    )
    batch = service.repo.create_batch(
        pack_id=pack.id,
        requested_by_user_id=1,
        status="enqueued",
        total_slots=1,
        total_variants=1,
        planned_count=1,
    )
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )

    await worker.handle_generate_variant(
        {
            "pack_id": pack.id,
            "slot_id": slot.id,
            "variant_index": 0,
            "batch_id": batch["id"],
            "user_id": 1,
        }
    )

    assert "Orbital archive doors glow blue." in adapter.requests[0].prompt


@pytest.mark.asyncio
async def test_terminal_batch_cancels_remaining_jobs_and_skips_generation(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.jobs import vn_asset_batch_group
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.repo.create_batch(
        pack_id=pack_with_slots.id,
        requested_by_user_id=1,
        status="cancelled",
        total_slots=1,
        total_variants=1,
        planned_count=1,
    )
    fake_jobs.created.append(
        {
            "id": 10,
            "status": "queued",
            "domain": "vn_assets",
            "batch_group": vn_asset_batch_group(user_id=1, pack_id=pack_with_slots.id, batch_id=batch["id"]),
        }
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )

    with pytest.raises(ValueError, match="vn_asset_batch_terminal"):
        await worker.handle_generate_variant(
            {
                "pack_id": pack_with_slots.id,
                "slot_id": slot.id,
                "variant_index": 0,
                "batch_id": batch["id"],
                "user_id": 1,
            },
            job={"id": 99},
        )

    assert fake_jobs.cancelled_ids == [10]
    assert service.repo.list_items(pack_with_slots.id) == []


def test_record_generation_success_preserves_terminal_batch_state(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    batch = service.repo.create_batch(
        pack_id=pack_with_slots.id,
        requested_by_user_id=1,
        status="failed",
        total_slots=1,
        total_variants=2,
        planned_count=2,
    )
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    worker._record_generation_success(batch_id=batch["id"])

    updated = service.repo.get_batch(batch["id"])
    assert updated["status"] == "failed"
    assert updated["completed_count"] == 1
    assert updated["completed_at"] is None


@pytest.mark.asyncio
async def test_generate_variant_offloads_sync_image_generation(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import worker as worker_module
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    to_thread_calls: list[tuple[Any, tuple[Any, ...]]] = []

    async def fake_to_thread(func: Any, /, *args: Any, **_kwargs: Any) -> Any:
        to_thread_calls.append((func, args))
        return func(*args)

    monkeypatch.setattr(
        worker_module,
        "asyncio",
        SimpleNamespace(to_thread=fake_to_thread),
        raising=False,
    )

    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    slot = pack_with_slots.slots[0]
    batch = service.repo.create_batch(
        pack_id=pack_with_slots.id,
        requested_by_user_id=1,
        status="enqueued",
        total_slots=1,
        total_variants=1,
        planned_count=1,
    )

    await worker.handle_generate_variant(
        {
            "pack_id": pack_with_slots.id,
            "slot_id": slot.id,
            "variant_index": 0,
            "batch_id": batch["id"],
            "user_id": 1,
        }
    )

    assert to_thread_calls == [(adapter.generate, (adapter.requests[0],))]


def test_approved_background_item_enqueues_lazy_depth_generation(
    service: VNAssetPackService,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Depth Pack", primary_character_id=character_id)
    )
    background = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="background",
            slot_key="background.interior",
            variant_count=1,
        ),
    )
    depth = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="depth_companion",
            slot_key="depth.interior",
            variant_count=0,
            required_for_runtime=False,
            depends_on_slot_id=background.id,
        ),
    )
    item = service.repo.create_item(
        pack_id=pack.id,
        slot_id=background.id,
        variant_index=0,
        review_status="draft",
    )

    service.review_item_for_pack(
        pack.id,
        int(item["id"]),
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )

    assert len(fake_jobs.created) == 1
    job = fake_jobs.created[0]
    assert job["job_type"] == "vn_asset_enqueue_batch"
    batch = service.repo.get_batch(job["payload"]["batch_id"])
    assert batch is not None
    assert batch["total_variants"] == 1
    assert '"variant_count": 1' in batch["options_json"]
    assert f'"slot_ids": [{depth.id}]' in batch["options_json"]


def test_lazy_depth_generation_does_not_duplicate_active_depth_batch(
    service: VNAssetPackService,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Depth Pack", primary_character_id=character_id)
    )
    background = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="background",
            slot_key="background.interior",
            variant_count=1,
        ),
    )
    service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="depth_companion",
            slot_key="depth.interior",
            variant_count=0,
            required_for_runtime=False,
            depends_on_slot_id=background.id,
        ),
    )
    item = service.repo.create_item(
        pack_id=pack.id,
        slot_id=background.id,
        variant_index=0,
        review_status="draft",
    )

    service.review_item_for_pack(
        pack.id,
        int(item["id"]),
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )
    service.review_item_for_pack(
        pack.id,
        int(item["id"]),
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )

    assert len(fake_jobs.created) == 1


def test_lazy_depth_generation_treats_full_pack_batch_as_active(
    service: VNAssetPackService,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Depth Pack", primary_character_id=character_id)
    )
    background = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="background",
            slot_key="background.interior",
            variant_count=1,
        ),
    )
    service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="depth_companion",
            slot_key="depth.interior",
            variant_count=0,
            required_for_runtime=False,
            depends_on_slot_id=background.id,
        ),
    )
    service.repo.create_batch(
        pack_id=pack.id,
        requested_by_user_id=1,
        status="queued",
        options={},
    )
    item = service.repo.create_item(
        pack_id=pack.id,
        slot_id=background.id,
        variant_index=0,
        review_status="draft",
    )

    service.review_item_for_pack(
        pack.id,
        int(item["id"]),
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )

    assert fake_jobs.created == []


def test_failed_fanout_preserves_full_planned_count(
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    failing_jobs = FailingChildJobs(fail_after_children=2)
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=failing_jobs)

    with pytest.raises(ValueError, match="child job quota exceeded"):
        worker.handle_enqueue_batch(batch_with_slots.job_payload)

    batch = service.repo.get_batch(batch_with_slots.id)
    assert batch is not None
    assert batch["status"] == "queued"
    assert batch["planned_count"] == sum(slot.variant_count for slot in batch_with_slots.slots)
    assert batch["enqueued_count"] == 2
    assert batch["enqueue_error"] == "child job quota exceeded"

    failing_jobs.fail_after_children = 100
    recovered = worker.handle_enqueue_batch(batch_with_slots.job_payload)
    assert recovered["enqueued_count"] == batch["planned_count"]
    assert len(failing_jobs.created) == batch["planned_count"]
    assert service.repo.get_batch(batch_with_slots.id)["status"] == "enqueued"
    assert service.repo.get_batch(batch_with_slots.id)["enqueue_error"] is None


@pytest.mark.asyncio
async def test_worker_entrypoint_requires_job_owner_before_opening_user_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import vn_asset_jobs_worker

    async def fail_get_db(*_args: Any, **_kwargs: Any) -> CharactersRAGDB:
        raise AssertionError("user database should not be opened")

    monkeypatch.setattr(vn_asset_jobs_worker, "get_chacha_db_for_user_id", fail_get_db)

    with pytest.raises(ValueError, match="missing_owner_user_id"):
        await vn_asset_jobs_worker.handle_vn_asset_job(
            {
                "job_type": "vn_asset_enqueue_batch",
                "payload": {"pack_id": 1, "batch_id": 1, "user_id": 1},
            }
        )


@pytest.mark.asyncio
async def test_worker_entrypoint_rejects_payload_owner_mismatch_before_opening_user_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import vn_asset_jobs_worker

    async def fail_get_db(*_args: Any, **_kwargs: Any) -> CharactersRAGDB:
        raise AssertionError("user database should not be opened")

    monkeypatch.setattr(vn_asset_jobs_worker, "get_chacha_db_for_user_id", fail_get_db)

    with pytest.raises(ValueError, match="vn_asset_job_owner_mismatch"):
        await vn_asset_jobs_worker.handle_vn_asset_job(
            {
                "job_type": "vn_asset_enqueue_batch",
                "owner_user_id": "1",
                "payload": {"pack_id": 1, "batch_id": 1, "user_id": 2},
            }
        )


def test_generation_api_enqueues_parent_job(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="API Generated Pack", primary_character_id=character_id))
    slots = service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    planned_count = sum(slot.variant_count for slot in slots)
    fake_jobs.created.clear()

    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(id=1, username="vn-generator")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    def override_job_manager() -> FakeJobs:
        return fake_jobs

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    job_manager_dep = getattr(vn_assets_endpoint, "_job_manager", None)
    if job_manager_dep is not None:
        app.dependency_overrides[job_manager_dep] = override_job_manager

    client = TestClient(app)
    missing_key_response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/generate",
        json={},
    )
    generate_response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/generate",
        json={"idempotency_key": "api-generate-parent-job"},
    )

    assert missing_key_response.status_code in {400, 422}
    assert generate_response.status_code == 202
    assert generate_response.json()["status"] == "queued"
    assert len(fake_jobs.created) == 1

    status_response = client.get(f"/api/v1/vn/vn-assets/packs/{pack.id}/generation")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["batch_id"] == generate_response.json()["batch_id"]
    assert status_payload["planned_count"] == planned_count
    assert status_payload["enqueued_count"] == 0


def test_generation_api_replays_same_idempotency_key_and_conflicts_on_different_payload(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Idempotent Pack", primary_character_id=character_id))
    service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    fake_jobs.created.clear()

    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(id=1, username="vn-generator")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs

    client = TestClient(app)
    payload = {"idempotency_key": "generate-pack-1", "variant_count": 1}
    first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/generate", json=payload)
    replay = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/generate", json=payload)
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/generate",
        json={"idempotency_key": "generate-pack-1", "variant_count": 2},
    )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert service.repo.get_idempotency_record(
        owner_user_id=1, scope="vn_asset_generate", resource_id=f"pack:{pack.id}",
        idempotency_key="generate-pack-1",
    )["batch_id"] == first.json()["batch_id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_generation_api_recovers_unfinished_response_receipt(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Receipt Pack", primary_character_id=character_id))
    service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(id=1, username="vn-generator")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs
    original_record = vn_assets_endpoint._record_idempotency_response

    def lost_response(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("response lost after batch commit")

    monkeypatch.setattr(vn_assets_endpoint, "_record_idempotency_response", lost_response)
    client = TestClient(app, raise_server_exceptions=False)
    payload = {"idempotency_key": "recover-receipt-1"}
    first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/generate", json=payload)
    monkeypatch.setattr(vn_assets_endpoint, "_record_idempotency_response", original_record)
    second = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/generate", json=payload)

    assert first.status_code == 500
    assert second.status_code == 202
    assert second.json()["batch_id"] == service.repo.list_batches(pack.id)[0]["id"]
    assert len(service.repo.list_batches(pack.id)) == 1
    assert len(fake_jobs.created) == 1


def test_generation_receipt_recovers_parent_job_after_interruption(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import service as service_module

    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Queued Pack", primary_character_id=character_id))
    slot = service.create_slot(
        pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="primary", variant_count=1)
    )
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack.id}",
        "idempotency_key": "interrupted-parent-1",
        "payload_hash": "test-payload-hash",
    }
    service.repo.claim_idempotency_record(owner_user_id=1, **receipt)
    original_enqueue = service_module.create_enqueue_batch_job

    def interrupted_enqueue(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt("worker stopped before parent enqueue")

    monkeypatch.setattr(service_module, "create_enqueue_batch_job", interrupted_enqueue)
    with pytest.raises(KeyboardInterrupt):
        service.start_generation(
            pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]),
            idempotency_receipt=receipt,
        )
    monkeypatch.setattr(service_module, "create_enqueue_batch_job", original_enqueue)
    record = service.repo.get_idempotency_record(owner_user_id=1, **{
        key: receipt[key] for key in ("scope", "resource_id", "idempotency_key")
    })
    assert record is not None
    assert record["batch_id"] is not None
    assert fake_jobs.created == []

    recovered = service.recover_generation_receipt(record, pack_id=pack.id, jobs_manager=fake_jobs)
    again = service.recover_generation_receipt(record, pack_id=pack.id, jobs_manager=fake_jobs)

    assert recovered.batch_id == record["batch_id"]
    assert again.batch_id == recovered.batch_id
    assert len(service.repo.list_batches(pack.id)) == 1
    assert len(fake_jobs.created) == 1

    other_service = VNAssetPackService(chacha_db, owner_user_id=2, jobs_manager=fake_jobs)
    other_pack = other_service.create_pack(
        VNAssetPackCreate(title="Other Pack", primary_character_id=character_id)
    )
    with pytest.raises(ValueError, match="vn_asset_generation_receipt_not_found"):
        other_service.recover_generation_receipt(record, pack_id=other_pack.id)


def test_retry_slot_api_replays_same_idempotency_key_and_conflicts_on_different_payload(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Retry Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    fake_jobs.created.clear()

    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(id=1, username="vn-generator")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs

    client = TestClient(app)
    payload = {"idempotency_key": "retry-slot-1", "variant_count": 1}
    first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry", json=payload)
    replay = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry", json=payload)
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry",
        json={"idempotency_key": "retry-slot-1", "variant_count": 2},
    )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert service.repo.get_idempotency_record(
        owner_user_id=1, scope="vn_asset_slot_retry",
        resource_id=f"pack:{pack.id}:slot:{slot.id}", idempotency_key="retry-slot-1",
    )["batch_id"] == first.json()["batch_id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_regenerate_item_api_replays_same_idempotency_key_and_conflicts_on_different_payload(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Regenerate Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    item = service.repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=0)
    fake_jobs.created.clear()

    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(id=1, username="vn-generator")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs

    client = TestClient(app)
    payload = {"idempotency_key": "regenerate-item-1", "variant_count": 1}
    first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/items/{item['id']}/regenerate", json=payload)
    replay = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/items/{item['id']}/regenerate", json=payload)
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/{item['id']}/regenerate",
        json={"idempotency_key": "regenerate-item-1", "variant_count": 2},
    )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert service.repo.get_idempotency_record(
        owner_user_id=1, scope="vn_asset_item_regenerate",
        resource_id=f"pack:{pack.id}:item:{item['id']}", idempotency_key="regenerate-item-1",
    )["batch_id"] == first.json()["batch_id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"
