from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import vn_assets as vn_assets_endpoint
from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetGenerationRequest,
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenResult
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.concurrency import BackendGenerationLease
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


@pytest.fixture
def chacha_db(tmp_path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-assets-jobs-test-client")
    yield database
    database.close_connection()


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


def test_generation_marks_batch_failed_when_parent_enqueue_is_rejected(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="queued job quota exceeded"):
        service.start_generation(
            pack_with_slots.id,
            user_id=1,
            jobs_manager=RejectingJobs(),
        )

    batches = service.repo.list_batches(pack_with_slots.id)
    assert len(batches) == 1
    assert batches[0]["status"] == "failed"
    assert batches[0]["enqueue_error"] == "queued job quota exceeded"


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

    with pytest.raises(ValueError, match="vn_asset_recipe_not_found"):
        await worker.handle_generate_variant({
            "pack_id": pack_with_slots.id,
            "slot_id": slot.id,
            "variant_index": 0,
            "batch_id": batch.batch_id,
            "user_id": 1,
        })
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
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    }

    first = await worker.handle_generate_variant(payload)
    second = await worker.handle_generate_variant(payload)

    assert second == first
    assert len(adapter.requests) == 1
    assert len(service.repo.list_items(pack_with_slots.id)) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1


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
async def test_retry_recovers_registered_file_after_worker_interruption(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
        raise KeyboardInterrupt("worker interrupted after file registration")

    monkeypatch.setattr(service.repo, "update_item_storage", interrupted_update)
    payload = {
        "pack_id": pack_with_slots.id,
        "slot_id": slot.id,
        "variant_index": 0,
        "batch_id": batch.batch_id,
        "user_id": 1,
    }
    with pytest.raises(KeyboardInterrupt):
        await worker.handle_generate_variant(payload)
    monkeypatch.setattr(service.repo, "update_item_storage", original_update)

    result = await worker.handle_generate_variant(payload)

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
    result = await worker._replay_variant(
        batch_id=batch.batch_id,
        slot_id=slot.id,
        variant_index=0,
        user_id=1,
        pack_id=pack_with_slots.id,
    )

    assert result is None
    assert service.repo.get_item(item["id"])["generated_file_id"] is None


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
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id], variant_count=2),
    )
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
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
    assert batch["status"] == "failed"
    assert batch["planned_count"] == sum(slot.variant_count for slot in batch_with_slots.slots)
    assert batch["enqueued_count"] == 2
    assert batch["enqueue_error"] == "child job quota exceeded"


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
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


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
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"
