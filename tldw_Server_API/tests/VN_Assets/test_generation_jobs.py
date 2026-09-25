from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Generator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

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

pytestmark = pytest.mark.integration


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
    assert result.selected_slot_ids == [slot.id for slot in pack_with_slots.slots]
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


def test_generation_acceptance_freezes_authored_recipe(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(VNAssetPackCreate(
        title="Recipe Pack", primary_character_id=character_id,
        style_prompt="original watercolor", default_backend="stable_diffusion_cpp",
        default_dimensions={"width": 640, "height": 480, "steps": 18},
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
        seed_policy={"base_seed": 101},
    ))

    status = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    stored = service.repo.get_batch(status.batch_id)
    recipe = json.loads(stored["recipe_json"])
    service.repo.update_pack(pack.id, {"style_prompt": "edited oil", "default_backend": "new-backend"})
    service.repo.update_slot(slot.id, {"variant_count": 4, "prompt_template": "edited template"})

    assert recipe["version"] == 1
    assert recipe["pack_id"] == pack.id
    assert recipe["owner_user_id"] == 1
    assert recipe["slots"][0]["variant_count"] == 2
    assert recipe["slots"][0]["requested_backend"] == "stable_diffusion_cpp"
    assert recipe["slots"][0]["width"] == 640
    assert recipe["slots"][0]["seeds"] == [101, 102]
    assert "original watercolor" in recipe["slots"][0]["prompt_snapshot"]["prompt"]


@pytest.mark.asyncio
async def test_worker_uses_accepted_recipe_after_pack_and_slot_edits(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(
        title="Frozen", primary_character_id=character_id,
        style_prompt="original watercolor", default_backend="stable_diffusion_cpp",
        default_dimensions={"width": 640, "height": 480, "steps": 18},
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
        seed_policy={"base_seed": 101},
    ))
    status = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    service.repo.update_pack(pack.id, {"style_prompt": "edited oil", "default_backend": "changed"})
    service.repo.update_slot(slot.id, {"variant_count": 4, "prompt_template": "edited template"})
    character = service.repo.get_character(character_id)
    service.repo.db.update_character_card(
        character_id, {"description": "An edited cartographer."},
        expected_version=int(character["version"]),
    )
    adapter = FakeImageAdapter()
    registry = FakeImageRegistry(adapter)
    gate = FakeGenerationGate()
    worker = VNAssetGenerationWorker(
        repo=VNAssetPacksRepository.initialized(service.repo.db),
        jobs_manager=fake_jobs, image_registry=registry,
        backend_gate=gate, save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": status.batch_id, "user_id": 1}

    worker.handle_enqueue_batch(payload)
    worker.handle_enqueue_batch(payload)
    child = next(job for job in fake_jobs.created if job["job_type"] == "vn_asset_generate_variant")
    await worker.handle_generate_variant(child["payload"])

    assert len([job for job in fake_jobs.created if job["job_type"] == "vn_asset_generate_variant"]) == 2
    assert len(adapter.requests) == 1
    assert "original watercolor" in adapter.requests[0].prompt
    assert "A careful archivist." in adapter.requests[0].prompt
    assert "An edited cartographer." not in adapter.requests[0].prompt
    assert "edited oil" not in adapter.requests[0].prompt
    assert adapter.requests[0].width == 640
    assert adapter.requests[0].seed == 101
    assert gate.requests[0][0] == "stable_diffusion_cpp"
    assert (
        json.loads(service.repo.get_batch(status.batch_id)["execution_recipe_json"])["slots"][0]["backend"]
        == "stable_diffusion_cpp"
    )


def test_backend_resolution_failure_marks_batch_failed(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    status = service.start_generation(pack_with_slots.id)
    registry = FakeImageRegistry(FakeImageAdapter())
    registry.resolve_backend = lambda _requested: None  # type: ignore[method-assign]
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs, image_registry=registry)

    with pytest.raises(ValueError, match="image_backend_unavailable"):
        worker.handle_enqueue_batch({"pack_id": pack_with_slots.id, "batch_id": status.batch_id, "user_id": 1})

    batch = service.repo.get_batch(status.batch_id)
    assert batch["status"] == "failed"
    assert batch["enqueue_error"] == "image_backend_unavailable"


def test_duplicate_parent_delivery_does_not_reopen_terminal_batch(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    status = service.start_generation(pack_with_slots.id)
    payload = {"pack_id": pack_with_slots.id, "batch_id": status.batch_id, "user_id": 1}
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
    )
    worker.handle_enqueue_batch(payload)
    created_count = len(fake_jobs.created)
    service.repo.update_batch(status.batch_id, {"status": "completed"})

    result = worker.handle_enqueue_batch(payload)

    assert result["status"] == "completed"
    assert service.repo.get_batch(status.batch_id)["status"] == "completed"
    assert len(fake_jobs.created) == created_count


def test_parent_retry_resumes_partial_fanout_after_transient_enqueue_failure(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    status = service.start_generation(pack_with_slots.id)
    child_jobs = FailingChildJobs(fail_after_children=1)
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=child_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
    )
    payload = {"pack_id": pack_with_slots.id, "batch_id": status.batch_id, "user_id": 1}

    with pytest.raises(ValueError, match="child job quota exceeded"):
        worker.handle_enqueue_batch(payload)
    assert service.repo.get_batch(status.batch_id)["status"] == "failed"
    child_jobs.fail_after_children = 1000
    resumed = worker.handle_enqueue_batch(payload)

    assert resumed["status"] == "enqueued"
    assert resumed["enqueued_count"] == status.planned_count
    assert service.repo.get_batch(status.batch_id)["enqueue_error"] is None
    assert len(child_jobs.created) == status.planned_count


@pytest.mark.asyncio
async def test_fanout_retry_completes_batch_when_all_children_already_finished(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class PersistThenFailJobs(FakeJobs):
        failed = False

        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            job = super().create_job(**kwargs)
            if kwargs.get("job_type") == "vn_asset_generate_variant" and not self.failed:
                self.failed = True
                raise RuntimeError("response lost after child insert")
            return job

    pack = service.create_pack(VNAssetPackCreate(title="Fanout Recovery", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    jobs = PersistThenFailJobs()
    status = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]), jobs_manager=jobs)
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": status.batch_id, "user_id": 1}

    with pytest.raises(RuntimeError, match="response lost"):
        worker.handle_enqueue_batch(payload)
    child = next(job for job in jobs.created if job["job_type"] == "vn_asset_generate_variant")
    await worker.handle_generate_variant(child["payload"])
    resumed = worker.handle_enqueue_batch(payload)

    assert resumed["status"] == "completed"
    assert service.repo.get_batch(status.batch_id)["completed_count"] == 1
    assert len([job for job in jobs.created if job["job_type"] == "vn_asset_generate_variant"]) == 1


@pytest.mark.asyncio
async def test_queued_child_can_finish_during_retryable_parent_fanout_failure(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    status = service.start_generation(pack_with_slots.id)
    child_jobs = FailingChildJobs(fail_after_children=1)
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=child_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack_with_slots.id, "batch_id": status.batch_id, "user_id": 1}
    with pytest.raises(ValueError, match="child job quota exceeded"):
        worker.handle_enqueue_batch(payload)
    child = child_jobs.created[0]

    result = await worker.handle_generate_variant(child["payload"])

    assert result["status"] == "draft_created"
    assert len(adapter.requests) == 1


def test_parent_fanout_does_not_overwrite_child_terminal_status(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Concurrent", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=1,
    ))
    status = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))

    class CompletingJobs(FakeJobs):
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            job = super().create_job(**kwargs)
            if kwargs.get("job_type") == "vn_asset_generate_variant":
                service.repo.update_batch(status.batch_id, {"status": "completed"})
            return job

    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=CompletingJobs(),
        image_registry=FakeImageRegistry(FakeImageAdapter()),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": status.batch_id, "user_id": 1})

    assert service.repo.get_batch(status.batch_id)["status"] == "completed"


def test_late_parent_exception_does_not_reopen_completed_batch(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    status = service.start_generation(pack_with_slots.id)
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
    )

    def fail_after_child_completion(*_args: Any, **_kwargs: Any) -> None:
        service.repo.update_batch(status.batch_id, {"status": "completed"})
        raise RuntimeError("late fanout error")

    monkeypatch.setattr(service.repo, "complete_batch_fanout", fail_after_child_completion)
    with pytest.raises(RuntimeError, match="late fanout error"):
        worker.handle_enqueue_batch({
            "pack_id": pack_with_slots.id, "batch_id": status.batch_id, "user_id": 1,
        })

    assert service.repo.get_batch(status.batch_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_worker_pins_implicit_hosted_model_before_child_execution(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(
        title="Hosted Model", primary_character_id=character_id, default_backend="openrouter",
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=1,
    ))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "provider/model-a")
    monkeypatch.setenv("OPENROUTER_IMAGE_API_KEY", "private-test-credential")
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": batch.batch_id, "user_id": 1}
    worker.handle_enqueue_batch(payload)
    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "provider/model-b")
    worker.handle_enqueue_batch(payload)
    await worker.handle_generate_variant({**payload, "slot_id": slot.id, "variant_index": 0})

    assert adapter.requests[0].model == "provider/model-a"
    assert (
        json.loads(service.repo.get_batch(batch.batch_id)["execution_recipe_json"])["slots"][0]["model"]
        == "provider/model-a"
    )
    assert "private-test-credential" not in json.dumps(service.repo.get_batch(batch.batch_id))


@pytest.mark.asyncio
async def test_worker_rejects_changed_implicit_local_model_without_storing_path(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import worker as worker_module

    pack = service.create_pack(VNAssetPackCreate(
        title="Local Model", primary_character_id=character_id,
        default_backend="stable_diffusion_cpp",
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=1,
    ))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    config = worker_module.get_image_generation_config()
    monkeypatch.setattr(worker_module, "get_image_generation_config", lambda: replace(
        config, sd_cpp_diffusion_model_path=None, sd_cpp_model_path="/private/first-model.gguf",
    ))
    adapter = FakeImageAdapter()
    worker = worker_module.VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": batch.batch_id, "user_id": 1}
    worker.handle_enqueue_batch(payload)
    stored = service.repo.get_batch(batch.batch_id)["execution_recipe_json"]
    monkeypatch.setattr(worker_module, "get_image_generation_config", lambda: replace(
        config, sd_cpp_diffusion_model_path=None, sd_cpp_model_path="/private/second-model.gguf",
    ))

    with pytest.raises(ValueError, match="vn_asset_local_model_changed"):
        await worker.handle_generate_variant({**payload, "slot_id": slot.id, "variant_index": 0})
    assert "/private/first-model.gguf" not in stored
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_implicit_local_model_path_is_not_saved_in_item_metadata(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import worker as worker_module

    pack = service.create_pack(VNAssetPackCreate(
        title="Private Local Model", primary_character_id=character_id,
        default_backend="stable_diffusion_cpp",
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=1,
    ))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    private_path = "/private/secret-local-model.gguf"
    config = worker_module.get_image_generation_config()
    monkeypatch.setattr(worker_module, "get_image_generation_config", lambda: replace(
        config, sd_cpp_diffusion_model_path=None, sd_cpp_model_path=private_path,
    ))
    adapter = FakeImageAdapter()
    worker = worker_module.VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": batch.batch_id, "user_id": 1}
    worker.handle_enqueue_batch(payload)
    result = await worker.handle_generate_variant({**payload, "slot_id": slot.id, "variant_index": 0})

    assert adapter.requests[0].model == private_path
    assert private_path not in json.dumps(service.repo.get_item(result["item_id"]))


def test_retry_copies_failed_source_recipe_while_regenerate_reads_current_settings(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(VNAssetPackCreate(
        title="Retry Pack", primary_character_id=character_id, style_prompt="original watercolor",
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
    ))
    original = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    service.repo.update_batch(original.batch_id, {"status": "failed"})
    service.repo.update_slot(slot.id, {
        "status": "failed", "last_error": "provider failed", "last_failed_batch_id": original.batch_id,
    })
    service.repo.update_pack(pack.id, {"style_prompt": "edited oil"})

    retry = service.retry_slot(
        pack.id, slot.id, VNAssetGenerationRequest(source_batch_id=original.batch_id),
    )
    regenerate = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    original_recipe = json.loads(service.repo.get_batch(original.batch_id)["recipe_json"])
    retry_row = service.repo.get_batch(retry.batch_id)
    retry_recipe = json.loads(retry_row["recipe_json"])
    current_recipe = json.loads(service.repo.get_batch(regenerate.batch_id)["recipe_json"])

    assert retry.source_batch_id == original.batch_id
    assert retry_row["source_batch_id"] == original.batch_id
    assert retry_recipe["slots"] == original_recipe["slots"]
    assert "edited oil" in current_recipe["slots"][0]["prompt_snapshot"]["prompt"]


def test_retry_rejects_legacy_source_instead_of_using_current_settings(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    slot_id = pack_with_slots.slots[0].id
    legacy = service.repo.create_batch(
        pack_id=pack_with_slots.id, requested_by_user_id=1, status="failed",
        options={"slot_ids": [slot_id]},
    )

    with pytest.raises(ValueError, match="vn_asset_recipe_unavailable"):
        service.retry_slot(
            pack_with_slots.id, slot_id,
            VNAssetGenerationRequest(source_batch_id=legacy["id"]),
        )


def test_generation_status_reports_recipe_availability_for_each_failed_slot_source(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    slot_id = pack_with_slots.slots[0].id
    legacy = service.repo.create_batch(
        pack_id=pack_with_slots.id, requested_by_user_id=1, status="failed",
        options={"slot_ids": [slot_id]},
    )
    service.repo.update_slot(slot_id, {
        "status": "failed", "last_error": "legacy failure", "last_failed_batch_id": legacy["id"],
    })
    newer = service.start_generation(
        pack_with_slots.id, VNAssetGenerationRequest(slot_ids=[slot_id]),
    )
    service.repo.update_batch(newer.batch_id, {"status": "failed", "enqueue_error": "queue unavailable"})

    status = service.get_generation_status(pack_with_slots.id)

    assert status.recipe_available is True
    assert status.failed_slot_batch_ids[slot_id] == legacy["id"]
    assert status.failed_slot_recipe_available[slot_id] is False


def test_retry_without_source_uses_latest_failed_batch_for_that_slot(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    first_slot, second_slot = pack_with_slots.slots[:2]
    first = service.start_generation(
        pack_with_slots.id, VNAssetGenerationRequest(slot_ids=[first_slot.id]),
    )
    service.repo.update_batch(first.batch_id, {"status": "failed"})
    service.repo.update_slot(first_slot.id, {
        "status": "failed", "last_error": "provider failed", "last_failed_batch_id": first.batch_id,
    })
    second = service.start_generation(
        pack_with_slots.id, VNAssetGenerationRequest(slot_ids=[second_slot.id]),
    )
    service.repo.update_batch(second.batch_id, {"status": "failed"})
    service.repo.update_slot(second_slot.id, {
        "status": "failed", "last_error": "provider failed", "last_failed_batch_id": second.batch_id,
    })

    retry = service.retry_slot(pack_with_slots.id, first_slot.id)

    assert retry.source_batch_id == first.batch_id


def test_retry_uses_recorded_slot_failure_not_later_batch_that_only_selected_it(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Failure Provenance", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary",
    ))
    first = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
    )
    worker._record_generation_failure(batch_id=first.batch_id, slot_id=slot.id, error="provider failed")
    second = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    service.repo.update_batch(second.batch_id, {"status": "failed", "enqueue_error": "queue full"})

    status = service.get_generation_status(pack.id)
    with pytest.raises(ValueError, match="vn_asset_retry_source_unavailable"):
        service.retry_slot(pack.id, slot.id, VNAssetGenerationRequest(source_batch_id=second.batch_id))
    retry = service.retry_slot(pack.id, slot.id)

    assert status.failed_slot_batch_ids[slot.id] == first.batch_id
    assert retry.source_batch_id == first.batch_id


@pytest.mark.asyncio
async def test_late_variant_success_preserves_sibling_failure_for_retry(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Mixed Outcomes", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
    ))
    status = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": status.batch_id, "user_id": 1}
    worker.handle_enqueue_batch(payload)

    worker._record_generation_failure(batch_id=status.batch_id, slot_id=slot.id, error="provider failed")
    await worker._generate_variant(
        pack=service.repo.get_pack(pack.id), slot=service.repo.get_slot(slot.id),
        batch=service.repo.get_batch(status.batch_id), character=None,
        variant_index=1, user_id=1, job=None,
    )

    failed_slot = service.repo.get_slot(slot.id)
    assert failed_slot["status"] == "failed"
    assert failed_slot["last_error"] == "provider failed"
    assert failed_slot["last_failed_batch_id"] == status.batch_id
    assert service.retry_slot(pack.id, slot.id).source_batch_id == status.batch_id


@pytest.mark.parametrize("newer_failed", [False, True])
def test_older_batch_failure_cannot_replace_newer_slot_result(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
    newer_failed: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Ordered Outcomes", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    older = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    newer = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    if newer_failed:
        worker._record_generation_failure(batch_id=newer.batch_id, slot_id=slot.id, error="newer failure")
    else:
        service.repo.mark_slot_generation_succeeded(slot.id, newer.batch_id)
    worker._record_generation_failure(batch_id=older.batch_id, slot_id=slot.id, error="older failure")

    failed_slot = service.repo.get_slot(slot.id)
    assert failed_slot["status"] == ("failed" if newer_failed else "reviewing")
    assert failed_slot["last_error"] == ("newer failure" if newer_failed else None)
    assert failed_slot["last_failed_batch_id"] == (newer.batch_id if newer_failed else None)


@pytest.mark.asyncio
async def test_older_batch_success_cannot_clear_newer_failure(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Late Success", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    older = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": older.batch_id, "user_id": 1})
    newer = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker._record_generation_failure(batch_id=newer.batch_id, slot_id=slot.id, error="newer failure")

    await worker._generate_variant(
        pack=service.repo.get_pack(pack.id), slot=service.repo.get_slot(slot.id),
        batch=service.repo.get_batch(older.batch_id), character=None,
        variant_index=0, user_id=1, job=None,
    )

    failed_slot = service.repo.get_slot(slot.id)
    assert failed_slot["status"] == "failed"
    assert failed_slot["last_error"] == "newer failure"
    assert failed_slot["last_failed_batch_id"] == newer.batch_id


def test_stale_success_read_cannot_reopen_failed_batch(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Sibling Race", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
    ))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)
    original_get_batch = service.repo.get_batch
    failure_recorded = False

    def read_before_sibling_failure(batch_id: int) -> dict[str, Any] | None:
        nonlocal failure_recorded
        snapshot = original_get_batch(batch_id)
        if not failure_recorded:
            failure_recorded = True
            worker._record_generation_failure(batch_id=batch_id, slot_id=slot.id, error="sibling failure")
        return snapshot

    monkeypatch.setattr(service.repo, "get_batch", read_before_sibling_failure)
    worker._record_generation_success(batch_id=batch.batch_id)
    if not failure_recorded:
        worker._record_generation_failure(batch_id=batch.batch_id, slot_id=slot.id, error="sibling failure")

    stored = original_get_batch(batch.batch_id)
    assert stored["status"] == "failed"
    assert stored["failed_count"] == 1
    assert stored["completed_count"] == 1


def test_retry_rejects_source_from_another_pack(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    first_pack = service.create_pack(VNAssetPackCreate(
        title="First", primary_character_id=character_id,
    ))
    first_slot = service.create_slot(first_pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary",
    ))
    second_pack = service.create_pack(VNAssetPackCreate(
        title="Second", primary_character_id=character_id,
    ))
    second_slot = service.create_slot(second_pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary",
    ))
    source = service.start_generation(
        first_pack.id, VNAssetGenerationRequest(slot_ids=[first_slot.id]),
    )
    service.repo.update_batch(source.batch_id, {"status": "failed"})

    with pytest.raises(ValueError, match="vn_asset_retry_source_unavailable"):
        service.retry_slot(
            second_pack.id, second_slot.id,
            VNAssetGenerationRequest(source_batch_id=source.batch_id),
        )
    assert len(service.repo.list_batches(second_pack.id)) == 0


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
async def test_accepted_recipe_ignores_later_world_book_edits(
    chacha_db: CharactersRAGDB,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    books = WorldBookService(chacha_db)
    book_id = books.create_world_book("Archive Lore")
    books.add_entry(world_book_id=book_id, keywords=["archive"], content="Original blue doors.")
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(
        title="Lore Pack", primary_character_id=character_id, source_world_book_ids=[book_id],
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=1,
    ))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    books.add_entry(world_book_id=book_id, keywords=["archive"], content="Edited red doors.")
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": batch.batch_id, "user_id": 1})
    await worker.handle_generate_variant({
        "pack_id": pack.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    })

    assert "Original blue doors." in adapter.requests[0].prompt
    assert "Edited red doors." not in adapter.requests[0].prompt


def test_generation_rejects_unreadable_configured_world_book(
    chacha_db: CharactersRAGDB,
    fake_jobs: FakeJobs,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

    books = WorldBookService(chacha_db)
    book_id = books.create_world_book("Unreadable")
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(
        title="Unreadable Lore", primary_character_id=character_id,
        source_world_book_ids=[book_id],
    ))
    service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))

    def fail_entries(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError("world book disk unavailable")

    monkeypatch.setattr(WorldBookService, "get_entries", fail_entries)
    with pytest.raises(ValueError, match="vn_asset_world_book_unavailable"):
        service.start_generation(pack.id)
    assert service.repo.list_batches(pack.id) == []


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


@pytest.mark.asyncio
async def test_generation_api_keeps_event_loop_responsive_during_recipe_capture(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import service as service_module

    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Async Capture", primary_character_id=character_id))
    service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    entered = threading.Event()
    release = threading.Event()
    original = service_module.build_authored_recipe

    def slow_recipe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        entered.set()
        release.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "build_authored_recipe", slow_recipe)
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs
    url = f"/api/v1/vn/vn-assets/packs/{pack.id}/generate"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        request_task = asyncio.create_task(client.post(url, json={"idempotency_key": "capture-offload"}))
        try:
            await asyncio.to_thread(entered.wait, 2)
            assert entered.is_set()
            assert loop.time() - started_at < 1.0
        finally:
            release.set()
        response = await request_task

    assert response.status_code == 202


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
    source = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    service.repo.update_batch(source.batch_id, {"status": "failed"})
    service.repo.update_slot(slot.id, {
        "status": "failed", "last_error": "provider failed", "last_failed_batch_id": source.batch_id,
    })
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
    payload = {"idempotency_key": "retry-slot-1", "variant_count": 1, "source_batch_id": source.batch_id}
    first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry", json=payload)
    replay = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry", json=payload)
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry",
        json={"idempotency_key": "retry-slot-1", "variant_count": 2, "source_batch_id": source.batch_id},
    )

    assert first.status_code == 202
    assert first.json()["source_batch_id"] == source.batch_id
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_retry_slot_api_reports_legacy_recipe_recovery(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Legacy", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    legacy = service.repo.create_batch(
        pack_id=pack.id, requested_by_user_id=1, status="failed",
        options={"slot_ids": [slot.id]},
    )
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs

    response = TestClient(app).post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry",
        json={"idempotency_key": "legacy-retry", "source_batch_id": legacy["id"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "vn_asset_recipe_unavailable"
    assert "Start generation" in response.json()["detail"]["message"]


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
