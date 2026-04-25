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
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenResult
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.concurrency import BackendGenerationLease
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
    app.include_router(vn_assets_router, prefix="/api/v1")

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
    generate_response = client.post(f"/api/v1/vn-assets/packs/{pack.id}/generate", json={})

    assert generate_response.status_code == 202
    assert generate_response.json()["status"] == "queued"
    assert len(fake_jobs.created) == 1

    status_response = client.get(f"/api/v1/vn-assets/packs/{pack.id}/generation")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["batch_id"] == generate_response.json()["batch_id"]
    assert status_payload["planned_count"] == planned_count
    assert status_payload["enqueued_count"] == 0
