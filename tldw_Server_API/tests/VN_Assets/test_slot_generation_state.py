"""Shared-slot generation transitions against the real SQLite VN repository."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetGenerationRequest, VNAssetReviewRequest
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.VN_Assets.concurrency import BackendGenerationLease
from tldw_Server_API.app.core.VN_Assets.jobs import create_generate_variant_job, vn_asset_generation_jobs_queue
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    BlockingFirstImageAdapter,
    EmptyGeneratedFiles,
    FakeGenerationGate,
    FakeImageAdapter,
    FakeImageRegistry,
    FakeJobs,
    RecordingVNSaver,
)
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    chacha_db as chacha_db,
)
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    character_id as character_id,
)
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    pack_with_slots as pack_with_slots,
)
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    service as service,
)

pytestmark = pytest.mark.integration


class LeaseAwareFakeJobs(FakeJobs):
    """Own-scope Jobs double reusing the real read-only lease-health facade."""

    def __init__(self) -> None:
        super().__init__()
        self._clock = JobManager.Clock()

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        """Mirror the stable UUID present on actual Jobs delivery snapshots."""
        job = super().create_job(**kwargs)
        job.setdefault("uuid", f"slot-state-job-{job['id']}")
        return job

    def has_live_processing_lease(self, job_id: int, *, owner_user_id: str) -> bool:
        """Use the actual facade against this test double's rows and Jobs clock."""
        return JobManager.has_live_processing_lease(self, job_id, owner_user_id=owner_user_id)


@pytest.fixture
def fake_jobs() -> LeaseAwareFakeJobs:
    """Extend only this file's existing doubles with the production health API."""
    return LeaseAwareFakeJobs()


def _legacy_batch(service: VNAssetPackService, pack: SimpleNamespace) -> int:
    """Create the recipe-free V0 submission shape for the first slot."""
    return int(
        service.repo.create_batch(
            pack_id=pack.id,
            requested_by_user_id=1,
            status="queued",
            total_slots=1,
            total_variants=1,
            options={"slot_ids": [pack.slots[0].id], "variant_count": 1},
        )["id"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("jobs_delivery", [False, True], ids=["inline", "real-jobs"])
async def test_v1_cancel_preserves_blocked_legacy_generation(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    tmp_path: Path,
    approved: bool,
    jobs_delivery: bool,
) -> None:
    """V1 cancellation cannot hide an actual recipe-free adapter invocation."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    if approved:
        service.repo.create_item(pack_id=pack.id, slot_id=slot_id, review_status="approved")
    v1 = _batch(service, pack)
    legacy = _legacy_batch(service, pack)
    jobs = JobManager(db_path=tmp_path / "legacy-jobs.db") if jobs_delivery else fake_jobs
    job = None
    cancellation_service = service
    if jobs_delivery:
        create_generate_variant_job(
            jobs,
            pack_id=pack.id,
            slot_id=slot_id,
            batch_id=legacy,
            variant_index=0,
            user_id=1,
        )
        job = jobs.acquire_next_job(
            domain="vn_assets",
            queue=vn_asset_generation_jobs_queue(),
            worker_id="legacy",
            lease_seconds=120,
        )
        assert job is not None
        cancellation_service = VNAssetPackService(service.repo.db, owner_user_id=1, jobs_manager=jobs)
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    pending = asyncio.create_task(
        worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": slot_id,
                "batch_id": legacy,
                "variant_index": 0,
                "user_id": 1,
            },
            job=job,
        )
    )
    try:
        assert await asyncio.to_thread(adapter.started.wait, 5)
        assert service.repo.list_batch_recipes(legacy) == []
        assert service.repo.get_slot(slot_id)["status"] == "generating"
        cancellation_service.repo.cancel_batch(v1)
        status = service.repo.get_slot(slot_id)["status"]
        readiness = service.get_readiness(pack.id).status
    finally:
        adapter.release.set()
        await pending

    assert (status, readiness) == ("generating", "generating")
    assert service.repo.get_slot(slot_id)["status"] == "reviewing"
    assert service.repo.get_batch(legacy)["completed_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_failure", [False, True])
async def test_legacy_terminal_transition_preserves_blocked_v1_generation(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    legacy_failure: bool,
) -> None:
    """Legacy completion/failure cannot demote another live V1 adapter call."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    v1 = _batch(service, pack)
    legacy = _legacy_batch(service, pack)
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    pending = asyncio.create_task(
        worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": slot_id,
                "batch_id": v1,
                "variant_index": 0,
                "user_id": 1,
            }
        )
    )
    try:
        assert await asyncio.to_thread(adapter.started.wait, 5)
        if legacy_failure:

            class FailingAdapter(FakeImageAdapter):
                """Fail only the legacy model call, not metadata persistence."""

                def generate(self, request: Any) -> Any:
                    raise RuntimeError("legacy adapter failed")

            worker.image_registry = FakeImageRegistry(FailingAdapter())
        payload = {
            "pack_id": pack.id,
            "slot_id": slot_id,
            "batch_id": legacy,
            "variant_index": 0,
            "user_id": 1,
        }
        if legacy_failure:
            with pytest.raises(RuntimeError, match="legacy adapter failed"):
                await worker.handle_generate_variant(payload)
        else:
            await worker.handle_generate_variant(payload)
        status = service.repo.get_slot(slot_id)["status"]
    finally:
        adapter.release.set()
        await pending

    assert status == "generating"
    assert service.repo.get_slot(slot_id)["status"] == "reviewing"
    stored = service.repo.get_batch(legacy)
    assert (stored["completed_count"], stored["failed_count"], stored["status"]) == (
        (0, 1, "failed") if legacy_failure else (1, 0, "completed")
    )


def _batch(service: VNAssetPackService, pack: SimpleNamespace, variants: int = 1) -> int:
    """Start a real V1 batch for the first fixture slot."""
    return service.start_generation(
        pack.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[pack.slots[0].id], variant_count=variants),
    ).batch_id


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("processing", "generating"),
        ("queued", "queued"),
        ("failed", "cancelled"),
        ("cancelled", "cancelled"),
        ("expired", "cancelled"),
        ("no-lease", "cancelled"),
        ("owner", "cancelled"),
        ("domain", "cancelled"),
        ("type", "cancelled"),
        ("queue", "cancelled"),
        ("payload", "cancelled"),
        ("key", "cancelled"),
        ("bool-variant", "cancelled"),
        ("cancel-requested", "cancelled"),
        ("bad-expiry", "cancelled"),
        ("vn-cancelled", "cancelled"),
        ("vn-failed-active", "generating"),
        ("vn-failed-queued", "cancelled"),
    ],
)
def test_legacy_jobs_display_requires_canonical_live_child(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    mutation: str,
    expected: str,
) -> None:
    """Only exact live legacy children affect cancellation's shared-slot display."""
    pack = pack_with_slots
    v1 = _batch(service, pack)
    legacy = _legacy_batch(service, pack)
    child = create_generate_variant_job(
        fake_jobs,
        pack_id=pack.id,
        slot_id=pack.slots[0].id,
        batch_id=legacy,
        variant_index=0,
        user_id=1,
    )
    child.update(status="processing", worker_id="legacy", lease_id="live", leased_until="2099-01-01 00:00:00")
    if mutation in {"queued", "failed", "cancelled"}:
        child["status"] = mutation
    elif mutation == "expired":
        child["leased_until"] = "2000-01-01 00:00:00"
    elif mutation == "no-lease":
        child["lease_id"] = None
    elif mutation == "cancel-requested":
        child["cancel_requested_at"] = "2026-09-25 00:00:00"
    elif mutation == "bad-expiry":
        child["leased_until"] = "invalid"
    elif mutation.startswith("vn-"):
        service.repo.update_batch(legacy, {"status": "cancelled" if mutation == "vn-cancelled" else "failed"})
        if mutation == "vn-failed-queued":
            child["status"] = "queued"
    elif mutation in {"owner", "domain", "type", "queue", "key"}:
        field = {"owner": "owner_user_id", "type": "job_type", "key": "idempotency_key"}.get(mutation, mutation)
        child[field] = "wrong"
    elif mutation == "payload":
        child["payload"] = {**child["payload"], "user_id": 2}
    elif mutation == "bool-variant":
        child["payload"] = {**child["payload"], "variant_index": False}

    service.repo.cancel_batch(v1)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == expected


def test_legacy_jobs_lookup_reads_beyond_first_page(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
) -> None:
    """A valid older child after 100 different-slot children still queues its slot."""
    pack = pack_with_slots
    legacy = _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    jobs = JobManager(db_path=tmp_path / "paged-legacy-jobs.db")
    for index in range(102):
        create_generate_variant_job(
            jobs,
            pack_id=pack.id,
            batch_id=legacy,
            user_id=1,
            slot_id=pack.slots[0 if index == 0 else 1].id,
            variant_index=index,
        )
    observing_service = VNAssetPackService(service.repo.db, owner_user_id=1, jobs_manager=jobs)

    observing_service.repo.cancel_batch(v1)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "queued"


@pytest.mark.parametrize(
    ("year", "expire", "expected"),
    [(2000, False, "generating"), (2100, True, "cancelled")],
)
def test_legacy_display_uses_authoritative_jobs_clock(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    year: int,
    expire: bool,
    expected: str,
) -> None:
    """Lease display follows real Jobs time, whether ahead of or behind wall time."""

    class FixedClock(JobManager.Clock):
        """Inject a mutable Jobs clock without patching the reader's wall clock."""

        def __init__(self) -> None:
            self.now = datetime(year, 1, 1, tzinfo=timezone.utc)

        def now_utc(self) -> datetime:
            return self.now

    pack = pack_with_slots
    legacy = _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    clock = FixedClock()
    jobs = JobManager(db_path=tmp_path / "legacy-clock.db", clock=clock)
    create_generate_variant_job(
        jobs,
        pack_id=pack.id,
        slot_id=pack.slots[0].id,
        batch_id=legacy,
        variant_index=0,
        user_id=1,
    )
    child = jobs.acquire_next_job(
        domain="vn_assets",
        queue=vn_asset_generation_jobs_queue(),
        worker_id="legacy",
        lease_seconds=120,
    )
    assert child is not None
    if expire:
        clock.now += timedelta(seconds=121)
    observer = VNAssetPackService(service.repo.db, owner_user_id=1, jobs_manager=jobs)

    observer.repo.cancel_batch(v1)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model_failure", [False, True], ids=["success", "original-failure"])
async def test_legacy_display_outage_preserves_worker_sdk_disposition(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_failure: bool,
) -> None:
    """Display outages cannot retry published legacy work or replace model errors."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    legacy = _legacy_batch(service, pack)
    service.repo.update_batch(
        legacy,
        {
            "total_variants": 2,
            "planned_count": 2,
            "options": {"slot_ids": [slot_id], "variant_count": 2},
        },
    )
    service.repo.cancel_batch(_batch(service, pack))
    jobs = JobManager(db_path=tmp_path / "legacy-sdk-outage.db")
    children = [
        create_generate_variant_job(
            jobs,
            pack_id=pack.id,
            slot_id=slot_id,
            batch_id=legacy,
            variant_index=index,
            user_id=1,
        )
        for index in range(2)
    ]
    original_error = VNAssetGenerationError("legacy_model_failure")

    class ModelAdapter(FakeImageAdapter):
        """Keep the real worker model/outcome path while optionally failing it."""

        def generate(self, request: Any) -> Any:
            if model_failure:
                self.requests.append(request)
                raise original_error
            return super().generate(request)

    adapter = ModelAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
    )
    original_list = jobs.list_jobs

    def unavailable_jobs(**kwargs: Any) -> Any:
        """Raise from the actual Jobs-backed reader after the legacy outcome."""
        raise OSError("Jobs display unavailable")

    monkeypatch.setattr(jobs, "list_jobs", unavailable_jobs)
    observed_errors: list[Exception] = []

    async def run_one() -> None:
        """Run one real SDK delivery with its default retry-on-exception policy."""
        sdk = WorkerSDK(
            jobs,
            WorkerConfig(
                domain="vn_assets",
                queue=vn_asset_generation_jobs_queue(),
                worker_id="legacy-sdk",
            ),
        )

        async def handle(acquired: dict[str, Any]) -> dict[str, Any]:
            """Stop after this delivery, allowing SDK to apply its disposition."""
            try:
                return await worker.handle_job_async(acquired)
            except Exception as exc:
                observed_errors.append(exc)
                raise
            finally:
                sdk.stop()

        await asyncio.wait_for(sdk.run(handler=handle, owner_user_id="1"), timeout=5)

    display_records: list[dict[str, Any]] = []
    sink = logger.add(lambda message: display_records.append(message.record), level="WARNING")
    try:
        await run_one()
    finally:
        logger.remove(sink)

    stored = jobs.get_job(int(children[0]["id"]))
    batch = service.repo.get_batch(legacy)
    assert (stored["status"], observed_errors) == (("failed", [original_error]) if model_failure else ("completed", []))
    display_warnings = [
        record for record in display_records if record["message"] == "VN legacy display reconciliation failed"
    ]
    assert len(display_warnings) == 1
    assert display_warnings[0]["extra"]["job_id"] == int(children[0]["id"])
    assert display_warnings[0]["extra"]["error_type"] == "OSError"
    assert len(adapter.requests) == 1
    assert len(saver.calls) == (0 if model_failure else 1)
    assert len(service.repo.list_items(pack.id)) == (0 if model_failure else 1)
    assert (batch["completed_count"], batch["failed_count"], batch["status"]) == (
        (0, 1, "failed") if model_failure else (1, 0, "processing")
    )
    assert jobs.get_job(int(children[1]["id"]))["status"] == "queued"
    if not model_failure:
        monkeypatch.setattr(jobs, "list_jobs", original_list)
        await run_one()
        assert [jobs.get_job(int(child["id"]))["status"] for child in children] == ["completed", "completed"]
        assert sorted(
            json.loads(item["source_context_snapshot_json"])["variant_index"]
            for item in service.repo.list_items(pack.id)
        ) == [0, 1]
        assert (len(adapter.requests), len(saver.calls), service.repo.get_batch(legacy)["completed_count"]) == (2, 2, 2)


@pytest.mark.asyncio
async def test_legacy_display_outage_emits_safe_structured_diagnostic(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The display warning includes identities/frames but no messages, sources or locals."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    legacy = _legacy_batch(service, pack)
    service.repo.cancel_batch(_batch(service, pack))
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )

    def unavailable_jobs(**kwargs: Any) -> Any:
        """Put distinct secrets in the message, chain, source literal and local."""
        secret_local = "DISPLAY_LOCAL_SECRET"
        try:
            raise RuntimeError("DISPLAY_CHAIN_SECRET")
        except RuntimeError as cause:
            raise OSError("DISPLAY_MESSAGE_SECRET", secret_local) from cause

    monkeypatch.setattr(fake_jobs, "list_jobs", unavailable_jobs)
    records: list[dict[str, Any]] = []
    sink = logger.add(lambda message: records.append(message.record), level="TRACE")
    result = None
    try:
        try:
            result = await worker.handle_generate_variant(
                {
                    "pack_id": pack.id,
                    "slot_id": slot_id,
                    "batch_id": legacy,
                    "variant_index": 0,
                    "user_id": 1,
                }
            )
        except OSError:
            pass
    finally:
        logger.remove(sink)

    warnings = [record for record in records if record["message"] == "VN legacy display reconciliation failed"]
    assert len(warnings) == 1
    warning = warnings[0]
    extra = warning["extra"]
    assert {
        key: extra[key] for key in ("operation", "user_id", "pack_id", "batch_id", "slot_id", "job_id", "error_type")
    } == {
        "operation": "finish_legacy_display",
        "user_id": 1,
        "pack_id": pack.id,
        "batch_id": legacy,
        "slot_id": slot_id,
        "job_id": None,
        "error_type": "OSError",
    }
    frames = extra["traceback_frames"]
    assert any(frame["function"] == "unavailable_jobs" for frame in frames)
    assert all(set(frame) == {"file", "function", "line"} and type(frame["line"]) is int for frame in frames)
    assert warning["exception"] is None
    assert not any(
        secret in str(warning)
        for secret in (
            "DISPLAY_LOCAL_SECRET",
            "DISPLAY_CHAIN_SECRET",
            "DISPLAY_MESSAGE_SECRET",
        )
    )
    assert any(record["function"] == "__exit__" and "rolling back" in record["message"] for record in records)
    assert not any(
        secret in str(records)
        for secret in (
            "DISPLAY_LOCAL_SECRET",
            "DISPLAY_CHAIN_SECRET",
            "DISPLAY_MESSAGE_SECRET",
        )
    )
    assert result is not None


def test_lazily_created_jobs_manager_wires_legacy_display(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core callers without injected Jobs still preserve genuine legacy child activity."""
    from tldw_Server_API.app.core.Jobs import manager as jobs_module

    monkeypatch.setattr(jobs_module, "JobManager", LeaseAwareFakeJobs)
    lazy_service = VNAssetPackService(service.repo.db, owner_user_id=1)
    pack = pack_with_slots
    v1 = _batch(lazy_service, pack)
    legacy = _legacy_batch(lazy_service, pack)
    child = create_generate_variant_job(
        lazy_service.jobs_manager,
        pack_id=pack.id,
        slot_id=pack.slots[0].id,
        batch_id=legacy,
        variant_index=0,
        user_id=1,
    )
    child.update(status="processing", worker_id="legacy", lease_id="live", leased_until="2099-01-01 00:00:00")

    lazy_service.repo.cancel_batch(v1)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "generating"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "other_work", "expected"),
    [
        ("failed", "none", "failed"),
        ("failed", "sibling", "generating"),
        ("failed", "replacement", "generating"),
        ("deleted-success", "none", "reviewing"),
    ],
)
async def test_finished_legacy_jobs_handoff_excludes_only_its_own_lease(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    other_work: str,
    expected: str,
) -> None:
    """Finishing delivery is not activity, but another child/current lease still is."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    legacy = _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    service.repo.cancel_batch(v1)
    jobs = JobManager(db_path=tmp_path / "legacy-handoff.db")
    create_generate_variant_job(
        jobs,
        pack_id=pack.id,
        slot_id=slot_id,
        batch_id=legacy,
        variant_index=0,
        user_id=1,
    )
    child = jobs.acquire_next_job(
        domain="vn_assets",
        queue=vn_asset_generation_jobs_queue(),
        worker_id="legacy",
        lease_seconds=120,
    )
    assert child is not None
    if other_work == "sibling":
        service.repo.update_batch(
            legacy,
            {
                "planned_count": 2,
                "total_variants": 2,
                "options": {"slot_ids": [slot_id], "variant_count": 2},
            },
        )
        create_generate_variant_job(
            jobs,
            pack_id=pack.id,
            slot_id=slot_id,
            batch_id=legacy,
            variant_index=1,
            user_id=1,
        )
        assert (
            jobs.acquire_next_job(
                domain="vn_assets",
                queue=vn_asset_generation_jobs_queue(),
                worker_id="sibling",
                lease_seconds=120,
            )
            is not None
        )

    class FailingAdapter(FakeImageAdapter):
        """Optionally replace the authoritative lease before failing the old model call."""

        def generate(self, request: Any) -> Any:
            if other_work == "replacement":
                assert jobs.release_job(child["id"], worker_id=child["worker_id"], lease_id=child["lease_id"])
                replacement = jobs.acquire_next_job(
                    domain="vn_assets",
                    queue=vn_asset_generation_jobs_queue(),
                    worker_id="replacement",
                    lease_seconds=120,
                )
                assert replacement is not None and replacement["lease_id"] != child["lease_id"]
            raise RuntimeError("legacy adapter failed")

    adapter = FakeImageAdapter() if outcome == "deleted-success" else FailingAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    if outcome == "deleted-success":
        original_finish = service.repo.finish_legacy_display

        def delete_before_finish(batch_id: int, slot_id: int, **options: Any) -> None:
            """Remove only the now-published V0 item immediately before final display."""
            for item in service.repo.list_items(pack.id):
                service.repo.delete_item(item["id"])
            original_finish(batch_id, slot_id, **options)

        monkeypatch.setattr(service.repo, "finish_legacy_display", delete_before_finish)
    if outcome == "failed":
        with pytest.raises(RuntimeError, match="legacy adapter failed"):
            await worker.handle_generate_variant(child["payload"], job=dict(child))
    else:
        await worker.handle_generate_variant(child["payload"], job=dict(child))
    # WorkerSDK has not acknowledged this delivery yet: its Jobs lease is live.
    assert jobs.has_live_processing_lease(child["id"], owner_user_id="1")
    status_at_handoff = service.repo.get_slot(slot_id)["status"]
    if other_work == "none":
        if outcome == "failed":
            assert jobs.fail_job(
                child["id"],
                error="legacy adapter failed",
                retryable=False,
                worker_id=child["worker_id"],
                lease_id=child["lease_id"],
            )
        else:
            assert jobs.complete_job(child["id"], worker_id=child["worker_id"], lease_id=child["lease_id"])
        assert service.repo.get_slot(slot_id)["status"] == expected

    assert status_at_handoff == expected
    stored = service.repo.get_batch(legacy)
    assert (stored["completed_count"], stored["failed_count"]) == ((1, 0) if outcome == "deleted-success" else (0, 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("other_work", ["none", "sibling", "replacement"])
async def test_published_legacy_handoff_settles_only_original_delivery(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    other_work: str,
) -> None:
    """Publishing A cannot hide blocked B; exact A stays settled before Jobs ack."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    legacy = _legacy_batch(service, pack)
    service.repo.update_batch(legacy, {
        "planned_count": 2, "total_variants": 2,
        "options": {"slot_ids": [slot_id], "variant_count": 2},
    })
    v1 = _batch(service, pack)
    service.repo.cancel_batch(v1)
    jobs = JobManager(db_path=tmp_path / "published-handoff.db")
    create_generate_variant_job(
        jobs, pack_id=pack.id, slot_id=slot_id, batch_id=legacy, variant_index=0, user_id=1,
    )
    original = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="original", lease_seconds=120,
    )
    assert original is not None
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=RecordingVNSaver(),
    )
    original_pending = asyncio.create_task(worker.handle_generate_variant(original["payload"], job=dict(original)))
    other_adapter = BlockingFirstImageAdapter()
    other_pending = None
    try:
        assert await asyncio.to_thread(adapter.started.wait, 5)
        if other_work != "none":
            if other_work == "replacement":
                assert jobs.release_job(
                    original["id"], worker_id=original["worker_id"], lease_id=original["lease_id"],
                )
            else:
                create_generate_variant_job(
                    jobs, pack_id=pack.id, slot_id=slot_id, batch_id=legacy, variant_index=1, user_id=1,
                )
            other = jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="other", lease_seconds=120,
            )
            assert other is not None
            if other_work == "replacement":
                assert other["id"] == original["id"] and other["lease_id"] != original["lease_id"]
            other_worker = VNAssetGenerationWorker(
                repo=VNAssetPacksRepository(service.repo.db), jobs_manager=jobs,
                image_registry=FakeImageRegistry(other_adapter), backend_gate=FakeGenerationGate(),
                save_vn_asset_image=RecordingVNSaver(),
            )
            other_pending = asyncio.create_task(other_worker.handle_generate_variant(other["payload"], job=dict(other)))
            assert await asyncio.to_thread(other_adapter.started.wait, 5)
        adapter.release.set()
        result = await original_pending
        expected = "reviewing" if other_work == "none" else "generating"
        handoff_status = service.repo.get_slot(slot_id)["status"]
        # No WorkerSDK acknowledgement: the current canonical delivery is still live.
        current = jobs.get_job(original["id"], owner_user_id="1")
        assert current["status"] == "processing"
        assert jobs.has_live_processing_lease(current["id"], owner_user_id="1")
        assert service.repo.get_batch(legacy)["completed_count"] == 1
        assert len(service.repo.list_items(pack.id)) == 1
        cancellation_service = VNAssetPackService(service.repo.db, owner_user_id=1, jobs_manager=jobs)
        assert cancellation_service.repo is not service.repo
        cancellation_service.repo.cancel_batch(v1)
        cancellation_status = cancellation_service.repo.get_slot(slot_id)["status"]
        assert (handoff_status, cancellation_status) == (expected, expected)
        item = service.repo.get_item(result["item_id"])
        context = json.loads(item["source_context_snapshot_json"])
        fingerprint = context.get("legacy_delivery_fingerprint")
        assert isinstance(fingerprint, str) and len(fingerprint) == 64
        assert all(char in "0123456789abcdef" for char in fingerprint)
        assert original["lease_id"] not in json.dumps(item)
        assert current["lease_id"] not in json.dumps(item)
        assert "legacy_delivery_fingerprint" not in json.dumps(adapter.requests[0].__dict__)
    finally:
        adapter.release.set()
        other_adapter.release.set()
        await original_pending
        if other_pending is not None:
            await other_pending


@pytest.mark.parametrize("provenance", [None, "", "0" * 64, 7])
@pytest.mark.parametrize("status", ["processing", "queued"])
def test_unknown_published_legacy_provenance_preserves_jobs_activity(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    provenance: Any,
    status: str,
) -> None:
    """Unknown/mismatched historical provenance cannot settle valid live Jobs work."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    legacy = _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    jobs = JobManager(db_path=tmp_path / "historical-provenance.db")
    create_generate_variant_job(
        jobs, pack_id=pack.id, slot_id=slot_id, batch_id=legacy, variant_index=0, user_id=1,
    )
    if status == "processing":
        assert jobs.acquire_next_job(
            domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="live", lease_seconds=120,
        ) is not None
    context = {"batch_id": legacy, "variant_index": 0}
    if provenance is not None:
        context["legacy_delivery_fingerprint"] = provenance
    service.repo.create_item(
        pack_id=pack.id, slot_id=slot_id, generated_file_id=123, source_context_snapshot=context,
    )
    cancellation_service = VNAssetPackService(service.repo.db, owner_user_id=1, jobs_manager=jobs)
    cancellation_service.repo.cancel_batch(v1)
    assert service.repo.get_slot(slot_id)["status"] == ("generating" if status == "processing" else "queued")


def test_legacy_jobs_read_failure_rolls_back_write_admitted_cancellation(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infrastructure failure cannot commit terminal outcomes without shared state."""
    pack = pack_with_slots
    _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    service.repo.update_slot(pack.slots[0].id, {"status": "generating", "last_error": "keep"})

    def unavailable(**filters: Any) -> list[dict[str, Any]]:
        """Observe protected cancellation before simulating a Jobs read outage."""
        assert service.repo.db.get_connection().in_transaction
        assert service.repo.get_batch(v1)["status"] == "cancelled"
        raise OSError("Jobs unavailable")

    monkeypatch.setattr(fake_jobs, "list_jobs", unavailable)
    with pytest.raises(OSError, match="Jobs unavailable"):
        service.repo.cancel_batch(v1)

    assert service.repo.get_batch(v1)["status"] == "queued"
    assert service.repo.get_batch(v1)["cancelled_count"] == 0
    assert service.repo.get_variant_outcome(v1, pack.slots[0].id, 0)["outcome_status"] == "planned"
    assert service.repo.get_slot(pack.slots[0].id)["last_error"] == "keep"
    assert service.repo.get_slot(pack.slots[0].id)["status"] == "generating"


def test_legacy_jobs_lookup_rechecks_current_cancellation(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listed processing child loses display authority when its current row cancels."""
    pack = pack_with_slots
    legacy = _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    child = create_generate_variant_job(
        fake_jobs,
        pack_id=pack.id,
        slot_id=pack.slots[0].id,
        batch_id=legacy,
        variant_index=0,
        user_id=1,
    )
    child.update(status="processing", worker_id="legacy", lease_id="live", leased_until="2099-01-01 00:00:00")
    original_read = fake_jobs.get_job

    def cancelled(identity: int, **filters: Any) -> dict[str, Any] | None:
        """Change only the authoritative row observed after listing candidates."""
        row = original_read(identity, **filters)
        return {**row, "status": "cancelled"} if row is not None else None

    monkeypatch.setattr(fake_jobs, "get_job", cancelled)
    service.repo.cancel_batch(v1)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_completed_legacy_slot_not_active_while_another_slot_runs(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
) -> None:
    """A stored V0 variant does not stick generating while its Job completes."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    legacy = service.repo.create_batch(
        pack_id=pack.id,
        requested_by_user_id=1,
        status="queued",
        total_variants=2,
        options={"slot_ids": [slot_id, pack.slots[1].id], "variant_count": 1},
    )["id"]
    v1 = _batch(service, pack)
    children = [
        create_generate_variant_job(
            fake_jobs,
            pack_id=pack.id,
            batch_id=legacy,
            user_id=1,
            slot_id=slot.id,
            variant_index=0,
        )
        for slot in pack.slots[:2]
    ]
    for child in children:
        child.update(status="processing", worker_id="legacy", lease_id="live", leased_until="2099-01-01 00:00:00")
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    await worker.handle_generate_variant(children[0]["payload"], job=dict(children[0]))

    service.repo.cancel_batch(v1)

    assert service.repo.get_slot(slot_id)["status"] == "reviewing"
    assert service.repo.get_batch(legacy)["status"] == "processing"
    assert service.repo.get_batch(legacy)["completed_count"] == 1


@pytest.mark.asyncio
async def test_failed_legacy_batch_preserves_already_executing_sibling(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
) -> None:
    """V0 fail-fast batch status cannot erase another child already in its adapter."""
    pack = pack_with_slots
    legacy = service.repo.create_batch(
        pack_id=pack.id,
        requested_by_user_id=1,
        status="queued",
        total_variants=2,
        options={"slot_ids": [pack.slots[0].id], "variant_count": 2},
    )["id"]
    v1 = _batch(service, pack)
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "slot_id": pack.slots[0].id, "batch_id": legacy, "user_id": 1}
    pending = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 0}))
    try:
        assert await asyncio.to_thread(adapter.started.wait, 5)

        class FailingAdapter(FakeImageAdapter):
            """Fail the second V0 child while leaving the first model call blocked."""

            def generate(self, request: Any) -> Any:
                raise RuntimeError("sibling failed")

        worker.image_registry = FakeImageRegistry(FailingAdapter())
        with pytest.raises(RuntimeError, match="sibling failed"):
            await worker.handle_generate_variant({**payload, "variant_index": 1})
        assert service.repo.get_batch(legacy)["status"] == "failed"
        service.repo.cancel_batch(v1)
        status = service.repo.get_slot(pack.slots[0].id)["status"]
    finally:
        adapter.release.set()
        await pending

    assert status == "generating"
    stored = service.repo.get_batch(legacy)
    assert (stored["status"], stored["completed_count"], stored["failed_count"]) == ("failed", 1, 1)
    assert service.repo.get_slot(pack.slots[0].id)["status"] == "reviewing"


@pytest.mark.asyncio
@pytest.mark.parametrize("display_outage", [False, True])
async def test_inline_legacy_task_cancellation_clears_local_display(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
    display_outage: bool,
) -> None:
    """Cancelling the coroutine clears display even though its model thread may finish."""
    pack = pack_with_slots
    legacy = _legacy_batch(service, pack)
    v1 = _batch(service, pack)
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    pending = asyncio.create_task(
        worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": pack.slots[0].id,
                "batch_id": legacy,
                "variant_index": 0,
                "user_id": 1,
            }
        )
    )
    try:
        assert await asyncio.to_thread(adapter.started.wait, 5)
        original_list = fake_jobs.list_jobs

        def unavailable_jobs(**kwargs: Any) -> Any:
            """Fail only the post-cancellation display read."""
            raise OSError("Jobs display unavailable")

        if display_outage:
            monkeypatch.setattr(fake_jobs, "list_jobs", unavailable_jobs)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        monkeypatch.setattr(fake_jobs, "list_jobs", original_list)
        service.repo.cancel_batch(v1)
        assert service.repo.get_slot(pack.slots[0].id)["status"] == "cancelled"
        assert service.repo.get_batch(legacy)["completed_count"] == 0
    finally:
        adapter.release.set()


@pytest.mark.asyncio
async def test_blocked_legacy_other_slot_does_not_manufacture_activity(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
) -> None:
    """An executing legacy child contributes only to its exact target slot."""
    pack = pack_with_slots
    v1 = _batch(service, pack)
    other_slot = pack.slots[1].id
    legacy = service.repo.create_batch(
        pack_id=pack.id,
        requested_by_user_id=1,
        status="queued",
        total_variants=1,
        options={"slot_ids": [other_slot], "variant_count": 1},
    )["id"]
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    pending = asyncio.create_task(
        worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": other_slot,
                "batch_id": legacy,
                "variant_index": 0,
                "user_id": 1,
            }
        )
    )
    try:
        assert await asyncio.to_thread(adapter.started.wait, 5)
        service.repo.cancel_batch(v1)
        assert service.repo.get_slot(pack.slots[0].id)["status"] == "cancelled"
        assert service.repo.get_slot(other_slot)["status"] == "generating"
    finally:
        adapter.release.set()
        await pending


def _claim(repo: VNAssetPacksRepository, batch: int, pack: SimpleNamespace, index: int = 0) -> dict[str, Any]:
    """Reserve a fenced inline item with storage for transition-only tests."""
    return repo.claim_variant(
        batch_id=batch,
        slot_id=pack.slots[0].id,
        variant_index=index,
        lease_id="inline",
        attempt_token=f"{batch}-{index}",
        item_fields={"pack_id": pack.id, "generated_file_id": batch * 10 + index},
    )


def _finish(repo: VNAssetPacksRepository, batch: int, pack: SimpleNamespace, item: dict[str, Any], kind: str) -> None:
    """Apply a terminal transition through the repository's fenced APIs."""
    identity = {"batch_id": batch, "slot_id": pack.slots[0].id, "variant_index": int(item["variant_index"])}
    token = f"{batch}-{item['variant_index']}"
    if kind == "completed":
        repo.complete_variant(**identity, item_id=item["id"], attempt_token=token)
    elif kind == "failed":
        repo.fail_variant(**identity, error="provider failed", attempt_token=token)
    else:
        repo.cancel_batch(batch)


@pytest.mark.parametrize("fallback", ["failed", "reviewing", None])
@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ("approved", "approved"), ("draft", "reviewing"),
        ("skipped", "skipped"), ("failed", "failed"),
        ("cancelled", "cancelled"), ("empty", "planned"),
        ("active", "generating"), ("queued", "queued"),
    ],
)
def test_legacy_fallback_preserves_derived_slot_precedence(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    existing: str, expected: str, fallback: str | None,
) -> None:
    """Fallback fills empty terminal display, never review/skip/failure/live work."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    batch = _batch(service, pack)
    if existing in {"approved", "draft", "skipped"}:
        item = _claim(service.repo, batch, pack)
        _finish(service.repo, batch, pack, item, "completed")
        service.repo.update_item_review(item["id"], review_status="approved" if existing == "approved" else "draft")
        if existing == "skipped":
            service.repo.update_slot(slot_id, {"status": "skipped"})
    elif existing == "failed":
        item = _claim(service.repo, batch, pack)
        _finish(service.repo, batch, pack, item, "failed")
        service.repo.cancel_batch(_batch(service, pack))
    elif existing == "cancelled":
        service.repo.cancel_batch(batch)
    elif existing == "empty":
        # Historical completion without publication cannot manufacture readiness.
        service.repo.update_batch(batch, {"status": "completed"})
    elif existing == "active":
        _claim(service.repo, batch, pack)
    before = service.repo.get_batch(batch)
    items = service.repo.list_items(pack.id)

    service.repo.finish_legacy_display(_legacy_batch(service, pack), slot_id, inline=False, fallback_status=fallback)

    if existing in {"empty", "cancelled"} and fallback is not None:
        expected = fallback
    assert service.repo.get_slot(slot_id)["status"] == expected
    assert service.repo.get_batch(batch) == before
    assert service.repo.list_items(pack.id) == items


@pytest.mark.asyncio
@pytest.mark.parametrize("review_status", ["approved", "draft"])
@pytest.mark.parametrize("jobs_delivery", [False, True], ids=["inline", "real-jobs"])
async def test_failed_legacy_worker_preserves_completed_v1_review_and_readiness(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs, tmp_path: Path, review_status: str, jobs_delivery: bool,
) -> None:
    """A genuine V0 model failure cannot demote the same slot's published V1 item."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    for slot in pack.slots[1:]:
        service.repo.update_slot(slot.id, {"required_for_runtime": False})
    batch = _batch(service, pack)
    adapter = FakeImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=saver,
    )
    result = await worker.handle_generate_variant(
        {"pack_id": pack.id, "slot_id": slot_id, "batch_id": batch, "variant_index": 0, "user_id": 1},
    )
    service.review_item_for_pack(
        pack.id, result["item_id"], VNAssetReviewRequest(review_status=review_status, preferred=review_status == "approved"),
    )
    v1_before = service.repo.get_batch(batch)
    outcome_before = service.repo.get_variant_outcome(batch, slot_id, 0)
    item_before = service.repo.get_item(result["item_id"])
    readiness_before = service.get_readiness(pack.id)
    assert readiness_before.ready is (review_status == "approved")
    legacy = _legacy_batch(service, pack)
    jobs = JobManager(db_path=tmp_path / "legacy-review.db") if jobs_delivery else fake_jobs
    job = None
    if jobs_delivery:
        create_generate_variant_job(jobs, pack_id=pack.id, slot_id=slot_id, batch_id=legacy, variant_index=0, user_id=1)
        job = jobs.acquire_next_job(
            domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="legacy-review", lease_seconds=120,
        )
        assert job is not None
    original_error = VNAssetGenerationError("legacy_model_failure")

    class FailingAdapter(FakeImageAdapter):
        """Fail the actual legacy model call, recording its unchanged invocation."""

        def generate(self, request: Any) -> Any:
            self.requests.append(request)
            raise original_error

    failing = FailingAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs, image_registry=FakeImageRegistry(failing),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=saver,
    )
    with pytest.raises(VNAssetGenerationError) as raised:
        await worker.handle_generate_variant(
            {"pack_id": pack.id, "slot_id": slot_id, "batch_id": legacy, "variant_index": 0, "user_id": 1}, job=job,
        )

    assert raised.value is original_error
    assert service.repo.get_slot(slot_id)["status"] == ("approved" if review_status == "approved" else "reviewing")
    assert service.get_readiness(pack.id) == readiness_before
    assert service.repo.get_item(result["item_id"]) == item_before
    assert service.repo.get_variant_outcome(batch, slot_id, 0) == outcome_before
    assert service.repo.get_batch(batch) == v1_before
    legacy_row = service.repo.get_batch(legacy)
    assert (legacy_row["status"], legacy_row["completed_count"], legacy_row["failed_count"]) == ("failed", 0, 1)
    assert service.repo.list_batch_recipes(legacy) == []
    assert (len(adapter.requests), len(failing.requests), len(saver.calls)) == (1, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("history", ["empty-completed", "cancelled", "failed"])
async def test_legacy_worker_failure_without_publication_remains_failed(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace, fake_jobs: FakeJobs, history: str,
) -> None:
    """Actual empty legacy failure outranks cancellation without creating a V1 outcome."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    batch = _batch(service, pack)
    if history == "empty-completed":
        service.repo.update_batch(batch, {"status": "completed"})
    elif history == "cancelled":
        service.repo.cancel_batch(batch)
    else:
        service.repo.fail_variant(batch_id=batch, slot_id=slot_id, variant_index=0, error="V1 failed")
    before = service.repo.get_variant_outcome(batch, slot_id, 0)
    legacy = _legacy_batch(service, pack)

    class FailingAdapter(FakeImageAdapter):
        """Make one actual legacy model attempt fail before publication."""

        def generate(self, request: Any) -> Any:
            self.requests.append(request)
            raise VNAssetGenerationError("legacy_model_failure")

    adapter = FailingAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=saver,
    )
    with pytest.raises(VNAssetGenerationError, match="legacy_model_failure"):
        await worker.handle_generate_variant(
            {"pack_id": pack.id, "slot_id": slot_id, "batch_id": legacy, "variant_index": 0, "user_id": 1},
        )
    assert service.repo.get_slot(slot_id)["status"] == "failed"
    assert service.repo.get_variant_outcome(batch, slot_id, 0) == before
    assert service.repo.list_items(pack.id) == []
    assert service.repo.get_batch(legacy)["failed_count"] == 1
    assert (len(adapter.requests), len(saver.calls)) == (1, 0)


@pytest.mark.parametrize(
    ("other_batch", "kind"),
    [(False, "completed"), (False, "failed"), (True, "completed"), (True, "failed"), (True, "cancelled")],
)
def test_terminal_transition_preserves_other_claimed_work(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    other_batch: bool,
    kind: str,
) -> None:
    """Ending one delivery must not hide another active delivery's slot."""
    pack = pack_with_slots
    first = _batch(service, pack, variants=1 if other_batch else 2)
    second = _batch(service, pack) if other_batch else first
    item = _claim(service.repo, first, pack)
    _claim(service.repo, second, pack, index=0 if other_batch else 1)
    service.repo.update_slot(pack.slots[0].id, {"status": "generating"})

    _finish(service.repo, first, pack, item, kind)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "generating"
    assert service.get_readiness(pack.id).status == "generating"


@pytest.mark.parametrize("kind", ["completed", "failed", "cancelled"])
def test_terminal_transition_preserves_other_queued_batch(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    kind: str,
) -> None:
    """Unclaimed planned work remains queued rather than review-ready."""
    pack = pack_with_slots
    first = _batch(service, pack)
    _batch(service, pack)
    item = _claim(service.repo, first, pack)
    service.repo.update_slot(pack.slots[0].id, {"status": "generating"})

    _finish(service.repo, first, pack, item, kind)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "queued"


@pytest.mark.parametrize(
    ("reviews", "kind", "expected"),
    [
        ([], "cancelled", "cancelled"),
        ([], "failed", "failed"),
        (["approved"], "cancelled", "approved"),
        (["approved"], "failed", "approved"),
        (["draft"], "cancelled", "reviewing"),
        (["draft"], "failed", "reviewing"),
        (["approved", "draft"], "cancelled", "reviewing"),
        (["rejected"], "cancelled", "reviewing"),
    ],
)
def test_terminal_state_uses_published_review_precedence(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    reviews: list[str],
    kind: str,
    expected: str,
) -> None:
    """Terminal hidden reservations cannot override published review decisions."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    published = [service.repo.create_item(pack_id=pack.id, slot_id=slot_id, review_status=review) for review in reviews]
    batch = _batch(service, pack)
    item = _claim(service.repo, batch, pack)
    service.repo.update_slot(slot_id, {"status": "generating"})

    _finish(service.repo, batch, pack, item, kind)

    assert service.repo.get_slot(slot_id)["status"] == expected
    assert [row["review_status"] for row in service.repo.list_items(pack.id)] == reviews
    assert [service.repo.get_item(row["id"])["review_status"] for row in published] == reviews
    assert service.repo.item_is_unpublished(item["id"])
    with pytest.raises(ValueError, match="item_not_found"):
        service.review_item(item["id"], VNAssetReviewRequest(review_status="approved"))


@pytest.mark.asyncio
async def test_cancel_during_adapter_reconciles_slot_without_publishing(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
) -> None:
    """Cancellation while the adapter runs releases readiness and keeps bytes hidden."""
    pack = pack_with_slots
    batch = _batch(service, pack)
    adapter = BlockingFirstImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver,
    )
    pending = asyncio.create_task(
        worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": pack.slots[0].id,
                "variant_index": 0,
                "batch_id": batch,
                "user_id": 1,
            }
        )
    )
    assert await asyncio.to_thread(adapter.started.wait, 5)
    try:
        service.repo.cancel_batch(batch)
        status_at_cancel = service.repo.get_slot(pack.slots[0].id)["status"]
    finally:
        adapter.release.set()
        with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
            await pending

    assert status_at_cancel == "cancelled"
    assert service.repo.list_items(pack.id) == []
    assert service.get_readiness(pack.id).status != "generating"
    assert service.repo.get_batch(batch)["cancelled_count"] == 1
    assert saver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["lease", "claim", "cancel"])
async def test_obsolete_worker_cannot_write_generating_or_call_adapter(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    loss: str,
) -> None:
    """Lose authority after claim but before generating admission."""
    pack = pack_with_slots
    slot_id = pack.slots[0].id
    batch = _batch(service, pack)
    approved = service.repo.create_item(pack_id=pack.id, slot_id=slot_id, review_status="approved")
    service.repo.update_slot(slot_id, {"status": "approved"})
    job = fake_jobs.create_job(job_type="vn_asset_generate_variant", owner_user_id="1")
    job.update(status="processing", lease_id="lease-1", leased_until="2099-01-01 00:00:00")
    adapter = FakeImageAdapter()

    class RevokingRegistry(FakeImageRegistry):
        """Inject a competing transition before the worker's visible admission."""

        def resolve_backend(self, requested: str | None) -> str | None:
            if loss == "lease":
                job["lease_id"] = "replacement-lease"
            elif loss == "cancel":
                service.repo.cancel_batch(batch)
            else:
                outcome = service.repo.get_variant_outcome(batch, slot_id, 0)
                service.repo.release_variant_claim(
                    batch_id=batch,
                    slot_id=slot_id,
                    variant_index=0,
                    attempt_token=outcome["claim_token"],
                )
                service.repo.claim_variant(
                    batch_id=batch,
                    slot_id=slot_id,
                    variant_index=0,
                    lease_id="inline",
                    attempt_token="replacement",
                    item_fields={"pack_id": pack.id},
                )
            return super().resolve_backend(requested)

    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=RevokingRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    with pytest.raises(VNAssetGenerationError):
        await worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": slot_id,
                "variant_index": 0,
                "batch_id": batch,
                "user_id": 1,
            },
            job=dict(job) if loss == "lease" else None,
        )

    assert service.repo.get_slot(slot_id)["status"] == ("queued" if loss == "claim" else "approved")
    assert adapter.requests == []
    assert service.repo.get_item(approved["id"])["review_status"] == "approved"


@pytest.mark.asyncio
async def test_parallel_variants_remain_generating_until_both_finish(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
) -> None:
    """A fast sibling cannot make a blocked adapter look finished."""
    pack = pack_with_slots
    batch = _batch(service, pack, variants=2)
    adapter = BlockingFirstImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
        generated_files_repo=EmptyGeneratedFiles(),
    )
    payload = {"pack_id": pack.id, "slot_id": pack.slots[0].id, "batch_id": batch, "user_id": 1}
    first = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 0}))
    assert await asyncio.to_thread(adapter.started.wait, 5)
    try:
        await worker.handle_generate_variant({**payload, "variant_index": 1})
        status_while_blocked = service.repo.get_slot(pack.slots[0].id)["status"]
        readiness_while_blocked = service.get_readiness(pack.id).status
    finally:
        adapter.release.set()
        await first

    assert status_while_blocked == "generating"
    assert readiness_while_blocked == "generating"
    assert service.repo.get_slot(pack.slots[0].id)["status"] == "reviewing"
    assert service.repo.get_batch(batch)["completed_count"] == 2


@pytest.mark.asyncio
async def test_generating_admission_checks_real_jobs_after_write_lock(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Jobs cancellation at the VN lock boundary must prevent visible admission."""
    from tldw_Server_API.app.core.DB_Management import VNAssetPacks_DB as db_module

    pack = pack_with_slots
    slot_id = pack.slots[0].id
    batch = _batch(service, pack)
    service.repo.create_item(pack_id=pack.id, slot_id=slot_id, review_status="approved")
    service.repo.update_slot(slot_id, {"status": "approved"})
    jobs = JobManager(db_path=tmp_path / "admission-jobs.db")
    jobs.create_job(
        domain="vn_assets",
        queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant",
        owner_user_id="1",
        payload={},
    )
    job = jobs.acquire_next_job(
        domain="vn_assets",
        queue=vn_asset_generation_jobs_queue(),
        worker_id="vn-worker",
        lease_seconds=120,
    )
    assert job is not None
    ready_for_admission = False
    original_lock = db_module._lock_variant

    def lock_then_cancel(conn: Any, batch_id: int, slot_id: int, variant_index: int) -> None:
        """Revoke the real Jobs lease after acquiring the VN write lock."""
        original_lock(conn, batch_id, slot_id, variant_index)
        if ready_for_admission:
            assert jobs.cancel_job(int(job["id"]))

    class BoundaryRegistry(FakeImageRegistry):
        """Arm cancellation after the initial claim admission has succeeded."""

        def resolve_backend(self, requested: str | None) -> str | None:
            nonlocal ready_for_admission
            ready_for_admission = True
            return super().resolve_backend(requested)

    monkeypatch.setattr(db_module, "_lock_variant", lock_then_cancel)
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=jobs,
        image_registry=BoundaryRegistry(adapter),
        backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
        await worker.handle_generate_variant(
            {
                "pack_id": pack.id,
                "slot_id": slot_id,
                "variant_index": 0,
                "batch_id": batch,
                "user_id": 1,
            },
            job=job,
        )

    assert service.repo.get_slot(slot_id)["status"] == "approved"
    assert adapter.requests == []
    assert service.repo.get_variant_outcome(batch, slot_id, 0)["outcome_status"] == "planned"


@pytest.mark.parametrize("kind", ["completed", "failed", "cancelled"])
def test_slot_reconciliation_failure_rolls_back_outcome_and_counters(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    kind: str,
) -> None:
    """A failed slot write cannot leave a terminal outcome committed separately."""
    pack = pack_with_slots
    batch = _batch(service, pack)
    item = _claim(service.repo, batch, pack)
    service.repo.update_slot(pack.slots[0].id, {"status": "generating"})
    service.repo.db.execute_query(
        """
        CREATE TRIGGER prevent_slot_change BEFORE UPDATE OF status ON vn_asset_slots
        BEGIN SELECT RAISE(ABORT, 'blocked slot reconciliation'); END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="blocked slot reconciliation"):
        _finish(service.repo, batch, pack, item, kind)

    assert service.repo.get_variant_outcome(batch, pack.slots[0].id, 0)["outcome_status"] == "planned"
    assert service.repo.get_item(item["id"])["review_status"] == "hidden"
    stored = service.repo.get_batch(batch)
    assert (stored["completed_count"], stored["failed_count"], stored["cancelled_count"]) == (0, 0, 0)


@pytest.mark.parametrize("first_kind", ["completed", "failed"])
@pytest.mark.concurrent
def test_concurrent_sqlite_outcomes_reconcile_after_serialized_writes(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    first_kind: str,
) -> None:
    """Concurrent siblings cannot commit mutually stale shared slot states."""
    pack = pack_with_slots
    batch = _batch(service, pack, variants=2)
    items = [_claim(service.repo, batch, pack, index=index) for index in range(2)]
    service.repo.update_slot(pack.slots[0].id, {"status": "generating"})
    barrier = threading.Barrier(2, timeout=5)

    def finish(item: dict[str, Any], kind: str) -> None:
        """Use a separate native thread's SQLite connection for each outcome."""
        try:
            barrier.wait()
            _finish(service.repo, batch, pack, item, kind)
        finally:
            service.repo.db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(finish, items[0], first_kind), executor.submit(finish, items[1], "failed")]
        for future in futures:
            future.result(timeout=10)

    stored = service.repo.get_batch(batch)
    assert (stored["completed_count"], stored["failed_count"]) == ((1, 1) if first_kind == "completed" else (0, 2))
    assert service.repo.get_slot(pack.slots[0].id)["status"] == ("reviewing" if first_kind == "completed" else "failed")


@pytest.mark.concurrent
def test_generating_admission_waits_for_sqlite_cancellation_commit(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """A contending admission observes cancellation only after owning the write lock."""
    pack = pack_with_slots
    batch = _batch(service, pack)
    _claim(service.repo, batch, pack)
    attempting_transaction = threading.Event()
    authority_called = threading.Event()

    def trace(statement: str) -> None:
        """Signal when SQLite attempts the contender's transaction entry."""
        if statement.startswith("BEGIN"):
            attempting_transaction.set()

    def validate() -> None:
        """Observe admission callback order without replacing SQLite locking."""
        authority_called.set()

    def start() -> None:
        """Attempt a fenced transition from a competing SQLite connection."""
        connection = service.repo.db.get_connection()
        connection.set_trace_callback(trace)
        try:
            service.repo.start_variant_generation(
                batch_id=batch,
                slot_id=pack.slots[0].id,
                variant_index=0,
                attempt_token=f"{batch}-0",
                validate_authority=validate,
            )
        finally:
            connection.set_trace_callback(None)
            service.repo.db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with service.repo.db.transaction():
            service.repo.cancel_batch(batch)
            future = executor.submit(start)
            assert attempting_transaction.wait(5)
            assert not authority_called.is_set()
        with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
            future.result(timeout=10)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "cancelled"
    assert service.repo.get_batch(batch)["cancelled_count"] == 1


@pytest.mark.parametrize("guard", ["wrong-token", "missing-recipe", "jobs-without-authority", "callback-error"])
def test_generating_admission_rejection_preserves_visible_state(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    guard: str,
) -> None:
    """Admission failures must preserve both review display and diagnostic state."""
    pack = pack_with_slots
    batch = _batch(service, pack)
    slot_id = pack.slots[0].id
    service.repo.claim_variant(
        batch_id=batch,
        slot_id=slot_id,
        variant_index=0,
        lease_id="jobs-lease" if guard == "jobs-without-authority" else "inline",
        attempt_token="current",
        item_fields={"pack_id": pack.id},
    )
    service.repo.update_slot(slot_id, {"status": "approved", "last_error": "keep diagnostics"})

    def unavailable_jobs() -> None:
        """Model a propagated Jobs read failure during protected admission."""
        raise OSError("Jobs unavailable")

    with pytest.raises((VNAssetGenerationError, OSError)):
        service.repo.start_variant_generation(
            batch_id=batch,
            slot_id=slot_id,
            variant_index=99 if guard == "missing-recipe" else 0,
            attempt_token="obsolete" if guard == "wrong-token" else "current",
            validate_authority=unavailable_jobs if guard == "callback-error" else None,
        )

    slot = service.repo.get_slot(slot_id)
    assert (slot["status"], slot["last_error"]) == ("approved", "keep diagnostics")


def test_cancellation_reconciles_every_slot_and_preserves_mixed_outcomes(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Repeated cancellation preserves terminal counters, approvals, and other slots."""
    pack = pack_with_slots
    batch = service.start_generation(pack.id, user_id=1, request=VNAssetGenerationRequest(variant_count=3)).batch_id
    item = _claim(service.repo, batch, pack)
    _finish(service.repo, batch, pack, item, "completed")
    service.repo.update_item_review(item["id"], review_status="approved", preferred=True)
    service.repo.fail_variant(batch_id=batch, slot_id=pack.slots[0].id, variant_index=1, error="provider failed")

    service.repo.cancel_batch(batch)
    service.repo.cancel_batch(batch)

    stored = service.repo.get_batch(batch)
    assert (stored["completed_count"], stored["failed_count"], stored["cancelled_count"]) == (
        1,
        1,
        3 * len(pack.slots) - 2,
    )
    assert [slot["status"] for slot in service.repo.list_slots(pack.id)] == [
        "approved",
        *["cancelled"] * (len(pack.slots) - 1),
    ]
    assert service.repo.get_item(item["id"])["review_status"] == "approved"
    assert service.repo.get_item(item["id"])["preferred"] == 1


def test_mixed_failure_and_cancellation_with_only_hidden_items_is_failed(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Cancelled reservations cannot turn a definitive failure into review candidates."""
    pack = pack_with_slots
    batch = _batch(service, pack, variants=2)
    item = _claim(service.repo, batch, pack)
    _claim(service.repo, batch, pack, index=1)
    _finish(service.repo, batch, pack, item, "failed")
    service.repo.cancel_batch(batch)

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "failed"
    assert service.repo.list_items(pack.id) == []
    stored = service.repo.get_batch(batch)
    assert (stored["failed_count"], stored["cancelled_count"]) == (1, 1)


def test_completed_replay_and_stale_failure_cannot_regress_approval(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Terminal replays leave the published item and slot review state immutable."""
    pack = pack_with_slots
    batch = _batch(service, pack)
    item = _claim(service.repo, batch, pack)
    _finish(service.repo, batch, pack, item, "completed")
    service.repo.update_item_review(item["id"], review_status="approved", preferred=True)
    service.repo.update_slot(pack.slots[0].id, {"status": "approved"})

    _finish(service.repo, batch, pack, item, "completed")
    _finish(service.repo, batch, pack, item, "failed")
    service.repo.cancel_batch(batch)
    service.repo.release_variant_claim(
        batch_id=batch,
        slot_id=pack.slots[0].id,
        variant_index=0,
        attempt_token=f"{batch}-0",
    )

    assert service.repo.get_slot(pack.slots[0].id)["status"] == "approved"
    assert service.repo.get_item(item["id"])["review_status"] == "approved"
    assert service.repo.get_batch(batch)["completed_count"] == 1
    assert not service.repo.item_is_unpublished(item["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("jobs_delivery", [False, True], ids=["inline", "jobs"])
async def test_backend_busy_preserves_retry_and_stable_reservation(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    jobs_delivery: bool,
) -> None:
    """Busy work remains retryable; only inline delivery releases its own claim."""
    pack = pack_with_slots
    batch = _batch(service, pack)
    job = None
    if jobs_delivery:
        job = fake_jobs.create_job(job_type="vn_asset_generate_variant", owner_user_id="1")
        job.update(status="processing", lease_id="first-lease", leased_until="2099-01-01 00:00:00")

    class BusyGate(FakeGenerationGate):
        """Model a backend that cannot currently admit another model call."""

        def try_acquire(self, backend: str, *, model: str | None = None) -> BackendGenerationLease:
            return BackendGenerationLease(acquired=False, backend=backend, model=model)

    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter),
        backend_gate=BusyGate(),
        save_vn_asset_image=RecordingVNSaver(),
        generated_files_repo=EmptyGeneratedFiles(),
    )
    payload = {"pack_id": pack.id, "slot_id": pack.slots[0].id, "variant_index": 0, "batch_id": batch, "user_id": 1}
    with pytest.raises(VNAssetGenerationError, match="vn_asset_backend_busy") as raised:
        await worker.handle_generate_variant(payload, job=dict(job) if job else None)
    assert raised.value.retryable
    outcome = service.repo.get_variant_outcome(batch, pack.slots[0].id, 0)
    assert outcome["outcome_status"] == "planned"
    assert (outcome["claim_token"] is not None) == jobs_delivery
    assert service.repo.get_slot(pack.slots[0].id)["status"] == ("generating" if jobs_delivery else "queued")
    assert adapter.requests == []

    worker.backend_gate = FakeGenerationGate()
    if job is not None:
        job["lease_id"] = "retry-lease"
    result = await worker.handle_generate_variant(payload, job=dict(job) if job else None)
    assert result["item_id"] == outcome["item_id"]
    assert service.repo.get_batch(batch)["completed_count"] == 1
