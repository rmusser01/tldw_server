from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Generator, Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
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
    VNAssetBulkReviewRequest,
    VNAssetGenerationRequest,
    VNAssetGenerationStatusResponse,
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management._vn_asset_corruption_test_support import (
    corrupt_recipe_version,
    delete_recipe_rows,
)
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
    vn_asset_jobs_queue,
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

    def get_job(self, job_id: int, **_kwargs: Any) -> dict[str, Any] | None:
        """Return the current recorded Jobs state by identity."""
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
        """Keep the output directory and registered-file records."""
        self.outputs_dir = outputs_dir
        self.records: dict[int, dict[str, Any]] = {}

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Persist generated bytes and record their item source reference."""
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
        """Look up the saved registration without changing its bytes."""
        return self.records.get(file_id)


class EmptyGeneratedFiles:
    """Represent an owner with no registered variant to replay."""
    async def get_file_by_source_ref(self, **_kwargs: Any) -> None:
        """Report no previous registration for the requested source."""
        return None


class BlockingFirstImageAdapter(FakeImageAdapter):
    """Hold the first model call to exercise concurrent delivery fences."""
    def __init__(self) -> None:
        """Create bounded release signals and a synchronized call counter."""
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self._calls = 0
        self._lock = threading.Lock()

    def generate(self, request: Any) -> ImageGenResult:
        """Wait on the first call only, then return ordinary generated bytes."""
        with self._lock:
            self._calls += 1
            first = self._calls == 1
        if first:
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError("test adapter was not released")
        return super().generate(request)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["parent", "child", "replay", "completed-replay"])
async def test_integrity_admission_releases_surviving_partial_fanout_reservations(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs, admission: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Damaged-ledger admission and replay release capacity and reject late workers."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id], variant_count=3),
    )
    fake_jobs.create_job(job_type="vn_asset_generate_variant", owner_user_id="1")
    identity = {"batch_id": batch.batch_id, "slot_id": slot.id, "variant_index": 0}
    attempt = "late"
    item = service.repo.claim_variant(
        **identity, lease_id="lease", attempt_token=attempt,
        item_fields={"pack_id": pack_with_slots.id, "generated_file_id": 17},
    )
    service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=2,
        item_fields={"pack_id": pack_with_slots.id},
    )
    service.repo.update_batch(batch.batch_id, {"enqueued_count": 1})
    delete_recipe_rows(service.repo, batch.batch_id, variant_index=1)
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(adapter),
    )
    if admission == "completed-replay":
        service.repo.complete_variant(**identity, item_id=item["id"], attempt_token=attempt)
        original_read = service.repo.get_variant_outcome_async

        async def disappear_after_observation(batch_id: int, slot_id: int, variant_index: int) -> dict[str, Any] | None:
            """Remove the completed row after admission observes it, before replay reads."""
            outcome = await original_read(batch_id, slot_id, variant_index)
            if outcome is not None and outcome["outcome_status"] == "completed":
                delete_recipe_rows(service.repo, batch_id, variant_index=0)
            return outcome

        monkeypatch.setattr(service.repo, "get_variant_outcome_async", disappear_after_observation)
    code = "vn_asset_recipe_count_mismatch" if admission == "parent" else "vn_asset_recipe_not_found"
    with pytest.raises(VNAssetGenerationError, match=code):
        if admission == "parent":
            worker.handle_enqueue_batch({"pack_id": pack_with_slots.id, "batch_id": batch.batch_id, "user_id": 1})
        elif admission == "child":
            await worker.handle_generate_variant({
                "pack_id": pack_with_slots.id, "batch_id": batch.batch_id, "user_id": 1,
                "slot_id": slot.id, "variant_index": 1,
            })
        elif admission == "completed-replay":
            await worker.handle_generate_variant({
                "pack_id": pack_with_slots.id, "batch_id": batch.batch_id, "user_id": 1,
                "slot_id": slot.id, "variant_index": 0,
            })
        else:
            await worker._replay_variant(
                pack_id=pack_with_slots.id, batch_id=batch.batch_id, user_id=1,
                slot_id=slot.id, variant_index=1,
            )
    assert service.repo.count_items_for_generation(pack_with_slots.id) == (1 if admission == "completed-replay" else 0)
    assert service.repo.get_batch(batch.batch_id)["failed_count"] == (1 if admission == "completed-replay" else 2)
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == (1 if admission == "completed-replay" else 0)
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 1) is None
    assert adapter.requests == []
    with pytest.raises(VNAssetGenerationError):
        service.repo.start_variant_generation(**identity, attempt_token=attempt)
    with pytest.raises(VNAssetGenerationError):
        service.repo.complete_variant(**identity, item_id=item["id"], attempt_token=attempt)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["child", "replay"])
async def test_integrity_reconciliation_does_not_block_async_worker_loop(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs, admission: str,
) -> None:
    """Both missing-recipe paths yield while the real slot reconciliation waits."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slots = pack_with_slots.slots[:2]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id for slot in slots], variant_count=2),
    )
    delete_recipe_rows(service.repo, batch.batch_id, slot_id=slots[0].id, variant_index=1)
    service.repo.create_batch(
        pack_id=pack_with_slots.id, requested_by_user_id=1, total_variants=2,
        options={"slot_ids": [slot.id for slot in slots], "variant_count": 1},
    )
    owner = service.repo.db.get_connection()
    main_thread = threading.get_ident()
    owned: list[tuple[threading.Thread, sqlite3.Connection]] = []
    blocked, release, responsive = threading.Event(), threading.Event(), threading.Event()

    def blocked_reader(
        _pack_id: int, slot_id: int, _user_id: int, _batches: Mapping[int, str],
        _settled: set[tuple[int, int, str]], _finishing: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Hold the supported activity read after real first-slot reconciliation."""
        connection = service.repo.db.get_connection()
        owned.append((threading.current_thread(), connection))
        assert connection.in_transaction
        if slot_id == slots[1].id:
            assert service.repo.get_batch(batch.batch_id)["failed_count"] == 3
            assert service.repo.get_slot(slots[0].id)["status"] == "failed"
            blocked.set()
            assert release.wait(3), "reconciliation safety release timed out"
        return False, False

    def safety_release() -> None:
        """Bound the RED loop stall without needing the blocked event loop."""
        if blocked.wait(3):
            responsive.wait(0.5)
        release.set()

    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)
    service.repo.legacy_activity_reader = blocked_reader
    payload = {"pack_id": pack_with_slots.id, "batch_id": batch.batch_id, "user_id": 1,
               "slot_id": slots[0].id, "variant_index": 1}
    watchdog = threading.Thread(target=safety_release)
    watchdog.start()
    operation = worker.handle_generate_variant(payload) if admission == "child" else worker._replay_variant(**payload)
    pending = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(blocked.wait, 3)
        was_blocked = not release.is_set()
        responsive.set()
        with pytest.raises(VNAssetGenerationError, match="vn_asset_recipe_not_found"):
            await pending
        assert was_blocked, "event loop resumed only after synchronous reconciliation unblocked"
        assert owned
        for thread, connection in owned:
            assert thread.ident != main_thread and not thread.is_alive()
            assert connection is not owner
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
        assert service.repo.get_batch(batch.batch_id)["failed_count"] == 3
        assert all(service.repo.get_slot(slot.id)["status"] == "failed" for slot in slots)
        assert service.repo.db.get_connection() is owner
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        watchdog.join(timeout=3)
        assert not watchdog.is_alive()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_outcome_query_does_not_block_async_worker_loop(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlled blocked real outcome query leaves the event loop responsive."""
    from tldw_Server_API.app.core.DB_Management import VNAssetPacks_DB as db_module
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    original = service.repo.db.execute_query
    main_thread = threading.get_ident()
    query_threads: list[int] = []
    blocked = threading.Event()
    release = threading.Event()
    responsive = threading.Event()
    owned_connections: list[sqlite3.Connection] = []
    closed_connection_ids: set[int] = set()
    reader_timeouts: list[int] = []

    class ObservedConnection(sqlite3.Connection):
        """Observe owned read execution while retaining the real SQLite engine."""

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            """Block only outcome reads on the connection that executes them."""
            if "SELECT outcome_status, item_id, claim_token" in sql:
                query_threads.append(threading.get_ident())
                blocked.set()
                assert release.wait(3), "query safety release timed out"
            return super().execute(sql, *args, **kwargs)

        def close(self) -> None:
            """Record disposal on the same thread that owned the reader."""
            closed_connection_ids.add(id(self))
            super().close()

    connect = sqlite3.connect
    reader_uri = f"{service.repo.db.db_path.as_uri()}?mode=ro"

    def observed_connect(database: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Observe only this VN database's independent read-only handles.

        Args:
            database (str): Original SQLite database path or URI.
            args (Any): Unchanged positional connection arguments.
            kwargs (Any): Unchanged keyword connection arguments.

        Returns:
            sqlite3.Connection: Native handle, instrumented only for reader_uri.
        """
        if database != reader_uri:
            return connect(database, *args, **kwargs)
        connection = connect(database, *args, **kwargs, factory=ObservedConnection)
        owned_connections.append(connection)
        with closing(connection.execute("PRAGMA busy_timeout")) as cursor:
            reader_timeouts.append(cursor.fetchone()[0])
        return connection

    def blocked_query(sql: str, *args: Any, **kwargs: Any) -> Any:
        """Block only outcome observations, then perform the real SQLite read."""
        if "SELECT outcome_status, item_id, claim_token" in sql:
            query_threads.append(threading.get_ident())
            blocked.set()
            assert release.wait(3), "query safety release timed out"
        return original(sql, *args, **kwargs)

    def safety_release() -> None:
        """Release on responsiveness or bounded timeout, even for the RED deadlock."""
        if blocked.wait(3):
            responsive.wait(0.5)
        release.set()

    monkeypatch.setattr(service.repo.db, "execute_query", blocked_query)
    monkeypatch.setattr(db_module.sqlite3, "connect", observed_connect)
    watchdog = threading.Thread(target=safety_release)
    watchdog.start()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        generated_files_repo=EmptyGeneratedFiles(), save_vn_asset_image=RecordingVNSaver(),
    )
    pending = asyncio.create_task(worker.handle_generate_variant({
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }))
    try:
        assert await asyncio.to_thread(blocked.wait, 3)
        was_blocked = not release.is_set()
        responsive.set()
        await pending
        assert was_blocked, "event loop could only resume after synchronous query unblocked"
        assert query_threads and all(thread != main_thread for thread in query_threads)
        assert owned_connections
        assert closed_connection_ids == {id(connection) for connection in owned_connections}
        assert reader_timeouts == [10000] * len(owned_connections)
    finally:
        release.set()
        await pending
        watchdog.join(timeout=3)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_delivery_with_same_lease_does_not_call_adapter_twice(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """A duplicate live delivery cannot generate or store the same variant twice."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_lease_cannot_publish_after_new_lease_takes_claim(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """A replaced lease cannot publish after the new delivery takes its claim."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancellation_during_adapter_does_not_register_asset(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Cancellation while the model runs prevents asset registration."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_takeover_backend_contention_is_retryable_until_stale_adapter_exits(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace, tmp_path: Path,
) -> None:
    """A takeover retries backend contention until the stale model call exits."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_backend_contention_keeps_failure_behavior(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
) -> None:
    """Legacy generation retains its terminal backend-contention behavior."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_job_lease_cannot_claim_a_variant(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
) -> None:
    """An expired Jobs lease cannot reserve a versioned variant."""
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation_phase", ["before", "during"])
async def test_cancellation_requested_live_job_cannot_publish(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, cancellation_phase: str,
) -> None:
    """Reject requested cancellation without changing the processing lease."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    jobs_path = tmp_path / "cancellation-request-jobs.db"
    jobs = JobManager(db_path=jobs_path)
    jobs.create_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        job_type="vn_asset_generate_variant", owner_user_id="1", payload={},
    )
    job = jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
        worker_id="vn-worker", lease_seconds=120,
    )
    assert job is not None
    adapter = BlockingFirstImageAdapter()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=saver, generated_files_repo=EmptyGeneratedFiles(),
    )

    def request_cancellation() -> None:
        """Persist a cancellation request while retaining the live lease."""
        with sqlite3.connect(jobs_path) as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested_at=CURRENT_TIMESTAMP WHERE id=?",
                (job["id"],),
            )
        current = jobs.get_job(int(job["id"]), owner_user_id="1")
        assert current is not None
        assert current["status"] == "processing"
        assert current["lease_id"] == job["lease_id"]
        assert current["leased_until"] == job["leased_until"]
        assert current["cancel_requested_at"] is not None
        assert jobs.has_live_processing_lease(int(job["id"]), owner_user_id="1") is False

    if cancellation_phase == "before":
        request_cancellation()
        adapter.release.set()
    pending = asyncio.create_task(worker.handle_generate_variant({
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }, job=job))
    try:
        if cancellation_phase == "during":
            assert await asyncio.to_thread(adapter.started.wait, 5)
            request_cancellation()
    finally:
        adapter.release.set()
        with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost") as caught:
            await pending

    assert caught.value.retryable is True
    assert len(adapter.requests) == (0 if cancellation_phase == "before" else 1)
    assert saver.calls == []
    outcome = service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)
    assert outcome["outcome_status"] == "planned"
    if cancellation_phase == "before":
        assert outcome["item_id"] is None
    else:
        item = service.repo.get_item(outcome["item_id"])
        assert item["review_status"] == "hidden"
        assert item["generated_file_id"] is None
    current_batch = service.repo.get_batch(batch.batch_id)
    assert current_batch["completed_count"] == 0
    assert current_batch["failed_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("lease_loss", ["cancelled", "replaced", "expired"])
async def test_lease_loss_after_storage_attachment_blocks_publication(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_loss: str,
) -> None:
    """Lease loss after storage attachment still fences variant publication."""
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
        """Attach real metadata, then revoke Jobs authority before publication."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delayed_jobs_validation_cannot_replace_new_claim(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delayed authority validation cannot overwrite a newer variant claim."""
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
        """Replace the lease during authority observation to exercise admission."""
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


@pytest.mark.integration
@pytest.mark.parametrize("transition", ["claim", "attach", "complete", "fail"])
@pytest.mark.parametrize("cancellation_requested", [False, True])
def test_repository_checks_jobs_authority_after_variant_write_lock(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str,
    cancellation_requested: bool,
) -> None:
    """Reject terminal and requested cancellation at mutation admission."""
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
        """Cancel Jobs only after the VN transition acquires write admission."""
        original_lock(*args)
        if cancellation_requested:
            with sqlite3.connect(tmp_path / "lock-validation-jobs.db") as conn:
                conn.execute(
                    "UPDATE jobs SET cancel_requested_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job["id"],),
                )
            current = jobs.get_job(int(job["id"]), owner_user_id="1")
            assert current is not None
            assert current["status"] == "processing"
            assert current["lease_id"] == job["lease_id"]
            assert current["leased_until"] == job["leased_until"]
        else:
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelled_legacy_reservation_is_reconciled_on_redelivery(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
) -> None:
    """Redelivery releases a cancelled legacy reservation without generating."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adapter_failure_after_jobs_cancellation_keeps_variant_planned(
    service: VNAssetPackService, pack_with_slots: SimpleNamespace, tmp_path: Path,
) -> None:
    """Model failure after Jobs cancellation cannot terminalize the stale claim."""
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
        """Cancel the authoritative delivery while its model call fails."""
        def generate(self, request: Any) -> ImageGenResult:
            """Revoke the job before raising a definitive adapter failure."""
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


def start_legacy_authored_batch(
    service: VNAssetPackService,
    pack_id: int,
    request: VNAssetGenerationRequest | None = None,
    *,
    jobs_manager: Any | None = None,
    freeze_recipe: bool = True,
) -> VNAssetGenerationStatusResponse:
    """Create real pre-ledger work without weakening new Start's V1 contract.

    Args:
        service (VNAssetPackService): Native repository and actual enqueue boundary.
        pack_id (int): Pack whose fixture inputs become a historical batch.
        request (VNAssetGenerationRequest | None): Legacy slot selection and count.
        jobs_manager (Any | None): Actual Jobs instance instead of the service fixture.
        freeze_recipe (bool): Whether this batch retains an authored snapshot.

    Returns:
        VNAssetGenerationStatusResponse: Submitted status after actual parent enqueue.

    Raises:
        ValueError: Fixture inputs or the actual enqueue operation are rejected.
    """
    from tldw_Server_API.app.core.VN_Assets.recipe import build_authored_recipe

    request = request or VNAssetGenerationRequest()
    pack = service.repo.get_pack(pack_id)
    assert pack is not None
    slots = service.repo.list_slots(pack_id)
    if request.slot_ids:
        slots = [slot for slot in slots if slot["id"] in request.slot_ids]
    recipe = build_authored_recipe(
        service.repo, pack, slots, owner_user_id=service.owner_user_id,
        variant_count=request.variant_count,
    )
    count = sum(slot["variant_count"] for slot in recipe["slots"])
    options = dict(request.options)
    if request.slot_ids:
        options["slot_ids"] = sorted(request.slot_ids)
    if request.variant_count is not None:
        options["variant_count"] = request.variant_count
    batch = service.repo.create_batch(
        pack_id=pack_id, requested_by_user_id=service.owner_user_id,
        status="queued", total_slots=len(slots), total_variants=count,
        planned_count=count, options=options, recipe=recipe if freeze_recipe else None,
    )
    assert batch["recipe_version"] == 0
    assert service.repo.list_batch_recipes(batch["id"]) == []
    return service._enqueue_generation_batch(
        batch, pack_id=pack_id, user_id=service.owner_user_id, jobs_manager=jobs_manager,
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

    status = start_legacy_authored_batch(service, pack_with_slots.id)
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

    status = start_legacy_authored_batch(service, pack_with_slots.id)
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
@pytest.mark.parametrize("resolution_failure", [False, True])
@pytest.mark.parametrize("async_dispatch", [False, True])
@pytest.mark.parametrize("retry_remaining", [False, True])
async def test_parent_fanout_failure_provenance_waits_for_retry_exhaustion(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    resolution_failure: bool,
    async_dispatch: bool,
    retry_remaining: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    started = start_legacy_authored_batch(service, pack_with_slots.id)
    parent = {**fake_jobs.created[0], "max_retries": 1, "retry_count": 0 if retry_remaining else 1}
    registry = FakeImageRegistry(FakeImageAdapter())
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs, image_registry=registry)
    create_job = fake_jobs.create_job

    def reject_second_child(**kwargs: Any) -> dict[str, Any]:
        children = [row for row in fake_jobs.created if row["job_type"] == "vn_asset_generate_variant"]
        if kwargs.get("job_type") == "vn_asset_generate_variant" and children:
            raise ValueError("child job quota exceeded")
        return create_job(**kwargs)

    error = "image_backend_unavailable" if resolution_failure else "child job quota exceeded"
    with monkeypatch.context() as patch:
        if resolution_failure:
            patch.setattr(registry, "resolve_backend", lambda _requested: None)
        else:
            patch.setattr(fake_jobs, "create_job", reject_second_child)
        with pytest.raises(ValueError, match=error):
            if async_dispatch:
                await worker.handle_job_async(parent)
            else:
                worker.handle_job(parent)

    status = service.get_generation_status(pack_with_slots.id)
    assert status.status == "failed"
    assert service.repo.get_batch(started.batch_id)["failed_count"] == 0
    queued_slot_id = None if resolution_failure else next(
        row["payload"]["slot_id"] for row in fake_jobs.created
        if row["job_type"] == "vn_asset_generate_variant"
    )
    for slot in pack_with_slots.slots:
        stored = service.repo.get_slot(slot.id)
        exhausted = slot.variant_count > 0 and not retry_remaining and slot.id != queued_slot_id
        assert stored["last_failed_batch_id"] == (started.batch_id if exhausted else None)
        assert stored["last_error"] == (error if exhausted else None)
        assert (stored["status"] == "failed") is exhausted
        assert status.failed_slot_recipe_available.get(slot.id, False) is exhausted

    recovered = await worker.handle_job_async(parent)
    assert recovered["status"] == "enqueued"
    assert recovered["enqueued_count"] == started.planned_count
    assert service.repo.get_batch(started.batch_id)["enqueue_error"] is None
    for slot in pack_with_slots.slots:
        stored = service.repo.get_slot(slot.id)
        assert stored["last_failed_batch_id"] is None
        assert stored["last_error"] is None
        assert stored["status"] != "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_recipe", [False, True])
@pytest.mark.parametrize("async_dispatch", [False, True])
@pytest.mark.parametrize("completion_failure", [False, True])
async def test_exhausted_fanout_preserves_fully_queued_slot_outcomes(
    service: VNAssetPackService,
    character_id: int,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_recipe: bool,
    async_dispatch: bool,
    completion_failure: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Partial fanout", primary_character_id=character_id))
    queued_slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
    ))
    later_slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="background", slot_key="background.interior", variant_count=2,
    ))
    zero_slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="depth_companion", slot_key="depth.interior", variant_count=0,
        required_for_runtime=False, depends_on_slot_id=later_slot.id,
    ))
    jobs = JobManager(db_path=tmp_path / "partial-fanout-jobs.db")
    started = start_legacy_authored_batch(
        service, pack.id, jobs_manager=jobs, freeze_recipe=not legacy_recipe,
    )
    adapter = FakeImageAdapter()
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=RecordingVNSaver(),
    )
    create_job = jobs.create_job

    def reject_later_variant(**kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload", {})
        if (kwargs.get("job_type") == "vn_asset_generate_variant"
                and payload.get("slot_id") == later_slot.id and payload.get("variant_index") == 1):
            raise RuntimeError("later variant enqueue rejected")
        return create_job(**kwargs)

    def reject_fanout_completion(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("fanout completion unavailable")

    error = "fanout completion unavailable" if completion_failure else "later variant enqueue rejected"
    with monkeypatch.context() as patch:
        if completion_failure:
            patch.setattr(service.repo, "complete_batch_fanout", reject_fanout_completion)
        else:
            patch.setattr(jobs, "create_job", reject_later_variant)
        while True:
            parent = jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_jobs_queue(), lease_seconds=120,
                worker_id="partial-fanout-worker", owner_user_id="1",
            )
            assert parent is not None
            with pytest.raises(RuntimeError, match=error):
                if async_dispatch:
                    await worker.handle_job_async(parent)
                else:
                    worker.handle_job(parent)
            exhausted = parent["retry_count"] >= parent["max_retries"]
            if not exhausted:
                assert service.repo.get_slot(later_slot.id)["last_failed_batch_id"] is None
            assert jobs.fail_job(
                parent["id"], error=error, retryable=True, backoff_seconds=0,
                worker_id="partial-fanout-worker", lease_id=parent["lease_id"],
            )
            if exhausted:
                break

    before_children = service.repo.get_slot(queued_slot.id)
    completed_children = 0
    while child := jobs.acquire_next_job(
        domain="vn_assets", queue=vn_asset_generation_jobs_queue(), lease_seconds=120,
        worker_id="variant-worker", owner_user_id="1",
    ):
        result = await worker.handle_job_async(child)
        assert result["status"] == "draft_created"
        assert jobs.complete_job(
            child["id"], result=result, worker_id="variant-worker", lease_id=child["lease_id"],
        )
        completed_children += 1

    assert completed_children == (4 if completion_failure else 3)
    assert len(service.repo.list_items(pack.id)) == completed_children
    queued = service.repo.get_slot(queued_slot.id)
    assert queued["status"] == "reviewing"
    assert queued["last_error"] is None
    assert queued["last_failed_batch_id"] is None
    assert before_children["status"] != "failed"
    assert before_children["last_failed_batch_id"] is None
    later = service.repo.get_slot(later_slot.id)
    assert later["status"] == ("reviewing" if completion_failure else "failed")
    assert later["last_failed_batch_id"] == (None if completion_failure else started.batch_id)
    assert later["last_error"] == (None if completion_failure else error)
    assert service.repo.get_slot(zero_slot.id)["last_failed_batch_id"] is None
    status = service.get_generation_status(pack.id)
    assert queued_slot.id not in status.failed_slot_batch_ids
    assert not status.failed_slot_recipe_available.get(queued_slot.id, False)
    assert status.failed_slot_recipe_available.get(later_slot.id, False) is (
        not completion_failure and not legacy_recipe
    )
    batch = service.repo.get_batch(started.batch_id)
    assert batch["completed_count"] == completed_children
    assert batch["failed_count"] == 0


@pytest.mark.parametrize("terminal_status", ["cancelled", "completed", None])
def test_exhausted_fanout_failure_preserves_terminal_batches_and_newer_slot_ownership(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str | None,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    started = start_legacy_authored_batch(service, pack_with_slots.id)
    parent = {**fake_jobs.created[0], "max_retries": 1, "retry_count": 1}
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(FakeImageAdapter()),
    )
    if terminal_status is None:
        service.start_generation(pack_with_slots.id)
    before = [service.repo.get_slot(slot.id) for slot in pack_with_slots.slots]

    def fail_after_batch_advances(*_args: Any, **_kwargs: Any) -> None:
        if terminal_status:
            service.repo.update_batch(started.batch_id, {"status": terminal_status})
        raise RuntimeError("late fanout failure")

    monkeypatch.setattr(service.repo, "complete_batch_fanout", fail_after_batch_advances)
    with pytest.raises(RuntimeError, match="late fanout failure"):
        worker.handle_job(parent)

    assert [service.repo.get_slot(slot.id) for slot in pack_with_slots.slots] == before
    assert service.repo.get_batch(started.batch_id)["status"] == (terminal_status or "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution_failure", [False, True])
async def test_retry_api_replays_source_after_real_parent_retry_exhaustion(
    service: VNAssetPackService,
    character_id: int,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    resolution_failure: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Exhausted fanout", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    started = start_legacy_authored_batch(service, pack.id, jobs_manager=jobs)
    registry = FakeImageRegistry(FakeImageAdapter())
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=jobs, image_registry=registry)
    create_job = jobs.create_job

    def reject_child(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("job_type") == "vn_asset_generate_variant":
            raise ValueError("child job quota exceeded")
        return create_job(**kwargs)

    error = "image_backend_unavailable" if resolution_failure else "child job quota exceeded"
    with monkeypatch.context() as patch:
        if resolution_failure:
            patch.setattr(registry, "resolve_backend", lambda _requested: None)
        else:
            patch.setattr(jobs, "create_job", reject_child)
        while True:
            parent = jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_jobs_queue(), lease_seconds=120,
                worker_id="fanout-worker", owner_user_id="1",
            )
            assert parent is not None
            exhausted = parent["retry_count"] >= parent["max_retries"]
            with pytest.raises(ValueError, match=error):
                await worker.handle_job_async(parent)
            status = service.get_generation_status(pack.id)
            assert status.failed_slot_batch_ids.get(slot.id) == (started.batch_id if exhausted else None)
            assert jobs.fail_job(
                parent["id"], error=error, retryable=True, backoff_seconds=0,
                worker_id="fanout-worker", lease_id=parent["lease_id"],
            )
            if exhausted:
                break

    assert jobs.list_jobs(domain="vn_assets", owner_user_id="1")[0]["status"] == "failed"
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[vn_assets_endpoint._service] = lambda: service
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: jobs
    response = TestClient(app).post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry",
        json={"idempotency_key": "exhausted-parent-retry"},
    )
    assert response.status_code == 202
    assert response.json()["source_batch_id"] == started.batch_id
    assert json.loads(service.repo.get_batch(response.json()["batch_id"])["recipe_json"])["slots"] == (
        json.loads(service.repo.get_batch(started.batch_id)["recipe_json"])["slots"]
    )


@pytest.mark.parametrize("processing", [False, True])
@pytest.mark.parametrize("lost_response", [False, True])
@pytest.mark.parametrize("active_child", [False, True])
def test_retry_api_rejects_active_source_fanout_without_cancelling_it(
    service: VNAssetPackService,
    character_id: int,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    processing: bool,
    lost_response: bool,
    active_child: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    jobs = JobManager(db_path=tmp_path / "retry-source-jobs.db")
    pack = service.create_pack(VNAssetPackCreate(title="Resumable source", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2 if active_child else 1,
    ))
    registry = FakeImageRegistry(FakeImageAdapter())
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=jobs, image_registry=registry)
    if lost_response:
        create_job = jobs.create_job

        def persist_then_fail(**kwargs: Any) -> dict[str, Any]:
            create_job(**kwargs)
            raise RuntimeError("parent enqueue response lost")

        with monkeypatch.context() as patch:
            patch.setattr(jobs, "create_job", persist_then_fail)
            with pytest.raises(RuntimeError, match="parent enqueue response lost"):
                start_legacy_authored_batch(service, pack.id, jobs_manager=jobs)
        source = service.repo.list_batches(pack.id)[0]
        assert source["job_batch_id"] is None
    else:
        status = start_legacy_authored_batch(service, pack.id, jobs_manager=jobs)
        with monkeypatch.context() as patch:
            patch.setattr(registry, "resolve_backend", lambda _requested: None)
            with pytest.raises(ValueError, match="image_backend_unavailable"):
                worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": status.batch_id, "user_id": 1})
        source = service.repo.get_batch(status.batch_id)
    parent = jobs.list_jobs(domain="vn_assets", owner_user_id="1")[0]
    parent_payload = parent["payload"]
    if active_child:
        create_job = jobs.create_job

        def reject_second_variant(**kwargs: Any) -> dict[str, Any]:
            if kwargs.get("payload", {}).get("variant_index") == 1:
                raise ValueError("child job quota exceeded")
            return create_job(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(jobs, "create_job", reject_second_variant)
            with pytest.raises(ValueError, match="child job quota exceeded"):
                worker.handle_enqueue_batch(parent_payload)
        assert jobs.cancel_job(parent["id"])
        parent = jobs.list_jobs(domain="vn_assets", owner_user_id="1", job_type="vn_asset_generate_variant")[0]
    if processing:
        parent = jobs.acquire_next_job(
            domain="vn_assets", queue=vn_asset_generation_jobs_queue() if active_child else vn_asset_jobs_queue(),
            lease_seconds=120, worker_id="source-worker", owner_user_id="1",
        )
        assert parent is not None
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[vn_assets_endpoint._service] = lambda: service
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: jobs

    response = TestClient(app).post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry",
        json={"idempotency_key": "active-source-retry", "source_batch_id": source["id"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "vn_asset_retry_source_active"
    assert len(service.repo.list_batches(pack.id)) == 1
    assert len(jobs.list_jobs(domain="vn_assets", owner_user_id="1")) == (2 if active_child else 1)
    assert jobs.get_job(parent["id"])["status"] == ("processing" if processing else "queued")
    if not processing and not active_child:
        assert jobs.cancel_job(parent["id"])
        accepted = TestClient(app).post(
            f"/api/v1/vn/vn-assets/packs/{pack.id}/slots/{slot.id}/retry",
            json={"idempotency_key": "active-source-retry", "source_batch_id": source["id"]},
        )
        assert accepted.status_code == 202
        assert accepted.json()["source_batch_id"] == source["id"]
        assert len(service.repo.list_batches(pack.id)) == 2
    else:
        resumed = worker.handle_enqueue_batch(parent_payload)
        assert resumed["enqueued_count"] == (2 if active_child else 1)
        assert len(service.repo.list_batches(pack.id)) == 1


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

    status = start_legacy_authored_batch(service, pack_with_slots.id)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_generation_status_matches_retry_eligibility_after_final_failure(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    cancelled: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FinalFailureAdapter(FakeImageAdapter):
        def generate(self, request: Any) -> ImageGenResult:
            if cancelled:
                service.cancel_generation(pack_with_slots.id)
            raise RuntimeError("final variant failure")

    slot_id = pack_with_slots.slots[0].id
    started = start_legacy_authored_batch(
        service,
        pack_with_slots.id, VNAssetGenerationRequest(slot_ids=[slot_id]),
    )
    parent = fake_jobs.created[-1]
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FinalFailureAdapter()), backend_gate=FakeGenerationGate(),
    )
    worker.handle_enqueue_batch(parent["payload"])
    child = fake_jobs.created[-1]

    with pytest.raises(RuntimeError, match="final variant failure"):
        await worker.handle_generate_variant(child["payload"])

    status = service.get_generation_status(pack_with_slots.id)
    assert status.status == ("cancelled" if cancelled else "failed")
    assert status.failed_slot_batch_ids[slot_id] == started.batch_id
    assert status.failed_slot_recipe_available[slot_id] is (not cancelled)
    if cancelled:
        with pytest.raises(ValueError, match="vn_asset_retry_source_unavailable"):
            service.retry_slot(pack_with_slots.id, slot_id)
    else:
        retry = service.retry_slot(pack_with_slots.id, slot_id)
        assert retry.source_batch_id == started.batch_id


def test_generation_status_rejects_non_owner_failure_source(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    slot_id = pack_with_slots.slots[0].id
    started = service.start_generation(
        pack_with_slots.id, VNAssetGenerationRequest(slot_ids=[slot_id]),
    )
    recipe = json.loads(service.repo.get_batch(started.batch_id)["recipe_json"])
    source = service.repo.create_batch(
        pack_id=pack_with_slots.id, requested_by_user_id=2, status="queued",
        recipe=recipe, options={"slot_ids": [slot_id]},
    )
    service.repo.record_batch_variant_failure(source["id"], slot_id=slot_id, error="foreign source")

    status = service.get_generation_status(pack_with_slots.id)
    assert status.failed_slot_batch_ids[slot_id] == source["id"]
    assert status.failed_slot_recipe_available[slot_id] is False
    with pytest.raises(ValueError, match="vn_asset_retry_source_unavailable"):
        service.retry_slot(pack_with_slots.id, slot_id)


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
    worker._record_generation_failure(batch_id=first.batch_id, slot_id=slot.id, variant_index=0, error="provider failed")
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
    status = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": status.batch_id, "user_id": 1}
    worker.handle_enqueue_batch(payload)

    worker._record_generation_failure(batch_id=status.batch_id, slot_id=slot.id, variant_index=0, error="provider failed")
    await worker._generate_variant(
        pack=service.repo.get_pack(pack.id), slot=service.repo.get_slot(slot.id),
        batch=service.repo.get_batch(status.batch_id), character=None, recipe=None,
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
    older = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    newer = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    if newer_failed:
        worker._record_generation_failure(batch_id=newer.batch_id, slot_id=slot.id, variant_index=0, error="newer failure")
    else:
        service.repo.mark_slot_generation_succeeded(slot.id, newer.batch_id)
    worker._record_generation_failure(batch_id=older.batch_id, slot_id=slot.id, variant_index=0, error="older failure")

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
    older = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": older.batch_id, "user_id": 1})
    newer = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker._record_generation_failure(batch_id=newer.batch_id, slot_id=slot.id, variant_index=0, error="newer failure")

    await worker._generate_variant(
        pack=service.repo.get_pack(pack.id), slot=service.repo.get_slot(slot.id),
        batch=service.repo.get_batch(older.batch_id), character=None, recipe=None,
        variant_index=0, user_id=1, job=None,
    )

    failed_slot = service.repo.get_slot(slot.id)
    assert failed_slot["status"] == "failed"
    assert failed_slot["last_error"] == "newer failure"
    assert failed_slot["last_failed_batch_id"] == newer.batch_id


def test_success_admission_cannot_reopen_failed_legacy_batch(
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
    batch = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)
    native_success = service.repo.record_batch_variant_success
    failure_recorded = False

    def complete_after_sibling_failure(batch_id: int) -> None:
        """Inject the sibling outcome before native completion, exactly once."""
        nonlocal failure_recorded
        assert not failure_recorded
        failure_recorded = True
        worker._record_generation_failure(batch_id=batch_id, slot_id=slot.id, variant_index=0, error="sibling failure")
        native_success(batch_id)

    monkeypatch.setattr(service.repo, "record_batch_variant_success", complete_after_sibling_failure)
    worker._record_generation_success(batch_id=batch.batch_id)

    stored = service.repo.get_batch(batch.batch_id)
    assert failure_recorded
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


@pytest.mark.integration
def test_generation_retries_original_batch_when_parent_enqueue_is_rejected(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Receipt recovery retries the original batch after parent admission fails."""
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


def test_rejected_parent_enqueue_preserves_zero_variant_slots(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    depth = next(slot for slot in pack_with_slots.slots if slot.variant_count == 0)
    before = service.repo.get_slot(depth.id)
    with pytest.raises(ValueError, match="queued job quota exceeded"):
        start_legacy_authored_batch(service, pack_with_slots.id, jobs_manager=RejectingJobs())

    batch = service.repo.list_batches(pack_with_slots.id)[0]
    stored_depth = service.repo.get_slot(depth.id)
    assert stored_depth["status"] == before["status"]
    assert stored_depth["last_error"] == before["last_error"]
    assert stored_depth["last_failed_batch_id"] == before["last_failed_batch_id"]
    assert depth.id not in service.get_generation_status(pack_with_slots.id).failed_slot_batch_ids
    for slot in pack_with_slots.slots:
        if slot.variant_count > 0:
            assert service.repo.get_slot(slot.id)["last_failed_batch_id"] == batch["id"]


@pytest.mark.parametrize("explicit_source", [False, True])
@pytest.mark.parametrize("recorded_failure", [False, True])
def test_retry_slot_api_rejects_zero_variant_source_without_enqueue(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    explicit_source: bool,
    recorded_failure: bool,
) -> None:
    depth = next(slot for slot in pack_with_slots.slots if slot.variant_count == 0)
    with pytest.raises(ValueError, match="queued job quota exceeded"):
        start_legacy_authored_batch(service, pack_with_slots.id, jobs_manager=RejectingJobs())
    source = service.repo.list_batches(pack_with_slots.id)[0]
    # Cover older enqueue failures as well as sources without recorded slot failure.
    service.repo.update_slot(depth.id, {
        "status": "failed" if recorded_failure else depth.status,
        "last_error": "queued job quota exceeded" if recorded_failure else None,
        "last_failed_batch_id": source["id"] if recorded_failure else None,
    })
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[vn_assets_endpoint._service] = lambda: service
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs
    payload: dict[str, Any] = {"idempotency_key": "zero-variant-retry"}
    if explicit_source:
        payload["source_batch_id"] = source["id"]

    response = TestClient(app).post(
        f"/api/v1/vn/vn-assets/packs/{pack_with_slots.id}/slots/{depth.id}/retry",
        json=payload,
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "vn_asset_retry_source_unavailable"
    assert len(service.repo.list_batches(pack_with_slots.id)) == 1
    assert fake_jobs.created == []
    background_item = service.repo.create_item(
        pack_id=pack_with_slots.id, slot_id=depth.depends_on_slot_id,
        variant_index=0, review_status="draft",
    )
    service.review_item_for_pack(
        pack_with_slots.id, int(background_item["id"]),
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )
    lazy_batch = service.repo.list_batches(pack_with_slots.id)[0]
    assert lazy_batch["planned_count"] == 1
    assert json.loads(lazy_batch["options_json"])["slot_ids"] == [depth.id]


@pytest.mark.asyncio
async def test_lost_parent_enqueue_response_recovers_slot_status(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class PersistThenFailJobs(FakeJobs):
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            job = super().create_job(**kwargs)
            if kwargs.get("job_type") == "vn_asset_enqueue_batch":
                raise RuntimeError("parent enqueue response lost")
            return job

    pack = service.create_pack(VNAssetPackCreate(title="Lost enqueue response", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    jobs = PersistThenFailJobs()
    with pytest.raises(RuntimeError, match="parent enqueue response lost"):
        start_legacy_authored_batch(
            service,
            pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]), jobs_manager=jobs,
        )
    batch = service.repo.list_batches(pack.id)[0]
    assert service.repo.get_slot(slot.id)["status"] == "failed"

    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": batch["id"], "user_id": 1})
    assert service.repo.get_slot(slot.id)["status"] != "failed"

    child = next(job for job in jobs.created if job["job_type"] == "vn_asset_generate_variant")
    await worker.handle_generate_variant(child["payload"], job=child)
    assert service.repo.get_batch(batch["id"])["status"] == "completed"
    assert service.repo.get_slot(slot.id)["status"] == "reviewing"


@pytest.mark.asyncio
async def test_resumed_fanout_preserves_child_success_after_lost_parent_response(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class PersistThenFailJobs(FakeJobs):
        failed_types: set[str] = set()

        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            job = super().create_job(**kwargs)
            job_type = str(kwargs.get("job_type"))
            if job_type not in self.failed_types:
                self.failed_types.add(job_type)
                raise RuntimeError(f"{job_type} response lost")
            return job

    pack = service.create_pack(VNAssetPackCreate(title="Lost responses", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    jobs = PersistThenFailJobs()
    with pytest.raises(RuntimeError, match="vn_asset_enqueue_batch response lost"):
        start_legacy_authored_batch(
            service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]), jobs_manager=jobs,
        )
    batch = service.repo.list_batches(pack.id)[0]
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    payload = {"pack_id": pack.id, "batch_id": batch["id"], "user_id": 1}

    with pytest.raises(RuntimeError, match="vn_asset_generate_variant response lost"):
        worker.handle_enqueue_batch(payload)
    child = next(job for job in jobs.created if job["job_type"] == "vn_asset_generate_variant")
    await worker.handle_generate_variant(child["payload"], job=child)
    worker.handle_enqueue_batch(payload)

    assert service.repo.get_batch(batch["id"])["status"] == "completed"
    assert service.repo.get_slot(slot.id)["status"] == "reviewing"
    assert service.repo.get_slot(slot.id)["last_failed_batch_id"] is None


def test_late_parent_enqueue_error_does_not_replace_completed_batch(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(VNAssetPackCreate(title="Late enqueue error", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    service.repo.mark_slot_generation_succeeded(slot.id, batch.batch_id)
    service.repo.update_batch(batch.batch_id, {"status": "completed"})

    service.repo.fail_batch_enqueue(batch.batch_id, "late response error")

    assert service.repo.get_batch(batch.batch_id)["status"] == "completed"
    assert service.repo.get_slot(slot.id)["status"] == "reviewing"


@pytest.mark.asyncio
async def test_rejected_newer_enqueue_records_slot_failure_after_older_success(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Overlapping batches", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    older = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FakeImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": older.batch_id, "user_id": 1})
    service.repo.mark_slot_generation_started(slot.id, older.batch_id)
    assert service.repo.get_slot(slot.id)["status"] == "generating"

    with pytest.raises(ValueError, match="queued job quota exceeded"):
        start_legacy_authored_batch(
            service,
            pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]), jobs_manager=RejectingJobs(),
        )
    newer = service.repo.list_batches(pack.id)[0]
    await worker.handle_generate_variant({
        "pack_id": pack.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": older.batch_id, "user_id": 1,
    })

    stored_slot = service.repo.get_slot(slot.id)
    assert stored_slot["status"] == "failed"
    assert stored_slot["last_failed_batch_id"] == newer["id"]
    assert stored_slot["last_error"] == "queued job quota exceeded"
    assert service.retry_slot(pack.id, slot.id).source_batch_id == newer["id"]


@pytest.mark.asyncio
async def test_retryable_variant_failure_does_not_terminalize_batch(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FlakyImageAdapter(FakeImageAdapter):
        def generate(self, request: Any) -> ImageGenResult:
            if not self.requests:
                self.requests.append(request)
                raise RuntimeError("temporary image failure")
            return super().generate(request)

    pack = service.create_pack(VNAssetPackCreate(title="Retryable variant", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    batch = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FlakyImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": batch.batch_id, "user_id": 1})
    job = fake_jobs.created[-1]

    with pytest.raises(RuntimeError, match="temporary image failure"):
        await worker.handle_generate_variant(job["payload"], job=job)
    assert service.repo.get_batch(batch.batch_id)["status"] != "failed"
    assert service.repo.get_slot(slot.id)["last_failed_batch_id"] is None

    job["retry_count"] = 1
    await worker.handle_generate_variant(job["payload"], job=job)
    assert service.repo.get_batch(batch.batch_id)["status"] == "completed"
    assert service.repo.get_batch(batch.batch_id)["failed_count"] == 0
    assert service.repo.get_slot(slot.id)["status"] == "reviewing"


@pytest.mark.asyncio
async def test_variant_failure_after_job_retry_is_recorded(
    service: VNAssetPackService,
    fake_jobs: FakeJobs,
    character_id: int,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FailingImageAdapter(FakeImageAdapter):
        def generate(self, request: Any) -> ImageGenResult:
            raise RuntimeError("persistent image failure")

    pack = service.create_pack(VNAssetPackCreate(title="Exhausted variant", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="sprite.primary"))
    batch = start_legacy_authored_batch(service, pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(FailingImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch({"pack_id": pack.id, "batch_id": batch.batch_id, "user_id": 1})
    job = fake_jobs.created[-1]

    with pytest.raises(RuntimeError, match="persistent image failure"):
        await worker.handle_generate_variant(job["payload"], job=job)
    assert service.repo.get_batch(batch.batch_id)["status"] != "failed"

    job["retry_count"] = 1
    with pytest.raises(RuntimeError, match="persistent image failure"):
        await worker.handle_generate_variant(job["payload"], job=job)
    assert service.repo.get_batch(batch.batch_id)["status"] == "failed"
    assert service.repo.get_slot(slot.id)["last_failed_batch_id"] == batch.batch_id


def test_sibling_success_does_not_clear_final_variant_failure(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(VNAssetPackCreate(title="Sibling outcome", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(
        asset_type="sprite", slot_key="sprite.primary", variant_count=2,
    ))
    batch = service.start_generation(pack.id, VNAssetGenerationRequest(slot_ids=[slot.id]))

    service.repo.mark_slot_generation_failed(slot.id, batch.batch_id, "final variant failure")
    service.repo.mark_slot_generation_succeeded(slot.id, batch.batch_id)

    stored_slot = service.repo.get_slot(slot.id)
    assert stored_slot["status"] == "failed"
    assert stored_slot["last_error"] == "final variant failure"
    assert stored_slot["last_failed_batch_id"] == batch.batch_id


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


@pytest.mark.integration
def test_unpublished_reserved_item_cannot_be_read_or_reviewed(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """An unpublished reservation is excluded from public reads and reviews."""
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


@pytest.mark.integration
def test_active_batch_prevents_deleting_frozen_slot(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    """Active generation prevents deletion of a recipe's frozen slot."""
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


@pytest.mark.integration
def test_fanout_uses_original_variants_after_slot_edit(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
) -> None:
    """Fanout retains the frozen variant count after the slot changes."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    first_slot = batch_with_slots.slots[0]
    service.repo.update_slot(first_slot.id, {"variant_count": 3})
    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs)

    worker.handle_enqueue_batch(batch_with_slots.job_payload)

    assert len(fake_jobs.created) == sum(slot.variant_count for slot in batch_with_slots.slots)
    assert all(job["payload"]["variant_index"] == 0 for job in fake_jobs.created)


@pytest.mark.integration
def test_fanout_does_not_regress_a_batch_completed_by_fast_children(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    batch_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late fanout completion cannot regress a batch settled by its children."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    original_create = fake_jobs.create_job

    def complete_during_fanout(**kwargs: Any) -> dict[str, Any]:
        """Commit a child outcome before parent fanout bookkeeping finishes."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_uses_original_recipe_after_source_edits(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    chacha_db: CharactersRAGDB,
) -> None:
    """Generation uses frozen recipe inputs rather than edited live sources."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_versioned_batch_with_missing_recipe_fails_closed(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing recipes and translate the native error through routed HTTP.

    Args:
        fake_jobs: Jobs boundary used by submission and the route dependency.
        service: Real SQLite-backed generation service.
        pack_with_slots: Owned pack and its configured generation slots.
        monkeypatch: Isolate the service error at the public route boundary.

    Returns:
        None; asserts fail-closed generation and the public error response.
    """
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    delete_recipe_rows(service.repo, batch.batch_id)
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

    def reject_start(*args: Any, **kwargs: Any) -> Any:
        """Carry the native worker error across the public service boundary.

        Args:
            args: Generation operation positional arguments, unused on failure.
            kwargs: Generation operation keyword arguments, unused on failure.

        Returns:
            Never returns; raises the actual captured damaged-ledger error.
        """
        raise error.value

    monkeypatch.setattr(service, "start_generation", reject_start)
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[vn_assets_endpoint._service] = lambda: service
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/vn/vn-assets/packs/{pack_with_slots.id}/generate",
            json={"idempotency_key": "damaged-ledger-http"},
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "vn_asset_recipe_not_found"
    assert adapter.requests == []
    assert service.repo.get_batch(batch.batch_id)["status"] == "failed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_recipe_version_never_uses_mutable_sources(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """An unsupported recipe version cannot fall back to mutable sources."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id,
        user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    corrupt_recipe_version(service.repo, batch.batch_id)
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completed_variant_redelivery_reuses_item_without_generating_again(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed redelivery replays its item without another model call."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_competing_registration_size_survives_completed_redelivery(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A converged registration, not the losing render, defines stored bytes."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    winner = b"previous-delivery-winner"
    storage = StoredVNSaver(tmp_path)
    saves = []

    async def save_winner(**kwargs: Any) -> dict[str, Any]:
        saves.append(kwargs)
        return await storage(**{**kwargs, "image_bytes": winner})

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1,
        request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    adapter = FakeImageAdapter()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=save_winner, generated_files_repo=storage,
    )
    payload = {
        "pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
        "batch_id": batch.batch_id, "user_id": 1,
    }
    first = await worker.handle_generate_variant(payload)
    item = service.repo.get_item(first["item_id"])
    assert item["bytes"] == len(winner)
    assert json.loads(item["backend_metadata_json"])["bytes_len"] == len(winner)
    service.repo.update_item_review(first["item_id"], review_status="approved", preferred=True)
    assert await worker.handle_generate_variant(payload) == first
    assert len(adapter.requests) == len(saves) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1
    assert service.repo.get_item(first["item_id"])["review_status"] == "approved"
    assert service.repo.get_item(first["item_id"])["preferred"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelled_batch_does_not_publish_reserved_variant(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Cancelled generation cannot publish a reserved variant on redelivery."""
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


@pytest.mark.integration
def test_cancellation_before_publication_rejects_reserved_variant(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Cancellation fences publication of an already reserved variant."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_variant_does_not_strand_queued_sibling(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """One failed variant leaves its queued sibling able to complete."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FailOnceAdapter(FakeImageAdapter):
        """Fail the first variant while allowing a queued sibling to complete."""
        def generate(self, request: Any) -> ImageGenResult:
            """Raise on the first request and generate subsequent requests normally."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_late_failure_does_not_regress_completed_variant(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """A late failure cannot regress a completed variant or its slot."""
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


@pytest.mark.integration
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
    """Retry recovers registered bytes off-loop without another model call."""
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
        """Expose a durable registration after an interrupted worker handoff."""
        async def get_file_by_source_ref(
            self, *, user_id: int, source_feature: str, source_ref: str
        ) -> dict[str, Any] | None:
            """Return the previously registered bytes for this exact source."""
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
        """Interrupt metadata attachment after durable file registration."""
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
    original_stat = Path.stat

    def checked_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        """Verify replay filesystem checks execute off the event-loop thread."""
        if path == saved_file:
            file_check_threads.append(threading.get_ident())
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", checked_stat)

    result = await worker.handle_generate_variant(payload)

    assert len(file_check_threads) == 1
    assert file_check_threads[0] != event_loop_thread
    assert result["generated_file_id"] == 77
    assert len(adapter.requests) == 1
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1
    assert service.repo.list_items(pack_with_slots.id)[0]["review_status"] == "draft"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replay_rejects_registered_file_owned_by_another_user(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Public replay rejects foreign bytes without generation or storage effects."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
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
    before = service.repo.get_item(item["id"])
    image_bytes = b"12345678"
    (tmp_path / "foreign.png").write_bytes(image_bytes)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))

    class ForeignFiles:
        """Expose an invalid cross-owner record to the replay boundary."""
        async def get_file_by_source_ref(
            self, *, user_id: int, source_feature: str, source_ref: str
        ) -> dict[str, Any]:
            """Return the foreign registration without concealing ownership."""
            return {
                "id": 77,
                "user_id": 2,
                "source_feature": source_feature,
                "source_ref": source_ref,
                "is_deleted": False,
                "storage_path": "foreign.png",
                "file_size_bytes": len(image_bytes),
                "mime_type": "image/png",
                "checksum": hashlib.sha256(image_bytes).hexdigest(),
            }

    adapter = FakeImageAdapter()
    gate = FakeGenerationGate()
    saver = RecordingVNSaver()
    worker = VNAssetGenerationWorker(
        repo=service.repo,
        jobs_manager=fake_jobs,
        generated_files_repo=ForeignFiles(),
        image_registry=FakeImageRegistry(adapter),
        backend_gate=gate,
        save_vn_asset_image=saver,
    )
    with pytest.raises(VNAssetGenerationError, match="vn_asset_item_storage_missing") as raised:
        await worker.handle_generate_variant({
            "batch_id": batch.batch_id, "slot_id": slot.id, "variant_index": 0,
            "user_id": 1, "pack_id": pack_with_slots.id,
        })
    assert raised.value.retryable is False
    after = service.repo.get_item(item["id"])
    assert after["generated_file_id"] is None
    registration_fields = ("generated_file_id", "storage_ref", "mime_type", "width", "height", "bytes")
    assert {key: after[key] for key in registration_fields} == {key: before[key] for key in registration_fields}
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 0
    assert service.repo.list_items(pack_with_slots.id) == []
    assert adapter.requests == []
    assert gate.requests == []
    assert saver.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("attached,invalid", [
    (attached, invalid) for attached in (False, True)
    for invalid in ("missing", "zero", "truncated", "corrupt", "owner", "deleted", "id", "feature", "source", "outside")
] + [(True, "attached_id"), (True, "attached_path")])
async def test_planned_replay_rejects_invalid_storage_without_generation_or_publication(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attached: bool, invalid: str,
) -> None:
    """Invalid planned storage fails once without generation or publication."""
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
        "checksum": hashlib.sha256(b"12345678").hexdigest(),
    }
    if attached:
        service.repo.update_item_storage(
            item["id"], generated_file_id=77, storage_ref="replay.png", mime_type="image/png",
            width=512, height=512, bytes=8,
        )
    if invalid == "corrupt":
        image.write_bytes(b"abcdefgh")
    elif invalid == "missing":
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
        """Expose the selected registration for replay integrity validation."""
        async def get_file_by_source_ref(self, **_kwargs: Any) -> dict[str, Any]:
            """Return the existing item-source registration for validation."""
            return record

        async def get_file_by_id(self, _file_id: int) -> dict[str, Any]:
            """Return attached file metadata for replay byte verification."""
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
    assert raised.value.retryable is False
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "failed"
    assert service.repo.get_item(item["id"])["review_status"] == "hidden"
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 0
    assert service.repo.get_batch(batch.batch_id)["failed_count"] == 1
    assert service.repo.count_items_for_generation(pack_with_slots.id) == 0
    assert service.repo.list_items(pack_with_slots.id) == []
    assert adapter.requests == []
    assert saver.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["valid", "missing", "corrupt"])
async def test_completed_replay_validates_bytes_without_changing_approved_review(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid: str,
) -> None:
    """Completed replay validates bytes while preserving approved history."""
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
    if invalid == "missing":
        image.unlink()
    elif invalid == "corrupt":
        image.write_bytes(b"abcdefgh")

    class Files:
        """Expose the selected registration for replay integrity validation."""
        async def get_file_by_id(self, _file_id: int) -> dict[str, Any]:
            """Return attached file metadata for replay byte verification."""
            return {
                "id": 77, "user_id": 1, "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{item['id']}", "is_deleted": False,
                "storage_path": "completed.png", "file_size_bytes": 8,
                "checksum": hashlib.sha256(b"12345678").hexdigest(),
            }

    worker = VNAssetGenerationWorker(repo=service.repo, jobs_manager=fake_jobs, generated_files_repo=Files())
    payload = {"pack_id": pack_with_slots.id, "slot_id": slot.id, "variant_index": 0,
               "batch_id": batch.batch_id, "user_id": 1}
    if invalid != "valid":
        with pytest.raises(VNAssetGenerationError) as raised:
            await worker.handle_generate_variant(payload)
        assert raised.value.retryable is False
    else:
        assert (await worker.handle_generate_variant(payload))["item_id"] == item["id"]
    assert service.repo.get_item(item["id"]) == before
    assert service.repo.get_batch(batch.batch_id)["completed_count"] == 1

    if invalid != "valid":
        replacement = service.regenerate_item(pack_with_slots.id, item["id"], user_id=1)
        adapter = FakeImageAdapter()
        saver = StoredVNSaver(tmp_path)
        replacement_worker = VNAssetGenerationWorker(
            repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(adapter),
            backend_gate=FakeGenerationGate(), save_vn_asset_image=saver, generated_files_repo=saver,
        )
        result = await replacement_worker.handle_generate_variant({**payload, "batch_id": replacement.batch_id})
        assert result["item_id"] != item["id"]
        assert service.repo.get_item(result["item_id"])["review_status"] == "draft"
        assert service.repo.get_item(item["id"]) == before


@pytest.mark.asyncio
@pytest.mark.integration
async def test_transient_replay_io_does_not_terminalize_variant(
    fake_jobs: FakeJobs, service: VNAssetPackService, pack_with_slots: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An I/O outage retries the existing fenced attempt without calling the model."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.VN_Assets import worker as worker_module

    slot = pack_with_slots.slots[0]
    batch = service.start_generation(
        pack_with_slots.id, user_id=1, request=VNAssetGenerationRequest(slot_ids=[slot.id]),
    )
    item = service.repo.reserve_variant_item(
        batch_id=batch.batch_id, slot_id=slot.id, variant_index=0,
        item_fields={"pack_id": pack_with_slots.id},
    )
    record = {"id": 77, "user_id": 1, "source_feature": "vn_assets",
              "source_ref": f"vn_asset_item:{item['id']}", "is_deleted": False,
              "storage_path": "asset.png", "file_size_bytes": 8}

    class Files:
        """Expose a valid registration while the filesystem is unavailable."""

        async def get_file_by_source_ref(self, **_kwargs: Any) -> dict[str, Any]:
            """Return the live registered file."""
            return record

    def unavailable(_path: Path, **_kwargs: Any) -> bool:
        """Simulate a temporary filesystem permission outage."""
        raise PermissionError("temporary filesystem outage")

    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: tmp_path))
    monkeypatch.setattr(worker_module, "generated_file_bytes_match", unavailable)
    adapter = FakeImageAdapter()
    worker = worker_module.VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, generated_files_repo=Files(),
        image_registry=FakeImageRegistry(adapter), backend_gate=FakeGenerationGate(),
    )
    with pytest.raises(VNAssetGenerationError) as raised:
        await worker.handle_generate_variant({"pack_id": pack_with_slots.id, "slot_id": slot.id,
                                             "variant_index": 0, "batch_id": batch.batch_id, "user_id": 1})
    assert raised.value.retryable is True
    assert service.repo.get_variant_outcome(batch.batch_id, slot.id, 0)["outcome_status"] == "planned"
    assert service.repo.get_batch(batch.batch_id)["failed_count"] == 0
    assert adapter.requests == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_variant_redelivery_does_not_increment_failure_count(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
) -> None:
    """Redelivery of a failed variant cannot charge its failure count twice."""
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class FailingAdapter:
        """Fail model generation to exercise idempotent failed redelivery."""
        def generate(self, _request: Any) -> None:
            """Raise a definitive model failure without generating bytes."""
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_versioned_batch_counts_only_committed_variants(
    fake_jobs: FakeJobs,
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Versioned batch counters include only committed variant outcomes."""
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
    with pytest.raises(VNAssetGenerationError, match="vn_asset_world_book_context_unavailable") as caught:
        service.start_generation(pack.id)
    assert caught.value.retryable
    assert "world book disk unavailable" not in str(caught.value)
    assert caught.value.__cause__ is None
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


@pytest.mark.asyncio
@pytest.mark.parametrize("variant_fails", [False, True])
@pytest.mark.parametrize("full_enqueue_rejected", [False, True])
async def test_zero_variant_full_batch_preserves_active_lazy_depth_outcome(
    service: VNAssetPackService,
    character_id: int,
    fake_jobs: FakeJobs,
    variant_fails: bool,
    full_enqueue_rejected: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    class DepthImageAdapter(FakeImageAdapter):
        def generate(self, request: Any) -> ImageGenResult:
            if variant_fails:
                raise RuntimeError("depth generation failed")
            return super().generate(request)

    pack = service.create_pack(VNAssetPackCreate(title="Active depth", primary_character_id=character_id))
    background = service.create_slot(
        pack.id, VNAssetSlotCreate(asset_type="background", slot_key="background.interior", variant_count=1),
    )
    depth = service.create_slot(
        pack.id, VNAssetSlotCreate(
            asset_type="depth_companion", slot_key="depth.interior", variant_count=0,
            required_for_runtime=False, depends_on_slot_id=background.id,
        ),
    )
    item = service.repo.create_item(pack_id=pack.id, slot_id=background.id, review_status="draft")
    service.review_item_for_pack(
        pack.id, int(item["id"]), VNAssetReviewRequest(review_status="approved", preferred=True),
    )
    parent = fake_jobs.created[-1]
    lazy_batch_id = parent["payload"]["batch_id"]
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs,
        image_registry=FakeImageRegistry(DepthImageAdapter()), backend_gate=FakeGenerationGate(),
        save_vn_asset_image=RecordingVNSaver(),
    )
    worker.handle_enqueue_batch(parent["payload"])
    child = fake_jobs.created[-1]
    service.repo.mark_slot_generation_started(depth.id, lazy_batch_id)
    assert service.get_readiness(pack.id).status == "generating"

    if full_enqueue_rejected:
        with pytest.raises(ValueError, match="queued job quota exceeded"):
            service.start_generation(pack.id, jobs_manager=RejectingJobs())
    else:
        service.start_generation(pack.id)
    full_batch = service.repo.list_batches(pack.id)[0]
    full_recipe = json.loads(full_batch["recipe_json"])
    assert next(slot for slot in full_recipe["slots"] if slot["slot_id"] == depth.id)["variant_count"] == 0

    if variant_fails:
        with pytest.raises(RuntimeError, match="depth generation failed"):
            await worker.handle_generate_variant(child["payload"])
    else:
        await worker.handle_generate_variant(child["payload"])

    stored_depth = service.repo.get_slot(depth.id)
    assert stored_depth["status"] == ("failed" if variant_fails else "reviewing")
    assert service.get_readiness(pack.id).status != "generating"
    assert stored_depth["latest_generation_batch_id"] == lazy_batch_id
    assert stored_depth["last_failed_batch_id"] == (lazy_batch_id if variant_fails else None)
    assert stored_depth["last_error"] == ("depth generation failed" if variant_fails else None)
    assert service.repo.get_slot(background.id)["latest_generation_batch_id"] == full_batch["id"]


@pytest.mark.parametrize("explicit_slot", [False, True])
@pytest.mark.parametrize("lost_parent_response", [False, True])
@pytest.mark.parametrize("legacy_recipe", [False, True])
def test_zero_work_fanout_completes_and_allows_later_lazy_depth(
    service: VNAssetPackService,
    character_id: int,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
    explicit_slot: bool,
    lost_parent_response: bool,
    legacy_recipe: bool,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack = service.create_pack(VNAssetPackCreate(title="Empty fanout", primary_character_id=character_id))
    background = service.create_slot(
        pack.id, VNAssetSlotCreate(asset_type="background", slot_key="background.interior", variant_count=0),
    )
    depth = service.create_slot(
        pack.id, VNAssetSlotCreate(
            asset_type="depth_companion", slot_key="depth.interior", variant_count=0,
            required_for_runtime=False, depends_on_slot_id=background.id,
        ),
    )
    request = VNAssetGenerationRequest(slot_ids=[depth.id] if explicit_slot else [])
    if lost_parent_response:
        create_job = fake_jobs.create_job

        def persist_then_fail(**kwargs: Any) -> dict[str, Any]:
            create_job(**kwargs)
            raise RuntimeError("parent response lost")

        with monkeypatch.context() as patch:
            patch.setattr(fake_jobs, "create_job", persist_then_fail)
            with pytest.raises(RuntimeError, match="parent response lost"):
                service.start_generation(pack.id, request)
    else:
        service.start_generation(pack.id, request)
    parent = fake_jobs.created[-1]
    batch_id = parent["payload"]["batch_id"]
    if legacy_recipe:
        with service.repo.db.transaction() as conn:
            conn.execute("UPDATE vn_asset_batches SET recipe_json = NULL WHERE id = ?", (batch_id,))
        assert service.repo.get_batch(batch_id)["recipe_json"] is None
    worker = VNAssetGenerationWorker(
        repo=service.repo, jobs_manager=fake_jobs, image_registry=FakeImageRegistry(FakeImageAdapter()),
    )

    result = worker.handle_enqueue_batch(parent["payload"])

    assert result["status"] == "completed"
    assert result["planned_count"] == 0
    assert result["enqueued_count"] == 0
    batch = service.repo.get_batch(batch_id)
    assert batch["completed_at"] is not None
    assert batch["enqueue_error"] is None
    assert len(fake_jobs.created) == 1
    assert service.get_generation_status(pack.id).status == "completed"
    worker.handle_enqueue_batch(parent["payload"])
    assert service.repo.get_batch(batch_id)["completed_at"] == batch["completed_at"]
    assert len(fake_jobs.created) == 1

    item = service.repo.create_item(pack_id=pack.id, slot_id=background.id, review_status="draft")
    service.review_item_for_pack(
        pack.id, int(item["id"]), VNAssetReviewRequest(review_status="approved", preferred=True),
    )
    lazy_batch = service.repo.get_batch(fake_jobs.created[-1]["payload"]["batch_id"])
    assert lazy_batch["id"] != batch_id
    assert lazy_batch["planned_count"] == 1
    assert json.loads(lazy_batch["options_json"])["slot_ids"] == [depth.id]


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
    release_observed = threading.Event()
    original = service_module.build_authored_recipe

    def slow_recipe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        entered.set()
        if release.wait(timeout=2):
            release_observed.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "build_authored_recipe", slow_recipe)
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs
    url = f"/api/v1/vn/vn-assets/packs/{pack.id}/generate"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request_task = asyncio.create_task(client.post(url, json={"idempotency_key": "capture-offload"}))
        try:
            await asyncio.to_thread(entered.wait, 2)
            assert entered.is_set()
            assert not request_task.done()
        finally:
            release.set()
            response = await request_task

    assert release_observed.is_set()
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
    assert service.repo.get_idempotency_record(
        owner_user_id=1, scope="vn_asset_generate", resource_id=f"pack:{pack.id}",
        idempotency_key="generate-pack-1",
    )["batch_id"] == first.json()["batch_id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


@pytest.mark.integration
def test_generation_api_recovers_unfinished_response_receipt(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the original batch when public receipt completion fails pre-ack."""
    import json

    from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository

    service = VNAssetPackService(chacha_db, owner_user_id=1, jobs_manager=fake_jobs)
    pack = service.create_pack(VNAssetPackCreate(title="Receipt Pack", primary_character_id=character_id))
    service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        """Bind the receipt recovery request to the test owner."""
        return User(id=1, username="vn-generator")

    async def override_chacha_db() -> CharactersRAGDB:
        """Use the same real metadata database across recovery requests."""
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs
    original_complete = VNAssetPacksRepository.complete_idempotency_record
    completion_calls: list[dict[str, Any]] = []

    def interrupted_completion(
        repo: VNAssetPacksRepository,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
        payload_hash: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fail before acknowledgement once, then forward unchanged store arguments."""
        arguments = {
            "owner_user_id": owner_user_id, "scope": scope, "resource_id": resource_id,
            "idempotency_key": idempotency_key, "payload_hash": payload_hash, "response": response,
        }
        completion_calls.append(arguments)
        if len(completion_calls) == 1:
            raise sqlite3.OperationalError("receipt completion unavailable")
        return original_complete(repo, **arguments)

    monkeypatch.setattr(VNAssetPacksRepository, "complete_idempotency_record", interrupted_completion)
    client = TestClient(app, raise_server_exceptions=False)
    payload = {"idempotency_key": "recover-receipt-1"}
    receipt_identity = {
        "owner_user_id": 1, "scope": "vn_asset_generate", "resource_id": f"pack:{pack.id}",
        "idempotency_key": payload["idempotency_key"],
    }
    with client:
        first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/generate", json=payload)
        assert first.status_code == 500
        assert first.text == "Internal Server Error"
        pending = service.repo.get_idempotency_record(**receipt_identity)
        assert pending is not None
        assert pending["status"] == "in_progress"
        assert json.loads(pending["response_json"]) == {}
        original_batch_id = pending["batch_id"]
        original_batch = service.repo.get_batch(original_batch_id)
        assert original_batch is not None
        assert original_batch["status"] == "queued"
        assert len(service.repo.list_batches(pack.id)) == 1
        assert len(fake_jobs.created) == 1
        original_parent_id = fake_jobs.created[0]["id"]
        assert original_batch["job_batch_id"] == str(original_parent_id)
        stable_response = {
            "batch_id": original_batch_id, "job_batch_id": str(original_parent_id),
            "status": "queued", "total_slots": 8, "total_variants": 6,
            "planned_count": 6, "enqueued_count": 0, "completed_count": 0,
            "failed_count": 0, "cancelled_count": 0, "enqueue_error": None,
            "source_batch_id": None, "recipe_available": True,
            "selected_slot_ids": [slot["id"] for slot in service.repo.list_slots(pack.id)],
        }
        first_response = {
            **stable_response, "failed_slot_batch_ids": {}, "failed_slot_recipe_available": {},
        }
        assert completion_calls == [{
            **receipt_identity, "payload_hash": pending["payload_hash"],
            "response": first_response,
        }]
        second = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/generate", json=payload)

    assert second.status_code == 202
    assert second.json()["batch_id"] == original_batch_id
    assert second.json()["job_batch_id"] == str(original_parent_id)
    assert len(service.repo.list_batches(pack.id)) == 1
    assert len(fake_jobs.created) == 1
    assert fake_jobs.created[0]["id"] == original_parent_id
    assert completion_calls == [completion_calls[0], {
        **receipt_identity, "payload_hash": pending["payload_hash"], "response": stable_response,
    }]
    completed = service.repo.get_idempotency_record(**receipt_identity)
    assert completed is not None
    assert completed["id"] == pending["id"]
    assert completed["batch_id"] == original_batch_id
    assert completed["status"] == "completed"
    assert json.loads(completed["response_json"]) == stable_response
    assert second.json() == first_response
    assert completed["payload_hash"] == pending["payload_hash"]


@pytest.mark.integration
def test_generation_receipt_recovers_parent_job_after_interruption(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt recovery creates one original parent after interrupted enqueue."""
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
        """Interrupt before the deterministic parent Job is submitted."""
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
    assert service.repo.get_idempotency_record(
        owner_user_id=1, scope="vn_asset_slot_retry",
        resource_id=f"pack:{pack.id}:slot:{slot.id}", idempotency_key="retry-slot-1",
    )["batch_id"] == first.json()["batch_id"]
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


@pytest.mark.parametrize(
    ("field", "corruption"),
    [
        ("recipe_json", "invalid_json"),
        ("recipe_json", "bool_version"),
        ("recipe_json", "float_version"),
        ("recipe_json", "missing_variant_count"),
        ("recipe_json", "string_variant_count"),
        ("recipe_json", "bool_variant_count"),
        ("recipe_json", "zero_variant_count"),
        ("recipe_json", "missing_prompt"),
        ("recipe_json", "missing_width"),
        ("recipe_json", "invalid_labels"),
        ("recipe_json", "short_seeds"),
        ("recipe_json", "null_slot"),
        ("execution_recipe_json", "invalid_json"),
        ("execution_recipe_json", "bool_version"),
        ("execution_recipe_json", "float_version"),
        ("execution_recipe_json", "missing_slots"),
        ("execution_recipe_json", "null_slots"),
        ("execution_recipe_json", "null_slot"),
        ("execution_recipe_json", "missing_backend"),
        ("execution_recipe_json", "invalid_backend"),
        ("execution_recipe_json", "invalid_model"),
        ("execution_recipe_json", "missing_target"),
    ],
)
def test_retry_slot_api_rejects_malformed_stored_snapshots_without_enqueue(
    service: VNAssetPackService,
    pack_with_slots: SimpleNamespace,
    fake_jobs: FakeJobs,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    corruption: str,
) -> None:
    """Reject corrupted recorded inputs before accepting any Retry work."""
    slot_id = pack_with_slots.slots[0].id
    source = service.start_generation(pack_with_slots.id, VNAssetGenerationRequest(slot_ids=[slot_id]))
    service.repo.update_batch(source.batch_id, {"status": "failed"})
    service.repo.update_slot(slot_id, {
        "status": "failed", "last_error": "provider failed", "last_failed_batch_id": source.batch_id,
    })
    snapshot = json.loads(service.repo.get_batch(source.batch_id)["recipe_json"])
    if field == "execution_recipe_json":
        snapshot = {"version": 1, "slots": [{"slot_id": slot_id, "backend": "test", "model": None}]}
    entry = snapshot["slots"][0]
    if corruption == "bool_version":
        snapshot["version"] = True
    elif corruption == "float_version":
        snapshot["version"] = 1.0
    elif corruption.startswith("missing_") and corruption not in {"missing_prompt", "missing_target"}:
        target = snapshot if corruption == "missing_slots" else entry
        target.pop(corruption.removeprefix("missing_"))
    elif corruption == "missing_prompt":
        entry["prompt_snapshot"].pop("prompt")
    elif corruption == "string_variant_count":
        entry["variant_count"] = "1"
    elif corruption == "bool_variant_count":
        entry["variant_count"] = True
    elif corruption == "zero_variant_count":
        entry["variant_count"] = 0
    elif corruption == "invalid_labels":
        entry["labels"] = []
    elif corruption == "short_seeds":
        entry["seeds"] = []
    elif corruption == "null_slot":
        snapshot["slots"] = [None]
    elif corruption == "null_slots":
        snapshot["slots"] = None
    elif corruption == "invalid_backend":
        entry["backend"] = []
    elif corruption == "invalid_model":
        entry["model"] = {}
    elif corruption == "missing_target":
        snapshot["slots"] = []
    raw_snapshot = "{" if corruption == "invalid_json" else json.dumps(snapshot)
    get_batch = service.repo.get_batch

    def read_corrupt_batch(batch_id: int) -> dict[str, Any] | None:
        """Simulate a malformed persisted snapshot at the repository boundary."""
        batch = get_batch(batch_id)
        return {**batch, field: raw_snapshot} if batch and batch_id == source.batch_id else batch

    monkeypatch.setattr(service.repo, "get_batch", read_corrupt_batch)
    batches_before = len(service.repo.list_batches(pack_with_slots.id))
    jobs_before = len(fake_jobs.created)
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="vn-generator")
    app.dependency_overrides[vn_assets_endpoint._service] = lambda: service
    app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: fake_jobs

    response = TestClient(app, raise_server_exceptions=False).post(
        f"/api/v1/vn/vn-assets/packs/{pack_with_slots.id}/slots/{slot_id}/retry",
        json={"idempotency_key": "corrupt-retry", "source_batch_id": source.batch_id},
    )

    assert response.status_code == 409
    expected_code = "vn_asset_recipe_invalid" if field == "recipe_json" else "vn_asset_execution_recipe_invalid"
    assert response.json()["detail"]["code"] == expected_code
    assert len(service.repo.list_batches(pack_with_slots.id)) == batches_before
    assert len(fake_jobs.created) == jobs_before


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
