"""Public generation requests retain submission ownership without blocking the loop."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import vn_assets
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService

pytestmark = pytest.mark.integration
OPERATIONS = ("generate", "retry", "regenerate")


@dataclass
class ApiCase:
    """Real owner-scoped VN/Jobs state behind the public ASGI router."""

    service: VNAssetPackService
    jobs: JobManager
    app: FastAPI
    pack_id: int
    slot_id: int
    item_id: int

    def route(self, operation: str) -> str:
        """Return the public submission route for the selected operation."""
        base = f"/api/v1/vn/vn-assets/packs/{self.pack_id}"
        suffix = {
            "generate": "/generate",
            "retry": f"/slots/{self.slot_id}/retry",
            "regenerate": f"/items/{self.item_id}/regenerate",
        }
        return base + suffix[operation]

    def receipt(self, operation: str) -> dict[str, Any] | None:
        """Observe the receipt without completing it or claiming another request."""
        scope, resource = {
            "generate": ("vn_asset_generate", f"pack:{self.pack_id}"),
            "retry": ("vn_asset_slot_retry", f"pack:{self.pack_id}:slot:{self.slot_id}"),
            "regenerate": ("vn_asset_item_regenerate", f"pack:{self.pack_id}:item:{self.item_id}"),
        }[operation]
        return self.service.repo.get_idempotency_record(
            owner_user_id=42, scope=scope, resource_id=resource, idempotency_key="thread-receipt",
        )


@pytest.fixture
def api_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[ApiCase]:
    """Initialize real databases before any asynchronous owning-thread operation."""
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("VN_ASSET_JOBS_QUEUE", "default")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,generation")
    database_path = ":memory:" if getattr(request, "param", None) == "memory" else str(tmp_path / "vn.db")
    db = CharactersRAGDB(database_path, client_id="vn-api-thread-boundary")
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    service = VNAssetPackService(db, owner_user_id=42, jobs_manager=jobs)
    character_id = db.add_character_card({"name": "Archive keeper"})
    books = WorldBookService(db)
    book_id = books.create_world_book("Submission lore")
    books.add_entry(world_book_id=book_id, keywords=["archive"], content="Archive doors glow green.")
    pack = service.create_pack(VNAssetPackCreate(
        title="Thread-owned submission", primary_character_id=character_id, source_world_book_ids=[book_id],
    ))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="pose", variant_count=1))
    item = service.repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=0, generated_file_id=1001)
    service.review_item(item["id"], VNAssetReviewRequest(review_status="approved"))
    app = FastAPI()
    app.include_router(vn_assets.router, prefix="/api/v1/vn")

    async def current_user() -> User:
        """Supply the authenticated owner without changing route authorization."""
        return User(id=42, username="vn-api-owner")

    async def current_service() -> VNAssetPackService:
        """Keep the initialized repository and caller handle on the test loop."""
        return service

    async def current_jobs() -> JobManager:
        """Supply the actual Jobs manager, not a successful submission double."""
        return jobs

    app.dependency_overrides[get_request_user] = current_user
    app.dependency_overrides[vn_assets._service] = current_service
    app.dependency_overrides[vn_assets._job_manager] = current_jobs
    try:
        yield ApiCase(service, jobs, app, pack.id, slot.id, int(item["id"]))
    finally:
        db.close_connection()


def observe_loop(
    loop: asyncio.AbstractEventLoop, task: asyncio.Task[httpx.Response],
    started: threading.Event, release: threading.Event, cancel_count: int,
    responsive: list[bool], pending_at_cancel: list[bool],
) -> None:
    """Release held real work even when the current endpoint blocks the loop.

    A foreign observer schedules a real loop pulse, then optional cancellation.
    It does not perform database operations or manufacture a passing response.
    """
    try:
        if not started.wait(4):
            return
        pulse = threading.Event()
        loop.call_soon_threadsafe(pulse.set)
        responsive.append(pulse.wait(1))
        if responsive[-1]:
            for _request in range(cancel_count):
                acknowledged = threading.Event()

                def cancel_once(acknowledged: threading.Event = acknowledged) -> None:
                    """Request cancellation on the loop and observe unfinished ownership."""
                    task.cancel()
                    pending_at_cancel.append(not task.done())
                    acknowledged.set()

                loop.call_soon_threadsafe(cancel_once)
                if not acknowledged.wait(1):
                    return
    finally:
        release.set()


def assert_owned_close(
    observed: list[tuple[threading.Thread, sqlite3.Connection]], owner: sqlite3.Connection,
) -> None:
    """Verify public handle closure/thread exit without inspecting pool internals."""
    thread, connection = observed[0]
    assert thread is not threading.current_thread()
    assert not thread.is_alive()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    assert owner.execute("SELECT 1").fetchone()[0] == 1
    assert not owner.in_transaction


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_public_submission_holds_real_world_book_query_off_loop(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """Held snapshot queries remain responsive, materialized and replay-identical.

    Args:
        api_case: File-backed VN/Jobs state with configured lore and an approved item.
        monkeypatch: Wrap public query/boundary calls while retaining their real work.
        operation: Public generation, slot retry or item regeneration route.

    Returns:
        None: Verify loop progress, transaction ownership, closure and HTTP parity.
    """
    repo = api_case.service.repo
    owner = repo.db.get_connection()
    approved = repo.get_item(api_case.item_id)
    original_entries = WorldBookService.get_entries
    original_boundary = repo.run_worker_replay_operation
    started, release = threading.Event(), threading.Event()
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    responsive: list[bool] = []
    materialized: list[dict[str, Any]] = []

    def held_entries(book_service: WorldBookService, **kwargs: Any) -> list[dict[str, Any]]:
        """Read the actual configured book and hold its enclosing submit transaction."""
        entries = original_entries(book_service, **kwargs)
        connection = book_service.db.get_connection()
        assert connection.in_transaction
        observed.append((threading.current_thread(), connection))
        started.set()
        assert release.wait(5), "world-book observer failed to release submission"
        return entries

    async def materialized_boundary(operation: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
        """Observe only detached transport data leaving the established boundary."""
        def observe_result() -> dict[str, Any] | None:
            """Validate the callback result on its owning thread, before transfer."""
            result = operation()
            assert isinstance(result, dict)
            materialized.append(result)
            return result

        return await original_boundary(observe_result)

    monkeypatch.setattr(WorldBookService, "get_entries", held_entries)
    monkeypatch.setattr(repo, "run_worker_replay_operation", materialized_boundary)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        task = asyncio.create_task(client.post(api_case.route(operation), json={"idempotency_key": "thread-receipt"}))
        observer = threading.Thread(
            target=observe_loop,
            args=(asyncio.get_running_loop(), task, started, release, 0, responsive, []),
            name="submission-loop-observer",
        )
        observer.start()
        try:
            response = await task
            observer.join(timeout=2)
            assert not observer.is_alive()
            assert responsive == [True]
            assert_owned_close(observed, owner)
            assert response.status_code == 202
            assert materialized == [response.json()]
            replay = await client.post(api_case.route(operation), json={"idempotency_key": "thread-receipt"})
            status = await client.get(f"/api/v1/vn/vn-assets/packs/{api_case.pack_id}/generation")
            assert (replay.status_code, status.status_code) == (202, 200)
            assert replay.json() == status.json() == response.json()
            conflict = await client.post(
                api_case.route(operation), json={"idempotency_key": "thread-receipt", "variant_count": 2},
            )
            assert conflict.status_code == 409
            assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"
            assert len(repo.list_batches(api_case.pack_id)) == 1
            assert len(api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")) == 1
            assert api_case.receipt(operation)["status"] == "completed"
            recipe = repo.list_batch_recipes(response.json()["batch_id"])[0]["recipe"]
            assert "Archive doors glow green." in recipe["prompt"]
            assert repo.get_item(api_case.item_id) == approved
        finally:
            release.set()
            observer.join(timeout=6)
            if not task.done():
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("cancel_count,fail", [(0, True), (1, False), (2, False), (1, True), (2, True)])
async def test_public_submission_drains_commit_rollback_and_cancellation(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, operation: str, cancel_count: int, fail: bool,
) -> None:
    """A real receipt-linked write finishes before native error or cancellation.

    Args:
        api_case: File-backed VN/Jobs state with an existing approved item.
        monkeypatch: Hold the public batch creation inside the service transaction.
        operation: Public generation, slot retry or item regeneration route.
        cancel_count: Zero preserves the native OSError; one/two request cancellation.
        fail: Raise after the real nested write to test rollback/cancel precedence.

    Returns:
        None: Verify owned close, unchanged approvals and preserved recovery receipts.
    """
    repo = api_case.service.repo
    owner = repo.db.get_connection()
    approved = repo.get_item(api_case.item_id)
    original_batch = repo.create_batch
    started, release = threading.Event(), threading.Event()
    responsive: list[bool] = []
    pending_at_cancel: list[bool] = []
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    native_error = OSError("native submission write failure")

    def held_batch(**kwargs: Any) -> dict[str, Any]:
        """Execute real linked batch/recipe creation before holding commit or rollback."""
        batch = original_batch(**kwargs)
        connection = repo.db.get_connection()
        assert connection.in_transaction
        observed.append((threading.current_thread(), connection))
        started.set()
        assert release.wait(5), "batch observer failed to release submission"
        if fail:
            raise native_error
        return batch

    monkeypatch.setattr(repo, "create_batch", held_batch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        task = asyncio.create_task(client.post(api_case.route(operation), json={"idempotency_key": "thread-receipt"}))
        observer = threading.Thread(
            target=observe_loop,
            args=(asyncio.get_running_loop(), task, started, release, cancel_count, responsive, pending_at_cancel),
            name="submission-cancel-observer",
        )
        observer.start()
        try:
            expected = asyncio.CancelledError if cancel_count else OSError
            with pytest.raises(expected) as caught:
                await task
            if not cancel_count:
                assert caught.value is native_error
            observer.join(timeout=2)
            assert not observer.is_alive()
            assert responsive == [True]
            assert pending_at_cancel == [True] * cancel_count
            assert_owned_close(observed, owner)
            assert repo.get_item(api_case.item_id) == approved
            batches = repo.list_batches(api_case.pack_id)
            parents = api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")
            record = api_case.receipt(operation)
            if fail:
                assert batches == parents == []
                if cancel_count:
                    assert record is not None and record["status"] == "in_progress" and record["batch_id"] is None
                else:
                    assert record is None
            else:
                assert len(batches) == len(parents) == 1
                assert record is not None and record["status"] == "in_progress"
                assert record["batch_id"] == batches[0]["id"]
                replay = await client.post(api_case.route(operation), json={"idempotency_key": "thread-receipt"})
                assert replay.status_code == 202
                assert replay.json()["batch_id"] == batches[0]["id"]
                assert api_case.receipt(operation)["status"] == "completed"
                assert len(repo.list_batches(api_case.pack_id)) == len(parents) == 1
        finally:
            release.set()
            observer.join(timeout=6)
            if not task.done():
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [0, 1, 2])
async def test_postcommit_enqueue_failure_keeps_receipt_for_normal_recovery(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, cancel_count: int,
) -> None:
    """A drained failure after real Jobs commit never releases the linked VN receipt."""
    repo = api_case.service.repo
    owner = repo.db.get_connection()
    original_create = api_case.jobs.create_job
    started, release = threading.Event(), threading.Event()
    responsive: list[bool] = []
    pending_at_cancel: list[bool] = []
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    native_error = OSError("enqueue acknowledgement lost")

    def held_create(**kwargs: Any) -> dict[str, Any]:
        """Commit the real deterministic parent before losing its acknowledgement."""
        original_create(**kwargs)
        connection = repo.db.get_connection()
        assert not connection.in_transaction
        observed.append((threading.current_thread(), connection))
        started.set()
        assert release.wait(5), "enqueue observer failed to release submission"
        raise native_error

    monkeypatch.setattr(api_case.jobs, "create_job", held_create)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        task = asyncio.create_task(client.post(api_case.route("generate"), json={"idempotency_key": "thread-receipt"}))
        observer = threading.Thread(
            target=observe_loop,
            args=(asyncio.get_running_loop(), task, started, release, cancel_count, responsive, pending_at_cancel),
            name="enqueue-cancel-observer",
        )
        observer.start()
        try:
            with pytest.raises(asyncio.CancelledError if cancel_count else OSError) as caught:
                await task
            if not cancel_count:
                assert caught.value is native_error
            observer.join(timeout=2)
            assert not observer.is_alive()
            assert responsive == [True]
            assert pending_at_cancel == [True] * cancel_count
            assert_owned_close(observed, owner)
            batches = repo.list_batches(api_case.pack_id)
            parents = api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")
            assert len(batches) == len(parents) == 1
            record = api_case.receipt("generate")
            assert record is not None and record["status"] == "in_progress"
            assert record["batch_id"] == batches[0]["id"]
            assert batches[0]["enqueue_error"] == str(native_error)
            monkeypatch.setattr(api_case.jobs, "create_job", original_create)
            replay = await client.post(api_case.route("generate"), json={"idempotency_key": "thread-receipt"})
            assert replay.status_code == 202
            assert replay.json()["batch_id"] == batches[0]["id"]
            assert replay.json()["job_batch_id"] == str(parents[0]["id"])
            assert replay.json()["enqueue_error"] is None
            assert api_case.receipt("generate")["status"] == "completed"
            assert len(api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")) == 1
            assert len(repo.list_batches(api_case.pack_id)) == 1
        finally:
            release.set()
            observer.join(timeout=6)
            if not task.done():
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_public_submission_preserves_missing_target_http_mapping(api_case: ApiCase, operation: str) -> None:
    """Native target validation still returns 404 and releases only an unlinked claim."""
    original_route = api_case.route(operation)
    route = original_route
    payload: dict[str, Any] = {"idempotency_key": "thread-receipt"}
    if operation == "generate":
        payload["slot_ids"] = [999999]
    elif operation == "retry":
        route = route.replace(f"/slots/{api_case.slot_id}/", "/slots/999999/")
    else:
        route = route.replace(f"/items/{api_case.item_id}/", "/items/999999/")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        missing_key = await client.post(original_route, json={})
        assert missing_key.status_code == 422
        response = await client.post(route, json=payload)
        assert response.status_code == 404
        assert response.json()["detail"] == ("item_not_found" if operation == "regenerate" else "slot_not_found")
        assert api_case.service.repo.list_batches(api_case.pack_id) == []
        assert api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch") == []
        if operation == "generate":
            assert api_case.receipt(operation) is None
        # Correcting an unlinked invalid request with the same key must remain possible.
        corrected = await client.post(original_route, json={"idempotency_key": "thread-receipt"})
        assert corrected.status_code == 202


@pytest.mark.asyncio
async def test_strict_world_book_failure_retains_existing_http400_mapping(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict snapshot error maps to existing HTTP400 after owned rollback/close."""
    repo = api_case.service.repo
    owner = repo.db.get_connection()
    approved = repo.get_item(api_case.item_id)
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []

    def unavailable_entries(book_service: WorldBookService, **kwargs: Any) -> list[dict[str, Any]]:
        """Expose a configured-book outage through its established service boundary."""
        connection = book_service.db.get_connection()
        assert connection.in_transaction
        observed.append((threading.current_thread(), connection))
        raise sqlite3.OperationalError("private world-book read failure")

    monkeypatch.setattr(WorldBookService, "get_entries", unavailable_entries)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        response = await client.post(api_case.route("generate"), json={"idempotency_key": "thread-receipt"})
    assert response.status_code == 400
    assert response.json() == {"detail": "vn_asset_world_book_context_unavailable"}
    assert_owned_close(observed, owner)
    assert repo.list_batches(api_case.pack_id) == []
    assert api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch") == []
    assert api_case.receipt("generate") is None
    assert repo.get_item(api_case.item_id) == approved


@pytest.mark.asyncio
@pytest.mark.parametrize("api_case,active_transaction", [("memory", False), ("file", True)], indirect=["api_case"])
async def test_public_submission_preserves_caller_owned_fallback(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, active_transaction: bool,
) -> None:
    """Private memory and active caller transactions retain their live owner handle."""
    repo = api_case.service.repo
    owner = repo.db.get_connection()
    caller_thread = threading.current_thread()
    original_entries = WorldBookService.get_entries
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []

    def observe_entries(book_service: WorldBookService, **kwargs: Any) -> list[dict[str, Any]]:
        """Observe the real world-book read without blocking the intentional fallback."""
        observed.append((threading.current_thread(), book_service.db.get_connection()))
        return original_entries(book_service, **kwargs)

    monkeypatch.setattr(WorldBookService, "get_entries", observe_entries)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        if active_transaction:
            with repo.db.transaction():
                response = await client.post(api_case.route("generate"), json={"idempotency_key": "thread-receipt"})
                assert owner.in_transaction
        else:
            response = await client.post(api_case.route("generate"), json={"idempotency_key": "thread-receipt"})
    assert response.status_code == 202
    assert observed == [(caller_thread, owner)]
    assert owner.execute("SELECT 1").fetchone()[0] == 1
    assert not owner.in_transaction
