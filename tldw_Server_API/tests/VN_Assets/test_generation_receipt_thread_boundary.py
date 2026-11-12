"""Generation receipt phases drain owned native work without blocking ASGI callers."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.api.v1.endpoints import vn_assets
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetGenerationStatusResponse
from tldw_Server_API.tests.VN_Assets.test_generation_api_thread_boundary import (
    OPERATIONS,
    ApiCase,
    assert_owned_close,
    observe_loop,
)
from tldw_Server_API.tests.VN_Assets.test_generation_api_thread_boundary import (
    api_case as api_case,
)

pytestmark = pytest.mark.integration
PHASES = ("claim", "recovery", "completion", "release")
PAYLOAD = {"idempotency_key": "thread-receipt"}


async def prepare_phase(
    case: ApiCase, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch,
    operation: str, phase: str,
) -> None:
    """Prepare an actual pending linked receipt or a side-effect failure for release."""
    if phase == "recovery":
        original = vn_assets._record_idempotency_response

        def lost_response(*args: Any, **kwargs: Any) -> None:
            """Retain the supported lost-ack seam before receipt completion."""
            raise RuntimeError("receipt acknowledgement lost")

        monkeypatch.setattr(vn_assets, "_record_idempotency_response", lost_response)
        with pytest.raises(RuntimeError, match="receipt acknowledgement lost"):
            await client.post(case.route(operation), json=PAYLOAD)
        monkeypatch.setattr(vn_assets, "_record_idempotency_response", original)
        assert case.receipt(operation)["status"] == "in_progress"
        assert case.receipt(operation)["batch_id"] is not None
    elif phase == "release":
        method = {"generate": "start_generation", "retry": "retry_slot", "regenerate": "regenerate_item"}[operation]

        def reject_submission(*args: Any, **kwargs: Any) -> VNAssetGenerationStatusResponse:
            """Reject before batch creation so real receipt release must run."""
            raise ValueError("slot_not_found")

        monkeypatch.setattr(case.service, method, reject_submission)


async def held_request(
    case: ApiCase, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch,
    operation: str, phase: str, cancel_count: int, fail: bool,
) -> tuple[httpx.Response | None, BaseException | None]:
    """Hold a real public repository phase and observe scheduling, draining and close.

    Args:
        case: Real file-backed owner-scoped VN/Jobs state.
        client: Native ASGI client with server exceptions retained.
        monkeypatch: Wrap supported public repository methods, not private algorithms.
        operation: Generation, slot retry or item regeneration route.
        phase: Receipt claim, linked recovery, response completion or error release.
        cancel_count: Actual once/repeated loop cancellation while native work is held.
        fail: Raise the identical native error after the real operation has run.

    Returns:
        The HTTP response or propagated exception after owned work is drained.
    """
    repo = case.service.repo
    owner = repo.db.get_connection()
    method = {"claim": "claim_idempotency_record", "recovery": "get_idempotency_record",
              "completion": "complete_idempotency_record", "release": "release_idempotency_claim"}[phase]
    original = getattr(repo, method)
    started, release = threading.Event(), threading.Event()
    responsive: list[bool] = []
    pending: list[bool] = []
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    native = sqlite3.OperationalError("native receipt phase failure")

    def held_native(**kwargs: Any) -> Any:
        """Run the original receipt write/read and hold its exact owning connection."""
        result = original(**kwargs)
        if started.is_set():
            return result
        connection = repo.db.get_connection()
        observed.append((threading.current_thread(), connection))
        if phase == "recovery":
            assert connection.in_transaction
        started.set()
        assert release.wait(5), "receipt phase observer did not release native work"
        if fail:
            raise native
        return result

    original_boundary = repo.run_worker_replay_operation

    async def plain_boundary(callback: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
        """Validate detached callback output on its owning thread before transfer."""
        def checked_callback() -> dict[str, Any] | None:
            """Prevent a live response model or repository resource crossing threads."""
            result = callback()
            assert result is None or isinstance(result, dict)
            return result

        return await original_boundary(checked_callback)

    monkeypatch.setattr(repo, "run_worker_replay_operation", plain_boundary)
    monkeypatch.setattr(repo, method, held_native)
    task = asyncio.create_task(client.post(case.route(operation), json=PAYLOAD))
    observer = threading.Thread(target=observe_loop,
        args=(asyncio.get_running_loop(), task, started, release, cancel_count, responsive, pending),
        name="receipt-phase-observer")
    observer.start()
    response, error = None, None
    try:
        try:
            response = await task
        except (asyncio.CancelledError, sqlite3.OperationalError) as exc:
            error = exc
        observer.join(timeout=2)
        assert not observer.is_alive()
        assert responsive == [True]
        assert len(observed) == 1
        assert_owned_close(observed, owner)
        if cancel_count:
            assert pending == [True] * cancel_count
            assert isinstance(error, asyncio.CancelledError)
        elif fail:
            assert error is native
        else:
            assert error is None
        return response, error
    finally:
        release.set()
        observer.join(timeout=6)
        monkeypatch.setattr(repo, method, original)
        monkeypatch.setattr(repo, "run_worker_replay_operation", original_boundary)
        if not task.done():
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("phase", PHASES)
async def test_public_receipt_phases_hold_native_work_off_loop(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, operation: str, phase: str,
) -> None:
    """All three routes offload each complete receipt phase without changing HTTP state."""
    approved = api_case.service.repo.get_item(api_case.item_id)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        await prepare_phase(api_case, client, monkeypatch, operation, phase)
        response, _error = await held_request(api_case, client, monkeypatch, operation, phase, 0, False)
        if phase == "release":
            assert response.status_code == 404
            assert response.json()["detail"] == "slot_not_found"
            assert api_case.receipt(operation) is None
            assert api_case.submitted_batches() == []
        else:
            assert response.status_code == 202
            replay = await client.post(api_case.route(operation), json=PAYLOAD)
            assert replay.status_code == 202
            assert replay.json() == response.json()
            conflict = await client.post(api_case.route(operation), json={**PAYLOAD, "variant_count": 2})
            assert conflict.status_code == 409
            assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"
            assert api_case.receipt(operation)["status"] == "completed"
            assert len(api_case.submitted_batches()) == 1
            assert len(api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")) == 1
    assert api_case.service.repo.get_item(api_case.item_id) == approved


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("cancel_count,fail", [(0, True), (1, False), (2, False), (1, True), (2, True)])
async def test_receipt_native_error_and_once_repeated_cancel_drain_exact_phase(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, phase: str, cancel_count: int, fail: bool,
) -> None:
    """Native errors retain identity; observed cancellation wins after close and commit."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        await prepare_phase(api_case, client, monkeypatch, "generate", phase)
        await held_request(api_case, client, monkeypatch, "generate", phase, cancel_count, fail)
        record = api_case.receipt("generate")
        if phase == "release":
            assert record is None
            assert api_case.submitted_batches() == []
        elif phase == "claim":
            assert record["status"] == "in_progress"
            assert record["batch_id"] is None
            assert api_case.submitted_batches() == []
            replay = await client.post(api_case.route("generate"), json=PAYLOAD)
            assert replay.status_code == 409
            assert replay.json()["detail"]["code"] == "idempotency_key_in_progress"
        else:
            assert record["batch_id"] is not None
            assert record["status"] == ("in_progress" if phase == "recovery" and fail else "completed")
            replay = await client.post(api_case.route("generate"), json=PAYLOAD)
            assert replay.status_code == 202
            assert replay.json()["batch_id"] == record["batch_id"]
            assert api_case.receipt("generate")["status"] == "completed"
            assert len(api_case.submitted_batches()) == 1
            assert len(api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_receipt_acknowledgement_callback_failure_stays_pending_and_recovers(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """The existing HTTP acknowledgement seam remains meaningful on each public route."""
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    owner = api_case.service.repo.db.get_connection()
    original = vn_assets._record_idempotency_response

    def lose_acknowledgement(*args: Any, **kwargs: Any) -> None:
        """Fail before completing the linked committed receipt on its owning thread."""
        observed.append((threading.current_thread(), api_case.service.repo.db.get_connection()))
        raise RuntimeError("lost HTTP acknowledgement")

    monkeypatch.setattr(vn_assets, "_record_idempotency_response", lose_acknowledgement)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        with pytest.raises(RuntimeError, match="lost HTTP acknowledgement"):
            await client.post(api_case.route(operation), json=PAYLOAD)
        assert_owned_close(observed, owner)
        pending = api_case.receipt(operation)
        assert pending["status"] == "in_progress"
        assert pending["batch_id"] is not None
        monkeypatch.setattr(vn_assets, "_record_idempotency_response", original)
        response = await client.post(api_case.route(operation), json=PAYLOAD)
        assert response.status_code == 202
        assert response.json()["batch_id"] == pending["batch_id"]
        assert len(api_case.submitted_batches()) == 1
        assert len(api_case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("message,expected", [("slot_not_found", 404), ("idempotency_key_conflict", 409)])
async def test_receipt_claim_domain_failures_keep_native_http_mapping(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, operation: str, message: str, expected: int,
) -> None:
    """Offloaded receipt domain failures retain existing HTTP dispositions and no effects."""
    from tldw_Server_API.app.core.exceptions import VNAssetGenerationError

    def reject_claim(**kwargs: Any) -> tuple[dict[str, Any], bool]:
        """Reject through the supported public repository claim boundary."""
        raise VNAssetGenerationError(message)

    monkeypatch.setattr(api_case.service.repo, "claim_idempotency_record", reject_claim)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
        response = await client.post(api_case.route(operation), json=PAYLOAD)
    assert response.status_code == expected
    detail = response.json()["detail"]
    assert (detail["code"] if isinstance(detail, dict) else detail) == message
    assert api_case.submitted_batches() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("api_case,active", [("memory", False), (None, True)], indirect=["api_case"])
async def test_receipt_memory_and_active_caller_fallback_keep_owner_and_plain_data(
    api_case: ApiCase, monkeypatch: pytest.MonkeyPatch, active: bool,
) -> None:
    """Memory/active transactions remain caller-owned, with only detached dict/None output."""
    repo = api_case.service.repo
    owner = repo.db.get_connection()
    original = repo.run_worker_replay_operation
    observed: list[tuple[threading.Thread, Any]] = []

    async def observe_boundary(operation: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
        """Inspect transport data without replacing the established ownership boundary."""
        def observe_result() -> dict[str, Any] | None:
            """Capture only materialized results on the actual operation thread."""
            result = operation()
            assert result is None or isinstance(result, dict)
            observed.append((threading.current_thread(), result))
            return result

        return await original(observe_result)

    monkeypatch.setattr(repo, "run_worker_replay_operation", observe_boundary)
    if active:
        owner.execute("BEGIN")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_case.app), base_url="http://test") as client:
            response = await client.post(api_case.route("generate"), json=PAYLOAD)
        assert response.status_code == 202
        assert len(observed) == 3
        assert all(thread is threading.current_thread() for thread, _result in observed)
        assert [result for _thread, result in observed] == [None, response.json(), None]
        assert repo.db.get_connection() is owner
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        assert owner.in_transaction == active
    finally:
        if active:
            owner.rollback()
