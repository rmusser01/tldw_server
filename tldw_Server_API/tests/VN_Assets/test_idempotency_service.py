"""Core receipt recovery and response persistence without endpoint callbacks."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints.vn_assets import _claim_or_replay_idempotency
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetGenerationStatusResponse,
    VNAssetPackCreate,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService

pytestmark = pytest.mark.integration
GENERATION_SCOPES = ("vn_asset_generate", "vn_asset_slot_retry", "vn_asset_item_regenerate")


@pytest.fixture
def service(tmp_path: Path) -> Iterator[VNAssetPackService]:
    """Provide real, isolated VN metadata and Jobs databases."""
    db = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-receipt-service-test")
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    try:
        yield VNAssetPackService(db, owner_user_id=42, jobs_manager=jobs)
    finally:
        db.close_connection()


@pytest.fixture
def pack_id(service: VNAssetPackService) -> int:
    """Create an owned pack with one planned sprite variant."""
    character_id = service.repo.db.add_character_card({"name": "Mira"})
    pack = service.create_pack(VNAssetPackCreate(title="Receipt Pack", primary_character_id=character_id))
    service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="primary", variant_count=1))
    return pack.id


@pytest.mark.parametrize("operation", ("start", "recover", "claim"))
def test_generation_receipt_outcomes_reuse_original_batch_and_job(
    service: VNAssetPackService,
    pack_id: int,
    operation: str,
) -> None:
    """Retain generation outcomes and replay without duplicating persisted work.

    Args:
        service (VNAssetPackService): Isolated real metadata and Jobs databases.
        pack_id (int): Owned pack with one variant.
        operation (str): Service entrypoint whose outcome is exercised.

    Returns:
        None: Checks the original batch and persisted receipt outcome.
    """
    receipt = {
        "scope": "vn_asset_generate", "resource_id": f"pack:{pack_id}",
        "idempotency_key": "db-owned-transaction", "payload_hash": "payload",
    }
    if operation != "start":
        service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
        original = service.start_generation(pack_id, idempotency_receipt=receipt)
        record = service.repo.get_idempotency_record(
            owner_user_id=42, **{key: value for key, value in receipt.items() if key != "payload_hash"},
        )
    if operation == "start":
        response = service.start_generation(pack_id)
    elif operation == "recover":
        response = service.recover_generation_receipt(record, pack_id=pack_id)
    else:
        response = VNAssetGenerationStatusResponse(**service.claim_or_replay_idempotency(
            owner_user_id=42, generation_pack_id=pack_id, **receipt,
        ))
        assert service.repo.get_idempotency_record(
            owner_user_id=42, **{key: value for key, value in receipt.items() if key != "payload_hash"},
        )["status"] == "completed"
    assert response.status == "queued"
    assert len(service.repo.list_batches(pack_id)) == 1
    assert len(service.jobs_manager.list_jobs(domain="vn_assets")) == 1
    if operation != "start":
        assert response.batch_id == original.batch_id


def test_start_generation_rolls_back_batch_recipes_and_receipt_on_failure(
    service: VNAssetPackService,
    pack_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roll back a persisted batch and receipt link when submission aborts.

    Args:
        service (VNAssetPackService): Real SQLite metadata and Jobs service.
        pack_id (int): Owned pack with one planned variant.
        monkeypatch (pytest.MonkeyPatch): Injects failure after real batch writes.

    Returns:
        None: Checks that no partial batch, recipe or receipt link survives.
    """
    receipt = {
        "scope": "vn_asset_generate", "resource_id": f"pack:{pack_id}",
        "idempotency_key": "rollback-start", "payload_hash": "payload",
    }
    identity = {"owner_user_id": 42, **{key: receipt[key] for key in ("scope", "resource_id", "idempotency_key")}}
    before, _claimed = service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    create_batch = service.repo.create_batch
    written_batch_ids: list[int] = []

    def create_then_fail(**kwargs: Any) -> dict[str, Any]:
        """Persist the actual batch and recipe before injecting an interruption.

        Args:
            kwargs (Any): Original repository batch creation arguments.

        Returns:
            dict[str, Any]: Never returns; raises after the real writes.
        """
        batch = create_batch(**kwargs)
        written_batch_ids.append(batch["id"])
        assert len(service.repo.list_batch_recipes(batch["id"])) == 1
        assert service.repo.get_idempotency_record(**identity)["batch_id"] == batch["id"]
        raise RuntimeError("batch persistence interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(service.repo, "create_batch", create_then_fail)
        with pytest.raises(RuntimeError, match="^batch persistence interrupted$"):
            service.start_generation(pack_id, idempotency_receipt=receipt)
    assert service.repo.list_batches(pack_id) == [], "start rollback left a persisted batch"
    assert len(written_batch_ids) == 1
    assert service.repo.list_batch_recipes(written_batch_ids[0]) == []
    assert service.repo.get_idempotency_record(**identity) == before
    assert service.jobs_manager.list_jobs(domain="vn_assets") == []


def test_recover_generation_rolls_back_batch_update_on_failure(
    service: VNAssetPackService,
    pack_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the original batch retryable if recovery fails after metadata writes.

    Args:
        service (VNAssetPackService): Real SQLite metadata and Jobs service.
        pack_id (int): Owned pack with one planned variant.
        monkeypatch (pytest.MonkeyPatch): Injects failure after real batch update.

    Returns:
        None: Checks batch rollback without changing the durable parent Job.
    """
    receipt = {
        "scope": "vn_asset_generate", "resource_id": f"pack:{pack_id}",
        "idempotency_key": "rollback-recover", "payload_hash": "payload",
    }
    identity = {"owner_user_id": 42, **{key: receipt[key] for key in ("scope", "resource_id", "idempotency_key")}}
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    original = service.start_generation(pack_id, idempotency_receipt=receipt)
    before = service.repo.update_batch(original.batch_id, {"enqueue_error": "pending recovery"})
    record = service.repo.get_idempotency_record(**identity)
    jobs_before = service.jobs_manager.list_jobs(domain="vn_assets")
    update_batch = service.repo.update_batch

    def update_then_fail(batch_id: int, fields: Mapping[str, Any]) -> dict[str, Any] | None:
        """Perform the actual recovery update before injecting an interruption.

        Args:
            batch_id (int): Original generation batch.
            fields (Mapping[str, Any]): Unchanged recovery update fields.

        Returns:
            dict[str, Any] | None: Never returns; raises after the real update.
        """
        updated = update_batch(batch_id, fields)
        assert updated["enqueue_error"] is None
        raise RuntimeError("recovery persistence interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(service.repo, "update_batch", update_then_fail)
        with pytest.raises(RuntimeError, match="^recovery persistence interrupted$"):
            service.recover_generation_receipt(record, pack_id=pack_id)
    assert service.repo.get_batch(original.batch_id) == before, "recover rollback lost the original batch state"
    assert service.repo.get_idempotency_record(**identity) == record
    assert service.jobs_manager.list_jobs(domain="vn_assets") == jobs_before
    recovered = service.recover_generation_receipt(record, pack_id=pack_id)
    assert recovered.batch_id == original.batch_id
    assert recovered.job_batch_id == original.job_batch_id
    assert recovered.enqueue_error is None
    assert len(service.repo.list_batches(pack_id)) == 1
    assert service.jobs_manager.list_jobs(domain="vn_assets") == jobs_before


def test_claim_generation_rolls_back_completed_receipt_and_batch_on_failure(
    service: VNAssetPackService,
    pack_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roll back receipt completion and recovered batch writes together.

    Args:
        service (VNAssetPackService): Real SQLite metadata and Jobs service.
        pack_id (int): Owned pack with one planned variant.
        monkeypatch (pytest.MonkeyPatch): Injects failure after real completion.

    Returns:
        None: Checks atomic rollback and retry of the original response and Job.
    """
    receipt = {
        "scope": "vn_asset_generate", "resource_id": f"pack:{pack_id}",
        "idempotency_key": "rollback-claim", "payload_hash": "payload",
    }
    identity = {"owner_user_id": 42, **{key: receipt[key] for key in ("scope", "resource_id", "idempotency_key")}}
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    original = service.start_generation(pack_id, idempotency_receipt=receipt)
    before = service.repo.update_batch(original.batch_id, {"enqueue_error": "pending recovery"})
    record = service.repo.get_idempotency_record(**identity)
    jobs_before = service.jobs_manager.list_jobs(domain="vn_assets")
    complete_record = service.repo.complete_idempotency_record

    def complete_then_fail(**kwargs: Any) -> dict[str, Any]:
        """Complete the real receipt before injecting an acknowledgement failure.

        Args:
            kwargs (Any): Original repository receipt completion arguments.

        Returns:
            dict[str, Any]: Never returns; raises after the real completion.
        """
        completed = complete_record(**kwargs)
        assert completed["status"] == "completed"
        assert service.repo.get_batch(original.batch_id)["enqueue_error"] is None
        raise RuntimeError("receipt acknowledgement interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(service.repo, "complete_idempotency_record", complete_then_fail)
        with pytest.raises(RuntimeError, match="^receipt acknowledgement interrupted$"):
            service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id)
    assert service.repo.get_idempotency_record(**identity) == record, "claim rollback left a completed receipt"
    assert service.repo.get_batch(original.batch_id) == before
    assert service.jobs_manager.list_jobs(domain="vn_assets") == jobs_before
    recovered = service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id)
    assert recovered["batch_id"] == original.batch_id
    assert recovered["job_batch_id"] == original.job_batch_id
    assert recovered["enqueue_error"] is None
    assert service.repo.get_idempotency_record(**identity)["status"] == "completed"
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id) == recovered
    assert len(service.repo.list_batches(pack_id)) == 1
    assert service.jobs_manager.list_jobs(domain="vn_assets") == jobs_before


@pytest.mark.parametrize("scope", GENERATION_SCOPES)
def test_claim_recovers_original_batch_and_persists_response(
    service: VNAssetPackService,
    pack_id: int,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    """Recover each generation scope and replay its immutable response snapshot."""
    receipt = {
        "scope": scope,
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "recover-parent",
        "payload_hash": "same-request",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)

    def reject_enqueue(**_kwargs: Any) -> dict[str, Any]:
        """Simulate queue admission failure after the batch commits."""
        raise ValueError("queued job quota exceeded")

    with monkeypatch.context() as patch:
        patch.setattr(service.jobs_manager, "create_job", reject_enqueue)
        with pytest.raises(ValueError, match="queued job quota exceeded"):
            service.start_generation(pack_id, idempotency_receipt=receipt)
    original_batch = service.repo.list_batches(pack_id)[0]
    assert original_batch["status"] == "queued"
    assert original_batch["enqueue_error"] == "queued job quota exceeded"

    recovered = service.claim_or_replay_idempotency(
        owner_user_id=42,
        **receipt,
        generation_pack_id=pack_id,
    )
    assert recovered == {
        "batch_id": original_batch["id"],
        "job_batch_id": "1",
        "status": "queued",
        "total_slots": 1,
        "total_variants": 1,
        "planned_count": 1,
        "enqueued_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "cancelled_count": 0,
        "enqueue_error": None,
    }
    record = service.repo.get_idempotency_record(
        owner_user_id=42,
        scope=scope,
        resource_id=receipt["resource_id"],
        idempotency_key=receipt["idempotency_key"],
    )
    assert record["status"] == "completed"
    assert record["batch_id"] == original_batch["id"]
    service.repo.update_batch(original_batch["id"], {"status": "failed", "failed_count": 1})
    assert (
        service.claim_or_replay_idempotency(
            owner_user_id=42,
            **receipt,
            generation_pack_id=pack_id,
        )
        == recovered
    )
    assert len(service.repo.list_batches(pack_id)) == 1
    jobs = service.jobs_manager.list_jobs(domain="vn_assets")
    assert len(jobs) == 1
    assert jobs[0]["idempotency_key"] == f"vn_assets:user:42:pack:{pack_id}:batch:{original_batch['id']}:enqueue"


@pytest.mark.parametrize(
    "scope",
    (
        *GENERATION_SCOPES,
        "vn_asset_item_upload",
        "vn_asset_cleanup",
        "vn_asset_export",
        "vn_asset_import_preview",
        "vn_asset_import_commit",
    ),
)
def test_completed_response_is_a_json_snapshot_per_scope(
    service: VNAssetPackService,
    scope: str,
) -> None:
    """Preserve JSON values and isolate receipts by owner, resource, and scope."""
    receipt = {
        "scope": scope,
        "resource_id": "resource:1",
        "idempotency_key": "shared-key",
        "payload_hash": "payload",
    }
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt) is None
    snapshot = {"status": "queued", "optional": None, "nested": {"ids": [1, 2], "approved": False}}
    service.complete_idempotency_response(owner_user_id=42, **receipt, response=snapshot)
    snapshot["status"] = "changed"
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt) == {
        "status": "queued",
        "optional": None,
        "nested": {"ids": [1, 2], "approved": False},
    }
    for overrides in ({"owner_user_id": 43}, {"resource_id": "resource:2"}, {"scope": "another_scope"}):
        assert service.claim_or_replay_idempotency(**{"owner_user_id": 42, **receipt, **overrides}) is None
    with pytest.raises(ValueError, match="^idempotency_key_conflict$"):
        service.claim_or_replay_idempotency(owner_user_id=42, **{**receipt, "payload_hash": "different"})


@pytest.mark.parametrize("scope", (*GENERATION_SCOPES, "vn_asset_item_upload"))
def test_unlinked_receipt_stays_in_progress(
    service: VNAssetPackService,
    pack_id: int,
    scope: str,
) -> None:
    """Reject fresh duplicate claims that have no recoverable batch link."""
    receipt = {
        "scope": scope,
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "pending",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$") as caught:
        service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id)
    assert caught.value.retryable is True
    assert caught.value.context["operation"] == "claim_or_replay_idempotency"
    assert service.repo.list_batches(pack_id) == []


@pytest.mark.parametrize("batch_status", ("completed", "failed", "cancelled"))
def test_terminal_legacy_batch_is_recovered_without_enqueue(
    service: VNAssetPackService,
    pack_id: int,
    batch_status: str,
) -> None:
    """Persist terminal V0 status without reopening or enqueueing the batch."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "legacy",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    batch = service.repo.create_batch(
        pack_id=pack_id,
        requested_by_user_id=42,
        status=batch_status,
        total_slots=1,
        total_variants=1,
        idempotency_receipt=receipt,
    )
    recovered = service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id)
    assert recovered["batch_id"] == batch["id"]
    assert recovered["status"] == batch_status
    assert recovered["job_batch_id"] is None
    assert service.jobs_manager.list_jobs(domain="vn_assets") == []


def test_recovery_failure_preserves_linked_receipt_for_retry(
    service: VNAssetPackService,
    pack_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the same linked receipt recoverable after Jobs infrastructure fails."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "retry",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    batch = service.repo.create_batch(
        pack_id=pack_id,
        requested_by_user_id=42,
        status="queued",
        total_slots=1,
        total_variants=1,
        idempotency_receipt=receipt,
    )

    def reject_enqueue(**_kwargs: Any) -> dict[str, Any]:
        """Simulate a transient Jobs outage during recovery."""
        raise RuntimeError("jobs unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(service.jobs_manager, "create_job", reject_enqueue)
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id)
    record = service.repo.get_idempotency_record(
        owner_user_id=42,
        scope=receipt["scope"],
        resource_id=receipt["resource_id"],
        idempotency_key="retry",
    )
    assert record["status"] == "in_progress"
    assert record["batch_id"] == batch["id"]
    assert (
        service.claim_or_replay_idempotency(
            owner_user_id=42,
            **receipt,
            generation_pack_id=pack_id,
        )["batch_id"]
        == batch["id"]
    )


def test_optional_key_skips_claim_and_persistence(service: VNAssetPackService) -> None:
    """Retain the optional-key no-op contract for shared receipt helpers."""
    receipt = {
        "scope": "vn_asset_item_upload",
        "resource_id": "item:1",
        "idempotency_key": None,
        "payload_hash": "payload",
    }
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt) is None
    service.complete_idempotency_response(owner_user_id=42, **receipt, response={"id": 1})
    assert (
        service.repo.get_idempotency_record(
            owner_user_id=42,
            scope="vn_asset_item_upload",
            resource_id="item:1",
            idempotency_key="",
        )
        is None
    )


def test_linked_generation_receipt_requires_explicit_recovery_pack(
    service: VNAssetPackService,
    pack_id: int,
) -> None:
    """Leave linked receipts pending unless the caller opts into recovery."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "no-recovery",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    service.repo.create_batch(
        pack_id=pack_id,
        requested_by_user_id=42,
        status="queued",
        total_slots=1,
        total_variants=1,
        idempotency_receipt=receipt,
    )
    with pytest.raises(ValueError, match="^idempotency_key_in_progress$"):
        service.claim_or_replay_idempotency(owner_user_id=42, **receipt)
    assert service.jobs_manager.list_jobs(domain="vn_assets") == []


def test_response_persistence_failure_retries_without_another_parent_job(
    service: VNAssetPackService,
    pack_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry a lost completion write using the already persisted parent Job."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "lost-response",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    batch = service.repo.create_batch(
        pack_id=pack_id,
        requested_by_user_id=42,
        status="queued",
        total_slots=1,
        total_variants=1,
        idempotency_receipt=receipt,
    )

    def fail_completion(**_kwargs: Any) -> dict[str, Any]:
        """Simulate a database failure before the receipt is marked complete."""
        raise RuntimeError("response persistence interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(service.repo, "complete_idempotency_record", fail_completion)
        with pytest.raises(RuntimeError, match="response persistence interrupted"):
            service.claim_or_replay_idempotency(owner_user_id=42, **receipt, generation_pack_id=pack_id)
    assert (
        service.repo.get_idempotency_record(
            owner_user_id=42,
            scope=receipt["scope"],
            resource_id=receipt["resource_id"],
            idempotency_key="lost-response",
        )["status"]
        == "in_progress"
    )
    assert (
        service.claim_or_replay_idempotency(
            owner_user_id=42,
            **receipt,
            generation_pack_id=pack_id,
        )["batch_id"]
        == batch["id"]
    )
    assert len(service.jobs_manager.list_jobs(domain="vn_assets")) == 1


@pytest.mark.parametrize(
    ("receipt_owner", "batch_owner", "wrong_pack"),
    (
        (43, 43, False),
        (42, 43, False),
        (42, 42, True),
    ),
)
def test_recovery_rejects_unowned_or_wrong_pack_batch(
    service: VNAssetPackService,
    pack_id: int,
    receipt_owner: int,
    batch_owner: int,
    wrong_pack: bool,
) -> None:
    """Enforce receipt owner, requesting owner, and pack linkage on recovery."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "wrong-owner",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=receipt_owner, **receipt)
    batch_pack_id = pack_id
    if wrong_pack:
        batch_pack_id = service.create_pack(
            VNAssetPackCreate(
                title="Other Pack",
                primary_character_id=service.get_pack(pack_id).primary_character_id,
            )
        ).id
    batch = service.repo.create_batch(
        pack_id=batch_pack_id,
        requested_by_user_id=receipt_owner,
        status="queued",
        total_slots=1,
        total_variants=1,
        idempotency_receipt=receipt,
    )
    if batch_owner != receipt_owner:
        # Model an inconsistent historical link; new batch creation rejects it.
        service.repo.db.execute_query(
            "UPDATE vn_asset_batches SET requested_by_user_id = ? WHERE id = ?",
            (batch_owner, batch["id"]),
        )
    with pytest.raises(ValueError, match="^vn_asset_generation_receipt_not_found$"):
        service.claim_or_replay_idempotency(owner_user_id=receipt_owner, **receipt, generation_pack_id=pack_id)
    assert service.jobs_manager.list_jobs(domain="vn_assets") == []
    assert (
        service.repo.get_idempotency_record(
            owner_user_id=receipt_owner,
            scope=receipt["scope"],
            resource_id=receipt["resource_id"],
            idempotency_key="wrong-owner",
        )["status"]
        == "in_progress"
    )


@pytest.mark.parametrize("failure", ("conflict", "in_progress", "missing_pack", "wrong_batch_owner"))
def test_endpoint_keeps_receipt_http_error_mapping(
    service: VNAssetPackService,
    pack_id: int,
    failure: str,
) -> None:
    """Keep structured 409 details and legacy generation 404 error strings."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack_id}",
        "idempotency_key": "http-error",
        "payload_hash": "payload",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    if failure in {"missing_pack", "wrong_batch_owner"}:
        batch = service.repo.create_batch(
            pack_id=pack_id,
            requested_by_user_id=42,
            status="queued",
            total_slots=1,
            total_variants=1,
            idempotency_receipt=receipt,
        )
        if failure == "wrong_batch_owner":
            service.repo.db.execute_query(
                "UPDATE vn_asset_batches SET requested_by_user_id = ? WHERE id = ?",
                (43, batch["id"]),
            )
    if failure == "conflict":
        receipt["payload_hash"] = "different"
    with pytest.raises(HTTPException) as caught:
        _claim_or_replay_idempotency(
            service,
            owner_user_id=42,
            **receipt,
            generation_pack_id=pack_id + 1 if failure == "missing_pack" else pack_id,
            response_model=VNAssetGenerationStatusResponse,
        )
    error = caught.value
    if failure == "conflict":
        assert error.status_code == 409
        assert error.detail["code"] == "idempotency_key_conflict"
        assert error.detail["details"] == {"scope": receipt["scope"], "resource_id": receipt["resource_id"]}
    elif failure == "in_progress":
        assert error.status_code == 409
        assert error.detail["code"] == "idempotency_key_in_progress"
        assert error.detail["retryable"] is True
    else:
        assert error.status_code == 404
        assert error.detail == (
            "pack_not_found" if failure == "missing_pack" else "vn_asset_generation_receipt_not_found"
        )


def test_endpoint_validates_completed_response_not_core(
    service: VNAssetPackService,
) -> None:
    """Apply response schema validation only at the endpoint boundary."""
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": "pack:1",
        "idempotency_key": "invalid-response",
        "payload_hash": "payload",
    }
    service.complete_idempotency_response(owner_user_id=42, **receipt, response={"total_slots": "not-an-integer"})
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt) == {"total_slots": "not-an-integer"}
    with pytest.raises(ValidationError):
        _claim_or_replay_idempotency(
            service,
            owner_user_id=42,
            **receipt,
            response_model=VNAssetGenerationStatusResponse,
        )
