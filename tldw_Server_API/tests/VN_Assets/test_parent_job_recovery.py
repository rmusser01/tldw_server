"""Parent health recovery against real SQLite Jobs transitions and VN receipts."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetPackCreate, VNAssetSlotCreate
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.jobs import create_enqueue_batch_job, create_generate_variant_job
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

pytestmark = [pytest.mark.integration]


@dataclass
class RecoveryCase:
    """An unfinished linked V1 receipt with one of two children enqueued."""

    service: VNAssetPackService
    jobs: JobManager
    receipt: dict[str, str]
    pack_id: int
    batch_id: int
    child_id: int

    def replay(self) -> dict[str, Any]:
        """Retry the original receipt through the public core entry point."""
        result = self.service.claim_or_replay_idempotency(
            owner_user_id=42, **self.receipt, generation_pack_id=self.pack_id,
        )
        assert result is not None
        return result

    def record(self) -> dict[str, Any]:
        """Read the persisted receipt without claiming or completing it."""
        record = self.service.repo.get_idempotency_record(
            owner_user_id=42,
            scope=self.receipt["scope"],
            resource_id=self.receipt["resource_id"],
            idempotency_key=self.receipt["idempotency_key"],
        )
        assert record is not None
        return record


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RecoveryCase]:
    """Use native-temp databases and deterministic real lease/failure operations."""
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("VN_ASSET_JOBS_QUEUE", "default")
    monkeypatch.setenv("VN_ASSET_GENERATION_JOBS_QUEUE", "generation")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,generation")
    db = CharactersRAGDB(str(tmp_path / "vn.db"), client_id="vn-parent-recovery")
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    service = VNAssetPackService(db, owner_user_id=42, jobs_manager=jobs)
    character_id = db.add_character_card({"name": "Parent Recovery"})
    pack = service.create_pack(VNAssetPackCreate(title="Recovery", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="pose", variant_count=2))
    receipt = {
        "scope": "vn_asset_generate",
        "resource_id": f"pack:{pack.id}",
        "idempotency_key": "unfinished-response",
        "payload_hash": "original-request",
    }
    service.repo.claim_idempotency_record(owner_user_id=42, **receipt)
    batch = service.repo.create_batch(
        pack_id=pack.id, requested_by_user_id=42, status="queued",
        total_slots=1, total_variants=2, planned_count=2,
        options={"variant_count": 2},
        recipes=[
            {"slot_id": slot.id, "variant_index": index, "recipe": {"prompt": "frozen", "seed": index + 7}}
            for index in range(2)
        ],
        idempotency_receipt=receipt,
    )
    child = create_generate_variant_job(
        jobs, pack_id=pack.id, batch_id=batch["id"], slot_id=slot.id, variant_index=0, user_id=42,
    )
    service.repo.update_batch(batch["id"], {"enqueued_count": 1, "enqueue_error": "interrupted fanout"})
    try:
        yield RecoveryCase(service, jobs, receipt, pack.id, int(batch["id"]), int(child["id"]))
    finally:
        db.close_connection()


def exhaust_parent(case: RecoveryCase) -> dict[str, Any]:
    """Spend the real parent retry budget, retaining its deterministic identity."""
    job = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": str(job["id"])})
    for _attempt in range(int(job["max_retries"]) + 1):
        acquired = case.jobs.acquire_next_job(
            domain="vn_assets", queue="default", job_type="vn_asset_enqueue_batch",
            owner_user_id="42", worker_id="parent-worker", lease_seconds=60,
        )
        assert acquired is not None
        assert case.jobs.fail_job(
            acquired["id"], error="fanout interrupted", retryable=True, backoff_seconds=0,
            worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
        )
    failed = case.jobs.get_job(job["id"], owner_user_id="42")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["retry_count"] == failed["max_retries"]
    return failed


def test_missing_parent_id_recovers_original_partial_batch(case: RecoveryCase) -> None:
    """A nonempty stale ID must not finalize a receipt with no live parent."""
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": "999999"})
    recipes = case.service.repo.list_batch_recipes(case.batch_id)
    recovered = case.replay()
    parent = case.jobs.get_job(int(recovered["job_batch_id"]), owner_user_id="42")
    assert parent is not None
    assert parent["status"] == "queued"
    assert parent["payload"] == {"pack_id": case.pack_id, "batch_id": case.batch_id, "user_id": 42}
    assert recovered["batch_id"] == case.batch_id
    assert recovered["enqueue_error"] is None
    assert case.service.repo.list_batch_recipes(case.batch_id) == recipes
    worker = VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs)
    worker.handle_enqueue_batch(parent["payload"])
    children = case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_generate_variant")
    assert len(children) == 2
    assert case.child_id in {child["id"] for child in children}


def test_exhausted_parent_recovers_same_identity_and_original_fanout(case: RecoveryCase) -> None:
    """Explicit unfinished-receipt retry renews the bounded parent attempt budget."""
    parent = exhaust_parent(case)
    recipes = case.service.repo.list_batch_recipes(case.batch_id)
    recovered = case.replay()
    retried = case.jobs.get_job(int(recovered["job_batch_id"]), owner_user_id="42")
    assert retried["status"] == "queued"
    assert (retried["id"], retried["uuid"], retried["idempotency_key"], retried["max_retries"]) == (
        parent["id"], parent["uuid"], parent["idempotency_key"], 3,
    )
    assert retried["retry_count"] == 0
    assert recovered["enqueue_error"] is None
    assert case.service.repo.list_batch_recipes(case.batch_id) == recipes
    assert case.record()["status"] == "completed"
    VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(retried["payload"])
    assert len(case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_generate_variant")) == 2


def test_existing_jobs_retry_cannot_restore_exhausted_parent(case: RecoveryCase) -> None:
    """Characterize the API limitation without inventing new keys or raw SQL."""
    parent = exhaust_parent(case)
    replay = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    assert replay["id"] == parent["id"]
    assert replay["status"] == "failed"
    assert case.jobs.retry_now_jobs(
        job_id=parent["id"], domain="vn_assets", queue="default", job_type="vn_asset_enqueue_batch",
    ) == 0
    assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "failed"


@pytest.mark.parametrize("failure", ("quota", "pause", "drain", "outage", "bad_id"))
def test_failed_recovery_keeps_receipt_pending(
    case: RecoveryCase, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """No rejected recovery can finalize a receipt or rewrite its batch link."""
    parent = exhaust_parent(case)
    if failure == "quota":
        monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_VN_ASSETS_USER_42", "1")
    elif failure in {"pause", "drain"}:
        case.jobs.set_queue_control("vn_assets", "default", failure)
    elif failure == "bad_id":
        case.service.repo.update_batch(case.batch_id, {"job_batch_id": "-1"})
    else:
        def unavailable(_job_id: int, *, owner_user_id: str) -> dict[str, Any] | None:
            """Model a read outage at the external Jobs boundary only."""
            raise RuntimeError("jobs unavailable")

        monkeypatch.setattr(case.jobs, "get_job", unavailable)
    with pytest.raises((VNAssetGenerationError, RuntimeError)):
        case.replay()
    assert case.record()["status"] == "in_progress"
    expected_id = "-1" if failure == "bad_id" else str(parent["id"])
    assert case.service.repo.get_batch(case.batch_id)["job_batch_id"] == expected_id


@pytest.mark.parametrize("mismatch", ("owner", "domain", "queue", "type", "payload", "key"))
def test_parent_identity_mismatch_never_admits_recovery(case: RecoveryCase, mismatch: str) -> None:
    """A linked foreign or misrouted row is never retried as this batch's parent."""
    facts: dict[str, Any] = {
        "domain": "vn_assets", "queue": "default", "job_type": "vn_asset_enqueue_batch",
        "owner_user_id": "42", "payload": {"pack_id": case.pack_id, "batch_id": case.batch_id, "user_id": 42},
        "idempotency_key": f"vn_assets:user:42:pack:{case.pack_id}:batch:{case.batch_id}:enqueue",
    }
    field, value = {
        "owner": ("owner_user_id", "43"), "domain": ("domain", "other"),
        "queue": ("queue", "generation"), "type": ("job_type", "vn_asset_generate_variant"),
        "payload": ("payload", {"pack_id": case.pack_id, "batch_id": case.batch_id + 1, "user_id": 42}),
        "key": ("idempotency_key", "unrelated"),
    }[mismatch]
    facts[field] = value
    parent = case.jobs.create_job(**facts)
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": str(parent["id"])})
    with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
        case.replay()
    assert case.record()["status"] == "in_progress"
    assert case.jobs.get_job(parent["id"])["status"] == "queued"


@pytest.mark.parametrize("full_fanout", (False, True))
def test_completed_parent_only_replays_full_fanout(case: RecoveryCase, full_fanout: bool) -> None:
    """Completed full fanout stays healthy; incomplete completion is not reopened."""
    parent = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": str(parent["id"])})
    if full_fanout:
        parent = case.jobs.get_job(parent["id"], owner_user_id="42")
        VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(parent["payload"])
    acquired = case.jobs.acquire_next_job(
        domain="vn_assets", queue="default", owner_user_id="42", worker_id="complete-parent", lease_seconds=60,
    )
    assert case.jobs.complete_job(parent["id"], worker_id=acquired["worker_id"], lease_id=acquired["lease_id"])
    if full_fanout:
        assert case.replay()["job_batch_id"] == str(parent["id"])
    else:
        with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
            case.replay()
        assert case.record()["status"] == "in_progress"
    assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "completed"
    assert all(job["status"] == "queued" for job in case.jobs.list_jobs(
        domain="vn_assets", job_type="vn_asset_generate_variant",
    ))


@pytest.mark.parametrize("saved_id", [None, "999999"])
@pytest.mark.parametrize("status", ["queued", "processing", "completed", "failed"])
def test_full_fanout_recovers_unsaved_parent_without_queue_mutation(
    case: RecoveryCase, monkeypatch: pytest.MonkeyPatch, saved_id: str | None, status: str,
) -> None:
    """An interrupted parent-ID save must not strand the original fully enqueued receipt."""
    parent = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    parent = case.jobs.get_job(parent["id"], owner_user_id="42")
    assert parent is not None
    VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(parent["payload"])
    if status != "queued":
        acquired = case.jobs.acquire_next_job(
            domain="vn_assets", queue="default", job_type="vn_asset_enqueue_batch",
            owner_user_id="42", worker_id="unsaved-parent", lease_seconds=60,
        )
        assert acquired is not None
        if status == "completed":
            assert case.jobs.complete_job(parent["id"], worker_id=acquired["worker_id"], lease_id=acquired["lease_id"])
        elif status == "failed":
            assert case.jobs.fail_job(
                parent["id"], error="interrupted response", retryable=False,
                worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
            )
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": saved_id})
    recipes = case.service.repo.list_batch_recipes(case.batch_id)
    jobs = case.jobs.list_jobs(domain="vn_assets")
    before = case.service.repo.get_batch(case.batch_id)

    def no_queue_mutation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Completed fanout is read-only even when the parent link was never saved."""
        pytest.fail("full fanout attempted Jobs creation or retry admission")

    monkeypatch.setattr(case.jobs, "create_job", no_queue_mutation)
    monkeypatch.setattr(case.jobs, "retry_failed_job_admission", no_queue_mutation)
    recovered = case.replay()
    assert recovered["job_batch_id"] == str(parent["id"])
    assert recovered["batch_id"] == case.batch_id
    assert recovered["enqueue_error"] is None
    assert case.record()["status"] == "completed"
    assert case.replay() == recovered
    assert case.service.repo.list_batch_recipes(case.batch_id) == recipes
    assert case.jobs.list_jobs(domain="vn_assets") == jobs
    after = case.service.repo.get_batch(case.batch_id)
    for field in ("status", "planned_count", "enqueued_count", "completed_count", "failed_count", "cancelled_count"):
        assert after[field] == before[field]


@pytest.mark.parametrize("status", ["cancelled", "cancel_requested"])
def test_full_fanout_unsaved_cancelled_parent_stays_pending(case: RecoveryCase, status: str) -> None:
    """Canonical lookup must not bypass terminal or in-flight administrative cancellation."""
    parent = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    parent = case.jobs.get_job(parent["id"], owner_user_id="42")
    assert parent is not None
    VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(parent["payload"])
    if status == "cancel_requested":
        assert case.jobs.acquire_next_job(
            domain="vn_assets", queue="default", job_type="vn_asset_enqueue_batch",
            owner_user_id="42", worker_id="cancel-unsaved", lease_seconds=60,
        ) is not None
    assert case.jobs.cancel_job(parent["id"], reason="admin request")
    before = case.jobs.get_job(parent["id"], owner_user_id="42")
    assert case.service.repo.get_batch(case.batch_id)["job_batch_id"] is None
    with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
        case.replay()
    assert case.record()["status"] == "in_progress"
    assert case.service.repo.get_batch(case.batch_id)["job_batch_id"] is None
    assert case.jobs.get_job(parent["id"], owner_user_id="42") == before


@pytest.mark.parametrize("mismatch", ["owner", "domain", "queue", "type", "payload", "key"])
def test_full_fanout_unsaved_foreign_parent_stays_pending(case: RecoveryCase, mismatch: str) -> None:
    """A full fanout cannot invent parent authority from a foreign or mismatched canonical row."""
    payload = {"pack_id": case.pack_id, "batch_id": case.batch_id, "user_id": 42}
    VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(payload)
    facts: dict[str, Any] = {
        "domain": "vn_assets", "queue": "default", "job_type": "vn_asset_enqueue_batch",
        "owner_user_id": "42", "payload": payload,
        "idempotency_key": f"vn_assets:user:42:pack:{case.pack_id}:batch:{case.batch_id}:enqueue",
    }
    field, value = {
        "owner": ("owner_user_id", "43"), "domain": ("domain", "other"),
        "queue": ("queue", "generation"), "type": ("job_type", "vn_asset_generate_variant"),
        "payload": ("payload", {**payload, "batch_id": case.batch_id + 1}),
        "key": ("idempotency_key", "unrelated"),
    }[mismatch]
    facts[field] = value
    parent = case.jobs.create_job(**facts)
    jobs = case.jobs.list_jobs()
    with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
        case.replay()
    assert case.record()["status"] == "in_progress"
    assert case.service.repo.get_batch(case.batch_id)["job_batch_id"] is None
    assert case.jobs.list_jobs() == jobs
    assert case.jobs.get_job(parent["id"])["status"] == "queued"


def test_full_fanout_without_any_parent_stays_pending(case: RecoveryCase) -> None:
    """A complete ledger alone cannot authorize creating a replacement parent."""
    VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(
        {"pack_id": case.pack_id, "batch_id": case.batch_id, "user_id": 42}
    )
    jobs = case.jobs.list_jobs(domain="vn_assets")
    with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
        case.replay()
    assert case.record()["status"] == "in_progress"
    assert case.jobs.list_jobs(domain="vn_assets") == jobs


def terminal_full_fanout_parent(case: RecoveryCase, status: str) -> dict[str, Any]:
    """Finish real parent fanout and its lease, retaining an approved history row."""
    slot_id = int(case.service.repo.list_batch_recipes(case.batch_id)[0]["slot_id"])
    case.service.repo.create_item(
        pack_id=case.pack_id, slot_id=slot_id, review_status="approved", preferred=True,
        source="uploaded", source_prompt_snapshot={"prompt": "historical approval"},
    )
    created = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    parent = case.jobs.get_job(created["id"], owner_user_id="42")
    assert parent is not None
    VNAssetGenerationWorker(repo=case.service.repo, jobs_manager=case.jobs).handle_enqueue_batch(parent["payload"])
    batch = case.service.repo.get_batch(case.batch_id)
    assert batch is not None
    assert batch["enqueued_count"] == batch["planned_count"] == 2
    children = case.jobs.list_jobs(
        domain="vn_assets", queue="generation", job_type="vn_asset_generate_variant", owner_user_id="42",
    )
    assert len(children) == 2
    assert case.child_id in {child["id"] for child in children}
    acquired = case.jobs.acquire_next_job(
        domain="vn_assets", queue="default", job_type="vn_asset_enqueue_batch",
        owner_user_id="42", worker_id="archive-contract-parent", lease_seconds=60,
    )
    assert acquired is not None and acquired["id"] == parent["id"]
    if status == "completed":
        assert case.jobs.complete_job(parent["id"], worker_id=acquired["worker_id"], lease_id=acquired["lease_id"])
    else:
        assert status == "failed"
        assert case.jobs.fail_job(
            parent["id"], error="response interrupted after fanout", retryable=False,
            worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
        )
    terminal = case.jobs.get_job(parent["id"], owner_user_id="42")
    assert terminal is not None and terminal["status"] == status
    return terminal


@pytest.mark.parametrize("saved_parent", ["unsaved", "linked"])
@pytest.mark.parametrize("status", ["completed", "failed"])
def test_archive_only_full_fanout_parent_keeps_public_receipt_pending(
    case: RecoveryCase, monkeypatch: pytest.MonkeyPatch, saved_parent: str, status: str,
) -> None:
    """Native archived parents are evidence, not active authority, even after complete fanout."""
    parent = terminal_full_fanout_parent(case, status)
    saved_id = None if saved_parent == "unsaved" else str(parent["id"])
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": saved_id})
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "0")
    assert case.jobs.prune_jobs(
        statuses=[status], older_than_days=0, domain="vn_assets", queue="default",
        job_type="vn_asset_enqueue_batch",
    ) == 1
    scope = {
        "idempotency_key": f"vn_assets:user:42:pack:{case.pack_id}:batch:{case.batch_id}:enqueue",
        "domain": "vn_assets", "queue": "default", "job_type": "vn_asset_enqueue_batch", "owner_user_id": "42",
    }
    native_lookup = case.jobs.get_job_or_archived_by_idempotency_key
    native_get_job = case.jobs.get_job
    archived = native_lookup(**scope)
    assert archived is not None and archived["archived"] is True
    assert (archived["id"], archived["uuid"], archived["status"], archived["payload"]) == (
        parent["id"], parent["uuid"], status, {"pack_id": case.pack_id, "batch_id": case.batch_id, "user_id": 42},
    )
    assert all(archived[field] == value for field, value in scope.items())
    assert native_get_job(parent["id"], owner_user_id="42") is None
    batch_before = case.service.repo.get_batch(case.batch_id)
    recipes_before = case.service.repo.list_batch_recipes(case.batch_id)
    items_before = case.service.repo.list_items(case.pack_id)
    slots_before = case.service.repo.list_slots(case.pack_id)
    receipt_before = case.record()
    jobs_before = case.jobs.list_jobs(domain="vn_assets", owner_user_id="42")
    assert receipt_before["status"] == "in_progress"
    assert receipt_before["response_json"] == "{}"
    assert len(items_before) == 1 and items_before[0]["review_status"] == "approved"
    assert items_before[0]["preferred"]
    assert len(jobs_before) == 2
    scoped_reads: list[dict[str, str]] = []
    active_reads: list[tuple[int, str | None, dict[str, Any] | None]] = []

    def observe_scoped_lookup(
        *, idempotency_key: str, domain: str, queue: str, job_type: str, owner_user_id: str,
    ) -> dict[str, Any] | None:
        """Observe public recovery using the actual scoped archive-capable read."""
        request = {
            "idempotency_key": idempotency_key, "domain": domain, "queue": queue,
            "job_type": job_type, "owner_user_id": owner_user_id,
        }
        scoped_reads.append(request)
        result = native_lookup(**request)
        assert result == archived
        return result

    def observe_active_lookup(job_id: int, *, owner_user_id: str | None = None) -> dict[str, Any] | None:
        """Keep native active authority checks observable, without substituting a row."""
        result = native_get_job(job_id, owner_user_id=owner_user_id)
        active_reads.append((job_id, owner_user_id, result))
        return result

    def no_queue_mutation(*args: Any, **kwargs: Any) -> None:
        """Complete fanout cannot create or retry a parent to replace missing authority."""
        pytest.fail("archive-only full fanout attempted Jobs creation or retry admission")

    monkeypatch.setattr(case.jobs, "get_job_or_archived_by_idempotency_key", observe_scoped_lookup)
    monkeypatch.setattr(case.jobs, "get_job", observe_active_lookup)
    monkeypatch.setattr(case.jobs, "create_job", no_queue_mutation)
    monkeypatch.setattr(case.jobs, "retry_failed_job_admission", no_queue_mutation)
    for _replay in range(2):
        with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$") as error:
            case.replay()
        assert error.value.retryable is True
        assert case.record() == receipt_before
        assert case.service.repo.get_batch(case.batch_id) == batch_before
        assert case.service.repo.list_batch_recipes(case.batch_id) == recipes_before
        assert case.service.repo.list_items(case.pack_id) == items_before
        assert case.service.repo.list_slots(case.pack_id) == slots_before
        assert case.jobs.list_jobs(domain="vn_assets", owner_user_id="42") == jobs_before
        assert native_lookup(**scope) == archived
    assert scoped_reads and all(request == scope for request in scoped_reads)
    assert active_reads and all(read == (parent["id"], "42", None) for read in active_reads)


def test_active_full_fanout_parent_completes_public_receipt_without_queue_mutation(
    case: RecoveryCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native active completed parent still recovers an unsaved ID without altering history."""
    parent = terminal_full_fanout_parent(case, "completed")
    assert case.service.repo.get_batch(case.batch_id)["job_batch_id"] is None
    batch_before = case.service.repo.get_batch(case.batch_id)
    recipes_before = case.service.repo.list_batch_recipes(case.batch_id)
    items_before = case.service.repo.list_items(case.pack_id)
    slots_before = case.service.repo.list_slots(case.pack_id)
    jobs_before = case.jobs.list_jobs(domain="vn_assets", owner_user_id="42")

    def no_queue_mutation(*args: Any, **kwargs: Any) -> None:
        """An authoritative active parent needs neither creation nor retry after full fanout."""
        pytest.fail("active full fanout attempted Jobs creation or retry admission")

    monkeypatch.setattr(case.jobs, "create_job", no_queue_mutation)
    monkeypatch.setattr(case.jobs, "retry_failed_job_admission", no_queue_mutation)
    recovered = case.replay()
    assert recovered["batch_id"] == case.batch_id
    assert recovered["job_batch_id"] == str(parent["id"])
    assert recovered["enqueue_error"] is None
    completed_receipt = case.record()
    assert completed_receipt["status"] == "completed"
    assert case.replay() == recovered
    assert case.record() == completed_receipt
    batch_after = case.service.repo.get_batch(case.batch_id)
    assert batch_after is not None
    assert {field: value for field, value in batch_after.items() if field != "updated_at"} == {
        **{field: value for field, value in batch_before.items() if field != "updated_at"},
        "job_batch_id": str(parent["id"]), "enqueue_error": None,
    }
    assert case.service.repo.list_batch_recipes(case.batch_id) == recipes_before
    assert case.service.repo.list_items(case.pack_id) == items_before
    assert case.service.repo.list_slots(case.pack_id) == slots_before
    assert case.jobs.list_jobs(domain="vn_assets", owner_user_id="42") == jobs_before
    assert case.jobs.get_job(parent["id"], owner_user_id="42") == parent


def test_admin_cancelled_parent_is_not_resumed(case: RecoveryCase) -> None:
    """An explicit client retry cannot override deliberate admin cancellation."""
    parent = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": str(parent["id"])})
    assert case.jobs.cancel_job(parent["id"], reason="admin request")
    with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
        case.replay()
    assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "cancelled"
    assert case.record()["status"] == "in_progress"


def test_concurrent_failed_parent_recovery_admits_once(case: RecoveryCase) -> None:
    """Two receipt recoveries converge on one parent and immutable response."""
    parent = exhaust_parent(case)
    barrier = Barrier(2)

    def recover(_index: int) -> dict[str, Any]:
        """Use separate caller-owned connections for concurrent client requests."""
        db = CharactersRAGDB(str(case.service.repo.db.db_path), client_id=f"recovery-{_index}")
        service = VNAssetPackService(db, owner_user_id=42, jobs_manager=case.jobs)
        try:
            barrier.wait(timeout=10)
            result = service.claim_or_replay_idempotency(
                owner_user_id=42, **case.receipt, generation_pack_id=case.pack_id,
            )
            assert result is not None
            return result
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(recover, range(2)))
    assert responses[0] == responses[1]
    assert responses[0]["job_batch_id"] == str(parent["id"])
    assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "queued"
    assert len(case.jobs.list_jobs(domain="vn_assets", job_type="vn_asset_enqueue_batch")) == 1


def test_response_completed_after_claim_remains_immutable(
    case: RecoveryCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrently completed receipt must not be overwritten by stale recovery."""
    original = case.service.repo.claim_idempotency_record
    snapshot = {"acknowledged": True, "snapshot": "immutable"}

    def complete_after_claim(**kwargs: Any) -> tuple[dict[str, Any], bool]:
        """Commit the other caller's response after this caller observes pending."""
        record, claimed = original(**kwargs)
        case.service.complete_idempotency_response(owner_user_id=42, **case.receipt, response=snapshot)
        return record, claimed

    monkeypatch.setattr(case.service.repo, "claim_idempotency_record", complete_after_claim)
    assert case.replay() == snapshot


@pytest.mark.parametrize("health", ["live", "expired", "cancel_requested"])
def test_processing_parent_requires_live_uncancelled_jobs_lease(case: RecoveryCase, health: str) -> None:
    """A dead final attempt cannot permanently complete a partial-fanout receipt."""
    parent = create_enqueue_batch_job(case.jobs, pack_id=case.pack_id, batch_id=case.batch_id, user_id=42)
    case.service.repo.update_batch(case.batch_id, {"job_batch_id": str(parent["id"])})
    for attempt in range(int(parent["max_retries"]) + 1):
        acquired = case.jobs.acquire_next_job(
            domain="vn_assets", queue="default", worker_id="final-parent", lease_seconds=60, owner_user_id="42",
        )
        assert acquired is not None
        if attempt < int(parent["max_retries"]):
            assert case.jobs.fail_job(
                parent["id"], error="partial fanout", retryable=True, backoff_seconds=0,
                worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
            )
    if health != "live":
        conn = case.jobs._connect()
        try:
            with conn:
                if health == "expired":
                    conn.execute("UPDATE jobs SET leased_until='2000-01-01 00:00:00' WHERE id=?", (parent["id"],))
                else:
                    conn.execute("UPDATE jobs SET cancel_requested_at=CURRENT_TIMESTAMP WHERE id=?", (parent["id"],))
        finally:
            conn.close()
    processing = case.jobs.get_job(parent["id"], owner_user_id="42")
    assert processing["status"] == "processing" and processing["retry_count"] == processing["max_retries"]
    if health == "live":
        assert case.replay()["job_batch_id"] == str(parent["id"])
        assert case.record()["status"] == "completed"
    else:
        with pytest.raises(VNAssetGenerationError, match="^idempotency_key_in_progress$"):
            case.replay()
        assert case.record()["status"] == "in_progress"
        assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "processing"
        if health == "expired":
            assert case.jobs.acquire_next_job(
                domain="vn_assets", queue="default", worker_id="reconcile", lease_seconds=60, owner_user_id="42",
            ) is None
            assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "failed"
            assert case.replay()["job_batch_id"] == str(parent["id"])
            assert case.jobs.get_job(parent["id"], owner_user_id="42")["status"] == "queued"


def test_delayed_original_completion_cannot_overwrite_recovered_snapshot(case: RecoveryCase) -> None:
    """An original caller completing after recovery cannot replace the first response."""
    exhaust_parent(case)
    original_waiting, resume_original = Event(), Event()

    def complete_original() -> None:
        """Delay a real independent original caller at its completion boundary."""
        db = CharactersRAGDB(str(case.service.repo.db.db_path), client_id="delayed-original")
        service = VNAssetPackService(db, owner_user_id=42, jobs_manager=case.jobs)
        try:
            original_waiting.set()
            assert resume_original.wait(timeout=15)
            service.complete_idempotency_response(
                owner_user_id=42, **case.receipt, response={"status": "queued", "enqueue_error": "original stale view"},
            )
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        original = executor.submit(complete_original)
        try:
            assert original_waiting.wait(timeout=15)
            recovered = case.replay()
        finally:
            resume_original.set()
        original.result(timeout=15)
    assert case.replay() == recovered
