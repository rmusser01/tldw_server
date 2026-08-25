"""PostgreSQL parity coverage for durable Jobs idempotency receipts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import SLIDES_ARCHIVE_EXACT_FIELDS
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    CreateJobCommand,
    IdempotentOperationCommand,
    IdempotentOperationConflict,
    IdempotentOperationConflictReason,
    IdempotentOperationDisposition,
    IdempotentOperationUnavailableError,
)

pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]


def _operation_command(
    *,
    owner_user_id: str = "recipient-1",
    key_digest: str = "a" * 64,
    request_fingerprint: str = "b" * 64,
    operation_scope: str = "share:share-1",
) -> IdempotentOperationCommand:
    return IdempotentOperationCommand(
        job=CreateJobCommand(
            domain="sharing",
            queue="workspace-clone",
            job_type="workspace_clone",
            payload={"schema_version": 1},
            owner_user_id=owner_user_id,
            batch_group=operation_scope,
            priority=5,
            max_retries=0,
        ),
        key_digest=key_digest,
        request_fingerprint=request_fingerprint,
        operation_scope=operation_scope,
        receipt_expires_at=datetime.now(timezone.utc) + timedelta(days=31),
    )


@pytest.fixture
def receipt_manager(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


def _counts(jobs_pg_dsn) -> tuple[int, int, int]:
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs")
        jobs = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM job_idempotency_receipts")
        receipts = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM job_events")
        events = int(cur.fetchone()[0])
    return jobs, receipts, events


def _archive_job(
    jobs_pg_dsn: str,
    job_uuid: str,
    *,
    retain_active: bool = False,
) -> None:
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO jobs_archive ({projection}) "  # nosec B608
            f"SELECT {projection} FROM jobs WHERE uuid = %s",  # nosec B608
            (job_uuid,),
        )
        if not retain_active:
            cur.execute("DELETE FROM jobs WHERE uuid = %s", (job_uuid,))


def test_postgres_first_request_and_exact_replay(receipt_manager, jobs_pg_dsn):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    replay = receipt_manager.admit_idempotent_operation(_operation_command())

    assert first.disposition is IdempotentOperationDisposition.CREATED
    assert replay.disposition is IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == first.job["uuid"]
    assert _counts(jobs_pg_dsn) == (1, 1, 1)


def test_postgres_uuid_lookup_and_replay_survive_archival(
    receipt_manager,
    jobs_pg_dsn,
):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    active = receipt_manager.get_job_or_archived_by_uuid(
        first.job["uuid"],
        domain="sharing",
        owner_user_id="recipient-1",
    )
    _archive_job(jobs_pg_dsn, first.job["uuid"])

    archived = receipt_manager.get_job_or_archived_by_uuid(
        first.job["uuid"],
        domain="sharing",
        owner_user_id="recipient-1",
    )
    replay = receipt_manager.admit_idempotent_operation(_operation_command())

    assert active is not None
    assert archived is not None
    assert active["archived"] is False
    assert archived["archived"] is True
    assert {key: value for key, value in active.items() if key != "archived"} == {
        key: value for key, value in archived.items() if key != "archived"
    }
    assert active["payload"] == archived["payload"] == {"schema_version": 1}
    assert replay.disposition is IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == first.job["uuid"]
    assert replay.job["archived"] is True


def test_postgres_uuid_lookup_rejects_duplicate_authority(
    receipt_manager,
    jobs_pg_dsn,
):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    _archive_job(jobs_pg_dsn, first.job["uuid"], retain_active=True)

    with pytest.raises(IdempotentOperationUnavailableError, match="exactly one Job"):
        receipt_manager.get_job_or_archived_by_uuid(first.job["uuid"])


def test_postgres_corrupt_receipt_correlation_fails_closed(
    receipt_manager,
    jobs_pg_dsn,
):
    receipt_manager.admit_idempotent_operation(_operation_command())
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE job_idempotency_receipts SET job_id = %s",
            (999_999,),
        )

    with pytest.raises(IdempotentOperationUnavailableError):
        receipt_manager.admit_idempotent_operation(_operation_command())


def test_postgres_same_key_with_different_fingerprint_conflicts(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())

    with pytest.raises(IdempotentOperationConflict) as exc_info:
        receipt_manager.admit_idempotent_operation(
            _operation_command(request_fingerprint="c" * 64)
        )

    assert exc_info.value.reason is IdempotentOperationConflictReason.KEY_REUSED
    assert exc_info.value.job_uuid == first.job["uuid"]


def test_postgres_second_key_converges_or_conflicts_by_fingerprint(
    receipt_manager,
    jobs_pg_dsn,
):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    converged = receipt_manager.admit_idempotent_operation(
        _operation_command(key_digest="d" * 64)
    )

    assert converged.disposition is IdempotentOperationDisposition.CONVERGED
    assert converged.job["uuid"] == first.job["uuid"]
    assert _counts(jobs_pg_dsn) == (1, 2, 1)

    with pytest.raises(IdempotentOperationConflict) as exc_info:
        receipt_manager.admit_idempotent_operation(
            _operation_command(
                key_digest="e" * 64,
                request_fingerprint="c" * 64,
            )
        )
    assert exc_info.value.reason is IdempotentOperationConflictReason.SCOPE_ACTIVE
    assert exc_info.value.job_uuid == first.job["uuid"]


def test_postgres_receipts_are_owner_isolated(receipt_manager, jobs_pg_dsn):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    second = receipt_manager.admit_idempotent_operation(
        _operation_command(owner_user_id="recipient-2")
    )

    assert first.job["uuid"] != second.job["uuid"]
    assert _counts(jobs_pg_dsn) == (2, 2, 2)


def test_postgres_receipt_failure_rolls_back_job(
    receipt_manager,
    jobs_pg_dsn,
    monkeypatch,
):
    from tldw_Server_API.app.core.Jobs.operations.postgres import idempotency

    def _fail_receipt_insert(*_args, **_kwargs):
        raise psycopg.IntegrityError("forced receipt failure")

    monkeypatch.setattr(idempotency, "_insert_receipt", _fail_receipt_insert)

    with pytest.raises(psycopg.IntegrityError, match="forced receipt failure"):
        receipt_manager.admit_idempotent_operation(_operation_command())

    assert _counts(jobs_pg_dsn) == (0, 0, 0)


def test_postgres_concurrent_keys_converge_without_deadlock(
    receipt_manager,
    jobs_pg_dsn,
):
    managers = [
        JobManager(None, backend="postgres", db_url=jobs_pg_dsn) for _ in range(8)
    ]
    barrier = Barrier(len(managers))

    def _admit(index: int):
        barrier.wait(timeout=10)
        return managers[index].admit_idempotent_operation(
            _operation_command(key_digest=("a" if index % 2 == 0 else "d") * 64)
        )

    with ThreadPoolExecutor(max_workers=len(managers)) as executor:
        results = list(executor.map(_admit, range(len(managers))))

    assert len({result.job["uuid"] for result in results}) == 1
    assert _counts(jobs_pg_dsn) == (1, 2, 1)


def test_postgres_concurrent_same_key_across_scopes_has_one_authority(
    receipt_manager,
    jobs_pg_dsn,
):
    managers = [
        JobManager(None, backend="postgres", db_url=jobs_pg_dsn) for _ in range(2)
    ]
    barrier = Barrier(len(managers))

    def _admit(index: int):
        barrier.wait(timeout=10)
        try:
            return managers[index].admit_idempotent_operation(
                _operation_command(operation_scope=f"share:share-{index + 1}")
            )
        except IdempotentOperationConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(managers)) as executor:
        outcomes = list(executor.map(_admit, range(len(managers))))

    admissions = [
        outcome for outcome in outcomes if not isinstance(outcome, Exception)
    ]
    conflicts = [
        outcome for outcome in outcomes if isinstance(outcome, IdempotentOperationConflict)
    ]
    assert len(admissions) == 1
    assert admissions[0].disposition is IdempotentOperationDisposition.CREATED
    assert len(conflicts) == 1
    assert conflicts[0].reason is IdempotentOperationConflictReason.KEY_REUSED
    assert conflicts[0].job_uuid == admissions[0].job["uuid"]
    assert _counts(jobs_pg_dsn) == (1, 1, 1)
