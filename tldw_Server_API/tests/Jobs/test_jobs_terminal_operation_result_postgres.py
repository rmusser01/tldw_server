"""PostgreSQL parity tests for exact terminal operation-result patching."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import SLIDES_ARCHIVE_EXACT_FIELDS
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    TerminalOperationResultPatchCommand,
    TerminalOperationResultPatchOutcome,
)

pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]


def _fingerprint(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture
def terminal_manager(jobs_pg_dsn, monkeypatch) -> JobManager:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


def _create_terminal_job(
    manager: JobManager,
    jobs_pg_dsn: str,
) -> dict[str, object]:
    result = {"schema_version": 1, "cleanup_state": "pending"}
    job = manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        payload={"schema_version": 1},
        owner_user_id="recipient-7",
        batch_group="share:7",
        max_retries=0,
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status = 'completed', result = %s::jsonb,
                completed_at = NOW(), updated_at = '2000-01-01T00:00:00Z'
            WHERE uuid = %s
            """,
            (json.dumps(result), job["uuid"]),
        )
    return {**job, "result": result, "status": "completed"}


def _command(
    job: dict[str, object],
    *,
    replacement: dict[str, object] | None = None,
) -> TerminalOperationResultPatchCommand:
    current = {"schema_version": 1, "cleanup_state": "pending"}
    return TerminalOperationResultPatchCommand(
        job_uuid=str(job["uuid"]),
        owner_user_id="recipient-7",
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        operation_scope="share:7",
        allowed_statuses=("completed",),
        expected_result_fingerprint=_fingerprint(current),
        replacement_result=replacement
        or {"schema_version": 1, "cleanup_state": "complete"},
    )


def _archive_job(jobs_pg_dsn: str, job_uuid: str, *, retain_active: bool = False) -> None:
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO jobs_archive ({projection}) "  # nosec B608
            f"SELECT {projection} FROM jobs WHERE uuid = %s",  # nosec B608
            (job_uuid,),
        )
        if not retain_active:
            cur.execute("DELETE FROM jobs WHERE uuid = %s", (job_uuid,))


def _stored_result(jobs_pg_dsn: str, table: str, job_uuid: str) -> dict:
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT result FROM {table} WHERE uuid = %s",  # nosec B608
            (job_uuid,),
        )
        row = cur.fetchone()
    assert row is not None
    value = row[0]
    return value if isinstance(value, dict) else json.loads(value)


def test_postgres_active_and_archived_terminal_patches(
    terminal_manager: JobManager,
    jobs_pg_dsn: str,
) -> None:
    active = _create_terminal_job(terminal_manager, jobs_pg_dsn)
    archived = _create_terminal_job(terminal_manager, jobs_pg_dsn)
    _archive_job(jobs_pg_dsn, str(archived["uuid"]))

    active_outcome = terminal_manager.patch_terminal_operation_result(_command(active))
    archived_outcome = terminal_manager.patch_terminal_operation_result(_command(archived))

    assert active_outcome is TerminalOperationResultPatchOutcome.APPLIED
    assert archived_outcome is TerminalOperationResultPatchOutcome.APPLIED
    assert _stored_result(jobs_pg_dsn, "jobs", str(active["uuid"]))["cleanup_state"] == "complete"
    assert _stored_result(jobs_pg_dsn, "jobs_archive", str(archived["uuid"]))["cleanup_state"] == "complete"


def test_postgres_failed_job_without_result_can_record_proven_cleanup(
    terminal_manager: JobManager,
    jobs_pg_dsn: str,
) -> None:
    job = _create_terminal_job(terminal_manager, jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'failed', result = NULL WHERE uuid = %s",
            (job["uuid"],),
        )
    command = TerminalOperationResultPatchCommand(
        job_uuid=str(job["uuid"]),
        owner_user_id="recipient-7",
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        operation_scope="share:7",
        allowed_statuses=("failed",),
        expected_result_fingerprint=_fingerprint({}),
        replacement_result={"schema_version": 1, "cleanup_state": "complete"},
    )

    outcome = terminal_manager.patch_terminal_operation_result(command)

    assert outcome is TerminalOperationResultPatchOutcome.APPLIED
    assert _stored_result(jobs_pg_dsn, "jobs", str(job["uuid"])) == {
        "schema_version": 1,
        "cleanup_state": "complete",
    }


def test_postgres_duplicate_authority_fails_closed(
    terminal_manager: JobManager,
    jobs_pg_dsn: str,
) -> None:
    job = _create_terminal_job(terminal_manager, jobs_pg_dsn)
    _archive_job(jobs_pg_dsn, str(job["uuid"]), retain_active=True)

    outcome = terminal_manager.patch_terminal_operation_result(_command(job))

    assert outcome is TerminalOperationResultPatchOutcome.CONFLICT


def test_postgres_terminal_patch_has_one_concurrent_winner(
    terminal_manager: JobManager,
    jobs_pg_dsn: str,
) -> None:
    job = _create_terminal_job(terminal_manager, jobs_pg_dsn)
    commands = (
        _command(job, replacement={"schema_version": 1, "cleanup_state": "complete"}),
        _command(job, replacement={"schema_version": 1, "cleanup_state": "failed"}),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(terminal_manager.patch_terminal_operation_result, commands))

    assert sorted(outcome.value for outcome in outcomes) == ["applied", "conflict"]
