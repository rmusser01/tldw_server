"""SQLite contract tests for exact terminal operation-result patching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import SLIDES_ARCHIVE_EXACT_FIELDS
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    TerminalOperationResultPatchCommand,
    TerminalOperationResultPatchOutcome,
)


def _fingerprint(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture
def terminal_manager(tmp_path, monkeypatch) -> JobManager:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    return JobManager(tmp_path / "terminal-result.db")


def _create_terminal_job(
    manager: JobManager,
    *,
    result: dict[str, object] | None = None,
) -> dict[str, object]:
    stored_result = result or {"schema_version": 1, "cleanup_state": "pending"}
    job = manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        payload={"schema_version": 1},
        owner_user_id="recipient-7",
        batch_group="share:7",
        max_retries=0,
    )
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'completed', result = ?, completed_at = DATETIME('now'),
                updated_at = '2000-01-01 00:00:00'
            WHERE uuid = ?
            """,
            (json.dumps(stored_result), job["uuid"]),
        )
    return {**job, "result": stored_result, "status": "completed"}


def _command(
    job: dict[str, object],
    *,
    current: dict[str, object] | None = None,
    replacement: dict[str, object] | None = None,
    **overrides: object,
) -> TerminalOperationResultPatchCommand:
    current_result = (
        current
        if current is not None
        else {"schema_version": 1, "cleanup_state": "pending"}
    )
    values: dict[str, object] = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "recipient-7",
        "domain": "sharing",
        "queue": "workspace-clone",
        "job_type": "workspace_clone",
        "operation_scope": "share:7",
        "allowed_statuses": ("completed",),
        "expected_result_fingerprint": _fingerprint(current_result),
        "replacement_result": replacement
        or {"schema_version": 1, "cleanup_state": "complete"},
    }
    values.update(overrides)
    return TerminalOperationResultPatchCommand(**values)


def _archive_job(
    manager: JobManager,
    job_uuid: str,
    *,
    retain_active: bool = False,
) -> None:
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            f"INSERT INTO jobs_archive ({projection}) "  # nosec B608
            f"SELECT {projection} FROM jobs WHERE uuid = ?",  # nosec B608
            (job_uuid,),
        )
        if not retain_active:
            conn.execute("DELETE FROM jobs WHERE uuid = ?", (job_uuid,))


def _stored_result(manager: JobManager, table: str, job_uuid: str) -> tuple[dict, str]:
    with sqlite3.connect(manager.db_path) as conn:
        row = conn.execute(
            f"SELECT result, updated_at FROM {table} WHERE uuid = ?",  # nosec B608
            (job_uuid,),
        ).fetchone()
    assert row is not None
    return json.loads(row[0]), str(row[1])


def test_active_terminal_result_patch_is_exact_and_advances_updated_at(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)

    outcome = terminal_manager.patch_terminal_operation_result(_command(job))

    result, updated_at = _stored_result(terminal_manager, "jobs", str(job["uuid"]))
    assert outcome is TerminalOperationResultPatchOutcome.APPLIED
    assert result == {"schema_version": 1, "cleanup_state": "complete"}
    assert updated_at != "2000-01-01 00:00:00"


def test_failed_job_without_result_can_record_proven_cleanup(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)
    with sqlite3.connect(terminal_manager.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', result = NULL WHERE uuid = ?",
            (job["uuid"],),
        )

    outcome = terminal_manager.patch_terminal_operation_result(
        _command(
            job,
            current={},
            allowed_statuses=("failed",),
        )
    )

    result, _updated_at = _stored_result(
        terminal_manager,
        "jobs",
        str(job["uuid"]),
    )
    assert outcome is TerminalOperationResultPatchOutcome.APPLIED
    assert result == {"schema_version": 1, "cleanup_state": "complete"}


def test_archived_terminal_result_patch_clears_compressed_authority(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)
    _archive_job(terminal_manager, str(job["uuid"]))
    with sqlite3.connect(terminal_manager.db_path) as conn:
        conn.execute(
            "UPDATE jobs_archive SET result_compressed = 'stale' WHERE uuid = ?",
            (job["uuid"],),
        )

    outcome = terminal_manager.patch_terminal_operation_result(_command(job))

    result, _updated_at = _stored_result(
        terminal_manager,
        "jobs_archive",
        str(job["uuid"]),
    )
    with sqlite3.connect(terminal_manager.db_path) as conn:
        compressed = conn.execute(
            "SELECT result_compressed FROM jobs_archive WHERE uuid = ?",
            (job["uuid"],),
        ).fetchone()[0]
    assert outcome is TerminalOperationResultPatchOutcome.APPLIED
    assert result["cleanup_state"] == "complete"
    assert compressed is None


def test_terminal_result_patch_replay_is_idempotent(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)
    command = _command(job)

    assert (
        terminal_manager.patch_terminal_operation_result(command)
        is TerminalOperationResultPatchOutcome.APPLIED
    )
    assert (
        terminal_manager.patch_terminal_operation_result(command)
        is TerminalOperationResultPatchOutcome.IDEMPOTENT
    )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("owner_user_id", "recipient-8"),
        ("domain", "other"),
        ("queue", "default"),
        ("job_type", "other"),
        ("operation_scope", "share:8"),
        ("allowed_statuses", ("failed",)),
        ("expected_result_fingerprint", "f" * 64),
    ],
)
def test_terminal_result_patch_rejects_wrong_correlation_without_mutation(
    terminal_manager: JobManager,
    override: str,
    value: object,
) -> None:
    job = _create_terminal_job(terminal_manager)

    outcome = terminal_manager.patch_terminal_operation_result(
        _command(job, **{override: value})
    )

    result, _updated_at = _stored_result(terminal_manager, "jobs", str(job["uuid"]))
    assert outcome is TerminalOperationResultPatchOutcome.CONFLICT
    assert result["cleanup_state"] == "pending"


def test_terminal_result_patch_distinguishes_missing_uuid(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)

    outcome = terminal_manager.patch_terminal_operation_result(
        _command(job, job_uuid="00000000-0000-0000-0000-000000000000")
    )

    assert outcome is TerminalOperationResultPatchOutcome.MISSING


def test_terminal_result_patch_rejects_duplicate_active_archive_authority(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)
    _archive_job(terminal_manager, str(job["uuid"]), retain_active=True)

    outcome = terminal_manager.patch_terminal_operation_result(_command(job))

    assert outcome is TerminalOperationResultPatchOutcome.CONFLICT


def test_terminal_result_patch_has_one_concurrent_winner(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)
    commands = (
        _command(job, replacement={"schema_version": 1, "cleanup_state": "complete"}),
        _command(job, replacement={"schema_version": 1, "cleanup_state": "failed"}),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(terminal_manager.patch_terminal_operation_result, commands))

    assert sorted(outcome.value for outcome in outcomes) == ["applied", "conflict"]


def test_terminal_result_patch_enforces_json_size_before_mutation(
    terminal_manager: JobManager,
    monkeypatch,
) -> None:
    job = _create_terminal_job(terminal_manager)
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "64")

    with pytest.raises(ValueError, match="too large"):
        terminal_manager.patch_terminal_operation_result(
            _command(job, replacement={"value": "x" * 128})
        )

    result, _updated_at = _stored_result(terminal_manager, "jobs", str(job["uuid"]))
    assert result["cleanup_state"] == "pending"


def test_terminal_result_patch_command_defensively_copies_result(
    terminal_manager: JobManager,
) -> None:
    job = _create_terminal_job(terminal_manager)
    replacement = {"schema_version": 1, "cleanup_state": "complete"}
    command = _command(job, replacement=replacement)

    replacement["cleanup_state"] = "tampered"

    assert command.replacement_result["cleanup_state"] == "complete"


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_uuid": ""},
        {"allowed_statuses": ()},
        {"allowed_statuses": ("processing",)},
        {"expected_result_fingerprint": "A" * 64},
        {"replacement_result": {"percent": float("nan")}},
    ],
)
def test_terminal_result_patch_command_rejects_malformed_correlation(
    terminal_manager: JobManager,
    overrides: dict[str, object],
) -> None:
    job = _create_terminal_job(terminal_manager)

    with pytest.raises(ValueError):
        _command(job, **overrides)
