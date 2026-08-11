"""Direct SQLite single-job lease renewal and release operation tests."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    BatchRenewLeaseItem,
    BatchRenewLeasesCommand,
    BatchRenewLeasesResult,
    NoTransitionReason,
    OperationOutcome,
    ReleaseJobCommand,
    RenewLeaseCommand,
)
from tldw_Server_API.app.core.Jobs.operations.sqlite import renew_leases_batch
from tldw_Server_API.app.core.Jobs.operations.sqlite.lifecycle import (
    release_job,
    renew_lease,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
RENEW_RESULT_FIELDS = {
    "id",
    "leased_until",
    "progress_percent",
    "progress_message",
}
RELEASE_RESULT_FIELDS = {
    "id",
    "domain",
    "queue",
    "job_type",
    "status",
    "available_at",
    "leased_until",
    "worker_id",
    "lease_id",
    "acquired_at",
    "started_at",
    "completion_token",
    "updated_at",
}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return ensure_jobs_tables(tmp_path / "jobs.db")


@pytest.fixture()
def conn(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _insert_job(
    conn: sqlite3.Connection,
    *,
    uuid: str = "job-1",
    status: str = "processing",
    available_at: str | None = "2026-01-02 11:30:00",
    leased_until: str | None = "2026-01-02 11:45:00",
    worker_id: str | None = "worker-1",
    lease_id: str | None = "lease-1",
) -> int:
    cursor = conn.execute(
        (
            "INSERT INTO jobs("
            "uuid, domain, queue, job_type, owner_user_id, project_id, batch_group, payload, result, "
            "status, retry_count, available_at, started_at, leased_until, lease_id, worker_id, "
            "acquired_at, error_message, error_code, last_error, completion_token, progress_percent, "
            "progress_message, request_id, trace_id, created_at, updated_at"
            ") VALUES(?, 'lifecycle', 'default', 'work', 'owner-1', 17, 'batch-1', ?, ?, ?, 2, ?, "
            "'2026-01-02 11:00:00', ?, ?, ?, '2026-01-02 11:00:00', 'old error', 'old-code', "
            "'old failure', 'completion-1', 10.0, 'old progress', 'request-1', 'trace-1', "
            "'2026-01-01 00:00:00', '2001-01-01 00:00:00')"
        ),
        (
            uuid,
            '{"input": 1}',
            '{"partial": true}',
            status,
            available_at,
            leased_until,
            lease_id,
            worker_id,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _renew_command(
    job_id: int,
    *,
    enforce: bool = True,
    worker_id: str | None = "worker-1",
    lease_id: str | None = "lease-1",
    progress_percent: float | None = None,
    progress_message: str | None = None,
) -> RenewLeaseCommand:
    return RenewLeaseCommand(
        job_id=job_id,
        seconds=30,
        enforce=enforce,
        worker_id=worker_id,
        lease_id=lease_id,
        progress_percent=progress_percent,
        progress_message=progress_message,
    )


def _release_command(
    job_id: int,
    *,
    enforce: bool = True,
    worker_id: str | None = "worker-1",
    lease_id: str | None = "lease-1",
) -> ReleaseJobCommand:
    return ReleaseJobCommand(
        job_id=job_id,
        enforce=enforce,
        worker_id=worker_id,
        lease_id=lease_id,
        reason="yield",
    )


class RecordingClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return NOW


class FailOnSecondClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("forced clock failure")
        return NOW


def test_sqlite_batch_renew_counts_attempts_and_commits_expected_noops(
    conn: sqlite3.Connection,
) -> None:
    valid_id = _insert_job(conn, uuid="valid")
    queued_id = _insert_job(conn, uuid="queued", status="queued")
    stale_id = _insert_job(conn, uuid="stale")
    long_id = _insert_job(conn, uuid="long", leased_until="2026-01-02 13:00:00")
    command = BatchRenewLeasesCommand(
        items=(
            BatchRenewLeaseItem(valid_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(999_999, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(queued_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(stale_id, 30, "worker-2", "lease-1"),
            BatchRenewLeaseItem(valid_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(long_id, 30, "worker-1", "lease-1"),
        ),
        enforce=True,
    )
    clock = RecordingClock()

    result = renew_leases_batch(conn, command=command, clock=clock)

    assert result == BatchRenewLeasesResult(requested_count=6, applied_count=3)
    assert clock.calls == 6
    assert conn.execute("SELECT leased_until FROM jobs WHERE id = ?", (long_id,)).fetchone()[0] == "2026-01-02 13:00:00"


def test_sqlite_batch_renew_empty_command_skips_clock(
    conn: sqlite3.Connection,
) -> None:
    clock = RecordingClock()

    result = renew_leases_batch(
        conn,
        command=BatchRenewLeasesCommand(items=(), enforce=True),
        clock=clock,
    )

    assert result == BatchRenewLeasesResult(requested_count=0, applied_count=0)
    assert clock.calls == 0


def test_sqlite_batch_renew_leaves_caller_transaction_open(
    conn: sqlite3.Connection,
) -> None:
    job_id = _insert_job(conn, uuid="caller-transaction")
    conn.execute("BEGIN IMMEDIATE")

    result = renew_leases_batch(
        conn,
        command=BatchRenewLeasesCommand(
            items=(BatchRenewLeaseItem(job_id, 30, "worker-1", "lease-1"),),
            enforce=True,
        ),
        clock=RecordingClock(),
    )

    assert result == BatchRenewLeasesResult(requested_count=1, applied_count=1)
    assert conn.in_transaction is True
    conn.rollback()
    current = conn.execute(
        "SELECT leased_until FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert current[0] == "2026-01-02 11:45:00"


def test_sqlite_batch_renew_failure_rolls_back_savepoint_only(
    conn: sqlite3.Connection,
) -> None:
    first_job_id = _insert_job(conn, uuid="first")
    second_job_id = _insert_job(conn, uuid="second")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE jobs SET progress_message = 'caller-owned' WHERE id = ?",
        (second_job_id,),
    )
    command = BatchRenewLeasesCommand(
        items=(
            BatchRenewLeaseItem(first_job_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(second_job_id, 30, "worker-1", "lease-1"),
        ),
        enforce=True,
    )

    with pytest.raises(RuntimeError, match="forced clock failure"):
        renew_leases_batch(conn, command=command, clock=FailOnSecondClock())

    assert conn.in_transaction is True
    rows = conn.execute(
        "SELECT leased_until, progress_message FROM jobs WHERE id IN (?, ?) ORDER BY id",
        (first_job_id, second_job_id),
    ).fetchall()
    assert [row["leased_until"] for row in rows] == [
        "2026-01-02 11:45:00",
        "2026-01-02 11:45:00",
    ]
    assert rows[1]["progress_message"] == "caller-owned"
    conn.rollback()
    persisted = conn.execute(
        "SELECT progress_message FROM jobs WHERE id = ?",
        (second_job_id,),
    ).fetchone()
    assert persisted[0] == "old progress"


def test_sqlite_batch_renew_rolls_back_when_clock_fails(
    conn: sqlite3.Connection,
    db_path: Path,
) -> None:
    first_job_id = _insert_job(conn, uuid="first")
    second_job_id = _insert_job(conn, uuid="second")
    command = BatchRenewLeasesCommand(
        items=(
            BatchRenewLeaseItem(first_job_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(second_job_id, 30, "worker-1", "lease-1"),
        ),
        enforce=True,
    )

    with pytest.raises(RuntimeError, match="forced clock failure"):
        renew_leases_batch(conn, command=command, clock=FailOnSecondClock())

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        leases = connection.execute(
            "SELECT id, leased_until FROM jobs WHERE id IN (?, ?) ORDER BY id",
            (first_job_id, second_job_id),
        ).fetchall()
    finally:
        connection.close()
    assert [(row["id"], row["leased_until"]) for row in leases] == [
        (first_job_id, "2026-01-02 11:45:00"),
        (second_job_id, "2026-01-02 11:45:00"),
    ]


def test_sqlite_batch_renew_rolls_back_when_second_update_fails(
    conn: sqlite3.Connection,
    db_path: Path,
) -> None:
    first_job_id = _insert_job(conn, uuid="first")
    second_job_id = _insert_job(conn, uuid="second", worker_id="worker-2")
    conn.execute(
        "CREATE TRIGGER fail_batch_renewal BEFORE UPDATE ON jobs "
        "WHEN OLD.worker_id = 'worker-2' "
        "BEGIN SELECT RAISE(ABORT, 'forced batch renewal failure'); END"
    )
    conn.commit()
    command = BatchRenewLeasesCommand(
        items=(
            BatchRenewLeaseItem(first_job_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(second_job_id, 30, "worker-2", "lease-1"),
        ),
        enforce=True,
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced batch renewal failure"):
        renew_leases_batch(conn, command=command, clock=RecordingClock())

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        leases = connection.execute(
            "SELECT id, leased_until FROM jobs WHERE id IN (?, ?) ORDER BY id",
            (first_job_id, second_job_id),
        ).fetchall()
    finally:
        connection.close()
    assert [(row["id"], row["leased_until"]) for row in leases] == [
        (first_job_id, "2026-01-02 11:45:00"),
        (second_job_id, "2026-01-02 11:45:00"),
    ]


@pytest.mark.parametrize(
    ("operation", "command"),
    [
        (renew_lease, RenewLeaseCommand(job_id=999, seconds=30, enforce=False)),
        (release_job, ReleaseJobCommand(job_id=999, enforce=False)),
    ],
    ids=["renew", "release"],
)
def test_sqlite_lifecycle_reports_missing_job(
    conn: sqlite3.Connection,
    operation: Any,
    command: RenewLeaseCommand | ReleaseJobCommand,
) -> None:
    if isinstance(command, RenewLeaseCommand):
        result = operation(conn, command=command, now=NOW)
    else:
        result = operation(conn, command=command, counters_enabled=False)

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.MISSING


@pytest.mark.parametrize("operation", [renew_lease, release_job], ids=["renew", "release"])
def test_sqlite_lifecycle_reports_wrong_status(
    conn: sqlite3.Connection,
    operation: Any,
) -> None:
    job_id = _insert_job(conn, status="queued")

    if operation is renew_lease:
        result = operation(conn, command=_renew_command(job_id), now=NOW)
    else:
        result = operation(conn, command=_release_command(job_id), counters_enabled=False)

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.WRONG_STATUS


@pytest.mark.parametrize(
    ("operation", "worker_id", "lease_id"),
    [
        (renew_lease, "worker-2", "lease-1"),
        (renew_lease, "worker-1", "lease-2"),
        (release_job, "worker-2", "lease-1"),
        (release_job, "worker-1", "lease-2"),
    ],
    ids=["renew-worker", "renew-lease", "release-worker", "release-lease"],
)
def test_sqlite_lifecycle_reports_stale_enforced_identity(
    conn: sqlite3.Connection,
    operation: Any,
    worker_id: str,
    lease_id: str,
) -> None:
    job_id = _insert_job(conn)

    if operation is renew_lease:
        result = operation(
            conn,
            command=_renew_command(job_id, worker_id=worker_id, lease_id=lease_id),
            now=NOW,
        )
    else:
        result = operation(
            conn,
            command=_release_command(job_id, worker_id=worker_id, lease_id=lease_id),
            counters_enabled=False,
        )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.STALE_LEASE
    current = conn.execute("SELECT status, lease_id, worker_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert tuple(current) == ("processing", "lease-1", "worker-1")


@pytest.mark.parametrize("operation", [renew_lease, release_job], ids=["renew", "release"])
def test_sqlite_lifecycle_ignores_supplied_identity_without_enforcement(
    conn: sqlite3.Connection,
    operation: Any,
) -> None:
    job_id = _insert_job(conn)

    if operation is renew_lease:
        result = operation(
            conn,
            command=_renew_command(
                job_id,
                enforce=False,
                worker_id="stale-worker",
                lease_id="stale-lease",
            ),
            now=NOW,
        )
    else:
        result = operation(
            conn,
            command=_release_command(
                job_id,
                enforce=False,
                worker_id="stale-worker",
                lease_id="stale-lease",
            ),
            counters_enabled=False,
        )

    assert result.outcome is OperationOutcome.APPLIED


def test_sqlite_renew_preserves_longer_current_lease_and_returns_transition_facts(
    conn: sqlite3.Connection,
) -> None:
    job_id = _insert_job(conn, leased_until="2026-01-02 13:00:00")

    result = renew_lease(conn, command=_renew_command(job_id), now=NOW)

    assert result.outcome is OperationOutcome.APPLIED
    assert result.row is not None
    assert set(result.row) == RENEW_RESULT_FIELDS
    assert result.row["leased_until"] == "2026-01-02 13:00:00"
    persisted = dict(
        conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    )
    assert persisted["uuid"] == "job-1"
    assert persisted["payload"] == '{"input": 1}'
    assert persisted["owner_user_id"] == "owner-1"


def test_sqlite_renew_joins_caller_owned_transaction(
    conn: sqlite3.Connection,
) -> None:
    """Leave a caller-owned transaction open after renewing a lease."""

    job_id = _insert_job(conn)
    conn.execute("BEGIN IMMEDIATE")

    result = renew_lease(
        conn,
        command=_renew_command(job_id, progress_message="caller transaction"),
        now=NOW,
    )

    assert result.outcome is OperationOutcome.APPLIED
    assert conn.in_transaction is True
    conn.rollback()
    current = conn.execute(
        "SELECT progress_message FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert current[0] == "old progress"


@pytest.mark.parametrize("enforce", [False, True], ids=["unenforced", "enforced"])
@pytest.mark.parametrize("progress_percent", [None, 55.5], ids=["percent-unchanged", "percent-set"])
@pytest.mark.parametrize("progress_message", [None, "halfway"], ids=["message-unchanged", "message-set"])
def test_sqlite_renew_supports_every_progress_and_enforcement_variant(
    conn: sqlite3.Connection,
    enforce: bool,
    progress_percent: float | None,
    progress_message: str | None,
) -> None:
    job_id = _insert_job(conn)

    result = renew_lease(
        conn,
        command=_renew_command(
            job_id,
            enforce=enforce,
            progress_percent=progress_percent,
            progress_message=progress_message,
        ),
        now=NOW,
    )

    assert result.row is not None
    assert set(result.row) == RENEW_RESULT_FIELDS
    assert result.row["leased_until"] == "2026-01-02 12:00:30"
    assert result.row["progress_percent"] == (10.0 if progress_percent is None else 55.5)
    assert result.row["progress_message"] == (
        "old progress" if progress_message is None else "halfway"
    )


def test_sqlite_release_clears_lease_fields_and_preserves_unrelated_facts(
    conn: sqlite3.Connection,
) -> None:
    job_id = _insert_job(conn)

    result = release_job(
        conn,
        command=_release_command(job_id),
        counters_enabled=False,
    )

    assert result.outcome is OperationOutcome.APPLIED
    assert result.row is not None
    assert set(result.row) == RELEASE_RESULT_FIELDS
    assert result.row["status"] == "queued"
    for field in (
        "available_at",
        "leased_until",
        "worker_id",
        "lease_id",
        "acquired_at",
        "started_at",
        "completion_token",
    ):
        assert result.row[field] is None
    persisted = dict(
        conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    )
    assert persisted["uuid"] == "job-1"
    assert persisted["payload"] == '{"input": 1}'
    assert persisted["result"] == '{"partial": true}'
    assert persisted["owner_user_id"] == "owner-1"
    assert persisted["project_id"] == 17
    assert persisted["batch_group"] == "batch-1"
    assert persisted["retry_count"] == 2
    assert persisted["progress_percent"] == 10.0
    assert persisted["progress_message"] == "old progress"
    assert persisted["request_id"] == "request-1"
    assert persisted["trace_id"] == "trace-1"
    assert persisted["error_message"] == "old error"
    assert persisted["error_code"] == "old-code"
    assert persisted["last_error"] == "old failure"
    assert result.row["updated_at"] != "2001-01-01 00:00:00"


@pytest.mark.parametrize(
    ("processing_count", "expected_processing"),
    [(2, 1), (0, 0)],
    ids=["decrement", "floor-zero"],
)
def test_sqlite_release_moves_processing_counter_to_ready(
    conn: sqlite3.Connection,
    processing_count: int,
    expected_processing: int,
) -> None:
    job_id = _insert_job(conn)
    conn.execute(
        (
            "INSERT INTO job_counters(domain, queue, job_type, ready_count, scheduled_count, "
            "processing_count, quarantined_count) VALUES('lifecycle', 'default', 'work', 4, 3, ?, 0)"
        ),
        (processing_count,),
    )
    conn.commit()

    release_job(
        conn,
        command=_release_command(job_id),
        counters_enabled=True,
    )

    counter = conn.execute(
        "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
        "WHERE domain='lifecycle' AND queue='default' AND job_type='work'"
    ).fetchone()
    assert tuple(counter) == (5, 3, expected_processing)


def test_sqlite_release_creates_missing_counter_as_one_ready(
    conn: sqlite3.Connection,
) -> None:
    job_id = _insert_job(conn)

    release_job(
        conn,
        command=_release_command(job_id),
        counters_enabled=True,
    )

    counter = conn.execute(
        "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
        "WHERE domain='lifecycle' AND queue='default' AND job_type='work'"
    ).fetchone()
    assert tuple(counter) == (1, 0, 0)


def test_sqlite_release_counter_failure_rolls_back_transition(
    conn: sqlite3.Connection,
) -> None:
    job_id = _insert_job(conn)
    conn.execute(
        "CREATE TRIGGER fail_release_counter BEFORE INSERT ON job_counters "
        "BEGIN SELECT RAISE(ABORT, 'forced counter failure'); END"
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="forced counter failure"):
        release_job(
            conn,
            command=_release_command(job_id),
            counters_enabled=True,
        )

    current = conn.execute(
        "SELECT status, available_at, leased_until, worker_id, lease_id, completion_token "
        "FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert tuple(current) == (
        "processing",
        "2026-01-02 11:30:00",
        "2026-01-02 11:45:00",
        "worker-1",
        "lease-1",
        "completion-1",
    )


def test_sqlite_release_joins_caller_owned_transaction(
    conn: sqlite3.Connection,
) -> None:
    """Leave a caller-owned transaction open after releasing a job."""

    job_id = _insert_job(conn)
    conn.execute("BEGIN IMMEDIATE")

    result = release_job(
        conn,
        command=_release_command(job_id),
        counters_enabled=False,
    )

    assert result.outcome is OperationOutcome.APPLIED
    assert conn.in_transaction is True
    conn.rollback()
    current = conn.execute(
        "SELECT status, worker_id, lease_id FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert tuple(current) == ("processing", "worker-1", "lease-1")


class _PauseAfterValidationConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        selected: threading.Event,
        resume: threading.Event,
    ) -> None:
        self._connection = connection
        self._selected = selected
        self._resume = resume
        self.validation_was_locked = False
        self._paused = False

    def __enter__(self) -> _PauseAfterValidationConnection:
        self._connection.__enter__()
        return self

    def __exit__(self, *args: Any) -> bool | None:
        return self._connection.__exit__(*args)

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        cursor = self._connection.execute(sql, parameters)
        if not self._paused and "FROM jobs WHERE id = ?" in sql:
            self._paused = True
            self.validation_was_locked = self._connection.in_transaction
            self._selected.set()
            if not self._resume.wait(timeout=5):
                raise TimeoutError("competing writer did not attempt reacquisition")
        return cursor


def test_sqlite_release_locks_before_validation_against_competing_reacquisition(
    tmp_path: Path,
) -> None:
    db_path = ensure_jobs_tables(tmp_path / "locking.db")
    setup = sqlite3.connect(db_path)
    setup.row_factory = sqlite3.Row
    job_id = _insert_job(setup)
    setup.close()

    selected = threading.Event()
    resume = threading.Event()
    released: dict[str, Any] = {}

    def run_release() -> None:
        release_conn = sqlite3.connect(db_path, timeout=1)
        release_conn.row_factory = sqlite3.Row
        proxy = _PauseAfterValidationConnection(release_conn, selected, resume)
        try:
            released["result"] = release_job(
                proxy,
                command=_release_command(job_id, enforce=False),
                counters_enabled=False,
            )
            released["validation_was_locked"] = proxy.validation_was_locked
        except (sqlite3.Error, RuntimeError, TimeoutError) as exc:  # pragma: no cover - asserted below
            released["error"] = exc
        finally:
            release_conn.close()

    release_thread = threading.Thread(target=run_release)
    release_thread.start()
    assert selected.wait(timeout=5)

    writer = sqlite3.connect(db_path, timeout=0.05)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with writer:
                writer.execute(
                    "UPDATE jobs SET worker_id = 'worker-2', lease_id = 'lease-2' WHERE id = ?",
                    (job_id,),
                )
    finally:
        writer.close()
        resume.set()
        release_thread.join(timeout=5)

    assert not release_thread.is_alive()
    assert "error" not in released
    assert released["validation_was_locked"] is True
    assert released["result"].outcome is OperationOutcome.APPLIED
    assert released["result"].row["worker_id"] is None
    assert released["result"].row["lease_id"] is None
