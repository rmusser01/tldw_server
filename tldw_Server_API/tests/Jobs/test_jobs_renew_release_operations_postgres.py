"""Direct Postgres single-job lease renewal and release operation tests."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    BatchRenewLeaseItem,
    BatchRenewLeasesCommand,
    BatchRenewLeasesResult,
    NoTransitionReason,
    OperationOutcome,
    ReleaseJobCommand,
    RenewLeaseCommand,
)
from tldw_Server_API.app.core.Jobs.operations.postgres import renew_leases_batch
from tldw_Server_API.app.core.Jobs.operations.postgres.lifecycle import (
    release_job,
    renew_lease,
)

psycopg = pytest.importorskip("psycopg")
pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]

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
def manager(jobs_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> JobManager:
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "false")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


@pytest.fixture()
def conn(manager: JobManager) -> Iterator[Any]:
    connection = manager._connect()
    try:
        yield connection
    finally:
        connection.close()


def _execute(manager: JobManager, sql: str, params: tuple[Any, ...] = ()) -> None:
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cur:
            cur.execute(sql, params)
    finally:
        connection.close()


def _fetch_job(manager: JobManager, job_id: int) -> dict[str, Any]:
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        connection.close()
    assert row is not None
    return dict(row)


def _insert_job(
    manager: JobManager,
    *,
    status: str = "processing",
    available_at: datetime | None = NOW - timedelta(minutes=30),
    leased_until: datetime | None = NOW - timedelta(minutes=15),
    worker_id: str | None = "worker-1",
    lease_id: str | None = "lease-1",
) -> int:
    job = manager.create_job(
        domain="lifecycle",
        queue="default",
        job_type="work",
        payload={"input": 1},
        owner_user_id="owner-1",
        project_id=17,
        batch_group="batch-1",
        request_id="request-1",
        trace_id="trace-1",
    )
    job_id = int(job["id"])
    _execute(
        manager,
        (
            "UPDATE jobs SET status = %s, retry_count = 2, available_at = %s, "
            "started_at = %s, leased_until = %s, lease_id = %s, worker_id = %s, "
            "acquired_at = %s, result = '{\"partial\": true}'::jsonb, "
            "error_message = 'old error', error_code = 'old-code', "
            "last_error = 'old failure', completion_token = 'completion-1', "
            "progress_percent = 10.0, progress_message = 'old progress', "
            "created_at = %s, updated_at = %s WHERE id = %s"
        ),
        (
            status,
            available_at,
            NOW - timedelta(hours=1),
            leased_until,
            lease_id,
            worker_id,
            NOW - timedelta(hours=1),
            NOW - timedelta(days=1),
            datetime(2001, 1, 1, tzinfo=timezone.utc),
            job_id,
        ),
    )
    return job_id


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
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return self.now


def _run_batch_renew_cleanup(
    cleanup: Callable[[], None],
    *,
    primary_error: BaseException | None,
) -> None:
    """Run trigger cleanup without obscuring an earlier test-body failure."""

    try:
        cleanup()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.__cause__ = cleanup_error


def test_batch_renew_cleanup_preserves_primary_error_and_chains_cleanup_failure() -> None:
    primary_error = AssertionError("expected lease rollback")

    def fail_cleanup() -> None:
        raise RuntimeError("forced cleanup failure")

    def fail_test_body() -> None:
        try:
            raise primary_error
        except BaseException as error:
            _run_batch_renew_cleanup(fail_cleanup, primary_error=error)
            raise

    with pytest.raises(AssertionError, match="expected lease rollback") as exc_info:
        fail_test_body()

    assert exc_info.value is primary_error
    assert isinstance(primary_error.__cause__, RuntimeError)
    assert str(primary_error.__cause__) == "forced cleanup failure"


def test_batch_renew_cleanup_raises_failure_without_primary_error() -> None:
    def fail_cleanup() -> None:
        raise RuntimeError("forced cleanup failure")

    with pytest.raises(RuntimeError, match="forced cleanup failure"):
        _run_batch_renew_cleanup(fail_cleanup, primary_error=None)


def test_postgres_batch_renew_counts_ordered_attempts_and_preserves_longer_lease(
    manager: JobManager,
    conn: Any,
) -> None:
    valid_id = _insert_job(manager)
    queued_id = _insert_job(manager, status="queued", worker_id=None, lease_id=None)
    stale_id = _insert_job(manager)
    long_expiry = NOW + timedelta(hours=1)
    long_id = _insert_job(manager, leased_until=long_expiry)
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
    clock = RecordingClock(NOW)

    result = renew_leases_batch(
        conn,
        manager._pg_cursor,
        command=command,
        clock=clock,
    )

    assert result == BatchRenewLeasesResult(requested_count=6, applied_count=3)
    assert clock.calls == 1
    assert _fetch_job(manager, valid_id)["leased_until"] == NOW + timedelta(seconds=30)
    assert _fetch_job(manager, queued_id)["leased_until"] == NOW - timedelta(minutes=15)
    assert _fetch_job(manager, stale_id)["leased_until"] == NOW - timedelta(minutes=15)
    assert _fetch_job(manager, long_id)["leased_until"] == long_expiry


def test_postgres_batch_renew_empty_command_reads_clock_and_enters_cursor_once(
    manager: JobManager,
    conn: Any,
) -> None:
    cursor_entries = 0
    clock = RecordingClock(NOW)

    @contextmanager
    def tracking_cursor_factory(connection: Any) -> Iterator[Any]:
        nonlocal cursor_entries
        cursor_entries += 1
        with manager._pg_cursor(connection) as cur:
            yield cur

    result = renew_leases_batch(
        conn,
        tracking_cursor_factory,
        command=BatchRenewLeasesCommand(items=(), enforce=False),
        clock=clock,
    )

    assert result == BatchRenewLeasesResult(requested_count=0, applied_count=0)
    assert clock.calls == 1
    assert cursor_entries == 1


def test_postgres_batch_renew_rolls_back_when_later_update_trigger_fails(
    manager: JobManager,
    conn: Any,
) -> None:
    first_job_id = _insert_job(manager)
    second_job_id = _insert_job(manager)
    original_first_lease = _fetch_job(manager, first_job_id)["leased_until"]
    original_second_lease = _fetch_job(manager, second_job_id)["leased_until"]
    function_name = f"force_direct_batch_renew_function_{uuid.uuid4().hex}"
    trigger_name = f"force_direct_batch_renew_trigger_{uuid.uuid4().hex}"
    setup_connection = manager._connect()
    try:
        with setup_connection, manager._pg_cursor(setup_connection) as cur:
            cur.execute(
                f'CREATE FUNCTION "{function_name}"() RETURNS trigger LANGUAGE plpgsql AS $$ '
                "BEGIN RAISE EXCEPTION 'forced direct batch renewal failure'; END; $$"
            )
            cur.execute(
                f'CREATE TRIGGER "{trigger_name}" BEFORE UPDATE ON jobs '
                f'FOR EACH ROW WHEN (OLD.id = {int(second_job_id)}) '
                f'EXECUTE FUNCTION "{function_name}"()'
            )
    finally:
        setup_connection.close()

    command = BatchRenewLeasesCommand(
        items=(
            BatchRenewLeaseItem(first_job_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(second_job_id, 30, "worker-1", "lease-1"),
        ),
        enforce=True,
    )
    primary_error: BaseException | None = None

    def cleanup_trigger() -> None:
        cleanup_connection = manager._connect()
        try:
            with cleanup_connection, manager._pg_cursor(cleanup_connection) as cur:
                cur.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}" ON jobs')
                cur.execute(f'DROP FUNCTION IF EXISTS "{function_name}"()')
        finally:
            cleanup_connection.close()

    try:
        with pytest.raises(psycopg.Error, match="forced direct batch renewal failure"):
            renew_leases_batch(
                conn,
                manager._pg_cursor,
                command=command,
                clock=RecordingClock(NOW),
            )
        assert _fetch_job(manager, first_job_id)["leased_until"] == original_first_lease
        assert _fetch_job(manager, second_job_id)["leased_until"] == original_second_lease
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _run_batch_renew_cleanup(cleanup_trigger, primary_error=primary_error)


@pytest.mark.parametrize(
    ("operation", "command"),
    [
        (renew_lease, RenewLeaseCommand(job_id=999, seconds=30, enforce=False)),
        (release_job, ReleaseJobCommand(job_id=999, enforce=False)),
    ],
    ids=["renew", "release"],
)
def test_postgres_lifecycle_reports_missing_job(
    manager: JobManager,
    conn: Any,
    operation: Any,
    command: RenewLeaseCommand | ReleaseJobCommand,
) -> None:
    if isinstance(command, RenewLeaseCommand):
        result = operation(conn, manager._pg_cursor, command=command, now=NOW)
    else:
        result = operation(
            conn,
            manager._pg_cursor,
            command=command,
            counters_enabled=False,
        )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.MISSING


@pytest.mark.parametrize("operation", [renew_lease, release_job], ids=["renew", "release"])
def test_postgres_lifecycle_reports_wrong_status(
    manager: JobManager,
    conn: Any,
    operation: Any,
) -> None:
    job_id = _insert_job(manager, status="queued")

    if operation is renew_lease:
        result = operation(
            conn,
            manager._pg_cursor,
            command=_renew_command(job_id),
            now=NOW,
        )
    else:
        result = operation(
            conn,
            manager._pg_cursor,
            command=_release_command(job_id),
            counters_enabled=False,
        )

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
def test_postgres_lifecycle_reports_stale_enforced_identity(
    manager: JobManager,
    conn: Any,
    operation: Any,
    worker_id: str,
    lease_id: str,
) -> None:
    job_id = _insert_job(manager)

    if operation is renew_lease:
        result = operation(
            conn,
            manager._pg_cursor,
            command=_renew_command(job_id, worker_id=worker_id, lease_id=lease_id),
            now=NOW,
        )
    else:
        result = operation(
            conn,
            manager._pg_cursor,
            command=_release_command(job_id, worker_id=worker_id, lease_id=lease_id),
            counters_enabled=False,
        )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.STALE_LEASE
    current = _fetch_job(manager, job_id)
    assert (current["status"], current["lease_id"], current["worker_id"]) == (
        "processing",
        "lease-1",
        "worker-1",
    )


@pytest.mark.parametrize("operation", [renew_lease, release_job], ids=["renew", "release"])
def test_postgres_lifecycle_ignores_identity_without_enforcement(
    manager: JobManager,
    conn: Any,
    operation: Any,
) -> None:
    job_id = _insert_job(manager)

    if operation is renew_lease:
        result = operation(
            conn,
            manager._pg_cursor,
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
            manager._pg_cursor,
            command=_release_command(
                job_id,
                enforce=False,
                worker_id="stale-worker",
                lease_id="stale-lease",
            ),
            counters_enabled=False,
        )

    assert result.outcome is OperationOutcome.APPLIED


def test_postgres_renew_preserves_longer_current_lease_and_returns_transition_facts(
    manager: JobManager,
    conn: Any,
) -> None:
    current_expiry = NOW + timedelta(hours=1)
    job_id = _insert_job(manager, leased_until=current_expiry)

    result = renew_lease(
        conn,
        manager._pg_cursor,
        command=_renew_command(job_id),
        now=NOW,
    )

    assert result.outcome is OperationOutcome.APPLIED
    assert result.row is not None
    assert set(result.row) == RENEW_RESULT_FIELDS
    assert result.row["leased_until"] == current_expiry
    persisted = _fetch_job(manager, job_id)
    assert persisted["payload"] == {"input": 1}
    assert persisted["owner_user_id"] == "owner-1"
    assert persisted["project_id"] == 17


@pytest.mark.parametrize("enforce", [False, True], ids=["unenforced", "enforced"])
@pytest.mark.parametrize("progress_percent", [None, 55.5], ids=["percent-unchanged", "percent-set"])
@pytest.mark.parametrize("progress_message", [None, "halfway"], ids=["message-unchanged", "message-set"])
def test_postgres_renew_supports_every_progress_and_enforcement_variant(
    manager: JobManager,
    conn: Any,
    enforce: bool,
    progress_percent: float | None,
    progress_message: str | None,
) -> None:
    job_id = _insert_job(manager)

    result = renew_lease(
        conn,
        manager._pg_cursor,
        command=_renew_command(
            job_id,
            enforce=enforce,
            progress_percent=progress_percent,
            progress_message=progress_message,
        ),
        now=NOW,
    )

    assert result.row is not None
    assert result.row["leased_until"] == NOW + timedelta(seconds=30)
    assert result.row["progress_percent"] == (10.0 if progress_percent is None else 55.5)
    assert result.row["progress_message"] == (
        "old progress" if progress_message is None else "halfway"
    )


def test_postgres_release_clears_lease_fields_and_preserves_unrelated_facts(
    manager: JobManager,
    conn: Any,
) -> None:
    job_id = _insert_job(manager)

    result = release_job(
        conn,
        manager._pg_cursor,
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
    persisted = _fetch_job(manager, job_id)
    assert persisted["payload"] == {"input": 1}
    assert persisted["result"] == {"partial": True}
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
    assert result.row["updated_at"] != datetime(2001, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("processing_count", "expected_processing"),
    [(2, 1), (0, 0)],
    ids=["decrement", "floor-zero"],
)
def test_postgres_release_moves_processing_counter_to_ready(
    manager: JobManager,
    conn: Any,
    processing_count: int,
    expected_processing: int,
) -> None:
    job_id = _insert_job(manager)
    _execute(
        manager,
        (
            "INSERT INTO job_counters(domain, queue, job_type, ready_count, scheduled_count, "
            "processing_count, quarantined_count) "
            "VALUES('lifecycle', 'default', 'work', 4, 3, %s, 0)"
        ),
        (processing_count,),
    )

    release_job(
        conn,
        manager._pg_cursor,
        command=_release_command(job_id),
        counters_enabled=True,
    )

    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
                "WHERE domain = 'lifecycle' AND queue = 'default' AND job_type = 'work'"
            )
            counter = cur.fetchone()
    finally:
        connection.close()
    assert counter is not None
    assert tuple(counter.values()) == (5, 3, expected_processing)


def test_postgres_release_creates_missing_counter_as_one_ready(
    manager: JobManager,
    conn: Any,
) -> None:
    job_id = _insert_job(manager)

    release_job(
        conn,
        manager._pg_cursor,
        command=_release_command(job_id),
        counters_enabled=True,
    )

    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
                "WHERE domain = 'lifecycle' AND queue = 'default' AND job_type = 'work'"
            )
            counter = cur.fetchone()
    finally:
        connection.close()
    assert counter is not None
    assert tuple(counter.values()) == (1, 0, 0)


def test_postgres_release_counter_failure_rolls_back_transition(
    manager: JobManager,
    conn: Any,
) -> None:
    job_id = _insert_job(manager)
    _execute(
        manager,
        (
            "CREATE FUNCTION fail_release_counter() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'forced counter failure'; END $$"
        ),
    )
    _execute(
        manager,
        (
            "CREATE TRIGGER fail_release_counter BEFORE INSERT OR UPDATE ON job_counters "
            "FOR EACH ROW EXECUTE FUNCTION fail_release_counter()"
        ),
    )

    with pytest.raises(psycopg.Error, match="forced counter failure"):
        release_job(
            conn,
            manager._pg_cursor,
            command=_release_command(job_id),
            counters_enabled=True,
        )

    current = _fetch_job(manager, job_id)
    assert current["status"] == "processing"
    assert current["available_at"] == NOW - timedelta(minutes=30)
    assert current["leased_until"] == NOW - timedelta(minutes=15)
    assert current["worker_id"] == "worker-1"
    assert current["lease_id"] == "lease-1"
    assert current["completion_token"] == "completion-1"


class _PauseAfterValidationCursor:
    def __init__(
        self,
        inner: Any,
        selected: threading.Event,
        writer_attempt_finished: threading.Event,
    ) -> None:
        self._inner = inner
        self._selected = selected
        self._writer_attempt_finished = writer_attempt_finished

    def execute(self, sql: Any, params: Any = ()) -> Any:
        result = self._inner.execute(sql, params)
        normalized = " ".join(str(sql).lower().split())
        if "from jobs where id = %s for update" in normalized:
            self._selected.set()
            if not self._writer_attempt_finished.wait(timeout=5):
                raise TimeoutError("competing writer did not finish its first attempt")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_postgres_release_locks_validation_against_competing_reassignment(
    manager: JobManager,
) -> None:
    job_id = _insert_job(manager)
    selected = threading.Event()
    writer_attempt_finished = threading.Event()
    release_done = threading.Event()
    released: dict[str, Any] = {}
    writer: dict[str, Any] = {}

    @contextmanager
    def pausing_cursor_factory(connection: Any) -> Iterator[Any]:
        with manager._pg_cursor(connection) as cur:
            yield _PauseAfterValidationCursor(cur, selected, writer_attempt_finished)

    def run_release() -> None:
        connection = manager._connect()
        try:
            released["result"] = release_job(
                connection,
                pausing_cursor_factory,
                command=_release_command(job_id, enforce=False),
                counters_enabled=False,
            )
        except (RuntimeError, TimeoutError, psycopg.Error) as exc:  # pragma: no cover - asserted below
            released["error"] = exc
        finally:
            connection.close()
            release_done.set()

    def assign_replacement(cur: Any) -> None:
        cur.execute(
            (
                "UPDATE jobs SET status = 'processing', available_at = NULL, "
                "leased_until = NOW() + interval '1 hour', worker_id = 'worker-2', "
                "lease_id = 'lease-2', acquired_at = NOW(), started_at = NOW(), "
                "completion_token = NULL WHERE id = %s"
            ),
            (job_id,),
        )

    def run_writer() -> None:
        connection = manager._connect()
        try:
            if not selected.wait(timeout=5):
                raise TimeoutError("release did not reach validation")
            try:
                with manager._pg_cursor(connection) as cur:
                    cur.execute("SET lock_timeout = '200ms'")
                    assign_replacement(cur)
                connection.commit()
                writer["blocked"] = False
            except psycopg.errors.LockNotAvailable:
                connection.rollback()
                writer["blocked"] = True
            finally:
                writer_attempt_finished.set()

            if writer.get("blocked"):
                if not release_done.wait(timeout=5):
                    raise TimeoutError("release did not commit after the blocked reassignment")
                with connection, manager._pg_cursor(connection) as cur:
                    assign_replacement(cur)
        except (RuntimeError, TimeoutError, psycopg.Error) as exc:  # pragma: no cover - asserted below
            writer["error"] = exc
            writer_attempt_finished.set()
        finally:
            connection.close()

    release_thread = threading.Thread(target=run_release)
    writer_thread = threading.Thread(target=run_writer)
    release_thread.start()
    writer_thread.start()
    release_thread.join(timeout=10)
    writer_thread.join(timeout=10)

    assert not release_thread.is_alive()
    assert not writer_thread.is_alive()
    assert "error" not in released
    assert "error" not in writer
    assert writer["blocked"] is True
    assert released["result"].outcome is OperationOutcome.APPLIED
    assert released["result"].row["status"] == "queued"
    current = _fetch_job(manager, job_id)
    assert current["status"] == "processing"
    assert current["worker_id"] == "worker-2"
    assert current["lease_id"] == "lease-2"
