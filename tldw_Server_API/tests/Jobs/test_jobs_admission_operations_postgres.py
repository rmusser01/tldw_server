from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import AdmissionResult
from tldw_Server_API.app.core.Jobs.operations.postgres import admission

psycopg = pytest.importorskip("psycopg")


def test_quota_transaction_without_psycopg_raises_clear_error(monkeypatch):
    class FakeConnection:
        isolation_level = object()
        closed = False

    monkeypatch.setattr(admission, "_psycopg", None)

    with pytest.raises(RuntimeError, match="psycopg is required for PostgreSQL quota admission"):
        with admission._read_committed_quota_transaction(FakeConnection(), enabled=True):
            pytest.fail("quota transaction unexpectedly started")


@pytest.mark.pg_jobs
def test_counter_failure_rolls_back_to_savepoint_and_commits_job_event(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")

    def fail_counter(cur, *, command, available_at):
        del command, available_at
        cur.execute("SELECT definitely_missing_column FROM job_counters")

    monkeypatch.setattr(admission, "_bump_counters", fail_counter)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    created = manager.create_job(
        domain="admission-fault",
        queue="default",
        job_type="counter",
        payload={},
        owner_user_id="owner-1",
    )

    assert created["status"] == "queued"
    events = manager.list_job_events_after(after_id=0, domain="admission-fault", limit=10)
    assert [event["event_type"] for event in events] == ["job.created"]


def test_quota_rejection_propagates_quota_query_errors():
    class FailingQuotaCursor:
        def execute(self, sql, params=()):
            del sql, params
            raise psycopg.ProgrammingError("quota read failed")

    command = admission.CreateJobCommand(
        domain="admission",
        queue="default",
        job_type="quota-fail",
        payload={},
        owner_user_id="owner-1",
    )

    with pytest.raises(psycopg.ProgrammingError, match="quota read failed"):
        admission._quota_rejection(
            FailingQuotaCursor(),
            command=command,
            now=datetime.now(timezone.utc),
            max_queued_quota=1,
            submits_per_minute_quota=0,
        )


def test_postgres_admission_lock_precedes_owner_scope_policy_check() -> None:
    events: list[object] = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=()):
            events.append(("sql", str(sql), params))

    class Connection:
        def __enter__(self):
            events.append("transaction")
            return self

        def __exit__(self, *_args):
            return False

    def reject_after_lock(_cursor):
        events.append("policy")
        raise RuntimeError("policy rejected")

    with pytest.raises(RuntimeError, match="policy rejected"):
        admission.create_job_admission(
            Connection(),
            lambda _connection: Cursor(),
            command=admission.CreateJobCommand(
                domain="notes",
                queue="graph-suggestions",
                job_type="note_graph_suggestions",
                payload={},
                owner_user_id="owner-1",
                idempotency_key="run-1",
            ),
            uuid_value="job-1",
            now=datetime(2026, 8, 27, tzinfo=timezone.utc),
            max_queued_quota=0,
            submits_per_minute_quota=0,
            counters_enabled=False,
            advisory_xact_lock_key=42,
            pre_admission_lookup=reject_after_lock,
        )

    assert events == [
        "transaction",
        ("sql", "SELECT pg_advisory_xact_lock(%s)", (42,)),
        "policy",
    ]


def test_postgres_manager_wires_owner_and_exact_job_scope_into_admission_lock(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", "graph-suggestions")
    manager = JobManager(tmp_path / "jobs.db")
    manager.backend = "postgres"
    captured: dict[str, object] = {}

    class Connection:
        @staticmethod
        def close():
            return None

    def capture_operation(_conn, _cursor_factory, **kwargs):
        captured.update(kwargs)
        command = kwargs["command"]
        return AdmissionResult.applied(
            row={
                "id": 1,
                "uuid": kwargs["uuid_value"],
                "domain": command.domain,
                "queue": command.queue,
                "job_type": command.job_type,
                "owner_user_id": command.owner_user_id,
                "status": "queued",
            }
        )

    monkeypatch.setattr(manager, "_connect", lambda: Connection())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager._postgres_create_job_admission",
        capture_operation,
    )
    policy = SimpleNamespace(
        active_statuses=("queued", "processing"),
        active_limit=1,
        admission_limit=20,
        created_after=datetime.now(timezone.utc) - timedelta(hours=1),
    )

    manager.create_job(
        domain="notes",
        queue="graph-suggestions",
        job_type="note_graph_suggestions",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="run-1",
        owner_scope_admission=policy,
    )

    assert captured["advisory_xact_lock_key"] == manager._pg_advisory_key(
        "owner-scope-admission",
        "owner-1",
        "notes",
        "graph-suggestions",
        "note_graph_suggestions",
    )
    assert callable(captured["pre_admission_lookup"])
