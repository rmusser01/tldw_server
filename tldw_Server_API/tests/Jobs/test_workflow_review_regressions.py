"""Admission and acquisition parity regressions from PR 3319 review."""

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import psycopg
import pytest

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome, WorkflowScope


@pytest.mark.integration
@pytest.mark.parametrize("column", ["owner_user_id", "idempotency_key"])
def test_root_bounded_identity_rejects_embedded_nul(workflow_case, column):
    case = workflow_case
    errors = (sqlite3.IntegrityError,) if case.manager.backend == "sqlite" else (ValueError, psycopg.DataError)
    with case.connection() as conn:
        with pytest.raises(errors, match="CHECK constraint failed|NUL|0x00"):
            case.insert_root(conn, **{column: "identity\x00suffix"})


@pytest.mark.integration
def test_receipt_bounded_facts_reject_embedded_nul(workflow_case):
    case = workflow_case
    errors = (sqlite3.IntegrityError,) if case.manager.backend == "sqlite" else (ValueError, psycopg.DataError)
    with case.connection() as conn:
        case.insert_root(conn)
        with pytest.raises(errors, match="CHECK constraint failed|NUL|0x00"):
            case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(facts="{}\x00suffix"))


@pytest.mark.unit
@pytest.mark.parametrize("column", ["owner_user_id", "idempotency_key"])
def test_admission_identity_rejects_nul_before_database(uninstalled_workflow_case, column):
    with pytest.raises(ValueError):
        replace(uninstalled_workflow_case.command, **{column: "identity\x00suffix"})


@pytest.mark.integration
def test_admission_generates_persisted_trace_and_replay_retains_it(workflow_case):
    case = workflow_case
    assert case.admit().outcome is WorkflowOutcome.APPLIED
    trace = case.job_row()["trace_id"]
    assert trace is not None
    assert str(UUID(trace, version=4)) == trace
    assert case.admit().outcome is WorkflowOutcome.ALREADY_APPLIED
    assert case.job_row()["trace_id"] == trace
    with case.connection() as conn:
        events = case.execute(
            conn,
            "SELECT trace_id FROM job_events WHERE job_id=? AND event_type='job.created'",
            (case.root.job_id_hint,),
        ).fetchall()
    assert [row[0] for row in events] == [trace]


@pytest.mark.integration
@pytest.mark.parametrize(
    "column,value",
    [
        ("max_retries", 101),
        ("max_retries", 2**63),
        ("quarantine_threshold", 2**31),
        ("quarantine_threshold", 2**63),
    ],
)
def test_controls_reject_unpersistable_integers_before_admission(workflow_case, column, value):
    case = workflow_case
    with pytest.raises(ValueError, match="bounded integer"):
        controls = replace(case.command.controls, **{column: value})
        case.capability.admit(replace(case.command, controls=controls))
    assert case.job_count() == 0


@pytest.mark.integration
def test_controls_storage_boundaries_admit_and_replay_exactly(workflow_case):
    case = workflow_case
    result = case.admit(max_retries=100, quarantine_threshold=2**31 - 1)
    assert result.outcome is WorkflowOutcome.APPLIED
    assert case.job_row()["max_retries"] == 100
    assert case.job_row()["quarantine_threshold"] == 2**31 - 1
    assert case.admit(max_retries=100, quarantine_threshold=2**31 - 1).root == result.root


@pytest.fixture(params=[False, True], ids=["two-step", "single-update"])
def acquisition_mode(request, monkeypatch):
    for backend in ("SQLITE", "PG"):
        monkeypatch.setenv(f"JOBS_{backend}_SINGLE_UPDATE_ACQUIRE", str(request.param))
    for prefix in ("JOBS", "JOBS_SQLITE", "JOBS_PG", "JOBS_POSTGRES"):
        for suffix in ("", "_CHATBOOKS"):
            monkeypatch.delenv(f"{prefix}_ACQUIRE_TIE_BREAK{suffix}", raising=False)
    return request.param


@pytest.mark.integration
@pytest.mark.parametrize("enrolled", [True, False], ids=["enrolled", "legacy"])
@pytest.mark.parametrize("future", [True, False], ids=["future-ahead-clock", "due-behind-clock"])
def test_chatbooks_implicit_order_uses_per_row_schedule_clock(
    workflow_case, monkeypatch, acquisition_mode, enrolled, future
):
    case = workflow_case
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CHATBOOKS", "default")
    scope = WorkflowScope("chatbooks", "default", "export")
    case.capability = case.manager.workflow_capability(replace(case.capability.policy, scope=scope))
    ready = [str(uuid4()), str(uuid4())]
    with case.connection() as conn:
        for index, job_uuid in enumerate(ready):
            case.insert(
                conn,
                "jobs",
                {
                    "uuid": job_uuid,
                    "domain": "chatbooks",
                    "queue": "default",
                    "job_type": "export",
                    "owner_user_id": "owner-a",
                    "payload": "{}",
                    "status": "queued",
                    "priority": 5,
                    "created_at": f"2020-01-0{index + 1} 00:00:00",
                },
            )
        now = (
            conn.execute("SELECT clock_timestamp()").fetchone()[0]
            if case.manager.backend == "postgres"
            else datetime.fromisoformat(conn.execute("SELECT DATETIME('now')").fetchone()[0])
        )
        schedule = now + timedelta(days=1 if future else -1)
        conn.commit()
    if enrolled:
        assert case.admit().outcome is WorkflowOutcome.APPLIED
        with case.connection() as conn:
            case.execute(
                conn,
                "UPDATE jobs SET available_at=?,priority=10 WHERE uuid=?",
                (schedule, case.root.job_uuid),
            )
            conn.commit()
    else:
        with case.connection() as conn:
            case.insert(
                conn,
                "jobs",
                {
                    "uuid": str(uuid4()),
                    "domain": "chatbooks",
                    "queue": "default",
                    "job_type": "export",
                    "owner_user_id": "owner-a",
                    "payload": "{}",
                    "status": "queued",
                    "priority": 10,
                    "available_at": schedule,
                },
            )
            conn.commit()
    monkeypatch.setattr(
        case.manager._clock, "now_utc", lambda: datetime(2100 if future else 2000, 1, 1, tzinfo=timezone.utc)
    )
    case.manager.set_rls_context(owner_user_id="owner-a", is_admin=False, domain_allowlist=None)
    try:
        acquired = case.manager.acquire_next_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            owner_user_id="owner-a",
            worker_id="review-worker",
            lease_seconds=120,
        )
    finally:
        case.manager.clear_rls_context()
    # Only SQLite's implicit two-step chatbooks policy switches FIFO/LIFO.
    expected_index = 0
    if case.manager.backend == "sqlite" and not acquisition_mode:
        expected_index = int(not future) if enrolled else int(future)
    assert acquired["uuid"] == ready[expected_index]


@pytest.mark.integration
@pytest.mark.parametrize("table", ["jobs", "jobs_archive"], ids=["active", "archive"])
@pytest.mark.parametrize("owner", ["owner-a", "owner-b", None], ids=["same-owner", "other-owner", "unowned"])
def test_legacy_correlation_respects_owner_without_bypassing_active_uniqueness(workflow_case, table, owner):
    case = workflow_case
    legacy_uuid = str(uuid4())
    lookup = {
        "jobs": "SELECT * FROM jobs WHERE uuid=?",
        "jobs_archive": "SELECT * FROM jobs_archive WHERE uuid=?",
    }[table]
    with case.connection() as conn:
        case.insert(
            conn,
            table,
            {
                "uuid": legacy_uuid,
                "domain": "embeddings",
                "queue": "workflow",
                "job_type": "embedding_workflow",
                "owner_user_id": owner,
                "idempotency_key": case.command.idempotency_key,
                "payload": "{}",
                "status": "queued" if table == "jobs" else "completed",
            },
        )
        conn.commit()
        before = case.execute(conn, lookup, (legacy_uuid,)).fetchone()
    result = case.admit()
    if table == "jobs" or owner == "owner-a":
        assert result.outcome is WorkflowOutcome.CONFLICT
        assert result.root is None
        with case.connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0
    else:
        assert result.outcome is WorkflowOutcome.APPLIED
        assert result.root.owner_user_id == "owner-a"
        assert result.root.job_uuid != legacy_uuid
        assert case.admit().outcome is WorkflowOutcome.ALREADY_APPLIED
    with case.connection() as conn:
        after = case.execute(conn, lookup, (legacy_uuid,)).fetchone()
    assert tuple(after) == tuple(before)
