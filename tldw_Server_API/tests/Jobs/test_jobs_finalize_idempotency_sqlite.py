import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Jobs import manager as manager_module
from tldw_Server_API.app.core.Jobs import metrics as metrics_module
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


def _set_env(monkeypatch, tmp_path):

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    # SQLite DB path under test dir
    db_path = os.path.join(os.getcwd(), "Databases", "jobs.db")
    monkeypatch.setenv("JOBS_DB_PATH", db_path)
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")


def test_complete_idempotent_with_token(monkeypatch, tmp_path):

    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1")
    assert acq and acq["id"] == j["id"]
    token = str(acq.get("lease_id"))
    ok1 = jm.complete_job(
        int(j["id"]), result={"ok": True}, worker_id="w1", lease_id=str(acq.get("lease_id")), completion_token=token
    )
    assert ok1 is True
    # Repeat with same token: should be idempotent success
    ok2 = jm.complete_job(
        int(j["id"]), result={"ok": True}, worker_id="w1", lease_id=str(acq.get("lease_id")), completion_token=token
    )
    assert ok2 is True
    # Different token: should not re-complete (status terminal); returns False
    ok3 = jm.complete_job(
        int(j["id"]),
        result={"ok": True},
        worker_id="w1",
        lease_id=str(acq.get("lease_id")),
        completion_token=token + "-x",
    )
    assert ok3 is False


def test_fail_idempotent_with_token(monkeypatch, tmp_path):

    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2")
    assert acq and acq["id"] == j["id"]
    token = str(acq.get("lease_id"))
    ok1 = jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq.get("lease_id")),
        completion_token=token,
    )
    assert ok1 is True
    # Repeat with same token: idempotent success
    ok2 = jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq.get("lease_id")),
        completion_token=token,
    )
    assert ok2 is True
    # Different token: job is already terminal
    ok3 = jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq.get("lease_id")),
        completion_token=token + "-x",
    )
    assert ok3 is False


def test_worker_terminalizer_is_exact_idempotent_cas(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="terminal-cas",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])
    arguments = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": lease_id,
        "completion_token": lease_id,
        "status": "failed",
        "error_code": "slides_render_failed",
        "error_message": "stable safe detail",
    }

    assert jm.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert jm.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"
    assert (
        jm.terminalize_job_from_worker(
            **{**arguments, "error_message": "different detail"},
        )
        == "CONFLICT"
    )
    assert (
        jm.terminalize_job_from_worker(
            **{**arguments, "completion_token": lease_id + "-different"},
        )
        == "CONFLICT"
    )


@pytest.mark.parametrize("winner_status", ["failed", "cancelled"])
def test_worker_terminalizer_reports_exact_generic_terminal_winner(
    monkeypatch,
    tmp_path,
    winner_status,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "false")
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"terminal-winner-{winner_status}",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])
    if winner_status == "failed":
        assert jm.fail_job(
            int(job["id"]),
            error="generic terminal winner",
            retryable=False,
            worker_id="slides-worker",
            lease_id=lease_id,
            completion_token=lease_id,
            enforce=True,
            error_code="terminal_first_failure",
            error_class="TerminalFirstFailure",
        )
    else:
        assert jm.cancel_job(int(job["id"]), reason="generic terminal winner")

    assert (
        jm.terminalize_job_from_worker(
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner_user_id="owner-1",
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            worker_id="slides-worker",
            lease_id=lease_id,
            completion_token=lease_id,
            status=winner_status,
            error_code=f"handler_{winner_status}",
            error_message="bounded handler outcome",
        )
        == "ALREADY_TERMINAL"
    )


@pytest.mark.parametrize(
    ("status", "event_type"),
    [("failed", "job.failed"), ("cancelled", "job.cancelled")],
)
def test_worker_terminalizer_applied_bookkeeping_happens_exactly_once(
    monkeypatch,
    tmp_path,
    status,
    event_type,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"terminal-bookkeeping-{status}",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        conn.execute(
            """
            UPDATE job_counters SET processing_count=1
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        )
        conn.commit()
    arguments = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": lease_id,
        "completion_token": lease_id,
        "status": status,
        "error_code": f"slides_render_{status}",
        "error_message": f"{status} safely",
    }

    assert jm.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert jm.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"

    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        processing_count = conn.execute(
            """
            SELECT processing_count FROM job_counters
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        ).fetchone()[0]
        event_count = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type=?",
            (int(job["id"]), event_type),
        ).fetchone()[0]
    assert processing_count == 0
    assert event_count == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_worker_terminalizer_applied_side_effects_are_not_repeated(
    monkeypatch,
    tmp_path,
    status,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"terminal-side-effects-{status}",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    calls = {
        "gauges": 0,
        "failures": 0,
        "failure_codes": 0,
        "cancelled": 0,
        "events": 0,
        "audits": 0,
    }

    def record(name):
        calls[name] += 1

    monkeypatch.setattr(jm, "_update_gauges", lambda **_kwargs: record("gauges"))
    monkeypatch.setattr(
        manager_module,
        "increment_failures",
        lambda *_args, **_kwargs: record("failures"),
    )
    monkeypatch.setattr(
        metrics_module,
        "increment_failures_by_code",
        lambda *_args, **_kwargs: record("failure_codes"),
    )
    monkeypatch.setattr(
        manager_module,
        "increment_cancelled",
        lambda *_args, **_kwargs: record("cancelled"),
    )
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: record("events"),
    )
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: record("audits"),
    )
    lease_id = str(acquired["lease_id"])
    arguments = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": lease_id,
        "completion_token": lease_id,
        "status": status,
        "error_code": f"slides_render_{status}",
        "error_message": f"{status} safely",
    }

    assert jm.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert jm.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"

    assert calls["gauges"] == 1
    assert calls["events"] == 1
    assert calls["audits"] == 1
    assert calls["failures"] == (1 if status == "failed" else 0)
    assert calls["failure_codes"] == (1 if status == "failed" else 0)
    assert calls["cancelled"] == (1 if status == "cancelled" else 0)


def test_worker_terminalizer_rejects_wrong_uuid_owner_scope_and_lease(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="terminal-correlation",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    base = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": str(acquired["lease_id"]),
        "completion_token": str(acquired["lease_id"]),
        "status": "cancelled",
        "error_code": "slides_render_cancelled",
        "error_message": "cancelled safely",
    }

    for changed in (
        {"job_uuid": "00000000-0000-0000-0000-000000000000"},
        {"owner_user_id": "owner-2"},
        {"domain": "other"},
        {"queue": "other"},
        {"job_type": "other"},
        {"worker_id": "other-worker"},
        {"lease_id": "other-lease"},
    ):
        assert jm.terminalize_job_from_worker(**{**base, **changed}) == "CONFLICT"

    assert jm.get_job(int(job["id"]))["status"] == "processing"


def test_worker_terminalizer_rejects_exact_non_slides_job(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )
    acquired = jm.acquire_next_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        lease_seconds=30,
        worker_id="generic-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])

    assert (
        jm.terminalize_job_from_worker(
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner_user_id="owner-1",
            domain="chatbooks",
            queue="default",
            job_type="export",
            worker_id="generic-worker",
            lease_id=lease_id,
            completion_token=lease_id,
            status="failed",
            error_code="generic_failure",
            error_message="safe detail",
        )
        == "CONFLICT"
    )
    assert jm.get_job(int(job["id"]))["status"] == "processing"


def _create_reconciler_job(jm, *, key, available_at=None):
    return jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=key,
        available_at=available_at,
    )


def test_reconciler_terminalizer_is_uuid_authoritative_and_idempotent(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = _create_reconciler_job(jm, key="reconciler-idempotent")
    arguments = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "expected_status": "queued",
        "status": "failed",
        "error_code": "generation_expired",
        "error_message": "generation input expired",
        "completion_token": "reconciler:expiration:v1",
        "job_id": int(job["id"]),
    }

    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"
    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "IDEMPOTENT"
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{**arguments, "error_message": "different safe detail"},
        )
        == "CONFLICT"
    )
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{**arguments, "completion_token": "reconciler:expiration:v2"},
        )
        == "CONFLICT"
    )

    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "failed"
    assert stored["error_code"] == "generation_expired"
    assert stored["completion_token"] == "reconciler:expiration:v1"
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        terminal_events = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.failed'",
            (int(job["id"]),),
        ).fetchone()[0]
    assert terminal_events == 1


def test_reconciler_terminalizer_replay_requires_exact_cancelled_state(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = _create_reconciler_job(jm, key="reconciler-cancelled-state")
    arguments = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "expected_status": "queued",
        "status": "cancelled",
        "error_code": "generation_cancelled",
        "error_message": "generation cancelled",
        "completion_token": "reconciler:cancelled-state:v1",
    }
    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        conn.execute(
            "UPDATE jobs SET cancellation_reason='different reason' WHERE uuid=?",
            (str(job["uuid"]),),
        )
        conn.commit()

    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "CONFLICT"


def test_reconciler_terminalizer_fails_closed_when_coordination_is_not_ready(
    monkeypatch,
    tmp_path,
):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = _create_reconciler_job(jm, key="reconciler-not-ready")
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        conn.execute("DROP INDEX idx_jobs_archive_uuid_unique")
        conn.commit()

    with pytest.raises(ValueError, match="coordination is unavailable"):
        jm.terminalize_slides_generation_job_from_reconciler(
            job_uuid=str(job["uuid"]),
            job_id=int(job["id"]),
            owner_user_id="owner-1",
            expected_status="queued",
            status="failed",
            error_code="generation_unavailable",
            error_message="coordination unavailable",
            completion_token="reconciler:not-ready:v1",
        )

    assert jm.get_job(int(job["id"]))["status"] == "queued"


def test_reconciler_terminalizer_never_mutates_ambiguous_active_uuid_rows(
    monkeypatch,
    tmp_path,
):
    _set_env(monkeypatch, tmp_path)
    JobManager()
    db_path = os.environ["JOBS_DB_PATH"]
    with sqlite3.connect(db_path) as conn:
        create_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone()[0]
        assert "uuid TEXT UNIQUE" in create_sql
        shadow_sql = create_sql.replace(
            "CREATE TABLE jobs",
            "CREATE TABLE jobs_without_uuid_unique",
            1,
        ).replace("uuid TEXT UNIQUE", "uuid TEXT", 1)
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(shadow_sql)
        conn.execute("INSERT INTO jobs_without_uuid_unique SELECT * FROM jobs")
        conn.execute("DROP TABLE jobs")
        conn.execute("ALTER TABLE jobs_without_uuid_unique RENAME TO jobs")
        conn.commit()

    ensure_jobs_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, payload, status
            ) VALUES (
                'ambiguous-active-uuid', 'slides', 'default',
                'presentation.generate', 'owner-1', ?, '{}', 'queued'
            )
            """,
            (("ambiguous-a",), ("ambiguous-b",)),
        )
        conn.commit()

    jm = JobManager(db_path)
    with pytest.raises(ValueError, match="correlation is unsafe"):
        jm.terminalize_slides_generation_job_from_reconciler(
            job_uuid="ambiguous-active-uuid",
            owner_user_id="owner-1",
            expected_status="queued",
            status="failed",
            error_code="generation_ambiguous",
            error_message="ambiguous generation correlation",
            completion_token="reconciler:ambiguous:v1",
        )

    with sqlite3.connect(db_path) as conn:
        statuses = conn.execute("SELECT status FROM jobs WHERE uuid='ambiguous-active-uuid' ORDER BY id").fetchall()
        event_count = conn.execute("SELECT COUNT(*) FROM job_events WHERE event_type='job.failed'").fetchone()[0]
        diagnostic = conn.execute(
            """
            SELECT diagnostic_code, diagnostic_count
            FROM slides_standalone_reconciliation WHERE singleton_id=1
            """
        ).fetchone()
    assert statuses == [("queued",), ("queued",)]
    assert event_count == 0
    assert diagnostic[0] == "ambiguous_generation_legacy_row"
    assert diagnostic[1] >= 2


@pytest.mark.parametrize(
    ("source_status", "available_at", "expected_counts"),
    [
        ("queued", None, (2, 4, 5)),
        ("queued", "future", (3, 3, 5)),
        ("processing", None, (3, 4, 4)),
    ],
)
def test_reconciler_terminalizer_decrements_the_source_counter_once(
    monkeypatch,
    tmp_path,
    source_status,
    available_at,
    expected_counts,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = JobManager()
    scheduled_at = datetime.now(timezone.utc) + timedelta(hours=1) if available_at else None
    job = _create_reconciler_job(
        jm,
        key=f"reconciler-counter-{source_status}-{available_at}",
        available_at=scheduled_at,
    )
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        if source_status == "processing":
            conn.execute(
                """
                UPDATE jobs
                SET status='processing', worker_id='worker', lease_id='lease',
                    leased_until=DATETIME('now', '+1 hour')
                WHERE id=?
                """,
                (int(job["id"]),),
            )
        conn.execute(
            """
            INSERT INTO job_counters(
                domain, queue, job_type, ready_count, scheduled_count,
                processing_count, quarantined_count
            ) VALUES('slides', 'default', 'presentation.generate', 3, 4, 5, 0)
            ON CONFLICT(domain, queue, job_type) DO UPDATE SET
                ready_count=3, scheduled_count=4, processing_count=5
            """
        )
        conn.commit()
    arguments = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "expected_status": source_status,
        "status": "cancelled",
        "error_code": "generation_cancelled",
        "error_message": "generation cancelled",
        "completion_token": f"reconciler:cancel:{source_status}:{available_at}",
        "job_id": int(job["id"]),
    }

    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"
    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "IDEMPOTENT"

    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        counts = conn.execute(
            """
            SELECT ready_count, scheduled_count, processing_count
            FROM job_counters
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        ).fetchone()
        terminal_events = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.cancelled'",
            (int(job["id"]),),
        ).fetchone()[0]
    assert counts == expected_counts
    assert terminal_events == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_reconciler_terminalizer_applied_side_effects_are_not_repeated(
    monkeypatch,
    tmp_path,
    status,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = JobManager()
    job = _create_reconciler_job(
        jm,
        key=f"reconciler-side-effects-{status}",
    )
    calls = {
        "gauges": 0,
        "failures": 0,
        "failure_codes": 0,
        "cancelled": 0,
        "events": 0,
        "audits": 0,
    }

    def record(name):
        calls[name] += 1

    monkeypatch.setattr(jm, "_update_gauges", lambda **_kwargs: record("gauges"))
    monkeypatch.setattr(
        manager_module,
        "increment_failures",
        lambda *_args, **_kwargs: record("failures"),
    )
    monkeypatch.setattr(
        metrics_module,
        "increment_failures_by_code",
        lambda *_args, **_kwargs: record("failure_codes"),
    )
    monkeypatch.setattr(
        manager_module,
        "increment_cancelled",
        lambda *_args, **_kwargs: record("cancelled"),
    )
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: record("events"),
    )
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: record("audits"),
    )
    arguments = {
        "job_uuid": str(job["uuid"]),
        "job_id": int(job["id"]),
        "owner_user_id": "owner-1",
        "expected_status": "queued",
        "status": status,
        "error_code": f"generation_{status}",
        "error_message": f"generation {status}",
        "completion_token": f"reconciler:side-effects:{status}",
    }

    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"
    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "IDEMPOTENT"

    assert calls["gauges"] == 1
    assert calls["events"] == 1
    assert calls["audits"] == 1
    assert calls["failures"] == (1 if status == "failed" else 0)
    assert calls["failure_codes"] == (1 if status == "failed" else 0)
    assert calls["cancelled"] == (1 if status == "cancelled" else 0)


def test_reconciler_terminalizer_rejects_wrong_correlation_without_numeric_fallback(
    monkeypatch,
    tmp_path,
):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = _create_reconciler_job(jm, key="reconciler-correlation")
    generic = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )
    base = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "expected_status": "queued",
        "status": "failed",
        "error_code": "generation_correlation_mismatch",
        "error_message": "generation correlation mismatch",
        "completion_token": "reconciler:correlation:v1",
        "job_id": int(job["id"]),
    }

    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{**base, "job_uuid": "00000000-0000-0000-0000-000000000000"},
        )
        == "MISSING"
    )
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{**base, "owner_user_id": "owner-2"},
        )
        == "CONFLICT"
    )
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{**base, "job_id": int(job["id"]) + 1},
        )
        == "CONFLICT"
    )
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{**base, "expected_status": "processing"},
        )
        == "CONFLICT"
    )
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{
                **base,
                "job_uuid": str(generic["uuid"]),
                "job_id": int(generic["id"]),
            },
        )
        == "CONFLICT"
    )
    assert jm.get_job(int(job["id"]))["status"] == "queued"


def test_reconciler_processing_terminalizer_can_require_no_live_lease(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = _create_reconciler_job(jm, key="reconciler-expired-lease")
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    arguments = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "expected_status": "processing",
        "status": "failed",
        "error_code": "generation_expired",
        "error_message": "generation input expired",
        "completion_token": "reconciler:expired-lease:v1",
        "job_id": int(job["id"]),
        "require_processing_lease_expired": True,
    }

    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "CONFLICT"
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        conn.execute(
            "UPDATE jobs SET leased_until='2000-01-01 00:00:00' WHERE id=?",
            (int(job["id"]),),
        )
        conn.commit()
    assert jm.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"

    live_job = _create_reconciler_job(jm, key="reconciler-live-lease-allowed")
    live_acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert live_acquired is not None
    assert int(live_acquired["id"]) == int(live_job["id"])
    assert (
        jm.terminalize_slides_generation_job_from_reconciler(
            **{
                **arguments,
                "job_uuid": str(live_job["uuid"]),
                "job_id": int(live_job["id"]),
                "completion_token": "reconciler:force-live-lease:v1",
                "require_processing_lease_expired": False,
            },
        )
        == "APPLIED"
    )


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"expected_status": "completed"}, "expected status"),
        ({"status": "completed"}, "terminal status"),
        ({"error_code": "INVALID CODE"}, "error_code"),
        ({"error_message": "x" * 1025}, "error_message"),
        ({"completion_token": ""}, "correlation"),
        ({"completion_token": "   "}, "correlation"),
        ({"job_uuid": ""}, "correlation"),
        ({"owner_user_id": ""}, "correlation"),
    ],
)
def test_reconciler_terminalizer_validates_closed_inputs(monkeypatch, tmp_path, changed, message):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = _create_reconciler_job(jm, key="reconciler-validation")
    arguments = {
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "expected_status": "queued",
        "status": "failed",
        "error_code": "generation_failed",
        "error_message": "generation failed",
        "completion_token": "reconciler:failure:v1",
        "job_id": int(job["id"]),
    }

    with pytest.raises(ValueError, match=message):
        jm.terminalize_slides_generation_job_from_reconciler(**{**arguments, **changed})
