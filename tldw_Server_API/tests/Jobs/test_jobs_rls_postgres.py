import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    NoTransitionReason,
    OperationOutcome,
    ReleaseJobCommand,
    RenewLeaseCommand,
)
from tldw_Server_API.app.core.Jobs.operations.postgres.lifecycle import (
    release_job,
    renew_lease,
)
from tldw_Server_API.app.core.Jobs.pg_migrations import (
    ensure_jobs_rls_policies_pg,
    ensure_jobs_tables_pg,
)

pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]
RLS_NOW = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)


def _dsn_or_skip(monkeypatch):


    base_dsn = os.getenv("JOBS_DB_URL")
    if not base_dsn:
        pytest.skip("JOBS_DB_URL not configured for Postgres RLS tests")
    # Enable single-update acquire path for consistency (not strictly needed here)
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "true")
    monkeypatch.setenv("JOBS_PG_RLS_ENABLE", "true")
    role = "jobs_rls"
    monkeypatch.setenv("JOBS_PG_RLS_ROLE", role)
    password = os.getenv("JOBS_PG_RLS_PASSWORD", "jobs_rls_pw")
    monkeypatch.setenv("JOBS_PG_SKIP_SCHEMA_INIT", "true")
    # Ensure role exists with login and grants for RLS enforcement
    import psycopg
    from psycopg import sql as _sql
    with psycopg.connect(base_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            role_ident = _sql.Identifier(role)
            pwd_literal = _sql.Literal(password)
            if not cur.fetchone():
                cur.execute(_sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role_ident, pwd_literal))
            else:
                try:
                    cur.execute(_sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(role_ident, pwd_literal))
                except Exception:
                    _ = None
            cur.execute("SELECT current_schema()")
            schema_row = cur.fetchone()
            schema_name = (schema_row[0] if schema_row else None) or "public"
            cur.execute(
                _sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                    _sql.Identifier(schema_name),
                    role_ident,
                )
            )
            cur.execute(
                _sql.SQL("GRANT SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}").format(
                    _sql.Identifier(schema_name),
                    role_ident,
                )
            )

    def _with_role(dsn: str, user: str, pwd: str) -> str:
        parsed = urlparse(dsn)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        auth = quote(user)
        if pwd:
            auth = f"{auth}:{quote(pwd)}"
        netloc = f"{auth}@{host}{port}"
        return urlunparse(
            (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )

    rls_dsn = _with_role(base_dsn, role, password)
    monkeypatch.setenv("JOBS_DB_URL", rls_dsn)
    return base_dsn, rls_dsn


def _row_val(row, key, idx):
    if isinstance(row, dict):
        return row.get(key)
    return row[idx] if row is not None else None


def _seed(dsn):


    import psycopg
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Minimal cleanup to keep test deterministic
            cur.execute("DELETE FROM job_idempotency_receipts")
            cur.execute("DELETE FROM job_events")
            cur.execute("DELETE FROM jobs")
            cur.execute("DELETE FROM job_counters")
            cur.execute("DELETE FROM job_queue_controls")
            cur.execute("DELETE FROM job_sla_policies")
            # Seed jobs across domains/owners
            cur.execute(
                "INSERT INTO jobs(domain,queue,job_type,owner_user_id,status,priority,created_at) VALUES"
                "('chatbooks','default','export','u1','queued',5,NOW()),"
                "('chatbooks','default','export','u2','queued',5,NOW()),"
                "('web','crawler','fetch','u1','queued',5,NOW()),"
                "('web','crawler','fetch','u2','queued',5,NOW())"
            )
            cur.execute(
                "INSERT INTO job_queue_controls(domain,queue,paused,drain) VALUES"
                "('chatbooks','default',false,false) ON CONFLICT (domain,queue) DO NOTHING"
            )
            cur.execute(
                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES"
                "('chatbooks','default','export',2,0,0,0) ON CONFLICT (domain,queue,job_type) DO NOTHING"
            )
            cur.execute(
                "INSERT INTO job_sla_policies(domain,queue,job_type,max_queue_latency_seconds,max_duration_seconds,enabled) VALUES"
                "('chatbooks','default','export', 60, 300, true) ON CONFLICT (domain,queue,job_type) DO NOTHING"
            )
            cur.execute(
                "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,created_at) VALUES"
                "(NULL,'chatbooks','default','export','jobs.seed','{}'::jsonb,'u1',NOW()),"
                "(NULL,'web','crawler','fetch','jobs.seed','{}'::jsonb,'u2',NOW())"
            )


def test_rls_scopes_idempotency_receipt_reads_and_inserts(monkeypatch):
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM job_idempotency_receipts")
            cur.execute(
                "INSERT INTO job_idempotency_receipts "
                "(domain, queue, job_type, owner_user_id, key_digest, "
                "request_fingerprint, operation_scope, job_uuid, job_id, "
                "created_at, expires_at) VALUES "
                "('chatbooks', 'default', 'export', 'u1', %s, %s, "
                "'share:1', 'job-1', 1, NOW(), NOW() + INTERVAL '30 days'), "
                "('chatbooks', 'default', 'export', 'u2', %s, %s, "
                "'share:2', 'job-2', 2, NOW(), NOW() + INTERVAL '30 days')",
                ("a" * 64, "b" * 64, "c" * 64, "d" * 64),
            )

    manager = JobManager(backend="postgres", db_url=rls_dsn)
    JobManager.set_rls_context(
        is_admin=False,
        domain_allowlist="chatbooks",
        owner_user_id="u1",
    )
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cur:
            cur.execute(
                "SELECT owner_user_id FROM job_idempotency_receipts "
                "ORDER BY owner_user_id"
            )
            assert [row["owner_user_id"] for row in cur.fetchall()] == ["u1"]
            cur.execute(
                "INSERT INTO job_idempotency_receipts "
                "(domain, queue, job_type, owner_user_id, key_digest, "
                "request_fingerprint, operation_scope, job_uuid, job_id, "
                "created_at, expires_at) VALUES "
                "('chatbooks', 'default', 'export', 'u1', %s, %s, "
                "'share:3', 'job-3', 3, NOW(), NOW() + INTERVAL '30 days')",
                ("e" * 64, "f" * 64),
            )
        connection.commit()
    finally:
        connection.close()
        JobManager.clear_rls_context()

    with psycopg.connect(admin_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM job_idempotency_receipts "
                "WHERE owner_user_id = 'u1'"
            )
            assert cur.fetchone()[0] == 2


def _seed_processing_job(dsn: str, *, owner_user_id: str) -> int:
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                (
                    "INSERT INTO jobs(domain, queue, job_type, owner_user_id, status, priority, "
                    "leased_until, worker_id, lease_id, acquired_at, started_at, "
                    "progress_percent, progress_message, created_at) "
                    "VALUES('chatbooks', 'default', 'export', %s, 'processing', 5, %s, "
                    "'worker-1', 'lease-1', %s, %s, 10.0, 'old progress', %s) RETURNING id"
                ),
                (
                    owner_user_id,
                    RLS_NOW - timedelta(minutes=15),
                    RLS_NOW - timedelta(hours=1),
                    RLS_NOW - timedelta(hours=1),
                    RLS_NOW - timedelta(days=1),
                ),
            )
            row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _read_lifecycle_facts(dsn: str, job_id: int) -> tuple[Any, ...]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                (
                    "SELECT status, leased_until, worker_id, lease_id, acquired_at, started_at, "
                    "progress_percent, progress_message FROM jobs WHERE id = %s"
                ),
                (job_id,),
            )
            row = cur.fetchone()
    assert row is not None
    return tuple(row)


def test_rls_context_filters_results(monkeypatch):


    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed(admin_dsn)

    jm = JobManager(backend="postgres", db_url=rls_dsn)

    # Admin: see all rows (bypass)
    JobManager.set_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None)
    all_rows = jm.list_jobs()
    assert len(all_rows) >= 4

    # chatbooks:u1: see exactly one job (domain + owner)
    JobManager.set_rls_context(is_admin=False, domain_allowlist="chatbooks", owner_user_id="u1")
    cb_u1 = jm.list_jobs()
    assert len(cb_u1) == 1
    assert cb_u1[0]["domain"] == "chatbooks" and cb_u1[0]["owner_user_id"] == "u1"

    # web:u2: see exactly one
    JobManager.set_rls_context(is_admin=False, domain_allowlist="web", owner_user_id="u2")
    web_u2 = jm.list_jobs()
    assert len(web_u2) == 1
    assert web_u2[0]["domain"] == "web" and web_u2[0]["owner_user_id"] == "u2"


def test_rls_applies_to_events_and_controls(monkeypatch):


    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed(admin_dsn)
    jm = JobManager(backend="postgres", db_url=rls_dsn)

    # chatbooks:u1 context
    JobManager.set_rls_context(is_admin=False, domain_allowlist="chatbooks", owner_user_id="u1")
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            # job_events should only show chatbooks/u1 rows
            cur.execute("SELECT COUNT(*) FROM job_events")
            ev_count = int(_row_val(cur.fetchone(), "count", 0) or 0)
            assert ev_count == 1
            # job_queue_controls should only show chatbooks rows
            cur.execute("SELECT COUNT(*) FROM job_queue_controls")
            qc_count = int(_row_val(cur.fetchone(), "count", 0) or 0)
            assert qc_count == 1
            # job_sla_policies should only show chatbooks rows
            cur.execute("SELECT COUNT(*) FROM job_sla_policies")
            sla_count = int(_row_val(cur.fetchone(), "count", 0) or 0)
            assert sla_count >= 1
    finally:
        conn.close()


@pytest.mark.parametrize("operation", ["renew", "release"])
def test_rls_visible_lifecycle_operation_applies(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    job_id = _seed_processing_job(admin_dsn, owner_user_id="u1")
    manager = JobManager(backend="postgres", db_url=rls_dsn)
    JobManager.set_rls_context(
        is_admin=False,
        domain_allowlist="chatbooks",
        owner_user_id="u1",
    )
    connection = manager._connect()
    try:
        if operation == "renew":
            result = renew_lease(
                connection,
                manager._pg_cursor,
                command=RenewLeaseCommand(
                    job_id=job_id,
                    seconds=30,
                    enforce=True,
                    worker_id="worker-1",
                    lease_id="lease-1",
                    progress_percent=75.0,
                    progress_message="visible renewal",
                ),
                now=RLS_NOW,
            )
        else:
            result = release_job(
                connection,
                manager._pg_cursor,
                command=ReleaseJobCommand(
                    job_id=job_id,
                    enforce=True,
                    worker_id="worker-1",
                    lease_id="lease-1",
                ),
                counters_enabled=False,
            )
    finally:
        connection.close()
        JobManager.clear_rls_context()

    assert result.outcome is OperationOutcome.APPLIED
    assert result.row is not None
    if operation == "renew":
        assert result.row["leased_until"] == RLS_NOW + timedelta(seconds=30)
        assert result.row["progress_percent"] == 75.0
        assert result.row["progress_message"] == "visible renewal"
    else:
        assert result.row["status"] == "queued"
        assert result.row["worker_id"] is None
        assert result.row["lease_id"] is None


def test_rls_visible_release_can_insert_missing_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    job_id = _seed_processing_job(admin_dsn, owner_user_id="u1")
    manager = JobManager(backend="postgres", db_url=rls_dsn)
    JobManager.set_rls_context(
        is_admin=False,
        domain_allowlist="chatbooks",
        owner_user_id="u1",
    )
    connection = manager._connect()
    try:
        result = release_job(
            connection,
            manager._pg_cursor,
            command=ReleaseJobCommand(
                job_id=job_id,
                enforce=True,
                worker_id="worker-1",
                lease_id="lease-1",
            ),
            counters_enabled=True,
        )
    finally:
        connection.close()
        JobManager.clear_rls_context()

    assert result.outcome is OperationOutcome.APPLIED
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT ready_count, processing_count FROM job_counters "
                "WHERE domain = 'chatbooks' AND queue = 'default' AND job_type = 'export'"
            )
            counter = cursor.fetchone()
    assert counter == (1, 0)


@pytest.mark.parametrize("operation", ["renew", "release"])
def test_rls_hidden_lifecycle_operation_reports_missing_and_preserves_row(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    job_id = _seed_processing_job(admin_dsn, owner_user_id="u2")
    before = _read_lifecycle_facts(admin_dsn, job_id)
    manager = JobManager(backend="postgres", db_url=rls_dsn)
    JobManager.set_rls_context(
        is_admin=False,
        domain_allowlist="chatbooks",
        owner_user_id="u1",
    )
    connection = manager._connect()
    try:
        if operation == "renew":
            result = renew_lease(
                connection,
                manager._pg_cursor,
                command=RenewLeaseCommand(
                    job_id=job_id,
                    seconds=30,
                    enforce=False,
                    progress_percent=75.0,
                    progress_message="must remain hidden",
                ),
                now=RLS_NOW,
            )
        else:
            result = release_job(
                connection,
                manager._pg_cursor,
                command=ReleaseJobCommand(job_id=job_id, enforce=False),
                counters_enabled=False,
            )
    finally:
        connection.close()
        JobManager.clear_rls_context()

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.MISSING
    assert result.row is None
    assert _read_lifecycle_facts(admin_dsn, job_id) == before
