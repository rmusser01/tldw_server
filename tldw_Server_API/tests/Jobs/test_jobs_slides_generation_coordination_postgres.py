from __future__ import annotations

import inspect
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.pg_migrations import (
    ensure_jobs_rls_policies_pg,
    ensure_jobs_tables_pg,
    slides_archive_indexes_ready_pg,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


@pytest.mark.pg_jobs
def test_postgres_migration_adds_archive_indexes_shared_tables_and_narrow_uuid_constraint(
    jobs_pg_dsn,
):
    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_jobs_archive_slides_scope")
        cur.execute("DROP INDEX IF EXISTS idx_jobs_archive_uuid_unique")
        cur.execute("CREATE INDEX idx_jobs_archive_slides_scope ON jobs_archive(uuid)")
        cur.execute("CREATE UNIQUE INDEX idx_jobs_archive_uuid_unique ON jobs_archive(id)")

    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    assert manager.get_slides_generation_readiness()["archive_indexes_ready"] is False
    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexes.indexname, indexes.indexdef,
                   index_state.indisvalid, index_state.indisready, index_state.indisunique
            FROM pg_indexes AS indexes
            JOIN pg_class AS index_class ON index_class.relname=indexes.indexname
            JOIN pg_index AS index_state ON index_state.indexrelid=index_class.oid
            WHERE indexes.schemaname=current_schema()
              AND indexes.indexname IN (
                'idx_jobs_archive_slides_scope', 'idx_jobs_archive_uuid_unique'
              )
            """
        )
        indexes = {row[0]: (" ".join(row[1].lower().split()), row[2], row[3], row[4]) for row in cur.fetchall()}
        assert "idx_jobs_archive_slides_scope" in indexes
        assert (
            "(domain, queue, job_type, idempotency_key, owner_user_id, archived_at desc)"
            in indexes["idx_jobs_archive_slides_scope"][0]
        )
        assert "where (idempotency_key is not null)" in indexes["idx_jobs_archive_slides_scope"][0]
        assert indexes["idx_jobs_archive_slides_scope"][1:3] == (True, True)
        assert "idx_jobs_archive_uuid_unique" in indexes
        assert "unique index" in indexes["idx_jobs_archive_uuid_unique"][0]
        assert "where (uuid is not null)" in indexes["idx_jobs_archive_uuid_unique"][0]
        assert indexes["idx_jobs_archive_uuid_unique"][1:] == (True, True, True)

        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema=current_schema()
              AND table_name IN ('slides_standalone_key_registry', 'slides_standalone_reconciliation')
            """
        )
        assert {row[0] for row in cur.fetchall()} == {
            "slides_standalone_key_registry",
            "slides_standalone_reconciliation",
        }

        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=current_schema() AND table_name='slides_standalone_key_registry'
            """
        )
        registry_columns = {row[0] for row in cur.fetchall()}
        assert registry_columns == {
            "key_id",
            "state",
            "activated_at",
            "retired_at",
            "config_revision",
        }
        assert not any("secret" in column or "digest" in column for column in registry_columns)

        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO jobs (uuid, domain, queue, job_type, status)
                VALUES (NULL, 'slides', 'default', 'presentation.generate', 'queued')
                """
            )
        cur.execute(
            """
            INSERT INTO jobs (uuid, domain, queue, job_type, status)
            VALUES (NULL, 'unrelated', 'default', 'legacy', 'queued')
            """
        )
        cur.execute(
            """
            INSERT INTO jobs (uuid, domain, queue, job_type, status)
            VALUES ('immutable-active', 'slides', 'default', 'presentation.generate', 'queued')
            """
        )
        with pytest.raises(psycopg.Error):
            cur.execute("UPDATE jobs SET uuid='replacement' WHERE uuid='immutable-active'")
        cur.execute(
            """
            INSERT INTO jobs_archive (
                id, uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, status, archived_at
            ) VALUES (
                2, 'immutable-archive', 'slides', 'default', 'presentation.generate',
                'owner-1', 'immutable-archive', 'completed', %s
            )
            """,
            (NOW,),
        )
        with pytest.raises(psycopg.Error):
            cur.execute("UPDATE jobs_archive SET uuid='replacement' WHERE uuid='immutable-archive'")
        for table, job_uuid in (("jobs", "immutable-active"), ("jobs_archive", "immutable-archive")):
            for column in ("domain", "queue", "job_type"):
                with pytest.raises(psycopg.Error):
                    cur.execute(
                        psycopg.sql.SQL("UPDATE {} SET {}='other' WHERE uuid=%s").format(
                            psycopg.sql.Identifier(table),
                            psycopg.sql.Identifier(column),
                        ),
                        (job_uuid,),
                    )


@pytest.mark.pg_jobs
def test_postgres_duplicate_archive_uuid_is_diagnosed_without_breaking_generic_jobs(
    jobs_pg_dsn,
    monkeypatch,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_jobs_archive_uuid_unique")
        cur.execute(
            """
            INSERT INTO jobs_archive (
                id, uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, status, archived_at
            ) VALUES
                (1, 'duplicate-uuid', 'slides', 'default', 'presentation.generate',
                 'owner-1', 'idem-a', 'completed', %s),
                (2, 'duplicate-uuid', 'slides', 'default', 'presentation.generate',
                 'owner-1', 'idem-b', 'completed', %s)
            """,
            (NOW, NOW),
        )

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs_archive WHERE uuid='duplicate-uuid'")
        assert cur.fetchone()[0] == 2
        cur.execute(
            """
            SELECT diagnostic_code, diagnostic_count, diagnostic_at
            FROM slides_standalone_reconciliation WHERE singleton_id=1
            """
        )
        diagnostic = cur.fetchone()
        assert diagnostic[0] == "duplicate_archive_uuid"
        assert diagnostic[1] >= 2
        assert diagnostic[2] is not None
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname=current_schema() AND indexname='idx_jobs_archive_uuid_unique'"
        )
        assert cur.fetchone() is None

    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    assert jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )["uuid"]
    assert jm.get_slides_generation_readiness()["diagnostic_code"] == "duplicate_archive_uuid"

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, payload, status, completed_at
            ) VALUES (
                'duplicate-uuid', 'slides', 'default', 'presentation.generate',
                'owner-3', 'idem-c', '{"new":true}', 'completed', %s
            )
            """,
            (NOW - timedelta(days=60),),
        )
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    with pytest.raises(ValueError, match="unsafe presentation.generate archive collision"):
        jm.prune_jobs(older_than_days=1, domain="slides")
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs WHERE uuid='duplicate-uuid'")
        assert cur.fetchone()[0] == 1


@pytest.mark.pg_jobs
def test_postgres_forward_migration_adds_archive_terminal_projection_before_audit(
    jobs_pg_dsn,
):
    removed_columns = (
        "batch_group",
        "completion_token",
        "failure_streak_code",
        "failure_streak_count",
        "quarantined_at",
        "request_id",
        "trace_id",
        "failure_timeline",
        "error_code",
    )
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        for column in removed_columns:
            cur.execute(
                psycopg.sql.SQL("ALTER TABLE jobs_archive DROP COLUMN IF EXISTS {}").format(
                    psycopg.sql.Identifier(column)
                )
            )

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=current_schema() AND table_name='jobs_archive'
            """
        )
        assert set(removed_columns) <= {row[0] for row in cur.fetchall()}
        cur.execute("SELECT diagnostic_code FROM slides_standalone_reconciliation WHERE singleton_id=1")
        assert cur.fetchone()[0] is None


@pytest.mark.pg_jobs
def test_postgres_reconciliation_takeover_and_revision_fencing_matches_sqlite(jobs_pg_dsn):
    first = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    second = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    lease = first.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW,
    )
    assert lease is not None and lease["fencing_token"] == 1
    assert (
        first.acquire_slides_reconciliation_lease(
            holder_uuid="holder-a",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=1),
        )
        is None
    )
    assert first.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        cursor="cursor-1",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=2,
        now=NOW + timedelta(seconds=2),
    )

    takeover = second.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=31),
    )
    assert takeover is not None
    assert takeover["fencing_token"] == 2
    assert takeover["cursor"] == "cursor-1"
    assert not first.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=32),
    )
    assert not first.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        cursor="stale",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(seconds=32),
    )
    assert not first.release_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=32),
    )
    assert second.release_slides_reconciliation_lease(
        holder_uuid="holder-b",
        fencing_token=2,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=70),
    )

    changed = first.acquire_slides_reconciliation_lease(
        holder_uuid="holder-c",
        lease_seconds=30,
        config_revision="revision-b",
        now=NOW + timedelta(seconds=71),
    )
    assert changed is not None
    assert changed["fencing_token"] == 3
    assert changed["cursor"] is None
    assert changed["startup_complete_epoch"] is None
    assert changed["last_complete_epoch"] is None
    assert changed["lag"] == 0


@pytest.mark.pg_jobs
def test_postgres_same_revision_takeover_invalidates_prior_fenced_sweep(jobs_pg_dsn):
    first = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    second = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    activated_at = NOW - timedelta(days=90)
    retired_at = NOW - timedelta(days=40)
    assert first.compare_and_swap_slides_current_key(
        expected_current_key_id=None,
        expected_config_revision=None,
        new_current_key_id="old-key",
        new_config_revision="revision-a",
        changed_at=activated_at,
    )
    assert first.compare_and_swap_slides_current_key(
        expected_current_key_id="old-key",
        expected_config_revision="revision-a",
        new_current_key_id="new-key",
        new_config_revision="revision-b",
        changed_at=retired_at,
    )
    lease = first.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=1,
        config_revision="revision-b",
        now=NOW,
    )
    assert lease is not None
    assert first.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-b",
        cursor=None,
        startup_complete_epoch="revision-b",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(milliseconds=500),
        completed=True,
        sweep_key_id="old-key",
        sweep_started_at=retired_at + timedelta(days=32),
        unexpired_reference_count=0,
    )
    prior_proof = first.load_slides_dormant_sweep_proof(key_id="old-key")
    assert prior_proof is not None

    takeover = second.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-b",
        now=NOW + timedelta(seconds=2),
    )
    assert takeover is not None
    assert takeover["fencing_token"] > prior_proof["fencing_token"]
    assert second.load_slides_dormant_sweep_proof(key_id="old-key") is None
    assert (
        second.compare_and_swap_remove_slides_key(
            key_id="old-key",
            expected_retired_at=retired_at,
            expected_config_revision="revision-b",
        )
        is None
    )


def test_postgres_reconciliation_lock_order_and_registry_grant_are_narrow():
    acquire_source = inspect.getsource(JobManager.acquire_slides_reconciliation_lease)
    postgres_source = acquire_source.split('conn.execute("BEGIN IMMEDIATE")', 1)[0]
    singleton_lock = postgres_source.index("WHERE singleton_id=1 FOR UPDATE")
    registry_lock = postgres_source.index("ORDER BY key_id FOR UPDATE")
    assert singleton_lock < registry_lock
    assert "SELECT DISTINCT config_revision" not in postgres_source

    prune_source = inspect.getsource(JobManager.prune_jobs)
    assert "pg_advisory_xact_lock" in prune_source
    assert "_SLIDES_GENERATION_CORRELATION_LOCK_PARTS" in prune_source

    rls_source = inspect.getsource(ensure_jobs_rls_policies_pg)
    assert "GRANT INSERT ON {}.slides_standalone_key_registry TO {}" in rls_source
    assert "GRANT INSERT ON {}.job_events TO {}" in rls_source
    assert "GRANT USAGE, SELECT ON SEQUENCE {}.job_events_id_seq TO {}" in rls_source
    assert "GRANT INSERT ON ALL TABLES" not in rls_source


def test_postgres_archive_index_shape_helper_rejects_wrong_catalog_rows():
    class StubCursor:
        def __init__(self, rows):
            self.rows = iter(rows)

        def execute(self, _query, _params):
            return None

        def fetchone(self):
            return next(self.rows)

    exact_rows = (
        (
            True,
            True,
            False,
            6,
            ["domain", "queue", "job_type", "idempotency_key", "owner_user_id", "archived_at DESC"],
            "(idempotency_key IS NOT NULL)",
        ),
        (True, True, True, 1, ["uuid"], "(uuid IS NOT NULL)"),
    )
    assert slides_archive_indexes_ready_pg(StubCursor(exact_rows)) is True
    wrong_rows = (
        (True, True, False, 1, ["uuid"], "(uuid IS NOT NULL)"),
        (True, True, True, 1, ["uuid"], "(uuid IS NOT NULL)"),
    )
    assert slides_archive_indexes_ready_pg(StubCursor(wrong_rows)) is False
    included_rows = (
        (
            True,
            True,
            False,
            7,
            ["domain", "queue", "job_type", "idempotency_key", "owner_user_id", "archived_at DESC"],
            "(idempotency_key IS NOT NULL)",
        ),
        (True, True, True, 2, ["uuid"], "(uuid IS NOT NULL)"),
    )
    assert slides_archive_indexes_ready_pg(StubCursor(included_rows)) is False


@pytest.mark.pg_jobs
def test_postgres_rls_role_gets_only_registry_insert(jobs_pg_dsn, monkeypatch):
    role = f"jobs_registry_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("JOBS_PG_RLS_ROLE", role)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    try:
        with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT has_table_privilege(%s, 'slides_standalone_key_registry', 'INSERT')",
                (role,),
            )
            assert cur.fetchone()[0] is True
            cur.execute(
                "SELECT has_table_privilege(%s, 'job_events', 'INSERT')",
                (role,),
            )
            assert cur.fetchone()[0] is True
            cur.execute(
                "SELECT has_sequence_privilege(%s, 'job_events_id_seq', 'USAGE')",
                (role,),
            )
            assert cur.fetchone()[0] is True
            cur.execute(
                "SELECT has_table_privilege(%s, 'jobs', 'INSERT')",
                (role,),
            )
            assert cur.fetchone()[0] is False
    finally:
        with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
            identifier = psycopg.sql.Identifier(role)
            cur.execute(psycopg.sql.SQL("DROP OWNED BY {}").format(identifier))
            cur.execute(psycopg.sql.SQL("REVOKE {} FROM CURRENT_USER").format(identifier))
            cur.execute(psycopg.sql.SQL("DROP ROLE {}").format(identifier))


@pytest.mark.pg_jobs
def test_postgres_exact_archive_collision_is_idempotent(jobs_pg_dsn, monkeypatch):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    payload = {"receipt_id": "receipt-1"}
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload=payload,
        owner_user_id="owner-1",
        idempotency_key="idempotent-archive",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='completed', completed_at=%s WHERE id=%s",
            (NOW - timedelta(days=60), int(job["id"])),
        )
        cur.execute(
            """
            INSERT INTO jobs_archive (
                id, uuid, domain, queue, job_type, owner_user_id, project_id,
                batch_group, idempotency_key, payload, result, status, priority,
                max_retries, retry_count, available_at, started_at, leased_until,
                lease_id, worker_id, acquired_at, error_message, last_error, error_code,
                cancel_requested_at, cancelled_at, cancellation_reason,
                completion_token, failure_streak_code, failure_streak_count,
                quarantined_at, progress_percent, progress_message, request_id,
                trace_id, failure_timeline, created_at, updated_at, completed_at
            )
            SELECT id, uuid, domain, queue, job_type, owner_user_id, project_id,
                   batch_group, idempotency_key, payload, result, status, priority,
                   max_retries, retry_count, available_at, started_at, leased_until,
                   lease_id, worker_id, acquired_at, error_message, last_error, error_code,
                   cancel_requested_at, cancelled_at, cancellation_reason,
                   completion_token, failure_streak_code, failure_streak_count,
                   quarantined_at, progress_percent, progress_message, request_id,
                   trace_id, failure_timeline, created_at, updated_at, completed_at
            FROM jobs WHERE id=%s
            """,
            (int(job["id"]),),
        )
    ensure_jobs_tables_pg(jobs_pg_dsn)
    assert manager.get_slides_generation_readiness()["ready"] is True
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    assert (
        manager.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )
        == 1
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs_archive WHERE uuid=%s", (job["uuid"],))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM jobs WHERE uuid=%s", (job["uuid"],))
        assert cur.fetchone()[0] == 0


@pytest.mark.pg_jobs
def test_postgres_archive_collision_with_different_result_stays_diagnosed(
    jobs_pg_dsn,
    monkeypatch,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    payload = {"receipt_id": "receipt-1"}
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload=payload,
        owner_user_id="owner-1",
        idempotency_key="result-collision",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status='completed', result=%s, completed_at=%s
            WHERE id=%s
            """,
            (json.dumps({"artifact": "active"}), NOW - timedelta(days=60), int(job["id"])),
        )
        cur.execute(
            """
            INSERT INTO jobs_archive (
                id, uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, payload, result, status, completed_at, archived_at
            ) VALUES (%s, %s, 'slides', 'default', 'presentation.generate',
                      'owner-1', 'result-collision', %s, %s, 'completed', %s, %s)
            """,
            (
                int(job["id"]),
                job["uuid"],
                json.dumps(payload),
                json.dumps({"artifact": "archive"}),
                NOW - timedelta(days=60),
                NOW,
            ),
        )

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    with pytest.raises(ValueError, match="unsafe presentation.generate archive collision"):
        manager.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )

    ensure_jobs_tables_pg(jobs_pg_dsn)
    readiness = manager.get_slides_generation_readiness()
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs WHERE uuid=%s", (job["uuid"],))
        assert cur.fetchone()[0] == 1


@pytest.mark.pg_jobs
def test_postgres_archive_preserves_terminal_error_projection(jobs_pg_dsn, monkeypatch):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key="terminal-error",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status='failed', error_code='provider_failed',
                error_message='safe failure', completed_at=%s
            WHERE id=%s
            """,
            (NOW - timedelta(days=60), int(job["id"])),
        )

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    assert manager.prune_jobs(older_than_days=1, domain="slides") == 1
    archived = manager.resolve_slides_generation_job(
        job_uuid=str(job["uuid"]),
        owner_user_id="owner-1",
        idempotency_key="terminal-error",
    )
    assert archived is not None
    assert archived["archived"] is True
    assert archived["status"] == "failed"
    assert archived["error_code"] == "provider_failed"


@pytest.mark.pg_jobs
@pytest.mark.parametrize(
    ("status", "event_type"),
    (("failed", "job.failed"), ("cancelled", "job.cancelled")),
)
def test_postgres_worker_terminalizer_bookkeeping_is_exactly_once(
    jobs_pg_dsn,
    monkeypatch,
    status,
    event_type,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"terminal-bookkeeping-{status}",
    )
    acquired = manager.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE job_counters SET processing_count=1
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        )
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

    assert manager.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert manager.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT processing_count FROM job_counters
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        )
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=%s AND event_type=%s",
            (int(job["id"]), event_type),
        )
        assert cur.fetchone()[0] == 1
