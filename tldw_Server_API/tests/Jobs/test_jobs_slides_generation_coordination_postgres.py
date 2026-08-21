from __future__ import annotations

import gzip
import inspect
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs import pg_migrations as jobs_pg_migrations
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import SLIDES_ARCHIVE_EXACT_FIELDS
from tldw_Server_API.app.core.Jobs.pg_migrations import (
    ensure_jobs_rls_policies_pg,
    ensure_jobs_tables_pg,
    slides_archive_indexes_ready_pg,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def test_active_owner_first_page_types_nullable_cursor_parameter(monkeypatch) -> None:
    """Keep PostgreSQL from inferring an unknown type for the first-page NULL."""

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def close(self) -> None:
            return None

    class _Cursor:
        def __init__(self) -> None:
            self.sql = ""
            self.params = ()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, sql, params) -> None:
            self.sql = " ".join(str(sql).split())
            self.params = tuple(params)

        def fetchall(self):
            return [{"owner_user_id": "owner-1"}]

    manager = object.__new__(JobManager)
    manager.backend = "postgres"
    cursor = _Cursor()
    monkeypatch.setattr(manager, "_connect", _Connection)
    monkeypatch.setattr(manager, "_pg_cursor", lambda _conn: cursor)

    assert manager.list_active_slides_generation_owner_ids(limit=1) == ["owner-1"]
    assert "CAST(%s AS TEXT) IS NULL" in cursor.sql
    assert cursor.params[3:] == (None, None, 1)


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


def test_postgres_audit_locks_before_scans_and_normalizes_compressed_rows():
    source = inspect.getsource(jobs_pg_migrations._audit_slides_generation_pg)
    ensure_source = inspect.getsource(jobs_pg_migrations.ensure_jobs_tables_pg)
    forward_block = ensure_source.split("# Forward-migrate older installs:", 1)[1].split(
        "# Audit before creating the standalone archive indexes.",
        1,
    )[0]
    audit_block = ensure_source.split(
        "# Audit before creating the standalone archive indexes.",
        1,
    )[1].split(
        "# Create hot-path indexes", 1
    )[0]

    assert source.index("FOR UPDATE") < source.index("SELECT COALESCE(SUM(candidate_count), 0)")
    assert "normalize_slides_archive_projection" in source
    assert forward_block.count("except psycopg.Error") == 1
    assert "except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS" in audit_block
    assert audit_block.index("_mark_slides_audit_failure_pg(audit_cur)") < audit_block.index(
        "_audit_slides_generation_pg(audit_cur)"
    )
    assert "SAVEPOINT slides_generation_audit" in audit_block
    assert "ROLLBACK TO SAVEPOINT slides_generation_audit" in audit_block


@pytest.mark.pg_jobs
def test_postgres_incomplete_archive_projection_fails_generation_readiness(
    jobs_pg_dsn,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE jobs_archive DROP COLUMN error_code")

    readiness = manager.get_slides_generation_readiness()

    assert readiness["ready"] is False
    assert readiness["archive_projection_ready"] is False


@pytest.mark.pg_jobs
@pytest.mark.parametrize("divergent", (False, True))
def test_postgres_audit_compares_logical_compressed_archive_projection(
    jobs_pg_dsn,
    divergent,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key=f"compressed-audit-{divergent}",
    )
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO jobs_archive ({projection}) "  # nosec B608 - closed projection
            f"SELECT {projection} FROM jobs WHERE id=%s",  # nosec B608 - closed projection
            (int(job["id"]),),
        )
        cur.execute("SELECT payload FROM jobs WHERE id=%s", (int(job["id"]),))
        stored_payload = cur.fetchone()[0]
        logical_payload = {"receipt_id": "different"} if divergent else stored_payload
        if divergent:
            cur.execute("UPDATE jobs SET payload=NULL WHERE id=%s", (int(job["id"]),))
        cur.execute(
            """
            UPDATE jobs_archive
            SET payload=NULL, payload_compressed=%s
            WHERE uuid=%s
            """,
            (
                gzip.compress(json.dumps(logical_payload).encode("utf-8")),
                str(job["uuid"]),
            ),
        )

    ensure_jobs_tables_pg(jobs_pg_dsn)
    readiness = manager.get_slides_generation_readiness()

    assert readiness["ready"] is (not divergent)
    assert readiness["diagnostic_code"] == ("ambiguous_generation_legacy_row" if divergent else None)


@pytest.mark.pg_jobs
def test_postgres_audit_holds_singleton_lock_before_scans(jobs_pg_dsn):
    scan_reached = Event()
    release_scan = Event()

    class PausingCursor:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(str(sql).split())
            if "SELECT COALESCE(SUM(candidate_count), 0)" in normalized:
                scan_reached.set()
                assert release_scan.wait(timeout=5)
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def run_audit():
        with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
            return jobs_pg_migrations._audit_slides_generation_pg(PausingCursor(cur))

    with ThreadPoolExecutor(max_workers=1) as executor:
        audit_future = executor.submit(run_audit)
        assert scan_reached.wait(timeout=5)
        with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='250ms'")
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cur.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET diagnostic_code='ambiguous_generation_legacy_row'
                    WHERE singleton_id=1
                    """
                )
            conn.rollback()
            cur.execute("SET LOCAL lock_timeout='250ms'")
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cur.execute(
                    """
                    SELECT diagnostic_code
                    FROM slides_standalone_reconciliation
                    WHERE singleton_id=1
                    FOR SHARE
                    """
                )
            conn.rollback()
        release_scan.set()
        audit_future.result(timeout=5)


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
def test_postgres_active_slides_generation_owners_match_sqlite_scope_and_order(
    jobs_pg_dsn,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    for owner, idempotency_key in (
        ("2", "pg-owner-2-a"),
        ("10", "pg-owner-10-a"),
        ("10", "pg-owner-10-b"),
    ):
        manager.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={"receipt_id": idempotency_key},
            owner_user_id=owner,
            idempotency_key=idempotency_key,
        )
    manager.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
    )
    terminal = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "pg-terminal"},
        owner_user_id="3",
        idempotency_key="pg-owner-3-terminal",
    )
    with psycopg.connect(jobs_pg_dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE jobs SET status='completed' WHERE id=%s",
            (terminal["id"],),
        )

    first = manager.list_active_slides_generation_owner_ids(limit=1)
    second = manager.list_active_slides_generation_owner_ids(
        after_owner_user_id=first[-1],
        limit=10,
    )

    assert first == ["10"]
    assert second == ["2"]


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


def test_postgres_prune_uses_one_locked_candidate_set_for_every_mutation():
    prune_source = inspect.getsource(JobManager.prune_jobs)
    batch_source = inspect.getsource(JobManager._prune_postgres_batch)

    candidate_lock = prune_source.index('f"SELECT id FROM jobs{where_clause} "')
    batch_dispatch = prune_source.index("_prune_postgres_batch")
    collision_check = batch_source.index("_idempotent_slides_archive_collisions")
    archive_copy = batch_source.index("SELECT {archive_projection} FROM locked_jobs")
    delete = batch_source.index("DELETE FROM jobs{candidate_where_clause}")

    assert candidate_lock < batch_dispatch
    assert "ORDER BY id FOR UPDATE" in prune_source
    assert "locked_ids" in prune_source
    assert collision_check < archive_copy < delete


def test_postgres_reconciler_terminalizer_serializes_before_uuid_authority_check():
    source = inspect.getsource(JobManager.terminalize_slides_generation_job_from_reconciler)

    advisory_lock = source.index("pg_advisory_xact_lock")
    readiness = source.index("_slides_generation_ready_in_connection")
    authority_check = source.index("ORDER BY id LIMIT 2 FOR UPDATE")
    update = source.index("UPDATE jobs SET status=%s")

    assert advisory_lock < readiness < authority_check < update
    assert 'conn.execute("BEGIN IMMEDIATE")' in source
    assert "len(authority_rows) > 1" in source


def test_postgres_admission_and_public_lookup_hold_serialized_readiness_lock():
    readiness_source = inspect.getsource(JobManager._slides_generation_ready_in_connection)
    lookup_source = inspect.getsource(JobManager.lookup_slides_generation_job)
    audit_source = inspect.getsource(jobs_pg_migrations._audit_slides_generation_pg)

    assert "FOR SHARE" in readiness_source
    assert "_serialized_slides_generation_replay" in lookup_source
    assert "get_slides_generation_readiness" not in lookup_source
    assert "COUNT(DISTINCT uuid)" not in audit_source


def test_postgres_archive_lookup_queries_are_bounded_by_authority():
    source = inspect.getsource(JobManager._lookup_slides_generation_job_in_connection)

    assert "ORDER BY archived_at DESC, uuid LIMIT 1" in source
    assert "AND idempotency_key=%s AND uuid=%s" in source
    assert "ORDER BY archived_at DESC, uuid LIMIT 2" in source


@pytest.mark.pg_jobs
def test_postgres_archive_lookup_selects_expected_uuid_or_newest_distinct_candidate(
    jobs_pg_dsn,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    older = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "older"},
        owner_user_id="owner-archive-authority",
        idempotency_key="archive-authority-older",
    )
    newest = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "newest"},
        owner_user_id="owner-archive-authority",
        idempotency_key="archive-authority-newest",
    )
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    shared_key = "archive-authority-shared"
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        for job, archived_at in (
            (older, NOW - timedelta(hours=1)),
            (newest, NOW),
        ):
            cur.execute(
                f"INSERT INTO jobs_archive ({projection}) "  # nosec B608 - closed projection
                f"SELECT {projection} FROM jobs WHERE id=%s",  # nosec B608 - closed projection
                (int(job["id"]),),
            )
            cur.execute(
                "UPDATE jobs_archive SET idempotency_key=%s, archived_at=%s WHERE uuid=%s",
                (shared_key, archived_at, str(job["uuid"])),
            )
            cur.execute("DELETE FROM jobs WHERE id=%s", (int(job["id"]),))

    ensure_jobs_tables_pg(jobs_pg_dsn)
    assert manager.get_slides_generation_readiness()["ready"] is True
    selected = manager.lookup_slides_generation_job(
        owner_user_id="owner-archive-authority",
        idempotency_key=shared_key,
    )
    assert selected is not None
    assert selected["uuid"] == newest["uuid"]
    expected = manager.lookup_slides_generation_job(
        owner_user_id="owner-archive-authority",
        idempotency_key=shared_key,
        expected_job_uuid=str(older["uuid"]),
        expected_job_id=int(older["id"]),
    )
    assert expected is not None
    assert expected["uuid"] == older["uuid"]
    active = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "active"},
        owner_user_id="owner-archive-authority",
        idempotency_key="archive-authority-active",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET idempotency_key=%s WHERE id=%s",
            (shared_key, int(active["id"])),
        )
    ensure_jobs_tables_pg(jobs_pg_dsn)
    active_first = manager.lookup_slides_generation_job(
        owner_user_id="owner-archive-authority",
        idempotency_key=shared_key,
    )
    assert active_first is not None
    assert active_first["uuid"] == active["uuid"]
    assert active_first["archived"] is False


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
def test_postgres_prune_row_lock_prevents_candidate_status_phantom(
    jobs_pg_dsn,
    monkeypatch,
):
    prune_manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = prune_manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "prune-lock"},
        owner_user_id="owner-1",
        idempotency_key="prune-lock",
    )
    phantom = prune_manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "must-not-enter-prune-set"},
        owner_user_id="owner-1",
        idempotency_key="prune-phantom",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status='failed', created_at=NOW() - INTERVAL '60 days',
                completed_at=NOW() - INTERVAL '60 days'
            WHERE id=%s
            """,
            (int(job["id"]),),
        )

    collision_reached = Event()
    release_collision = Event()
    original_collision_check = prune_manager._idempotent_slides_archive_collisions

    def pause_before_collision(*args, **kwargs):
        collision_reached.set()
        assert release_collision.wait(timeout=5)
        return original_collision_check(*args, **kwargs)

    monkeypatch.setattr(
        prune_manager,
        "_idempotent_slides_archive_collisions",
        pause_before_collision,
    )
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")

    with ThreadPoolExecutor(max_workers=1) as executor:
        prune_future = executor.submit(
            prune_manager.prune_jobs,
            statuses=["failed"],
            older_than_days=0,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )
        assert collision_reached.wait(timeout=5)
        try:
            with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='250ms'")
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    cur.execute(
                        "UPDATE jobs SET error_message='racing update' WHERE id=%s",
                        (int(job["id"]),),
                    )
                conn.rollback()
                cur.execute(
                    """
                    UPDATE jobs
                    SET status='failed', error_code='slides_orphaned',
                        error_message='terminalized while prune is paused',
                        completion_token='prune-phantom', completed_at=NOW()
                    WHERE id=%s AND status='queued'
                    """,
                    (int(phantom["id"]),),
                )
                assert cur.rowcount == 1
        finally:
            release_collision.set()

        assert prune_future.result(timeout=5) == 1

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs WHERE id=%s", (int(job["id"]),))
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT status FROM jobs_archive WHERE uuid=%s",
            (str(job["uuid"]),),
        )
        assert cur.fetchone()[0] == "failed"
        cur.execute("SELECT status FROM jobs WHERE uuid=%s", (str(phantom["uuid"]),))
        assert cur.fetchone()[0] == "failed"
        cur.execute(
            "SELECT COUNT(*) FROM jobs_archive WHERE uuid=%s",
            (str(phantom["uuid"]),),
        )
        assert cur.fetchone()[0] == 0


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
            "UPDATE jobs SET last_error='prior_retry' WHERE id=%s",
            (int(job["id"]),),
        )
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
        cur.execute("SELECT last_error FROM jobs WHERE id=%s", (int(job["id"]),))
        assert cur.fetchone()[0] is None


@pytest.mark.pg_jobs
def test_postgres_worker_terminalizer_accepts_reconciler_cas_winner(jobs_pg_dsn):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="worker-reconciler-race-pg",
    )
    acquired = manager.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    assert (
        manager.terminalize_slides_generation_job_from_reconciler(
            job_uuid=str(job["uuid"]),
            job_id=int(job["id"]),
            owner_user_id="owner-1",
            expected_status="processing",
            status="failed",
            error_code="generation_expired",
            error_message="Generation input expired.",
            completion_token="reconciler:pg:expiry:v1",
        )
        == "APPLIED"
    )

    assert (
        manager.terminalize_job_from_worker(
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner_user_id="owner-1",
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            worker_id="slides-worker",
            lease_id=str(acquired["lease_id"]),
            completion_token=str(acquired["lease_id"]),
            status="failed",
            error_code="slides_render_failed",
            error_message="bounded worker-safe detail",
        )
        == "ALREADY_TERMINAL"
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize(
    ("source_status", "available_at", "expected_counts"),
    (
        ("queued", None, (2, 4, 5)),
        ("queued", "future", (3, 3, 5)),
        ("processing", None, (3, 4, 4)),
    ),
)
def test_postgres_reconciler_terminalizer_bookkeeping_and_replay_match_sqlite(
    jobs_pg_dsn,
    monkeypatch,
    source_status,
    available_at,
    expected_counts,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    scheduled_at = datetime.now(UTC) + timedelta(hours=1) if available_at else None
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"reconciler-pg-{source_status}-{available_at}",
        available_at=scheduled_at,
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        if source_status == "processing":
            cur.execute(
                """
                UPDATE jobs
                SET status='processing', worker_id='slides-worker',
                    lease_id='slides-lease', leased_until=NOW() + INTERVAL '1 hour'
                WHERE id=%s
                """,
                (int(job["id"]),),
            )
        cur.execute(
            """
            INSERT INTO job_counters(
                domain, queue, job_type, ready_count, scheduled_count,
                processing_count, quarantined_count
            ) VALUES('slides', 'default', 'presentation.generate', 3, 4, 5, 0)
            ON CONFLICT(domain, queue, job_type) DO UPDATE SET
                ready_count=3, scheduled_count=4, processing_count=5
            """
        )
    arguments = {
        "job_uuid": str(job["uuid"]),
        "job_id": int(job["id"]),
        "owner_user_id": "owner-1",
        "expected_status": source_status,
        "status": "cancelled",
        "error_code": "generation_cancelled",
        "error_message": "generation cancelled",
        "completion_token": f"reconciler:pg:{source_status}:{available_at}",
    }

    assert manager.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"
    assert manager.terminalize_slides_generation_job_from_reconciler(**arguments) == "IDEMPOTENT"
    assert (
        manager.terminalize_slides_generation_job_from_reconciler(**{**arguments, "owner_user_id": "owner-2"})
        == "CONFLICT"
    )

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT ready_count, scheduled_count, processing_count
            FROM job_counters
            WHERE domain='slides' AND queue='default'
              AND job_type='presentation.generate'
            """
        )
        assert cur.fetchone() == expected_counts
        cur.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=%s AND event_type='job.cancelled'",
            (int(job["id"]),),
        )
        assert cur.fetchone()[0] == 1


@pytest.mark.pg_jobs
def test_postgres_reconciler_terminalizer_respects_live_processing_lease(
    jobs_pg_dsn,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="reconciler-pg-lease",
    )
    acquired = manager.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    arguments = {
        "job_uuid": str(job["uuid"]),
        "job_id": int(job["id"]),
        "owner_user_id": "owner-1",
        "expected_status": "processing",
        "status": "failed",
        "error_code": "generation_expired",
        "error_message": "generation input expired",
        "completion_token": "reconciler:pg:lease",
        "require_processing_lease_expired": True,
    }

    assert manager.terminalize_slides_generation_job_from_reconciler(**arguments) == "CONFLICT"
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET leased_until=NOW() - INTERVAL '1 second' WHERE id=%s",
            (int(job["id"]),),
        )
    assert manager.terminalize_slides_generation_job_from_reconciler(**arguments) == "APPLIED"


@pytest.mark.pg_jobs
def test_postgres_reconciler_terminalizer_never_mutates_duplicate_active_uuid(
    jobs_pg_dsn,
):
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE jobs DROP CONSTRAINT jobs_uuid_key")
        cur.execute(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, payload, status
            ) VALUES
                ('ambiguous-active-pg', 'slides', 'default',
                 'presentation.generate', 'owner-1', 'ambiguous-pg-a', '{}', 'queued'),
                ('ambiguous-active-pg', 'slides', 'default',
                 'presentation.generate', 'owner-1', 'ambiguous-pg-b', '{}', 'queued')
            """
        )

    with pytest.raises(ValueError, match="correlation is unsafe"):
        manager.terminalize_slides_generation_job_from_reconciler(
            job_uuid="ambiguous-active-pg",
            owner_user_id="owner-1",
            expected_status="queued",
            status="failed",
            error_code="generation_ambiguous",
            error_message="ambiguous generation correlation",
            completion_token="reconciler:ambiguous:pg",
        )

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM jobs WHERE uuid='ambiguous-active-pg' ORDER BY id")
        assert cur.fetchall() == [("queued",), ("queued",)]
        cur.execute(
            """
            SELECT diagnostic_code, diagnostic_count
            FROM slides_standalone_reconciliation WHERE singleton_id=1
            """
        )
        diagnostic = cur.fetchone()
        assert diagnostic[0] == "ambiguous_generation_legacy_row"
        assert diagnostic[1] >= 2
