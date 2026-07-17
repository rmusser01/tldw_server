from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

pytestmark = [pytest.mark.pg_jobs]
UTC = timezone.utc
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def test_postgres_migration_adds_archive_indexes_shared_tables_and_narrow_uuid_constraint(
    jobs_pg_dsn,
):
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


def test_postgres_duplicate_archive_uuid_is_diagnosed_without_breaking_generic_jobs(jobs_pg_dsn):
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
