import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

pytestmark = [
    pytest.mark.pg_jobs,
]


def test_pg_schema_has_aux_tables_and_indexes(jobs_pg_dsn):


    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            # Tables exist
            def table_exists(name: str) -> bool:
                cur.execute("SELECT to_regclass(%s)", (name,))
                return cur.fetchone()[0] is not None

            assert table_exists("jobs")
            assert table_exists("job_events")
            assert table_exists("job_queue_controls")
            assert table_exists("job_attachments")
            assert table_exists("job_sla_policies")
            assert table_exists("job_counters")

            # Indexes present
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = 'jobs'"
            )
            idxs = {r[0] for r in cur.fetchall()}
            assert "idx_jobs_status_available_at" in idxs
            assert "idx_jobs_idempotent_unique" in idxs
            cur.execute(
                "SELECT column_name, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='jobs' "
                "AND column_name IN ('expired_lease_policy', 'quarantine_threshold', "
                "'no_attempt_recovery_fingerprint')"
            )
            columns = {row[0]: row[1:] for row in cur.fetchall()}
            assert columns["expired_lease_policy"] == (
                "NO",
                "'consume_retry'::text",
            )
            assert columns["quarantine_threshold"] == ("YES", None)
            assert columns["no_attempt_recovery_fingerprint"] == ("YES", None)

    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('job_events')")
            assert cur.fetchone()[0] is not None
