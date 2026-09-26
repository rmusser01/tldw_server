"""Real migration and rate-query access paths for explicit retry admissions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.jobs_failed_requeue import (
    count_recent_job_admissions,
    ensure_retry_admission_index,
)
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]
pytestmark = pytest.mark.integration
USE_SHARED_JOBS_POSTGRES = True
INDEX_NAME = "idx_job_events_retry_admissions"
NOW = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


@dataclass
class IndexDatabase:
    """A native-temp SQLite database or the official isolated PostgreSQL fixture."""

    backend: str
    location: Path | str

    def connect(self) -> Any:
        """Return a real connection without altering migration or planner behavior."""
        if self.backend == "sqlite":
            return sqlite3.connect(self.location)
        import psycopg

        return psycopg.connect(self.location)

    def ensure(self) -> None:
        """Run the existing fresh/upgrade schema entry point."""
        if self.backend == "sqlite":
            ensure_jobs_tables(self.location)
        else:
            ensure_jobs_tables_pg(str(self.location))


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def database(request: pytest.FixtureRequest, tmp_path: Path) -> IndexDatabase:
    """Provision through the established migrations and official PG test fixture."""
    if request.param == "postgres":
        result = IndexDatabase("postgres", request.getfixturevalue("jobs_pg_dsn"))
        _, db_name = request.getfixturevalue("isolated_test_environment")
        assert "pg_temp_db" not in request.fixturenames, "alternate pg_temp_db was requested"
        with closing(result.connect()) as conn, conn:
            assert conn.execute("SELECT current_database()").fetchone() == (db_name,)
        assert conn.closed
        return result
    assert "pg_temp_db" not in request.fixturenames
    assert "isolated_test_environment" not in request.fixturenames
    result = IndexDatabase("sqlite", tmp_path / "jobs.db")
    result.ensure()
    return result


def index_definition(database: IndexDatabase) -> tuple[Any, ...]:
    """Inspect the actual schema, including index keys, predicate and readiness."""
    with closing(database.connect()) as conn, conn:
        with closing(conn.cursor()) as cur:
            if database.backend == "sqlite":
                cur.execute("SELECT rootpage,sql FROM sqlite_master WHERE type='index' AND name=?", (INDEX_NAME,))
                row = cur.fetchone()
                assert row is not None, "retry admission rate index is missing"
                cur.execute("PRAGMA index_info(idx_job_events_retry_admissions)")
                assert [column[2] for column in cur.fetchall()] == ["domain", "owner_user_id", "created_at"]
                assert "WHERE event_type='job.retry_admitted'" in row[1]
            else:
                cur.execute(
                    """SELECT i.indexrelid,pg_get_indexdef(i.indexrelid),
                        pg_get_expr(i.indpred,i.indrelid),i.indisvalid,i.indisready
                        FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
                        JOIN pg_namespace n ON n.oid=c.relnamespace
                        WHERE n.nspname=current_schema() AND c.relname=%s""", (INDEX_NAME,),
                )
                row = cur.fetchone()
                assert row is not None, "retry admission rate index is missing"
                assert "(domain, owner_user_id, created_at)" in row[1]
                assert row[2] == "(event_type = 'job.retry_admitted'::text)"
                assert row[3:] == (True, True)
            return row


def seed_event_history(database: IndexDatabase, *, include_retries: bool) -> None:
    """Create unrelated history and independently chosen rate-window boundaries."""
    with closing(database.connect()) as conn, conn:
        with closing(conn.cursor()) as cur:
            sql = (
                "INSERT INTO job_events(domain,owner_user_id,event_type,created_at) VALUES(?,?,?,?)"
                if database.backend == "sqlite" else
                "INSERT INTO job_events(domain,owner_user_id,event_type,created_at) VALUES(%s,%s,%s,%s)"
            )
            cur.executemany(sql, [("vn_assets", "42", "job.completed", "2010-01-01 00:00:00")] * 400)
            if include_retries:
                cur.executemany(sql, [
                    ("vn_assets", "42", "job.retry_admitted", "2026-09-25 12:00:00"),
                    ("vn_assets", "43", "job.retry_admitted", "2026-09-25 12:00:00"),
                    ("other", "42", "job.retry_admitted", "2026-09-25 12:00:00"),
                    ("vn_assets", "42", "job.retry_admitted", "2010-01-01 00:00:00"),
                ])


class CapturedQuery:
    """Record the production rate query while executing it on a real cursor."""

    def __init__(self, cursor: Any) -> None:
        """Keep the real executor and the last bound query for EXPLAIN."""
        self.cursor = cursor
        self.sql = ""
        self.params: tuple[Any, ...] = ()

    def execute(self, sql: str, params: tuple[Any, ...]) -> CapturedQuery:
        """Execute unchanged SQL and preserve its parameters."""
        self.sql, self.params = sql, params
        self.cursor.execute(sql, params)
        return self

    def fetchone(self) -> Any:
        """Return the real database result, not a synthetic rate count."""
        return self.cursor.fetchone()


def plan_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Traverse a PostgreSQL JSON plan without relying on a fixed tree shape."""
    yield node
    for child in node.get("Plans", []):
        yield from plan_nodes(child)


def test_fresh_schema_has_partial_retry_admission_index(database: IndexDatabase) -> None:
    """Fresh provisioning supports the new shared rate query immediately."""
    index_definition(database)


@pytest.mark.pg_jobs
def test_pre_events_jobs_upgrade_creates_events_before_retry_index(jobs_pg_dsn: str) -> None:
    """Current production ensure upgrades existing Jobs with no events table."""
    import psycopg

    database = IndexDatabase("postgres", jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn:
        row = conn.execute(
            """INSERT INTO jobs(domain,queue,job_type,payload,status,owner_user_id)
                VALUES('vn_assets','default','legacy','{}','failed','42') RETURNING id""",
        ).fetchone()
        assert row is not None
        job_id = row[0]
        conn.execute("DROP TABLE job_events")
        assert conn.execute("SELECT to_regclass('jobs'),to_regclass('job_events')").fetchone() == ("jobs", None)

    # Do not patch current DDL or invoke ensure_job_events_pg separately.
    assert ensure_jobs_tables_pg(jobs_pg_dsn) == jobs_pg_dsn
    first = index_definition(database)
    assert ensure_jobs_tables_pg(jobs_pg_dsn) == jobs_pg_dsn
    assert index_definition(database) == first
    with psycopg.connect(jobs_pg_dsn) as conn:
        assert conn.execute("SELECT status,owner_user_id FROM jobs WHERE id=%s", (job_id,)).fetchone() == ("failed", "42")
        assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone() == (0,)


def test_existing_event_history_upgrade_is_idempotent(database: IndexDatabase) -> None:
    """Existing installations gain the index without losing events or rebuilding it."""
    with closing(database.connect()) as conn, conn:
        conn.cursor().execute("DROP INDEX IF EXISTS idx_job_events_retry_admissions")
    seed_event_history(database, include_retries=True)
    database.ensure()
    first = index_definition(database)
    database.ensure()
    assert index_definition(database) == first
    with closing(database.connect()) as conn, conn:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) FROM job_events")
            assert cur.fetchone()[0] == 404


@pytest.mark.parametrize("include_retries", [False, True])
def test_rate_query_uses_retry_index_not_global_event_history(
    database: IndexDatabase, include_retries: bool,
) -> None:
    """The actual rate predicate has an indexed path even with zero retry events."""
    seed_event_history(database, include_retries=include_retries)
    with closing(database.connect()) as conn, conn:
        with closing(conn.cursor()) as cur:
            captured = CapturedQuery(cur)
            now = NOW.strftime("%Y-%m-%d %H:%M:%S") if database.backend == "sqlite" else NOW
            assert count_recent_job_admissions(
                captured, backend=database.backend, domain="vn_assets", owner_user_id="42", now=now,
            ) == (1 if include_retries else 0)
            if database.backend == "sqlite":
                cur.execute("EXPLAIN QUERY PLAN " + captured.sql, captured.params)
                steps = [row[3] for row in cur.fetchall()]
                assert not any("SCAN job_events" in step for step in steps), steps
                assert any(INDEX_NAME in step and "SEARCH" in step for step in steps), steps
            else:
                # Tiny fixtures may legitimately prefer a sequential scan. Show
                # index support, not a production planner or throughput claim.
                cur.execute("SET LOCAL enable_seqscan=off")
                cur.execute("EXPLAIN (FORMAT JSON) " + captured.sql, captured.params)
                nodes = list(plan_nodes(cur.fetchone()[0][0]["Plan"]))
                assert any(node.get("Index Name") == INDEX_NAME for node in nodes), nodes


@pytest.fixture
def failed_pg_retry_index(jobs_pg_dsn: str) -> Iterator[tuple[IndexDatabase, Any]]:
    """Leave a real invalid concurrent-build index behind a live writer lock."""
    import psycopg

    database = IndexDatabase("postgres", jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        conn.execute("DROP INDEX idx_job_events_retry_admissions")
    with database.connect() as blocker:
        blocker.execute(
            "INSERT INTO job_events(domain,owner_user_id,event_type) VALUES('vn_assets','42','job.completed')"
        )
        with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SET lock_timeout='1000ms'")
            cur.execute("SET statement_timeout='5000ms'")
            with pytest.raises(psycopg.errors.LockNotAvailable):
                ensure_retry_admission_index(cur, backend="postgres")
            cur.execute(
                "SELECT indisvalid,indisready FROM pg_index "
                "WHERE indexrelid='idx_job_events_retry_admissions'::regclass"
            )
            assert cur.fetchone() == (False, False), "fault must leave an actual failed-build catalog entry"
        try:
            yield database, blocker
        finally:
            blocker.rollback()


@pytest.mark.pg_jobs
def test_failed_concurrent_build_is_repaired_on_migration_retry(
    failed_pg_retry_index: tuple[IndexDatabase, Any],
) -> None:
    """A failed real build must not be accepted by IF NOT EXISTS on retry."""
    database, blocker = failed_pg_retry_index
    blocker.rollback()
    seed_event_history(database, include_retries=True)
    database.ensure()
    first = index_definition(database)
    database.ensure()
    assert index_definition(database) == first
    with closing(database.connect()) as conn, conn, conn.cursor() as cur:
        captured = CapturedQuery(cur)
        assert count_recent_job_admissions(
            captured, backend="postgres", domain="vn_assets", owner_user_id="42", now=NOW,
        ) == 1
        cur.execute("SET LOCAL enable_seqscan=off")
        cur.execute("EXPLAIN (FORMAT JSON) " + captured.sql, captured.params)
        nodes = list(plan_nodes(cur.fetchone()[0][0]["Plan"]))
        assert any(node.get("Index Name") == INDEX_NAME for node in nodes), nodes


@pytest.mark.pg_jobs
def test_failed_concurrent_repair_does_not_mark_migration_verified(
    failed_pg_retry_index: tuple[IndexDatabase, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A still-blocked repair propagates configured migration timeout failure."""
    import psycopg

    from tldw_Server_API.app.core.Jobs import pg_migrations

    database, blocker = failed_pg_retry_index
    blocker.rollback()

    def block_repair(executor: Any, *, backend: str) -> None:
        """Hold a real writer only when migration reaches its index phase."""
        blocker.execute(
            "INSERT INTO job_events(domain,owner_user_id,event_type) VALUES('vn_assets','42','job.completed')"
        )
        try:
            ensure_retry_admission_index(executor, backend=backend)
        finally:
            blocker.rollback()

    monkeypatch.setenv("JOBS_PG_ARCHIVE_MIGRATION_LOCK_TIMEOUT_MS", "1000")
    monkeypatch.setenv("JOBS_PG_ARCHIVE_MIGRATION_STATEMENT_TIMEOUT_MS", "5000")
    monkeypatch.setattr(pg_migrations, "ensure_retry_admission_index", block_repair)
    with pytest.raises(RuntimeError, match="retry-admission index migration failed") as failure:
        database.ensure()
    assert isinstance(failure.value.__cause__.__cause__, psycopg.errors.LockNotAvailable)
    blocker.rollback()
    monkeypatch.setattr(pg_migrations, "ensure_retry_admission_index", ensure_retry_admission_index)
    database.ensure()
    index_definition(database)


@pytest.mark.pg_jobs
def test_concurrent_invalid_retry_index_repairs_preserve_one_ready_index(
    failed_pg_retry_index: tuple[IndexDatabase, Any],
) -> None:
    """Concurrent initializers serialize repair rather than dropping each other's build."""
    import psycopg

    database, blocker = failed_pg_retry_index
    blocker.rollback()
    barrier = Barrier(2)

    def repair() -> tuple[Any, ...]:
        """Use independent real autocommit migration-phase connections."""
        with psycopg.connect(str(database.location), autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SET lock_timeout='5000ms'")
            cur.execute("SET statement_timeout='10000ms'")
            barrier.wait(timeout=10)
            ensure_retry_admission_index(cur, backend="postgres")
            cur.execute(
                "SELECT indexrelid,indisvalid,indisready FROM pg_index "
                "WHERE indexrelid='idx_job_events_retry_admissions'::regclass"
            )
            return cur.fetchone()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(repair) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert results[0] == results[1]
    assert results[0][1:] == (True, True)


@pytest.mark.pg_jobs
@pytest.mark.parametrize("lock_ms,statement_ms", [(250, 500), (500, 250)])
def test_retry_index_coordination_respects_configured_timeout(
    jobs_pg_dsn: str, lock_ms: int, statement_ms: int,
) -> None:
    """A competing initializer's session lock times out without touching its index."""
    import psycopg

    lock_key = "tldw.jobs.job_events.retry_admission_index.v1"
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as leader:
        leader.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (lock_key,))
        with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT set_config('lock_timeout',%s,FALSE)", (f"{lock_ms}ms",))
            cur.execute("SELECT set_config('statement_timeout',%s,FALSE)", (f"{statement_ms}ms",))
            with pytest.raises(RuntimeError, match="retry.admission.*advisory lock.*timeout"):
                ensure_retry_admission_index(cur, backend="postgres")
            leader.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (lock_key,))
            ensure_retry_admission_index(cur, backend="postgres")
    index_definition(IndexDatabase("postgres", jobs_pg_dsn))


@pytest.mark.pg_jobs
def test_disabled_retry_index_timeouts_use_bounded_thirty_second_fallback(
    jobs_pg_dsn: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real lock contention stays bounded even with both server timeouts disabled."""
    import psycopg

    from tldw_Server_API.app.core.DB_Management import jobs_failed_requeue as retry_db

    lock_key = "tldw.jobs.job_events.retry_admission_index.v1"
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as leader:
        leader.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (lock_key,))
        with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SET lock_timeout=0")
            cur.execute("SET statement_timeout=0")
            # Advance only the Python wait budget, not PostgreSQL state or clocks.
            ticks = iter([100.0, 129.99, 130.01])
            sleeps: list[float] = []
            with monkeypatch.context() as patch:
                patch.setattr(retry_db, "monotonic", ticks.__next__)
                patch.setattr(retry_db, "sleep", sleeps.append)
                with pytest.raises(RuntimeError, match="retry.admission.*advisory lock.*timeout"):
                    ensure_retry_admission_index(cur, backend="postgres")
            assert sleeps == pytest.approx([0.01])
            leader.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (lock_key,))
            ensure_retry_admission_index(cur, backend="postgres")


@pytest.mark.pg_jobs
@pytest.mark.parametrize("collision_ddl", [
    "CREATE INDEX idx_job_events_retry_admissions ON jobs(domain,owner_user_id,created_at)",
    "CREATE INDEX idx_job_events_retry_admissions ON job_events(domain,owner_user_id,created_at) "
    "WHERE event_type='job.completed'",
    "CREATE INDEX idx_job_events_retry_admissions ON job_events(domain,owner_user_id,created_at DESC) "
    "WHERE event_type='job.retry_admitted'",
    "CREATE INDEX idx_job_events_retry_admissions ON job_events(domain,owner_user_id,created_at) "
    "INCLUDE(job_id) WHERE event_type='job.retry_admitted'",
    "CREATE INDEX idx_job_events_retry_admissions ON job_events(domain text_pattern_ops,owner_user_id,created_at) "
    "WHERE event_type='job.retry_admitted'",
    "CREATE TABLE idx_job_events_retry_admissions(id INTEGER)",
])
def test_foreign_same_name_retry_index_collision_fails_closed(
    jobs_pg_dsn: str, collision_ddl: str,
) -> None:
    """Never accept or drop another object's same-name but different definition."""
    import psycopg

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX idx_job_events_retry_admissions")
        cur.execute(collision_ddl)
        cur.execute("SELECT 'idx_job_events_retry_admissions'::regclass::oid")
        original_oid = cur.fetchone()[0]
        with pytest.raises(RuntimeError, match="retry.admission.*(definition|collision)"):
            ensure_retry_admission_index(cur, backend="postgres")
        cur.execute("SELECT 'idx_job_events_retry_admissions'::regclass::oid")
        assert cur.fetchone()[0] == original_oid
