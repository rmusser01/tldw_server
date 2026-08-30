import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs import pg_migrations as jobs_pg_migrations
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.pg_migrations import (
    _configure_pg_archive_migration_session,
    _ensure_pg_archive_locators,
    ensure_jobs_tables_pg,
)

pytestmark = [pytest.mark.pg_jobs]

_POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS = {
    "idx_jobs_archive_lookup_id": ("id", "archive_id DESC"),
    "idx_jobs_archive_batch_group_scope": (
        "batch_group",
        "domain",
        "owner_user_id",
        "job_type",
        "archive_id DESC",
    ),
}


def test_pg_schema_persists_owner_scoped_idempotency_receipts(jobs_pg_dsn):
    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'job_idempotency_receipts'"
            )
            columns = {row[0] for row in cur.fetchall()}
            assert columns == {
                "receipt_id",
                "domain",
                "queue",
                "job_type",
                "owner_user_id",
                "key_digest",
                "request_fingerprint",
                "operation_scope",
                "job_uuid",
                "job_id",
                "created_at",
                "expires_at",
            }
            assert not {"idempotency_key", "raw_key", "client_key"} & columns

            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = current_schema() "
                "AND tablename = 'job_idempotency_receipts'"
            )
            index_definitions = {row[0]: row[1] for row in cur.fetchall()}
            assert (
                "(domain, queue, job_type, owner_user_id, key_digest)"
                in index_definitions[
                    "idx_job_idempotency_receipts_owner_key"
                ]
            )
            assert "(job_uuid)" in index_definitions[
                "idx_job_idempotency_receipts_job_uuid"
            ]
            assert "(job_id)" in index_definitions[
                "idx_job_idempotency_receipts_job_id"
            ]
            assert (
                "(operation_scope, owner_user_id, expires_at)"
                in index_definitions["idx_job_idempotency_receipts_scope"]
            )


def test_pg_forward_migration_adds_missing_columns_and_partial_indexes(jobs_pg_dsn):


    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Try to drop a new-ish column to simulate an older schema
            try:
                cur.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS progress_message")
            except Exception:
                _ = None
            cur.execute("DROP INDEX IF EXISTS idx_jobs_archive_lookup_id")
            cur.execute(
                "DROP INDEX IF EXISTS idx_jobs_archive_batch_group_scope"
            )

    # Run ensure to forward-migrate
    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            # Column should exist now
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='jobs' AND column_name='progress_message'")
            row = cur.fetchone()
            assert row is not None
            # idx_jobs_acquire_order partial index exists and is queued-only
            cur.execute("""
                SELECT indexname, indexdef FROM pg_indexes
                WHERE schemaname = current_schema() AND tablename = 'jobs' AND indexname = 'idx_jobs_acquire_order'
            """)
            row2 = cur.fetchone()
            assert row2 is not None
            assert "status = 'queued'" in (row2[1] or "")
            archive_index_states = {
                index_name: _read_pg_archive_batch_index_state(
                    cur, index_name
                )
                for index_name in _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS
            }

    for index_name, expected_columns in (
        _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS.items()
    ):
        state = archive_index_states[index_name]
        assert state is not None
        assert state[:8] == (
            True,
            True,
            True,
            False,
            True,
            True,
            "btree",
            len(expected_columns),
        )
        assert tuple(state[8]) == tuple(
            column.removesuffix(" DESC") for column in expected_columns
        )
        assert tuple(state[9]) == tuple(
            3 if column.endswith(" DESC") else 0
            for column in expected_columns
        )


def test_pg_forward_migration_backfills_execution_controls(jobs_pg_dsn):
    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS expired_lease_policy")
            cur.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS quarantine_threshold")
            cur.execute(
                "ALTER TABLE jobs DROP COLUMN IF EXISTS "
                "prepared_disposition_fingerprint"
            )
            cur.execute(
                "ALTER TABLE jobs DROP COLUMN IF EXISTS "
                "no_attempt_recovery_fingerprint"
            )
            for column in (
                "expired_lease_policy",
                "quarantine_threshold",
                "prepared_disposition_fingerprint",
                "no_attempt_recovery_fingerprint",
            ):
                cur.execute(
                    f"ALTER TABLE jobs_archive DROP COLUMN IF EXISTS {column}"
                )
            cur.execute(
                "INSERT INTO jobs(uuid, domain, queue, job_type, payload, status) "
                "VALUES('legacy-controls', 'legacy', 'default', 'work', '{}'::jsonb, 'queued')"
            )
            cur.execute(
                "INSERT INTO jobs_archive(uuid, domain, queue, job_type, payload, status) "
                "VALUES('legacy-archive-controls', 'legacy', 'default', 'work', "
                "'{}'::jsonb, 'completed')"
            )

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT expired_lease_policy, quarantine_threshold, "
                "prepared_disposition_fingerprint, no_attempt_recovery_fingerprint "
                "FROM jobs "
                "WHERE uuid='legacy-controls'"
            )
            assert cur.fetchone() == ("consume_retry", None, None, None)
            cur.execute(
                "SELECT expired_lease_policy, quarantine_threshold, "
                "prepared_disposition_fingerprint, no_attempt_recovery_fingerprint "
                "FROM jobs_archive WHERE uuid='legacy-archive-controls'"
            )
            assert cur.fetchone() == ("consume_retry", None, None, None)
            cur.execute("SAVEPOINT invalid_policy")
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "UPDATE jobs SET expired_lease_policy='invalid' "
                    "WHERE uuid='legacy-controls'"
                )
            cur.execute("ROLLBACK TO SAVEPOINT invalid_policy")
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "UPDATE jobs SET quarantine_threshold=0 "
                    "WHERE uuid='legacy-controls'"
                )
            cur.execute("ROLLBACK TO SAVEPOINT invalid_policy")
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "UPDATE jobs_archive SET prepared_disposition_fingerprint='invalid' "
                    "WHERE uuid='legacy-archive-controls'"
                )
            cur.execute("ROLLBACK TO SAVEPOINT invalid_policy")
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "UPDATE jobs SET no_attempt_recovery_fingerprint='invalid' "
                    "WHERE uuid='legacy-controls'"
                )

    ensure_jobs_tables_pg(jobs_pg_dsn)


def _read_pg_archive_batch_index_state(cur, index_name):
    cur.execute(
        "SELECT i.indrelid = 'jobs_archive'::regclass, i.indisvalid, "
        "i.indisready, i.indisunique, i.indpred IS NULL, "
        "i.indexprs IS NULL, am.amname, i.indnkeyatts, "
        "ARRAY(SELECT pg_get_indexdef(i.indexrelid, key_position, true) "
        "FROM generate_series(1, i.indnkeyatts) AS key_position "
        "ORDER BY key_position), "
        "ARRAY(SELECT i.indoption[key_position - 1]::integer "
        "FROM generate_series(1, i.indnkeyatts) AS key_position "
        "ORDER BY key_position) "
        "FROM pg_class idx "
        "JOIN pg_namespace ns ON ns.oid = idx.relnamespace "
        "JOIN pg_index i ON i.indexrelid = idx.oid "
        "JOIN pg_am am ON am.oid = idx.relam "
        "WHERE ns.nspname = current_schema() AND idx.relname = %s",
        (index_name,),
    )
    return cur.fetchone()


@pytest.mark.parametrize(
    ("index_name", "misdefined_ddl"),
    (
        (
            "idx_jobs_archive_lookup_id",
            "CREATE INDEX idx_jobs_archive_lookup_id "
            "ON jobs_archive(id, archive_id)",
        ),
        (
            "idx_jobs_archive_batch_group_scope",
            "CREATE INDEX idx_jobs_archive_batch_group_scope "
            "ON jobs_archive(domain, batch_group, owner_user_id, job_type, "
            "archive_id)",
        ),
    ),
)
def test_pg_archive_batch_read_index_migration_repairs_misdefined_index(
    jobs_pg_dsn,
    index_name,
    misdefined_ddl,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP INDEX {index_name}")
            cur.execute(misdefined_ddl)

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            state = _read_pg_archive_batch_index_state(cur, index_name)

    assert state is not None
    assert state[:8] == (
        True,
        True,
        True,
        False,
        True,
        True,
        "btree",
        len(_POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS[index_name]),
    )
    expected_columns = _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS[index_name]
    assert tuple(state[8]) == tuple(
        column.removesuffix(" DESC") for column in expected_columns
    )
    assert tuple(state[9]) == tuple(
        3 if column.endswith(" DESC") else 0 for column in expected_columns
    )


@pytest.mark.parametrize(
    "index_name",
    tuple(_POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS),
)
def test_pg_archive_batch_read_index_migration_rejects_name_collision(
    jobs_pg_dsn,
    index_name,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP INDEX {index_name}")
            cur.execute("CREATE TABLE archive_batch_index_name_owner (id INTEGER)")
            cur.execute(
                f"CREATE INDEX {index_name} "
                "ON archive_batch_index_name_owner(id)"
            )

    with pytest.raises(RuntimeError, match=f"{index_name} belongs to another table"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            state = _read_pg_archive_batch_index_state(cur, index_name)

    assert state is not None
    assert state[0] is False


class _ArchiveBatchReadIndexCursor:
    def __init__(self, states, *, repair_succeeds=True):
        self.states = dict(states)
        self.repair_succeeds = repair_succeeds
        self.calls = []
        self._selected_index = None

    def execute(self, query, params=None):
        rendered = str(query)
        self.calls.append((rendered, params))
        if "SELECT i.indrelid = 'jobs_archive'::regclass" in rendered:
            self._selected_index = str(params[0])
        elif rendered.startswith("DROP INDEX CONCURRENTLY"):
            index_name = rendered.rsplit(" ", 1)[-1]
            self.states[index_name] = None
        elif rendered.startswith("CREATE INDEX CONCURRENTLY"):
            index_name = next(
                name for name in self.states if name in rendered
            )
            columns = _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS[index_name]
            column_names = tuple(
                column.removesuffix(" DESC") for column in columns
            )
            options = tuple(
                3 if column.endswith(" DESC") else 0 for column in columns
            )
            valid = self.repair_succeeds
            self.states[index_name] = (
                True,
                valid,
                valid,
                False,
                True,
                True,
                "btree",
                len(columns),
                column_names,
                options,
                None,
            )

    def fetchone(self):
        return self.states[self._selected_index]


def _ready_pg_archive_batch_index_state(index_name):
    columns = _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS[index_name]
    column_names = tuple(column.removesuffix(" DESC") for column in columns)
    options = tuple(3 if column.endswith(" DESC") else 0 for column in columns)
    return (
        True,
        True,
        True,
        False,
        True,
        True,
        "btree",
        len(columns),
        column_names,
        options,
        None,
    )


@pytest.mark.parametrize("bad_state", ("invalid", "misdefined"))
def test_pg_archive_batch_read_index_mock_repairs_bad_state_once(bad_state):
    bad_index = "idx_jobs_archive_lookup_id"
    states = {
        index_name: _ready_pg_archive_batch_index_state(index_name)
        for index_name in _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS
    }
    if bad_state == "invalid":
        states[bad_index] = (
            True,
            False,
            False,
            False,
            True,
            True,
            "btree",
            2,
            ("id", "archive_id"),
            (0, 3),
            None,
        )
    else:
        states[bad_index] = (
            True,
            True,
            True,
            False,
            True,
            True,
            "btree",
            2,
            ("archive_id", "id"),
            (3, 0),
            None,
        )
    cursor = _ArchiveBatchReadIndexCursor(states)

    jobs_pg_migrations._ensure_pg_archive_batch_read_indexes(cursor)

    drop_calls = [
        query for query, _ in cursor.calls if query.startswith("DROP INDEX")
    ]
    create_calls = [
        query for query, _ in cursor.calls if query.startswith("CREATE INDEX")
    ]
    assert drop_calls == [f"DROP INDEX CONCURRENTLY {bad_index}"]
    assert len(create_calls) == 1
    assert bad_index in create_calls[0]
    assert cursor.states[bad_index] == _ready_pg_archive_batch_index_state(
        bad_index
    )


def test_pg_archive_batch_read_index_mock_rejects_failed_repair():
    bad_index = "idx_jobs_archive_lookup_id"
    states = {
        index_name: _ready_pg_archive_batch_index_state(index_name)
        for index_name in _POSTGRES_ARCHIVE_BATCH_READ_INDEX_COLUMNS
    }
    states[bad_index] = (
        True,
        False,
        False,
        False,
        True,
        True,
        "btree",
        2,
        ("id", "archive_id"),
        (0, 3),
        None,
    )
    cursor = _ArchiveBatchReadIndexCursor(states, repair_succeeds=False)

    with pytest.raises(RuntimeError, match=f"{bad_index} verification failed"):
        jobs_pg_migrations._ensure_pg_archive_batch_read_indexes(cursor)

    assert sum(
        query.startswith("DROP INDEX") for query, _ in cursor.calls
    ) == 1
    assert sum(
        query.startswith("CREATE INDEX") for query, _ in cursor.calls
    ) == 1


def test_pg_legacy_archive_migration_backfills_stable_locators_and_paginates(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute(
                """
                CREATE TABLE jobs_archive (
                    id INTEGER,
                    uuid TEXT,
                    domain TEXT NOT NULL,
                    queue TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    payload JSONB,
                    result JSONB,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ,
                    archived_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            for version in ("first", "second", "third"):
                cur.execute(
                    "INSERT INTO jobs_archive "
                    "(id, uuid, domain, queue, job_type, payload, status, "
                    "created_at, archived_at) "
                    "VALUES (9, NULL, 'prompt_studio', 'default', "
                    "'optimization', %s::jsonb, 'cancelled', NULL, %s)",
                    (
                        '{"version":"' + version + '"}',
                        datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            cur.execute(
                "CREATE INDEX idx_jobs_archive_id ON jobs_archive(id)"
            )

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT archive_id FROM jobs_archive ORDER BY archive_id"
            )
            locators = [row[0] for row in cur.fetchall()]
            cur.execute(
                "SELECT column_default, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'jobs_archive' "
                "AND column_name = 'archive_id'"
            )
            default, nullable = cur.fetchone()
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = current_schema() "
                "AND tablename = 'jobs_archive'"
            )
            indexes = {row[0] for row in cur.fetchall()}
            cur.execute(
                "SELECT i.indisunique, i.indisvalid, "
                "pg_get_indexdef(i.indexrelid) "
                "FROM pg_class idx "
                "JOIN pg_namespace ns ON ns.oid = idx.relnamespace "
                "JOIN pg_index i ON i.indexrelid = idx.oid "
                "WHERE ns.nspname = current_schema() "
                "AND idx.relname = 'idx_jobs_archive_id'"
            )
            archive_index_state = cur.fetchone()

    assert len(locators) == len(set(locators)) == 3
    assert all(locator is not None for locator in locators)
    assert "nextval" in str(default)
    assert nullable == "NO"
    assert "idx_jobs_archive_id" in indexes
    assert archive_index_state is not None
    assert archive_index_state[:2] == (True, True)
    assert "(archive_id)" in str(archive_index_state[2])
    assert "idx_jobs_archive_migration" in indexes
    assert "idx_jobs_archive_cursor_v2" in indexes
    assert "idx_jobs_archive_lookup_id" in indexes
    assert "idx_jobs_archive_batch_group_scope" in indexes

    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    seen: list[int] = []
    cursor: dict[str, object] = {}
    while True:
        page = manager.list_archived_jobs(
            domain="prompt_studio",
            job_type="optimization",
            status="cancelled",
            limit=1,
            **cursor,
        )
        if not page:
            break
        row = page[0]
        seen.append(int(row["_archive_locator"]))
        cursor = {
            "created_before": row["_archive_cursor_created_at"],
            "before_id": int(row["id"]),
            "before_uuid": str(row["_archive_cursor_uuid"]),
            "before_archive_locator": row["_archive_locator"],
        }

    assert sorted(seen) == sorted(int(locator) for locator in locators)


def test_pg_steady_state_ensure_does_not_reset_sequence_during_concurrent_insert(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status) "
                "VALUES (40, 'seed-archive', 'prompt_studio', 'default', "
                "'optimization', '{}'::jsonb, 'cancelled') "
                "RETURNING archive_id"
            )
            seed_locator = int(cur.fetchone()[0])
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_before = cur.fetchone()

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_after = cur.fetchone()

    assert sequence_state_after == sequence_state_before

    start = threading.Barrier(2)

    def _ensure_concurrently() -> None:
        start.wait(timeout=5)
        ensure_jobs_tables_pg(jobs_pg_dsn)

    def _insert_concurrently() -> int:
        start.wait(timeout=5)
        with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO jobs_archive "
                    "(id, uuid, domain, queue, job_type, payload, status) "
                    "VALUES (41, 'concurrent-archive', 'prompt_studio', "
                    "'default', 'optimization', '{}'::jsonb, 'cancelled') "
                    "RETURNING archive_id"
                )
                return int(cur.fetchone()[0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        ensure_future = executor.submit(_ensure_concurrently)
        insert_future = executor.submit(_insert_concurrently)
        concurrent_locator = insert_future.result(timeout=10)
        ensure_future.result(timeout=10)

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status) "
                "VALUES (42, 'after-concurrent-archive', 'prompt_studio', "
                "'default', 'optimization', '{}'::jsonb, 'cancelled') "
                "RETURNING archive_id"
            )
            following_locator = int(cur.fetchone()[0])

    assert concurrent_locator > seed_locator
    assert following_locator > concurrent_locator


def test_pg_ensure_repairs_archive_sequence_behind_existing_max(jobs_pg_dsn):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(archive_id, id, uuid, domain, queue, job_type, payload, "
                "status) VALUES (100, 100, 'explicit-high-locator', "
                "'prompt_studio', 'default', 'optimization', '{}'::jsonb, "
                "'cancelled')"
            )
            cur.execute(
                "SELECT setval("
                "'jobs_archive_archive_id_seq'::regclass, 1, false)"
            )
            cur.execute(
                "SELECT contype FROM pg_constraint "
                "WHERE conrelid = 'jobs_archive'::regclass "
                "AND conname = 'idx_jobs_archive_id'"
            )
            constraint_type_before = cur.fetchone()[0]

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status) "
                "VALUES (101, 'after-sequence-repair', 'prompt_studio', "
                "'default', 'optimization', '{}'::jsonb, 'cancelled') "
                "RETURNING archive_id"
            )
            repaired_locator = int(cur.fetchone()[0])
            cur.execute(
                "SELECT contype FROM pg_constraint "
                "WHERE conrelid = 'jobs_archive'::regclass "
                "AND conname = 'idx_jobs_archive_id'"
            )
            constraint_type_after = cur.fetchone()[0]

    assert repaired_locator > 100
    assert constraint_type_before == constraint_type_after == "p"


def test_pg_archive_locator_migration_rejects_cross_table_index_name_collision(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute("CREATE TABLE archive_index_name_owner (id INTEGER)")
            cur.execute(
                "CREATE INDEX idx_jobs_archive_id "
                "ON archive_index_name_owner(id)"
            )
            cur.execute(
                "CREATE TABLE jobs_archive ("
                "id INTEGER, uuid TEXT, domain TEXT NOT NULL, "
                "queue TEXT NOT NULL, job_type TEXT NOT NULL, payload JSONB, "
                "result JSONB, status TEXT NOT NULL, created_at TIMESTAMPTZ, "
                "archived_at TIMESTAMPTZ DEFAULT NOW())"
            )

    with pytest.raises(RuntimeError, match="belongs to another table"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT i.indrelid = 'archive_index_name_owner'::regclass "
                "FROM pg_class idx JOIN pg_index i ON i.indexrelid = idx.oid "
                "WHERE idx.relname = 'idx_jobs_archive_id'"
            )
            index_still_owned_by_other_table = cur.fetchone()[0]
            cur.execute(
                "SELECT to_regclass('jobs_archive_archive_id_seq')"
            )
            archive_sequence = cur.fetchone()[0]

    assert index_still_owned_by_other_table is True
    assert archive_sequence is None


def test_pg_archive_locator_migration_preserves_misdefined_constraint(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute(
                "CREATE TABLE jobs_archive ("
                "archive_id BIGINT, id INTEGER, uuid TEXT, "
                "domain TEXT NOT NULL, queue TEXT NOT NULL, "
                "job_type TEXT NOT NULL, payload JSONB, result JSONB, "
                "status TEXT NOT NULL, created_at TIMESTAMPTZ, "
                "archived_at TIMESTAMPTZ DEFAULT NOW(), "
                "CONSTRAINT idx_jobs_archive_id PRIMARY KEY (id))"
            )

    with pytest.raises(RuntimeError, match="misdefined constraint-backed"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT contype, pg_get_constraintdef(oid) "
                "FROM pg_constraint WHERE conrelid = 'jobs_archive'::regclass "
                "AND conname = 'idx_jobs_archive_id'"
            )
            constraint_state = cur.fetchone()
            cur.execute(
                "SELECT to_regclass('jobs_archive_archive_id_seq')"
            )
            archive_sequence = cur.fetchone()[0]

    assert constraint_state is not None
    assert constraint_state[0] == "p"
    assert "PRIMARY KEY (id)" in str(constraint_state[1])
    assert archive_sequence is None


def test_pg_archive_locator_migration_widens_legacy_integer_column(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute(
                "CREATE TABLE jobs_archive ("
                "archive_id INTEGER, id INTEGER, uuid TEXT, "
                "domain TEXT NOT NULL, queue TEXT NOT NULL, "
                "job_type TEXT NOT NULL, payload JSONB, result JSONB, "
                "status TEXT NOT NULL, created_at TIMESTAMPTZ, "
                "archived_at TIMESTAMPTZ DEFAULT NOW())"
            )
            cur.execute(
                "INSERT INTO jobs_archive "
                "(archive_id, id, domain, queue, job_type, status) "
                "VALUES (2147483647, 1, 'prompt_studio', 'default', "
                "'optimization', 'cancelled')"
            )
            cur.execute(
                "CREATE UNIQUE INDEX idx_jobs_archive_id "
                "ON jobs_archive(archive_id)"
            )

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data_type, is_nullable FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'jobs_archive' "
                "AND column_name = 'archive_id'"
            )
            column_state = cur.fetchone()
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, domain, queue, job_type, status) VALUES "
                "(2, 'prompt_studio', 'default', 'optimization', 'cancelled') "
                "RETURNING archive_id"
            )
            next_locator = int(cur.fetchone()[0])

    assert column_state == ("bigint", "NO")
    assert next_locator > 2147483647


def test_pg_archive_locator_migration_preserves_sequence_owned_elsewhere(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute("CREATE TABLE archive_sequence_owner (id BIGINT)")
            cur.execute("CREATE SEQUENCE jobs_archive_archive_id_seq")
            cur.execute(
                "ALTER SEQUENCE jobs_archive_archive_id_seq "
                "OWNED BY archive_sequence_owner.id"
            )
            cur.execute(
                "ALTER TABLE archive_sequence_owner ALTER COLUMN id "
                "SET DEFAULT nextval('jobs_archive_archive_id_seq'::regclass)"
            )
            cur.execute(
                "CREATE TABLE jobs_archive ("
                "id INTEGER, uuid TEXT, domain TEXT NOT NULL, "
                "queue TEXT NOT NULL, job_type TEXT NOT NULL, payload JSONB, "
                "result JSONB, status TEXT NOT NULL, created_at TIMESTAMPTZ, "
                "archived_at TIMESTAMPTZ DEFAULT NOW())"
            )
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_before = cur.fetchone()

    with pytest.raises(RuntimeError, match="owned by another"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_get_serial_sequence("
                "'archive_sequence_owner', 'id')"
            )
            sequence_owner = cur.fetchone()[0]
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_after = cur.fetchone()
            cur.execute(
                "INSERT INTO archive_sequence_owner DEFAULT VALUES "
                "RETURNING id"
            )
            owner_locator = int(cur.fetchone()[0])

    assert str(sequence_owner).endswith("jobs_archive_archive_id_seq")
    assert sequence_state_after == sequence_state_before
    assert owner_locator == 1


def test_pg_archive_locator_migration_rejects_duplicates_before_sequence_use(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute("CREATE SEQUENCE jobs_archive_archive_id_seq")
            cur.execute(
                "CREATE TABLE jobs_archive ("
                "archive_id BIGINT, id INTEGER, uuid TEXT, "
                "domain TEXT NOT NULL, queue TEXT NOT NULL, "
                "job_type TEXT NOT NULL, payload JSONB, result JSONB, "
                "status TEXT NOT NULL, created_at TIMESTAMPTZ, "
                "archived_at TIMESTAMPTZ DEFAULT NOW())"
            )
            cur.execute(
                "ALTER SEQUENCE jobs_archive_archive_id_seq "
                "OWNED BY jobs_archive.archive_id"
            )
            cur.execute(
                "ALTER TABLE jobs_archive ALTER COLUMN archive_id "
                "SET DEFAULT nextval('jobs_archive_archive_id_seq'::regclass)"
            )
            cur.execute(
                "INSERT INTO jobs_archive "
                "(archive_id, id, domain, queue, job_type, status) VALUES "
                "(7, 1, 'prompt_studio', 'default', 'optimization', "
                "'cancelled'), "
                "(7, 2, 'prompt_studio', 'default', 'optimization', "
                "'cancelled')"
            )
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_before = cur.fetchone()

    with pytest.raises(RuntimeError, match="duplicate archive_id"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_after = cur.fetchone()
            cur.execute(
                "SELECT COUNT(*) FROM jobs_archive WHERE archive_id = 7"
            )
            duplicate_count = int(cur.fetchone()[0])

    assert sequence_state_after == sequence_state_before
    assert duplicate_count == 2


def test_pg_archive_locator_migration_rejects_ambiguous_existing_sequence(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE jobs_archive")
            cur.execute("CREATE SEQUENCE jobs_archive_archive_id_seq")
            cur.execute(
                "CREATE TABLE jobs_archive ("
                "id INTEGER, uuid TEXT, domain TEXT NOT NULL, "
                "queue TEXT NOT NULL, job_type TEXT NOT NULL, payload JSONB, "
                "result JSONB, status TEXT NOT NULL, created_at TIMESTAMPTZ, "
                "archived_at TIMESTAMPTZ DEFAULT NOW())"
            )
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_before = cur.fetchone()

    with pytest.raises(RuntimeError, match="without archive ownership"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_after = cur.fetchone()

    assert sequence_state_after == sequence_state_before


def test_pg_healthy_archive_rejects_sequence_shared_by_another_default(
    jobs_pg_dsn,
):
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE archive_sequence_consumer (id BIGINT)")
            cur.execute(
                "ALTER TABLE archive_sequence_consumer ALTER COLUMN id "
                "SET DEFAULT nextval('jobs_archive_archive_id_seq'::regclass)"
            )
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_before = cur.fetchone()

    with pytest.raises(RuntimeError, match="another table or column default"):
        ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_value, is_called "
                "FROM jobs_archive_archive_id_seq"
            )
            sequence_state_after = cur.fetchone()
            cur.execute(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'archive_sequence_consumer' "
                "AND column_name = 'id'"
            )
            consumer_default = cur.fetchone()[0]

    assert sequence_state_after == sequence_state_before
    assert "jobs_archive_archive_id_seq" in str(consumer_default)


def test_pg_archive_locator_repair_waits_for_writer_then_rechecks_sequence(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_PG_ARCHIVE_MIGRATION_LOCK_TIMEOUT_MS", "10000")
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(archive_id, id, uuid, domain, queue, job_type, payload, "
                "status) VALUES (100, 100, 'explicit-high-locator', "
                "'prompt_studio', 'default', 'optimization', '{}'::jsonb, "
                "'cancelled')"
            )
            cur.execute(
                "SELECT pg_catalog.setval("
                "'jobs_archive_archive_id_seq'::regclass, 1, false)"
            )

    blocker = psycopg.connect(jobs_pg_dsn)
    try:
        with blocker.cursor() as cur:
            cur.execute(
                "LOCK TABLE jobs_archive IN ROW EXCLUSIVE MODE"
            )
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status) "
                "VALUES (101, 'writer-before-repair', 'prompt_studio', "
                "'default', 'optimization', '{}'::jsonb, 'cancelled') "
                "RETURNING archive_id"
            )
            writer_locator = int(cur.fetchone()[0])

        with ThreadPoolExecutor(max_workers=1) as executor:
            repair_future = executor.submit(
                _ensure_pg_archive_locators,
                jobs_pg_dsn,
            )
            deadline = time.monotonic() + 5
            waiting_for_table_lock = False
            with psycopg.connect(jobs_pg_dsn, autocommit=True) as poll_conn:
                with poll_conn.cursor() as poll_cur:
                    while time.monotonic() < deadline:
                        poll_cur.execute(
                            "SELECT EXISTS(SELECT 1 FROM pg_locks "
                            "WHERE relation = 'jobs_archive'::regclass "
                            "AND mode = 'AccessExclusiveLock' AND NOT granted)"
                        )
                        waiting_for_table_lock = bool(poll_cur.fetchone()[0])
                        if waiting_for_table_lock:
                            break
                        time.sleep(0.01)
            assert waiting_for_table_lock

            blocker.commit()
            repair_future.result(timeout=10)
    finally:
        blocker.rollback()
        blocker.close()

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status) "
                "VALUES (102, 'writer-after-repair', 'prompt_studio', "
                "'default', 'optimization', '{}'::jsonb, 'cancelled') "
                "RETURNING archive_id"
            )
            repaired_locator = int(cur.fetchone()[0])

    assert writer_locator == 1
    assert repaired_locator > 100


def test_pg_archive_migration_session_overrides_request_timeouts(monkeypatch):
    monkeypatch.setenv(
        "JOBS_PG_ARCHIVE_MIGRATION_STATEMENT_TIMEOUT_MS",
        "123456",
    )
    monkeypatch.setenv("JOBS_PG_ARCHIVE_MIGRATION_LOCK_TIMEOUT_MS", "6543")

    class _RecordingCursor:
        def __init__(self):
            self.calls = []

        def execute(self, query, params):
            self.calls.append((query, params))

    cursor = _RecordingCursor()
    _configure_pg_archive_migration_session(cursor)

    assert cursor.calls == [
        (
            "SELECT set_config('statement_timeout', %s, %s)",
            ("123456ms", True),
        ),
        (
            "SELECT set_config('lock_timeout', %s, %s)",
            ("6543ms", True),
        ),
    ]


def test_pg_ensure_configures_timeouts_before_each_schema_phase(monkeypatch):
    connections = []

    class _RecordingCursor:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            self.calls.append((str(query), params))

    class _RecordingConnection:
        def __init__(self, *, autocommit):
            self.autocommit = autocommit
            self.recording_cursor = _RecordingCursor()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return self.recording_cursor

        def commit(self):
            return None

    def _connect(_dsn, *, autocommit=False):
        connection = _RecordingConnection(autocommit=autocommit)
        connections.append(connection)
        return connection

    monkeypatch.setattr(psycopg, "connect", _connect)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.pg_util.negotiate_pg_dsn",
        lambda dsn: dsn,
    )
    monkeypatch.setattr(
        jobs_pg_migrations,
        "_ensure_pg_archive_locators",
        lambda _dsn: None,
    )
    monkeypatch.setattr(
        jobs_pg_migrations,
        "_ensure_pg_archive_batch_read_indexes",
        lambda _cursor: None,
        raising=False,
    )
    monkeypatch.setattr(
        jobs_pg_migrations,
        "_mark_slides_audit_failure_pg",
        lambda _cursor: None,
    )
    monkeypatch.setattr(
        jobs_pg_migrations,
        "_audit_slides_generation_pg",
        lambda _cursor: (None, 0),
    )
    monkeypatch.setattr(
        jobs_pg_migrations,
        "ensure_job_events_pg",
        lambda _dsn: None,
    )
    monkeypatch.setattr(
        jobs_pg_migrations,
        "ensure_job_counters_pg",
        lambda _dsn: None,
    )
    monkeypatch.delenv("JOBS_PG_RLS_ENABLE", raising=False)

    jobs_pg_migrations.ensure_jobs_tables_pg("postgresql://jobs.test/jobs")

    configured_connections = [
        connection
        for connection in connections
        if connection.recording_cursor.calls
        and "set_config('statement_timeout'" in connection.recording_cursor.calls[0][0]
    ]
    assert [connection.autocommit for connection in configured_connections] == [
        False,
        True,
        True,
    ]
    expected_local = (True, False, False)
    for connection, local in zip(
        configured_connections,
        expected_local,
        strict=True,
    ):
        first_calls = connection.recording_cursor.calls[:2]
        assert first_calls == [
            (
                "SELECT set_config('statement_timeout', %s, %s)",
                ("300000ms", local),
            ),
            (
                "SELECT set_config('lock_timeout', %s, %s)",
                ("30000ms", local),
            ),
        ]
