from __future__ import annotations

import base64
import gzip
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event

import pytest

from tldw_Server_API.app.core.Jobs import manager as jobs_manager
from tldw_Server_API.app.core.Jobs import migrations as jobs_migrations
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Slides import standalone_html_registry as registry_module

UTC = timezone.utc
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def _insert_archive(
    conn: sqlite3.Connection,
    *,
    job_id: int | None,
    job_uuid: str | None,
    owner: str | None = "owner-1",
    idempotency_key: str | None = "idem-1",
    payload: str = "{}",
    result: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO jobs_archive (
            id, uuid, domain, queue, job_type, owner_user_id,
            idempotency_key, payload, result, status, archived_at
        ) VALUES (?, ?, 'slides', 'default', 'presentation.generate', ?, ?, ?, ?, 'completed', ?)
        """,
        (job_id, job_uuid, owner, idempotency_key, payload, result, NOW.isoformat()),
    )


def _copy_job_to_archive(conn: sqlite3.Connection, *, job_id: int) -> None:
    active_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    copied_columns = [row[1] for row in conn.execute("PRAGMA table_info(jobs_archive)") if row[1] in active_columns]
    columns_sql = ", ".join(f'"{column}"' for column in copied_columns)
    conn.execute(
        f"INSERT INTO jobs_archive ({columns_sql}) "  # nosec B608 - trusted schema metadata
        f"SELECT {columns_sql} FROM jobs WHERE id=?",  # nosec B608 - trusted schema metadata
        (job_id,),
    )


def test_sqlite_migration_adds_exact_archive_indexes_shared_tables_and_narrow_uuid_trigger(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-schema.db")
    with sqlite3.connect(db_path) as conn:
        objects = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')")
        }
        assert "slides_standalone_key_registry" in objects
        assert "slides_standalone_reconciliation" in objects
        assert "idx_jobs_archive_slides_scope" in objects
        assert "idx_jobs_archive_uuid_unique" in objects
        assert "trg_jobs_slides_generation_uuid_required" in objects

        scope_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_jobs_archive_slides_scope'"
        ).fetchone()[0]
        normalized_scope = " ".join(scope_sql.split()).lower().replace("( ", "(").replace(" )", ")")
        assert "(domain, queue, job_type, idempotency_key, owner_user_id, archived_at desc)" in normalized_scope
        assert "where idempotency_key is not null" in normalized_scope

        uuid_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_jobs_archive_uuid_unique'"
        ).fetchone()[0]
        normalized_uuid = " ".join(uuid_sql.split()).lower()
        assert "unique index" in normalized_uuid
        assert "where uuid is not null" in normalized_uuid

        registry_columns = {row[1] for row in conn.execute("PRAGMA table_info(slides_standalone_key_registry)")}
        assert registry_columns == {
            "key_id",
            "state",
            "activated_at",
            "retired_at",
            "config_revision",
        }
        assert not any("secret" in column or "digest" in column for column in registry_columns)
        coordination_columns = {row[1] for row in conn.execute("PRAGMA table_info(slides_standalone_reconciliation)")}
        for expected in (
            "holder_uuid",
            "lease_expires_at",
            "fencing_token",
            "cursor",
            "config_revision",
            "startup_complete_epoch",
            "last_complete_epoch",
            "lag",
            "diagnostic_code",
            "diagnostic_count",
            "diagnostic_at",
        ):
            assert expected in coordination_columns

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO jobs (uuid, domain, queue, job_type, status)
                VALUES (NULL, 'slides', 'default', 'presentation.generate', 'queued')
                """
            )
        conn.execute(
            """
            INSERT INTO jobs (uuid, domain, queue, job_type, status)
            VALUES (NULL, 'unrelated', 'default', 'legacy', 'queued')
            """
        )


def test_sqlite_generation_uuid_is_immutable_in_active_and_archive_rows(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-uuid-immutable.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (uuid, domain, queue, job_type, status)
            VALUES ('active-uuid', 'slides', 'default', 'presentation.generate', 'queued')
            """
        )
        _insert_archive(conn, job_id=1, job_uuid="archive-uuid")

        for replacement in ("replacement", None, ""):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("UPDATE jobs SET uuid=? WHERE uuid='active-uuid'", (replacement,))
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("UPDATE jobs_archive SET uuid=? WHERE uuid='archive-uuid'", (replacement,))

        for column, replacement in (
            ("domain", "other"),
            ("queue", "other"),
            ("job_type", "other"),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE jobs SET {column}=? WHERE uuid='active-uuid'", (replacement,))
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    f"UPDATE jobs_archive SET {column}=? WHERE uuid='archive-uuid'",
                    (replacement,),
                )

        conn.execute("UPDATE jobs SET priority=6 WHERE uuid='active-uuid'")
        conn.execute("UPDATE jobs_archive SET priority=6 WHERE uuid='archive-uuid'")
        conn.execute(
            """
            INSERT INTO jobs (uuid, domain, queue, job_type, status)
            VALUES (NULL, 'unrelated', 'default', 'legacy', 'queued')
            """
        )
        conn.execute("UPDATE jobs SET uuid='' WHERE domain='unrelated'")


def test_sqlite_forward_migration_adds_archive_terminal_projection_before_audit(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-archive-forward.db")
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
    with sqlite3.connect(db_path) as conn:
        for column in removed_columns:
            conn.execute(f"ALTER TABLE jobs_archive DROP COLUMN {column}")

    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        archive_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs_archive)")}
        assert set(removed_columns) <= archive_columns
        diagnostic = conn.execute(
            "SELECT diagnostic_code FROM slides_standalone_reconciliation WHERE singleton_id=1"
        ).fetchone()[0]
        assert diagnostic is None


def test_sqlite_incomplete_archive_projection_fails_generation_readiness(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-archive-incomplete.db")
    manager = JobManager(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE jobs_archive DROP COLUMN error_code")

    readiness = manager.get_slides_generation_readiness()

    assert readiness["ready"] is False
    assert readiness["archive_projection_ready"] is False
    with pytest.raises(ValueError, match="Jobs coordination is unavailable"):
        manager.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={},
            owner_user_id="owner-1",
            idempotency_key="schema-incomplete",
        )
    assert manager.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )["uuid"]


def test_sqlite_failed_archive_forward_alter_persists_fail_closed_diagnostic(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-archive-alter-failure.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE jobs_archive DROP COLUMN error_code")

    real_connect = sqlite3.connect
    injected_failures: list[tuple[str, str]] = []

    class FailingArchiveAlterConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(str(sql).lower().split())
            if normalized == "alter table jobs_archive add column error_code text":
                injected_failures.append((normalized, "OperationalError"))
                raise sqlite3.OperationalError("injected archive ALTER failure")
            try:
                return self._inner.execute(sql, *args, **kwargs)
            except sqlite3.Error as exc:
                injected_failures.append((normalized, type(exc).__name__))
                raise

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        jobs_migrations.sqlite3,
        "connect",
        lambda *args, **kwargs: FailingArchiveAlterConnection(real_connect(*args, **kwargs)),
    )

    ensure_jobs_tables(db_path)

    with real_connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs_archive)")}
        diagnostic = conn.execute(
            "SELECT diagnostic_code FROM slides_standalone_reconciliation WHERE singleton_id=1"
        ).fetchone()[0]
    assert "error_code" not in columns
    assert diagnostic == "ambiguous_generation_legacy_row", injected_failures[-4:]


def test_sqlite_schema_audit_holds_immediate_writer_lock_before_scans(tmp_path, monkeypatch):
    db_path = ensure_jobs_tables(tmp_path / "slides-audit-lock.db")
    statements: list[str] = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(jobs_migrations.sqlite3, "connect", traced_connect)
    ensure_jobs_tables(db_path)

    normalized = [" ".join(statement.upper().split()) for statement in statements]
    begin_index = normalized.index("BEGIN IMMEDIATE")
    first_audit_index = next(
        index for index, statement in enumerate(normalized) if "SELECT COALESCE(SUM(CANDIDATE_COUNT), 0)" in statement
    )
    assert begin_index < first_audit_index


def test_sqlite_audit_exception_persists_fail_closed_diagnostic(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-audit-exception.db")
    manager = JobManager(db_path)
    assert manager.get_slides_generation_readiness()["ready"] is True
    diagnostic_seen_during_audit: list[str | None] = []

    def fail_audit(conn):
        diagnostic_seen_during_audit.append(
            conn.execute(
                "SELECT diagnostic_code FROM slides_standalone_reconciliation " "WHERE singleton_id=1"
            ).fetchone()[0]
        )
        raise RecursionError("malformed legacy JSON nesting")

    monkeypatch.setattr(
        jobs_migrations,
        "_audit_and_index_slides_generation",
        fail_audit,
    )
    ensure_jobs_tables(db_path)

    readiness = manager.get_slides_generation_readiness()
    assert diagnostic_seen_during_audit == ["ambiguous_generation_legacy_row"]
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"
    assert manager.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )["uuid"]


@pytest.mark.parametrize("divergent", (False, True))
def test_sqlite_audit_compares_logical_compressed_archive_projection(
    tmp_path,
    divergent,
):
    db_path = ensure_jobs_tables(tmp_path / f"slides-compressed-audit-{divergent}.db")
    manager = JobManager(db_path)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key=f"compressed-audit-{divergent}",
    )
    with sqlite3.connect(db_path) as conn:
        _copy_job_to_archive(conn, job_id=int(job["id"]))
        payload = (
            '{"receipt_id":"different"}'
            if divergent
            else conn.execute(
                "SELECT payload FROM jobs WHERE id=?",
                (int(job["id"]),),
            ).fetchone()[0]
        )
        payload_compressed = "gzip64:" + base64.b64encode(gzip.compress(payload.encode("utf-8"))).decode("ascii")
        if divergent:
            conn.execute(
                "UPDATE jobs SET payload=NULL WHERE id=?",
                (int(job["id"]),),
            )
        conn.execute(
            """
            UPDATE jobs_archive
            SET payload=NULL, payload_compressed=?
            WHERE uuid=?
            """,
            (payload_compressed, str(job["uuid"])),
        )

    ensure_jobs_tables(db_path)
    readiness = manager.get_slides_generation_readiness()

    assert readiness["ready"] is (not divergent)
    assert readiness["diagnostic_code"] == ("ambiguous_generation_legacy_row" if divergent else None)


def test_sqlite_wrong_definition_archive_indexes_fail_readiness_and_are_repaired(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-index-repair.db")
    jm = JobManager(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_jobs_archive_slides_scope")
        conn.execute("DROP INDEX idx_jobs_archive_uuid_unique")
        conn.execute(
            """
            CREATE INDEX idx_jobs_archive_slides_scope
            ON jobs_archive(uuid) WHERE uuid IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_jobs_archive_uuid_unique
            ON jobs_archive(id) WHERE id IS NOT NULL
            """
        )
        conn.commit()

    assert jm.get_slides_generation_readiness()["archive_indexes_ready"] is False
    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        scope_columns = [row[2] for row in conn.execute("PRAGMA index_info(idx_jobs_archive_slides_scope)")]
        uuid_columns = [row[2] for row in conn.execute("PRAGMA index_info(idx_jobs_archive_uuid_unique)")]
        scope_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='idx_jobs_archive_slides_scope'").fetchone()[
            0
        ]
        uuid_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='idx_jobs_archive_uuid_unique'").fetchone()[0]
    assert scope_columns == [
        "domain",
        "queue",
        "job_type",
        "idempotency_key",
        "owner_user_id",
        "archived_at",
    ]
    assert uuid_columns == ["uuid"]
    assert "where idempotency_key is not null" in " ".join(scope_sql.lower().split())
    assert "where uuid is not null" in " ".join(uuid_sql.lower().split())
    assert jm.get_slides_generation_readiness()["archive_indexes_ready"] is True

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_jobs_archive_uuid_unique")
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_jobs_archive_uuid_unique
            ON jobs_archive(uuid COLLATE NOCASE) WHERE uuid IS NOT NULL
            """
        )
        conn.commit()
    assert jm.get_slides_generation_readiness()["archive_indexes_ready"] is False
    ensure_jobs_tables(db_path)
    assert jm.get_slides_generation_readiness()["archive_indexes_ready"] is True


def test_sqlite_duplicate_archive_uuid_is_diagnosed_without_deduping_or_breaking_jobs(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-duplicate.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_jobs_archive_uuid_unique")
        _insert_archive(conn, job_id=1, job_uuid="duplicate-uuid", idempotency_key="idem-a")
        _insert_archive(conn, job_id=2, job_uuid="duplicate-uuid", idempotency_key="idem-b")
        conn.commit()

    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs_archive WHERE uuid='duplicate-uuid'").fetchone()[0] == 2
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_jobs_archive_uuid_unique'"
            ).fetchone()
            is None
        )
        diagnostic = conn.execute(
            """
            SELECT diagnostic_code, diagnostic_count, diagnostic_at
            FROM slides_standalone_reconciliation WHERE singleton_id=1
            """
        ).fetchone()
        assert diagnostic[0] == "duplicate_archive_uuid"
        assert diagnostic[1] >= 2
        assert diagnostic[2]

    jm = JobManager(db_path)
    unrelated = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )
    assert unrelated["uuid"]
    readiness = jm.get_slides_generation_readiness()
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "duplicate_archive_uuid"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              uuid, domain, queue, job_type, owner_user_id, idempotency_key,
              payload, status, completed_at
            ) VALUES (
              'duplicate-uuid', 'slides', 'default', 'presentation.generate',
              'owner-3', 'idem-c', '{"new":true}', 'completed', '2000-01-01 00:00:00'
            )
            """
        )
        conn.commit()
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
    with pytest.raises(ValueError, match="unsafe presentation.generate archive collision"):
        jm.prune_jobs(older_than_days=1, domain="slides")
    with sqlite3.connect(db_path) as conn:
        archived_payloads = conn.execute(
            """
            SELECT payload, payload_compressed FROM jobs_archive
            WHERE uuid='duplicate-uuid' ORDER BY id
            """
        ).fetchall()
        active_count = conn.execute("SELECT COUNT(*) FROM jobs WHERE uuid='duplicate-uuid'").fetchone()[0]
    assert archived_payloads == [
        ("{}", None),
        ("{}", None),
    ]
    assert active_count == 1


def test_sqlite_exact_archive_collision_is_idempotent_and_deletes_active_row(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-idempotent-archive.db")
    manager = JobManager(db_path)
    payload = {"receipt_id": "receipt-1"}
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload=payload,
        owner_user_id="owner-1",
        idempotency_key="idempotent-archive",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status='completed', completed_at='2000-01-01 00:00:00' WHERE id=?",
            (int(job["id"]),),
        )
        _copy_job_to_archive(conn, job_id=int(job["id"]))
        conn.commit()

    reopened = JobManager(db_path)
    assert reopened.get_slides_generation_readiness()["ready"] is True

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    statements: list[str] = []
    original_connect = manager._connect

    def traced_connect():
        conn = original_connect()
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(manager, "_connect", traced_connect)
    assert (
        manager.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )
        == 1
    )
    assert any(statement.strip().upper() == "BEGIN IMMEDIATE" for statement in statements)
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM jobs_archive WHERE uuid=?",
                (job["uuid"],),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE uuid=?",
                (job["uuid"],),
            ).fetchone()[0]
            == 0
        )


def test_sqlite_archive_collision_with_different_terminal_result_is_unsafe(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-result-collision.db")
    manager = JobManager(db_path)
    payload = {"receipt_id": "receipt-1"}
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload=payload,
        owner_user_id="owner-1",
        idempotency_key="result-collision",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status='completed', result='{"artifact":"active"}',
                completed_at='2000-01-01 00:00:00'
            WHERE id=?
            """,
            (int(job["id"]),),
        )
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner="owner-1",
            idempotency_key="result-collision",
            payload=json.dumps(payload),
            result='{"artifact":"archive"}',
        )
        conn.commit()

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    with pytest.raises(ValueError, match="unsafe presentation.generate archive collision"):
        manager.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE uuid=?", (job["uuid"],)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM jobs_archive WHERE uuid=?", (job["uuid"],)).fetchone()[0] == 1


def test_sqlite_unsafe_archive_collision_poisons_readiness_and_preserves_active(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-unsafe-archive.db")
    manager = JobManager(db_path)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "active"},
        owner_user_id="owner-1",
        idempotency_key="unsafe-archive",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status='completed', completed_at='2000-01-01 00:00:00' WHERE id=?",
            (int(job["id"]),),
        )
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner="owner-2",
            idempotency_key="different-correlation",
            payload='{"receipt_id":"archive"}',
        )
        conn.commit()
    assert manager.get_slides_generation_readiness()["ready"] is True

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    with pytest.raises(ValueError, match="unsafe presentation.generate archive collision"):
        manager.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )

    readiness = manager.get_slides_generation_readiness()
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE uuid=?",
                (job["uuid"],),
            ).fetchone()[0]
            == 1
        )

    reopened = JobManager(db_path)
    reopened_readiness = reopened.get_slides_generation_readiness()
    assert reopened_readiness["ready"] is False
    assert reopened_readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"


@pytest.mark.parametrize(
    ("create_owner", "create_key", "receipt_id"),
    (
        ("owner-2", "must-not-slip-through", "new"),
        ("owner-1", "unsafe-archive", "replay"),
    ),
)
def test_sqlite_unsafe_archive_poison_commits_before_waiting_create_rechecks(
    tmp_path,
    monkeypatch,
    create_owner,
    create_key,
    receipt_id,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-poison-create-race.db")
    prune_manager = JobManager(db_path)
    create_manager = JobManager(db_path)
    job = prune_manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "active"},
        owner_user_id="owner-1",
        idempotency_key="unsafe-archive",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status='completed', completed_at='2000-01-01 00:00:00' WHERE id=?",
            (int(job["id"]),),
        )
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner="other-owner",
            idempotency_key="other-key",
            payload='{"receipt_id":"archive"}',
        )
        conn.commit()

    collision_seen = Event()
    allow_poison = Event()
    create_waiting = Event()
    original_collision_check = prune_manager._idempotent_slides_archive_collisions

    def pause_on_collision(*args, **kwargs):
        try:
            return original_collision_check(*args, **kwargs)
        except ValueError:
            collision_seen.set()
            assert allow_poison.wait(timeout=5)
            raise

    class SignalingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if str(sql).strip().upper() == "BEGIN IMMEDIATE":
                create_waiting.set()
            return self._inner.execute(sql, *args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    original_create_connect = create_manager._connect
    monkeypatch.setattr(prune_manager, "_idempotent_slides_archive_collisions", pause_on_collision)
    monkeypatch.setattr(
        create_manager,
        "_connect",
        lambda: SignalingConnection(original_create_connect()),
    )
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")

    with ThreadPoolExecutor(max_workers=2) as executor:
        prune_future = executor.submit(
            prune_manager.prune_jobs,
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )
        assert collision_seen.wait(timeout=5)
        create_future = executor.submit(
            create_manager.create_job,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={"receipt_id": receipt_id},
            owner_user_id=create_owner,
            idempotency_key=create_key,
        )
        assert create_waiting.wait(timeout=5)
        allow_poison.set()
        with pytest.raises(ValueError, match="unsafe presentation.generate archive collision"):
            prune_future.result(timeout=5)
        with pytest.raises(ValueError, match="coordination is unavailable"):
            create_future.result(timeout=5)

    with sqlite3.connect(db_path) as conn:
        expected_count = int(create_key == "unsafe-archive")
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE idempotency_key=?",
                (create_key,),
            ).fetchone()[0]
            == expected_count
        )


def test_sqlite_archive_preserves_terminal_error_projection(tmp_path, monkeypatch):
    db_path = ensure_jobs_tables(tmp_path / "slides-archive-terminal-error.db")
    manager = JobManager(db_path)
    job = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key="terminal-error",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status='failed', error_code='provider_failed',
                error_message='safe failure', completed_at='2000-01-01 00:00:00'
            WHERE id=?
            """,
            (int(job["id"]),),
        )
        conn.commit()

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


def test_sqlite_two_managers_create_one_generation_correlation(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-create-race.db")
    managers = (JobManager(db_path), JobManager(db_path))
    start = Barrier(2)

    def create(manager: JobManager) -> dict:
        start.wait()
        return manager.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={"receipt_id": "receipt-1"},
            owner_user_id="owner-1",
            idempotency_key="same-correlation",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, managers))

    assert {result["uuid"] for result in results} == {results[0]["uuid"]}
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                """
            SELECT COUNT(*) FROM jobs
            WHERE domain='slides' AND queue='default'
              AND job_type='presentation.generate'
              AND owner_user_id='owner-1' AND idempotency_key='same-correlation'
            """
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    ("job_uuid", "owner", "idempotency_key"),
    [
        (None, "owner-1", "idem-1"),
        ("", "owner-1", "idem-1"),
        ("uuid-1", None, "idem-1"),
        ("uuid-1", "owner-1", None),
    ],
)
def test_sqlite_ambiguous_legacy_generation_rows_fail_only_standalone_readiness(
    tmp_path,
    job_uuid,
    owner,
    idempotency_key,
):
    db_path = ensure_jobs_tables(tmp_path / f"slides-ambiguous-{job_uuid}-{owner}-{idempotency_key}.db")
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=1,
            job_uuid=job_uuid,
            owner=owner,
            idempotency_key=idempotency_key,
        )
        conn.commit()

    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    readiness = jm.get_slides_generation_readiness()
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"
    assert readiness["diagnostic_count"] >= 1
    assert jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )["uuid"]


def test_sqlite_multiple_uuid_candidates_for_one_full_scope_are_diagnosed(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-multiple-candidates.db")
    jm = JobManager(db_path)
    active = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="same-correlation",
    )
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=int(active["id"]) + 100,
            job_uuid="different-uuid",
            owner="owner-1",
            idempotency_key="same-correlation",
        )
        conn.commit()

    ensure_jobs_tables(db_path)

    readiness = JobManager(db_path).get_slides_generation_readiness()
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"


def test_sqlite_generation_lookup_requires_uuid_owner_scope_key_and_rejects_numeric_reuse(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-lookup.db")
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"digest_key_id": "key-1"},
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
    )
    assert job["uuid"]

    found = jm.resolve_slides_generation_job(
        job_uuid=str(job["uuid"]),
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
        job_id=int(job["id"]),
    )
    assert found is not None
    assert found["uuid"] == job["uuid"]
    assert found["archived"] is False
    assert (
        jm.resolve_slides_generation_job(
            job_uuid=str(job["uuid"]),
            owner_user_id="owner-2",
            idempotency_key="idem-lookup",
            job_id=int(job["id"]),
        )
        is None
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (int(job["id"]),)).fetchone()
        columns = [description[0] for description in conn.execute("SELECT * FROM jobs LIMIT 0").description]
        values = dict(zip(columns, row))
        archive_columns = {item[1] for item in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()}
        insert_columns = [column for column in columns if column in archive_columns]
        conn.execute(
            f"INSERT INTO jobs_archive ({','.join(insert_columns)}) VALUES ({','.join('?' for _ in insert_columns)})",
            tuple(values[column] for column in insert_columns),
        )
        conn.execute("DELETE FROM jobs WHERE id=?", (int(job["id"]),))
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid="reused-numeric-id-uuid",
            owner="owner-2",
            idempotency_key="other-key",
        )
        conn.commit()

    archived = jm.resolve_slides_generation_job(
        job_uuid=str(job["uuid"]),
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
        job_id=int(job["id"]),
    )
    assert archived is not None
    assert archived["archived"] is True
    assert (
        jm.resolve_slides_generation_job(
            job_uuid="reused-numeric-id-uuid",
            owner_user_id="owner-1",
            idempotency_key="idem-lookup",
            job_id=int(job["id"]),
        )
        is None
    )
    replayed = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"digest_key_id": "key-1"},
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
    )
    assert replayed["uuid"] == job["uuid"]
    assert replayed["archived"] is True


def test_sqlite_nullable_legacy_archive_id_never_matches_expected_numeric_id(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-null-archive-id.db")
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=None,
            job_uuid="archive-without-numeric-id",
            owner="owner-1",
            idempotency_key="nullable-id",
        )
        conn.commit()
    manager = JobManager(db_path)

    assert (
        manager.resolve_slides_generation_job(
            job_uuid="archive-without-numeric-id",
            owner_user_id="owner-1",
            idempotency_key="nullable-id",
            job_id=42,
        )
        is None
    )


def test_sqlite_public_generation_lookup_is_active_first_and_needs_no_expected_uuid(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-public-lookup.db")
    jm = JobManager(db_path)
    active = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key="idem-public-lookup",
    )
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=int(active["id"]) + 100,
            job_uuid="stale-archive-uuid",
            owner="owner-1",
            idempotency_key="idem-public-lookup",
        )
        conn.commit()

    found = jm.lookup_slides_generation_job(
        owner_user_id="owner-1",
        idempotency_key="idem-public-lookup",
    )
    assert found is not None
    assert found["uuid"] == active["uuid"]
    assert found["archived"] is False
    assert (
        jm.resolve_slides_generation_job(
            job_uuid="wrong-expected-uuid",
            owner_user_id="owner-1",
            idempotency_key="idem-public-lookup",
        )
        is None
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM jobs WHERE uuid=?", (active["uuid"],))
        conn.commit()
    archived = jm.lookup_slides_generation_job(
        owner_user_id="owner-1",
        idempotency_key="idem-public-lookup",
    )
    assert archived is not None
    assert archived["uuid"] == "stale-archive-uuid"
    assert archived["archived"] is True


def test_sqlite_archived_generation_replay_precedes_quota_admission(tmp_path, monkeypatch):
    db_path = ensure_jobs_tables(tmp_path / "slides-pre-admission-replay.db")
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=1,
            job_uuid="archived-authority",
            owner="owner-1",
            idempotency_key="idem-replay",
            payload='{"receipt_id":"receipt-1"}',
        )
        conn.execute(
            """
            INSERT INTO jobs (
              uuid, domain, queue, job_type, owner_user_id,
              idempotency_key, payload, status
            ) VALUES (
              'quota-filler', 'slides', 'default', 'presentation.generate',
              'owner-1', 'different-key', '{}', 'queued'
            )
            """
        )
        conn.commit()
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_SLIDES", "1")

    replayed = JobManager(db_path).create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key="idem-replay",
    )

    assert replayed["uuid"] == "archived-authority"
    assert replayed["archived"] is True


@pytest.mark.parametrize("archived", (False, True))
def test_sqlite_generation_replay_precedes_queue_policy_rejection(
    tmp_path,
    monkeypatch,
    archived,
):
    db_path = ensure_jobs_tables(tmp_path / f"slides-queue-policy-replay-{archived}.db")
    manager = JobManager(db_path)
    original = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key="queue-policy-replay",
    )
    if archived:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status='completed', completed_at=? WHERE id=?",
                (NOW.isoformat(), int(original["id"])),
            )
            _copy_job_to_archive(conn, job_id=int(original["id"]))
            conn.execute("DELETE FROM jobs WHERE id=?", (int(original["id"]),))
    monkeypatch.setattr(manager, "_get_allowed_queues", lambda _domain: ["high"])

    replayed = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-1"},
        owner_user_id="owner-1",
        idempotency_key="queue-policy-replay",
    )

    assert replayed["uuid"] == original["uuid"]
    assert replayed["archived"] is archived


def test_sqlite_not_ready_fails_closed_only_for_exact_generation_scope(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-readiness-enforcement.db")
    jm = JobManager(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_jobs_archive_uuid_unique")
        conn.commit()

    with pytest.raises(ValueError) as lookup_error:
        jm.lookup_slides_generation_job(
            owner_user_id="owner-1",
            idempotency_key="idem-not-ready",
        )
    assert type(lookup_error.value).__name__ == "SlidesGenerationJobsUnavailableError"
    with pytest.raises(ValueError) as create_error:
        jm.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={},
            owner_user_id="owner-1",
            idempotency_key="idem-not-ready",
        )
    assert type(create_error.value).__name__ == "SlidesGenerationJobsUnavailableError"
    assert jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )["uuid"]


def test_sqlite_exact_generation_create_uses_immediate_correlation_fence(tmp_path, monkeypatch):
    db_path = ensure_jobs_tables(tmp_path / "slides-create-fence.db")
    jm = JobManager(db_path)
    statements: list[str] = []
    original_connect = jm._connect

    def traced_connect():
        conn = original_connect()
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(jm, "_connect", traced_connect)
    jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="idem-fenced-create",
    )

    assert any(statement.strip().upper() == "BEGIN IMMEDIATE" for statement in statements)


def test_sqlite_fair_share_rejection_replays_concurrent_generation_winner(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-fair-share-race.db")
    manager = JobManager(db_path)

    class RacingScheduler:
        max_per_user = 1

        def can_submit(self, owner_user_id, active_count):
            assert owner_user_id == "owner-1"
            assert active_count == 0
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        uuid, domain, queue, job_type, owner_user_id,
                        idempotency_key, payload, status
                    ) VALUES (
                        'racing-winner', 'slides', 'default',
                        'presentation.generate', 'owner-1',
                        'fair-share-race', '{}', 'queued'
                    )
                    """
                )
            return False

    monkeypatch.setattr(jobs_manager, "_fair_share_enabled", lambda: True)
    monkeypatch.setattr(jobs_manager, "_get_fair_share", RacingScheduler)
    monkeypatch.setattr(manager, "_count_active_jobs_for_user", lambda _owner: 0)

    replayed = manager.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "loser"},
        owner_user_id="owner-1",
        idempotency_key="fair-share-race",
    )

    assert replayed["uuid"] == "racing-winner"
    assert replayed["archived"] is False
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE idempotency_key='fair-share-race'").fetchone()[0] == 1


def test_sqlite_generation_idempotency_never_returns_wrong_owner_or_legacy_null_uuid(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-create-authority.db")
    jm = JobManager(db_path)
    first = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="idem-owner",
    )
    assert first["uuid"]
    with pytest.raises(ValueError):
        jm.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={},
            owner_user_id="owner-2",
            idempotency_key="idem-owner",
        )

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TRIGGER trg_jobs_slides_generation_uuid_required")
        conn.execute(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, payload, status
            ) VALUES (NULL, 'slides', 'default', 'presentation.generate', 'owner-legacy',
                      'legacy-null-winner', '{}', 'queued')
            """
        )
        conn.commit()

    legacy_manager = JobManager(db_path)
    with pytest.raises(ValueError):
        legacy_manager.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={},
            owner_user_id="owner-legacy",
            idempotency_key="legacy-null-winner",
        )


def test_sqlite_archive_compression_targets_uuid_and_skips_legacy_null_uuid(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-archive-compress.db")
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"digest_key_id": "key-1"},
        owner_user_id="owner-1",
        idempotency_key="archive-compress",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs SET status='completed', completed_at='2000-01-01 00:00:00'
            WHERE id=?
            """,
            (int(job["id"]),),
        )
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid="reused-before-prune",
            owner="owner-2",
            idempotency_key="other",
            payload='{"sentinel":true}',
        )
        conn.execute("UPDATE jobs_archive SET payload_compressed='sentinel' WHERE uuid='reused-before-prune'")
        conn.execute("DROP TRIGGER trg_jobs_slides_generation_uuid_required")
        conn.execute(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id, idempotency_key,
                payload, status, completed_at
            ) VALUES (NULL, 'slides', 'default', 'presentation.generate', 'legacy-owner',
                      'legacy-null-archive', '{"legacy":true}', 'completed', '2000-01-01 00:00:00')
            """
        )
        conn.commit()

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
    assert (
        jm.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )
        == 2
    )

    with sqlite3.connect(db_path) as conn:
        reused = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive WHERE uuid='reused-before-prune'"
        ).fetchone()
        archived = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive WHERE uuid=?",
            (job["uuid"],),
        ).fetchone()
        legacy = conn.execute(
            """
            SELECT payload, payload_compressed FROM jobs_archive
            WHERE uuid IS NULL AND idempotency_key='legacy-null-archive'
            """
        ).fetchone()
        assert reused == ('{"sentinel":true}', "sentinel")
        assert archived[0] is None
        assert archived[1].startswith("gzip64:")
        assert legacy == ('{"legacy":true}', None)
        diagnostic = conn.execute(
            "SELECT diagnostic_code FROM slides_standalone_reconciliation WHERE singleton_id=1"
        ).fetchone()[0]
        assert diagnostic == "ambiguous_generation_legacy_row"


def test_sqlite_reconciliation_lease_fences_takeover_and_preserves_progress(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-fencing.db")
    first_manager = JobManager(db_path)
    second_manager = JobManager(db_path)
    first = first_manager.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW,
    )
    assert first is not None
    assert first["fencing_token"] == 1
    assert (
        first_manager.acquire_slides_reconciliation_lease(
            holder_uuid="holder-a",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=1),
        )
        is None
    )
    assert (
        second_manager.acquire_slides_reconciliation_lease(
            holder_uuid="holder-b",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=1),
        )
        is None
    )
    assert first_manager.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        cursor="cursor-10",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=4,
        now=NOW + timedelta(seconds=2),
    )
    assert first_manager.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=3),
    )

    takeover = second_manager.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=40),
    )
    assert takeover is not None
    assert takeover["fencing_token"] == 2
    assert takeover["cursor"] == "cursor-10"
    assert not first_manager.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=41),
    )
    assert not first_manager.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        cursor="stale",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(seconds=41),
    )
    assert not first_manager.release_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=41),
    )
    assert second_manager.release_slides_reconciliation_lease(
        holder_uuid="holder-b",
        fencing_token=2,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=100),
    )
    released = second_manager.get_slides_reconciliation_state()
    assert released["holder_uuid"] is None
    assert released["fencing_token"] == 2
    assert released["cursor"] == "cursor-10"


def test_sqlite_revision_change_atomically_invalidates_readiness_cursor_and_lag(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-revision.db")
    jm = JobManager(db_path)
    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW,
    )
    assert lease is not None
    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        cursor="cursor-a",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=9,
        now=NOW + timedelta(seconds=1),
    )
    assert jm.release_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        now=NOW + timedelta(seconds=2),
    )

    changed = jm.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-b",
        now=NOW + timedelta(seconds=3),
    )
    assert changed is not None
    assert changed["fencing_token"] == lease["fencing_token"] + 1
    assert changed["config_revision"] == "revision-b"
    assert changed["cursor"] is None
    assert changed["startup_complete_epoch"] is None
    assert changed["last_complete_epoch"] is None
    assert changed["lag"] == 0


def test_sqlite_key_rotation_fences_reconciliation_and_stale_holder(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-key-revision-fence.db")
    jm = JobManager(db_path)
    initial = jm.compare_and_swap_slides_current_key(
        expected_current_key_id=None,
        expected_config_revision=None,
        new_current_key_id="key-a",
        new_config_revision="revision-a",
        changed_at=NOW,
    )
    assert initial is not None
    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=1),
    )
    assert lease is not None
    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        cursor="cursor-a",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=7,
        now=NOW + timedelta(seconds=2),
    )

    rotated = jm.compare_and_swap_slides_current_key(
        expected_current_key_id="key-a",
        expected_config_revision="revision-a",
        new_current_key_id="key-b",
        new_config_revision="revision-b",
        changed_at=NOW + timedelta(seconds=3),
    )
    assert rotated is not None and rotated["applied_here"]
    state = jm.get_slides_reconciliation_state()
    assert state["config_revision"] == "revision-b"
    assert state["fencing_token"] == lease["fencing_token"] + 1
    assert state["holder_uuid"] is None
    assert state["cursor"] is None
    assert state["startup_complete_epoch"] is None
    assert state["last_complete_epoch"] is None
    assert state["lag"] == 0
    assert not jm.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=4),
    )
    assert (
        jm.acquire_slides_reconciliation_lease(
            holder_uuid="stale-revision-holder",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=4),
        )
        is None
    )


def test_sqlite_completed_sweep_requires_and_stores_reconciler_reference_count(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-sweep-count.db")
    jm = JobManager(db_path)
    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="sweep-holder",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW,
    )
    assert lease is not None

    with pytest.raises(ValueError, match="unexpired_reference_count"):
        jm.checkpoint_slides_reconciliation(
            holder_uuid="sweep-holder",
            fencing_token=lease["fencing_token"],
            config_revision="revision-a",
            cursor=None,
            startup_complete_epoch="revision-a",
            last_complete_epoch=NOW.timestamp(),
            lag=0,
            now=NOW + timedelta(seconds=1),
            completed=True,
            sweep_key_id="old-key",
            sweep_started_at=NOW,
        )

    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="sweep-holder",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        cursor=None,
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(seconds=2),
        completed=True,
        sweep_key_id="old-key",
        sweep_started_at=NOW,
        unexpired_reference_count=3,
    )
    proof = jm.load_slides_dormant_sweep_proof(key_id="old-key")
    assert proof is not None
    assert proof["unexpired_reference_count"] == 3


def test_sqlite_same_revision_takeover_invalidates_prior_fenced_sweep(tmp_path):
    now = datetime.now(UTC) - timedelta(minutes=1)
    db_path = ensure_jobs_tables(tmp_path / "slides-sweep-takeover.db")
    first = JobManager(db_path)
    second = JobManager(db_path)
    activated_at = now - timedelta(days=90)
    retired_at = now - timedelta(days=40)
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
        now=now,
    )
    assert lease is not None
    assert first.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-b",
        cursor=None,
        startup_complete_epoch="revision-b",
        last_complete_epoch=now.timestamp(),
        lag=0,
        now=now + timedelta(milliseconds=500),
        completed=True,
        sweep_key_id="old-key",
        sweep_started_at=retired_at + timedelta(days=32),
        unexpired_reference_count=0,
    )
    prior_proof = first.load_slides_dormant_sweep_proof(key_id="old-key")
    assert prior_proof is not None
    assert prior_proof["fencing_token"] == lease["fencing_token"]

    takeover = second.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-b",
        now=now + timedelta(seconds=2),
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


def test_sqlite_key_removal_atomically_requires_current_zero_reference_proof(tmp_path):
    now = datetime.now(UTC) - timedelta(minutes=1)
    db_path = ensure_jobs_tables(tmp_path / "slides-key-removal-proof.db")
    jm = JobManager(db_path)
    activated_at = now - timedelta(days=90)
    retired_at = now - timedelta(days=40)
    assert jm.compare_and_swap_slides_current_key(
        expected_current_key_id=None,
        expected_config_revision=None,
        new_current_key_id="old-key",
        new_config_revision="revision-a",
        changed_at=activated_at,
    )
    assert jm.compare_and_swap_slides_current_key(
        expected_current_key_id="old-key",
        expected_config_revision="revision-a",
        new_current_key_id="new-key",
        new_config_revision="revision-b",
        changed_at=retired_at,
    )
    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="sweep-holder",
        lease_seconds=60,
        config_revision="revision-b",
        now=now,
    )
    assert lease is not None
    checkpoint = {
        "holder_uuid": "sweep-holder",
        "fencing_token": lease["fencing_token"],
        "config_revision": "revision-b",
        "cursor": None,
        "startup_complete_epoch": "revision-b",
        "last_complete_epoch": now.timestamp(),
        "lag": 0,
        "completed": True,
        "sweep_key_id": "old-key",
        "sweep_started_at": retired_at + timedelta(days=32),
    }
    assert jm.checkpoint_slides_reconciliation(
        **checkpoint,
        now=now + timedelta(seconds=1),
        unexpired_reference_count=1,
    )
    assert (
        jm.compare_and_swap_remove_slides_key(
            key_id="old-key",
            expected_retired_at=retired_at,
            expected_config_revision="revision-b",
        )
        is None
    )
    assert jm.checkpoint_slides_reconciliation(
        **checkpoint,
        now=now + timedelta(seconds=2),
        unexpired_reference_count=0,
    )
    removed = jm.compare_and_swap_remove_slides_key(
        key_id="old-key",
        expected_retired_at=retired_at,
        expected_config_revision="revision-b",
    )
    assert removed is not None
    assert [record["key_id"] for record in removed["records"]] == ["new-key"]


def test_sqlite_same_current_key_revision_advance_preserves_activated_at(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-same-key-revision.db")
    jm = JobManager(db_path)
    activated_at = NOW - timedelta(days=90)
    assert jm.compare_and_swap_slides_current_key(
        expected_current_key_id=None,
        expected_config_revision=None,
        new_current_key_id="same-key",
        new_config_revision="revision-a",
        changed_at=activated_at,
    )

    advanced = jm.compare_and_swap_slides_current_key(
        expected_current_key_id="same-key",
        expected_config_revision="revision-a",
        new_current_key_id="same-key",
        new_config_revision="revision-b",
        changed_at=NOW,
    )

    assert advanced is not None and advanced["applied_here"]
    current = next(record for record in advanced["state"]["records"] if record["state"] == "current")
    assert current["activated_at"] == activated_at


@pytest.mark.asyncio
async def test_job_manager_registry_adapter_uses_only_source_free_shared_state(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-registry-adapter.db")
    jm = JobManager(db_path)
    store = registry_module.JobManagerDigestKeyRegistryStore(jm)

    empty = await store.load_digest_key_registry()
    assert empty.records == ()
    activated_at = NOW - timedelta(days=90)
    initial = await store.compare_and_swap_current_key(
        expected_current_key_id=None,
        expected_config_epoch=None,
        new_current_key_id="old-key",
        new_config_epoch="revision-a",
        changed_at=activated_at,
    )
    assert initial is not None and initial.applied_here
    rotated_at = NOW - timedelta(days=33)
    rotated = await store.compare_and_swap_current_key(
        expected_current_key_id="old-key",
        expected_config_epoch="revision-a",
        new_current_key_id="new-key",
        new_config_epoch="revision-b",
        changed_at=rotated_at,
    )
    assert rotated is not None and rotated.applied_here
    assert all(record.activated_at.tzinfo == UTC for record in rotated.state.records)

    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="sweep-holder",
        lease_seconds=30,
        config_revision="revision-b",
        now=NOW,
    )
    assert lease is not None
    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="sweep-holder",
        fencing_token=lease["fencing_token"],
        config_revision="revision-b",
        cursor=None,
        startup_complete_epoch="revision-b",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(seconds=1),
        completed=True,
        sweep_key_id="old-key",
        sweep_started_at=NOW,
        unexpired_reference_count=0,
    )
    proof = await store.load_dormant_sweep_proof(key_id="old-key")
    assert proof is not None
    assert proof.complete
    assert proof.unexpired_reference_count == 0
    assert proof.sweep_started_at.tzinfo == UTC

    removed = await store.compare_and_swap_remove_key(
        key_id="old-key",
        expected_retired_at=rotated_at,
        expected_config_epoch="revision-b",
    )
    assert removed is not None
    assert [record.key_id for record in removed.records] == ["new-key"]

    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            "SELECT key_id, state, activated_at, retired_at, config_revision FROM slides_standalone_key_registry"
        ).fetchall()
        rendered = json.dumps(raw)
        assert "secret" not in rendered.lower()
        assert "hmac" not in rendered.lower()
