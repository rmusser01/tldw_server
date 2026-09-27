from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from sqlglot.errors import TokenError

from tldw_Server_API.app.core.AuthNZ import sqlite_profile_version_schema
from tldw_Server_API.app.core.AuthNZ.sqlite_profile_version_schema import (
    remediate_sqlite_profile_version_schema,
    validate_sqlite_profile_version_readiness,
)

pytestmark = pytest.mark.unit

_CANONICAL_PROFILE_VERSION = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"
)


def _schema_snapshot(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], ...]:
    return (
        conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall(),
        conn.execute("SELECT * FROM users ORDER BY id").fetchall(),
        conn.execute("PRAGMA user_version").fetchall(),
    )


def test_sync_remediation_canonicalizes_nullable_schema_and_restores_fk(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users (
                username, email, password_hash, updated_at, profile_version
            ) VALUES
                ('legacy', 'legacy@example.com', 'hash',
                 '2026-01-02 03:04:05.123456', NULL),
                ('canonical', 'canonical@example.com', 'hash',
                 '2026-02-03 04:05:06.654321',
                 '2026-02-03T04:05:06.654321Z');
            """
        )

        remediate_sqlite_profile_version_schema(conn)

        profile_column = {
            row[1]: row for row in conn.execute("PRAGMA table_info(users)")
        }["profile_version"]
        values = conn.execute(
            "SELECT username, profile_version FROM users ORDER BY id"
        ).fetchall()
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, updated_at)
            VALUES ('omitted', 'omitted@example.com', 'hash', CURRENT_TIMESTAMP)
            """
        )
        omitted = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'omitted'"
        ).fetchone()[0]

        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.in_transaction

    assert profile_column[2].upper() == "TEXT"
    assert profile_column[3] == 1
    assert profile_column[4] == "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')"
    assert values == [
        ("legacy", "2026-01-02T03:04:05.123456Z"),
        ("canonical", "2026-02-03T04:05:06.654321Z"),
    ]
    assert _CANONICAL_PROFILE_VERSION.fullmatch(omitted)


def test_sync_remediation_rolls_back_malformed_value_and_restores_fk(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "malformed.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users (
                username, email, password_hash, updated_at, profile_version
            ) VALUES (
                'bad', 'bad@example.com', 'hash',
                '2026-01-02 03:04:05.123456', 'not-a-timestamp'
            );
            PRAGMA user_version = 17;
            """
        )
        before = _schema_snapshot(conn)

        with pytest.raises(RuntimeError, match="invalid existing value"):
            remediate_sqlite_profile_version_schema(conn)

        assert _schema_snapshot(conn) == before
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.in_transaction


@pytest.mark.parametrize(
    "users_schema",
    [
        None,
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY
        )
        """,
    ],
)
def test_sync_remediation_fails_closed_for_missing_or_incomplete_users_schema(
    tmp_path: Path,
    users_schema: str | None,
) -> None:
    db_path = tmp_path / "incomplete.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        if users_schema is not None:
            conn.execute(users_schema)

        with pytest.raises(RuntimeError, match="profile_version|updated_at|users schema"):
            remediate_sqlite_profile_version_schema(conn)

        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.in_transaction


def test_sync_remediation_ignores_updated_at_once_profile_anchor_is_canonical(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "invalid-updated-at.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT,
                profile_version TEXT NOT NULL DEFAULT (
                    STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')
                )
            );
            INSERT INTO users (id, updated_at, profile_version)
            VALUES (1, 'not-a-timestamp', '2026-01-02T03:04:05.123456Z');
            """
        )

        remediate_sqlite_profile_version_schema(conn)

        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.in_transaction


def test_sync_remediation_rebuilds_populated_anchor_without_updated_at(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "canonical-anchor-no-updated-at.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                profile_version TEXT
            );
            INSERT INTO users (id, profile_version)
            VALUES (1, '2026-01-02T03:04:05.123456Z');
            """
        )

        remediate_sqlite_profile_version_schema(conn)

        profile_column = {
            row[1]: row for row in conn.execute("PRAGMA table_info(users)")
        }["profile_version"]
        stored = conn.execute(
            "SELECT profile_version FROM users WHERE id = 1"
        ).fetchone()[0]

    assert profile_column[3] == 1
    assert stored == "2026-01-02T03:04:05.123456Z"


def test_sync_remediation_rejects_null_anchor_without_updated_at(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "null-anchor-no-updated-at.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                profile_version TEXT
            );
            INSERT INTO users (id, profile_version) VALUES (1, NULL);
            """
        )
        before = _schema_snapshot(conn)

        with pytest.raises(RuntimeError, match="updated_at"):
            remediate_sqlite_profile_version_schema(conn)

        assert _schema_snapshot(conn) == before


def test_sync_readiness_rejects_trigger_that_writes_visible_user_field(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "unsafe-trigger.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                profile_version TEXT NOT NULL DEFAULT (
                    STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')
                )
            );
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL
            );
            CREATE TRIGGER rewrite_user_email
            AFTER INSERT ON audit_events
            BEGIN
                UPDATE users
                SET email = 'trigger@example.com'
                WHERE id = NEW.user_id;
            END;
            INSERT INTO users (id, email) VALUES (1, 'before@example.com');
            """
        )

        with pytest.raises(RuntimeError, match="unsafe.*trigger"):
            remediate_sqlite_profile_version_schema(conn)

        assert conn.execute(
            "SELECT email FROM users WHERE id = 1"
        ).fetchone()[0] == "before@example.com"


def test_sync_readiness_sanitizes_trigger_tokenizer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private trigger tokenizer detail"

    def _raise_token_error(_tokenizer: object, _sql: str) -> list[object]:
        raise TokenError(sentinel)

    monkeypatch.setattr(
        sqlite_profile_version_schema.Tokenizer,
        "tokenize",
        _raise_token_error,
    )
    with sqlite3.connect(":memory:", isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                profile_version TEXT NOT NULL DEFAULT (
                    STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')
                )
            );
            CREATE TABLE audit_events (id INTEGER PRIMARY KEY);
            CREATE TRIGGER audit_event_insert
            AFTER INSERT ON audit_events
            BEGIN
                SELECT NEW.id;
            END;
            """
        )

        with pytest.raises(RuntimeError) as raised:
            validate_sqlite_profile_version_readiness(conn)

    assert str(raised.value) == "AuthNZ profile_version found an unsafe SQLite trigger"
    assert raised.value.__cause__ is None
    assert sentinel not in str(raised.value)


def test_sync_remediation_restart_accepts_rows_with_default_anchor_and_null_updated_at(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "restart.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT,
                profile_version TEXT
            );
            INSERT INTO users VALUES (1, '2026-01-02 03:04:05', NULL);
            """
        )

        remediate_sqlite_profile_version_schema(conn)
        conn.execute("INSERT INTO users (id) VALUES (2)")
        remediate_sqlite_profile_version_schema(conn)

        rows = conn.execute(
            "SELECT updated_at, profile_version FROM users ORDER BY id"
        ).fetchall()

    assert rows[1][0] is None
    assert _CANONICAL_PROFILE_VERSION.fullmatch(rows[1][1])


@pytest.mark.parametrize(
    "updated_at",
    [
        "0001-01-01T00:00:00+14:00",
        "9999-12-31T23:59:59-14:00",
    ],
)
def test_sync_remediation_sanitizes_updated_at_utc_overflow(
    tmp_path: Path,
    updated_at: str,
) -> None:
    db_path = tmp_path / "overflow.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO users VALUES (1, ?, NULL)",
            (updated_at,),
        )

        with pytest.raises(RuntimeError, match="invalid timestamp") as raised:
            remediate_sqlite_profile_version_schema(conn)

        assert type(raised.value) is RuntimeError
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert not conn.in_transaction


def test_sync_remediation_canonical_schema_uses_validate_only_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "canonical.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                profile_version TEXT NOT NULL DEFAULT (
                    STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')
                )
            );
            INSERT INTO users (id) VALUES (1);
            """
        )
        schema_before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'users'"
        ).fetchone()[0]

        monkeypatch.setattr(
            sqlite_profile_version_schema,
            "rebuild_sqlite_users_with_profile_version",
            lambda _conn: pytest.fail("canonical schema must not be rebuilt"),
        )
        remediate_sqlite_profile_version_schema(conn)

        assert conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'users'"
        ).fetchone()[0] == schema_before
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.in_transaction


def test_sync_remediation_base_exception_rolls_back_and_restores_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StopBootstrap(BaseException):
        pass

    db_path = tmp_path / "base-exception.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users VALUES (1, '2026-01-02 03:04:05', NULL);
            """
        )

        def _interrupt(rebuild_conn: sqlite3.Connection) -> None:
            rebuild_conn.execute("CREATE TABLE partial_rebuild(id INTEGER)")
            raise _StopBootstrap()

        monkeypatch.setattr(
            sqlite_profile_version_schema,
            "rebuild_sqlite_users_with_profile_version",
            _interrupt,
        )
        with pytest.raises(_StopBootstrap):
            remediate_sqlite_profile_version_schema(conn)

        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'partial_rebuild'"
        ).fetchone() is None
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.in_transaction


def test_sync_remediation_preserves_primary_failure_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FaultingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        @property
        def in_transaction(self) -> bool:
            return self._connection.in_transaction

        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            if sql == "PRAGMA foreign_keys = 1":
                raise RuntimeError("restore failed")
            return self._connection.execute(sql, parameters)

        def rollback(self) -> None:
            raise RuntimeError("rollback failed")

        def commit(self) -> None:
            self._connection.commit()

    primary = RuntimeError("primary rebuild failure")
    db_path = tmp_path / "cleanup-failure.db"
    with sqlite3.connect(db_path, isolation_level=None) as raw_conn:
        raw_conn.execute("PRAGMA foreign_keys = ON")
        raw_conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users VALUES (1, '2026-01-02 03:04:05', NULL);
            """
        )

        def _fail_rebuild(_connection: sqlite3.Connection) -> None:
            raise primary

        monkeypatch.setattr(
            sqlite_profile_version_schema,
            "rebuild_sqlite_users_with_profile_version",
            _fail_rebuild,
        )
        with pytest.raises(RuntimeError) as raised:
            remediate_sqlite_profile_version_schema(_FaultingConnection(raw_conn))

        assert raised.value is primary
        assert sqlite_profile_version_schema.sqlite_profile_version_connection_invalid(
            raised.value
        )
        raw_conn.rollback()


def test_sync_remediation_commit_failure_after_durable_commit_is_retryable(
    tmp_path: Path,
) -> None:
    class _CommitAfterSuccessConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

        def commit(self) -> None:
            self._connection.commit()
            raise RuntimeError("commit acknowledgement failed")

    db_path = tmp_path / "ambiguous-commit.db"
    with sqlite3.connect(db_path, isolation_level=None) as raw_conn:
        raw_conn.execute("PRAGMA foreign_keys = ON")
        raw_conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users VALUES (1, '2026-01-02 03:04:05', NULL);
            """
        )

        with pytest.raises(RuntimeError, match="commit acknowledgement failed"):
            remediate_sqlite_profile_version_schema(
                _CommitAfterSuccessConnection(raw_conn)
            )

        remediate_sqlite_profile_version_schema(raw_conn)
        profile_column = {
            row[1]: row for row in raw_conn.execute("PRAGMA table_info(users)")
        }["profile_version"]
        assert profile_column[3] == 1
        assert raw_conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not raw_conn.in_transaction


def test_sync_remediation_marks_connection_invalid_on_post_commit_restore_failure(
    tmp_path: Path,
) -> None:
    class _RestoreFailureConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        @property
        def in_transaction(self) -> bool:
            return self._connection.in_transaction

        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            if sql == "PRAGMA foreign_keys = 1":
                raise RuntimeError("restore failed")
            return self._connection.execute(sql, parameters)

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

    db_path = tmp_path / "restore-after-commit.db"
    with sqlite3.connect(db_path, isolation_level=None) as raw_conn:
        raw_conn.execute("PRAGMA foreign_keys = ON")
        raw_conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users VALUES (1, '2026-01-02 03:04:05', NULL);
            """
        )

        with pytest.raises(RuntimeError, match="restore connection state") as raised:
            remediate_sqlite_profile_version_schema(
                _RestoreFailureConnection(raw_conn)
            )

        assert sqlite_profile_version_schema.sqlite_profile_version_connection_invalid(
            raised.value
        )
        assert raw_conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        assert not raw_conn.in_transaction
        raw_conn.execute("PRAGMA foreign_keys = ON")
        remediate_sqlite_profile_version_schema(raw_conn)


def test_sync_remediation_rejects_caller_owned_transaction_without_mutation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "caller-transaction.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO users VALUES (1, '2026-01-02 03:04:05', NULL);
            """
        )
        before = _schema_snapshot(conn)
        conn.execute("BEGIN IMMEDIATE")

        with pytest.raises(RuntimeError, match="no active transaction"):
            remediate_sqlite_profile_version_schema(conn)

        assert conn.in_transaction
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        conn.rollback()
        assert _schema_snapshot(conn) == before


def test_readiness_rejects_temp_users_shadowing_main_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "temp-users-shadow.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE main.users (
                id INTEGER PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_version TEXT
            );
            INSERT INTO main.users VALUES (1, '2026-01-02 03:04:05', NULL);
            CREATE TEMP TABLE users (
                id INTEGER PRIMARY KEY,
                profile_version TEXT NOT NULL DEFAULT
                    (STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))
            );
            INSERT INTO temp.users (id) VALUES (99);
            """
        )
        before = _schema_snapshot(conn)

        with pytest.raises(RuntimeError, match="temporary.*users"):
            validate_sqlite_profile_version_readiness(conn)
        with pytest.raises(RuntimeError, match="temporary.*users"):
            remediate_sqlite_profile_version_schema(conn)

        assert _schema_snapshot(conn) == before
        assert conn.execute(
            "SELECT profile_version FROM main.users WHERE id = 1"
        ).fetchone() == (None,)


def test_readiness_audits_temp_triggers_that_write_main_users(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "temp-trigger.db"
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.executescript(
            """
            CREATE TABLE main.users (
                id INTEGER PRIMARY KEY,
                email TEXT,
                profile_version TEXT NOT NULL DEFAULT
                    (STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))
            );
            CREATE TABLE main.api_keys (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL
            );
            CREATE TEMP TRIGGER rewrite_user
            AFTER UPDATE ON main.api_keys
            BEGIN
                UPDATE users SET email = 'trigger@example.com'
                WHERE id = NEW.user_id;
            END;
            """
        )

        with pytest.raises(RuntimeError, match="unsafe SQLite trigger"):
            validate_sqlite_profile_version_readiness(conn)
