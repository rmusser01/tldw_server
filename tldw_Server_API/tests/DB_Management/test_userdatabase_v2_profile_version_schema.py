from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    ProfileUserWriteRejected,
)
from tldw_Server_API.app.core.DB_Management import UserDatabase_v2 as user_database_module
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
    close_all_backends,
    is_factory_managed_backend,
)
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import (
    UserDatabase,
    UserDatabaseError,
)

pytestmark = pytest.mark.unit

_CANONICAL_PROFILE_VERSION = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"
)


@pytest.fixture(autouse=True)
def _reset_backends() -> None:
    close_all_backends()
    yield
    close_all_backends()


def _config(db_path: Path) -> DatabaseConfig:
    return DatabaseConfig(
        backend_type=BackendType.SQLITE,
        sqlite_path=str(db_path),
    )


def _profile_column(conn: sqlite3.Connection) -> tuple[object, ...]:
    return {
        row[1]: row for row in conn.execute("PRAGMA table_info(users)")
    }["profile_version"]


def _create_legacy_custom_schema(db_path: Path, *, malformed: bool = False) -> None:
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE tenants (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                role TEXT DEFAULT 'user',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL,
                last_login TEXT,
                email_verified INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0,
                profile_version TEXT,
                custom_rank INTEGER NOT NULL DEFAULT 7 CHECK (custom_rank > 0),
                tenant_id INTEGER,
                CONSTRAINT fk_users_tenant
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            );
            CREATE UNIQUE INDEX idx_users_custom_rank
                ON users(custom_rank, username);
            CREATE TABLE user_change_log (
                username TEXT NOT NULL,
                changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TRIGGER trg_users_custom_rank
            AFTER UPDATE OF custom_rank ON users
            BEGIN
                INSERT INTO user_change_log(username) VALUES (NEW.username);
            END;
            CREATE TABLE user_children (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
            );
            INSERT INTO tenants(id, name) VALUES (1, 'tenant');
            """
        )
        profile_value = (
            "not-a-timestamp"
            if malformed
            else "2026-02-03T04:05:06.654321Z"
        )
        conn.execute(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, updated_at,
                profile_version, custom_rank, tenant_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "preserved-uuid",
                "preserved",
                "preserved@example.com",
                "hash",
                "2026-02-03 04:05:06.654321",
                profile_value,
                11,
                1,
            ),
        )
        if not malformed:
            conn.execute(
                """
                INSERT INTO users (
                    uuid, username, email, password_hash, updated_at,
                    profile_version, custom_rank, tenant_id
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    "backfilled-uuid",
                    "backfilled",
                    "backfilled@example.com",
                    "hash",
                    "2026-01-02 03:04:05.123456",
                    12,
                    1,
                ),
            )
        conn.execute("UPDATE sqlite_sequence SET seq = 73 WHERE name = 'users'")


def test_fresh_bootstrap_uses_canonical_default_and_remains_guarded(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "fresh.db"
    db = UserDatabase(config=_config(db_path), client_id="schema-test")

    with sqlite3.connect(db_path) as conn:
        profile_column = _profile_column(conn)
        conn.execute(
            """
            INSERT INTO users (uuid, username, email, password_hash)
            VALUES ('omitted', 'omitted', 'omitted@example.com', 'hash')
            """
        )
        omitted = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'omitted'"
        ).fetchone()[0]

    assert profile_column[2].upper() == "TEXT"
    assert profile_column[3] == 1
    assert profile_column[4] == "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')"
    assert _CANONICAL_PROFILE_VERSION.fullmatch(omitted)
    with sqlite3.connect(db_path) as conn:
        org_member_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(org_members)")
        }
        override_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_config_overrides)")
        }
        org_member_pk = {
            row[1]
            for row in conn.execute("PRAGMA table_info(org_members)")
            if row[5]
        }
        override_pk = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_config_overrides)")
            if row[5]
        }
    assert {"org_id", "user_id", "role", "status", "added_at"} <= org_member_columns
    assert org_member_pk == {"org_id", "user_id"}
    assert {
        "user_id",
        "key",
        "value_json",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
    } <= override_columns
    assert override_pk == {"user_id", "key"}
    assert is_factory_managed_backend(db.backend)
    with pytest.raises(ProfileUserWriteRejected):
        db.backend.execute(
            "UPDATE users SET email = ? WHERE username = ?",
            ("blocked@example.com", "omitted"),
        )

    UserDatabase(config=_config(db_path), client_id="schema-test-2")
    with sqlite3.connect(db_path) as conn:
        assert _profile_column(conn) == profile_column
        assert conn.execute(
            "SELECT profile_version FROM users WHERE username = 'omitted'"
        ).fetchone()[0] == omitted


def test_legacy_bootstrap_preserves_custom_schema_objects_fk_and_sequence(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-custom.db"
    _create_legacy_custom_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        original_index = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_users_custom_rank'"
        ).fetchone()[0]
        original_trigger = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'trg_users_custom_rank'"
        ).fetchone()[0]
        original_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(users)")
        ]

    db = UserDatabase(config=_config(db_path), client_id="schema-test")

    with sqlite3.connect(db_path) as conn:
        profile_column = _profile_column(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        values = conn.execute(
            """
            SELECT username, profile_version, custom_rank, tenant_id
            FROM users ORDER BY id
            """
        ).fetchall()
        index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_users_custom_rank'"
        ).fetchone()[0]
        trigger_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'trg_users_custom_rank'"
        ).fetchone()[0]
        child_fk = conn.execute("PRAGMA foreign_key_list(user_children)").fetchall()
        users_fk = conn.execute("PRAGMA foreign_key_list(users)").fetchall()
        sequence = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'users'"
        ).fetchone()[0]

    assert columns[: len(original_columns)] == original_columns
    assert profile_column[2].upper() == "TEXT"
    assert profile_column[3] == 1
    assert profile_column[4] == "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')"
    assert values == [
        ("preserved", "2026-02-03T04:05:06.654321Z", 11, 1),
        ("backfilled", "2026-01-02T03:04:05.123456Z", 12, 1),
    ]
    assert index_sql == original_index
    assert trigger_sql == original_trigger
    assert any(row[2] == "users" for row in child_fk)
    assert any(row[2] == "tenants" for row in users_fk)
    assert sequence == 73
    assert is_factory_managed_backend(db.backend)


def test_legacy_bootstrap_malformed_value_fails_closed_without_partial_rebuild(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "malformed.db"
    _create_legacy_custom_schema(db_path, malformed=True)
    with sqlite3.connect(db_path) as conn:
        before_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
        ).fetchone()[0]
        before_rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        before_sequence = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'users'"
        ).fetchone()[0]

    raw_backend = DatabaseBackendFactory.create_backend(_config(db_path))
    raw_connection = raw_backend.get_pool().get_connection()
    raw_connection.execute("PRAGMA foreign_keys = ON")

    with pytest.raises(UserDatabaseError, match="profile_version"):
        UserDatabase(backend=raw_backend, client_id="schema-test")

    assert raw_connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert not raw_connection.in_transaction
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        assert conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
        ).fetchone()[0] == before_schema
        assert conn.execute("SELECT * FROM users ORDER BY id").fetchall() == before_rows
        assert conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'users'"
        ).fetchone()[0] == before_sequence
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?",
            ("__authnz_users_profile_version_v91",),
        ).fetchone() is None


def test_bootstrap_invalidates_pool_connection_after_remediation_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "cleanup-failure.db"
    raw_backend = DatabaseBackendFactory.create_backend(_config(db_path))
    raw_connection = raw_backend.get_pool().get_connection()
    failure = RuntimeError("cleanup failed")
    failure._authnz_sqlite_profile_version_connection_invalid = True

    def _fail_remediation(_connection: sqlite3.Connection) -> None:
        raise failure

    monkeypatch.setattr(
        user_database_module,
        "remediate_sqlite_profile_version_schema",
        _fail_remediation,
    )

    with pytest.raises(UserDatabaseError, match="profile_version"):
        UserDatabase(backend=raw_backend, client_id="schema-test")

    with pytest.raises(sqlite3.ProgrammingError):
        raw_connection.execute("SELECT 1")
