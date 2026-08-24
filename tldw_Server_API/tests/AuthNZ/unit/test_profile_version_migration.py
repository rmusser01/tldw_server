from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    apply_authnz_migrations,
    ensure_authnz_tables,
)

pytestmark = pytest.mark.unit

_LEGACY_SCHEMA_VERSION = 90
_CANONICAL_PROFILE_VERSION = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"
)
_SQLITE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Databases"
    / "SQLite"
    / "Schema"
    / "sqlite_users.sql"
)


def _create_legacy_users_db(
    db_path: Path,
    *,
    updated_at: str | None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, updated_at)
            VALUES ('legacy', 'legacy@example.com', 'hash', ?)
            """,
            (updated_at,),
        )
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (?, 'legacy current', CURRENT_TIMESTAMP)
            """,
            (_LEGACY_SCHEMA_VERSION,),
        )


def _profile_version(db_path: Path) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT profile_version FROM users").fetchone()
    assert row is not None
    return row[0]


def _create_current_users_db_with_profile_definition(
    db_path: Path,
    *,
    profile_definition: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                {profile_definition}
            )
            """
        )
        conn.execute(
            """
            INSERT INTO users (
                username,
                email,
                password_hash,
                updated_at,
                profile_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "current",
                "current@example.com",
                "hash",
                "2026-01-02 03:04:05.123456",
                "2026-01-02T03:04:05.123456Z",
            ),
        )
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (91, 'current', CURRENT_TIMESTAMP)
            """
        )


def test_fresh_sqlite_schema_has_canonical_profile_version_default(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SQLITE_SCHEMA_PATH.read_text(encoding="utf-8"))
        columns = {
            row[1]: row
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES ('fresh', 'fresh@example.com', 'hash')
            """
        )
        value = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'fresh'"
        ).fetchone()[0]

    profile_column = columns["profile_version"]
    assert profile_column[2].upper() == "TEXT"
    assert profile_column[3] == 1
    assert "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')" in profile_column[4]
    assert _CANONICAL_PROFILE_VERSION.fullmatch(value)


@pytest.mark.parametrize(
    ("legacy_value", "expected"),
    [
        ("2026-01-02 03:04:05", "2026-01-02T03:04:05.000000Z"),
        ("2026-01-02 03:04:05.123456", "2026-01-02T03:04:05.123456Z"),
        ("2026-01-02T05:04:05.123456+02:00", "2026-01-02T03:04:05.123456Z"),
        ("2026-01-02T03:04:05.123456Z", "2026-01-02T03:04:05.123456Z"),
    ],
)
def test_sqlite_upgrade_normalizes_updated_at_without_version_jump(
    tmp_path: Path,
    legacy_value: str,
    expected: str,
) -> None:
    db_path = tmp_path / "legacy.db"
    _create_legacy_users_db(db_path, updated_at=legacy_value)

    ensure_authnz_tables(db_path)

    assert _profile_version(db_path) == expected


def test_sqlite_upgrade_enforces_profile_version_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    _create_legacy_users_db(
        db_path,
        updated_at="2026-01-02 03:04:05.123456",
    )

    ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]: row
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
    profile_column = columns["profile_version"]
    assert profile_column[2].upper() == "TEXT"
    assert profile_column[3] == 1
    assert "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')" in profile_column[4]


def test_sqlite_upgrade_default_initializes_omitted_profile_version(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "omitted-column.db"
    _create_legacy_users_db(
        db_path,
        updated_at="2026-01-02 03:04:05.123456",
    )
    ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, updated_at)
            VALUES ('later', 'later@example.com', 'hash', CURRENT_TIMESTAMP)
            """
        )
        value = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'later'"
        ).fetchone()[0]

    assert _CANONICAL_PROFILE_VERSION.fullmatch(value)


@pytest.mark.parametrize(
    "profile_definition",
    [
        "profile_version TEXT",
        "profile_version TEXT NOT NULL",
        (
            "profile_version TEXT DEFAULT "
            "(STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))"
        ),
        (
            "profile_version DATETIME NOT NULL DEFAULT "
            "(STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))"
        ),
        (
            "profile_version TEXT NOT NULL DEFAULT "
            "'2026-01-02T03:04:05.123456Z'"
        ),
    ],
)
def test_current_sqlite_schema_fails_startup_on_profile_metadata_drift(
    tmp_path: Path,
    profile_definition: str,
) -> None:
    db_path = tmp_path / "metadata-drift.db"
    _create_current_users_db_with_profile_definition(
        db_path,
        profile_definition=profile_definition,
    )

    with pytest.raises(RuntimeError, match="profile_version"):
        ensure_authnz_tables(db_path)


def test_sqlite_upgrade_preserves_custom_users_schema_objects_and_foreign_keys(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "custom-schema.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE tenants (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tenant_id INTEGER NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
                display_name TEXT NOT NULL DEFAULT 'Legacy'
                    CHECK (length(display_name) > 0),
                CONSTRAINT uq_users_identity UNIQUE (username, email)
            );
            CREATE TABLE user_children (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX idx_users_display_name ON users(display_name);
            CREATE TRIGGER update_users_timestamp
                AFTER UPDATE OF username ON users
                FOR EACH ROW
            BEGIN
                UPDATE users
                SET updated_at = '2030-01-02 03:04:05.123456'
                WHERE id = NEW.id;
            END;
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            );
            INSERT INTO tenants (id, name) VALUES (3, 'preserved');
            INSERT INTO users (
                id,
                username,
                email,
                password_hash,
                updated_at,
                tenant_id,
                display_name
            ) VALUES (
                7,
                'legacy-custom',
                'legacy-custom@example.com',
                'hash',
                '2026-01-02 03:04:05.123456',
                3,
                'Custom'
            );
            INSERT INTO user_children (id, user_id) VALUES (11, 7);
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (90, 'legacy current', CURRENT_TIMESTAMP);
            UPDATE sqlite_sequence SET seq = 42 WHERE name = 'users';
            """
        )

    ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        ]
        row = conn.execute(
            """
            SELECT id, tenant_id, display_name, updated_at, profile_version
            FROM users WHERE id = 7
            """
        ).fetchone()
        schema_objects = {
            (object_type, name): sql
            for object_type, name, sql in conn.execute(
                """
                SELECT type, name, sql
                FROM sqlite_master
                WHERE tbl_name = 'users' AND type IN ('index', 'trigger')
                  AND sql IS NOT NULL
                """
            ).fetchall()
        }
        users_foreign_keys = conn.execute("PRAGMA foreign_key_list(users)").fetchall()
        child_foreign_keys = conn.execute(
            "PRAGMA foreign_key_list(user_children)"
        ).fetchall()
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        sequence = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'users'"
        ).fetchone()[0]

        conn.execute("UPDATE users SET username = 'renamed-custom' WHERE id = 7")
        triggered = conn.execute(
            "SELECT updated_at, profile_version FROM users WHERE id = 7"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO users (
                    username,
                    email,
                    password_hash,
                    updated_at,
                    tenant_id,
                    display_name
                ) VALUES (?, ?, ?, ?, ?, '')
                """,
                (
                    "blank-name",
                    "blank-name@example.com",
                    "hash",
                    "2026-01-02 03:04:05",
                    3,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO users (
                    username,
                    email,
                    password_hash,
                    updated_at,
                    tenant_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "renamed-custom",
                    "legacy-custom@example.com",
                    "hash",
                    "2026-01-02 03:04:05",
                    3,
                ),
            )
        inserted_id = conn.execute(
            """
            INSERT INTO users (
                username,
                email,
                password_hash,
                updated_at,
                tenant_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "post-upgrade",
                "post-upgrade@example.com",
                "hash",
                "2026-01-02 03:04:05",
                3,
            ),
        ).lastrowid
        inserted_defaults = conn.execute(
            "SELECT display_name, profile_version FROM users WHERE id = ?",
            (inserted_id,),
        ).fetchone()
        conn.execute("DELETE FROM users WHERE id = 7")
        child_count = conn.execute(
            "SELECT COUNT(*) FROM user_children WHERE user_id = 7"
        ).fetchone()[0]

    assert columns == [
        "id",
        "username",
        "email",
        "password_hash",
        "updated_at",
        "tenant_id",
        "display_name",
        "profile_version",
    ]
    assert row == (
        7,
        3,
        "Custom",
        "2026-01-02 03:04:05.123456",
        "2026-01-02T03:04:05.123456Z",
    )
    assert ("index", "idx_users_display_name") in schema_objects
    assert ("trigger", "update_users_timestamp") in schema_objects
    assert users_foreign_keys[0][2] == "tenants"
    assert child_foreign_keys[0][2] == "users"
    assert foreign_key_errors == []
    assert sequence == 42
    assert triggered == (
        "2030-01-02 03:04:05.123456",
        "2026-01-02T03:04:05.123456Z",
    )
    assert inserted_id == 43
    assert inserted_defaults[0] == "Legacy"
    assert _CANONICAL_PROFILE_VERSION.fullmatch(inserted_defaults[1])
    assert child_count == 0


def test_sqlite_profile_version_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.db"
    _create_legacy_users_db(
        db_path,
        updated_at="2026-01-02 03:04:05.123456",
    )

    ensure_authnz_tables(db_path)
    first_value = _profile_version(db_path)
    ensure_authnz_tables(db_path)

    assert _profile_version(db_path) == first_value
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 91"
        ).fetchone()[0]
    assert count == 1


def test_candidate_timestamp_migration_preserves_rows_and_indexes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "candidate-timestamps.db"
    apply_authnz_migrations(db_path, target_version=91)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("candidate-user", "candidate@example.com", "hash"),
        )
        user_id = int(conn.execute("SELECT id FROM users").fetchone()[0])
        conn.execute(
            "INSERT INTO organizations (name, updated_at) VALUES (?, NULL)",
            ("Candidate Org",),
        )
        org_id = int(conn.execute("SELECT id FROM organizations").fetchone()[0])
        conn.execute(
            "INSERT INTO org_members (org_id, user_id, added_at) "
            "VALUES (?, ?, NULL)",
            (org_id, user_id),
        )
        conn.execute(
            "CREATE INDEX candidate_org_slug_idx ON organizations(slug)"
        )
        conn.commit()

    apply_authnz_migrations(db_path)

    with sqlite3.connect(db_path) as conn:
        org_row = conn.execute(
            "SELECT name, updated_at FROM organizations WHERE id = ?",
            (org_id,),
        ).fetchone()
        member_added_at = conn.execute(
            "SELECT added_at FROM org_members WHERE org_id = ? AND user_id = ?",
            (org_id, user_id),
        ).fetchone()[0]
        index_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("candidate_org_slug_idx",),
        ).fetchone()
        updated_at_metadata = {
            row[1]: row for row in conn.execute("PRAGMA table_info(organizations)")
        }["updated_at"]

    assert org_row[0] == "Candidate Org"
    assert org_row[1] is not None
    assert member_added_at is not None
    assert index_exists is not None
    assert updated_at_metadata[3] == 1


@pytest.mark.parametrize("legacy_value", [None, "not-a-timestamp"])
def test_sqlite_upgrade_rejects_invalid_updated_at_and_rolls_back(
    tmp_path: Path,
    legacy_value: str | None,
) -> None:
    db_path = tmp_path / "invalid-legacy.db"
    _create_legacy_users_db(db_path, updated_at=legacy_value)
    with sqlite3.connect(db_path) as conn:
        schema_before = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            ORDER BY type, name
            """
        ).fetchall()
        users_before = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        sequence_before = conn.execute(
            "SELECT name, seq FROM sqlite_sequence ORDER BY name"
        ).fetchall()

    with pytest.raises(RuntimeError, match="profile_version"):
        ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        latest = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        schema_after = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            ORDER BY type, name
            """
        ).fetchall()
        users_after = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        sequence_after = conn.execute(
            "SELECT name, seq FROM sqlite_sequence ORDER BY name"
        ).fetchall()
    assert latest == _LEGACY_SCHEMA_VERSION
    assert "profile_version" not in columns
    assert schema_after == schema_before
    assert users_after == users_before
    assert sequence_after == sequence_before


@pytest.mark.parametrize("corrupt_value", [None, "not-a-profile-version"])
def test_current_sqlite_schema_fails_startup_when_profile_version_is_corrupt(
    tmp_path: Path,
    corrupt_value: str | None,
) -> None:
    db_path = tmp_path / "corrupt-current.db"
    if corrupt_value is None:
        _create_current_users_db_with_profile_definition(
            db_path,
            profile_definition=(
                "profile_version TEXT DEFAULT "
                "(STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))"
            ),
        )
    else:
        _create_legacy_users_db(
            db_path,
            updated_at="2026-01-02 03:04:05.123456",
        )
        ensure_authnz_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE users SET profile_version = ?", (corrupt_value,))

    with pytest.raises(RuntimeError, match="profile_version"):
        ensure_authnz_tables(db_path)


def test_current_sqlite_schema_fails_startup_when_candidate_schema_is_corrupt(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "corrupt-candidates.db"
    ensure_authnz_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(
            """
            ALTER TABLE org_members RENAME TO legacy_org_members;
            CREATE TABLE org_members (
                org_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT DEFAULT 'member',
                status TEXT DEFAULT 'active',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (org_id, user_id),
                FOREIGN KEY (org_id) REFERENCES teams(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            DROP TABLE legacy_org_members;
            """
        )

    with pytest.raises(RuntimeError, match="candidate schema"):
        ensure_authnz_tables(db_path)


def test_update_users_timestamp_trigger_cannot_change_profile_version(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "trigger.db"
    fixed_version = "2026-01-02T03:04:05.123456Z"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SQLITE_SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, profile_version)
            VALUES ('trigger-user', 'trigger@example.com', 'hash', ?)
            """,
            (fixed_version,),
        )
        conn.execute(
            "UPDATE users SET username = 'renamed-user' WHERE username = 'trigger-user'"
        )
        value = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'renamed-user'"
        ).fetchone()[0]
        trigger_sql = conn.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'trigger' AND name = 'update_users_timestamp'
            """
        ).fetchone()[0]

    assert value == fixed_version
    assert "profile_version" not in trigger_sql.lower()
