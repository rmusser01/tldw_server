"""Regression tests for SQLite users.uuid schema drift (#2875).

The packaged schema (sqlite_users.sql) declares users.uuid NOT NULL with a
randomblob-hex default, while the migrations path historically added the
column nullable with no default. Migration 097 backfills NULL/blank uuids and
migration 001 now creates the column with the packaged default so both fresh
creation paths agree that a user row always carries a uuid.
"""

import pathlib
import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    apply_authnz_migrations,
    ensure_authnz_tables,
)

pytestmark = pytest.mark.unit


def _connect(db_path: pathlib.Path) -> sqlite3.Connection:
    """Open a SQLite connection with dict-style row access for assertions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_fresh_migrations_path_defaults_user_uuid(tmp_path: pathlib.Path) -> None:
    """A user inserted without a uuid on a fresh migrations-path DB gets one."""
    db_path = tmp_path / "fresh.db"
    ensure_authnz_tables(db_path)

    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("drift_user", "drift@example.local", "hash"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT uuid FROM users WHERE username = 'drift_user'"
        ).fetchone()
        assert row is not None
        assert row["uuid"], "fresh migrations-path DB must default users.uuid"
    finally:
        conn.close()


def test_migration_097_backfills_null_and_blank_uuids(tmp_path: pathlib.Path) -> None:
    """Legacy NULL/blank uuids are backfilled; existing uuids are preserved."""
    db_path = tmp_path / "legacy.db"
    # Build the schema up to the version before the backfill.
    apply_authnz_migrations(db_path, target_version=96)

    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, uuid) VALUES (?, ?, ?, NULL)",
            ("null_uuid", "null@example.local", "hash"),
        )
        conn.execute(
            "INSERT INTO users (username, email, password_hash, uuid) VALUES (?, ?, ?, '')",
            ("blank_uuid", "blank@example.local", "hash"),
        )
        conn.execute(
            "INSERT INTO users (username, email, password_hash, uuid) VALUES (?, ?, ?, ?)",
            ("kept_uuid", "kept@example.local", "hash", "existing-uuid-value"),
        )
        conn.commit()
    finally:
        conn.close()

    apply_authnz_migrations(db_path)

    conn = _connect(db_path)
    try:
        rows = {
            row["username"]: row["uuid"]
            for row in conn.execute("SELECT username, uuid FROM users").fetchall()
        }
        assert rows["null_uuid"], "NULL uuid must be backfilled"
        assert rows["blank_uuid"].strip(), "blank uuid must be backfilled"
        assert rows["kept_uuid"] == "existing-uuid-value"
        assert len({rows["null_uuid"], rows["blank_uuid"]}) == 2, (
            "backfilled uuids must be unique per row"
        )
    finally:
        conn.close()
