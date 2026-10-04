"""Migration 100 copies non-default storage quotas into limits.storage_quota_mb overrides (spec 2 §8)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.AuthNZ import storage_quota_backfill
from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations

pytestmark = pytest.mark.unit
KEY = "limits.storage_quota_mb"


def _add_user(conn: sqlite3.Connection, name: str, quota_mb: int) -> int:
    """Insert a user row with the given legacy column value; return its id."""
    cur = conn.execute(
        "INSERT INTO users (username, email, password_hash, storage_quota_mb) VALUES (?, ?, 'x', ?)",
        (name, f"{name}@example.com", quota_mb),
    )
    return int(cur.lastrowid)


def _overrides(db_path: Path) -> dict[int, object]:
    """user_id -> decoded value of every limits.storage_quota_mb override."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id, value_json FROM user_config_overrides WHERE key = ?", (KEY,)).fetchall()
    return {int(uid): json.loads(val) for uid, val in rows}


def test_skip_values_include_the_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """5120 and the DEFAULT_STORAGE_QUOTA_MB configured now are both treated as 'never set'."""
    monkeypatch.setattr(authnz_settings, "get_settings", lambda: SimpleNamespace(DEFAULT_STORAGE_QUOTA_MB=10240))
    assert storage_quota_backfill.skip_values() == [5120, 10240]


def test_copies_only_non_default_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2048 is copied; 5120 and the configured default (10240) are dropped; an existing override is kept."""
    monkeypatch.setattr(storage_quota_backfill, "skip_values", lambda: [5120, 10240])
    db_path = tmp_path / "authnz.db"
    apply_authnz_migrations(db_path, target_version=99)
    with sqlite3.connect(db_path) as conn:
        u_default = _add_user(conn, "dflt", 5120)
        u_custom = _add_user(conn, "custom", 2048)
        u_configured = _add_user(conn, "configured", 10240)
        u_existing = _add_user(conn, "existing", 3000)
        conn.execute(
            "INSERT INTO user_config_overrides (user_id, key, value_json, created_at, updated_at) "
            "VALUES (?, ?, '999', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (u_existing, KEY),
        )
    apply_authnz_migrations(db_path)
    assert _overrides(db_path) == {u_custom: 2048, u_existing: 999}
    assert u_default not in _overrides(db_path) and u_configured not in _overrides(db_path)


def test_skips_when_users_table_absent(tmp_path: Path) -> None:
    """A synthetic version-99 DB without a users table migrates to 100 without error."""
    db_path = tmp_path / "synthetic.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TIMESTAMP NOT NULL)")
        conn.execute("INSERT INTO schema_migrations VALUES (99, 'synthetic', CURRENT_TIMESTAMP)")
    apply_authnz_migrations(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 100
