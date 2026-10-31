"""The spec-mandated (user_id, ts) index on llm_usage_log (spec 2 §4)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations

pytestmark = pytest.mark.unit


def _index_names(db_path: Path) -> set[str]:
    """The names of every index currently on llm_usage_log."""
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA index_list('llm_usage_log')").fetchall()}


def test_fresh_sqlite_db_has_the_user_ts_index(tmp_path: Path) -> None:
    """A brand-new database, migrated to latest, already carries the index."""
    db_path = tmp_path / "fresh.db"
    apply_authnz_migrations(db_path)
    assert "idx_llm_usage_log_user_ts" in _index_names(db_path)


def _create_legacy_llm_usage_log_database(db_path: Path) -> None:
    """A version-98 database: llm_usage_log exists with only the pre-migration-99 indexes."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE llm_usage_log (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP NOT NULL,
                user_id INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX idx_llm_usage_log_ts ON llm_usage_log(ts)")
        conn.execute("CREATE INDEX idx_llm_usage_log_user ON llm_usage_log(user_id)")
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
            "INSERT INTO schema_migrations (version, name, applied_at) "
            "VALUES (98, 'legacy current', CURRENT_TIMESTAMP)"
        )


def test_sqlite_db_migrated_before_the_index_was_added_still_gets_it(tmp_path: Path) -> None:
    """A database upgraded from before the index migration also gets the index."""
    db_path = tmp_path / "legacy.db"
    _create_legacy_llm_usage_log_database(db_path)
    assert "idx_llm_usage_log_user_ts" not in _index_names(db_path)

    apply_authnz_migrations(db_path)
    assert "idx_llm_usage_log_user_ts" in _index_names(db_path)
