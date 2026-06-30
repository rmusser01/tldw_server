from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ import migrations as authnz_migrations
from tldw_Server_API.app.core.AuthNZ.migrations import (
    migration_001_create_users_table,
    migration_003_create_api_keys_table,
    migration_012_create_rbac_tables,
    migration_013_create_rbac_limits_and_usage,
    migration_015_create_llm_usage_tables,
    migration_088_add_llm_usage_cache_accounting_columns,
)


def _prepare_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    migration_001_create_users_table(conn)
    migration_003_create_api_keys_table(conn)
    migration_012_create_rbac_tables(conn)
    return conn


def _fk_count(conn: sqlite3.Connection, table_name: str) -> int:
    rows = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
    return len(rows)


def test_migration_013_relaxes_fks_when_tldw_test_mode_is_y(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.delenv("DISABLE_USAGE_FOREIGN_KEYS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    conn = _prepare_conn()
    migration_013_create_rbac_limits_and_usage(conn)

    assert _fk_count(conn, "usage_log") == 0
    assert _fk_count(conn, "usage_daily") == 0


def test_migration_015_relaxes_fks_when_tldw_test_mode_is_y(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.delenv("DISABLE_USAGE_FOREIGN_KEYS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    conn = _prepare_conn()
    migration_015_create_llm_usage_tables(conn)

    assert _fk_count(conn, "llm_usage_log") == 0
    assert _fk_count(conn, "llm_usage_daily") == 0


def test_migrations_013_and_015_keep_fks_when_test_flags_disabled(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.delenv("DISABLE_USAGE_FOREIGN_KEYS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    conn = _prepare_conn()
    migration_013_create_rbac_limits_and_usage(conn)
    migration_015_create_llm_usage_tables(conn)

    assert _fk_count(conn, "usage_log") >= 1
    assert _fk_count(conn, "usage_daily") >= 1
    assert _fk_count(conn, "llm_usage_log") >= 1
    assert _fk_count(conn, "llm_usage_daily") >= 1


def test_migration_088_fails_when_llm_usage_log_table_is_missing() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(sqlite3.OperationalError, match="llm_usage_log"):
        migration_088_add_llm_usage_cache_accounting_columns(conn)


def test_migration_088_skips_existing_columns_and_adds_missing_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE llm_usage_log (id INTEGER PRIMARY KEY, cached_input_tokens INTEGER)")

    migration_088_add_llm_usage_cache_accounting_columns(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)").fetchall()}
    assert "cached_input_tokens" in columns
    assert "raw_usage_metadata_json" in columns


def test_apply_authnz_migrations_allows_redis_file_fallback_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_lock_kwargs: dict[str, object] = {}
    recorded_apply_args: dict[str, object] = {}

    @contextmanager
    def fake_acquire_migration_lock(**kwargs: object):
        recorded_lock_kwargs.update(kwargs)
        yield object()

    def fake_apply_locked(db_path: Path, target_version: int | None = None) -> None:
        recorded_apply_args["db_path"] = db_path
        recorded_apply_args["target_version"] = target_version

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(authnz_migrations, "_is_test_mode", lambda: True)
    monkeypatch.setattr(authnz_migrations, "_is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(authnz_migrations, "acquire_migration_lock", fake_acquire_migration_lock)
    monkeypatch.setattr(authnz_migrations, "_apply_authnz_migrations_locked", fake_apply_locked)

    authnz_migrations.apply_authnz_migrations(db_path, target_version=7)

    assert recorded_lock_kwargs["redis_url"] == "redis://127.0.0.1:1/0"
    assert recorded_lock_kwargs["lock_dir"] == str(tmp_path)
    assert recorded_lock_kwargs["allow_file_fallback_on_redis_error"] is True
    assert recorded_apply_args == {"db_path": db_path, "target_version": 7}


def test_apply_authnz_migrations_allows_redis_file_fallback_in_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_lock_kwargs: dict[str, object] = {}

    @contextmanager
    def fake_acquire_migration_lock(**kwargs: object):
        recorded_lock_kwargs.update(kwargs)
        yield object()

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(authnz_migrations, "_is_test_mode", lambda: False)
    monkeypatch.setattr(authnz_migrations, "_is_explicit_pytest_runtime", lambda: True)
    monkeypatch.setattr(authnz_migrations, "acquire_migration_lock", fake_acquire_migration_lock)
    monkeypatch.setattr(authnz_migrations, "_apply_authnz_migrations_locked", lambda *_args, **_kwargs: None)

    authnz_migrations.apply_authnz_migrations(db_path)

    assert recorded_lock_kwargs["redis_url"] == "redis://127.0.0.1:1/0"
    assert recorded_lock_kwargs["allow_file_fallback_on_redis_error"] is True


def test_apply_authnz_migrations_keeps_redis_fail_closed_outside_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_lock_kwargs: dict[str, object] = {}

    @contextmanager
    def fake_acquire_migration_lock(**kwargs: object):
        recorded_lock_kwargs.update(kwargs)
        yield object()

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(authnz_migrations, "_is_test_mode", lambda: False)
    monkeypatch.setattr(authnz_migrations, "_is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(authnz_migrations, "acquire_migration_lock", fake_acquire_migration_lock)
    monkeypatch.setattr(authnz_migrations, "_apply_authnz_migrations_locked", lambda *_args, **_kwargs: None)

    authnz_migrations.apply_authnz_migrations(db_path)

    assert recorded_lock_kwargs["allow_file_fallback_on_redis_error"] is False
