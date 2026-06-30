import importlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


def test_sqlite_schema_has_expected_columns_and_indexes(tmp_path):


    db_path = ensure_jobs_tables(tmp_path / "jobs_mig.db")
    conn = sqlite3.connect(db_path)
    try:
        # Columns present
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for expected in [
            "completion_token",
            "failure_timeline",
            "request_id",
            "trace_id",
            "progress_percent",
            "progress_message",
            "error_code",
            "error_class",
            "error_stack",
        ]:
            assert expected in cols
        # Archive table exists
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs_archive'").fetchone()
        assert row and row[0] == "jobs_archive"
        # Partial unique index for idempotency exists
        idx = [r[1] for r in conn.execute("PRAGMA index_list('jobs')").fetchall()]
        assert any("idx_jobs_idempotent" in x for x in idx)
        # job_counters exists
        row2 = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_counters'").fetchone()
        assert row2 and row2[0] == "job_counters"
        # Idempotent ensure re-runs without error
        ensure_jobs_tables(Path(db_path))
    finally:
        conn.close()


def test_jobs_migrations_uses_shared_sqlite_policy_helper(tmp_path):
    jobs_migrations = importlib.import_module("tldw_Server_API.app.core.Jobs.migrations")
    calls: list[dict[str, object]] = []

    def fake_configure(conn, **kwargs):
        calls.append(kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sqlite_policy, "configure_sqlite_connection", fake_configure)
        jobs_migrations = importlib.reload(jobs_migrations)
        jobs_migrations.ensure_jobs_tables(tmp_path / "jobs_helper.db")

    importlib.reload(jobs_migrations)

    assert calls == [{
        "use_wal": True,
        "synchronous": "NORMAL",
        "busy_timeout_ms": 5000,
        "foreign_keys": False,
        "temp_store": None,
    }]


def test_ensure_jobs_tables_sanitizes_schema_failure_log(tmp_path, monkeypatch):
    jobs_migrations = importlib.import_module("tldw_Server_API.app.core.Jobs.migrations")
    secret = "sk_jobsMigrationSecret1234567890"
    db_path = tmp_path / secret / "jobs.db"
    fake_logger = MagicMock()

    def fail_connect(path):
        raise sqlite3.OperationalError(f"unable to open database file {path} token={secret}")

    monkeypatch.setattr(jobs_migrations.sqlite3, "connect", fail_connect)
    monkeypatch.setattr(jobs_migrations, "logger", fake_logger)

    result = jobs_migrations.ensure_jobs_tables(db_path)

    assert result == db_path
    fake_logger.warning.assert_called_once()
    warning_args = fake_logger.warning.call_args.args
    rendered = " ".join(str(arg) for arg in warning_args)
    assert "Failed to ensure Jobs schema" in rendered
    assert str(db_path) not in rendered
    assert secret not in rendered
