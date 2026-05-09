from __future__ import annotations

import os
import random
import string
import uuid
from datetime import datetime, timezone

import pytest


class _FakePostgresCursor:
    def __init__(self, *, failing_table: str, failing_column: str) -> None:
        self.failing_table = failing_table
        self.failing_column = failing_column
        self._last_lookup: tuple[str, str] | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[str, str] | None = None) -> None:
        if "information_schema.columns" in query and params:
            self._last_lookup = params
            return
        failing_alter = f"ALTER TABLE {self.failing_table} ADD COLUMN {self.failing_column}"
        if failing_alter in query:
            raise OSError("simulated migration failure")

    def fetchone(self):
        return None


class _FakePostgresConnection:
    def __init__(self, cursor: _FakePostgresCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self) -> _FakePostgresCursor:
        return self._cursor


def _has_psycopg() -> bool:


    try:
        import psycopg  # noqa: F401
        return True
    except Exception:
        return False


def test_postgres_store_init_fails_fast_when_required_column_migration_fails(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.store import ClusterStoreUnavailable, PostgresStore

    cursor = _FakePostgresCursor(
        failing_table="sandbox_vz_sessions",
        failing_column="helper_instance_id",
    )
    store = PostgresStore.__new__(PostgresStore)
    monkeypatch.setattr(store, "_conn", lambda: _FakePostgresConnection(cursor))

    with pytest.raises(ClusterStoreUnavailable, match="sandbox_vz_sessions.helper_instance_id"):
        store._init_db()


@pytest.mark.integration
def test_postgres_store_adds_missing_columns(monkeypatch):
    dsn = os.getenv("SANDBOX_TEST_PG_DSN")
    if not dsn or not _has_psycopg():
        pytest.skip("Postgres DSN not provided or psycopg not installed")
    import psycopg

    # Prepare a clean slate: drop tables if exist and create old schema without new columns
    with psycopg.connect(dsn, autocommit=True) as con:
        with con.cursor() as cur:
            for tbl in ("sandbox_runs", "sandbox_idempotency", "sandbox_usage", "sandbox_sessions", "sandbox_acp_sessions"):
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {tbl}")
                except Exception:
                    _ = None
            # Old sandbox_runs schema (no runtime_version/resource_usage)
            cur.execute(
                """
                CREATE TABLE sandbox_runs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    spec_version TEXT,
                    runtime TEXT,
                    base_image TEXT,
                    phase TEXT,
                    exit_code INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    message TEXT,
                    image_digest TEXT,
                    policy_hash TEXT
                );
                """
            )
            # Minimal other tables
            cur.execute(
                """
                CREATE TABLE sandbox_idempotency (
                    endpoint TEXT,
                    user_key TEXT,
                    key TEXT,
                    fingerprint TEXT,
                    object_id TEXT,
                    response_body JSONB,
                    created_at DOUBLE PRECISION,
                    PRIMARY KEY (endpoint, user_key, key)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE sandbox_usage (
                    user_id TEXT PRIMARY KEY,
                    artifact_bytes BIGINT
                );
                """
            )

    # Instantiate store; __init__ runs _init_db which performs migrations
    from tldw_Server_API.app.core.Sandbox.store import PostgresStore
    st = PostgresStore(dsn=dsn)
    # Verify columns now exist
    with psycopg.connect(dsn, autocommit=True) as con:
        with con.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name='sandbox_runs'
                """
            )
            cols = {r[0] for r in cur.fetchall()}
            assert "runtime_version" in cols
            assert "resource_usage" in cols
            assert "session_id" in cols
            assert "persona_id" in cols
            assert "workspace_id" in cols
            assert "workspace_group_id" in cols
            assert "scope_snapshot_id" in cols
            assert "claim_owner" in cols
            assert "claim_expires_at" in cols

            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name='sandbox_acp_sessions'
                """
            )
            acp_cols = {r[0] for r in cur.fetchall()}
            assert "id" in acp_cols
            assert "user_id" in acp_cols
            assert "sandbox_session_id" in acp_cols
            assert "run_id" in acp_cols
            assert "ssh_private_key" in acp_cols
            assert "persona_id" in acp_cols
            assert "workspace_id" in acp_cols
            assert "workspace_group_id" in acp_cols
            assert "scope_snapshot_id" in acp_cols


@pytest.mark.integration
def test_postgres_update_run_preserves_owner(monkeypatch):
    dsn = os.getenv("SANDBOX_TEST_PG_DSN")
    if not dsn or not _has_psycopg():
        pytest.skip("Postgres DSN not provided or psycopg not installed")

    from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
    from tldw_Server_API.app.core.Sandbox.store import PostgresStore

    run_id = f"owner-preserve-{uuid.uuid4().hex[:12]}"
    owner = "owner-42"
    st = PostgresStore(dsn=dsn)
    initial = RunStatus(
        id=run_id,
        phase=RunPhase.queued,
        spec_version="1.0",
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
    )
    st.put_run(owner, initial)

    updated = RunStatus(
        id=run_id,
        phase=RunPhase.running,
        spec_version="1.0",
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        started_at=datetime.now(timezone.utc),
        message="started",
    )
    st.update_run(updated)

    assert st.get_run_owner(run_id) == owner

    import psycopg

    with psycopg.connect(dsn, autocommit=True) as con:
        with con.cursor() as cur:
            cur.execute("SELECT user_id FROM sandbox_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == owner

    with psycopg.connect(dsn, autocommit=True) as con:
        with con.cursor() as cur:
            cur.execute("DELETE FROM sandbox_runs WHERE id = %s", (run_id,))
