"""Unit contracts for PostgreSQL Jobs RLS policy construction."""

from __future__ import annotations

import psycopg
import pytest

from tldw_Server_API.app.core.Jobs import pg_migrations

pytestmark = pytest.mark.unit


def test_pg_rls_policy_setup_scopes_idempotency_receipts(monkeypatch):
    executed: list[str] = []

    class _RecordingCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            del params
            executed.append(" ".join(str(query).split()))

    class _RecordingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return _RecordingCursor()

    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda _dsn, **_kwargs: _RecordingConnection(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.pg_util.negotiate_pg_dsn",
        lambda dsn: dsn,
    )
    monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
    monkeypatch.delenv("JOBS_PG_RLS_DEBUG", raising=False)

    pg_migrations.ensure_jobs_rls_policies_pg("postgresql://jobs.test/jobs")

    combined = "\n".join(executed)
    assert "ALTER TABLE job_idempotency_receipts ENABLE ROW LEVEL SECURITY" in combined
    assert "ALTER TABLE job_idempotency_receipts FORCE ROW LEVEL SECURITY" in combined
    assert "CREATE POLICY job_idempotency_receipts_select" in combined
    assert "CREATE POLICY job_idempotency_receipts_modify" in combined
    receipt_policy_sql = "\n".join(
        query
        for query in executed
        if "CREATE POLICY job_idempotency_receipts" in query
    )
    assert "app.domain_allowlist" in receipt_policy_sql
    assert "app.owner_user_id" in receipt_policy_sql
    assert "WITH CHECK" in receipt_policy_sql


def test_pg_rls_role_receives_receipt_insert_and_sequence_grants(monkeypatch):
    executed: list[str] = []

    class _RecordingCursor:
        def __init__(self):
            self.last_query = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            del params
            self.last_query = " ".join(str(query).split())
            executed.append(self.last_query)

        def fetchone(self):
            if "current_schema" in self.last_query:
                return ("public",)
            if "FROM pg_roles" in self.last_query:
                return (1,)
            if "current_user" in self.last_query:
                return ("jobs_admin",)
            return None

    class _RecordingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return _RecordingCursor()

    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda _dsn, **_kwargs: _RecordingConnection(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.pg_util.negotiate_pg_dsn",
        lambda dsn: dsn,
    )
    monkeypatch.setenv("JOBS_PG_RLS_ROLE", "jobs_rls")
    monkeypatch.delenv("JOBS_PG_RLS_DEBUG", raising=False)

    pg_migrations.ensure_jobs_rls_policies_pg("postgresql://jobs.test/jobs")

    grant_sql = "\n".join(query for query in executed if "GRANT" in query)
    assert "GRANT INSERT ON TABLE" in grant_sql
    assert "job_idempotency_receipts" in grant_sql
    assert "GRANT USAGE, SELECT ON SEQUENCE" in grant_sql
    assert "job_idempotency_receipts_receipt_id_seq" in grant_sql
