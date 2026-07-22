"""PostgreSQL regressions for Prompt Studio tenant session reuse."""

from __future__ import annotations

import uuid
from importlib import import_module
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    ensure_prompt_studio_rls,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

prompt_db_module = import_module(
    "tldw_Server_API.app.core.DB_Management.PromptStudioDatabase"
)
pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def _scope(*, user_id: int, session_role: str | None):
    return scoped_context(
        user_id=user_id,
        org_ids=[],
        team_ids=[],
        is_admin=False,
        session_role=session_role,
    )


def _borrowed_session(db: PromptStudioDatabase) -> tuple[int, int, str]:
    conn = db.get_connection()
    row = conn.execute(
        "SELECT pg_backend_pid() AS backend_pid, "
        "current_setting('app.current_user_id', true) AS tenant"
    ).fetchone()
    assert row is not None
    return id(conn.raw_connection), int(row["backend_pid"]), str(row["tenant"])


def test_psycopg_sql_tenant_setting_is_reapplied_on_shared_pool_reuse(
    pg_database_config: DatabaseConfig,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second Prompt facade must replace, never inherit, the first tenant."""

    if prompt_db_module.psycopg_sql is None:
        pytest.skip("psycopg v3 SQL composition is unavailable")

    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
    pg_database_config.pool_size = 1
    pg_database_config.max_overflow = 0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    tenant_a = "prompt-tenant-a"
    tenant_b = "prompt-tenant-b"
    db_a = PromptStudioDatabase(
        db_path=str(tmp_path / "prompt-tenant-a.sqlite"),
        client_id="prompt-audit-a",
        tenant_user_id=tenant_a,
        backend=backend,
    )
    db_b = PromptStudioDatabase(
        db_path=str(tmp_path / "prompt-tenant-b.sqlite"),
        client_id="prompt-audit-b",
        tenant_user_id=tenant_b,
        backend=backend,
    )

    test_role: str | None = None
    try:
        assert ensure_prompt_studio_rls(backend) is True
        with backend.transaction() as conn:
            bypasses_rls = bool(
                backend.execute(
                    "SELECT rolsuper OR rolbypassrls "
                    "FROM pg_roles WHERE rolname = current_user",
                    connection=conn,
                ).scalar
            )

        if bypasses_rls:
            test_role = f"ps_rls_{uuid.uuid4().hex[:12]}"
            ident = backend.escape_identifier
            try:
                with backend.transaction() as conn:
                    backend.execute(
                        f"CREATE ROLE {ident(test_role)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                        connection=conn,
                    )
                    backend.execute(
                        f"GRANT USAGE ON SCHEMA public TO {ident(test_role)}",
                        connection=conn,
                    )
                    backend.execute(
                        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                        f"IN SCHEMA public TO {ident(test_role)}",
                        connection=conn,
                    )
                    backend.execute(
                        "GRANT USAGE, SELECT ON ALL SEQUENCES "
                        f"IN SCHEMA public TO {ident(test_role)}",
                        connection=conn,
                    )
                    backend.execute(
                        f"GRANT {ident(test_role)} TO CURRENT_USER",
                        connection=conn,
                    )
            except BackendDatabaseError as exc:
                pytest.skip(f"PostgreSQL test role could not be provisioned: {exc}")

        with _scope(user_id=901, session_role=test_role):
            project_a = db_a.create_project("Tenant A", user_id=tenant_a)
            raw_a, backend_pid_a, setting_a = _borrowed_session(db_a)
            visible_a = db_a.list_projects(per_page=10)["projects"]
            db_a.close_connection()

        with _scope(user_id=902, session_role=test_role):
            project_b = db_b.create_project("Tenant B", user_id=tenant_b)
            raw_b, backend_pid_b, setting_b = _borrowed_session(db_b)
            visible_b = db_b.list_projects(per_page=10)["projects"]
            db_b.close_connection()

        with _scope(user_id=903, session_role=test_role):
            raw_a_again, backend_pid_a_again, setting_a_again = _borrowed_session(db_a)
            visible_a_again = db_a.list_projects(per_page=10)["projects"]
            db_a.close_connection()

        assert raw_a == raw_b == raw_a_again
        assert backend_pid_a == backend_pid_b == backend_pid_a_again
        assert (setting_a, setting_b, setting_a_again) == (
            tenant_a,
            tenant_b,
            tenant_a,
        )
        assert [row["id"] for row in visible_a] == [project_a["id"]]
        assert [row["id"] for row in visible_b] == [project_b["id"]]
        assert [row["id"] for row in visible_a_again] == [project_a["id"]]
    finally:
        db_a.close_connection()
        db_b.close_connection()
        if test_role is not None:
            ident = backend.escape_identifier
            try:
                with backend.transaction() as conn:
                    backend.execute(
                        f"DROP OWNED BY {ident(test_role)}",
                        connection=conn,
                    )
                    backend.execute(
                        f"DROP ROLE IF EXISTS {ident(test_role)}",
                        connection=conn,
                    )
            except BackendDatabaseError:
                pass
        pool.close_all()
