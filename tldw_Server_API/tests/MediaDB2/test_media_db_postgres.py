import os
import urllib.parse

import pytest

# Reuse Postgres AuthNZ fixtures (setup/cleanup) as a plugin so
# this module's @usefixtures works without direct imports.
#
# Rationale: Running this file directly or from a subtree may skip
# project-level plugins declared in pyproject.toml (depending on CWD and
# environment, e.g., PYTEST_DISABLE_PLUGIN_AUTOLOAD=1). Declaring the plugin
# here keeps this test hermetic and consistently runnable in CI and locally
# without relying on a root conftest.
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


TEST_RLS_ROLE = "tldw_rls_tester"


@pytest.fixture(autouse=True)
def _enable_pg_role_switch(monkeypatch):
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")


def _has_postgres_dependencies() -> bool:


    try:
        import psycopg  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.skipif(
    not _has_postgres_dependencies(),
    reason="Postgres dependencies missing (install psycopg[binary])",
)
@pytest.mark.usefixtures("setup_test_database", "clean_database")
def test_media_rls_enforces_scope_postgres():
    dsn = (os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or "").strip()
    assert dsn, "Postgres test database URL not configured"

    base_config = DatabaseConfig.from_env()
    assert base_config.backend_type == BackendType.POSTGRESQL, "Expected PostgreSQL backend configuration"

    parsed = urllib.parse.urlparse(dsn)
    db_name = (parsed.path or "/").lstrip("/") or "postgres"
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 5432)

    admin_backend = DatabaseBackendFactory.create_backend(base_config)
    bootstrap_db = MediaDatabase(db_path=":memory:", client_id="bootstrap", backend=admin_backend)
    test_role = TEST_RLS_ROLE
    test_password = "ContentRlsR3g!"
    escaped_password = test_password.replace("'", "''")
    ident = admin_backend.escape_identifier  # type: ignore[attr-defined]

    with admin_backend.transaction() as conn:
        role_exists = admin_backend.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s::text LIMIT 1",
            (test_role,),
            connection=conn,
        ).scalar is not None
        if not role_exists:
            admin_backend.execute(
                f"CREATE ROLE {ident(test_role)} LOGIN PASSWORD '{escaped_password}'",
                connection=conn,
            )
        else:
            admin_backend.execute(
                f"ALTER ROLE {ident(test_role)} WITH LOGIN PASSWORD '{escaped_password}'",
                connection=conn,
            )
        admin_backend.execute(
            f"GRANT USAGE ON SCHEMA public TO {ident(test_role)}",
            connection=conn,
        )
        admin_backend.execute(
            f"GRANT CREATE ON SCHEMA public TO {ident(test_role)}",
            connection=conn,
        )
        admin_backend.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {ident(test_role)}",
            connection=conn,
        )
        admin_backend.execute(
            f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {ident(test_role)}",
            connection=conn,
        )
        admin_backend.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE CURRENT_USER IN SCHEMA public "
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {ident(test_role)}",
            connection=conn,
        )
        admin_backend.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE CURRENT_USER IN SCHEMA public "
            f"GRANT USAGE, SELECT ON SEQUENCES TO {ident(test_role)}",
            connection=conn,
        )
        admin_backend.execute(
            f"GRANT {ident(test_role)} TO CURRENT_USER",
            connection=conn,
        )

    try:
        bootstrap_db.close_connection()
    except Exception:
        _ = None

    backend = admin_backend

    db_owner = MediaDatabase(db_path=":memory:", client_id="101", backend=backend)
    db_other_user = MediaDatabase(db_path=":memory:", client_id="202", backend=backend)
    db_admin = MediaDatabase(db_path=":memory:", client_id="999", backend=backend)
    db_team_owner = MediaDatabase(db_path=":memory:", client_id="303", backend=backend)
    db_team_reader = MediaDatabase(db_path=":memory:", client_id="404", backend=backend)
    db_org_owner = MediaDatabase(db_path=":memory:", client_id="606", backend=backend)
    db_org_member = MediaDatabase(db_path=":memory:", client_id="707", backend=backend)

    try:
        with scoped_context(
            user_id=101,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            media_id, _, _ = db_owner.add_media_with_keywords(
                title="personal doc",
                content="sensitive content",
                media_type="text",
                chunks=None,
                keywords=[],
            )

        with scoped_context(
            user_id=202,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            current_role = db_other_user.backend.execute("SELECT current_role").scalar
            assert current_role == TEST_RLS_ROLE
            current_scope = db_other_user.backend.execute(
                "SELECT current_setting('app.current_user_id', true)"
            ).scalar
            assert current_scope == "202"
            assert (
                db_other_user.get_media_by_id(media_id) is None
            ), "Unrelated user should not see another user's media"
            hidden_rows, hidden_total = media_db_api.search_media(
                db_other_user,
                search_query=None,
                search_fields=[],
                page=1,
                results_per_page=20,
            )
            assert hidden_total == 0
            assert hidden_rows == []

        with scoped_context(
            user_id=101,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            owner_row = db_owner.get_media_by_id(media_id)
            assert owner_row is not None and owner_row["id"] == media_id
            owner_rows, owner_total = media_db_api.search_media(
                db_owner,
                search_query=None,
                search_fields=[],
                page=1,
                results_per_page=20,
            )
            assert owner_total == 1
            assert owner_rows[0]["id"] == media_id

        with scoped_context(
            user_id=999,
            org_ids=[],
            team_ids=[],
            is_admin=True,
        ):
            admin_row = db_admin.get_media_by_id(media_id)
            assert admin_row is not None and admin_row["id"] == media_id

        with scoped_context(
            user_id=606,
            org_ids=[12],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            org_media_id, _, _ = db_org_owner.add_media_with_keywords(
                title="org shared doc",
                content="org scope content",
                media_type="text",
                chunks=None,
                keywords=[],
                visibility="org",
                owner_user_id=606,
            )

        with scoped_context(
            user_id=707,
            org_ids=[12],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            org_row = db_org_member.get_media_by_id(org_media_id)
            assert org_row is not None and org_row["id"] == org_media_id

        with scoped_context(
            user_id=808,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            assert (
                db_other_user.get_media_by_id(org_media_id) is None
            ), "User without org membership should not see org media"

        with scoped_context(
            user_id=303,
            org_ids=[],
            team_ids=[77],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            team_media_id, _, _ = db_team_owner.add_media_with_keywords(
                title="team shared doc",
                content="shared content",
                media_type="text",
                chunks=None,
                keywords=[],
                visibility="team",
                owner_user_id=303,
            )

        with scoped_context(
            user_id=505,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            assert (
                db_other_user.get_media_by_id(team_media_id) is None
            ), "User without team membership should be denied"

        with scoped_context(
            user_id=404,
            org_ids=[],
            team_ids=[77],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            accessible = db_team_reader.get_media_by_id(team_media_id)
            assert accessible is not None and accessible["id"] == team_media_id
            team_rows, team_total = media_db_api.search_media(
                db_team_reader,
                search_query=None,
                search_fields=[],
                page=1,
                results_per_page=20,
            )
            assert team_total == 1
            assert team_rows[0]["id"] == team_media_id

        with scoped_context(
            user_id=202,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            update_attempt = db_other_user.backend.execute(
                "UPDATE media SET title = %s WHERE id = %s",
                ("malicious edit", media_id),
            )
            assert update_attempt.rowcount == 0

        with scoped_context(
            user_id=808,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=TEST_RLS_ROLE,
        ):
            org_update_attempt = db_other_user.backend.execute(
                "UPDATE media SET title = %s WHERE id = %s",
                ("bad org edit", org_media_id),
            )
            assert org_update_attempt.rowcount == 0

        with scoped_context(user_id=999, org_ids=[], team_ids=[], is_admin=True):
            admin_update = db_admin.backend.execute(
                "UPDATE media SET title = %s WHERE id = %s",
                ("admin retitle", media_id),
            )
            assert admin_update.rowcount == 1
            updated = db_admin.get_media_by_id(media_id)
            assert updated is not None and updated["title"] == "admin retitle"

    finally:
        for db in (
            db_owner,
            db_other_user,
            db_admin,
            db_team_owner,
            db_team_reader,
            db_org_owner,
            db_org_member,
        ):
            try:
                db.close_connection()
            except Exception:
                _ = None
        try:
            backend.close_all()
        except Exception:
            _ = None


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.skipif(
    not _has_postgres_dependencies(),
    reason="Postgres dependencies missing (install psycopg[binary])",
)
@pytest.mark.usefixtures("setup_test_database", "clean_database")
def test_add_media_with_chunks_postgres_persists_unvectorized_rows():
    dsn = (os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or "").strip()
    assert dsn, "Postgres test database URL not configured"

    base_config = DatabaseConfig.from_env()
    assert base_config.backend_type == BackendType.POSTGRESQL, "Expected PostgreSQL backend configuration"

    backend = DatabaseBackendFactory.create_backend(base_config)
    db = MediaDatabase(db_path=":memory:", client_id="909", backend=backend)

    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="chunked postgres doc",
            content="chunk one chunk two",
            media_type="text",
            keywords=[],
            overwrite=True,
            chunks=[
                {"text": "chunk one", "start_char": 0, "end_char": 9, "chunk_type": "text"},
                {"text": "chunk two", "start_char": 10, "end_char": 19, "chunk_type": "text"},
            ],
        )

        assert media_id is not None
        assert db.get_unvectorized_chunk_count(media_id) == 2
    finally:
        try:
            db.close_connection()
        except Exception:
            _ = None
        try:
            backend.close_all()
        except Exception:
            _ = None
