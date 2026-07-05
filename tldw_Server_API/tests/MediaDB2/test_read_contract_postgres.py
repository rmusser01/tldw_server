from __future__ import annotations

import os

import pytest

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


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
def test_search_media_contract_matches_postgres_shape():
    from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_search_repository import (
        MediaSearchRepository,
    )

    dsn = (os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or "").strip()
    assert dsn, "Postgres test database URL not configured"

    base_config = DatabaseConfig.from_env()
    assert base_config.backend_type == BackendType.POSTGRESQL

    backend = DatabaseBackendFactory.create_backend(base_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)

    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            media_id, _, _ = db.add_media_with_keywords(
                title="Postgres Contract Doc",
                content="postgres contract content",
                media_type="text",
                keywords=[],
            )
            rows, total = MediaSearchRepository.from_legacy_db(db).search(
                search_query=None,
                search_fields=[],
                page=1,
                results_per_page=10,
            )

        assert isinstance(rows, list)
        assert isinstance(total, int)
        assert total >= 1
        assert rows[0]["id"] == media_id
        assert rows[0]["title"] == "Postgres Contract Doc"
    finally:
        try:
            db.close_connection()
        except Exception:
            _ = None
        try:
            backend.close_all()
        except Exception:
            _ = None
