"""Cached ChaCha health checks must use PostgreSQL transaction semantics."""

import asyncio

import pytest

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def postgres_chacha(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_postgres_health_check_keeps_connection_usable(postgres_chacha):
    assert deps._health_check_instance(postgres_chacha) is True
    assert deps._health_check_instance(postgres_chacha) is True
    assert postgres_chacha.get_connection().execute("SELECT 1").fetchone() is not None


def test_postgres_failed_health_probe_reports_unhealthy_then_cleans_transaction(postgres_chacha):
    connection = postgres_chacha.get_connection()
    with pytest.raises(DatabaseError):
        connection.execute("SELECT 1 / 0")

    assert deps._health_check_instance(postgres_chacha) is False
    assert connection.execute("SELECT 1").fetchone() is not None
    assert deps._health_check_instance(postgres_chacha) is True


def test_repeated_postgres_cached_dependency_returns_same_healthy_instance(
    postgres_chacha, monkeypatch, tmp_path,
):
    monkeypatch.setattr(deps.DatabasePaths, "get_user_base_directory", lambda _user: tmp_path)
    monkeypatch.setattr(deps, "_chacha_db_instances", {str(tmp_path): postgres_chacha})

    def unexpected_rebuild(*_args):
        pytest.fail("Healthy PostgreSQL cache must not be rebuilt")

    monkeypatch.setattr(deps, "_create_and_prepare_db", unexpected_rebuild)

    async def repeated_reads():
        assert await deps._get_or_init_db_instance(1, "1") is postgres_chacha
        assert await deps._get_or_init_db_instance(1, "1") is postgres_chacha

    asyncio.run(repeated_reads())


def test_postgres_health_probe_does_not_commit_callers_transaction(postgres_chacha):
    connection = postgres_chacha.get_connection()
    connection.execute("CREATE TEMP TABLE health_pending (value INTEGER)")
    connection.commit()
    with pytest.raises(RuntimeError, match="rollback caller"):
        with postgres_chacha.transaction() as transaction:
            transaction.execute("INSERT INTO health_pending VALUES (1)")
            assert deps._health_check_instance(postgres_chacha) is True
            raise RuntimeError("rollback caller")
    row = connection.execute("SELECT COUNT(*) AS count FROM health_pending").fetchone()
    assert row["count"] == 0
