"""Protect the AuthNZ fixture against a session connecting just before DROP."""

import os
import re
from urllib.parse import unquote, urlsplit

import asyncpg
import pytest

from tldw_Server_API.tests._plugins.postgres import _drop_database


@pytest.fixture
def pending_cleanup():
    pending = []
    yield pending
    if pending:
        pending[0]()
        pytest.fail("Auth fixture teardown did not drop the database at the tested boundary")


@pytest.fixture
def monkeypatch(monkeypatch, pending_cleanup):
    # The Auth fixture depends on monkeypatch, so the probe outlives its teardown.
    return monkeypatch


@pytest.mark.integration
def test_isolated_auth_database_drops_after_a_late_connection(
    isolated_test_environment, monkeypatch, pending_cleanup,
):
    _, db_name = isolated_test_environment
    if not re.fullmatch(r"tldw_test_[0-9a-f]{8}", db_name):
        pytest.fail("Expected a uniquely named Auth fixture database")
    connect = asyncpg.connect
    fixture_dsn = os.environ["TEST_DATABASE_URL"]

    def recover_fixture_database():
        # Clean up our fixture database if the regression reappears.
        url = urlsplit(fixture_dsn)
        _drop_database(
            url.hostname, url.port, unquote(url.username), unquote(url.password), db_name,
        )

    pending_cleanup.append(recover_fixture_database)

    class AdminConnection(asyncpg.Connection):
        async def execute(self, query, *args, **kwargs):
            if not query.lstrip().upper().startswith("DROP DATABASE ") or db_name not in query:
                return await super().execute(query, *args, **kwargs)

            # Connect after any separate session-termination query has finished.
            late_session = await connect(fixture_dsn, timeout=5)
            try:
                if await late_session.fetchval("SELECT current_database()") != db_name:
                    pytest.fail("Late session connected outside the Auth fixture database")
                result = await super().execute(query, *args, **kwargs)
                if await self.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", db_name,
                ):
                    pytest.fail("Auth fixture database still exists after teardown")
                pending_cleanup.clear()
                return result
            finally:
                await late_session.close(timeout=5)

    async def connect_with_late_session(*args, **kwargs):
        if kwargs.get("database") == "postgres":
            kwargs["connection_class"] = AdminConnection
        return await connect(*args, **kwargs)

    monkeypatch.setattr(asyncpg, "connect", connect_with_late_session)
