"""Guard the retry-admission opt-in's shared PostgreSQL fixture lifecycle."""

from __future__ import annotations

import os
from contextlib import closing
from types import SimpleNamespace
from typing import Any, Literal, NoReturn, cast

import asyncpg
import psycopg
import pytest

from tldw_Server_API.app.core.Jobs import pg_migrations
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs import conftest as jobs_fixtures

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
USE_SHARED_JOBS_POSTGRES = True


@pytest.mark.integration
@pytest.mark.pg_jobs
def test_shared_explicit_dsn_uses_fixture_owned_database(
    jobs_pg_dsn: str,
    isolated_test_environment: tuple[Any, str],
    request: pytest.FixtureRequest,
) -> None:
    """The explicit DSN must not allocate the alternate Jobs scratch database."""
    assert "pg_temp_db" not in request.fixturenames, "alternate pg_temp_db was requested"
    assert os.environ["JOBS_DB_URL"] == jobs_pg_dsn == os.environ["TEST_DATABASE_URL"]
    with psycopg.connect(jobs_pg_dsn) as conn:
        assert conn.execute("SELECT current_database()").fetchone() == (isolated_test_environment[1],)
    assert conn.closed


@pytest.mark.integration
@pytest.mark.pg_jobs
def test_shared_autouse_url_uses_fixture_owned_database(
    isolated_test_environment: tuple[Any, str], request: pytest.FixtureRequest,
) -> None:
    """Autouse alone must bind Jobs to the shared database before test setup."""
    assert "pg_temp_db" not in request.fixturenames, "alternate pg_temp_db was requested"
    assert os.environ["JOBS_DB_URL"] == os.environ["TEST_DATABASE_URL"]
    jobs = JobManager()
    assert jobs.backend == "postgres"
    with closing(jobs._connect()) as conn, conn, jobs._pg_cursor(conn) as cur:
        cur.execute("SELECT current_database() AS name")
        assert cur.fetchone()["name"] == isolated_test_environment[1]
    assert conn.closed


@pytest.mark.integration
@pytest.mark.pg_jobs
def test_shared_autouse_ignores_legacy_env_bypass(
    jobs_pg_dsn: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in cannot escape shared ownership via the historical env bypass."""
    with monkeypatch.context() as patch:
        patch.setenv("JOBS_PG_USE_ENV_DB", "1")
        patch.setenv("JOBS_DB_URL", "postgresql://ignored.invalid/foreign")
        jobs_fixtures._pg_jobs_db_url.__wrapped__(request, patch)
        assert os.environ["JOBS_DB_URL"] == jobs_pg_dsn
    assert "pg_temp_db" not in request.fixturenames


@pytest.mark.unit
def test_shared_opt_in_without_pg_marker_allocates_no_database(request: pytest.FixtureRequest) -> None:
    """SQLite/non-PG tests must not instantiate either PostgreSQL lifecycle."""
    assert "pg_temp_db" not in request.fixturenames
    assert "isolated_test_environment" not in request.fixturenames


class _HistoricalRouteRequest:
    """Resolve only a DSN sentinel, never either PostgreSQL fixture lifecycle."""

    def __init__(self) -> None:
        """Supply the non-opted-in module and PG marker used by real fixture bodies."""
        self.module = SimpleNamespace(USE_SHARED_JOBS_POSTGRES=False)
        self.keywords = {"pg_jobs": True}
        self.resolved: list[str] = []

    def getfixturevalue(self, name: str) -> dict[str, str]:
        """Reject unexpected resolution; pg_temp_db is data, not an allocated DB."""
        self.resolved.append(name)
        assert name == "pg_temp_db", f"Unexpected fixture lifecycle: {name}"
        return {"dsn": "postgresql://sentinel.invalid/historical", "database": "historical"}


@pytest.mark.unit
@pytest.mark.parametrize("route", ["explicit", "autouse", "env_override"])
def test_legacy_jobs_routes_select_dsn_without_database(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
    route: Literal["explicit", "autouse", "env_override"],
) -> None:
    """Exercise historical fixture routing without allocating or connecting a DB.

    Only native schema I/O is replaced. The real fixture bodies choose the
    dependency and bind the DSN, with distinct sentinels for the environment
    bypass. The actual test has no pg_jobs/jobs marker, so RUN_JOBS=0 still
    exercises these unit controls without either DB lifecycle at setup.
    """
    selected = _HistoricalRouteRequest()
    fixture_request = cast(pytest.FixtureRequest, selected)
    migrations: list[tuple[str, str]] = []

    def ensure_tables(dsn: str) -> None:
        """Record the selected DSN at the native schema I/O boundary."""
        migrations.append(("tables", dsn))

    def ensure_counters(dsn: str) -> None:
        """Record the selected DSN at the native counter I/O boundary."""
        migrations.append(("counters", dsn))

    def forbid_connection(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail immediately if a routing unit control attempts real PG I/O."""
        pytest.fail("Historical route unit control attempted a PostgreSQL connection")

    monkeypatch.setattr(pg_migrations, "ensure_jobs_tables_pg", ensure_tables)
    monkeypatch.setattr(pg_migrations, "ensure_job_counters_pg", ensure_counters)
    monkeypatch.setattr(psycopg, "connect", forbid_connection)
    monkeypatch.setattr(asyncpg, "connect", forbid_connection)
    monkeypatch.delenv("JOBS_PG_USE_ENV_DB", raising=False)
    monkeypatch.setenv("JOBS_DB_URL", "postgresql://sentinel.invalid/environment")
    if route == "explicit":
        dsn = jobs_fixtures.jobs_pg_dsn.__wrapped__(fixture_request, monkeypatch)
        assert dsn == "postgresql://sentinel.invalid/historical"
    else:
        if route == "env_override":
            monkeypatch.setenv("JOBS_PG_USE_ENV_DB", "1")
        jobs_fixtures._pg_jobs_db_url.__wrapped__(fixture_request, monkeypatch)
    if route == "env_override":
        assert selected.resolved == []
        assert os.environ["JOBS_DB_URL"] == "postgresql://sentinel.invalid/environment"
        assert migrations == []
    else:
        assert selected.resolved == ["pg_temp_db"]
        assert os.environ["JOBS_DB_URL"] == "postgresql://sentinel.invalid/historical"
        assert migrations == [
            ("tables", "postgresql://sentinel.invalid/historical"),
            ("counters", "postgresql://sentinel.invalid/historical"),
        ]
    assert "pg_temp_db" not in request.fixturenames
    assert "isolated_test_environment" not in request.fixturenames
