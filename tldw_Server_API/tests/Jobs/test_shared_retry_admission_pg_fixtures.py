"""Guard the retry-admission opt-in's shared PostgreSQL fixture lifecycle."""

from __future__ import annotations

import os
from contextlib import closing
from typing import Any

import psycopg
import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs import conftest as jobs_fixtures

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]
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
    print(f"shared explicit native database={isolated_test_environment[1]}; closed={conn.closed}; no pg_temp_db")


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
    print(f"shared autouse wrapper database={isolated_test_environment[1]}; closed={conn.closed}; no pg_temp_db")


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


@pytest.mark.integration
@pytest.mark.jobs
@pytest.mark.parametrize("route", ["explicit", "autouse", "env_override"])
def test_legacy_jobs_routes_keep_alternate_lifecycle(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    """Non-opted-in Jobs still use pg_temp_db and retain the env bypass."""
    monkeypatch.setattr(request.module, "USE_SHARED_JOBS_POSTGRES", False)
    monkeypatch.delenv("JOBS_PG_USE_ENV_DB", raising=False)
    if route == "explicit":
        dsn = request.getfixturevalue("jobs_pg_dsn")
    else:
        request.node.add_marker(pytest.mark.pg_jobs)
        if route == "env_override":
            dsn = str(request.getfixturevalue("pg_temp_db")["dsn"])
            monkeypatch.setenv("JOBS_DB_URL", dsn)
            monkeypatch.setenv("JOBS_PG_USE_ENV_DB", "1")
        jobs_fixtures._pg_jobs_db_url.__wrapped__(request, monkeypatch)
        dsn = os.environ["JOBS_DB_URL"]
    assert "isolated_test_environment" not in request.fixturenames
    assert dsn == str(request.getfixturevalue("pg_temp_db")["dsn"])
    with psycopg.connect(dsn) as conn:
        assert conn.execute("SELECT current_database()").fetchone() == (
            request.getfixturevalue("pg_temp_db")["database"],
        )
    assert conn.closed
