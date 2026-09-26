"""Guard the retry-admission opt-in's shared PostgreSQL fixture lifecycle."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import closing
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

import psycopg
import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures", "pytester"]
USE_SHARED_JOBS_POSTGRES = True


@pytest.fixture(autouse=True)
def _legacy_env_before_shared_setup(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed the legacy bypass before normal Jobs autouse setup for its native guard."""
    if request.node.name == "test_shared_autouse_ignores_legacy_env_bypass":
        request.getfixturevalue("isolated_test_environment")
        monkeypatch.setenv("JOBS_PG_USE_ENV_DB", "1")
        monkeypatch.setenv("JOBS_DB_URL", "postgresql://ignored.invalid/foreign")


@pytest.fixture(autouse=True)
def _pg_jobs_db_url(_legacy_env_before_shared_setup: None, _pg_jobs_db_url: None) -> None:
    """Resolve the unchanged predecessor after seeding, via normal fixture override."""
    return _pg_jobs_db_url


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
    jobs_pg_dsn: str,
    isolated_test_environment: tuple[Any, str],
    request: pytest.FixtureRequest,
) -> None:
    """The opt-in cannot escape shared ownership via the historical env bypass."""
    assert os.environ["JOBS_PG_USE_ENV_DB"] == "1"
    assert os.environ["JOBS_DB_URL"] == jobs_pg_dsn == os.environ["TEST_DATABASE_URL"]
    with psycopg.connect(jobs_pg_dsn) as conn:
        assert conn.execute("SELECT current_database()").fetchone() == (isolated_test_environment[1],)
    assert conn.closed
    assert "pg_temp_db" not in request.fixturenames


@pytest.mark.unit
def test_shared_opt_in_without_pg_marker_allocates_no_database(request: pytest.FixtureRequest) -> None:
    """SQLite/non-PG tests must not instantiate either PostgreSQL lifecycle."""
    assert "pg_temp_db" not in request.fixturenames
    assert "isolated_test_environment" not in request.fixturenames


def _run_fixture_probe(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *args: str,
) -> Any:
    """Run bounded public fixture resolution; optionally retain generated evidence."""
    root = str(Path(__file__).resolve().parents[3])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [root, os.getenv("PYTHONPATH")])))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    config = pytester.makeini("[pytest]\nasyncio_mode = auto\nmarkers =\n    pg_jobs: routing probe\n")
    result = pytester.runpytest_subprocess("-q", "--tb=short", f"--junitxml={name}.xml", *args, timeout=90)
    if evidence_root := os.getenv("TASK20_PROBE_EVIDENCE"):
        evidence = Path(evidence_root) / name
        evidence.mkdir(parents=True, exist_ok=True)
        for source in pytester.path.glob("*.py"):
            shutil.copy2(source, evidence / source.name)
        shutil.copy2(config, evidence / config.name)
        shutil.copy2(pytester.path / f"{name}.xml", evidence / "results.xml")
        (evidence / "output.log").write_text(result.stdout.str() + "\n" + result.stderr.str(), encoding="utf-8")
        (evidence / "result.json").write_text(
            json.dumps(
                {
                    "exit_code": int(result.ret),
                    "outcomes": result.parseoutcomes(),
                    "duration": result.duration,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


_ROUTING_IO_BOUNDARY = dedent("""
    import os
    import asyncpg
    import psycopg
    import pytest
    from collections.abc import Iterator
    from typing import Any, NoReturn
    from tldw_Server_API.app.core.Jobs import pg_migrations
    from tldw_Server_API.tests.Jobs import conftest as original

    jobs_pg_dsn = original.jobs_pg_dsn
    _pg_jobs_db_url = original._pg_jobs_db_url
    events = []
    historical = "postgresql://sentinel.invalid/historical"
    shared = "postgresql://sentinel.invalid/shared"
    environment = "postgresql://sentinel.invalid/environment"

    @pytest.fixture(autouse=True)
    def _fixture_io_boundary(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
            pytest.fail("Routing probe attempted PostgreSQL I/O")
        monkeypatch.setattr(asyncpg, "connect", forbidden)
        monkeypatch.setattr(psycopg, "connect", forbidden)
        monkeypatch.setattr(pg_migrations, "ensure_jobs_tables_pg", lambda dsn: events.append(("tables", dsn)))
        monkeypatch.setattr(pg_migrations, "ensure_job_counters_pg", lambda dsn: events.append(("counters", dsn)))
        monkeypatch.setenv("JOBS_DB_URL", environment)
        monkeypatch.setenv("JOBS_PG_USE_ENV_DB", "1" if bypass else "0")
        yield
        if any(event == expected + "_setup" for event in events):
            assert events[-1] == expected + "_teardown", "Selected lifecycle did not tear down"

    @pytest.fixture
    def pg_temp_db() -> Iterator[dict[str, str]]:
        assert expected == "historical", "Forbidden alternative lifecycle resolution"
        events.append("historical_setup")
        yield {"dsn": historical, "database": "historical"}
        events.append("historical_teardown")

    @pytest.fixture
    def isolated_test_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[None, str]]:
        assert expected == "shared", "Forbidden shared lifecycle resolution"
        events.append("shared_setup")
        monkeypatch.setenv("TEST_DATABASE_URL", shared)
        yield None, "shared"
        events.append("shared_teardown")
""")


@pytest.mark.unit
@pytest.mark.parametrize("route", ["explicit", "autouse", "env_override"])
def test_legacy_jobs_routes_select_dsn_without_database(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    route: Literal["explicit", "autouse", "env_override"],
) -> None:
    """Exercise historical fixture routing without allocating or connecting a DB.

    Only native schema I/O is replaced. The real fixture bodies choose the
    dependency and bind the DSN, with distinct sentinels for the environment
    bypass. The actual test has no pg_jobs/jobs marker, so RUN_JOBS=0 still
    exercises these unit controls without either DB lifecycle at setup.
    """
    expected = "none" if route == "env_override" else "historical"
    pytester.makeconftest(_ROUTING_IO_BOUNDARY + f"\nexpected = {expected!r}\nbypass = {route == 'env_override'!r}\n")
    marker = "@pytest.mark.pg_jobs" if route != "explicit" else ""
    argument = ", jobs_pg_dsn: str" if route == "explicit" else ""
    source = f"""
        import os
        import pytest
        from conftest import events, historical, environment
        USE_SHARED_JOBS_POSTGRES = {{opt_in}}

        {marker}
        def test_route(request: pytest.FixtureRequest{argument}) -> None:
            assert "isolated_test_environment" not in request.fixturenames
            if {route == "env_override"!r}:
                assert os.environ["JOBS_DB_URL"] == environment
                assert events == []
                assert "pg_temp_db" not in request.fixturenames
            else:
                assert os.environ["JOBS_DB_URL"] == historical
                assert events == ["historical_setup", ("tables", historical), ("counters", historical)]
                assert "pg_temp_db" in request.fixturenames
            assert ("jobs_pg_dsn" in request.fixturenames) == {route == "explicit"!r}
    """
    if route == "explicit":
        pytester.makepyfile(test_probe=source.format(opt_in=True))
        red = _run_fixture_probe(pytester, monkeypatch, "historical-wrong-route")
        red.assert_outcomes(errors=1)
        assert "Forbidden shared lifecycle resolution" in red.stdout.str()
    pytester.makepyfile(test_probe=source.format(opt_in=False))
    _run_fixture_probe(pytester, monkeypatch, f"historical-{route}").assert_outcomes(passed=1)


@pytest.mark.unit
def test_shared_bypass_is_rejected_before_resolution(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed bypass before autouse; it must still resolve shared setup and teardown."""
    pytester.makeconftest(_ROUTING_IO_BOUNDARY + '\nexpected = "shared"\nbypass = True\n')
    source = """
        import os
        import pytest
        from conftest import events, shared
        USE_SHARED_JOBS_POSTGRES = {opt_in}

        @pytest.mark.pg_jobs
        def test_route(request: pytest.FixtureRequest) -> None:
            assert os.environ["JOBS_DB_URL"] == shared, "Shared bypass escaped ownership"
            assert events == ["shared_setup", ("tables", shared), ("counters", shared)]
            assert "jobs_pg_dsn" in request.fixturenames
            assert "isolated_test_environment" in request.fixturenames
            assert "pg_temp_db" not in request.fixturenames
    """
    pytester.makepyfile(test_probe=source.format(opt_in=False))
    red = _run_fixture_probe(pytester, monkeypatch, "shared-bypass-wrong-route")
    red.assert_outcomes(failed=1)
    assert "Shared bypass escaped ownership" in red.stdout.str()
    pytester.makepyfile(test_probe=source.format(opt_in=True))
    _run_fixture_probe(pytester, monkeypatch, "shared-bypass").assert_outcomes(passed=1)


@pytest.mark.unit
def test_sqlite_probe_rejects_unexpected_allocation(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-PG selection cannot resolve either lifecycle, including a bad autouse."""
    pytester.makeconftest(_ROUTING_IO_BOUNDARY + '\nexpected = "none"\nbypass = False\n')
    source = """
        import pytest
        from conftest import events
        USE_SHARED_JOBS_POSTGRES = True

        def test_sqlite(request: pytest.FixtureRequest) -> None:
            assert events == []
            assert "pg_temp_db" not in request.fixturenames
            assert "isolated_test_environment" not in request.fixturenames
    """
    pytester.makepyfile(
        test_probe=source
        + """
        @pytest.fixture(autouse=True)
        def unexpected_allocation(pg_temp_db: dict[str, str]) -> dict[str, str]:
            return pg_temp_db
    """
    )
    red = _run_fixture_probe(pytester, monkeypatch, "sqlite-wrong-allocation")
    red.assert_outcomes(errors=1)
    assert "Forbidden alternative lifecycle resolution" in red.stdout.str()
    pytester.makepyfile(test_probe=source)
    _run_fixture_probe(pytester, monkeypatch, "sqlite-no-allocation").assert_outcomes(passed=1)
