import os
import pytest

try:
    from tldw_Server_API.tests._plugins.postgres import *  # noqa: F401,F403
except Exception:
    _ = None


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy(v: str | None) -> bool:
    return bool(v and v.strip().lower() in _TRUTHY)


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_jobs_routes_env():
    """Session-scoped baseline env so Jobs admin routes are mounted eagerly.

    Ensures that when `tldw_Server_API.app.main` is first imported during test
    discovery or module import, the 'jobs' router is included regardless of the
    order in which fixtures or tests import modules.
    """
    # Core test toggles applied as early as possible
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("MINIMAL_TEST_APP", "1")
    os.environ.setdefault("ROUTES_STABLE_ONLY", "0")
    # Make sure 'jobs' is present in ROUTES_ENABLE for non-minimal codepaths
    prev = os.environ.get("ROUTES_ENABLE", "")
    parts = [p for p in prev.split(",") if p]
    if "jobs" not in parts:
        parts.append("jobs")
        os.environ["ROUTES_ENABLE"] = ",".join(parts)
    # Avoid privilege metadata validation aborts when config file is absent
    os.environ.setdefault("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")


@pytest.fixture(autouse=True)
def _jobs_minimal_env(monkeypatch):
    """Ensure a minimal, stable app environment for Jobs tests.

    - Enables TEST_MODE and MINIMAL_TEST_APP to avoid heavy startup dependencies
    - Ensures Jobs routes are mounted via ROUTES_ENABLE
    - Disables background Jobs services that can add flakiness
    - Stabilizes acquisition priority for chatbooks domain
    """
    # Core test toggles
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "0")
    prev = os.getenv("ROUTES_ENABLE", "")
    parts = [p for p in prev.split(",") if p]
    if "jobs" not in parts:
        parts.append("jobs")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))

    # Quiet background features for deterministic tests
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "false")
    # Default to no lease enforcement in Jobs tests unless a test opts in
    monkeypatch.delenv("JOBS_ENFORCE_LEASE_ACK", raising=False)
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    # Disable counters and outbox by default for determinism; tests can opt-in
    if os.getenv("JOBS_COUNTERS_ENABLED") is None:
        monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "false")
    if os.getenv("JOBS_EVENTS_OUTBOX") is None:
        monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    # Webhooks worker defaults to off; individual tests can opt-in as needed
    if os.getenv("JOBS_WEBHOOKS_ENABLED") is None:
        monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "false")

    # Disable core Chatbooks Jobs worker during tests to avoid races
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "false")

    # Skip global privilege metadata validation for Jobs tests
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    # Default to no domain-scoped RBAC unless a test opts in
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "false")
    monkeypatch.setenv("JOBS_REQUIRE_DOMAIN_FILTER", "false")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "false")
    # Disable Postgres RLS unless a test explicitly enables it
    monkeypatch.setenv("JOBS_PG_RLS_ENABLE", "false")
    # Avoid env leakage that could allow admin finalize on arbitrary domains
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "chatbooks")
    # Keep acquire ordering defaults deterministic across environments
    monkeypatch.delenv("JOBS_ACQUIRE_PRIORITY_DESC_DOMAINS", raising=False)
    monkeypatch.delenv("JOBS_PG_ACQUIRE_PRIORITY_DESC_DOMAINS", raising=False)
    monkeypatch.delenv("JOBS_POSTGRES_ACQUIRE_PRIORITY_DESC_DOMAINS", raising=False)


@pytest.fixture(autouse=True)
def _reset_jobs_acquire_gate():
    """Reset JobManager acquire gate before and after each test.

    App shutdown flips the acquire gate on; carrying that across tests
    blocks acquisitions. Ensure it's off for each test.
    """
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JM
        _JM.set_acquire_gate(False)
        yield
        _JM.set_acquire_gate(False)
    except Exception:
        # Tests that don't touch JobManager shouldn't fail on import issues
        yield

@pytest.fixture
def route_debugger():
    """Helper to print app routes when debugging 404s in tests.

    Usage:
        if resp.status_code == 404:
            route_debugger(app)
    """
    def _debug(app):
        try:
            from starlette.routing import BaseRoute
            lines = []
            for r in getattr(app, "routes", []):
                path = getattr(r, "path", None) or getattr(r, "path_format", None) or str(r)
                methods = sorted(list(getattr(r, "methods", set()))) if hasattr(r, "methods") else []
                name = getattr(r, "name", "")
                lines.append(f"- {path} [{','.join(methods)}] name={name}")
            print("[route-debug] Mounted routes:\n" + "\n".join(lines))
        except Exception as e:  # pragma: no cover - debugging helper
            print(f"[route-debug] failed: {e}")

    return _debug


@pytest.fixture(scope="function")
def jobs_pg_dsn(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Initialize Jobs on the selected fixture-owned per-test PostgreSQL DB.

    - Opted-in modules delegate lifecycle to isolated_test_environment.
    - Historical modules retain the unified pg_temp_db fixture.
    - Ensures Jobs schema exists on that DB.
    - Sets JOBS_DB_URL for the duration of the test.
    """
    # Minimal app footprint hints and ensure Jobs routes are enabled
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "0")
    prev = os.getenv("ROUTES_ENABLE", "")
    parts = [p for p in prev.split(",") if p]
    if "jobs" not in parts:
        parts.append("jobs")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    if getattr(request.module, "USE_SHARED_JOBS_POSTGRES", False):
        from psycopg.conninfo import conninfo_to_dict

        _, db_name = request.getfixturevalue("isolated_test_environment")
        dsn = os.environ["TEST_DATABASE_URL"]
        assert conninfo_to_dict(dsn)["dbname"] == db_name, "DSN must belong to the shared fixture"
    else:
        dsn = str(request.getfixturevalue("pg_temp_db")["dsn"])
    # Initialize Jobs schema
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg, ensure_job_counters_pg
    ensure_jobs_tables_pg(dsn)
    ensure_job_counters_pg(dsn)
    # Ensure acquisitions are allowed for these tests
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager
        JobManager.set_acquire_gate(False)
    except Exception:
        _ = None
    # Bind to env for code under test
    monkeypatch.setenv("JOBS_DB_URL", dsn)
    return dsn


@pytest.fixture(autouse=True)
def _pg_jobs_db_url(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind pg_jobs to the opted-in shared lifecycle or the historical route."""
    try:
        if "pg_jobs" not in request.keywords:
            return
    except Exception:
        return
    if getattr(request.module, "USE_SHARED_JOBS_POSTGRES", False):
        monkeypatch.setenv("JOBS_DB_URL", request.getfixturevalue("jobs_pg_dsn"))
        return
    # Default to the per-test temp DB for pg_jobs, unless explicitly overridden.
    if _truthy(os.getenv("JOBS_PG_USE_ENV_DB")) and os.getenv("JOBS_DB_URL", "").startswith("postgres"):
        return
    pg_temp_db = request.getfixturevalue("pg_temp_db")
    dsn = str(pg_temp_db["dsn"])  # type: ignore[index]
    monkeypatch.setenv("JOBS_DB_URL", dsn)
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg, ensure_job_counters_pg
    ensure_jobs_tables_pg(dsn)
    ensure_job_counters_pg(dsn)
