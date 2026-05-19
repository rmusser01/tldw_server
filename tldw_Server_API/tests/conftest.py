"""
Pytest configuration for the main test suite.

Registers shared test plugins and provides common fixtures.
"""
from __future__ import annotations

pytest_plugins = ["tldw_Server_API.tests._plugins.http_client_patch_guard"]

from collections.abc import Callable
import os
from pathlib import Path
try:
    # Ensure tests see provider keys from the canonical location
    # Load once at collection time, without overriding explicit env
    from dotenv import load_dotenv  # type: ignore
    _tests_root = Path(__file__).resolve()
    _project_root = _tests_root.parents[1]  # tldw_Server_API/
    _env_path = _project_root / "Config_Files" / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
        # If a real OpenAI key is present, prefer OpenAI as the default provider
        # to ensure real-integration tests hit OpenAI when provider is unspecified.
        if os.getenv("OPENAI_API_KEY") and not os.getenv("DEFAULT_LLM_PROVIDER"):
            os.environ.setdefault("DEFAULT_LLM_PROVIDER", "openai")
except Exception:
    # Never fail collection due to dotenv issues
    _ = None
# Force test-friendly env knobs
os.environ["MPLBACKEND"] = "Agg"
# Provide an explicit, deterministic API key for tests that rely on single-user/test-mode shortcuts.
# Production code no longer assumes a default for SINGLE_USER_TEST_API_KEY.
os.environ.setdefault("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
# Force a deterministic single-user key for pytest runs, regardless of developer .env.
os.environ["SINGLE_USER_API_KEY"] = os.environ["SINGLE_USER_TEST_API_KEY"]
# Default to single-user auth for tests; suites that need multi-user set it explicitly.
os.environ["AUTH_MODE"] = "single_user"
# Ensure a deterministic default AuthNZ DB for baseline restores.
# Tests that need Postgres must override DATABASE_URL explicitly.
os.environ.setdefault("DATABASE_URL", "sqlite:///./Databases/users.db")
# Ensure the AuthNZ PROFILE hint does not leak from developer shells into tests.
# Tests that need a profile should set it explicitly via monkeypatch.
os.environ.pop("PROFILE", None)
# Disable background schedulers/workers that spawn threads during tests
os.environ["DISABLE_AUTHNZ_SCHEDULER"] = "1"
os.environ["AUTHNZ_SCHEDULER_DISABLED"] = "1"
os.environ["WORKFLOWS_SCHEDULER_ENABLED"] = "false"
# Ensure ingestion backpressure/tenant quotas don't leak from developer envs into tests.
os.environ["EMBEDDINGS_TENANT_RPS"] = "0"
os.environ["INGEST_TENANT_RPS"] = "0"
os.environ["EMB_BACKPRESSURE_MAX_DEPTH"] = "999999999"
os.environ["EMB_BACKPRESSURE_MAX_AGE_SECONDS"] = "999999999"
# Relax webhook egress for test replay/egress simulations (no real network used in test short-circuit paths)
os.environ.setdefault("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
os.environ.setdefault("WORKFLOWS_WEBHOOK_ALLOWLIST", "*")
# Allow ephemeral localhost ports in tests.
os.environ.setdefault("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
# Disable AuthNZ scheduler functions proactively to avoid background threads
try:
    from tldw_Server_API.app.core.AuthNZ import scheduler as _auth_sched
    async def _noop():
        return None
    _auth_sched.start_authnz_scheduler = _noop  # type: ignore[assignment]
    _auth_sched.stop_authnz_scheduler = _noop  # type: ignore[assignment]
    _auth_sched.reset_authnz_scheduler = _noop  # type: ignore[assignment]
except Exception:
    _ = None
import logging
import weakref
# Dump lingering non-daemon threads at exit to avoid silent hangs
import threading
import atexit
import asyncio
try:
    import faulthandler
    import signal
    import sys as _sys
    if hasattr(signal, "SIGUSR2"):
        faulthandler.register(signal.SIGUSR2, file=_sys.stderr, all_threads=True)
except Exception:
    # Best-effort; tracing is optional
    _ = None
import pytest


_AIOSQLITE_CONNECTIONS: "weakref.WeakSet[object]" = weakref.WeakSet()
_AIOSQLITE_ORIG_CONNECT = None


def _install_aiosqlite_tracking() -> None:
    global _AIOSQLITE_ORIG_CONNECT
    if _AIOSQLITE_ORIG_CONNECT is not None:
        return
    try:
        import aiosqlite  # type: ignore
    except Exception:
        return
    _AIOSQLITE_ORIG_CONNECT = aiosqlite.connect

    def _tracked_connect(*args, **kwargs):
        conn = _AIOSQLITE_ORIG_CONNECT(*args, **kwargs)
        try:
            _AIOSQLITE_CONNECTIONS.add(conn)
        except Exception:
            _ = None
        return conn

    try:
        aiosqlite.connect = _tracked_connect  # type: ignore[assignment]
        aiosqlite.connect.__wrapped__ = _AIOSQLITE_ORIG_CONNECT  # type: ignore[attr-defined]
    except Exception:
        _AIOSQLITE_ORIG_CONNECT = None


async def _close_tracked_aiosqlite_connections() -> None:
    try:
        import aiosqlite  # type: ignore
    except Exception:
        return
    conns = list(_AIOSQLITE_CONNECTIONS)
    if not conns:
        return
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    for conn in conns:
        try:
            loop = getattr(conn, "_loop", None)
            if loop and loop is not current_loop and loop.is_running():
                try:
                    future = asyncio.run_coroutine_threadsafe(conn.close(), loop)
                    await asyncio.wrap_future(future)
                    continue
                except Exception:
                    _ = None
            await conn.close()
        except Exception:
            try:
                stop_future = conn.stop()
                if stop_future is not None:
                    await asyncio.wrap_future(stop_future)
            except Exception:
                _ = None


def _run_coro_sync_best_effort(coro):
    """Run an async cleanup coroutine from sync pytest hooks/fixtures."""
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except Exception:
        loop = None
    if loop is not None and not loop.is_closed():
        try:
            if loop.is_running():
                return None
            return loop.run_until_complete(coro)
        except Exception:
            return None
    try:
        return asyncio.run(coro)
    except Exception:
        return None


_AUTH_ENV_BASELINE_KEYS = (
    # AuthNZ mode + core configuration.
    "AUTH_MODE",
    "PROFILE",
    "JWT_SECRET_KEY",
    "DATABASE_URL",
    # Single-user auth header compatibility.
    "SINGLE_USER_API_KEY",
    "API_KEY",
    # Common guardrail toggles that can leak between tests when set via os.environ directly.
    "VIRTUAL_KEYS_ENABLED",
    "LLM_BUDGET_ENFORCE",
    "CSRF_ENABLED",
    # Route gating and backend knobs used by a handful of integration tests.
    "ROUTES_ENABLE",
    "TLDW_USER_DB_BACKEND",
    # Privilege metadata validation can be toggled in tests/fixtures.
    "PRIVILEGE_METADATA_VALIDATE_ON_STARTUP",
)

_AUTH_ENV_BASELINE = {k: os.environ.get(k) for k in _AUTH_ENV_BASELINE_KEYS}

_USER_DB_ENV_KEYS = (
    # Per-test DB base directory often overridden in Character Chat/Streaming tests.
    "USER_DB_BASE_DIR",
    # Legacy alias still honored in a few paths.
    "USER_DB_BASE",
)

_RISKY_ENV_KEYS = (
    "JOBS_DB_URL",
    "JOBS_DB_PATH",
    "JOBS_ALLOWED_QUEUES",
    "JOBS_ALLOWED_QUEUES_AUDIO",
    "JOBS_ALLOWED_QUEUES_EMBEDDINGS",
    "JOBS_POLL_INTERVAL_SECONDS",
    "JOBS_LEASE_SECONDS",
    "JOBS_LEASE_MAX_SECONDS",
    "JOBS_ENFORCE_LEASE_ACK",
    "JOBS_DISABLE_LEASE_ENFORCEMENT",
    "JOBS_REQUIRE_COMPLETION_TOKEN",
    "JOBS_EVENTS_OUTBOX",
    "JOBS_COUNTERS_ENABLED",
    "JOBS_METRICS_GAUGES_ENABLED",
    "JOBS_METRICS_RECONCILE_ENABLE",
    "JOBS_WEBHOOKS_ENABLED",
    "JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC",
    "LOGURU_LEVEL",
    "LOG_LEVEL",
    "SYSTEM_LOG_LEVEL",
    "LOG_STREAM",
    "LOG_COLOR",
    "FORCE_COLOR",
    "PY_COLORS",
    "EMB_BACKPRESSURE_MAX_DEPTH",
    "EMB_BACKPRESSURE_MAX_AGE_SECONDS",
    "EMBEDDINGS_TENANT_RPS",
    "INGEST_TENANT_RPS",
)


@pytest.fixture(autouse=True)
def _restore_risky_env_and_logging():
    """Reset noisy env/logging settings that can leak between tests."""
    baseline = {k: os.environ.get(k) for k in _RISKY_ENV_KEYS}
    yield

    for key, baseline_value in baseline.items():
        if baseline_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = baseline_value

    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        JobManager.set_acquire_gate(False)
        JobManager.clear_rls_context()
    except Exception as exc:
        logging.getLogger(__name__).debug("JobManager reset failed: %s", exc)

    try:
        from tldw_Server_API.app.core.Logging.system_log_buffer import ensure_system_log_buffer

        ensure_system_log_buffer()
    except Exception as exc:
        logging.getLogger(__name__).debug("system_log_buffer reset failed: %s", exc)


@pytest.fixture()
def auth_headers():
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    return {"X-API-KEY": settings.SINGLE_USER_API_KEY}


@pytest.fixture(autouse=True)
def _restore_auth_env_and_singletons():
    """Restore shared AuthNZ-related env and singleton state between tests.

    Many tests legitimately flip `AUTH_MODE` (and related env vars) to exercise
    multi-user/JWT paths. Some of those tests historically used `os.environ[...]`
    assignments without restoring them, which makes the suite order-dependent.

    This fixture restores a small set of high-impact environment keys to their
    baseline values and resets key singletons used by the auth/jobs stacks.
    """
    # Defensive pre-test reset: avoid stale singleton state inherited from
    # collection-time imports or prior tests.
    try:
        from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service

        reset_jwt_service()
    except Exception:
        _ = None

    yield

    for key, baseline_value in _AUTH_ENV_BASELINE.items():
        if baseline_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = baseline_value

    # Ensure subsequent tests rebuild Settings from the restored environment.
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service

        reset_jwt_service()
    except Exception:
        _ = None

    # Avoid leaking the process-wide jobs acquisition gate across tests.
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        JobManager.set_acquire_gate(False)
    except Exception:
        _ = None


@pytest.fixture(autouse=True)
def _restore_user_db_env_and_chacha_cache():
    """Restore USER_DB_BASE_DIR/USER_DB_BASE and clear ChaChaNotes DB cache per test."""
    baseline = {k: os.environ.get(k) for k in _USER_DB_ENV_KEYS}
    yield

    for key, baseline_value in baseline.items():
        if baseline_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = baseline_value

    # Clear config/Auth settings so restored env is honored next test.
    try:
        from tldw_Server_API.app.core.config import clear_config_cache

        clear_config_cache()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        _ = None

    # Drain ChaCha background tasks/futures before closing DB handles to avoid
    # races where sqlite connections are closed during active executor queries.
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
            reset_chacha_shutdown_state,
            shutdown_chacha_resources,
        )

        _run_coro_sync_best_effort(shutdown_chacha_resources(wait_timeout=5.0))
        reset_chacha_shutdown_state()
    except Exception:
        _ = None


@pytest.fixture(autouse=True)
def _reset_character_chat_complete_windows():
    """Ensure legacy /complete throttle cache is rebound per test loop."""
    try:
        from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as _chat_sessions

        _chat_sessions.reset_complete_windows()
    except Exception:
        _ = None
    yield
    try:
        from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as _chat_sessions

        _chat_sessions.reset_complete_windows()
    except Exception:
        _ = None


def _log_lingering_threads():


    try:
        import sys, traceback

        remaining = [
            t
            for t in threading.enumerate()
            if t is not threading.current_thread() and not t.daemon
        ]
        if remaining:
            details = []
            for t in remaining:
                stack = sys._current_frames().get(t.ident)
                formatted_stack = "".join(traceback.format_stack(stack)) if stack else ""
                details.append((t.name, getattr(t, "_target", None), formatted_stack))
                try:
                    # Best-effort shutdown to avoid interpreter hang
                    t.join(timeout=1.0)
                except Exception:
                    _ = None
                try:
                    t.daemon = True  # allow interpreter shutdown even if still alive
                except Exception:
                    _ = None
            summary = [(d[0], d[1]) for d in details]
            print(f"Non-daemon threads still running at exit: {summary}", file=sys.stderr)
            _log.warning("Non-daemon threads still running at exit: %s", summary)
            for name, target, formatted_stack in details:
                if formatted_stack:
                    _log.warning(
                        "Thread %s target=%s stack:\n%s", name, target, formatted_stack
                    )
                    print(
                        f"Thread {name} target={target} stack:\n{formatted_stack}",
                        file=sys.stderr,
                    )
    except Exception:
        _ = None


def _cleanup_lingering_threads(log: logging.Logger, context: str = "teardown") -> None:
    """Best-effort cleanup of lingering non-daemon threads during tests.

    Performs a first pass join with timeout for non-daemon threads and then logs
    any remaining threads with their stack frames before marking them as daemon
    to avoid interpreter shutdown hangs.
    """
    try:
        import sys

        current = threading.current_thread()
        # First pass: try to join all non-daemon threads with a timeout
        for t in threading.enumerate():
            if t is current or t.daemon:
                continue
            try:
                # Cancel timers so they don't keep the interpreter alive
                if isinstance(t, threading.Timer):
                    try:
                        t.cancel()
                    except Exception:
                        _ = None
                t.join(timeout=1.0)
            except Exception:
                _ = None

        # Second pass: log any remaining threads and mark them daemon
        for t in threading.enumerate():
            if t is current or t.daemon:
                continue
            try:
                stack = sys._current_frames().get(t.ident)
            except Exception:
                stack = None
            msg = (
                f"Lingering non-daemon thread during {context}: "
                f"name={t.name} target={getattr(t, '_target', None)}"
            )
            print(msg, file=sys.stderr)
            try:
                log.warning("%s stack=%s", msg, stack)
            except Exception:
                _ = None
            try:
                if isinstance(t, threading.Timer):
                    t.cancel()
                    t.join(timeout=1.0)
            except Exception:
                _ = None
            try:
                t.daemon = True  # allow interpreter shutdown to proceed
            except Exception:
                _ = None
    except Exception as e:
        try:
            import sys as _local_sys

            print(
                f"Failed to log lingering threads during {context}: {e}",
                file=_local_sys.stderr,
            )
        except Exception:
            _ = None


atexit.register(_log_lingering_threads)
# Ensure problematic optional routers don't import during test collection
# and enable test-friendly behaviors before importing the app.
_log = logging.getLogger(__name__)
try:
    # Disable heavy 'research' router to avoid importing Web_Scraping during collection
    existing_disable = os.getenv("ROUTES_DISABLE", "")
    if "research" not in existing_disable:
        os.environ["ROUTES_DISABLE"] = (existing_disable + ",research").strip(",")
    # Default: prefer minimal app profile for faster, deterministic tests.
    # Evaluations routes remain included by default; heavy suites are marker-gated.
    os.environ.setdefault("MINIMAL_TEST_APP", "1")
    # Ensure Workflows/Scheduler routes stay enabled in tests to avoid 404s when stable_only is true
    try:
        _re = os.getenv("ROUTES_ENABLE", "")
        parts = [p for p in _re.replace(" ", ",").split(",") if p]
        for k in ["workflows", "scheduler"]:
            if k not in [p.lower() for p in parts]:
                parts.append(k)
        os.environ["ROUTES_ENABLE"] = ",".join(dict.fromkeys(parts))
    except Exception:
        _ = None
    # Ensure notes endpoints stay enabled for health tests even if ROUTES_DISABLE includes them
    try:
        _rd = os.getenv("ROUTES_DISABLE", "")
        parts = [p for p in _rd.replace(" ", ",").split(",") if p]
        parts = [p for p in parts if p.lower() != "notes"]
        os.environ["ROUTES_DISABLE"] = ",".join(dict.fromkeys(parts))
    except Exception:
        _ = None
    # Enable deterministic test behaviors across subsystems
    os.environ.setdefault("TEST_MODE", "1")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    # Ensure Postgres helpers see consistent defaults immediately at import time.
    # Many PG tests call get_pg_env() at module import; set test user/password
    # here so precedence falls to the correct, compose-aligned credentials.
    os.environ.setdefault("POSTGRES_TEST_USER", "tldw_user")
    os.environ.setdefault("POSTGRES_TEST_PASSWORD", "TestPassword123!")
    # Also mirror to generic POSTGRES_* if unset to avoid helper drift.
    os.environ.setdefault("POSTGRES_USER", "tldw_user")
    os.environ.setdefault("POSTGRES_PASSWORD", "TestPassword123!")
    # Ensure Postgres tests use a proper DSN instead of falling back to a SQLite DATABASE_URL.
    # If a dedicated DSN is provided via TEST_DATABASE_URL or POSTGRES_TEST_DSN, prefer it.
    # Otherwise, if POSTGRES_TEST_HOST/USER/DB are present, synthesize a DSN.
    try:
        _pg_dsn = os.getenv("TEST_DATABASE_URL") or os.getenv("POSTGRES_TEST_DSN")
        if not _pg_dsn:
            _pg_host = os.getenv("POSTGRES_TEST_HOST")
            _pg_port = os.getenv("POSTGRES_TEST_PORT", "5432")
            _pg_user = os.getenv("POSTGRES_TEST_USER")
            _pg_pass = os.getenv("POSTGRES_TEST_PASSWORD", "")
            _pg_db = os.getenv("POSTGRES_TEST_DATABASE") or os.getenv("POSTGRES_TEST_DB")
            if _pg_host and _pg_user and _pg_db:
                # Compose a DSN and set TEST_DATABASE_URL so PG helpers don't pick SQLite DATABASE_URL
                _auth = f"{_pg_user}:{_pg_pass}" if _pg_pass else _pg_user
                _pg_dsn = f"postgresql://{_auth}@{_pg_host}:{int(_pg_port)}/{_pg_db}"
        if _pg_dsn and _pg_dsn.lower().startswith("postgres"):
            os.environ["TEST_DATABASE_URL"] = _pg_dsn
    except Exception:
        _ = None
except Exception as e:
    # Surface environment setup failures in test output
    _log.exception("Failed to apply test environment setup in conftest.py")
from fastapi.testclient import TestClient
import contextlib
import asyncio


# Skip Jobs-marked tests by default unless explicitly enabled via RUN_JOBS.
# This ensures general CI workflows never run Jobs tests; the dedicated
# jobs-suite workflow sets RUN_JOBS=1 to include them.
import pytest as _pytest_jobs_gate

@_pytest_jobs_gate.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):  # pragma: no cover - collection-time behavior
    try:
        run_jobs = str(os.getenv("RUN_JOBS", "")).lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        run_jobs = False
    try:
        run_evals = str(os.getenv("RUN_EVALUATIONS", "")).lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        run_evals = False

    skip_jobs = _pytest_jobs_gate.mark.skip(reason="Jobs tests run only in the jobs-suite CI workflow")
    skip_evals = _pytest_jobs_gate.mark.skip(reason="Evaluations tests run only when RUN_EVALUATIONS=1")
    jobs_markers = {"jobs", "pg_jobs", "pg_jobs_stress"}
    for item in items:
        try:
            if not run_jobs and any(m.name in jobs_markers for m in item.iter_markers()):
                item.add_marker(skip_jobs)
            if not run_evals and any(m.name == "evaluations" for m in item.iter_markers()):
                item.add_marker(skip_evals)
        except Exception:
            # Never break collection on marker inspection
            _ = None

def pytest_configure(config):  # pragma: no cover - registration only
    try:
        config.addinivalue_line("markers", "evaluations: heavy Evaluations tests (opt-in via RUN_EVALUATIONS=1)")
        config.addinivalue_line("markers", "stt_golden: real-audio STT adapter golden tests (opt-in via TLDW_STT_GOLDEN_ENABLE=1)")
    except Exception:
        _ = None
    _install_aiosqlite_tracking()


@pytest.fixture(scope="function")
def event_loop():
    """Provide a fresh event loop per test and shutdown its default executor."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    try:
        if hasattr(loop, "shutdown_default_executor"):
            loop.run_until_complete(loop.shutdown_default_executor())
    except Exception:
        _ = None
    try:
        loop.close()
    except Exception:
        _ = None


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover - diagnostics/cleanup
    """Log and relax any remaining non-daemon threads to avoid interpreter shutdown hangs."""
    try:
        import sys, traceback

        try:
            from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
            _run_coro_sync_best_effort(shutdown_all_audit_services(raise_on_error=False))
        except Exception:
            _ = None
        try:
            from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool as _reset_db_pool
            _run_coro_sync_best_effort(_reset_db_pool())
        except Exception:
            _ = None
        try:
            from tldw_Server_API.app.core.Scheduler import stop_global_scheduler as _stop_global_scheduler
            _run_coro_sync_best_effort(_stop_global_scheduler())
        except Exception:
            _ = None
        try:
            from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler as _get_wf_scheduler
            _run_coro_sync_best_effort(_get_wf_scheduler().stop())
        except Exception:
            _ = None
        try:
            _run_coro_sync_best_effort(_close_tracked_aiosqlite_connections())
        except Exception:
            _ = None
        try:
            from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
                shutdown_chacha_resources,
            )
            _run_coro_sync_best_effort(shutdown_chacha_resources(wait_timeout=5.0))
        except Exception:
            _ = None
        try:
            from tldw_Server_API.app.api.v1.endpoints.research import (
                shutdown_websearch_executor,
            )
            shutdown_websearch_executor(wait=True, cancel_futures=True)
        except Exception:
            _ = None
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            loop = None
        if loop is not None and not loop.is_closed():
            try:
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                _ = None
            try:
                loop.close()
            except Exception:
                _ = None
        # Best-effort cleanup for lingering ThreadPoolExecutor workers (e.g., asyncio.to_thread)
        try:
            import concurrent.futures.thread as _cf_thread
            lock = getattr(_cf_thread, "_global_shutdown_lock", None)
            if lock is not None:
                with lock:
                    setattr(_cf_thread, "_shutdown", True)
            worker_items = list(getattr(_cf_thread, "_threads_queues", {}).items())
            for _worker_thread, queue in worker_items:
                try:
                    queue.put_nowait(None)
                except Exception:
                    try:
                        queue.put(None)
                    except Exception:
                        _ = None
            for worker_thread, _queue in worker_items:
                try:
                    worker_thread.join(timeout=2.0)
                except Exception:
                    _ = None
        except Exception:
            _ = None
        current = threading.current_thread()
        threads = [t for t in threading.enumerate() if t is not current and not t.daemon]
        if threads:
            for t in threads:
                # Stop common offenders (e.g., aiosqlite worker threads) to avoid hangs
                try:
                    import aiosqlite  # type: ignore
                    if isinstance(t, getattr(aiosqlite, "Connection", (aiosqlite.core.Connection,))):  # type: ignore[attr-defined]
                        try:
                            t._stop_running()  # type: ignore[attr-defined]
                        except Exception:
                            _ = None
                        try:
                            t.join(timeout=2.0)
                        except Exception:
                            _ = None
                except Exception:
                    _ = None
                try:
                    if isinstance(t, threading.Timer):
                        t.cancel()
                        t.join(timeout=1.0)
                except Exception:
                    _ = None
            # Re-check after stopping attempts; only log remaining threads
            threads = [t for t in threading.enumerate() if t is not current and not t.daemon]
            if threads:
                summary = [(t.name, getattr(t, "_target", None)) for t in threads]
                print(f"[pytest_sessionfinish] Non-daemon threads before exit: {summary}", file=sys.stderr)
                for t in threads:
                    stack = sys._current_frames().get(t.ident)
                    if stack:
                        formatted_stack = "".join(traceback.format_stack(stack))
                        print(
                            f"[pytest_sessionfinish] Thread {t.name} target={getattr(t, '_target', None)} stack:\n{formatted_stack}",
                            file=sys.stderr,
                        )
                    try:
                        t.daemon = True
                    except Exception:
                        _ = None
    except Exception:
        # Do not interfere with pytest shutdown on logging failures
        _ = None


# Bump file-descriptor limit for macOS/Linux test runs to avoid spurious
# 'Too many open files' and SQLite 'unable to open database file' errors
# caused by module-level TestClient instances in some test modules.
@pytest.fixture(scope="session", autouse=True)
def _raise_fd_limit():  # pragma: no cover - platform-dependent behavior
    try:
        import resource  # POSIX only
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Aim for at least 4096 if permitted by the hard limit
        target = 4096
        new_soft = min(max(soft, target), hard if hard > 0 else target)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        # On platforms without 'resource' (e.g., Windows) or when permissions
        # disallow raising limits, silently continue.
        _ = None

class _TestUsageLogger:
    def __init__(self):
        self.events = []

    def log_event(self, name, resource_id=None, tags=None, metadata=None):

        self.events.append((name, resource_id, tags, metadata))


@pytest.fixture()
def client_with_single_user(monkeypatch):
    """Provide a TestClient for the full FastAPI app with a single-user auth override.

    Returns a tuple of (client, usage_logger) for tests that also need to inspect usage events.
    """
    # Ensure tests run in non-production behavior
    os.environ.setdefault("TESTING", "true")

    usage_logger = _TestUsageLogger()

    # Import the FastAPI app and dependencies lazily to avoid heavy imports during test collection
    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_usage_event_logger
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext

    async def _override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    def _override_logger():

        return usage_logger

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="single-user",
            token_type="single_user",
            jti=None,
            roles=["admin"],
            permissions=["media.create"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception as e:
                # Best-effort; don't fail tests if state attachment fails
                import logging
                logging.getLogger(__name__).debug("Failed to set request.state.auth: %s", e)
        return principal

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_usage_event_logger] = _override_logger
    fastapi_app.dependency_overrides[get_auth_principal] = _override_principal

    with TestClient(fastapi_app) as client:
        yield client, usage_logger

    fastapi_app.dependency_overrides.pop(get_request_user, None)
    fastapi_app.dependency_overrides.pop(get_usage_event_logger, None)
    fastapi_app.dependency_overrides.pop(get_auth_principal, None)


@pytest.fixture()
def client_user_only(client_with_single_user):
    """Shorthand fixture that returns only the TestClient from client_with_single_user."""
    client, _ = client_with_single_user
    return client


def _reset_test_media_db_runtime():
    """Restore canonical Media DB runtime defaults before seam-backed test DB creation."""
    from tldw_Server_API.app.core.config import load_comprehensive_config
    from tldw_Server_API.app.core.DB_Management import DB_Manager

    cfg = load_comprehensive_config()
    DB_Manager.reset_content_backend(config=cfg, reload=False)
    return cfg


@pytest.fixture()
def test_media_db_factory() -> Callable[[Path | str], object]:
    """Create temporary Media DB handles through the refactor seam and close them after the test."""
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database

    open_databases: list[object] = []

    def _build(db_path: Path | str, *, client_id: str = "test_client") -> object:
        cfg = _reset_test_media_db_runtime()
        db = create_media_database(client_id=client_id, db_path=str(db_path), config=cfg)
        open_databases.append(db)
        return db

    yield _build

    for db in reversed(open_databases):
        close_connection = getattr(db, "close_connection", None)
        if callable(close_connection):
            close_connection()


@pytest.fixture()
def managed_test_media_db():
    """Expose the seam-backed Media DB context manager for test overrides."""
    from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database

    def _managed(client_id: str, **kwargs):
        cfg = kwargs.pop("config", None) or _reset_test_media_db_runtime()
        return managed_media_database(client_id, config=cfg, **kwargs)

    return _managed


@pytest.fixture()
def data_tables_app_factory(
    monkeypatch,
    managed_test_media_db,
) -> Callable[[Path], tuple["FastAPI", Path]]:
    """Create FastAPI apps wired for data table endpoints with test auth and DB overrides."""
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    apps: list[FastAPI] = []

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True, is_admin=True)

    async def _override_principal(request=None) -> AuthPrincipal:
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",
            jti=None,
            roles=["admin"],
            permissions=["media.create", "media.read", "media.update", "media.delete"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    def _build_app(db_path: Path) -> tuple[FastAPI, Path]:
        monkeypatch.setenv("TEST_MODE", "1")
        jobs_db_path = db_path.parent / "jobs.db"
        monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

        app = FastAPI()
        app.include_router(data_tables_router, prefix="/api/v1", tags=["data-tables"])

        async def _override_db():
            with managed_test_media_db(
                "test_client",
                db_path=str(db_path),
                initialize=False,
            ) as override_db:
                yield override_db

        app.dependency_overrides[get_request_user] = _override_user
        app.dependency_overrides[get_auth_principal] = _override_principal
        app.dependency_overrides[get_media_db_for_user] = _override_db
        apps.append(app)
        return app, jobs_db_path

    yield _build_app

    for app in apps:
        app.dependency_overrides.clear()


# Global session teardown to prevent test-run hangs from lingering executors/threads
@pytest.fixture(scope="session", autouse=True)
def _shutdown_executors_and_evaluations_pool():
    """Ensure global executors and the Evaluations connection pool are shut down at session end.

    Prevents pytest from hanging due to non-daemon worker threads started by
    CPU-bound helpers and background maintenance in the Evaluations module when
    app lifespan teardown is not exercised during tests.
    """
    yield
    # Best-effort shutdown of registered executors (thread/process pools)
    try:
        from tldw_Server_API.app.core.Utils.executor_registry import (
            shutdown_all_registered_executors_sync,
        )
        shutdown_all_registered_executors_sync(wait=True, cancel_futures=True)
    except Exception:
        _ = None
    # Explicit CPU pools cleanup (idempotent)
    try:
        from tldw_Server_API.app.core.Utils.cpu_bound_handler import cleanup_pools
        cleanup_pools()
    except Exception:
        _ = None
    # Proactively join/mark any lingering non-daemon threads so interpreter shutdown won't hang
    _cleanup_lingering_threads(_log, context="teardown")


@pytest.fixture(autouse=True)
def _reset_workflow_scheduler():
    """Reset WorkflowScheduler singleton state between tests to avoid stale queues/active counts."""
    try:
        from tldw_Server_API.app.core.Workflows.engine import WorkflowScheduler
        WorkflowScheduler._inst = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
    yield
    try:
        from tldw_Server_API.app.core.Workflows.engine import WorkflowScheduler
        WorkflowScheduler._inst = None  # type: ignore[attr-defined]
    except Exception:
        _ = None

    # Ensure ChaCha executor threads are not reported as lingering non-daemon
    # workers by the per-test scheduler reset diagnostics.
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
            reset_chacha_shutdown_state,
            shutdown_chacha_resources,
        )

        _run_coro_sync_best_effort(shutdown_chacha_resources(wait_timeout=5.0))
        reset_chacha_shutdown_state()
    except Exception:
        _ = None


    # Log any lingering non-daemon threads with their stack frames to aid debugging hangs
    _cleanup_lingering_threads(_log, context="scheduler reset")


# Unified Postgres fixtures are provided by tldw_Server_API.tests._plugins.postgres


@pytest.fixture()
def bypass_api_limits(monkeypatch):
    """Context manager to bypass ingress rate limiting for a given FastAPI app.

    Usage:
        with bypass_api_limits(app):
            ... make requests ...

    - Sets TEST_MODE=true for deterministic behavior
    - Disables RGSimpleMiddleware by removing it from app.user_middleware
    - Disables any provided legacy limiter(s) during the context
    """

    @contextlib.contextmanager
    def _bypass(app, *, limiters: tuple = ()):  # type: ignore[override]
        # Ensure test-friendly behaviors
        monkeypatch.setenv("TEST_MODE", "true")
        monkeypatch.setenv("RG_ENABLED", "0")

        # Snapshot existing middleware stack
        original_user_middleware = getattr(app, "user_middleware", [])[:]
        # Remove RGSimpleMiddleware if present
        try:
            from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
            app.user_middleware = [
                m for m in original_user_middleware if getattr(m, "cls", None) is not RGSimpleMiddleware
            ]
            app.middleware_stack = app.build_middleware_stack()
        except Exception:
            _ = None

        # Disable provided legacy limiter(s)
        limiter_states = []
        for lim in limiters or ():
            try:
                limiter_states.append((lim, getattr(lim, "enabled", True)))
                lim.enabled = False
            except Exception:
                limiter_states.append((lim, None))

        try:
            yield
        finally:
            # Restore limiter states
            for lim, prev in limiter_states:
                if prev is not None:
                    try:
                        lim.enabled = prev
                    except Exception:
                        _ = None
            # Restore middleware stack
            try:
                app.user_middleware = original_user_middleware
                app.middleware_stack = app.build_middleware_stack()
            except Exception:
                _ = None

    return _bypass
