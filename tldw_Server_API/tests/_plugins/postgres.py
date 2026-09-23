"""Unified Postgres fixtures for tests.

Goals:
- Single source of truth for resolving Postgres connection settings.
- Best‑effort reachability check and optional Docker auto‑start for localhost.
- Function‑scoped temporary database creation and cleanup.

Usage patterns:
- Request `pg_temp_db` for a per‑test scratch DB. Returns a dict with
  host/port/user/password/database and a `dsn` field. Skips if Postgres is
  unreachable and not required.
- Request `pg_eval_params` for compatibility with existing tests that expect
  a dict of connection params (host/port/user/password/database).
- Request `pg_database_config` to get a DatabaseConfig ready for backend
  creation via DatabaseBackendFactory.

Environment knobs:
- TEST_DATABASE_URL / DATABASE_URL / POSTGRES_TEST_DSN / POSTGRES_TEST_*
- TLDW_TEST_POSTGRES_REQUIRED=1 to fail instead of skip when unavailable
- TLDW_TEST_NO_DOCKER=1 to disable Docker auto‑start
- TLDW_TEST_PG_IMAGE (default: postgres:18)
- TLDW_TEST_PG_CONTAINER_NAME (default: tldw_postgres_test)
"""
from __future__ import annotations

import os
import time
import uuid
import socket
import shutil
import subprocess
from typing import Dict, Generator

import pytest

try:  # Prefer psycopg v3
    import psycopg  # type: ignore
    _PG_DRIVER = "psycopg"
except Exception:  # pragma: no cover - optional dependency
    try:
        import psycopg2  # type: ignore
        _PG_DRIVER = "psycopg2"
    except Exception:  # pragma: no cover - optional dependency
        psycopg = None  # type: ignore
        psycopg2 = None  # type: ignore
        _PG_DRIVER = None


def _quote_ident(ident: str) -> str:
    """Safely quote a Postgres identifier for dynamic DDL."""
    return '"' + str(ident).replace('"', '""') + '"'


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _ensure_postgres_available(host: str, port: int, user: str, password: str, *, require_pg: bool) -> bool:
    """Try to connect; if not available and local, attempt to start docker, then retry.

    Returns True if Postgres becomes reachable; otherwise False (caller may skip tests).
    """
    # Quick TCP probe first
    if _tcp_reachable(host, port):
        return True

    # Only attempt Docker on local hostnames
    if str(host) not in {"localhost", "127.0.0.1", "::1"}:
        return False

    if os.getenv("TLDW_TEST_NO_DOCKER", "").lower() in ("1", "true", "yes"):
        return False

    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False

    image = os.getenv("TLDW_TEST_PG_IMAGE", "postgres:18")
    container = os.getenv("TLDW_TEST_PG_CONTAINER_NAME", "tldw_postgres_test")

    # Stop and remove an existing container with same name (best‑effort)
    try:
        subprocess.run([docker_bin, "rm", "-f", container], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        _ = None

    envs = [
        "-e", f"POSTGRES_USER={user}",
        "-e", f"POSTGRES_PASSWORD={password}",
        "-e", "POSTGRES_DB=postgres",
    ]
    ports = ["-p", f"{port}:5432"]

    run_cmd = [docker_bin, "run", "-d", "--name", container, *envs, *ports, image]
    try:
        subprocess.run(run_cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False

    # Wait up to ~30 seconds for readiness
    for _ in range(30):
        if _tcp_reachable(host, port):
            return True
        time.sleep(1)
    return False


def _connect_admin(host: str, port: int, user: str, password: str):
    """Return a connection to the 'postgres' DB using whichever driver is available.

    Retries briefly to tolerate startup races after Docker auto-start.
    """
    if _PG_DRIVER is None:
        raise RuntimeError("psycopg (or psycopg2) is required for Postgres‑backed tests")

    last_err = None
    debug = os.getenv("TLDW_TEST_PG_DEBUG", "").lower() in ("1", "true", "yes", "y", "on")
    for _ in range(10):
        try:
            if _PG_DRIVER == "psycopg":  # pragma: no cover - env dependent
                conn = psycopg.connect(host=host, port=int(port), dbname="postgres", user=user, password=password or None, autocommit=True)  # type: ignore[name-defined]
            else:  # psycopg2
                conn = psycopg2.connect(host=host, port=int(port), database="postgres", user=user, password=password or None)  # type: ignore[name-defined]
                conn.autocommit = True
            return conn
        except Exception as e:  # pragma: no cover - env/timing dependent
            last_err = e
            if debug:
                try:
                    print(f"[pg-fixture] admin connect failed: host={host} port={port} user={user} err={e}")
                except Exception:
                    _ = None
            time.sleep(0.5)
    raise last_err  # type: ignore[misc]


def _create_database(host: str, port: int, user: str, password: str, db_name: str) -> None:
    conn = _connect_admin(host, port, user, password)
    q_db = _quote_ident(db_name)
    q_owner = _quote_ident(user)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (db_name,))
            cur.execute(f"DROP DATABASE IF EXISTS {q_db}")
            cur.execute(f"CREATE DATABASE {q_db} OWNER {q_owner}")
    finally:
        conn.close()


def _drop_database(host: str, port: int, user: str, password: str, db_name: str) -> None:
    try:
        conn = _connect_admin(host, port, user, password)
    except Exception:
        return
    q_db = _quote_ident(db_name)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (db_name,))
            cur.execute(f"DROP DATABASE IF EXISTS {q_db}")
    finally:
        try:
            conn.close()
        except Exception:
            _ = None


@pytest.fixture(scope="session")
def pg_server() -> Dict[str, str | int]:
    """Resolve base Postgres params and ensure server reachability.

    Does not create a specific database; use `pg_temp_db` for per‑test DBs.
    Skips tests if Postgres is unreachable and not required.
    """
    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    env = get_pg_env()
    require_pg = os.getenv("TLDW_TEST_POSTGRES_REQUIRED", "").lower() in ("1", "true", "yes", "y", "on")
    debug = os.getenv("TLDW_TEST_PG_DEBUG", "").lower() in ("1", "true", "yes", "y", "on")

    ok = _ensure_postgres_available(env.host, env.port, env.user, env.password, require_pg=require_pg)
    if not ok:
        if require_pg:
            pytest.fail("Postgres required (TLDW_TEST_POSTGRES_REQUIRED=1) but not reachable")
        pytest.skip("Postgres not reachable; skipping Postgres‑backed tests")
    if debug:
        try:
            masked = "***" if env.password else ""
            print(
                "[pg-fixture] resolved server:",
                f"host={env.host} port={env.port} user={env.user} password={masked} database={env.database} dsn={env.dsn}"
            )
        except Exception:
            _ = None

    return {"host": env.host, "port": int(env.port), "user": env.user, "password": env.password}


def _temp_db_generator(pg_server) -> Generator[Dict[str, object], None, None]:
    host = str(pg_server["host"])  # type: ignore[index]
    port = int(pg_server["port"])  # type: ignore[index]
    user = str(pg_server["user"])  # type: ignore[index]
    password = str(pg_server.get("password") or "")  # type: ignore[index]
    db_name = f"tldw_test_{uuid.uuid4().hex[:8]}"

    if _PG_DRIVER is None:  # pragma: no cover - env dependent
        pytest.skip("psycopg not installed; skipping Postgres‑backed tests")

    require_pg = os.getenv("TLDW_TEST_POSTGRES_REQUIRED", "").lower() in ("1", "true", "yes", "y", "on")
    debug = os.getenv("TLDW_TEST_PG_DEBUG", "").lower() in ("1", "true", "yes", "y", "on")

    try:
        _create_database(host, port, user, password, db_name)
    except Exception as e:
        # First try a local Docker fallback on an alternate port if we're on localhost
        alt_port_env = os.getenv("TLDW_TEST_PG_ALT_PORT", "5434")
        alt_port = int(alt_port_env) if alt_port_env.isdigit() else 5434
        can_attempt_docker = str(host) in {"127.0.0.1", "localhost", "::1"} and os.getenv("TLDW_TEST_NO_DOCKER", "").lower() not in ("1", "true", "yes", "y", "on")
        tried_docker = False
        if can_attempt_docker and alt_port != int(port):
            tried_docker = True
            if debug:
                try:
                    print(f"[pg-fixture] attempting Docker fallback on 127.0.0.1:{alt_port} for user={user}")
                except Exception:
                    _ = None
            ok2 = _ensure_postgres_available("127.0.0.1", alt_port, user, password, require_pg=require_pg)
            if ok2:
                # Switch to alternate local container and retry create
                host = "127.0.0.1"
                port = alt_port
                try:
                    _create_database(host, port, user, password, db_name)
                except Exception as e2:
                    # Fall through to final skip/fail
                    e = e2

        msg = (
            f"Unable to create temporary Postgres database as user '{user}' on {host}:{port}. "
            f"This usually means the resolved credentials lack CREATEDB privileges or are incorrect.\n"
            f"Hint: set POSTGRES_TEST_DSN (or JOBS_DB_URL/TEST_DATABASE_URL) to a superuser DSN, e.g. postgresql://tldw_user:TestPassword123!@127.0.0.1:{port}/postgres (or postgres:postgres).\n"
            + ("Tried Docker fallback on alternate port and still failed.\n" if tried_docker else "")
            + f"Error: {e}"
        )
        if debug:
            try:
                print("[pg-fixture] " + msg)
            except Exception:
                _ = None
        if require_pg:
            pytest.fail(msg)
        else:
            pytest.skip(msg)
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    params: Dict[str, object] = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": db_name,
        "dsn": dsn,
    }
    try:
        yield params
    finally:
        _drop_database(host, port, user, password, db_name)


@pytest.fixture(scope="function")
def pg_temp_db(pg_server) -> Generator[Dict[str, object], None, None]:
    """Create a temporary database for the current test and drop it afterwards.

    Returns a dict with: host, port, user, password, database, dsn.
    """
    yield from _temp_db_generator(pg_server)


@pytest.fixture(scope="session")
def pg_temp_db_session(pg_server) -> Generator[Dict[str, object], None, None]:
    """Create a temporary database for the test session and drop it afterwards."""
    yield from _temp_db_generator(pg_server)


@pytest.fixture(scope="function")
def pg_eval_params(pg_temp_db) -> Dict[str, object]:
    """Compatibility fixture returning connection params for a live temp DB.

    Matches the signature expected by existing tests that use
    cfg = {"host", "port", "user", "password", "database"}.
    """
    return {k: v for k, v in pg_temp_db.items() if k in {"host", "port", "user", "password", "database"}}


@pytest.fixture(scope="function")
def pg_database_config(pg_temp_db):
    """Return a DatabaseConfig prepopulated with a temporary Postgres database."""
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
    return DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=str(pg_temp_db["host"]),
        pg_port=int(pg_temp_db["port"]),
        pg_database=str(pg_temp_db["database"]),
        pg_user=str(pg_temp_db["user"]),
        pg_password=str(pg_temp_db.get("password") or ""),
    )


@pytest.fixture(scope="session")
def pg_database_config_session(pg_temp_db_session):
    """Return a DatabaseConfig for a session-scoped Postgres database."""
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
    return DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=str(pg_temp_db_session["host"]),
        pg_port=int(pg_temp_db_session["port"]),
        pg_database=str(pg_temp_db_session["database"]),
        pg_user=str(pg_temp_db_session["user"]),
        pg_password=str(pg_temp_db_session.get("password") or ""),
    )


@pytest.fixture(scope="function")
def pg_restricted_backend(pg_database_config):
    """Yield a PostgreSQL backend whose role cannot bypass row-level security.

    ``pg_database_config`` connects with the admin DSN, and every supported
    setup documents that DSN as a superuser -- the bundled container's
    ``tldw_user`` is ``rolsuper=true, rolbypassrls=true``.  A superuser is
    exempt from RLS *even under* ``FORCE ROW LEVEL SECURITY``, so a test that
    asserts a policy through the plain config passes whether or not the policy
    exists.  That is finding F1 of the cross-user isolation audit, reproduced
    inside the test harness.

    The restricted role owns the objects it creates, which is why the policies
    under test must use ``FORCE ROW LEVEL SECURITY`` -- plain ``ENABLE`` exempts
    the owner.

    Roughly ten tests hand-rolled this setup before it lived here; prefer this
    fixture over another copy.
    """
    from dataclasses import replace

    from psycopg import sql

    from tldw_Server_API.app.core.DB_Management.backends.factory import (
        DatabaseBackendFactory,
    )

    role = "tldw_rls_" + uuid.uuid4().hex
    password = uuid.uuid4().hex
    admin = DatabaseBackendFactory.create_backend(pg_database_config)
    backend = None
    try:
        with admin.transaction() as conn:
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS "
                    "NOINHERIT NOCREATEDB NOCREATEROLE"
                ).format(sql.Identifier(role), sql.Literal(password))
            )
            conn.execute(
                sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(
                    sql.Identifier(role)
                )
            )
            conn.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(pg_database_config.pg_database),
                    sql.Identifier(role),
                )
            )
            # Migrations that verify `relowner = current_user::regrole` refuse to
            # run for a role that merely holds CREATE, so hand it the schema.
            # This also makes it the table owner, which is precisely why the
            # policies under test need FORCE ROW LEVEL SECURITY: plain ENABLE
            # exempts the owner and the assertions would pass vacuously.
            conn.execute(
                sql.SQL("ALTER SCHEMA public OWNER TO {}").format(sql.Identifier(role))
            )

        backend = DatabaseBackendFactory.create_backend(
            replace(
                pg_database_config,
                pg_user=role,
                pg_password=password,
                connection_string=None,
                pool_size=1,
                max_overflow=1,
            )
        )
        # Fail loudly rather than silently proving nothing.
        with backend.transaction() as conn:
            flags = conn.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
        assert not flags["rolsuper"], f"{role} unexpectedly holds SUPERUSER"
        assert not flags["rolbypassrls"], f"{role} unexpectedly holds BYPASSRLS"

        yield backend
    finally:
        if backend is not None:
            backend.get_pool().close_all()
        with admin.transaction() as conn:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
        admin.get_pool().close_all()
