"""A contended RBAC query must not block the async authentication event loop."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import asyncpg
import psycopg
import pytest
from psycopg import sql
from starlette.requests import Request

pytestmark = pytest.mark.integration


def _hold_rbac_lock_until_event_loop_progress(
    database_url: str,
    table: str,
    lock_acquired: threading.Event,
    query_blocked: threading.Event,
    event_loop_progress: threading.Event,
    stop: threading.Event,
) -> bool:
    """Release a real table lock even when the regression blocks the event loop."""
    options = {"connect_timeout": 5, "options": "-c statement_timeout=5000 -c lock_timeout=3000"}
    query_fragment = "%COALESCE(r.is_system%" if table == "roles" else "%JOIN role_permissions rp%"
    with psycopg.connect(database_url, autocommit=True, **options) as blocker:
        with psycopg.connect(database_url, autocommit=True, **options) as monitor:
            blocker.execute("BEGIN")
            try:
                blocker.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(sql.Identifier(table)))
                blocker_pid = blocker.info.backend_pid
                lock_acquired.set()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not stop.is_set():
                    waiting = monitor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_stat_activity
                            WHERE datname = current_database()
                              AND wait_event_type = 'Lock'
                              AND %s = ANY(pg_blocking_pids(pid))
                              AND query LIKE %s
                        )
                        """,
                        (blocker_pid, query_fragment),
                    ).fetchone()[0]
                    if waiting:
                        query_blocked.set()
                        # An independent thread is the watchdog: a blocked loop cannot
                        # prevent rollback, even when asyncio timeouts cannot fire.
                        return event_loop_progress.wait(timeout=2)
                    stop.wait(timeout=0.01)
                raise AssertionError("Authentication did not reach the contended RBAC query")
            finally:
                blocker.execute("ROLLBACK")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("auth_kind", ["jwt", "api-key"])
@pytest.mark.parametrize("locked_table", ["roles", "permissions"])
async def test_postgres_authentication_allows_event_loop_progress_during_rbac_lock(
    isolated_test_environment,
    auth_kind: str,
    locked_table: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_handling
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service

    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        user_id = await connection.fetchval(
            """
            INSERT INTO users (uuid, username, email, password_hash, is_active, is_verified)
            VALUES ($1, 'rbac-progress', 'rbac-progress@example.test', 'unused-test-hash', TRUE, TRUE)
            RETURNING id
            """,
            uuid.uuid4(),
        )
        await connection.execute(
            "INSERT INTO roles (name, is_system) VALUES ('admin', TRUE) ON CONFLICT (name) DO NOTHING"
        )
        await connection.execute("INSERT INTO permissions (name) VALUES ('media.read') ON CONFLICT (name) DO NOTHING")
        await connection.execute(
            """
            INSERT INTO user_roles (user_id, role_id)
            SELECT $1, id FROM roles WHERE name = 'admin'
            """,
            user_id,
        )
        await connection.execute(
            """
            INSERT INTO role_permissions (role_id, permission_id)
            SELECT r.id, p.id FROM roles r CROSS JOIN permissions p
            WHERE r.name = 'admin' AND p.name = 'media.read'
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await connection.close()

    if auth_kind == "jwt":
        token = get_jwt_service().create_access_token(user_id, "rbac-progress", "admin")
        api_key = None
    else:
        manager = await get_api_key_manager()
        key_info = await manager.create_api_key(user_id, name="rbac-progress", scope="admin")
        token = None
        api_key = key_info["key"]

    async def authenticate():
        request = Request({"type": "http", "method": "GET", "path": "/test", "headers": [], "client": ("127.0.0.1", 0)})
        return await user_handling.get_request_user(request, api_key=api_key, token=token)

    # Initialize the actual synchronous repository before introducing contention.
    # Each authentication uses a fresh request, so no resolved principal is reused.
    await authenticate()
    lock_acquired = threading.Event()
    query_blocked = threading.Event()
    event_loop_progress = threading.Event()
    stop = threading.Event()
    blocker = asyncio.create_task(
        asyncio.to_thread(
            _hold_rbac_lock_until_event_loop_progress,
            pool.settings.DATABASE_URL,
            locked_table,
            lock_acquired,
            query_blocked,
            event_loop_progress,
            stop,
        )
    )

    async def probe_event_loop() -> None:
        if await asyncio.to_thread(query_blocked.wait, 10):
            event_loop_progress.set()

    probe = asyncio.create_task(probe_event_loop())
    try:
        assert await asyncio.to_thread(lock_acquired.wait, 10), "RBAC lock was not acquired"
        user = await asyncio.wait_for(authenticate(), timeout=15)
    finally:
        stop.set()
        probe.cancel()
        await asyncio.gather(probe, return_exceptions=True)
        responsive = await asyncio.wait_for(blocker, timeout=15)

    assert (
        responsive
    ), "Synchronous RBAC authentication prevented event-loop progress until the watchdog released the lock"
    assert user.id == user_id
    assert user.roles == ["admin"]
    assert user.is_admin is True
    assert {"media.read", "system.configure"} <= set(user.permissions)
