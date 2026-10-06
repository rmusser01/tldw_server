"""Private PostgreSQL lookup failures; imported only by isolated VN tests."""

from typing import Any


async def cancel_postgres_quota_read(conn: Any) -> None:
    """Cancel a real server read inside the caller's transaction or savepoint."""
    await conn.execute("SET LOCAL statement_timeout = '10ms'")
    await conn.fetch("SELECT pg_sleep(0.05)")


async def postgres_connection_usable(conn: Any) -> bool:
    """Probe the same connection without resetting its transaction state."""
    return await conn.fetchval("SELECT 1") == 1
