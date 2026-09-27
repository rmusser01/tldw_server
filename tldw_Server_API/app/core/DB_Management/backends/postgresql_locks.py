"""Transaction-scoped PostgreSQL coordination for asynchronous repositories."""

from collections.abc import Awaitable, Callable
from typing import Any


async def acquire_schema_lock(
    execute: Callable[..., Awaitable[Any]], namespace: str, resource: str
) -> None:
    """Serialize schema work using the caller's active transaction executor.

    The caller owns commit/rollback, which releases the lock automatically.
    Database errors and cancellation propagate without retries or new connections.
    Namespace and resource are bound values, never interpolated SQL identifiers.
    """
    await execute(
        "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))",
        namespace,
        resource,
    )
