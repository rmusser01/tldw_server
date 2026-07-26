"""Transport-neutral transaction ownership for UserProfiles commands."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseConcurrencyConflict,
    DatabaseLockError,
    RollbackSignal,
)
from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
    AuthnzTransactionPolicy,
    get_authnz_transaction_policy,
)
from tldw_Server_API.app.core.UserProfiles.backend import (
    ProfileBackendUnavailable,
    resolve_profile_backend,
)

T = TypeVar("T")


class ProfileTransactionError(RuntimeError):
    """Base class for sanitized, transport-neutral transaction failures."""

    code = "profile_update_failed"
    retry_after_seconds: int | None = None


class ProfileDatabaseBusy(ProfileTransactionError):
    code = "database_busy"

    def __init__(self, *, retry_after_seconds: int) -> None:
        super().__init__("Database is temporarily busy")
        self.retry_after_seconds = retry_after_seconds


class ProfileUpdateConcurrencyConflict(ProfileTransactionError):
    code = "profile_update_concurrency_conflict"

    def __init__(self) -> None:
        super().__init__("Profile update conflicted")


class ProfileTransactionFailed(ProfileTransactionError):
    def __init__(self) -> None:
        super().__init__("Profile update transaction failed")


class ProfileTransactionGateway:
    """Run one profile operation under the configured AuthNZ transaction policy."""

    def __init__(
        self,
        db_pool: Any,
        *,
        policy: AuthnzTransactionPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._db_pool = db_pool
        self._policy = policy or get_authnz_transaction_policy()
        self._sleep = sleep

    async def run(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        """Run once on PostgreSQL or retry SQLite transaction entry contention."""
        try:
            is_postgres = resolve_profile_backend(self._db_pool) == "postgres"
        except ProfileBackendUnavailable:
            raise ProfileTransactionFailed() from None
        retries_used = 0
        while True:
            entered = False
            try:
                async with self._db_pool.transaction(
                    acquire_timeout_seconds=(
                        self._policy.db_pool_acquire_timeout_seconds
                    )
                ) as conn:
                    entered = True
                    return await operation(conn)
            except RollbackSignal:
                raise
            except asyncio.CancelledError:
                raise
            except DatabaseConcurrencyConflict:
                raise ProfileUpdateConcurrencyConflict() from None
            except DatabaseLockError:
                can_retry_entry = (
                    not is_postgres
                    and not entered
                    and retries_used < self._policy.sqlite_lock_max_retries
                )
                if not can_retry_entry:
                    raise self._database_busy() from None
                delay = min(
                    self._policy.sqlite_lock_retry_base_seconds * (2**retries_used),
                    self._policy.sqlite_lock_retry_max_seconds,
                )
                retries_used += 1
                await self._sleep(delay)
            except (ConnectionPoolExhaustedError, TimeoutError):
                if is_postgres and not entered:
                    raise self._database_busy() from None
                raise ProfileTransactionFailed() from None
            except Exception:  # noqa: BLE001 - stable domain boundary
                raise ProfileTransactionFailed() from None

    def _database_busy(self) -> ProfileDatabaseBusy:
        return ProfileDatabaseBusy(
            retry_after_seconds=self._policy.busy_retry_after_seconds
        )


__all__ = [
    "ProfileDatabaseBusy",
    "ProfileTransactionError",
    "ProfileTransactionFailed",
    "ProfileTransactionGateway",
    "ProfileUpdateConcurrencyConflict",
]
