"""Per-user synchronous concurrency quotas (TASK-13435, ADR-076).

A request holds one lease for as long as it runs. The limit is the user's
``limits.*`` value: ``None`` is unlimited and costs nothing, ``0`` blocks
(ADR-064). Leases live in a Resource Governor built from ``RG_BACKEND``: on Redis
one limit holds across workers; on the memory backend it holds per worker process.

Every exit path calls ``ConcurrencySlot.release()``. Two backstops bound a missed
release: a lease lapses ``LEASE_TTL_SEC`` after its last renewal, and a slot stops
renewing once the task that took it has finished.
"""

from __future__ import annotations

import asyncio
from typing import Any

from anyio import CancelScope
from loguru import logger

from tldw_Server_API.app.core.config import rg_backend
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor, ResourceGovernor, RGRequest
from tldw_Server_API.app.core.Usage.quota_checks import as_quota_user_id
from tldw_Server_API.app.core.Usage.quota_resolver import user_quota

try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis is an optional deployment dependency
    _REDIS_ERRORS: tuple[type[Exception], ...] = ()
else:
    # redis-py's ConnectionError and TimeoutError are not the builtins: they derive from RedisError.
    _REDIS_ERRORS = (RedisError,)

LEASE_TTL_SEC = 120
RENEW_EVERY_SEC = 30.0

_NONCRITICAL = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    *_REDIS_ERRORS,
)
_governor: ResourceGovernor | None = None


class ConcurrencyQuotaExceeded(Exception):
    """Every one of the user's slots is held. ``retry_after`` is None when the limit is 0."""

    def __init__(self, limit: int, retry_after: int | None) -> None:
        """Record the limit that was hit and when a slot may free up."""
        super().__init__(f"Concurrency limit reached ({limit})")
        self.limit = limit
        self.retry_after = retry_after


class _QuotaPolicies:
    """Quota leases carry their own limit and TTL, so every policy id resolves to one bare policy."""

    def get_policy(self, _policy_id: str) -> dict[str, Any]:
        """A policy with no limits of its own."""
        return {"scopes": ["user"]}


def _lease_governor() -> ResourceGovernor:
    """This process's lease store, built once from RG_BACKEND. RG_ENABLED does not turn it off (ADR-076)."""
    global _governor
    if _governor is None:
        if rg_backend() == "redis":
            from tldw_Server_API.app.core.Resource_Governance.governor_redis import RedisResourceGovernor

            _governor = RedisResourceGovernor(policy_loader=_QuotaPolicies())
        else:
            _governor = MemoryResourceGovernor(policy_loader=_QuotaPolicies())
    return _governor


class ConcurrencySlot:
    """One held lease, or nothing for an unlimited user. ``release()`` is idempotent."""

    def __init__(self, governor: ResourceGovernor | None = None, handle_id: str | None = None) -> None:
        """Hold ``handle_id`` and renew it while the calling task runs."""
        self._governor = governor
        self._handle_id = handle_id
        self._owner = asyncio.current_task()
        self._keepalive = asyncio.create_task(self._keep_alive()) if handle_id else None

    async def _keep_alive(self) -> None:
        """Renew the lease while the task that took it runs; once that task ends, let the lease lapse."""
        while True:
            await asyncio.sleep(RENEW_EVERY_SEC)
            if self._owner is None or self._owner.done() or self._handle_id is None:
                return
            try:
                await self._governor.renew(self._handle_id, LEASE_TTL_SEC)
            except _NONCRITICAL as exc:
                logger.debug("Concurrency lease renew failed: {!r}", exc)

    async def release(self) -> None:
        """Give the slot back. Completes even inside a cancelled request scope."""
        handle_id, self._handle_id = self._handle_id, None
        if handle_id is None:
            return
        if self._keepalive is not None:
            self._keepalive.cancel()
        with CancelScope(shield=True):
            try:
                await self._governor.release(handle_id)
            except _NONCRITICAL as exc:
                logger.warning("Concurrency lease release failed; it lapses within {} s: {!r}", LEASE_TTL_SEC, exc)


async def acquire_slot(user_id: Any, key: str, category: str) -> ConcurrencySlot:
    """Take one of the user's ``key`` slots, or raise ConcurrencyQuotaExceeded.

    ``category`` is the lease kind, ``"jobs"`` or ``"streams"``; ``key`` names both the
    ``limits.*`` value and the lease bucket. A user id that is not numeric is unlimited.
    A lease-store error admits the request and logs a warning (fail open, like the resolver).
    """
    uid = as_quota_user_id(user_id)
    limit = await user_quota(uid, key)
    if limit is None:
        return ConcurrencySlot()
    if int(limit) < 1:
        raise ConcurrencyQuotaExceeded(0, retry_after=None)
    governor = _lease_governor()
    try:
        decision, handle_id = await governor.reserve(
            RGRequest(
                entity=f"user:{uid}",
                categories={category: {"units": 1, "max_concurrent": int(limit), "ttl_sec": LEASE_TTL_SEC}},
                tags={"policy_id": key, "endpoint": key},
            )
        )
    except _NONCRITICAL as exc:
        logger.warning("Concurrency quota {} unavailable for user {}; admitting: {!r}", key, uid, exc)
        return ConcurrencySlot()
    if not decision.allowed or not handle_id:
        raise ConcurrencyQuotaExceeded(int(limit), retry_after=int(decision.retry_after or 1))
    return ConcurrencySlot(governor, handle_id)
