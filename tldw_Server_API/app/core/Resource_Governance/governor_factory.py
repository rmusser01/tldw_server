"""
Governor backend factory.

Selects Redis-backed or in-memory governor based on configuration:
- If REDIS_URL is set and Redis is reachable -> use RedisResourceGovernor
- Otherwise -> use MemoryResourceGovernor (default)

This enables horizontal scaling: multiple app instances share rate limiting
state through Redis.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config_sections.redis import redis_url_env

from .governor import MemoryResourceGovernor, ResourceGovernor


def create_governor(
    *,
    redis_url: str | None = None,
    policy_loader: Any | None = None,
    policies: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ResourceGovernor:
    """Create the appropriate governor backend.

    Args:
        redis_url: Redis connection URL. If ``None``, checks the ``REDIS_URL``
            environment variable.
        policy_loader: Policy loader instance passed to the governor.
        policies: Static policy dict (only used by the memory backend when
            *policy_loader* is ``None``).
        **kwargs: Additional keyword arguments forwarded to the chosen
            governor constructor (e.g. ``time_source``, ``ns``).

    Returns:
        A :class:`ResourceGovernor` instance -- Redis-backed when Redis is
        available, in-memory otherwise.
    """
    url = redis_url or redis_url_env()

    if url:
        try:
            from .governor_redis import RedisResourceGovernor  # noqa: WPS433
        except ImportError:
            logger.warning(
                "Redis governor requested but redis package not installed; "
                "falling back to in-memory backend"
            )
            return _create_memory_governor(
                policy_loader=policy_loader, policies=policies, **kwargs
            )

        try:
            # Probe Redis connectivity with a short timeout so startup is not
            # blocked by an unreachable Redis instance.
            import redis as redis_lib

            probe = redis_lib.from_url(url, socket_connect_timeout=3)
            probe.ping()
            probe.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Redis at {} unreachable ({}); falling back to in-memory backend",
                url,
                exc,
            )
            return _create_memory_governor(
                policy_loader=policy_loader, policies=policies, **kwargs
            )

        # RedisResourceGovernor requires a policy_loader (not a static dict).
        redis_kwargs: dict[str, Any] = {}
        if "ns" in kwargs:
            redis_kwargs["ns"] = kwargs["ns"]
        if "time_source" in kwargs:
            redis_kwargs["time_source"] = kwargs["time_source"]

        governor = RedisResourceGovernor(
            policy_loader=policy_loader, **redis_kwargs
        )
        logger.info("Resource Governor: using Redis backend at {}", url)
        return governor

    return _create_memory_governor(
        policy_loader=policy_loader, policies=policies, **kwargs
    )


def _create_memory_governor(
    *,
    policy_loader: Any | None = None,
    policies: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> MemoryResourceGovernor:
    """Instantiate the in-memory governor with supported kwargs."""
    mem_kwargs: dict[str, Any] = {}
    if policy_loader is not None:
        mem_kwargs["policy_loader"] = policy_loader
    if policies is not None:
        mem_kwargs["policies"] = policies
    for key in ("time_source", "backend_label", "default_handle_ttl"):
        if key in kwargs:
            mem_kwargs[key] = kwargs[key]

    logger.info("Resource Governor: using in-memory backend")
    return MemoryResourceGovernor(**mem_kwargs)
