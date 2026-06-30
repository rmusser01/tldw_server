"""Database connection pool metrics exporter.

Provides a lightweight introspection helper that works with asyncpg pools,
queue-based pools, and generic pool wrappers commonly found in the codebase.
The returned dict is safe to serialise as JSON and can be published to
Prometheus / Grafana or included in health-check responses.
"""
from __future__ import annotations

from typing import Any

from loguru import logger


def collect_pool_metrics(pool: Any) -> dict[str, Any]:
    """Collect metrics from a database connection pool.

    Works with asyncpg pools, SQLite connection managers, and generic
    pools that expose a ``_pool`` attribute with a ``qsize()`` method.

    Parameters
    ----------
    pool:
        The pool object to inspect.

    Returns
    -------
    dict
        A JSON-serialisable dict of pool metrics.
    """
    metrics: dict[str, Any] = {
        "pool_type": type(pool).__name__,
        "available": False,
    }

    try:
        # asyncpg pool
        if hasattr(pool, "get_size"):
            metrics.update({
                "available": True,
                "size": pool.get_size(),
                "free_size": (
                    pool.get_idle_size() if hasattr(pool, "get_idle_size") else None
                ),
                "min_size": (
                    pool.get_min_size() if hasattr(pool, "get_min_size") else None
                ),
                "max_size": (
                    pool.get_max_size() if hasattr(pool, "get_max_size") else None
                ),
            })

        # Generic pool with _pool attribute (e.g. queue.Queue-based)
        elif hasattr(pool, "_pool"):
            inner = pool._pool
            if hasattr(inner, "qsize"):
                metrics.update({
                    "available": True,
                    "size": inner.qsize(),
                })
    except Exception as exc:  # noqa: BLE001 - health metrics must be best-effort
        logger.debug(
            "Failed to collect pool metrics for {}: {}",
            type(pool).__name__,
            type(exc).__name__,
        )
        metrics.update({
            "available": False,
            "error": type(exc).__name__,
        })

    return metrics
