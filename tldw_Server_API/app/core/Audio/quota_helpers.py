import os
import sqlite3

from loguru import logger

from tldw_Server_API.app.core.config import load_comprehensive_config

# Optional DB/Redis drivers (for precise exception handling without hard dependencies)
try:  # asyncpg is optional; used when PostgreSQL is configured
    import asyncpg  # type: ignore
except ImportError:  # pragma: no cover - absence is fine
    asyncpg = None  # type: ignore
try:  # aiosqlite may surface errors during SQLite operations
    import aiosqlite  # type: ignore
except ImportError:  # pragma: no cover
    aiosqlite = None  # type: ignore
try:  # redis is optional; used for active stream counters if enabled
    from redis import exceptions as redis_exceptions  # type: ignore
except ImportError:  # pragma: no cover
    redis_exceptions = None  # type: ignore
try:
    # Project-level DB error wrapper used by get_db_pool/DB layer
    from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError as AuthNZDatabaseError  # type: ignore
except ImportError:  # pragma: no cover
    AuthNZDatabaseError = None  # type: ignore

# Build precise exception tuples we’ll catch in quota-limit helpers.
EXPECTED_DB_EXC = ()
if hasattr(sqlite3, "Error"):
    EXPECTED_DB_EXC = (*EXPECTED_DB_EXC, sqlite3.Error)  # type: ignore[attr-defined]
if asyncpg and hasattr(asyncpg, "PostgresError"):
    EXPECTED_DB_EXC = (*EXPECTED_DB_EXC, asyncpg.PostgresError)  # type: ignore[attr-defined]
if aiosqlite and hasattr(aiosqlite, "Error"):
    EXPECTED_DB_EXC = (*EXPECTED_DB_EXC, aiosqlite.Error)  # type: ignore[attr-defined]
if AuthNZDatabaseError is not None:
    EXPECTED_DB_EXC = (*EXPECTED_DB_EXC, AuthNZDatabaseError)  # type: ignore

EXPECTED_REDIS_EXC = ()
if redis_exceptions and hasattr(redis_exceptions, "RedisError"):
    EXPECTED_REDIS_EXC = (*EXPECTED_REDIS_EXC, redis_exceptions.RedisError)  # type: ignore[attr-defined]


def _get_failopen_cap_minutes() -> float:
    """Return per-connection fail-open cap in minutes for streaming quotas.

    Resolution order:
      1) Env var AUDIO_FAILOPEN_CAP_MINUTES (>0)
      2) Config [Audio-Quota] failopen_cap_minutes (>0)
      3) Config [Audio] failopen_cap_minutes (>0)
      4) Default 5.0
    """
    # Env override
    v = os.getenv("AUDIO_FAILOPEN_CAP_MINUTES")
    if v is not None:
        try:
            f = float(v)
            if f > 0:
                return f
        except (ValueError, TypeError) as e:
            logger.debug(f"AUDIO_FAILOPEN_CAP_MINUTES parse failed: {e}")
    # Config-based override
    try:
        cfg = load_comprehensive_config()
        if cfg is not None:
            if cfg.has_section("Audio-Quota"):
                try:
                    f = float(cfg.get("Audio-Quota", "failopen_cap_minutes", fallback=""))
                    if f > 0:
                        return f
                except (ValueError, TypeError) as e:
                    logger.debug(f"[Audio-Quota].failopen_cap_minutes parse failed: {e}")
            if cfg.has_section("Audio"):
                try:
                    f = float(cfg.get("Audio", "failopen_cap_minutes", fallback=""))
                    if f > 0:
                        return f
                except (ValueError, TypeError) as e:
                    logger.debug(f"[Audio].failopen_cap_minutes parse failed: {e}")
    except Exception as e:
        logger.debug(f"Config read for failopen cap failed: {e}")
    return 5.0
