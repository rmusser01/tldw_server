"""JobManager dependency helpers with caching."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from loguru import logger

try:
    from cachetools import LRUCache

    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False
    logger.warning(
        "cachetools not found. JobManager cache will grow indefinitely. Install with: pip install cachetools"
    )

from tldw_Server_API.app.core.Jobs.manager import JobManager

MAX_CACHED_JOB_MANAGER_INSTANCES = 4

if _HAS_CACHETOOLS:
    _job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
else:
    _job_manager_cache: dict[str, JobManager] = {}

_job_manager_lock = threading.Lock()


def _build_job_manager(db_url: str, db_path: str | None = None) -> JobManager:
    """Create a JobManager instance for the resolved jobs DB URL."""
    backend = "postgres" if db_url.startswith("postgres") else None
    if backend == "postgres":
        return JobManager(backend=backend, db_url=db_url or None)
    return JobManager(db_path=Path(db_path) if db_path else None, backend=backend, db_url=db_url or None)


def _normalize_jobs_db_url(db_url: str) -> str:
    """Normalize JOBS_DB_URL and fall back to in-memory on invalid schemes."""
    if db_url and not (db_url.startswith("postgres") or db_url.startswith("sqlite")):
        logger.warning("Unexpected JOBS_DB_URL scheme; defaulting to in-memory jobs backend.")
        return ""
    return db_url


def get_job_manager() -> JobManager:
    """Return a cached JobManager keyed by JOBS_DB_URL."""
    db_url = _normalize_jobs_db_url((os.getenv("JOBS_DB_URL") or "").strip())
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    cache_key = db_url or f"sqlite:{db_path or 'default'}"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached
        job_manager = _build_job_manager(db_url, db_path or None)
        _job_manager_cache[cache_key] = job_manager
        return job_manager
