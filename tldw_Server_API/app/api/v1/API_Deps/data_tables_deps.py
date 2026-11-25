"""Job dependencies shared by native and Workspace Data Tables routes."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from cachetools import LRUCache

from tldw_Server_API.app.core.Jobs.manager import JobManager

MAX_CACHED_JOB_MANAGER_INSTANCES = 4
_job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
_job_manager_lock = threading.Lock()


def data_tables_jobs_queue() -> str:
    """Return the configured jobs queue for data tables workers."""
    return (os.getenv("DATA_TABLES_JOBS_QUEUE") or "default").strip() or "default"


def get_job_manager() -> JobManager:
    """Return a cached JobManager instance keyed by JOBS_DB_URL or JOBS_DB_PATH."""
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    cache_key = f"url:{db_url}" if db_url else f"path:{db_path or 'default'}"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached

        if db_url:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)
        elif db_path:
            job_manager = JobManager(db_path=Path(db_path))
        else:
            job_manager = JobManager()

        _job_manager_cache[cache_key] = job_manager
        return job_manager
