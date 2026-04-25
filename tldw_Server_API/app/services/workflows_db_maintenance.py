from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_workflows_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.testing import is_truthy

_WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    BackendDatabaseError,
    ConnectionError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return is_truthy(v.lower())


async def run_workflows_db_maintenance(stop_event: asyncio.Event) -> None:
    """Periodic DB maintenance for Workflows (checkpoint/VACUUM), gated by env.

    Env:
      WORKFLOWS_DB_MAINTENANCE_ENABLED=true|false  (caller decides to start)
      WORKFLOWS_DB_MAINTENANCE_INTERVAL_SEC=1800   (seconds between runs)

      For SQLite:
        WORKFLOWS_SQLITE_CHECKPOINT=TRUNCATE|RESTART|PASSIVE (default TRUNCATE)
        WORKFLOWS_SQLITE_VACUUM=true|false (default false)

      For Postgres:
        WORKFLOWS_POSTGRES_VACUUM=true|false (default false)
    """
    backend = get_content_backend_instance()
    db: WorkflowsDatabase = create_workflows_database(backend=backend)

    interval = int(os.getenv("WORKFLOWS_DB_MAINTENANCE_INTERVAL_SEC", "1800"))
    logger.info(
        f"Starting Workflows DB maintenance worker (interval={interval}s, backend={'postgres' if db.backend else 'sqlite'})"
    )

    while not stop_event.is_set():
        try:
            if db.backend and db.backend_type == BackendType.POSTGRESQL:
                # Optional manual VACUUM ANALYZE - autovacuum normally covers this
                if _env_bool("WORKFLOWS_POSTGRES_VACUUM", False):
                    try:
                        db.backend.vacuum()  # type: ignore[union-attr]
                        logger.info("Workflows DB maintenance: VACUUM ANALYZE completed for Postgres backend")
                    except _WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Workflows DB maintenance: Postgres VACUUM failed: {e}")
            else:
                # SQLite: run a checkpoint and optional compact vacuum
                try:
                    checkpoint_mode = os.getenv("WORKFLOWS_SQLITE_CHECKPOINT", "TRUNCATE").upper()
                    if checkpoint_mode not in {"PASSIVE", "RESTART", "TRUNCATE"}:
                        checkpoint_mode = "TRUNCATE"
                    try:
                        db._conn.execute(f"PRAGMA wal_checkpoint({checkpoint_mode});")  # type: ignore[attr-defined]
                    except _WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS as e:
                        # Some environments may not be in WAL mode; ignore
                        logger.debug(f"Workflows DB maintenance: WAL checkpoint skipped: {e}")

                    # PRAGMA optimize is a lightweight hint to SQLite
                    try:
                        db._conn.execute("PRAGMA optimize;")  # type: ignore[attr-defined]
                    except _WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Workflows DB maintenance: PRAGMA optimize skipped: {e}")

                    if _env_bool("WORKFLOWS_SQLITE_VACUUM", False):
                        try:
                            db._conn.execute("VACUUM;")  # type: ignore[attr-defined]
                            logger.info("Workflows DB maintenance: SQLite VACUUM completed")
                        except _WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS as ve:
                            logger.warning(f"Workflows DB maintenance: SQLite VACUUM failed: {ve}")
                except _WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Workflows DB maintenance (SQLite) failed: {e}")
        except _WORKFLOWS_DB_MAINTENANCE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Workflows DB maintenance loop error: {e}")

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval)

    logger.info("Workflows DB maintenance worker stopped")
