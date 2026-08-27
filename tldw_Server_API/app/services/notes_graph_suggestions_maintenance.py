"""Independent maintenance lifecycle for Notes graph suggestions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenanceScope,
    SuggestionMaintenance,
)


async def _open_owner_database(owner_user_id: str) -> Any:
    user_id = int(owner_user_id)
    return await get_chacha_db_for_user_id(user_id, client_id=str(user_id))


def _close_database(db: Any) -> None:
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
    elif hasattr(db, "close_connection"):
        db.close_connection()


async def _discover_scopes(jobs: JobManager) -> tuple[tuple[MaintenanceScope, ...], tuple[Any, ...]]:
    identities: set[tuple[str, str]] = set()
    for status in ("queued", "processing", "completed", "failed", "cancelled", "quarantined"):
        for job in jobs.list_jobs(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            job_type=JOB_TYPE,
            status=status,
            limit=100,
        ):
            owner = job.get("owner_user_id")
            payload = job.get("payload")
            dataset = payload.get("dataset_id") if isinstance(payload, dict) else None
            if isinstance(owner, str) and owner and isinstance(dataset, str) and dataset:
                identities.add((owner, dataset))
    scopes: list[MaintenanceScope] = []
    databases: list[Any] = []
    for owner, dataset in sorted(identities):
        db = await _open_owner_database(owner)
        databases.append(db)
        scopes.append(MaintenanceScope(db.note_graph_suggestion_store, dataset))
    return tuple(scopes), tuple(databases)


async def run_notes_graph_suggestions_maintenance(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run provider-independent maintenance at startup and once per minute."""

    stop = stop_event or asyncio.Event()
    jobs = JobManager()
    logger.info("Notes graph suggestions maintenance starting")
    while not stop.is_set():
        databases: tuple[Any, ...] = ()
        try:
            scopes, databases = await _discover_scopes(jobs)
            SuggestionMaintenance(jobs=jobs, scopes=scopes).run_pass(now=datetime.now(timezone.utc))
        except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            logger.warning("Notes graph suggestions maintenance pass failed safely")
        finally:
            for db in databases:
                try:
                    _close_database(db)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("Notes graph suggestions maintenance database close skipped")
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass


__all__ = ["run_notes_graph_suggestions_maintenance"]
