"""Lifecycle-managed maintenance for owner-scoped Notes graph projections."""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)
from tldw_Server_API.app.core.Notes.wikilinks import WIKILINK_PARSER_VERSION
from tldw_Server_API.app.core.Notes_Graph.projection_service import (
    NoteGraphProjectionService,
)
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

_DEFAULT_INTERVAL_SECONDS = 5.0
_DEFAULT_BATCH_LIMIT = 50
_DEFAULT_REBUILD_PAGE_LIMIT = 100


def snapshot_cached_chacha_db_instances() -> tuple[CharactersRAGDB, ...]:
    """Load the API cache lazily so disabled workers have no startup side effects."""

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        snapshot_cached_chacha_db_instances as snapshot,
    )

    return snapshot()


def run_notes_graph_projection_maintenance_once(
    db: CharactersRAGDB,
    *,
    batch_limit: int = _DEFAULT_BATCH_LIMIT,
    rebuild_page_limit: int = _DEFAULT_REBUILD_PAGE_LIMIT,
) -> int:
    """Run one bounded pass for one already-authenticated owner database."""

    service = NoteGraphProjectionService(db)
    status = db.note_graph_projection_store.get_projection_status()
    if status.parser_version != WIKILINK_PARSER_VERSION:
        service.prepare_rebuild()
        status = db.note_graph_projection_store.get_projection_status()
    if status.rebuild_state in {"pending", "running"}:
        service.queue_rebuild_page(limit=rebuild_page_limit)
    return service.process_dirty(limit=batch_limit)


async def run_notes_graph_projection_worker(
    stop_event: asyncio.Event,
    *,
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
) -> None:
    """Maintain only cached owner-bound databases until lifecycle shutdown."""

    while not stop_event.is_set():
        for db in snapshot_cached_chacha_db_instances():
            try:
                await asyncio.to_thread(run_notes_graph_projection_maintenance_once, db)
            except (CharactersRAGDBError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Notes graph projection maintenance pass failed ({})",
                    type(exc).__name__,
                )
        if stop_event.is_set():
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


def provide_notes_graph_projection_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return the declarative Notes graph maintenance worker spec."""

    return (
        stop_event_worker_spec(
            name="notes_graph_projection_maintenance_task",
            worker_service=run_notes_graph_projection_worker,
            category="notes-graph",
            phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            enabled=_projection_worker_enabled,
        ),
    )


def _projection_worker_enabled(context: WorkerLifecycleContext) -> bool:
    raw = os.getenv("NOTES_GRAPH_PROJECTION_MAINTENANCE_ENABLED")
    if raw is None or not raw.strip():
        return not context.test_mode
    return is_truthy(raw)


__all__ = [
    "provide_notes_graph_projection_worker_specs",
    "run_notes_graph_projection_maintenance_once",
    "run_notes_graph_projection_worker",
]
