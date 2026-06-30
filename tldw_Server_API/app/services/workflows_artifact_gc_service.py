from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlite3 import Error as SQLiteError

from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Manager import create_workflows_database, get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

_GC_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    SQLiteError,
    TypeError,
    ValueError,
)


def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def _record_artifact_gc_event(
    db: WorkflowsDatabase,
    *,
    tenant_id: str,
    run_id: str,
    artifact_id: str,
    uri: str,
    artifact_type: str | None,
    file_deleted: bool,
    retention_days: int,
) -> None:
    payload: dict[str, object] = {
        "artifact_id": artifact_id,
        "uri": uri,
        "status": "deleted",
        "file_deleted": bool(file_deleted),
        "row_deleted": True,
        "retention_days": int(retention_days),
        "source": "artifact_gc",
    }
    if artifact_type:
        payload["artifact_type"] = artifact_type
    try:
        db.append_event(tenant_id, run_id, "artifact_gc", payload)
    except _GC_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).warning(
            "Artifact GC: failed to append workflow evidence"
        )


async def run_workflows_artifact_gc_worker(stop_event: asyncio.Event) -> None:
    """Background loop to enforce artifact retention by deleting old files and DB rows.

    Env:
      WORKFLOWS_ARTIFACT_GC_ENABLED=true|false (caller controls start)
      WORKFLOWS_ARTIFACT_RETENTION_DAYS=30
      WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC=3600
    Policy:
      - Only file:// artifacts are removed from disk; DB row is removed regardless.
      - Runs are not checked; age is by artifact.created_at timestamp.
    """
    backend = get_content_backend_instance()
    db: WorkflowsDatabase = create_workflows_database(backend=backend)

    interval = int(os.getenv("WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC", "3600"))
    retention_days = int(os.getenv("WORKFLOWS_ARTIFACT_RETENTION_DAYS", "30"))
    logger.info(f"Starting Workflows artifact GC worker (interval={interval}s, retention_days={retention_days})")

    while not stop_event.is_set():
        try:
            cutoff = _now_utc() - timedelta(days=retention_days)
            cutoff_iso = cutoff.isoformat()
            rows = db.list_artifacts_older_than(cutoff_iso)
            deleted = 0
            for r in rows:
                try:
                    artifact_id = str(r.get("artifact_id") or "")
                    run_id = str(r.get("run_id") or "")
                    artifact_type = r.get("type")
                    uri = str(r.get("uri") or "")
                    file_deleted = False
                    if uri.startswith("file://"):
                        fp = Path(uri[7:])
                        try:
                            if fp.exists() and fp.is_file():
                                fp.unlink()
                                file_deleted = True
                        except OSError as fe:
                            logger.bind(error_type=type(fe).__name__).warning(
                                "Artifact GC: failed to delete artifact file"
                            )
                    db.delete_artifact(artifact_id)
                    _record_artifact_gc_event(
                        db,
                        tenant_id=str(r.get("tenant_id") or "default"),
                        run_id=run_id,
                        artifact_id=artifact_id,
                        uri=uri,
                        artifact_type=str(artifact_type) if artifact_type is not None else None,
                        file_deleted=file_deleted,
                        retention_days=retention_days,
                    )
                    deleted += 1
                except _GC_NONCRITICAL_EXCEPTIONS as e:
                    logger.bind(error_type=type(e).__name__).warning(
                        "Artifact GC: error deleting artifact"
                    )
            if deleted:
                logger.info(f"Artifact GC: deleted {deleted} artifacts older than {retention_days} days")
        except _GC_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(error_type=type(e).__name__).warning("Artifact GC loop error")

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval)

    logger.info("Workflows artifact GC worker stopped")
