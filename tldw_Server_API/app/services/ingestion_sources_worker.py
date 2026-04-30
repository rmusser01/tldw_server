from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
    build_archive_snapshot_from_bytes_with_failures,
    load_archive_artifact_bytes,
    prune_archive_source_retention,
)
from tldw_Server_API.app.core.Ingestion_Sources.diffing import diff_snapshots
from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
    build_git_repository_snapshot_with_failures,
)
from tldw_Server_API.app.core.Ingestion_Sources.jobs import DOMAIN, ingestion_sources_queue
from tldw_Server_API.app.core.Ingestion_Sources.local_directory import (
    build_local_directory_snapshot_with_failures,
)
from tldw_Server_API.app.core.Ingestion_Sources.service import (
    create_source_snapshot,
    ensure_ingestion_sources_schema,
    finish_source_sync_job,
    get_source_artifact_by_id,
    get_source_by_id,
    get_latest_source_snapshot,
    list_source_items,
    record_ingestion_item_event,
    start_source_sync_job,
    update_source_artifact,
    update_source_snapshot,
    upsert_source_item,
)
from tldw_Server_API.app.core.Ingestion_Sources.sinks.media_sink import apply_media_change
from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import apply_notes_change
from tldw_Server_API.app.core.testing import env_flag_enabled

try:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
except ImportError:  # pragma: no cover - optional
    get_db_pool = None  # type: ignore

try:
    from tldw_Server_API.app.core.External_Sources.connectors_service import (
        get_account_tokens,
    )
except ImportError:  # pragma: no cover - optional
    get_account_tokens = None  # type: ignore

try:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
        CharactersRAGDBError as NotesDatabaseError,
    )
except ImportError:  # pragma: no cover - optional
    NotesDatabaseError = None  # type: ignore

try:
    from tldw_Server_API.app.core.DB_Management.media_db.errors import (
        DatabaseError as MediaDatabaseError,
    )
except ImportError:  # pragma: no cover - optional
    MediaDatabaseError = None  # type: ignore

try:
    from tldw_Server_API.app.core.Jobs.manager import JobManager
except ImportError:  # pragma: no cover - optional
    JobManager = None  # type: ignore

_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_SINK_ITEM_EXCEPTIONS = tuple(
    exc
    for exc in (
        NotesDatabaseError,
        MediaDatabaseError,
    )
    if exc is not None
) + _NONCRITICAL_EXCEPTIONS
_SINK_FAILURE_ERROR_LABEL = "ingestion_source_sink_failed"
_ITEM_FAILURE_ERROR_LABEL = "ingestion_source_item_failed"
_SYNC_FAILURE_ERROR_LABEL = "ingestion_source_sync_failed"


class _AccountTokenPool(Protocol):
    """Minimal pool interface required for linked-account token lookups."""

    def transaction(self) -> Any:
        ...


def _previous_snapshot_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, str | None]]:
    previous: dict[str, dict[str, str | None]] = {}
    for row in rows:
        relative_path = str(row.get("normalized_relative_path") or "").strip()
        if not relative_path:
            continue
        previous[relative_path] = {
            "relative_path": relative_path,
            "content_hash": row.get("content_hash"),
        }
    return previous


def _iter_changes(diff_result: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    ordered: list[tuple[str, dict[str, Any]]] = []
    for event_type in ("created", "changed", "deleted"):
        for item in diff_result.get(event_type, []):
            ordered.append((event_type, item))
    return ordered


def _build_media_binding(result: dict[str, Any], previous_binding: dict[str, Any]) -> dict[str, Any]:
    action = str(result.get("action") or "").strip().lower()
    if action in {"created", "version_created"} and result.get("media_id") is not None:
        return {"media_id": int(result["media_id"])}
    if action == "archived":
        return {}
    return dict(previous_binding)


def _build_notes_binding(notes_db, result: dict[str, Any], previous_binding: dict[str, Any]) -> dict[str, Any]:
    action = str(result.get("action") or "").strip().lower()
    if action in {"created", "updated"} and result.get("note_id"):
        note_id = str(result["note_id"])
        note = notes_db.get_note_by_id(note_id=note_id) or {}
        return {
            "note_id": note_id,
            "sync_status": str(result.get("sync_status") or "sync_managed"),
            "current_version": int(note.get("version") or previous_binding.get("current_version") or 1),
        }
    if action in {"skipped_detached", "detached_conflict"}:
        binding = dict(previous_binding)
        binding["sync_status"] = "conflict_detached"
        note_id = binding.get("note_id")
        if note_id:
            note = notes_db.get_note_by_id(note_id=str(note_id)) or {}
            if note:
                binding["current_version"] = int(note.get("version") or binding.get("current_version") or 1)
        return binding
    if action == "archived":
        return {}
    return dict(previous_binding)


def _sync_status_for_result(
    *,
    sink_type: str,
    event_type: str,
    result: dict[str, Any],
    previous_row: dict[str, Any] | None,
) -> str:
    action = str(result.get("action") or "").strip().lower()
    if sink_type == "notes":
        if action in {"created", "updated"}:
            return str(result.get("sync_status") or "sync_managed")
        if action in {"skipped_detached", "detached_conflict"}:
            return "conflict_detached"
    else:
        if action in {"created", "version_created"}:
            return "active"

    if action == "archived":
        return "archived_upstream_removed"
    if event_type == "deleted":
        return str((previous_row or {}).get("sync_status") or "missing_upstream")
    return str((previous_row or {}).get("sync_status") or "active")


def _create_sink_db(*, sink_type: str, user_id: int):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    if sink_type == "notes":
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

        return CharactersRAGDB(
            db_path=str(DatabasePaths.get_chacha_db_path(user_id)),
            client_id=str(user_id),
        )

    return create_media_database(
        client_id=str(user_id),
        db_path=str(DatabasePaths.get_media_db_path(user_id)),
    )


def _apply_change_to_sink(
    *,
    sink_db,
    sink_type: str,
    binding: dict[str, Any],
    change: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    if sink_type == "notes":
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError as NotesConflictError

        try:
            return apply_notes_change(
                sink_db,
                binding=binding or None,
                change=change,
                policy=policy,
            )
        except NotesConflictError:
            return {"action": "detached_conflict", "sync_status": "conflict_detached"}

    return apply_media_change(
        sink_db,
        binding=binding or None,
        change=change,
        policy=policy,
    )


def _load_local_directory_snapshot_data(
    *,
    config: dict[str, Any],
    sink_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return build_local_directory_snapshot_with_failures(
        config,
        sink_type=sink_type,
    )


def _load_archive_snapshot_data(
    *,
    artifact: dict[str, Any],
    filename: str,
    sink_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    archive_bytes = load_archive_artifact_bytes(artifact)
    return build_archive_snapshot_from_bytes_with_failures(
        archive_bytes=archive_bytes,
        filename=filename,
        sink_type=sink_type,
    )


def _load_git_repository_snapshot_data(
    *,
    config: dict[str, Any],
    sink_type: str,
    access_token: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return build_git_repository_snapshot_with_failures(
        config,
        sink_type=sink_type,
        access_token=access_token,
    )


async def _resolve_git_repository_access_token(
    *,
    pool: _AccountTokenPool,
    source: dict[str, Any],
    user_id: int,
) -> str | None:
    if get_account_tokens is None:
        return None
    config = source.get("config") or {}
    mode = str(config.get("mode") or "").strip().lower()
    if mode != "remote_github_repo":
        return None
    account_id = config.get("account_id")
    if account_id in (None, ""):
        return None
    async with pool.transaction() as db:
        tokens = await get_account_tokens(db, user_id=user_id, account_id=int(account_id))
    if not tokens:
        return None
    token = str(tokens.get("access_token") or "").strip()
    return token or None


async def _load_current_source_snapshot(
    *,
    pool,
    source: dict[str, Any],
    source_id: int,
    sink_type: str,
    user_id: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    source_type = str(source.get("source_type") or "").strip().lower()
    if source_type == "local_directory":
        current_items, extraction_failures = await asyncio.to_thread(
            _load_local_directory_snapshot_data,
            config=source.get("config") or {},
            sink_type=sink_type,
        )
        return current_items, extraction_failures, None, None

    if source_type == "git_repository":
        access_token = await _resolve_git_repository_access_token(
            pool=pool,
            source=source,
            user_id=user_id,
        )
        current_items, extraction_failures = await asyncio.to_thread(
            _load_git_repository_snapshot_data,
            config=source.get("config") or {},
            sink_type=sink_type,
            access_token=access_token,
        )
        return current_items, extraction_failures, None, None

    if source_type != "archive_snapshot":
        raise NotImplementedError(f"Sync execution not implemented for source type '{source_type}'.")

    async with pool.transaction() as db:
        staged_snapshot = await get_latest_source_snapshot(
            db,
            source_id=source_id,
            status="staged",
        )
    if not staged_snapshot:
        raise ValueError(f"No staged archive snapshot is available for source {source_id}.")

    snapshot_summary = staged_snapshot.get("summary") or {}
    artifact_id = snapshot_summary.get("artifact_id")
    if artifact_id is None:
        current_items = {
            str(path): dict(item)
            for path, item in snapshot_summary.get("items", {}).items()
        }
        return current_items, {}, staged_snapshot, None

    async with pool.transaction() as db:
        staged_artifact = await get_source_artifact_by_id(
            db,
            artifact_id=int(artifact_id),
        )
    if not staged_artifact:
        raise ValueError(
            f"Archive artifact {artifact_id} is not available for source {source_id}."
        )

    current_items, extraction_failures = await asyncio.to_thread(
        _load_archive_snapshot_data,
        artifact=staged_artifact,
        filename=str(
            (staged_artifact.get("metadata") or {}).get("filename")
            or snapshot_summary.get("filename")
            or "archive.zip"
        ),
        sink_type=sink_type,
    )
    return current_items, extraction_failures, staged_snapshot, staged_artifact


async def _apply_snapshot_changes(
    *,
    db,
    sink_db,
    sink_type: str,
    policy: str,
    source_id: int,
    jid: int,
    current_items: dict[str, dict[str, Any]],
    extraction_failures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    previous_rows = await list_source_items(db, source_id=source_id, include_absent=False)
    all_rows = await list_source_items(db, source_id=source_id, include_absent=True)
    previous_map = _previous_snapshot_map(previous_rows)
    all_rows_map = {
        str(row.get("normalized_relative_path") or ""): row
        for row in all_rows
        if row.get("normalized_relative_path")
    }

    diff_result = diff_snapshots(previous=previous_map, current=current_items)
    failed_paths = set(extraction_failures)
    if failed_paths:
        diff_result["deleted"] = [
            item
            for item in diff_result.get("deleted", [])
            if str(item.get("relative_path") or "") not in failed_paths
        ]

    result_summary: dict[str, Any] = {
        "status": "completed",
        "source_id": source_id,
        "processed": 0,
        "created": len(diff_result.get("created", [])),
        "changed": len(diff_result.get("changed", [])),
        "deleted": len(diff_result.get("deleted", [])),
        "unchanged": len(diff_result.get("unchanged", [])),
        "detached_conflicts": 0,
        "ingestion_failed_items": len(extraction_failures),
        "sink_failed_items": 0,
        "degraded_items": len(extraction_failures),
        "current_item_count": len(current_items) + len(extraction_failures),
    }

    for event_type, raw_change in _iter_changes(diff_result):
        change = dict(raw_change)
        change["event_type"] = event_type
        change["source_id"] = source_id
        relative_path = str(change.get("relative_path") or "").strip()
        existing_row = all_rows_map.get(relative_path)
        existing_binding = (
            dict(existing_row.get("binding") or {})
            if existing_row
            else {}
        )
        try:
            result = _apply_change_to_sink(
                sink_db=sink_db,
                sink_type=sink_type,
                binding=existing_binding,
                change=change,
                policy=policy,
            )
        except _SINK_ITEM_EXCEPTIONS as exc:
            preserved_content_hash = None if existing_row is None else existing_row.get("content_hash")
            preserved_present_in_source = (
                bool(existing_row.get("present_in_source"))
                if existing_row is not None
                else event_type != "deleted"
            )
            updated_row = await upsert_source_item(
                db,
                source_id=source_id,
                normalized_relative_path=relative_path,
                content_hash=None if preserved_content_hash is None else str(preserved_content_hash),
                sync_status="degraded_sink_error",
                binding=existing_binding,
                present_in_source=preserved_present_in_source,
            )
            all_rows_map[relative_path] = updated_row
            await record_ingestion_item_event(
                db,
                source_id=source_id,
                item_path=relative_path,
                event_type="sink_failed",
                payload={
                    "action": "sink_failed",
                    "job_id": str(jid),
                    "sync_status": "degraded_sink_error",
                    "error": _SINK_FAILURE_ERROR_LABEL,
                    "error_type": type(exc).__name__,
                    "event_type": event_type,
                },
            )
            result_summary["degraded_items"] += 1
            result_summary["sink_failed_items"] += 1
            continue

        action = str(result.get("action") or "").strip().lower()
        if sink_type == "notes":
            binding = _build_notes_binding(sink_db, result, existing_binding)
        else:
            binding = _build_media_binding(result, existing_binding)
        sync_status = _sync_status_for_result(
            sink_type=sink_type,
            event_type=event_type,
            result=result,
            previous_row=existing_row,
        )
        if sync_status == "conflict_detached":
            result_summary["detached_conflicts"] += 1
        present_in_source = event_type != "deleted"
        content_hash = change.get("content_hash") if present_in_source else None

        updated_row = await upsert_source_item(
            db,
            source_id=source_id,
            normalized_relative_path=relative_path,
            content_hash=None if content_hash is None else str(content_hash),
            sync_status=sync_status,
            binding=binding,
            present_in_source=present_in_source,
        )
        all_rows_map[relative_path] = updated_row
        await record_ingestion_item_event(
            db,
            source_id=source_id,
            item_path=relative_path,
            event_type=event_type,
            payload={
                "action": action,
                "job_id": str(jid),
                "sync_status": sync_status,
            },
        )
        result_summary["processed"] += 1

    for relative_path in sorted(failed_paths):
        failure = extraction_failures[relative_path]
        existing_row = all_rows_map.get(relative_path)
        existing_binding = dict(existing_row.get("binding") or {}) if existing_row else {}
        preserved_content_hash = None if existing_row is None else existing_row.get("content_hash")
        updated_row = await upsert_source_item(
            db,
            source_id=source_id,
            normalized_relative_path=relative_path,
            content_hash=None if preserved_content_hash is None else str(preserved_content_hash),
            sync_status="degraded_ingestion_error",
            binding=existing_binding,
            present_in_source=True,
        )
        all_rows_map[relative_path] = updated_row
        await record_ingestion_item_event(
            db,
            source_id=source_id,
            item_path=relative_path,
            event_type="ingestion_failed",
            payload={
                "action": "ingestion_failed",
                "job_id": str(jid),
                "sync_status": "degraded_ingestion_error",
                "error": _ITEM_FAILURE_ERROR_LABEL,
            },
        )

    return result_summary


async def _process_sync_job(
    jm,
    jid: int,
    lease_id: str | None,
    worker_id: str,
    source_id: int,
    user_id: int,
) -> None:
    if get_db_pool is None:
        raise RuntimeError("Database pool unavailable for ingestion sources worker.")

    pool = await get_db_pool()
    staged_snapshot: dict[str, Any] | None = None
    staged_artifact: dict[str, Any] | None = None

    async with pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        source = await get_source_by_id(db, source_id=source_id, user_id=user_id)
        if not source:
            raise ValueError("Ingestion source not found or not owned by user.")
        state = await start_source_sync_job(db, source_id=source_id, job_id=str(jid))

    if str(state.get("active_job_id") or "") != str(jid):
        jm.fail_job(
            jid,
            error=f"Active sync job already exists for source {source_id}",
            retryable=True,
            backoff_seconds=30,
            worker_id=worker_id,
            lease_id=lease_id or None,
            completion_token=lease_id or None,
        )
        return

    try:
        if not source.get("enabled", True):
            raise ValueError(f"Ingestion source {source_id} is disabled.")
        sink_type = str(source.get("sink_type") or "").strip().lower()
        policy = str(source.get("policy") or "canonical").strip().lower()
        current_items, extraction_failures, staged_snapshot, staged_artifact = await _load_current_source_snapshot(
            pool=pool,
            source=source,
            source_id=source_id,
            sink_type=sink_type,
            user_id=user_id,
        )
        sink_db = _create_sink_db(sink_type=sink_type, user_id=user_id)

        async with pool.transaction() as db:
            result_summary = await _apply_snapshot_changes(
                db=db,
                sink_db=sink_db,
                sink_type=sink_type,
                policy=policy,
                source_id=source_id,
                jid=jid,
                current_items=current_items,
                extraction_failures=extraction_failures,
            )

            if staged_snapshot:
                snapshot = await update_source_snapshot(
                    db,
                    snapshot_id=int(staged_snapshot["id"]),
                    status="success",
                    summary=result_summary,
                )
                if staged_artifact:
                    await update_source_artifact(
                        db,
                        artifact_id=int(staged_artifact["id"]),
                        status="active",
                    )
            else:
                snapshot = await create_source_snapshot(
                    db,
                    source_id=source_id,
                    snapshot_kind=str(source.get("source_type") or "local_directory"),
                    status="success",
                    summary=result_summary,
                )
            await finish_source_sync_job(
                db,
                source_id=source_id,
                job_id=str(jid),
                outcome="success",
                snapshot_id=int(snapshot["id"]),
            )
            if staged_snapshot:
                with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                    await prune_archive_source_retention(db, source_id=source_id)

        jm.complete_job(
            jid,
            result=result_summary,
            worker_id=worker_id,
            lease_id=lease_id or None,
            completion_token=lease_id or None,
        )
    except _NONCRITICAL_EXCEPTIONS:
        with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
            async with pool.transaction() as db:
                if staged_snapshot:
                    await update_source_snapshot(
                        db,
                        snapshot_id=int(staged_snapshot["id"]),
                        status="failed",
                        summary={
                            "status": "failed",
                            "error": _SYNC_FAILURE_ERROR_LABEL,
                        },
                    )
                if staged_artifact:
                    await update_source_artifact(
                        db,
                        artifact_id=int(staged_artifact["id"]),
                        status="failed",
                        metadata={"error": _SYNC_FAILURE_ERROR_LABEL},
                    )
                await finish_source_sync_job(
                    db,
                    source_id=source_id,
                    job_id=str(jid),
                    outcome="failure",
                    error=_SYNC_FAILURE_ERROR_LABEL,
                )
                if staged_snapshot:
                    with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                        await prune_archive_source_retention(db, source_id=source_id)
        jm.fail_job(
            jid,
            error=_SYNC_FAILURE_ERROR_LABEL,
            retryable=False,
            worker_id=worker_id,
            lease_id=lease_id or None,
            completion_token=lease_id or None,
        )


async def run_ingestion_sources_worker(stop_event: asyncio.Event | None = None) -> None:
    if JobManager is None:
        logger.warning("Jobs manager unavailable; ingestion sources worker disabled")
        return

    jm = JobManager()
    worker_id = "ingestion-sources-worker"
    poll_sleep = float(os.getenv("INGESTION_SOURCES_POLL_INTERVAL_SECONDS", "1.0") or "1.0")
    queue_name = ingestion_sources_queue()

    logger.info("Starting ingestion sources worker")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping ingestion sources worker on shutdown signal")
            return
        try:
            job = jm.acquire_next_job(
                domain=DOMAIN,
                queue=queue_name,
                lease_seconds=120,
                worker_id=worker_id,
            )
            if not job:
                await asyncio.sleep(poll_sleep)
                continue

            payload = job.get("payload") or {}
            await _process_sync_job(
                jm,
                jid=int(job["id"]),
                lease_id=str(job.get("lease_id") or ""),
                worker_id=worker_id,
                source_id=int(payload.get("source_id")),
                user_id=int(job.get("owner_user_id") or payload.get("user_id") or 0),
            )
        except _NONCRITICAL_EXCEPTIONS:
            await asyncio.sleep(poll_sleep)


async def start_ingestion_sources_worker() -> asyncio.Task | None:
    if not env_flag_enabled("INGESTION_SOURCES_WORKER_ENABLED"):
        return None
    stop = asyncio.Event()
    return asyncio.create_task(run_ingestion_sources_worker(stop), name="ingestion-sources-worker")
