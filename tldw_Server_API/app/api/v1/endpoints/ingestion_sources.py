"""API endpoints for user-managed ingestion sources and sync operations."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from tldw_Server_API.app.api.v1.schemas.ingestion_sources import (
    IngestionSourceCapabilitiesResponse,
    IngestionSourceCreateRequest,
    IngestionSourceItemResponse,
    IngestionSourcePatchRequest,
    IngestionSourceResponse,
    IngestionSourceSyncTriggerResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, User
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
    inspect_archive_candidate_file,
    persist_archive_artifact_from_file,
    validate_archive_upload_filename,
)
from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
    validate_git_repository_source,
)
from tldw_Server_API.app.core.Ingestion_Sources.jobs import enqueue_ingestion_source_job
from tldw_Server_API.app.core.Ingestion_Sources.local_directory import validate_local_directory_source
from tldw_Server_API.app.core.Ingestion_Sources.access_policy import (
    can_create_local_directory_ingestion_source,
)
from tldw_Server_API.app.core.Ingestion_Sources.service import (
    create_source_snapshot,
    create_source,
    ensure_ingestion_sources_schema,
    get_source_by_id,
    get_source_item_by_id,
    list_sources_by_user,
    list_source_items,
    update_source,
    update_source_item_state,
    update_source_snapshot,
    validate_source_sink_pair,
)
from tldw_Server_API.app.core.exceptions import IngestionSourceValidationError

router = APIRouter(prefix="/ingestion-sources", tags=["ingestion-sources"])
_ARCHIVE_UPLOAD_CHUNK_SIZE = 1024 * 1024
_DEFAULT_ARCHIVE_UPLOAD_MAX_BYTES = 250 * 1024 * 1024


def _archive_upload_max_bytes() -> int:
    """Return the configured maximum accepted archive upload size in bytes."""
    raw = os.getenv("INGESTION_SOURCES_ARCHIVE_UPLOAD_MAX_BYTES")
    if raw is None or str(raw).strip() == "":
        return _DEFAULT_ARCHIVE_UPLOAD_MAX_BYTES
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return _DEFAULT_ARCHIVE_UPLOAD_MAX_BYTES


async def _stream_archive_upload_to_temp_file(
    archive: UploadFile,
    *,
    filename: str,
) -> dict[str, Any]:
    """Stream an uploaded archive into a temporary file with size accounting."""
    max_bytes = _archive_upload_max_bytes()
    suffix = "".join(Path(filename).suffixes) or ".archive"
    hasher = hashlib.sha256()
    byte_size = 0
    temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_handle.name)
    try:
        while True:
            chunk = await archive.read(_ARCHIVE_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            byte_size += len(chunk)
            if byte_size > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        "Archive upload exceeds the configured maximum size "
                        f"of {max_bytes} bytes"
                    ),
                )
            hasher.update(chunk)
            temp_handle.write(chunk)
        if byte_size <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Archive upload is empty",
            )
        temp_handle.flush()
        return {
            "temp_path": temp_path,
            "byte_size": byte_size,
            "checksum": hasher.hexdigest(),
        }
    except Exception:
        with contextlib.suppress(FileNotFoundError, OSError):
            temp_path.unlink()
        raise
    finally:
        temp_handle.close()
        with contextlib.suppress(Exception):
            await archive.close()


def _prepare_create_payload(payload: IngestionSourceCreateRequest) -> dict[str, Any]:
    """Normalize create payloads and validate local directory sources eagerly."""
    result = payload.model_dump()
    validate_source_sink_pair(
        source_type=str(payload.source_type),
        sink_type=str(payload.sink_type),
    )
    config = dict(result.get("config") or {})
    try:
        config = _prepare_source_config(payload.source_type, config)
    except ValueError as exc:
        raise IngestionSourceValidationError(str(exc)) from exc
    result["config"] = config
    return result


def _prepare_source_config(source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    if source_type == "local_directory":
        normalized = dict(config)
        normalized["path"] = str(validate_local_directory_source(normalized))
        return normalized
    if source_type == "git_repository":
        return validate_git_repository_source(config)
    return dict(config)


def _prepare_patch_payload(
    *,
    existing: dict[str, Any],
    payload: IngestionSourcePatchRequest,
) -> dict[str, Any]:
    patch = payload.model_dump(exclude_unset=True)
    effective_source_type = str(
        patch.get("source_type")
        if patch.get("source_type") is not None
        else existing.get("source_type") or ""
    ).strip()
    effective_sink_type = str(
        patch.get("sink_type")
        if patch.get("sink_type") is not None
        else existing.get("sink_type") or ""
    ).strip()
    if effective_source_type and effective_sink_type:
        validate_source_sink_pair(
            source_type=effective_source_type,
            sink_type=effective_sink_type,
        )
    if patch.get("source_type") is None:
        patch.pop("source_type", None)
    if any(key in patch for key in ("source_type", "config")):
        effective_config = (
            dict(patch.get("config") or {})
            if patch.get("config") is not None
            else dict(existing.get("config") or {})
        )
        try:
            patch["config"] = _prepare_source_config(effective_source_type, effective_config)
        except ValueError as exc:
            raise IngestionSourceValidationError(str(exc)) from exc
    return patch


def _requires_local_directory_entitlement(
    *,
    source_type: str | None,
    config_changed: bool = True,
) -> bool:
    return config_changed and str(source_type or "").strip().lower() == "local_directory"


def _can_create_local_directory_ingestion_source_for_request(
    *,
    current_user: User,
    request: Request,
) -> bool:
    state = getattr(request, "state", None)
    return can_create_local_directory_ingestion_source(
        current_user,
        active_org_id=getattr(state, "active_org_id", None),
        org_ids=getattr(state, "org_ids", None),
    )


def _local_directory_identity_changed(
    *,
    existing: dict[str, Any],
    patch: dict[str, Any],
) -> bool:
    effective_source_type = (
        patch.get("source_type")
        if patch.get("source_type") is not None
        else existing.get("source_type")
    )
    if not _requires_local_directory_entitlement(source_type=effective_source_type):
        return False
    existing_source_type = str(existing.get("source_type") or "").strip().lower()
    patch_source_type = str(patch.get("source_type") or "").strip().lower()
    if "source_type" in patch and patch_source_type != existing_source_type:
        return True
    if "sink_type" in patch and patch.get("sink_type") not in (None, existing.get("sink_type")):
        return True
    if "config" in patch:
        return dict(patch.get("config") or {}) != dict(existing.get("config") or {})
    return False


def _raw_local_directory_identity_changed(
    *,
    existing: dict[str, Any],
    patch: dict[str, Any],
) -> bool:
    effective_source_type = (
        patch.get("source_type")
        if patch.get("source_type") is not None
        else existing.get("source_type")
    )
    if not _requires_local_directory_entitlement(source_type=effective_source_type):
        return False
    if "source_type" in patch and patch.get("source_type") not in (None, existing.get("source_type")):
        return True
    if "sink_type" in patch and patch.get("sink_type") not in (None, existing.get("sink_type")):
        return True
    if "config" in patch and patch.get("config") is not None:
        return dict(patch.get("config") or {}) != dict(existing.get("config") or {})
    return False


@router.post(
    "/",
    response_model=IngestionSourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_ingestion_source(
    payload: IngestionSourceCreateRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
):
    """Create a new ingestion source for the authenticated user."""
    if (
        payload.source_type == "local_directory"
        and not _can_create_local_directory_ingestion_source_for_request(
            current_user=current_user,
            request=request,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local directory ingestion sources are not enabled for this user",
        )
    try:
        prepared_payload = _prepare_create_payload(payload)
    except IngestionSourceValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        row = await create_source(db, user_id=int(current_user.id), payload=prepared_payload)
    return row


@router.get("/", response_model=list[IngestionSourceResponse])
async def list_ingestion_sources(current_user: User = Depends(get_request_user)):
    """List ingestion sources owned by the authenticated user."""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        rows = await list_sources_by_user(db, user_id=int(current_user.id))
    return rows


@router.get("/capabilities", response_model=IngestionSourceCapabilitiesResponse)
async def get_ingestion_source_capabilities(
    request: Request,
    current_user: User = Depends(get_request_user),
) -> IngestionSourceCapabilitiesResponse:
    """Return ingestion-source capabilities for the authenticated user."""
    return IngestionSourceCapabilitiesResponse(
        can_create_local_directory=_can_create_local_directory_ingestion_source_for_request(
            current_user=current_user,
            request=request,
        ),
    )


@router.get("/{source_id}", response_model=IngestionSourceResponse)
async def get_ingestion_source(source_id: int, current_user: User = Depends(get_request_user)):
    """Fetch a single ingestion source by identifier for the authenticated user."""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        row = await get_source_by_id(db, source_id=source_id, user_id=int(current_user.id))
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source not found")
    return row


@router.patch("/{source_id}", response_model=IngestionSourceResponse)
async def patch_ingestion_source(
    source_id: int,
    payload: IngestionSourcePatchRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
):
    """Update mutable ingestion source settings while enforcing identity immutability."""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        existing = await get_source_by_id(db, source_id=source_id, user_id=int(current_user.id))
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source not found")
        raw_patch = payload.model_dump(exclude_unset=True)
        if _raw_local_directory_identity_changed(
            existing=existing,
            patch=raw_patch,
        ) and not _can_create_local_directory_ingestion_source_for_request(
            current_user=current_user,
            request=request,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Local directory ingestion sources are not enabled for this user",
            )
        try:
            patch = _prepare_patch_payload(existing=existing, payload=payload)
            if _local_directory_identity_changed(
                existing=existing,
                patch=patch,
            ) and not _can_create_local_directory_ingestion_source_for_request(
                current_user=current_user,
                request=request,
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Local directory ingestion sources are not enabled for this user",
                )
            row = await update_source(
                db,
                source_id=source_id,
                user_id=int(current_user.id),
                patch=patch,
            )
        except IngestionSourceValidationError as exc:
            detail = str(exc)
            status_code = (
                status.HTTP_409_CONFLICT
                if detail == "Source identity is immutable after the first successful sync"
                else status.HTTP_400_BAD_REQUEST
            )
            raise HTTPException(status_code=status_code, detail=detail) from exc
    return row


@router.get("/{source_id}/items", response_model=list[IngestionSourceItemResponse])
async def list_ingestion_source_items(source_id: int, current_user: User = Depends(get_request_user)):
    """List tracked items currently bound to an ingestion source."""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        row = await get_source_by_id(db, source_id=source_id, user_id=int(current_user.id))
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source not found")
        items = await list_source_items(db, source_id=source_id)
    return items


@router.post(
    "/{source_id}/sync",
    response_model=IngestionSourceSyncTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(check_rate_limit)],
)
async def trigger_ingestion_source_sync(source_id: int, current_user: User = Depends(get_request_user)):
    """Enqueue a manual sync job for an ingestion source."""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        row = await get_source_by_id(db, source_id=source_id, user_id=int(current_user.id))
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source not found")

    job = await asyncio.to_thread(
        enqueue_ingestion_source_job,
        user_id=int(current_user.id),
        source_id=source_id,
    )
    return {
        "status": "queued",
        "source_id": source_id,
        "job_id": job.get("id"),
    }


@router.post(
    "/{source_id}/archive",
    response_model=IngestionSourceSyncTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(check_rate_limit)],
)
async def upload_ingestion_source_archive(
    source_id: int,
    archive: UploadFile = File(...),
    current_user: User = Depends(get_request_user),
):
    """Stage a new archive payload for an archive-backed ingestion source."""
    try:
        filename = validate_archive_upload_filename(archive.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    artifact_storage_path: str | None = None
    staged_upload_path: Path | None = None
    db_pool = await get_db_pool()
    try:
        async with db_pool.transaction() as db:
            await ensure_ingestion_sources_schema(db)
            row = await get_source_by_id(db, source_id=source_id, user_id=int(current_user.id))
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source not found")
            if str(row.get("source_type") or "").strip().lower() != "archive_snapshot":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Archive upload is only supported for archive_snapshot sources",
                )
            previous_snapshot_id = (
                None
                if row.get("last_successful_snapshot_id") is None
                else int(row["last_successful_snapshot_id"])
            )
            sink_type = str(row.get("sink_type") or "notes")

        staged_upload = await _stream_archive_upload_to_temp_file(
            archive,
            filename=filename,
        )
        staged_upload_path = Path(staged_upload["temp_path"])
        try:
            inspection = await asyncio.to_thread(
                inspect_archive_candidate_file,
                archive_path=staged_upload_path,
                filename=filename,
                sink_type=sink_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        async with db_pool.transaction() as db:
            await ensure_ingestion_sources_schema(db)
            snapshot = await create_source_snapshot(
                db,
                source_id=source_id,
                snapshot_kind="archive_snapshot",
                status="staged",
                summary={
                    "filename": inspection["filename"],
                    "previous_snapshot_id": previous_snapshot_id,
                    "item_count": int(inspection["item_count"]),
                },
            )
            artifact = await persist_archive_artifact_from_file(
                db,
                user_id=int(current_user.id),
                source_id=source_id,
                snapshot_id=int(snapshot["id"]),
                filename=filename,
                staged_file_path=staged_upload_path,
                byte_size=int(staged_upload["byte_size"]),
                checksum=str(staged_upload["checksum"]),
            )
            artifact_storage_path = str(artifact.get("storage_path") or "")
            staged_upload_path = None
            await update_source_snapshot(
                db,
                snapshot_id=int(snapshot["id"]),
                summary={"artifact_id": int(artifact["id"])},
            )
    except HTTPException:
        if staged_upload_path is not None:
            with contextlib.suppress(FileNotFoundError, OSError):
                staged_upload_path.unlink()
        if artifact_storage_path:
            with contextlib.suppress(FileNotFoundError, OSError):
                Path(artifact_storage_path).unlink()
        raise
    except Exception:
        if staged_upload_path is not None:
            with contextlib.suppress(FileNotFoundError, OSError):
                staged_upload_path.unlink()
        if artifact_storage_path:
            with contextlib.suppress(FileNotFoundError, OSError):
                Path(artifact_storage_path).unlink()
        raise

    job = await asyncio.to_thread(
        enqueue_ingestion_source_job,
        user_id=int(current_user.id),
        source_id=source_id,
    )
    return {
        "status": "queued",
        "source_id": source_id,
        "job_id": job.get("id"),
        "snapshot_status": "staged",
    }


@router.post("/{source_id}/items/{item_id}/reattach", response_model=IngestionSourceItemResponse)
async def reattach_ingestion_source_item(
    source_id: int,
    item_id: int,
    current_user: User = Depends(get_request_user),
):
    """Reattach a detached notes-backed source item to resume managed sync updates."""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        source = await get_source_by_id(db, source_id=source_id, user_id=int(current_user.id))
        if not source:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source not found")
        if str(source.get("sink_type") or "").strip().lower() != "notes":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reattach is only supported for notes sinks",
            )
        item = await get_source_item_by_id(db, source_id=source_id, item_id=item_id)
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion source item not found")
        if str(item.get("sync_status") or "").strip().lower() != "conflict_detached":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only detached items can be reattached",
            )

        binding = dict(item.get("binding") or {})
        note_id = binding.get("note_id")
        if not note_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Detached item is missing a bound note",
            )

        notes_db = CharactersRAGDB(
            db_path=str(DatabasePaths.get_chacha_db_path(int(current_user.id))),
            client_id=str(current_user.id),
        )
        note = notes_db.get_note_by_id(note_id=str(note_id))
        if note is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Bound note no longer exists",
            )

        binding["sync_status"] = "sync_managed"
        binding["current_version"] = int(note["version"])
        updated = await update_source_item_state(
            db,
            item_id=item_id,
            sync_status="sync_managed",
            binding=binding,
            clear_content_hash=True,
        )
    return updated
