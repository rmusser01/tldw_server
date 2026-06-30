from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path as PathlibPath

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.file_artifacts_schemas import (
    FileArtifactResponse,
    FileArtifactsPurgeRequest,
    FileArtifactsPurgeResponse,
    FileCreateRequest,
    FileCreateResponse,
    FileDeleteResponse,
    ReferenceImageListItem,
    ReferenceImageListResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import FileArtifactsError, file_artifacts_http_status
from tldw_Server_API.app.core.File_Artifacts.file_artifacts_service import FileArtifactsService
from tldw_Server_API.app.core.Image_Generation.reference_images import (
    ReferenceImageOperationalError,
    list_reference_image_candidates,
)

router = APIRouter(prefix="/files", tags=["files"])

_EXPORT_MIME_TYPES = {
    "ics": "text/calendar",
    "md": "text/markdown",
    "html": "text/html",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "json": "application/json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}
_EXPORT_RESPONSE_CONTENT = {content_type: {} for content_type in sorted(set(_EXPORT_MIME_TYPES.values()))}

def _file_artifacts_http_exception(exc: FileArtifactsError) -> HTTPException:
    detail = exc.detail if exc.detail is not None else exc.code
    status_code = file_artifacts_http_status(exc)
    return HTTPException(status_code=status_code, detail=detail)


def _resolve_export_path_for_user(user_id: int, path_value: str) -> PathlibPath:
    base_dir = DatabasePaths.get_user_temp_outputs_dir(user_id)
    try:
        base_resolved = base_dir.resolve(strict=False)
    except Exception as exc:
        logger.error("files: failed to resolve temp outputs base dir")
        raise HTTPException(status_code=500, detail="storage_unavailable") from exc

    candidate = PathlibPath(path_value)
    if candidate.is_absolute():
        raise HTTPException(status_code=400, detail="invalid_path")
    if len(candidate.parts) != 1:
        raise HTTPException(status_code=400, detail="invalid_path")
    candidate_name = candidate.name
    if not candidate_name or candidate_name in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid_path")
    if os.sep in candidate_name or (os.altsep and os.altsep in candidate_name):
        raise HTTPException(status_code=400, detail="invalid_path")
    if not re.match(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$", candidate_name):
        raise HTTPException(status_code=400, detail="invalid_path")

    resolved = (base_resolved / candidate_name).resolve(strict=False)
    if not resolved.is_relative_to(base_resolved):
        raise HTTPException(status_code=400, detail="invalid_path")
    return resolved


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _clear_export_state(
    *,
    user_id: int,
    file_id: int,
    row,
    consumed_at: str | None,
) -> None:
    try:
        with CollectionsDatabase.for_user(user_id=user_id) as cdb:
            cdb.update_file_artifact_export(
                file_id,
                export_status="none",
                export_format=row.export_format,
                export_storage_path=None,
                export_bytes=row.export_bytes,
                export_content_type=row.export_content_type,
                export_job_id=row.export_job_id,
                export_expires_at=row.export_expires_at,
                export_consumed_at=consumed_at,
            )
    except Exception:
        logger.warning("files.export: failed to clear export state")


@router.post(
    "/create",
    response_model=FileCreateResponse,
    summary="Create a structured file artifact",
)
async def create_file_artifact(
    request: FileCreateRequest,
    response: Response,
    http_request: Request,
    cdb: CollectionsDatabase = Depends(get_collections_db_for_user),
    current_user: User = Depends(get_request_user),
) -> FileCreateResponse:
    service = FileArtifactsService(cdb, user_id=current_user.id)
    request_id = getattr(getattr(http_request, "state", None), "request_id", None) or http_request.headers.get("X-Request-ID")
    try:
        artifact, status_code = await service.create_artifact(request, request_id=request_id)
    except FileArtifactsError as exc:
        raise _file_artifacts_http_exception(exc) from exc
    response.status_code = status_code
    return FileCreateResponse(artifact=artifact)


@router.get(
    "/reference-images",
    response_model=ReferenceImageListResponse,
    summary="List picker-safe managed reference images",
)
async def get_reference_images(
    media_db = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> ReferenceImageListResponse:
    try:
        items = await list_reference_image_candidates(media_db, user_id=current_user.id)
    except ReferenceImageOperationalError as exc:
        raise HTTPException(status_code=503, detail="reference_image_storage_unavailable") from exc
    return ReferenceImageListResponse(
        items=[
            ReferenceImageListItem(
                file_id=item.file_id,
                title=item.title,
                mime_type=item.mime_type,
                width=item.width,
                height=item.height,
                created_at=item.created_at,
            )
            for item in items
        ]
    )


@router.get(
    "/{file_id}",
    response_model=FileArtifactResponse,
    summary="Get a structured file artifact",
)
async def get_file_artifact(
    file_id: int,
    cdb: CollectionsDatabase = Depends(get_collections_db_for_user),
    current_user: User = Depends(get_request_user),
) -> FileArtifactResponse:
    service = FileArtifactsService(cdb, user_id=current_user.id)
    try:
        artifact = service.get_artifact(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file_artifact_not_found") from None
    return FileArtifactResponse(artifact=artifact)


@router.get(
    "/{file_id}/export",
    summary="Download a file artifact export",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Raw file artifact export",
            "content": _EXPORT_RESPONSE_CONTENT,
        },
    },
)
async def export_file_artifact(
    file_id: int,
    format: str = Query(..., description="Export format (ics|md|html|xlsx|csv|json|png|jpg|webp)"),
    cdb: CollectionsDatabase = Depends(get_collections_db_for_user),
    current_user: User = Depends(get_request_user),
) -> FileResponse:
    if format not in _EXPORT_MIME_TYPES:
        raise HTTPException(status_code=422, detail="unsupported_export_format")
    try:
        row = cdb.get_file_artifact(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file_artifact_not_found") from None

    consumed_at = _parse_iso_datetime(getattr(row, "export_consumed_at", None))
    if consumed_at is not None:
        raise HTTPException(status_code=409, detail="export_consumed")

    if row.export_status != "ready" or not row.export_storage_path:
        raise HTTPException(status_code=404, detail="export_not_ready")
    if row.export_format and row.export_format != format:
        raise HTTPException(status_code=404, detail="export_format_mismatch")

    user_id = int(current_user.id)
    expires_at = _parse_iso_datetime(getattr(row, "export_expires_at", None))
    now = datetime.now(timezone.utc)
    if expires_at is not None and expires_at <= now:
        if row.export_storage_path:
            try:
                path = _resolve_export_path_for_user(user_id, row.export_storage_path)
                if path.exists():
                    path.unlink()
            except Exception:
                logger.warning("files.export: failed to delete expired export file")
        _clear_export_state(
            user_id=user_id,
            file_id=file_id,
            row=row,
            consumed_at=None,
        )
        raise HTTPException(status_code=404, detail="export_expired")

    path = _resolve_export_path_for_user(user_id, row.export_storage_path)
    if not path.exists():
        _clear_export_state(
            user_id=user_id,
            file_id=file_id,
            row=row,
            consumed_at=None,
        )
        raise HTTPException(status_code=404, detail="export_missing")

    consumed_at_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if not cdb.consume_file_artifact_export(file_id, consumed_at=consumed_at_iso):
        raise HTTPException(status_code=409, detail="export_consumed")

    media_type = row.export_content_type or _EXPORT_MIME_TYPES.get(format, "application/octet-stream")
    def _consume_export_file() -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            logger.warning("files.export: failed to delete export file")
        _clear_export_state(
            user_id=user_id,
            file_id=file_id,
            row=row,
            consumed_at=consumed_at_iso,
        )

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
        background=BackgroundTask(_consume_export_file),
    )


@router.delete(
    "/{file_id}",
    response_model=FileDeleteResponse,
    summary="Delete a file artifact (soft delete by default)",
)
async def delete_file_artifact(
    file_id: int,
    hard: bool = False,
    delete_file: bool = False,
    cdb: CollectionsDatabase = Depends(get_collections_db_for_user),
    current_user: User = Depends(get_request_user),
) -> FileDeleteResponse:
    fs_deleted = False
    user_id = int(current_user.id)
    if hard and delete_file:
        try:
            row = cdb.get_file_artifact(file_id, include_deleted=True)
        except KeyError:
            raise HTTPException(status_code=404, detail="file_artifact_not_found") from None
        if row.export_storage_path:
            try:
                path = _resolve_export_path_for_user(user_id, row.export_storage_path)
                if path.exists():
                    path.unlink()
                    fs_deleted = True
            except HTTPException:
                logger.warning("files.delete: invalid export path")
            except Exception:
                logger.warning("files.delete: failed to delete export file")
    ok = cdb.delete_file_artifact(file_id, hard=hard)
    if not ok:
        raise HTTPException(status_code=404, detail="file_artifact_not_found")
    return FileDeleteResponse(success=True, file_deleted=fs_deleted)


@router.post(
    "/purge",
    response_model=FileArtifactsPurgeResponse,
    summary="Purge expired and aged soft-deleted file artifacts",
)
async def purge_file_artifacts(
    payload: FileArtifactsPurgeRequest = Body(default=FileArtifactsPurgeRequest()),
    cdb: CollectionsDatabase = Depends(get_collections_db_for_user),
    current_user: User = Depends(get_request_user),
) -> FileArtifactsPurgeResponse:
    user_id = int(current_user.id)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    candidates = cdb.list_file_artifacts_for_purge(
        now_iso=now,
        soft_deleted_grace_days=payload.soft_deleted_grace_days,
        include_retention=payload.include_retention,
    )
    files_deleted = 0
    if payload.delete_files:
        for file_id, storage_path in list(candidates.items()):
            if not storage_path:
                continue
            try:
                path = _resolve_export_path_for_user(user_id, storage_path)
                if path.exists():
                    path.unlink()
                    files_deleted += 1
            except HTTPException:
                logger.warning("files.purge: invalid export path")
            except Exception:
                logger.warning("files.purge: failed to delete export file")

    removed = 0
    if candidates:
        removed = cdb.delete_file_artifacts_by_ids(list(candidates.keys()))
    return FileArtifactsPurgeResponse(removed=removed, files_deleted=files_deleted)
