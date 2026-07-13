"""Version-2 asynchronous playlist preflight resource routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import (
    PlaylistMaterializationCreateRequest,
    PlaylistMaterializationItemResponse,
    PlaylistMaterializationResponse,
    PlaylistPreflightAcceptedResponse,
    PlaylistPreflightCreateRequest,
    PlaylistPreflightItemResponse,
    PlaylistPreflightItemsPageResponse,
    PlaylistPreflightLimits,
    PlaylistPreflightSummaryResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_READ
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
    InvalidPlaylistUrlError,
    PlaylistIngestService,
    PlaylistPreflightBusyError,
    PlaylistPreflightIncompleteError,
    PlaylistPreflightUnavailableError,
    PlaylistSelectionError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
    PlaylistIngestNotFoundError,
    PlaylistItemRecord,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager

router = APIRouter()
_RESOURCE_BASE = "/api/v1/media/playlist-preflights"
_PREFLIGHT_DEPENDENCIES = [
    Depends(RequirePermission(MEDIA_CREATE)),
    Depends(rbac_rate_limit("media.create")),
]
_PREFLIGHT_READ_DEPENDENCIES = [
    Depends(RequirePermission(MEDIA_READ)),
    Depends(rbac_rate_limit("media.read")),
]


def _owner(current_user: User) -> str:
    owner = str(getattr(current_user, "id", "") or "").strip()
    if not owner:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication_required")
    return owner


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, PlaylistIngestNotFoundError):
        raise HTTPException(status_code=404, detail="preflight_not_found") from exc
    if isinstance(exc, InvalidPlaylistUrlError):
        raise HTTPException(status_code=422, detail="invalid_playlist_url") from exc
    if isinstance(exc, PlaylistPreflightBusyError):
        raise HTTPException(status_code=429, detail="preflight_busy") from exc
    if isinstance(exc, PlaylistPreflightIncompleteError):
        raise HTTPException(status_code=409, detail="preflight_incomplete") from exc
    if isinstance(exc, PlaylistSelectionError):
        raise HTTPException(status_code=422, detail="invalid_occurrence_selection") from exc
    if isinstance(exc, PlaylistPreflightUnavailableError):
        raise HTTPException(status_code=503, detail="preflight_unavailable") from exc
    logger.warning("Playlist preflight resource request failed")
    raise HTTPException(status_code=500, detail="playlist_preflight_failed") from exc


def _item_response(item: PlaylistItemRecord) -> PlaylistPreflightItemResponse:
    return PlaylistPreflightItemResponse(
        occurrence_id=item.occurrence_id,
        ordinal=item.ordinal,
        occurrence_index_for_source=item.occurrence_index_for_source,
        source_url=item.source_url,
        normalized_source_id=item.normalized_source_id,
        source_kind=item.source_kind,
        availability=item.availability,
        duplicate_status=item.duplicate_status,
        duplicate_of_occurrence_id=item.duplicate_of_occurrence_id,
        selected_by_default=item.selected_by_default,
        display_metadata=item.display_metadata,
    )


@router.post(
    "/playlist-preflights",
    response_model=PlaylistPreflightAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create an asynchronous playlist preflight resource",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
)
def create_playlist_preflight(
    payload: Any = Body(...),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistPreflightAcceptedResponse:
    try:
        request = PlaylistPreflightCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="invalid_playlist_url") from exc
    try:
        created = PlaylistIngestService(job_manager).create_preflight(
            _owner(current_user),
            url=request.url,
            max_items=request.max_items,
            timeout_seconds=request.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - trust boundary maps every failure to a safe code
        _raise_http(exc)
    status_url = f"{_RESOURCE_BASE}/{created.preflight_id}"
    return PlaylistPreflightAcceptedResponse(
        preflight_id=created.preflight_id,
        status_url=status_url,
        items_url=f"{status_url}/items",
        expires_at=created.record.expires_at,
        limits=PlaylistPreflightLimits(
            max_items=created.max_items,
            global_capacity=created.global_capacity,
            owner_capacity=created.owner_capacity,
        ),
    )


@router.get(
    "/playlist-preflights/{preflight_id}",
    response_model=PlaylistPreflightSummaryResponse,
    summary="Get an asynchronous playlist preflight summary",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_READ_DEPENDENCIES,
)
def get_playlist_preflight(
    preflight_id: str,
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistPreflightSummaryResponse:
    service = PlaylistIngestService(job_manager)
    try:
        record = service.get_preflight(_owner(current_user), preflight_id)
    except Exception as exc:  # noqa: BLE001 - trust boundary maps every failure to a safe code
        _raise_http(exc)
    return PlaylistPreflightSummaryResponse(
        preflight_id=record.preflight_id,
        status=record.status,
        source_url=record.source_url,
        source_kind=record.source_kind,
        playlist_id=record.playlist_id,
        summary=record.summary,
        error=service.public_error(record.error),
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
    )


@router.get(
    "/playlist-preflights/{preflight_id}/items",
    response_model=PlaylistPreflightItemsPageResponse,
    summary="List an immutable playlist preflight item page",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_READ_DEPENDENCIES,
)
def list_playlist_preflight_items(
    preflight_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistPreflightItemsPageResponse:
    try:
        page = PlaylistIngestService(job_manager).list_preflight_items(
            _owner(current_user),
            preflight_id,
            limit=limit,
            cursor=cursor,
        )
    except Exception as exc:  # noqa: BLE001 - trust boundary maps every failure to a safe code
        _raise_http(exc)
    return PlaylistPreflightItemsPageResponse(
        preflight_id=preflight_id,
        items=[_item_response(item) for item in page],
        next_cursor=page.next_cursor,
    )


@router.post(
    "/playlist-preflights/{preflight_id}/materializations",
    response_model=PlaylistMaterializationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Materialize selected playlist occurrences",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
)
def create_playlist_materialization(
    preflight_id: str,
    payload: Any = Body(...),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistMaterializationResponse:
    try:
        request = PlaylistMaterializationCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="invalid_materialization_request") from exc
    try:
        created = PlaylistIngestService(job_manager).create_materialization(
            _owner(current_user),
            preflight_id,
            request.occurrence_ids,
        )
    except Exception as exc:  # noqa: BLE001 - trust boundary maps every failure to a safe code
        _raise_http(exc)
    items = [
        PlaylistMaterializationItemResponse(
            occurrence_id=item.occurrence_id,
            ordinal=item.ordinal,
            source_url=str(item.source_url),
            normalized_source_id=item.normalized_source_id,
            source_kind=item.source_kind,
            display_metadata=item.display_metadata,
        )
        for item in created.items
    ]
    return PlaylistMaterializationResponse(
        materialization_id=created.record.materialization_id,
        preflight_id=created.record.preflight_id,
        items=items,
        expires_at=created.record.expires_at,
    )


@router.delete(
    "/playlist-preflights/{preflight_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel and expire a playlist preflight resource",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
)
def delete_playlist_preflight(
    preflight_id: str,
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> Response:
    try:
        PlaylistIngestService(job_manager).cancel_preflight(_owner(current_user), preflight_id)
    except Exception as exc:  # noqa: BLE001 - trust boundary maps every failure to a safe code
        _raise_http(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
