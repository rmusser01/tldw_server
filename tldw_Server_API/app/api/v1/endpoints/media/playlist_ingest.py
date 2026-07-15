"""Version-2 asynchronous playlist preflight resource routes."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
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
    PlaylistIngestProcessingOccurrence,
    PlaylistIngestRunCancelRequest,
    PlaylistIngestRunCreateRequest,
    PlaylistIngestRunCreateResponse,
    PlaylistIngestRunItemResponse,
    PlaylistIngestRunItemsPageResponse,
    PlaylistIngestRunRetryRequest,
    PlaylistIngestRunRetryResponse,
    PlaylistIngestRunSummaryResponse,
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
from tldw_Server_API.app.core.DB_Management.playlist_ingest_store import (
    MediaIngestRunEventRecord,
    MediaIngestRunItemRecord,
    MediaIngestRunRecord,
    PlaylistIngestStore,
    PlaylistItemRecord,
)
from tldw_Server_API.app.core.exceptions import (
    InvalidPlaylistUrlError,
    PlaylistIngestConflictError,
    PlaylistIngestNotFoundError,
    PlaylistPreflightBusyError,
    PlaylistPreflightIncompleteError,
    PlaylistPreflightRequiredError,
    PlaylistPreflightUnavailableError,
    PlaylistRunPendingError,
    PlaylistRunStatusUnavailableError,
    PlaylistRunValidationError,
    PlaylistSelectionError,
    ReviewRequiredError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import PlaylistIngestService
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work

router = APIRouter()
_MAX_RUN_EVENT_ID = 2**63 - 1
_PREFLIGHT_DEPENDENCIES = [
    Depends(RequirePermission(MEDIA_CREATE)),
    Depends(rbac_rate_limit("media.create")),
]
_PREFLIGHT_READ_DEPENDENCIES = [
    Depends(RequirePermission(MEDIA_READ)),
    Depends(rbac_rate_limit("media.read")),
]


def _request_body_schema(model: type, *, required: bool = True) -> dict[str, Any]:
    return {
        "requestBody": {
            "required": required,
            "content": {"application/json": {"schema": model.model_json_schema()}},
        }
    }


def _owner(current_user: User) -> str:
    owner = str(getattr(current_user, "id", "") or "").strip()
    if not owner:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication_required")
    return owner


def _playlist_ingest_sse_poll_seconds() -> float:
    raw_value = os.getenv(
        "TLDW_PLAYLIST_INGEST_SSE_POLL_SECONDS",
        os.getenv("PLAYLIST_RUN_SSE_POLL_INTERVAL", "1.0"),
    )
    try:
        value = float(raw_value or "1.0")
    except (OSError, TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return min(max(value, 0.01), 60.0)


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, PlaylistIngestNotFoundError):
        raise HTTPException(status_code=404, detail="preflight_not_found") from exc
    if isinstance(exc, InvalidPlaylistUrlError):
        raise HTTPException(status_code=422, detail="invalid_playlist_url") from exc
    if isinstance(exc, PlaylistPreflightBusyError):
        raise HTTPException(status_code=429, detail="preflight_busy", headers={"Retry-After": "5"}) from exc
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


def _raise_run_http(exc: Exception) -> None:
    if isinstance(exc, PlaylistIngestNotFoundError):
        raise HTTPException(status_code=404, detail="ingest_run_not_found") from exc
    if isinstance(exc, ReviewRequiredError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_required",
                "items": [item.model_dump(mode="json") for item in exc.items],
            },
        ) from exc
    if isinstance(exc, PlaylistPreflightRequiredError):
        raise HTTPException(status_code=422, detail="playlist_preflight_required") from exc
    if isinstance(exc, PlaylistSelectionError):
        raise HTTPException(status_code=422, detail="invalid_occurrence_selection") from exc
    if isinstance(exc, PlaylistRunStatusUnavailableError):
        raise HTTPException(status_code=503, detail="run_status_unavailable") from exc
    if isinstance(exc, PlaylistRunPendingError):
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_action_pending", "run_id": exc.run_id},
        ) from exc
    if isinstance(exc, PlaylistRunValidationError):
        safe_codes = {
            "invalid_run_request",
            "invalid_direct_url",
            "library_lookup_failed",
            "collection_planning_failed",
            "collection_planning_reconciliation_failed",
            "collection_planning_cleanup_failed",
        }
        code = str(exc) if str(exc) in safe_codes else "invalid_run_request"
        raise HTTPException(status_code=422, detail=code) from exc
    if isinstance(exc, PlaylistIngestConflictError):
        raise HTTPException(status_code=409, detail="ingest_run_conflict") from exc
    logger.warning("Playlist ingest run request failed")
    raise HTTPException(status_code=500, detail="playlist_ingest_run_failed") from exc


def _run_item_response(item: MediaIngestRunItemRecord) -> PlaylistIngestRunItemResponse:
    return PlaylistIngestRunItemResponse(
        occurrence_id=item.occurrence_id,
        ordinal=item.ordinal,
        input_kind=item.input_kind,
        source_url=item.source_url,
        normalized_source_id=item.normalized_source_id,
        source_kind=item.source_kind,
        display_metadata=item.display_metadata,
        action=item.action,
        state=item.state,
        outcome=item.outcome,
        progress_percent=item.progress_percent,
        progress_message=item.progress_message,
        job_id=item.job_id,
        batch_id=item.batch_id,
        media_id=item.media_id,
        planned_collection_item_id=item.planned_collection_item_id,
        attempt=item.attempt,
        retryable=item.retryable,
    )


def _processing_occurrence(item: MediaIngestRunItemRecord) -> PlaylistIngestProcessingOccurrence:
    return PlaylistIngestProcessingOccurrence(
        occurrence_id=item.occurrence_id,
        ordinal=item.ordinal,
        input_kind=item.input_kind,
        source_url=item.source_url,
        source_kind=item.source_kind,
        display_metadata=item.display_metadata,
        state=item.state,
        attempt=item.attempt,
        planned_collection_item_id=item.planned_collection_item_id,
    )


def _processing_occurrences(items: list[MediaIngestRunItemRecord]) -> list[PlaylistIngestProcessingOccurrence]:
    return [
        _processing_occurrence(item)
        for item in items
        if item.action in {"ingest", "overwrite"} and item.state in {"staged", "awaiting_upload"}
    ]


def _run_summary(
    run: MediaIngestRunRecord,
    items: list[MediaIngestRunItemRecord],
) -> PlaylistIngestRunSummaryResponse:
    counts = {"total": len(items)}
    for item in items:
        counts[item.state] = counts.get(item.state, 0) + 1
        if item.outcome is not None:
            counts[item.outcome] = counts.get(item.outcome, 0) + 1
    return PlaylistIngestRunSummaryResponse(
        run_id=run.run_id,
        status=run.status,
        counts=counts,
        version=run.version,
        collection_id=run.collection_id,
        batch_ids=list(run.batch_ids or []),
        created_at=run.created_at,
        updated_at=run.updated_at,
        expires_at=run.expires_at,
    )


def _event_payload(event: MediaIngestRunEventRecord) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "occurrence_id": event.occurrence_id,
        "job_id": event.job_id,
        "batch_id": event.batch_id,
        "event_type": event.event_type,
        "state": event.state,
        "outcome": event.outcome,
        "progress_percent": event.progress_percent,
        "progress_message": event.progress_message,
        "occurred_at": event.occurred_at.isoformat(),
    }


@router.post(
    "/playlist-preflights",
    response_model=PlaylistPreflightAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create an asynchronous playlist preflight resource",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
    openapi_extra=_request_body_schema(PlaylistPreflightCreateRequest),
)
def create_playlist_preflight(
    request_scope: Request,
    payload: Any = Body(...),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistPreflightAcceptedResponse:
    """Validate and enqueue one owner-scoped asynchronous playlist preflight."""
    assert_may_start_work(request_scope.app, "media.playlist.preflight.create")
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
    status_url = request_scope.url_for("playlist_preflight_summary", preflight_id=created.preflight_id).path
    items_url = request_scope.url_for("playlist_preflight_items", preflight_id=created.preflight_id).path
    return PlaylistPreflightAcceptedResponse(
        preflight_id=created.preflight_id,
        status_url=status_url,
        items_url=items_url,
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
    name="playlist_preflight_summary",
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
    name="playlist_preflight_items",
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
    openapi_extra=_request_body_schema(PlaylistMaterializationCreateRequest),
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


@router.post(
    "/ingest/runs",
    response_model=PlaylistIngestRunCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an owner-scoped media ingest run",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
    openapi_extra=_request_body_schema(PlaylistIngestRunCreateRequest),
    name="playlist_ingest_run_create",
)
def create_playlist_ingest_run(
    request_scope: Request,
    payload: Any = Body(...),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistIngestRunCreateResponse:
    """Create and reconcile one owner-scoped playlist ingest run."""
    assert_may_start_work(request_scope.app, "media.playlist.ingest.run.create")
    try:
        request = PlaylistIngestRunCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="invalid_run_request") from exc
    service = PlaylistIngestService(job_manager)
    owner = _owner(current_user)
    try:
        run = service.create_run(
            owner,
            client_request_id=request.client_request_id,
            inputs=request.inputs,
            review_overrides=request.review_overrides,
            processing_options=request.processing_options,
            playlist_summaries=request.playlist_summaries,
            new_collection=request.new_collection,
        )
        run = service.reconcile_run_jobs(owner, run.run_id)
        items = list(PlaylistIngestStore(job_manager).list_run_items(owner, run.run_id, limit=500))
    except Exception as exc:  # noqa: BLE001 - map every trust-boundary failure to a safe code
        _raise_run_http(exc)
    status_url = request_scope.url_for("playlist_ingest_run_summary", run_id=run.run_id).path
    return PlaylistIngestRunCreateResponse(
        run_id=run.run_id,
        status=run.status,
        version=run.version,
        status_url=status_url,
        items_url=f"{status_url}/items",
        events_url=f"{status_url}/events/stream",
        processing_occurrences=_processing_occurrences(items),
    )


@router.get(
    "/ingest/runs/{run_id}",
    response_model=PlaylistIngestRunSummaryResponse,
    summary="Get an owner-scoped media ingest run summary",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_READ_DEPENDENCIES,
    name="playlist_ingest_run_summary",
)
def get_playlist_ingest_run(
    run_id: str,
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistIngestRunSummaryResponse:
    owner = _owner(current_user)
    service = PlaylistIngestService(job_manager)
    store = PlaylistIngestStore(job_manager)
    try:
        run = service.reconcile_run_jobs(owner, run_id)
        items = list(store.list_run_items(owner, run.run_id, limit=500))
    except Exception as exc:  # noqa: BLE001 - map every trust-boundary failure to a safe code
        _raise_run_http(exc)
    return _run_summary(run, items)


@router.get(
    "/ingest/runs/{run_id}/items",
    response_model=PlaylistIngestRunItemsPageResponse,
    summary="List one owner-scoped media ingest run item page",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_READ_DEPENDENCIES,
)
def list_playlist_ingest_run_items(
    run_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistIngestRunItemsPageResponse:
    owner = _owner(current_user)
    service = PlaylistIngestService(job_manager)
    store = PlaylistIngestStore(job_manager)
    try:
        run = service.reconcile_run_jobs(owner, run_id)
        page = store.list_run_items(owner, run.run_id, limit=limit, cursor=cursor)
    except Exception as exc:  # noqa: BLE001 - map every trust-boundary failure to a safe code
        _raise_run_http(exc)
    return PlaylistIngestRunItemsPageResponse(
        run_id=run.run_id,
        version=run.version,
        items=[_run_item_response(item) for item in page],
        next_cursor=page.next_cursor,
    )


@router.post(
    "/ingest/runs/{run_id}/cancel",
    response_model=PlaylistIngestRunSummaryResponse,
    summary="Cancel selected occurrences or a whole media ingest run",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
    openapi_extra=_request_body_schema(PlaylistIngestRunCancelRequest, required=False),
)
def cancel_playlist_ingest_run(
    run_id: str,
    payload: Any = Body(default=None),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistIngestRunSummaryResponse:
    try:
        request = PlaylistIngestRunCancelRequest.model_validate({} if payload is None else payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="invalid_run_cancel_request") from exc
    owner = _owner(current_user)
    service = PlaylistIngestService(job_manager)
    store = PlaylistIngestStore(job_manager)
    try:
        run = service.cancel_run(
            owner,
            run_id,
            occurrence_ids=request.occurrence_ids,
            reason=request.reason,
        )
        items = list(store.list_run_items(owner, run.run_id, limit=500))
    except Exception as exc:  # noqa: BLE001 - map every trust-boundary failure to a safe code
        _raise_run_http(exc)
    return _run_summary(run, items)


@router.post(
    "/ingest/runs/{run_id}/retry",
    response_model=PlaylistIngestRunRetryResponse,
    summary="Retry selected failed media ingest occurrences",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_DEPENDENCIES,
    openapi_extra=_request_body_schema(PlaylistIngestRunRetryRequest),
)
def retry_playlist_ingest_run(
    run_id: str,
    request_scope: Request,
    payload: Any = Body(...),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> PlaylistIngestRunRetryResponse:
    """Retry selected failed occurrences without accepting work during drain."""
    assert_may_start_work(request_scope.app, "media.playlist.ingest.run.retry")
    try:
        request = PlaylistIngestRunRetryRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="invalid_run_retry_request") from exc
    owner = _owner(current_user)
    service = PlaylistIngestService(job_manager)
    try:
        processing = list(service.retry_run_items(owner, run_id, request.occurrence_ids))
        run = service.reconcile_run_jobs(owner, run_id)
    except Exception as exc:  # noqa: BLE001 - map every trust-boundary failure to a safe code
        _raise_run_http(exc)
    return PlaylistIngestRunRetryResponse(
        run_id=run.run_id,
        version=run.version,
        processing_occurrences=[_processing_occurrence(item) for item in processing],
    )


@router.get(
    "/ingest/runs/{run_id}/events/stream",
    summary="Stream owner-scoped media ingest run events",
    tags=["Media Playlist Ingest v2"],
    dependencies=_PREFLIGHT_READ_DEPENDENCIES,
    response_class=StreamingResponse,
    responses={status.HTTP_200_OK: {"content": {"text/event-stream": {}}}},
)
async def stream_playlist_ingest_run_events(
    request_scope: Request,
    run_id: str,
    after_id: int = Query(default=0, ge=0, le=_MAX_RUN_EVENT_ID),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(get_job_manager),
) -> StreamingResponse:
    raw_last_event_id = request_scope.headers.get("Last-Event-ID")
    if raw_last_event_id is not None:
        try:
            after_id = int(raw_last_event_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid_last_event_id") from exc
        if not 0 <= after_id <= _MAX_RUN_EVENT_ID:
            raise HTTPException(status_code=400, detail="invalid_last_event_id")

    owner = _owner(current_user)
    service = PlaylistIngestService(job_manager)
    store = PlaylistIngestStore(job_manager)
    try:
        run = service.reconcile_run_jobs(owner, run_id)
        items = list(store.list_run_items(owner, run.run_id, limit=500))
    except Exception as exc:  # noqa: BLE001 - map every trust-boundary failure to a safe code
        _raise_run_http(exc)

    poll_interval = _playlist_ingest_sse_poll_seconds()
    max_duration = None
    if is_test_mode():
        try:
            max_duration = float(os.getenv("PLAYLIST_RUN_SSE_TEST_MAX_SECONDS", "1.0") or "1.0")
        except (OSError, TypeError, ValueError):
            max_duration = 1.0
    stream = SSEStream(
        heartbeat_interval_s=poll_interval,
        heartbeat_mode="data",
        max_duration_s=max_duration,
        labels={"component": "playlist_ingest", "endpoint": "run_events_sse"},
    )

    async def _producer() -> None:
        cursor = int(after_id)
        await stream.send_event("snapshot", _run_summary(run, items).model_dump(mode="json"))
        while True:
            try:
                service.reconcile_run_jobs(owner, run_id)
                minimum, maximum = store.run_event_bounds(owner, run_id)
                outside_retained_bounds = cursor > 0 and (
                    minimum is None or maximum is None or cursor < minimum or cursor > maximum
                )
                if outside_retained_bounds:
                    await stream.send_event(
                        "resync_required",
                        {
                            "run_id": run_id,
                            "min_event_id": minimum,
                            "latest_event_id": maximum,
                        },
                        event_id=str(maximum) if maximum is not None else None,
                    )
                    cursor = maximum or 0
                else:
                    events = store.list_run_events(owner, run_id, after_event_id=cursor, limit=500)
                    for event in events:
                        await stream.send_event(
                            "occurrence" if event.occurrence_id is not None else "run",
                            _event_payload(event),
                            event_id=str(event.event_id),
                        )
                        cursor = event.event_id
            except PlaylistIngestNotFoundError:
                await stream.send_event("resync_required", {"run_id": run_id})
                await stream.done()
                return
            except Exception:  # noqa: BLE001 - keep transport errors safe and recoverable
                await stream.send_event(
                    "status_unavailable",
                    {"run_id": run_id, "code": "run_status_unavailable"},
                )
            await asyncio.sleep(poll_interval)

    async def _generate():
        producer = asyncio.create_task(_producer())
        try:
            async for line in stream.iter_sse():
                yield line
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    await producer

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["router"]
