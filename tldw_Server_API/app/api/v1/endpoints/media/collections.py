from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
    get_collections_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.media_collections import (
    MediaCollectionCreateRequest,
    MediaCollectionItemCreateRequest,
    MediaCollectionItemResponse,
    MediaCollectionItemUpdateRequest,
    MediaCollectionListResponse,
    MediaCollectionResponse,
    MediaCollectionUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    CollectionsDatabase,
    MediaCollectionItemRow,
    MediaCollectionRow,
)


router = APIRouter()


def _item_response(row: MediaCollectionItemRow) -> MediaCollectionItemResponse:
    return MediaCollectionItemResponse(
        id=row.id,
        collection_id=row.collection_id,
        ordinal=row.ordinal,
        source_url=row.source_url,
        normalized_source_id=row.normalized_source_id,
        source_kind=row.source_kind,
        title=row.title,
        speaker=row.speaker,
        published_at=row.published_at,
        track=row.track,
        duplicate_status=row.duplicate_status,
        status=row.status,
        media_id=row.media_id,
        content_item_id=row.content_item_id,
        latest_job_id=row.latest_job_id,
        latest_run_id=row.latest_run_id,
        idempotency_key=row.idempotency_key,
        retry_count=row.retry_count,
        error_summary=row.error_summary,
        warnings=row.warnings,
        metadata=row.metadata,
        tags=row.tags,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _collection_response(row: MediaCollectionRow) -> MediaCollectionResponse:
    return MediaCollectionResponse(
        id=row.id,
        name=row.name,
        kind=row.kind,
        description=row.description,
        source_url=row.source_url,
        metadata=row.metadata,
        default_tags=row.default_tags,
        created_at=row.created_at,
        updated_at=row.updated_at,
        items=[_item_response(item) for item in row.items],
    )


def _raise_collection_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    raise exc


@router.post(
    "/collections",
    response_model=MediaCollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a durable media collection",
    tags=["Media Collections"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
    ],
)
async def create_media_collection(
    payload: MediaCollectionCreateRequest,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionResponse:
    try:
        row = db.create_media_collection(
            name=payload.name,
            kind=payload.kind,
            description=payload.description,
            source_url=payload.source_url,
            metadata=payload.metadata,
            default_tags=payload.default_tags,
        )
        return _collection_response(row)
    except Exception as exc:
        _raise_collection_error(exc)
        raise


@router.get(
    "/collections",
    response_model=MediaCollectionListResponse,
    summary="List durable media collections",
    tags=["Media Collections"],
)
async def list_media_collections(
    kind: str | None = None,
    page: int = 1,
    size: int = 20,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionListResponse:
    rows, total = db.list_media_collections(kind=kind, page=page, size=size)
    return MediaCollectionListResponse(
        items=[_collection_response(row) for row in rows],
        total=total,
        page=max(1, int(page or 1)),
        size=min(100, max(1, int(size or 20))),
    )


@router.get(
    "/collections/{collection_id}",
    response_model=MediaCollectionResponse,
    summary="Get a durable media collection",
    tags=["Media Collections"],
)
async def get_media_collection(
    collection_id: int,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionResponse:
    try:
        return _collection_response(db.get_media_collection(collection_id))
    except Exception as exc:
        _raise_collection_error(exc)
        raise


@router.patch(
    "/collections/{collection_id}",
    response_model=MediaCollectionResponse,
    summary="Update a durable media collection",
    tags=["Media Collections"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
    ],
)
async def update_media_collection(
    collection_id: int,
    payload: MediaCollectionUpdateRequest,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionResponse:
    try:
        row = db.update_media_collection(
            collection_id,
            **payload.model_dump(exclude_unset=True),
        )
        return _collection_response(row)
    except Exception as exc:
        _raise_collection_error(exc)
        raise


@router.post(
    "/collections/{collection_id}/items",
    response_model=MediaCollectionItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an item to a durable media collection",
    tags=["Media Collections"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
    ],
)
async def add_media_collection_item(
    collection_id: int,
    payload: MediaCollectionItemCreateRequest,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionItemResponse:
    try:
        row = db.add_media_collection_item(
            collection_id=collection_id,
            source_url=payload.source_url,
            normalized_source_id=payload.normalized_source_id,
            source_kind=payload.source_kind,
            status=payload.status,
            ordinal=payload.ordinal,
            title=payload.title,
            speaker=payload.speaker,
            published_at=payload.published_at,
            track=payload.track,
            duplicate_status=payload.duplicate_status,
            media_id=payload.media_id,
            content_item_id=payload.content_item_id,
            latest_job_id=payload.latest_job_id,
            latest_run_id=payload.latest_run_id,
            idempotency_key=payload.idempotency_key,
            retry_count=payload.retry_count,
            error_summary=payload.error_summary,
            warnings=payload.warnings,
            metadata=payload.metadata,
            tags=payload.tags,
        )
        return _item_response(row)
    except Exception as exc:
        _raise_collection_error(exc)
        raise


@router.patch(
    "/collections/{collection_id}/items/{item_id}",
    response_model=MediaCollectionItemResponse,
    summary="Update a durable media collection item",
    tags=["Media Collections"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
    ],
)
async def update_media_collection_item(
    collection_id: int,
    item_id: int,
    payload: MediaCollectionItemUpdateRequest,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionItemResponse:
    try:
        current = db.get_media_collection_item(item_id)
        if current.collection_id != collection_id:
            raise KeyError("media_collection_item_not_found")
        row = db.update_media_collection_item(
            item_id,
            **payload.model_dump(exclude_unset=True),
        )
        return _item_response(row)
    except Exception as exc:
        _raise_collection_error(exc)
        raise
