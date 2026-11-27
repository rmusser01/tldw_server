from __future__ import annotations

from dataclasses import replace
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
    get_collections_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
from tldw_Server_API.app.api.v1.schemas.media_collections import (
    MediaCollectionCreateRequest,
    MediaCollectionItemCreateRequest,
    MediaCollectionItemResponse,
    MediaCollectionItemUpdateRequest,
    MediaCollectionListResponse,
    MediaCollectionResponse,
    MediaCollectionUpdateRequest,
    PlanKey,
    QuickIngestClaimResponse,
    QuickIngestPlanItemResponse,
    QuickIngestPlanRequest,
    QuickIngestPlanResponse,
    QuickIngestRetryClaim,
    QuickIngestSourceStatusResponse,
    positive_quick_ingest_media_id,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_READ
from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    CollectionsDatabase,
    MediaCollectionItemRow,
    MediaCollectionRow,
    QuickIngestPlanConflict,
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
    if isinstance(exc, QuickIngestPlanConflict):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    raise exc


def _recover_plan_job(
    db: CollectionsDatabase,
    plan_key: str,
    source_key: str,
    row: MediaCollectionItemRow,
    *,
    persist: bool = True,
) -> MediaCollectionItemRow:
    """Reconcile a Jobs commit even when its submitter lost the response before projecting its ID."""
    if row.dispatch_state != "jobs":
        return row
    key = row.idempotency_key
    is_file = row.source_url.startswith("local://")
    job_key = f"media-upload:{db.user_id}:{key}" if is_file else ingest_jobs._collection_url_job_key(db.user_id, key)
    # Direct saves can plan and resume independently of Jobs availability.
    with ingest_jobs.JobManager.scoped_rls_context(
        is_admin=False, domain_allowlist="media_ingest", owner_user_id=db.user_id
    ):
        job = ingest_jobs._find_collection_retry_job(ingest_jobs.get_job_manager(), db.user_id, job_key)
    if job is None:
        return row
    payload = ingest_jobs._normalize_payload(job.get("payload"))
    if (
        str(job.get("owner_user_id")) != db.user_id
        or str(payload.get("collection_id")) != str(row.collection_id)
        or str(payload.get("planned_item_id")) != str(row.id)
        or payload.get("idempotency_key") != key
        or payload.get("source_kind") != ("file" if is_file else "url")
        or (
            payload.get("original_filename") != unquote(row.source_url[8:])
            if is_file
            else payload.get("source") != row.source_url
        )
    ):
        raise QuickIngestPlanConflict("Stored job identity does not match this plan source.")
    status, media_id = _plan_job_result(job)
    job_id = str(job["id"])
    if row.latest_job_id is not None and row.latest_job_id != job_id:
        raise QuickIngestPlanConflict("The source has a different accepted job.")
    if row.latest_job_id == job_id and status == "processing":
        return row
    if not persist:
        current = db.get_quick_ingest_source(plan_key, source_key)
        if current.retry_count != row.retry_count or current.dispatch_state != "jobs":
            return current
        return replace(
            current,
            latest_job_id=job_id,
            status=status,
            media_id=media_id,
            dispatch_state="jobs" if status == "processing" else "settled",
        )
    try:
        return db.finish_quick_ingest_dispatch(
            plan_key,
            source_key,
            retry_count=row.retry_count,
            idempotency_key=key,
            transport="jobs",
            status=status,
            media_id=media_id,
            latest_job_id=job_id,
        )
    except QuickIngestPlanConflict:
        current = db.get_quick_ingest_source(plan_key, source_key)
        if current.retry_count == row.retry_count and current.latest_job_id not in (None, job_id):
            raise
        return current


def _plan_job_result(job: dict) -> tuple[str, int | None]:
    """Only an authoritative terminal Jobs result permits a retry or confirms a media ID."""
    status = job["status"]
    if status in {"failed", "quarantined", "cancelled"}:
        return ("cancelled" if status == "cancelled" else "failed"), None
    if status != "completed":
        return "processing", None
    from tldw_Server_API.app.services.media_ingest_jobs_worker import _is_failed_result, _is_skipped_existing_result

    result = ingest_jobs._normalize_payload(job.get("result"))
    if _is_failed_result(result):
        return "failed", None
    media_id = positive_quick_ingest_media_id(result.get("media_id") or result.get("db_id"))
    if _is_skipped_existing_result(result):
        return "skipped_existing", media_id
    return ("completed", media_id) if media_id is not None else ("processing", None)


@router.put(
    "/collections/quick-ingest-plans/{plan_key}",
    response_model=QuickIngestPlanResponse,
    tags=["Media Collections"],
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("media.create"))],
)
async def put_quick_ingest_plan(
    plan_key: PlanKey,
    payload: QuickIngestPlanRequest,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> QuickIngestPlanResponse:
    """Bind selected source identities atomically and return their current user-owned rows."""
    try:
        collection_id, items = db.put_quick_ingest_plan(
            plan_key,
            payload.collection.model_dump(),
            [(s.source_key, s.item.model_dump()) for s in payload.items],
        )
        items = {key: _recover_plan_job(db, plan_key, key, row) for key, row in items.items()}
        return QuickIngestPlanResponse(
            collection_id=collection_id,
            items=[
                QuickIngestPlanItemResponse(source_key=key, item=_item_response(row), dispatch_state=row.dispatch_state)
                for key, row in items.items()
            ],
        )
    except (KeyError, ValueError) as exc:
        _raise_collection_error(exc)
        raise


@router.get(
    "/collections/quick-ingest-plans/{plan_key}/sources/{source_key}",
    response_model=QuickIngestSourceStatusResponse,
    tags=["Media Collections"],
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("media.read"))],
)
async def get_quick_ingest_source_status(
    plan_key: PlanKey,
    source_key: PlanKey,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> QuickIngestSourceStatusResponse:
    """Inspect the original receipt without creating a plan or reserving a source."""
    try:
        row = _recover_plan_job(
            db, plan_key, source_key, db.get_quick_ingest_source(plan_key, source_key), persist=False
        )
        return QuickIngestSourceStatusResponse(
            collection_id=row.collection_id,
            source_key=source_key,
            item=_item_response(row),
            dispatch_state=row.dispatch_state,
        )
    except (KeyError, ValueError) as exc:
        _raise_collection_error(exc)
        raise


@router.post(
    "/collections/quick-ingest-plans/{plan_key}/sources/{source_key}/claim",
    response_model=QuickIngestClaimResponse,
    tags=["Media Collections"],
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("media.create"))],
)
async def claim_quick_ingest_source(
    plan_key: PlanKey,
    source_key: PlanKey,
    retry: QuickIngestRetryClaim | None = None,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> QuickIngestClaimResponse:
    """Claim one source attempt, preserving the previously admitted job or direct result on replay."""
    try:
        _recover_plan_job(db, plan_key, source_key, db.get_quick_ingest_source(plan_key, source_key))
        claimed, row = db.claim_quick_ingest_source(plan_key, source_key, **(retry.model_dump() if retry else {}))
        return QuickIngestClaimResponse(claimed=claimed, item=_item_response(row), dispatch_state=row.dispatch_state)
    except (KeyError, ValueError) as exc:
        _raise_collection_error(exc)
        raise


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
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("media.read")),
    ],
)
async def list_media_collections(
    kind: str | None = None,
    page: int = 1,
    size: int = 20,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> MediaCollectionListResponse:
    page_clean = max(1, int(page or 1))
    size_clean = min(100, max(1, int(size or 20)))
    rows, total = db.list_media_collections(kind=kind, page=page_clean, size=size_clean)
    return MediaCollectionListResponse(
        items=[_collection_response(row) for row in rows],
        total=total,
        page=page_clean,
        size=size_clean,
    )


@router.get(
    "/collections/{collection_id}",
    response_model=MediaCollectionResponse,
    summary="Get a durable media collection",
    tags=["Media Collections"],
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("media.read")),
    ],
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
        projected = db.apply_quick_ingest_item_patch(item_id, payload.model_dump(exclude_unset=True))
        if projected is not None:
            return _item_response(projected)
        if "source_url" in payload.model_fields_set:
            if (
                payload.model_fields_set != {"source_url", "retry_count", "idempotency_key"}
                or not (payload.source_url and payload.source_url.strip())
                or not (payload.retry_count and payload.idempotency_key and payload.idempotency_key.strip())
            ):
                raise ValueError("A source correction requires its URL, retry count and idempotency key.")
            repaired = db.repair_media_collection_item_source(
                current, payload.source_url.strip(), payload.retry_count, payload.idempotency_key.strip(),
            )
            if repaired is None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The collection source or its attempt has changed.")
            return _item_response(repaired)
        row = db.update_media_collection_item(
            item_id,
            **payload.model_dump(exclude_unset=True),
        )
        return _item_response(row)
    except Exception as exc:
        _raise_collection_error(exc)
        raise
