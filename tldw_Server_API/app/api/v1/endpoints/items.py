from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.items_schemas import (
    Item,
    ItemsBulkRequest,
    ItemsBulkResponse,
    ItemsBulkResult,
    ItemsListResponse,
)
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.utils.pagination import build_page_pagination_meta
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Collections_DB import ContentItemRow
from tldw_Server_API.app.core.DB_Management.media_db.api import search_media
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    InputError,
)

router = APIRouter(prefix="/items", tags=["items"])

_ITEMS_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_ITEMS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    DatabaseError,
    ImportError,
    InputError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _items_pagination(*, total: int, page: int, size: int) -> PagePaginationMeta:
    total_pages = (total + size - 1) // size if total > 0 else 0
    return build_page_pagination_meta(
        page=page,
        per_page=size,
        total=total,
        total_pages=total_pages,
    )


def _domain_from_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse

        return urlparse(url).hostname or ""
    except _ITEMS_COERCE_EXCEPTIONS:
        return ""


def _resolve_read_at_for_status(collections_db, item_id: int, status: str) -> tuple[str | None, bool]:
    current = collections_db.get_content_item(item_id)
    status_lower = status.lower()
    if status_lower == "read" and not current.read_at:
        return datetime.now(timezone.utc).isoformat(), False
    if status_lower != "read" and current.read_at:
        return None, True
    return None, False


@router.get("", response_model=ItemsListResponse, summary="Unified items list across origins")
async def list_items(
    ids: list[int] | None = Query(None, description="Filter by item IDs"),
    q: str | None = Query(None, description="Search query (title/content)"),
    tags: list[str] | None = Query(None, description="Require these tags (names)"),
    domain: str | None = Query(None, description="Filter by domain (hostname)"),
    date_from: str | None = Query(None, description="ISO start date inclusive"),
    date_to: str | None = Query(None, description="ISO end date inclusive"),
    status_filter: list[str] | None = Query(None, description="Filter by status (e.g., saved, read)"),
    favorite: bool | None = Query(None, description="Filter by favorite flag"),
    origin: str | None = Query(None, description="Filter by origin (watchlist, reading, etc.)"),
    job_id: int | None = Query(None, description="Filter by job_id"),
    run_id: int | None = Query(None, description="Filter by run_id"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    current_user: User = Depends(get_request_user),
    db=Depends(get_media_db_for_user),
    collections_db=Depends(get_collections_db_for_user),
):
    must_have_keywords = tags or []
    # Prepare date range
    dr = None
    date_from_iso = None
    date_to_iso = None
    try:
        start_dt = datetime.fromisoformat(date_from) if date_from else None
        end_dt = datetime.fromisoformat(date_to) if date_to else None
        if start_dt or end_dt:
            dr = {"start_date": start_dt, "end_date": end_dt}
        if start_dt:
            date_from_iso = start_dt.isoformat()
        if end_dt:
            date_to_iso = end_dt.isoformat()
    except _ITEMS_COERCE_EXCEPTIONS:
        raise HTTPException(status_code=422, detail="invalid_date_range") from None

    # Query collections layer first
    try:
        coll_rows, coll_total = collections_db.list_content_items(
            ids=ids,
            q=q,
            tags=tags,
            domain=domain,
            date_from=date_from_iso,
            date_to=date_to_iso,
            status=status_filter,
            favorite=favorite,
            job_id=job_id,
            run_id=run_id,
            origin=origin,
            page=page,
            size=size,
        )
    except _ITEMS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("collections items query failed")
        raise HTTPException(status_code=500, detail="items_query_failed") from e

    if coll_total > 0:
        return ItemsListResponse(
            items=[_content_item_to_schema(r) for r in coll_rows],
            total=coll_total,
            page=page,
            size=size,
            pagination=_items_pagination(total=coll_total, page=page, size=size),
        )

    if origin or (status_filter and len(status_filter) > 0) or favorite is not None:
        return ItemsListResponse(
            items=[],
            total=0,
            page=page,
            size=size,
            pagination=_items_pagination(total=0, page=page, size=size),
        )

    # Legacy fallback to Media DB when collections layer has no data
    try:
        rows, total = search_media(
            db,
            search_query=q,
            search_fields=["title", "content"],
            media_types=None,
            date_range=dr,
            must_have_keywords=must_have_keywords,
            must_not_have_keywords=None,
            sort_by="last_modified_desc",
            media_ids_filter=ids,
            page=page,
            results_per_page=size,
            include_trash=False,
            include_deleted=False,
        )
    except (InputError, DatabaseError) as e:
        raise map_db_error_to_http(e, default_detail="items_query_failed") from e
    except _ITEMS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("items list failed")
        raise HTTPException(status_code=500, detail="items_query_failed") from e

    # Build items and apply optional domain filter in-process
    items: list[Item] = []
    for r in rows:
        item = _media_row_to_item(r, db=db, domain_filter=domain)
        if item is not None:
            items.append(item)

    return ItemsListResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pagination=_items_pagination(total=total, page=page, size=size),
    )


@router.get("/{item_id}", response_model=Item, summary="Get unified item by ID")
async def get_item(
    item_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db=Depends(get_media_db_for_user),
    collections_db=Depends(get_collections_db_for_user),
):
    try:
        row = collections_db.get_content_item(item_id)
        return _content_item_to_schema(row)
    except KeyError:
        pass
    except _ITEMS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("collections item fetch failed")
        raise HTTPException(status_code=500, detail="item_fetch_failed") from e

    try:
        row = collections_db.get_content_item_by_media_id(item_id)
        return _content_item_to_schema(row)
    except KeyError:
        pass
    except _ITEMS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("collections item fetch by media_id failed")
        raise HTTPException(status_code=500, detail="item_fetch_failed") from e

    try:
        rows, _total = search_media(
            db,
            search_query=None,
            search_fields=["title", "content"],
            media_types=None,
            date_range=None,
            must_have_keywords=None,
            must_not_have_keywords=None,
            sort_by="last_modified_desc",
            media_ids_filter=[item_id],
            page=1,
            results_per_page=1,
            include_trash=False,
            include_deleted=False,
        )
    except (InputError, DatabaseError) as e:
        raise map_db_error_to_http(e, default_detail="item_fetch_failed") from e
    except _ITEMS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("media item fetch failed")
        raise HTTPException(status_code=500, detail="item_fetch_failed") from e

    if not rows:
        raise HTTPException(status_code=404, detail="item_not_found")
    item = _media_row_to_item(rows[0], db=db, domain_filter=None)
    if item is None:
        raise HTTPException(status_code=404, detail="item_not_found")
    return item


def _content_item_to_schema(row: ContentItemRow) -> Item:
    domain = row.domain or _domain_from_url(row.url)
    summary = row.summary
    if not summary:
        metadata = {}
        if row.metadata_json:
            try:
                import json as _json

                metadata = _json.loads(row.metadata_json)
            except _ITEMS_COERCE_EXCEPTIONS:
                metadata = {}
        summary = metadata.get("summary")
    item_id = row.media_id if row.media_id is not None else row.id
    return Item(
        id=int(item_id),
        content_item_id=int(row.id),
        media_id=(int(row.media_id) if row.media_id is not None else None),
        title=row.title or "Untitled",
        url=row.url,
        domain=domain or "",
        summary=summary,
        published_at=row.published_at,
        status=row.status,
        favorite=bool(row.favorite),
        tags=row.tags,
        type=row.origin,
    )


def _media_row_to_item(row, *, db, domain_filter: str | None) -> Item | None:
    url = row.get("url")
    dom = _domain_from_url(url)
    if domain_filter and dom and dom != domain_filter:
        return None
    # Fetch tags per item
    try:
        from tldw_Server_API.app.core.DB_Management.media_db.api import (
            fetch_keywords_for_media,
            get_document_version,
        )

        tag_list = fetch_keywords_for_media(db=db, media_id=int(row.get("id")))
    except _ITEMS_NONCRITICAL_EXCEPTIONS:
        tag_list = []

    # Derive summary/published_at best-effort
    summary = None
    published_at = None
    try:
        latest = get_document_version(db, media_id=int(row.get("id")), version_number=None, include_content=False)  # type: ignore[name-defined]
        if isinstance(latest, dict):
            summary = latest.get("analysis_content") or None
            sm = latest.get("safe_metadata")
            if isinstance(sm, str):
                import json as _json

                try:
                    sm = _json.loads(sm)
                except _ITEMS_COERCE_EXCEPTIONS:
                    sm = None
            if isinstance(sm, dict):
                published_at = sm.get("published_at") or sm.get("date")
    except _ITEMS_NONCRITICAL_EXCEPTIONS:
        pass
    if not summary:
        content = row.get("content") or ""
        if isinstance(content, str) and content:
            summary = (content[:500] + "...") if len(content) > 500 else content

    return Item(
        id=int(row.get("id")),
        content_item_id=None,
        media_id=int(row.get("id")),
        title=row.get("title") or "Untitled",
        url=url,
        domain=dom,
        summary=summary or None,
        published_at=published_at,
        status="saved",
        favorite=False,
        tags=tag_list,
        type=row.get("type") or None,
    )


@router.post("/bulk", response_model=ItemsBulkResponse, summary="Bulk update content items")
async def bulk_update_items(
    payload: ItemsBulkRequest,
    current_user: User = Depends(get_request_user),
    collections_db=Depends(get_collections_db_for_user),
) -> ItemsBulkResponse:
    item_ids = [int(item_id) for item_id in payload.item_ids or []]
    if not item_ids:
        raise HTTPException(status_code=422, detail="item_ids_required")

    action = payload.action
    if action == "set_status" and not payload.status:
        raise HTTPException(status_code=422, detail="status_required")
    if action == "set_favorite" and payload.favorite is None:
        raise HTTPException(status_code=422, detail="favorite_required")
    if action in {"add_tags", "remove_tags", "replace_tags"} and not payload.tags:
        raise HTTPException(status_code=422, detail="tags_required")

    seen: set[int] = set()
    unique_ids: list[int] = []
    for item_id in item_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        unique_ids.append(item_id)

    results: list[ItemsBulkResult] = []
    succeeded = 0
    failed = 0

    for item_id in unique_ids:
        try:
            if action == "set_status":
                read_at, clear_read_at = _resolve_read_at_for_status(collections_db, item_id, payload.status)
                collections_db.update_content_item(
                    item_id,
                    status=payload.status,
                    read_at=read_at,
                    clear_read_at=clear_read_at,
                )
            elif action == "set_favorite":
                collections_db.update_content_item(item_id, favorite=payload.favorite)
            elif action in {"add_tags", "remove_tags", "replace_tags"}:
                current = collections_db.get_content_item(item_id)
                incoming = payload.tags or []
                incoming_norm = [str(tag).strip().lower() for tag in incoming if str(tag).strip()]
                if action == "replace_tags":
                    next_tags = incoming_norm
                elif action == "add_tags":
                    next_tags = sorted(set((current.tags or []) + incoming_norm))
                else:
                    remove_set = set(incoming_norm)
                    next_tags = [tag for tag in (current.tags or []) if tag not in remove_set]
                collections_db.update_content_item(item_id, tags=next_tags)
            elif action == "delete":
                if payload.hard:
                    collections_db.delete_content_item(item_id)
                else:
                    read_at, clear_read_at = _resolve_read_at_for_status(collections_db, item_id, "archived")
                    collections_db.update_content_item(
                        item_id,
                        status="archived",
                        read_at=read_at,
                        clear_read_at=clear_read_at,
                    )
            else:
                raise HTTPException(status_code=422, detail="unsupported_action")
            results.append(ItemsBulkResult(item_id=item_id, success=True))
            succeeded += 1
        except KeyError:
            results.append(ItemsBulkResult(item_id=item_id, success=False, error="item_not_found"))
            failed += 1
        except HTTPException as exc:
            results.append(ItemsBulkResult(item_id=item_id, success=False, error=str(exc.detail)))
            failed += 1
        except _ITEMS_NONCRITICAL_EXCEPTIONS:
            logger.error("bulk_update_items failed")
            results.append(ItemsBulkResult(item_id=item_id, success=False, error="update_failed"))
            failed += 1

    return ItemsBulkResponse(
        total=len(unique_ids),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
