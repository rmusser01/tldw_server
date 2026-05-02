# tldw_Server_API/app/api/v1/endpoints/reading_highlights.py
"""
Reading Highlights API (create/list/update/delete)

Stub endpoints to anchor the unified Content Collections PRD.

Endpoints mirror PRD shape:
- POST /reading/items/{item_id}/highlight
- GET  /reading/items/{item_id}/highlights
- PATCH/DELETE by highlight id

DB access must be implemented via app.core.DB_Management (no raw SQL here).
"""

import json
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Path
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.schemas.reading_highlights_schemas import (
    Highlight,
    HighlightCreateRequest,
    HighlightUpdateRequest,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Collections.utils import (
    build_highlight_context,
    find_highlight_span,
    hash_text_sha256,
)
from tldw_Server_API.app.core.Personalization.companion_activity import (
    record_reading_highlight_created,
    record_reading_highlight_deleted,
    record_reading_highlight_updated,
)

router = APIRouter(tags=["reading-highlights"])  # no prefix; use absolute paths below


pass


def _extract_item_text_and_hash(db, item_id: int) -> tuple[Optional[str], Optional[str]]:
    try:
        item = db.get_content_item(item_id)
    except KeyError:
        return None, None

    metadata = {}
    if getattr(item, "metadata_json", None):
        try:
            metadata = json.loads(item.metadata_json) if item.metadata_json else {}
        except Exception:
            metadata = {}
    text = None
    if isinstance(metadata, dict):
        text = metadata.get("text")
    if not text:
        text = item.summary or item.notes or None

    content_hash = item.content_hash or hash_text_sha256(text)
    return text, content_hash


def _item_title_or_none(db, item_id: int) -> str | None:
    try:
        item = db.get_content_item(item_id)
    except KeyError:
        return None
    return getattr(item, "title", None)


@router.post("/reading/items/{item_id}/highlight", response_model=Highlight, summary="Create highlight for an item")
async def create_highlight(
    item_id: int = Path(..., ge=1),
    payload: HighlightCreateRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> Highlight:
    content_text, content_hash = _extract_item_text_and_hash(db, item_id)
    start_offset = payload.start_offset
    end_offset = payload.end_offset
    context_before = None
    context_after = None
    if content_text and payload.quote:
        span = find_highlight_span(
            content_text,
            payload.quote,
            start_offset=start_offset,
            end_offset=end_offset,
            anchor_strategy=payload.anchor_strategy,
        )
        if span is not None:
            start_offset, end_offset = span
            context_before, context_after = build_highlight_context(content_text, start_offset, end_offset)
    try:
        row = db.create_highlight(
            item_id=item_id,
            quote=payload.quote,
            start_offset=start_offset,
            end_offset=end_offset,
            color=payload.color,
            note=payload.note,
            anchor_strategy=payload.anchor_strategy,
            content_hash_ref=content_hash,
            context_before=context_before,
            context_after=context_after,
        )
    except Exception as e:
        logger.error("create_highlight failed")
        raise HTTPException(status_code=500, detail="highlight_create_failed") from e
    item_title = _item_title_or_none(db, item_id)
    record_reading_highlight_created(
        user_id=current_user.id,
        highlight=row,
        item_title=item_title,
    )
    return Highlight(
        id=row.id,
        item_id=row.item_id,
        quote=row.quote,
        start_offset=row.start_offset,
        end_offset=row.end_offset,
        color=row.color,
        note=row.note,
        created_at=__import__("datetime").datetime.fromisoformat(row.created_at),
        anchor_strategy=row.anchor_strategy,
        content_hash_ref=row.content_hash_ref,
        context_before=row.context_before,
        context_after=row.context_after,
        state=row.state,
    )


@router.get("/reading/items/{item_id}/highlights", response_model=list[Highlight], summary="List highlights for an item")
async def list_highlights_for_item(
    item_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> list[Highlight]:
    rows = db.list_highlights_by_item(item_id=item_id)
    return [
        Highlight(
            id=r.id,
            item_id=r.item_id,
            quote=r.quote,
            start_offset=r.start_offset,
            end_offset=r.end_offset,
            color=r.color,
            note=r.note,
            created_at=__import__("datetime").datetime.fromisoformat(r.created_at),
            anchor_strategy=r.anchor_strategy,
            content_hash_ref=r.content_hash_ref,
            context_before=r.context_before,
            context_after=r.context_after,
            state=r.state,
        )
        for r in rows
    ]


@router.patch("/reading/highlights/{highlight_id}", response_model=Highlight, summary="Update a highlight")
async def update_highlight(
    highlight_id: int = Path(..., ge=1),
    payload: HighlightUpdateRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> Highlight:
    patch = payload.dict(exclude_unset=True)
    try:
        row = db.update_highlight(highlight_id=highlight_id, patch=patch)
    except KeyError:
        raise HTTPException(status_code=404, detail="highlight_not_found") from None
    item_title = _item_title_or_none(db, row.item_id)
    record_reading_highlight_updated(
        user_id=current_user.id,
        highlight=row,
        item_title=item_title,
        patch=patch,
    )
    return Highlight(
        id=row.id,
        item_id=row.item_id,
        quote=row.quote,
        start_offset=row.start_offset,
        end_offset=row.end_offset,
        color=row.color,
        note=row.note,
        created_at=__import__("datetime").datetime.fromisoformat(row.created_at),
        anchor_strategy=row.anchor_strategy,
        content_hash_ref=row.content_hash_ref,
        context_before=row.context_before,
        context_after=row.context_after,
        state=row.state,
    )


@router.delete("/reading/highlights/{highlight_id}", summary="Delete a highlight")
async def delete_highlight(
    highlight_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> dict[str, Any]:
    try:
        row = db.get_highlight(highlight_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="highlight_not_found") from None
    ok = db.delete_highlight(highlight_id=highlight_id)
    if not ok:
        raise HTTPException(status_code=404, detail="highlight_not_found")
    item_title = _item_title_or_none(db, row.item_id)
    record_reading_highlight_deleted(
        user_id=current_user.id,
        highlight=row,
        item_title=item_title,
    )
    return {"success": True}
