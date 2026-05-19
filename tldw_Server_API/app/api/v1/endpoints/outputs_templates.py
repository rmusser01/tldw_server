# tldw_Server_API/app/api/v1/endpoints/outputs_templates.py
"""
Output Templates API (CRUD + preview)

Stub endpoints to anchor the unified Content Collections PRD.
Implements request/response models using outputs_templates_schemas.

DB access must be implemented via app.core.DB_Management (no raw SQL here).
"""

import json
import sqlite3
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.outputs_templates_schemas import (
    OutputTemplate,
    OutputTemplateCreate,
    OutputTemplateList,
    OutputTemplateUpdate,
    TemplatePreviewRequest,
    TemplatePreviewResponse,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import search_media
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.services.outputs_service import (
    build_items_context_from_content_items,
    render_output_template,
)

router = APIRouter(prefix="/outputs/templates", tags=["outputs-templates"])


@router.get("", response_model=OutputTemplateList, summary="List output templates")
async def list_output_templates(
    q: str | None = Query(None, description="Optional search query by name/description"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> OutputTemplateList:
    # DB via dependency
    items, total = db.list_output_templates(q=q, limit=limit, offset=offset)
    return OutputTemplateList(
        items=[
            OutputTemplate(
                id=i.id,
                user_id=str(i.user_id),
                name=i.name,
                type=i.type,
                format=i.format,
                body=i.body,
                description=i.description,
                is_default=bool(i.is_default),
                created_at=datetime.fromisoformat(i.created_at),
                updated_at=datetime.fromisoformat(i.updated_at),
                metadata=(json.loads(i.metadata_json) if i.metadata_json else None),
            )
            for i in items
        ],
        total=total,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=offset,
            limit=limit,
            count=len(items),
        ),
    )


@router.post("", response_model=OutputTemplate, summary="Create output template")
async def create_output_template(
    payload: OutputTemplateCreate = Body(...),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> OutputTemplate:
    row = db.create_output_template(
        name=payload.name,
        type_=payload.type,
        format_=payload.format,
        body=payload.body,
        description=payload.description,
        is_default=payload.is_default,
        metadata_json=(json.dumps(payload.metadata) if payload.metadata is not None else None),
    )
    return OutputTemplate(
        id=row.id,
        user_id=row.user_id,
        name=row.name,
        type=row.type,
        format=row.format,
        body=row.body,
        description=row.description,
        is_default=row.is_default,
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
        metadata=(json.loads(row.metadata_json) if row.metadata_json else None),
    )


@router.get("/{template_id}", response_model=OutputTemplate, summary="Get output template")
async def get_output_template(
    template_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> OutputTemplate:
    try:
        row = db.get_output_template(template_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="template_not_found") from None
    return OutputTemplate(
        id=row.id,
        user_id=row.user_id,
        name=row.name,
        type=row.type,
        format=row.format,
        body=row.body,
        description=row.description,
        is_default=row.is_default,
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
        metadata=(json.loads(row.metadata_json) if row.metadata_json else None),
    )


@router.patch("/{template_id}", response_model=OutputTemplate, summary="Update output template")
async def update_output_template(
    template_id: int = Path(..., ge=1),
    payload: OutputTemplateUpdate = Body(...),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> OutputTemplate:
    patch = payload.dict(exclude_unset=True)
    if "metadata" in patch:
        md = patch.pop("metadata")
        patch["metadata_json"] = json.dumps(md) if md is not None else None
    try:
        row = db.update_output_template(template_id, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail="template_not_found") from None
    return OutputTemplate(
        id=row.id,
        user_id=row.user_id,
        name=row.name,
        type=row.type,
        format=row.format,
        body=row.body,
        description=row.description,
        is_default=row.is_default,
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
        metadata=(json.loads(row.metadata_json) if row.metadata_json else None),
    )


@router.delete("/{template_id}", summary="Delete output template")
async def delete_output_template(
    template_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
) -> dict[str, Any]:
    ok = db.delete_output_template(template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="template_not_found")
    return {"success": True}


def _domain_from_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname or ""
    except (ValueError, TypeError, AttributeError):
        return ""


def _build_items_context_from_media_ids(media_db, item_ids: list[int], limit: int) -> list[dict[str, object]]:
    if not item_ids:
        return []
    # Leverage search_media_db to fetch multiple items at once
    try:
        rows, total = search_media(
            media_db,
            search_query=None,
            media_ids_filter=item_ids[:limit],
            page=1,
            results_per_page=min(limit, max(1, len(item_ids))),
            include_deleted=False,
            include_trash=False,
        )
    except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        rows, _total = [], 0

    items: list[dict[str, object]] = []
    for r in rows:
        mid = int(r.get("id"))
        title = r.get("title") or "Untitled"
        url = r.get("url") or None
        domain = _domain_from_url(url)
        # Latest version to get analysis/safe_metadata
        latest = None
        try:
            from tldw_Server_API.app.core.DB_Management.media_db.api import (
                fetch_keywords_for_media,
                get_document_version,
            )
            latest = get_document_version(media_db, media_id=mid, version_number=None, include_content=False)
            tags = fetch_keywords_for_media(db=media_db, media_id=mid) or []
        except (ImportError, AttributeError, ValueError, TypeError, KeyError, RuntimeError):
            tags = []
        published_at = None
        if isinstance(latest, dict):
            sm = latest.get("safe_metadata")
            if isinstance(sm, str):
                try:
                    import json as _json
                    sm = _json.loads(sm)
                except (json.JSONDecodeError, TypeError, ValueError):
                    sm = None
            if isinstance(sm, dict):
                published_at = sm.get("published_at") or sm.get("date")
        if not published_at:
            dt = r.get("ingestion_date")
            published_at = dt if isinstance(dt, str) else None
        summary = None
        if isinstance(latest, dict):
            summary = latest.get("analysis_content") or None
        if not summary:
            content = r.get("content") or ""
            if isinstance(content, str) and content:
                summary = (content[:500] + "...") if len(content) > 500 else content

        items.append(
            {
                "id": mid,
                "title": title,
                "url": url,
                "domain": domain,
                "summary": summary or "",
                "published_at": published_at or "",
                "tags": tags,
            }
        )
    return items


def _select_media_ids_for_run(media_db, run_id: int, limit: int) -> list[int]:
    """Attempt to resolve media IDs for a run across candidate mapping tables.

    This function is defensive: it tries several known table names and returns
    the first successful set. If no mapping tables exist, returns empty list.
    """
    candidates = [
        "scrape_run_items",
        "watchlist_run_items",
        "runs_items",
        "scrape_runs_items",
    ]
    for tbl in candidates:
        try:
            media_lookup_sql_template = "SELECT media_id FROM {tbl} WHERE run_id = ? ORDER BY media_id LIMIT ?"
            q = media_lookup_sql_template.format_map(locals())  # nosec B608
            cur = media_db.execute_query(q, (run_id, limit))
            rows = cur.fetchall()
            mids = [int(r["media_id"]) if isinstance(r, dict) else int(r[0]) for r in rows]
            if mids:
                return mids
        except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            continue
    # Fallback: try if Media has run_id (unlikely)
    try:
        cur = media_db.execute_query("SELECT id FROM Media WHERE run_id = ? ORDER BY id LIMIT ?", (run_id, limit))
        rows = cur.fetchall()
        mids = [int(r["id"]) if isinstance(r, dict) else int(r[0]) for r in rows]
        return mids
    except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        return []


@router.post("/{template_id}/preview", response_model=TemplatePreviewResponse, summary="Preview template rendering")
async def preview_output_template(
    template_id: int = Path(..., ge=1),
    payload: TemplatePreviewRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db = Depends(get_collections_db_for_user),
    media_db = Depends(get_media_db_for_user),
) -> TemplatePreviewResponse:
    # Basic Markdown/HTML preview using Jinja2-like rendering.
    # TTS/mp3 cannot be previewed as text.
    try:
        row = db.get_output_template(template_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="template_not_found") from None
    if row.format == "mp3":
        raise HTTPException(status_code=422, detail="tts_audio templates cannot be previewed as text")

    # Inline data (advanced users) takes precedence
    if payload.data:
        try:
            ctx = dict(payload.data)
            # Ensure required keys exist
            ctx.setdefault("items", [])
            ctx.setdefault("date", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
            context = ctx
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=f"invalid_inline_data: {e}") from e
    else:
        # Build real or sample context for rendering
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        items: list[dict[str, object]]
        if payload.item_ids:
            items = _build_items_context_from_media_ids(media_db, payload.item_ids, payload.limit)
        elif payload.run_id is not None:
            items = []
            try:
                rows, _ = db.list_content_items(run_id=payload.run_id, page=1, size=payload.limit)
                items = build_items_context_from_content_items(rows)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, sqlite3.Error) as exc:
                logger.opt(exception=True).warning(
                    f"Content items lookup failed for run_id={payload.run_id}, falling back to media IDs: {exc}"
                )
                items = []
            if not items:
                mids = _select_media_ids_for_run(media_db, payload.run_id, payload.limit)
                items = _build_items_context_from_media_ids(media_db, mids, payload.limit) if mids else []
        else:
            # Provide samples when no selection is provided.
            items_count = 3
            items = [
                {
                    "title": f"Sample Article {i+1}",
                    "url": f"https://example.com/article-{i+1}",
                    "domain": "example.com",
                    "summary": "A short summary of the article content.",
                    "published_at": now,
                    "tags": ["sample", "preview"],
                }
                for i in range(items_count)
            ]
        context = {
            "date": now,
            "job": {
                "name": "Preview",
                "run_id": payload.run_id,
                "selection": {
                    "item_ids": payload.item_ids or [],
                    "count": len(items),
                },
            },
            "items": items,
            "tags": sorted({t for it in items for t in it.get("tags", []) if isinstance(it.get("tags", []), list)}),
        }

    rendered = render_output_template(row.body, context)
    fmt = row.format if row.format in ("md", "html") else "md"
    return TemplatePreviewResponse(rendered=rendered, format=fmt)
