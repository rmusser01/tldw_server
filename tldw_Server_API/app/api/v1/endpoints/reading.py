from __future__ import annotations

import asyncio
import html
import json
import os
import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from loguru import logger

try:
    import bleach  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    bleach = None
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, User

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.items import bulk_update_items as bulk_update_items_handler
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.items_schemas import ItemsBulkRequest, ItemsBulkResponse
from tldw_Server_API.app.api.v1.schemas.reading_schemas import (
    ReadingArchiveCreateRequest,
    ReadingArchiveResponse,
    ReadingCitation,
    ReadingDeleteResponse,
    ReadingDigestOutput,
    ReadingDigestOutputsListResponse,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleFilters,
    ReadingDigestScheduleResponse,
    ReadingDigestScheduleUpdateRequest,
    ReadingImportJobResponse,
    ReadingImportJobsListResponse,
    ReadingImportJobStatus,
    ReadingImportResponse,
    ReadingItem,
    ReadingItemDetail,
    ReadingNoteLinkCreateRequest,
    ReadingNoteLinkResponse,
    ReadingNoteLinksListResponse,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchListResponse,
    ReadingSavedSearchResponse,
    ReadingSavedSearchUpdateRequest,
    ReadingItemsListResponse,
    ReadingSaveRequest,
    ReadingSummarizeRequest,
    ReadingSummaryResponse,
    ReadingTTSRequest,
    ReadingUpdateRequest,
)
from tldw_Server_API.app.core.Collections.reading_import_jobs import (
    MAX_READING_IMPORT_BYTES,
    READING_IMPORT_DOMAIN,
    READING_IMPORT_JOB_TYPE,
    reading_import_queue,
    stage_reading_import_file,
)
from tldw_Server_API.app.core.Collections.reading_service import ReadingService
from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    ContentItemNoteLinkRow,
    ContentItemRow,
    SavedSearchRow,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.feature_flags import (
    is_collections_reading_archive_controls_enabled,
    is_collections_reading_note_links_enabled,
    is_collections_reading_saved_searches_enabled,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze as summarize_analyze
from tldw_Server_API.app.core.Personalization.companion_activity import (
    record_reading_item_archived,
    record_reading_item_deleted,
    record_reading_item_saved,
    record_reading_note_linked,
    record_reading_note_unlinked,
    record_reading_item_updated,
)
from tldw_Server_API.app.core.TTS.tts_config import get_tts_config
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSError,
    TTSInvalidVoiceReferenceError,
    TTSProviderNotConfiguredError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSValidationError,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
from tldw_Server_API.app.core.TTS.tts_validation import TTSInputValidator
from tldw_Server_API.app.services.outputs_service import (
    _outputs_dir_for_user,
    _resolve_output_path_for_user,
    _sanitize_title_for_filename,
)
from tldw_Server_API.app.services.reading_digest_scheduler import get_reading_digest_scheduler

READING_ARCHIVE_MAX_BYTES = int(os.getenv("READING_ARCHIVE_MAX_BYTES", str(5 * 1024 * 1024)))
READING_ARCHIVE_RETENTION_DAYS = int(os.getenv("READING_ARCHIVE_RETENTION_DAYS", "30") or "30")

_ARCHIVE_ALLOWED_TAGS = [
    "a",
    "blockquote",
    "br",
    "code",
    "div",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "ul",
]
_ARCHIVE_ALLOWED_ATTRS = {
    "a": ["href", "title", "rel", "target"],
    "img": ["src", "alt", "title", "width", "height"],
}
_ARCHIVE_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


router = APIRouter(prefix="/reading", tags=["reading"])

_READING_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _service_for_user(user: User) -> ReadingService:
    if not user or user.id is None:
        raise HTTPException(status_code=500, detail="user_missing")
    return ReadingService(user.id)


def _parse_metadata(row: ContentItemRow) -> dict[str, object]:
    if getattr(row, "metadata_json", None):
        try:
            return json.loads(row.metadata_json) if row.metadata_json else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _derive_processing_status(row: ContentItemRow, metadata: dict[str, object]) -> str:
    status_raw = str(metadata.get("processing_status", "")).lower()
    if status_raw in {"processing", "ready"}:
        return status_raw
    if metadata.get("fetch_error"):
        return "ready"
    if row.media_id or row.content_hash or row.summary or metadata.get("text"):
        return "ready"
    return "processing"


def _to_reading_item(row) -> ReadingItem:
    metadata = _parse_metadata(row)
    fetch_error = metadata.get("last_fetch_error") or metadata.get("fetch_error")
    return ReadingItem(
        id=int(row.id),
        media_id=row.media_id,
        media_uuid=metadata.get("media_uuid") if metadata else None,
        title=row.title or "Untitled",
        url=row.url or row.canonical_url,
        canonical_url=row.canonical_url,
        domain=row.domain,
        summary=row.summary,
        notes=row.notes,
        published_at=row.published_at,
        status=row.status,
        processing_status=_derive_processing_status(row, metadata),
        archive_requested=bool(metadata.get("archive_requested", False)),
        has_archive_copy=bool(metadata.get("has_archive_copy", False)),
        last_fetch_error=str(fetch_error) if fetch_error else None,
        favorite=bool(row.favorite),
        tags=row.tags,
        created_at=row.created_at,
        updated_at=row.updated_at,
        read_at=row.read_at,
    )


def _to_reading_detail(row: ContentItemRow) -> ReadingItemDetail:
    metadata = _parse_metadata(row)
    return ReadingItemDetail(
        **_to_reading_item(row).model_dump(),
        text=metadata.get("text") if metadata else None,
        clean_html=metadata.get("clean_html") if metadata else None,
        metadata=metadata,
    )


def _to_saved_search_response(row: SavedSearchRow) -> ReadingSavedSearchResponse:
    query_payload: dict[str, object] = {}
    raw = getattr(row, "query_json", None)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                query_payload = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            query_payload = {}
    return ReadingSavedSearchResponse(
        id=int(row.id),
        name=str(row.name),
        query=query_payload,
        sort=row.sort,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_note_link_response(row: ContentItemNoteLinkRow) -> ReadingNoteLinkResponse:
    return ReadingNoteLinkResponse(
        item_id=int(row.item_id),
        note_id=str(row.note_id),
        created_at=row.created_at,
    )


def _ensure_note_exists_or_404(notes_db: CharactersRAGDB, note_id: str) -> None:
    note_data = notes_db.get_note_by_id(note_id=str(note_id))
    if not note_data:
        raise HTTPException(status_code=404, detail="note_not_found")


def _ensure_reading_saved_searches_enabled() -> None:
    if not is_collections_reading_saved_searches_enabled():
        raise HTTPException(status_code=404, detail="reading_saved_searches_disabled")


def _ensure_reading_note_links_enabled() -> None:
    if not is_collections_reading_note_links_enabled():
        raise HTTPException(status_code=404, detail="reading_note_links_disabled")


def _ensure_reading_archive_controls_enabled() -> None:
    if not is_collections_reading_archive_controls_enabled():
        raise HTTPException(status_code=404, detail="reading_archive_controls_disabled")


def _select_text_for_action(
    row: ContentItemRow,
    metadata: dict[str, object],
    text_source: str | None,
) -> str:
    if text_source == "summary":
        return row.summary or ""
    if text_source == "notes":
        return row.notes or ""
    if text_source == "text":
        return str(metadata.get("text") or "")
    return str(metadata.get("text") or row.summary or row.notes or "")


def _build_reading_citation(row: ContentItemRow) -> ReadingCitation:
    return ReadingCitation(
        item_id=int(row.id),
        url=row.url or row.canonical_url,
        canonical_url=row.canonical_url,
        title=row.title or None,
        source="reading",
    )


def _tts_content_type(response_format: str) -> str | None:
    mapping = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/L16; rate=24000; channels=1",
    }
    return mapping.get(response_format)


def _raise_for_tts_error(exc: Exception) -> None:
    if isinstance(exc, TTSInvalidVoiceReferenceError):
        raise HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, TTSValidationError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, TTSProviderNotConfiguredError):
        raise HTTPException(status_code=503, detail="TTS service unavailable")
    if isinstance(exc, TTSAuthenticationError):
        raise HTTPException(status_code=502, detail="TTS provider authentication failed")
    if isinstance(exc, TTSRateLimitError):
        raise HTTPException(status_code=429, detail="TTS provider rate limit exceeded")
    if isinstance(exc, TTSQuotaExceededError):
        raise HTTPException(status_code=402, detail="TTS quota exceeded")
    if isinstance(exc, TTSError):
        raise HTTPException(status_code=500, detail="TTS generation failed")
    raise HTTPException(status_code=500, detail="TTS generation failed")


def _parse_import_job_result(result: object) -> ReadingImportResponse | None:
    if result is None:
        return None
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(result, dict):
        try:
            return ReadingImportResponse(**result)
        except (TypeError, ValueError):
            return None
    return None


def _to_import_job_status(job: dict[str, object]) -> ReadingImportJobStatus:
    return ReadingImportJobStatus(
        job_id=int(job.get("id")),
        job_uuid=job.get("uuid"),
        status=str(job.get("status") or ""),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress_percent=job.get("progress_percent"),
        progress_message=job.get("progress_message"),
        error_message=job.get("error_message") or job.get("last_error"),
        result=_parse_import_job_result(job.get("result")),
    )


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_digest_filters(raw: str | None) -> ReadingDigestScheduleFilters | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    try:
        return ReadingDigestScheduleFilters(**payload)
    except (TypeError, ValueError):
        return None


def _validate_cron_or_422(cron: str, timezone_str: str | None) -> None:
    try:
        from apscheduler.triggers.cron import CronTrigger
        CronTrigger.from_crontab(cron, timezone=timezone_str or "UTC")
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid cron or timezone. Timezone must be an IANA name. "
                f"Details: {exc}"
            ),
        ) from exc


def _resolve_archive_retention(payload: ReadingArchiveCreateRequest) -> str | None:
    if payload.retention_until:
        dt = _parse_iso_datetime(payload.retention_until)
        if dt is None:
            raise HTTPException(status_code=422, detail="reading_archive_retention_invalid")
        return dt.isoformat()
    days = payload.retention_days
    if days is None:
        days = max(0, int(READING_ARCHIVE_RETENTION_DAYS))
    if days <= 0:
        return None
    return (datetime.now(tz=timezone.utc) + timedelta(days=days)).isoformat()


def _sanitize_archive_html(value: str) -> str:
    if bleach is None:
        return html.escape(value)
    return bleach.clean(
        value,
        tags=_ARCHIVE_ALLOWED_TAGS,
        attributes=_ARCHIVE_ALLOWED_ATTRS,
        protocols=_ARCHIVE_ALLOWED_PROTOCOLS,
        strip=True,
        strip_comments=True,
    )


def _render_archive_html(
    *,
    title: str,
    url: str | None,
    body_html: str | None,
    body_text: str | None,
) -> str:
    safe_title = html.escape(title or "Untitled")
    safe_url = html.escape(url or "")
    content_html = _sanitize_archive_html(body_html) if body_html else f"<pre>{html.escape(body_text or '')}</pre>"
    header = f"<h1>{safe_title}</h1>"
    if safe_url:
        header = f"{header}<p><a href=\"{safe_url}\">{safe_url}</a></p>"
    return (
        "<!doctype html>"
        "<html><head><meta charset=\"utf-8\">"
        f"<title>{safe_title}</title></head><body>{header}{content_html}</body></html>"
    )


def _strip_basic_html(html_value: str) -> str:
    stripped = re.sub(r"<[^>]+>", "", html_value)
    return html.unescape(stripped)


def _serialize_highlight_row(row) -> dict[str, object]:
    return {
        "id": row.id,
        "item_id": row.item_id,
        "quote": row.quote,
        "start_offset": row.start_offset,
        "end_offset": row.end_offset,
        "color": row.color,
        "note": row.note,
        "created_at": row.created_at,
        "anchor_strategy": row.anchor_strategy,
        "content_hash_ref": row.content_hash_ref,
        "context_before": row.context_before,
        "context_after": row.context_after,
        "state": row.state,
    }


@router.post(
    "/save",
    response_model=ReadingItem,
    summary="Save a URL into the reading list",
    dependencies=[Depends(rbac_rate_limit("reading.save"))],
)
async def save_reading_item(
    payload: ReadingSaveRequest = Body(
        ...,
        examples={
            "basic": {
                "summary": "Save a URL",
                "value": {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "tags": ["ai", "reading"],
                    "notes": "Why it matters",
                },
            },
            "inline_content": {
                "summary": "Save inline content (offline/testing)",
                "value": {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "content": "Inline article content used for tests.",
                },
            },
        },
    ),
    current_user: User = Depends(get_request_user),
) -> ReadingItem:
    if payload.archive_mode != "use_default":
        _ensure_reading_archive_controls_enabled()
    service = _service_for_user(current_user)
    try:
        result = await service.save_url(
            url=str(payload.url),
            tags=payload.tags,
            status=payload.status,
            favorite=payload.favorite,
            title_override=payload.title,
            summary_override=payload.summary,
            content_override=payload.content,
            notes=payload.notes,
            archive_mode=payload.archive_mode,
        )
        item = _to_reading_item(result.item)
        record_reading_item_saved(user_id=current_user.id, item=item)
        return item
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_save_failed")
        raise HTTPException(status_code=400, detail="reading_save_failed") from exc


@router.get(
    "/items",
    response_model=ReadingItemsListResponse,
    summary="List reading items",
    dependencies=[Depends(rbac_rate_limit("reading.list"))],
)
async def list_reading_items(
    status: list[str] | None = Query(None),
    tags: list[str] | None = Query(None),
    favorite: bool | None = Query(None),
    q: str | None = Query(None),
    domain: str | None = Query(None),
    date_from: str | None = Query(None, description="ISO start date inclusive"),
    date_to: str | None = Query(None, description="ISO end date inclusive"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    offset: int | None = Query(None, ge=0),
    limit: int | None = Query(None, ge=1, le=200),
    sort: str | None = Query(
        None,
        description="updated_desc|updated_asc|created_desc|created_asc|title_asc|title_desc|relevance",
    ),
    current_user: User = Depends(get_request_user),
) -> ReadingItemsListResponse:
    service = _service_for_user(current_user)
    date_from_iso = None
    date_to_iso = None
    try:
        start_dt = datetime.fromisoformat(date_from) if date_from else None
        end_dt = datetime.fromisoformat(date_to) if date_to else None
        if start_dt:
            date_from_iso = start_dt.isoformat()
        if end_dt:
            date_to_iso = end_dt.isoformat()
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="invalid_date_range") from None
    resolved_limit = limit if limit is not None else size
    resolved_offset = offset if offset is not None else max(0, (page - 1) * size)
    if limit is not None:
        page = int(resolved_offset / resolved_limit) + 1 if resolved_limit else page
        size = resolved_limit
    rows, total = service.list_items(
        status=status,
        tags=tags,
        favorite=favorite,
        q=q,
        domain=domain,
        date_from=date_from_iso,
        date_to=date_to_iso,
        page=page,
        size=size,
        offset=resolved_offset,
        limit=resolved_limit,
        sort=sort,
    )
    return ReadingItemsListResponse(
        items=[_to_reading_item(row) for row in rows],
        total=total,
        page=page,
        size=size,
        offset=resolved_offset,
        limit=resolved_limit,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=resolved_limit,
            offset=resolved_offset,
            count=len(rows),
        ),
    )


@router.post(
    "/saved-searches",
    response_model=ReadingSavedSearchResponse,
    status_code=201,
    summary="Create a reading saved search",
    dependencies=[Depends(rbac_rate_limit("reading.update"))],
)
async def create_reading_saved_search(
    payload: ReadingSavedSearchCreateRequest,
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingSavedSearchResponse:
    _ensure_reading_saved_searches_enabled()
    try:
        row = collections_db.create_saved_search(
            name=payload.name,
            query_json=json.dumps(payload.query or {}, ensure_ascii=False),
            sort=payload.sort,
        )
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_saved_search_create_failed")
        raise HTTPException(status_code=400, detail="reading_saved_search_create_failed") from exc
    return _to_saved_search_response(row)


@router.get(
    "/saved-searches",
    response_model=ReadingSavedSearchListResponse,
    summary="List reading saved searches",
    dependencies=[Depends(rbac_rate_limit("reading.list"))],
)
async def list_reading_saved_searches(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingSavedSearchListResponse:
    _ensure_reading_saved_searches_enabled()
    try:
        rows, total = collections_db.list_saved_searches(limit=limit, offset=offset)
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_saved_search_list_failed")
        raise HTTPException(status_code=400, detail="reading_saved_search_list_failed") from exc
    return ReadingSavedSearchListResponse(
        items=[_to_saved_search_response(row) for row in rows],
        total=int(total),
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=int(total),
            limit=limit,
            offset=offset,
            count=len(rows),
        ),
    )


@router.patch(
    "/saved-searches/{search_id}",
    response_model=ReadingSavedSearchResponse,
    summary="Update a reading saved search",
    dependencies=[Depends(rbac_rate_limit("reading.update"))],
)
async def update_reading_saved_search(
    search_id: int,
    payload: ReadingSavedSearchUpdateRequest,
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingSavedSearchResponse:
    _ensure_reading_saved_searches_enabled()
    patch: dict[str, object] = {}
    if payload.name is not None:
        patch["name"] = payload.name
    if payload.query is not None:
        patch["query_json"] = json.dumps(payload.query, ensure_ascii=False)
    if payload.sort is not None:
        patch["sort"] = payload.sort
    try:
        row = collections_db.update_saved_search(search_id, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_saved_search_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_saved_search_update_failed")
        raise HTTPException(status_code=400, detail="reading_saved_search_update_failed") from exc
    return _to_saved_search_response(row)


@router.delete(
    "/saved-searches/{search_id}",
    response_model=dict[str, bool],
    summary="Delete a reading saved search",
    dependencies=[Depends(rbac_rate_limit("reading.update"))],
)
async def delete_reading_saved_search(
    search_id: int,
    collections_db=Depends(get_collections_db_for_user),
) -> dict[str, bool]:
    _ensure_reading_saved_searches_enabled()
    try:
        ok = collections_db.delete_saved_search(search_id)
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_saved_search_delete_failed")
        raise HTTPException(status_code=400, detail="reading_saved_search_delete_failed") from exc
    if not ok:
        raise HTTPException(status_code=404, detail="reading_saved_search_not_found")
    return {"ok": True}


@router.post(
    "/items/{item_id}/links/note",
    response_model=ReadingNoteLinkResponse,
    summary="Link a note to a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.update"))],
)
async def link_note_to_reading_item(
    item_id: int,
    payload: ReadingNoteLinkCreateRequest,
    current_user: User = Depends(get_request_user),
    notes_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingNoteLinkResponse:
    _ensure_reading_note_links_enabled()
    _ensure_note_exists_or_404(notes_db, payload.note_id)
    try:
        item = collections_db.get_content_item(item_id)
        row = collections_db.link_note_to_content_item(item_id=item_id, note_id=payload.note_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_note_link_create_failed")
        raise HTTPException(status_code=400, detail="reading_note_link_create_failed") from exc
    record_reading_note_linked(
        user_id=current_user.id,
        item_id=item_id,
        note_id=payload.note_id,
        item_title=getattr(item, "title", None),
        link_created_at=getattr(row, "created_at", None),
    )
    return _to_note_link_response(row)


@router.get(
    "/items/{item_id}/links",
    response_model=ReadingNoteLinksListResponse,
    summary="List note links for a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.read"))],
)
async def list_reading_item_note_links(
    item_id: int,
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingNoteLinksListResponse:
    _ensure_reading_note_links_enabled()
    try:
        links = collections_db.list_note_links_for_content_item(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_note_link_list_failed")
        raise HTTPException(status_code=400, detail="reading_note_link_list_failed") from exc
    return ReadingNoteLinksListResponse(
        item_id=item_id,
        links=[_to_note_link_response(row) for row in links],
    )


@router.delete(
    "/items/{item_id}/links/note/{note_id}",
    response_model=dict[str, bool],
    summary="Unlink a note from a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.update"))],
)
async def unlink_note_from_reading_item(
    item_id: int,
    note_id: str,
    current_user: User = Depends(get_request_user),
    collections_db=Depends(get_collections_db_for_user),
) -> dict[str, bool]:
    _ensure_reading_note_links_enabled()
    try:
        item = collections_db.get_content_item(item_id)
        existing_link = next(
            (
                row
                for row in collections_db.list_note_links_for_content_item(item_id)
                if str(row.note_id) == str(note_id)
            ),
            None,
        )
        ok = collections_db.unlink_note_from_content_item(item_id=item_id, note_id=note_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_note_link_delete_failed")
        raise HTTPException(status_code=400, detail="reading_note_link_delete_failed") from exc
    if not ok:
        raise HTTPException(status_code=404, detail="reading_note_link_not_found")
    record_reading_note_unlinked(
        user_id=current_user.id,
        item_id=item_id,
        note_id=note_id,
        item_title=getattr(item, "title", None),
        link_created_at=getattr(existing_link, "created_at", None),
    )
    return {"ok": True}


@router.post("/items/bulk", response_model=ItemsBulkResponse, summary="Bulk update reading items (alias)")
async def bulk_update_reading_items(
    payload: ItemsBulkRequest,
    current_user: User = Depends(get_request_user),
    collections_db = Depends(get_collections_db_for_user),
) -> ItemsBulkResponse:
    return await bulk_update_items_handler(
        payload,
        current_user=current_user,
        collections_db=collections_db,
    )


@router.post(
    "/import",
    response_model=ReadingImportJobResponse,
    status_code=202,
    summary="Import Pocket/Instapaper export into reading list",
    dependencies=[Depends(rbac_rate_limit("reading.import"))],
)
async def import_reading_items(
    file: UploadFile = File(...),
    source: str = Form("auto"),
    merge_tags: bool = Form(True),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> ReadingImportJobResponse:
    try:
        raw = await file.read()
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_import_read_failed")
        raise HTTPException(status_code=400, detail="reading_import_failed") from exc
    if not raw:
        raise HTTPException(status_code=400, detail="reading_import_empty")
    if len(raw) > MAX_READING_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="reading_import_too_large")

    staged_path = None
    try:
        staged_path = stage_reading_import_file(
            user_id=current_user.id,
            filename=file.filename,
            raw_bytes=raw,
        )
        payload = {
            "file_token": staged_path.name,
            "source": source,
            "merge_tags": merge_tags,
            "filename": file.filename,
        }
        job = jm.create_job(
            domain=READING_IMPORT_DOMAIN,
            queue=reading_import_queue(),
            job_type=READING_IMPORT_JOB_TYPE,
            payload=payload,
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=3,
        )
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_import_job_create_failed")
        if staged_path is not None:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("reading_import_staged_file_cleanup_failed")
        raise HTTPException(status_code=500, detail="reading_import_failed") from exc

    return ReadingImportJobResponse(
        job_id=int(job.get("id")),
        job_uuid=job.get("uuid"),
        status=str(job.get("status") or "queued"),
    )


@router.get(
    "/import/jobs",
    response_model=ReadingImportJobsListResponse,
    summary="List reading import jobs",
    dependencies=[Depends(rbac_rate_limit("reading.import.status"))],
)
async def list_reading_import_jobs(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> ReadingImportJobsListResponse:
    fetch_limit = limit + offset
    rows = jm.list_jobs(
        domain=READING_IMPORT_DOMAIN,
        owner_user_id=str(current_user.id),
        job_type=READING_IMPORT_JOB_TYPE,
        status=status,
        limit=fetch_limit,
    )
    total = jm.count_jobs(
        domain=READING_IMPORT_DOMAIN,
        owner_user_id=str(current_user.id),
        status=status,
    )
    jobs = rows[offset: offset + limit] if offset else rows[:limit]
    return ReadingImportJobsListResponse(
        jobs=[_to_import_job_status(row) for row in jobs],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(jobs),
        ),
    )


@router.get(
    "/import/jobs/{job_id}",
    response_model=ReadingImportJobStatus,
    summary="Get a reading import job",
    dependencies=[Depends(rbac_rate_limit("reading.import.status"))],
)
async def get_reading_import_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> ReadingImportJobStatus:
    job = jm.get_job(job_id)
    if not job or str(job.get("domain")) != READING_IMPORT_DOMAIN:
        raise HTTPException(status_code=404, detail="reading_import_job_not_found")
    if str(job.get("owner_user_id")) != str(current_user.id):
        raise HTTPException(status_code=403, detail="reading_import_job_forbidden")
    return _to_import_job_status(job)


@router.get(
    "/items/{item_id}",
    response_model=ReadingItemDetail,
    summary="Get reading item detail",
    dependencies=[Depends(rbac_rate_limit("reading.read"))],
)
async def get_reading_item(
    item_id: int,
    current_user: User = Depends(get_request_user),
) -> ReadingItemDetail:
    service = _service_for_user(current_user)
    try:
        row = service.get_item(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_get_failed")
        raise HTTPException(status_code=400, detail="reading_get_failed") from exc
    return _to_reading_detail(row)


@router.post(
    "/items/{item_id}/summarize",
    response_model=ReadingSummaryResponse,
    summary="Summarize a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.summarize"))],
)
async def summarize_reading_item(
    item_id: int,
    payload: ReadingSummarizeRequest = Body(
        ...,
        examples={
            "default": {
                "summary": "Summarize with provider",
                "value": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "prompt": "Summarize for a product brief.",
                },
            }
        },
    ),
    current_user: User = Depends(get_request_user),
) -> ReadingSummaryResponse:
    service = _service_for_user(current_user)
    try:
        row = service.get_item(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_summary_get_failed")
        raise HTTPException(status_code=400, detail="reading_item_fetch_failed") from exc

    metadata = _parse_metadata(row)
    text = _select_text_for_action(row, metadata, None)
    if not text.strip():
        raise HTTPException(status_code=400, detail="reading_item_no_content")

    if payload.recursive and payload.chunked:
        raise HTTPException(status_code=400, detail="reading_summary_invalid_strategy")

    provider = (payload.provider or DEFAULT_LLM_PROVIDER).strip()
    if not provider:
        provider = DEFAULT_LLM_PROVIDER
    loop = asyncio.get_running_loop()
    try:
        summary = await loop.run_in_executor(
            None,
            lambda: summarize_analyze(
                api_name=provider,
                input_data=text,
                custom_prompt_arg=payload.prompt,
                api_key=None,
                system_message=payload.system_prompt,
                temp=payload.temperature,
                streaming=False,
                recursive_summarization=payload.recursive,
                chunked_summarization=payload.chunked,
                chunk_options=None,
                model_override=payload.model,
            ),
        )
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_summarize_failed")
        raise HTTPException(status_code=503, detail="reading_summarize_failed") from exc

    if not isinstance(summary, str):
        summary = str(summary)
    if not summary or summary.strip().lower().startswith("error:"):
        logger.error("reading_summarize_error")
        raise HTTPException(status_code=503, detail="reading_summarize_failed")

    citation = _build_reading_citation(row)
    generated_at = datetime.now(tz=timezone.utc).isoformat()
    return ReadingSummaryResponse(
        item_id=int(row.id),
        summary=summary,
        provider=str(provider),
        model=payload.model,
        citations=[citation],
        generated_at=generated_at,
    )


@router.post(
    "/items/{item_id}/tts",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Generated reading item audio.",
            "content": {
                "audio/aac": {},
                "audio/flac": {},
                "audio/L16": {},
                "audio/mpeg": {},
                "audio/opus": {},
                "audio/wav": {},
            },
        },
    },
    summary="Generate TTS audio for a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.tts"))],
)
async def tts_reading_item(
    item_id: int,
    payload: ReadingTTSRequest = Body(
        ...,
        examples={
            "stream_mp3": {
                "summary": "Stream MP3 audio",
                "value": {
                    "model": "kokoro",
                    "voice": "af_heart",
                    "response_format": "mp3",
                    "stream": True,
                    "text_source": "text",
                },
            }
        },
    ),
    current_user: User = Depends(get_request_user),
) -> Response:
    service = _service_for_user(current_user)
    try:
        row = service.get_item(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_tts_get_failed")
        raise HTTPException(status_code=400, detail="reading_item_fetch_failed") from exc

    metadata = _parse_metadata(row)
    text = _select_text_for_action(row, metadata, payload.text_source)
    if payload.text_source and not text.strip():
        raise HTTPException(status_code=400, detail="reading_item_text_source_empty")
    if not text.strip():
        raise HTTPException(status_code=400, detail="reading_item_no_content")
    if payload.max_chars:
        text = text[: payload.max_chars]

    tts_config = get_tts_config()
    validator = TTSInputValidator({"strict_validation": tts_config.strict_validation})
    try:
        sanitized_text = validator.sanitize_text(text)
    except TTSValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not sanitized_text.strip():
        raise HTTPException(status_code=400, detail="reading_tts_empty_input")

    response_format = payload.response_format
    content_type = _tts_content_type(response_format)
    if not content_type:
        raise HTTPException(status_code=400, detail="reading_tts_format_invalid")

    tts_request = OpenAISpeechRequest(
        model=payload.model,
        input=sanitized_text,
        voice=payload.voice,
        response_format=response_format,
        stream=payload.stream,
    )
    if payload.speed is not None:
        tts_request.speed = payload.speed

    try:
        tts_service = await get_tts_service_v2()
        speech_iter = tts_service.generate_speech(
            tts_request,
            fallback=True,
            user_id=current_user.id,
        )
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        _raise_for_tts_error(exc)

    headers = {
        "Content-Disposition": f"attachment; filename=reading_{item_id}.{response_format}",
        "X-Reading-Item-Id": str(item_id),
    }
    if row.url or row.canonical_url:
        headers["X-Reading-Url"] = row.url or row.canonical_url or ""

    async def _stream_chunks():
        try:
            async for chunk in speech_iter:
                if chunk:
                    yield chunk
        except _READING_NONCRITICAL_EXCEPTIONS as exc:
            _raise_for_tts_error(exc)

    if payload.stream:
        return StreamingResponse(_stream_chunks(), media_type=content_type, headers=headers)

    audio_bytes = b""
    try:
        async for chunk in speech_iter:
            if chunk:
                audio_bytes += chunk
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        _raise_for_tts_error(exc)

    audio_bytes = audio_bytes.replace(b"--final_boundary_for_non_streamed--", b"")
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="reading_tts_no_audio")
    return Response(content=audio_bytes, media_type=content_type, headers=headers)


@router.patch(
    "/items/{item_id}",
    response_model=ReadingItem,
    summary="Update reading item metadata",
    dependencies=[Depends(rbac_rate_limit("reading.update"))],
)
async def update_reading_item(
    item_id: int,
    payload: ReadingUpdateRequest = Body(
        ...,
        examples={
            "mark_read": {
                "summary": "Mark read and favorite",
                "value": {"status": "read", "favorite": True, "tags": ["ai", "priority"]},
            }
        },
    ),
    current_user: User = Depends(get_request_user),
) -> ReadingItem:
    service = _service_for_user(current_user)
    try:
        row = service.update_item(
            item_id=item_id,
            status=payload.status,
            favorite=payload.favorite,
            tags=payload.tags,
            notes=payload.notes,
            title=payload.title,
        )
        item = _to_reading_item(row)
        record_reading_item_updated(user_id=current_user.id, item=item)
        return item
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_update_failed")
        raise HTTPException(status_code=400, detail="reading_update_failed") from exc


@router.delete(
    "/items/{item_id}",
    response_model=ReadingDeleteResponse,
    summary="Delete a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.delete"))],
)
async def delete_reading_item(
    item_id: int,
    hard: bool = Query(False),
    current_user: User = Depends(get_request_user),
) -> ReadingDeleteResponse:
    service = _service_for_user(current_user)
    try:
        if hard:
            service.delete_item(item_id)
            record_reading_item_deleted(user_id=current_user.id, item_id=item_id)
            return ReadingDeleteResponse(status="deleted", item_id=item_id, hard=True)
        row = service.update_item(item_id, status="archived")
        item = _to_reading_item(row)
        record_reading_item_archived(user_id=current_user.id, item=item)
        return ReadingDeleteResponse(status=row.status or "archived", item_id=item_id, hard=False)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_delete_failed")
        raise HTTPException(status_code=400, detail="reading_delete_failed") from exc


@router.post(
    "/items/{item_id}/archive",
    response_model=ReadingArchiveResponse,
    summary="Create an archive snapshot for a reading item",
    dependencies=[Depends(rbac_rate_limit("reading.archive"))],
)
async def create_reading_archive(
    item_id: int,
    payload: ReadingArchiveCreateRequest = Body(...),
    current_user: User = Depends(get_request_user),
) -> ReadingArchiveResponse:
    service = _service_for_user(current_user)
    try:
        row = service.get_item(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="reading_item_not_found") from None

    metadata = _parse_metadata(row)
    clean_html = metadata.get("clean_html") if isinstance(metadata, dict) else None
    text = metadata.get("text") if isinstance(metadata, dict) else None
    fallback_text = text or row.summary or row.notes or ""

    source = payload.source
    format_value = payload.format
    body_html: str | None = None
    body_text: str | None = None
    if source == "clean_html":
        if not clean_html:
            raise HTTPException(status_code=409, detail="reading_archive_no_html")
        body_html = clean_html
    elif source == "text":
        if not fallback_text:
            raise HTTPException(status_code=409, detail="reading_archive_no_text")
        body_text = fallback_text
    else:
        if format_value == "html":
            if clean_html:
                body_html = clean_html
            elif fallback_text:
                body_text = fallback_text
            else:
                raise HTTPException(status_code=409, detail="reading_archive_no_content")
        else:
            if fallback_text:
                body_text = fallback_text
            elif clean_html:
                body_text = _strip_basic_html(clean_html)
            else:
                raise HTTPException(status_code=409, detail="reading_archive_no_content")

    base_title = payload.title or row.title or "Reading Archive"
    if format_value == "html":
        content = _render_archive_html(
            title=base_title,
            url=row.canonical_url or row.url,
            body_html=body_html,
            body_text=body_text,
        )
        ext = "html"
    else:
        parts = [f"# {base_title}"]
        url = row.canonical_url or row.url
        if url:
            parts.append("")
            parts.append(url)
        if body_text:
            parts.append("")
            parts.append(body_text)
        content = "\n".join(parts).strip() + "\n"
        ext = "md"

    content_bytes = content.encode("utf-8")
    if READING_ARCHIVE_MAX_BYTES > 0 and len(content_bytes) > READING_ARCHIVE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="reading_archive_too_large")

    user_id = int(current_user.id)
    out_dir = _outputs_dir_for_user(user_id)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_archive_outputs_dir_failed")
        raise HTTPException(status_code=500, detail="storage_unavailable") from exc

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_title = f"{base_title} (archive {ts})"
    safe_title = _sanitize_title_for_filename(base_title)
    filename = f"reading_archive_{item_id}_{safe_title}_{ts}.{ext}"
    path = _resolve_output_path_for_user(user_id, filename)

    try:
        path.write_text(content, encoding="utf-8")
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_archive_write_failed")
        raise HTTPException(status_code=500, detail="reading_archive_write_failed") from exc

    retention_until = _resolve_archive_retention(payload)
    meta = {
        "item_id": row.id,
        "url": row.url,
        "canonical_url": row.canonical_url,
        "source": source,
        "format": format_value,
        "title": row.title,
    }
    try:
        output_row = service.collections.create_output_artifact(
            type_="reading_archive",
            title=archive_title,
            format_=format_value,
            storage_path=filename,
            metadata_json=json.dumps(meta),
            media_item_id=row.media_id,
            retention_until=retention_until,
        )
    except _READING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("reading_archive_db_failed")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("reading_archive_cleanup_failed")
        raise HTTPException(status_code=500, detail="reading_archive_db_failed") from exc

    return ReadingArchiveResponse(
        output_id=output_row.id,
        title=output_row.title,
        format=format_value,
        storage_path=output_row.storage_path,
        created_at=output_row.created_at,
        retention_until=retention_until,
        download_url=f"/api/v1/outputs/{output_row.id}/download",
    )


@router.get(
    "/export",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Reading item export download.",
            "content": {
                "application/x-ndjson": {},
                "application/zip": {},
            },
        },
    },
    summary="Export reading list items",
    dependencies=[Depends(rbac_rate_limit("reading.export"))],
)
async def export_reading_items(
    status: list[str] | None = Query(None),
    tags: list[str] | None = Query(None),
    favorite: bool | None = Query(None),
    q: str | None = Query(None),
    domain: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(1000, ge=1, le=10000),
    include_metadata: bool = Query(True),
    include_clean_html: bool = Query(False),
    include_text: bool = Query(False),
    include_highlights: bool = Query(False),
    include_notes: bool = Query(True),
    format: str = Query("jsonl", description="Export format: jsonl or zip"),
    current_user: User = Depends(get_request_user),
) -> StreamingResponse:
    service = _service_for_user(current_user)
    rows, total = service.list_items(
        status=status,
        tags=tags,
        favorite=favorite,
        q=q,
        domain=domain,
        page=page,
        size=size,
    )

    def _serialize_row(row: ContentItemRow) -> dict:
        metadata = {}
        if (include_metadata or include_clean_html or include_text) and getattr(row, "metadata_json", None):
            try:
                metadata = json.loads(row.metadata_json) if row.metadata_json else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        payload = {
            "id": row.id,
            "url": row.url,
            "canonical_url": row.canonical_url,
            "domain": row.domain,
            "title": row.title,
            "summary": row.summary,
            "status": row.status,
            "favorite": row.favorite,
            "tags": row.tags,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "read_at": row.read_at,
            "published_at": row.published_at,
            "origin_type": row.origin_type,
        }
        if include_notes:
            payload["notes"] = row.notes
        if include_metadata:
            payload["metadata"] = metadata
        if include_clean_html:
            payload["clean_html"] = metadata.get("clean_html") if isinstance(metadata, dict) else None
        if include_text:
            payload["text"] = metadata.get("text") if isinstance(metadata, dict) else None
        if include_highlights:
            try:
                highlights = service.collections.list_highlights_by_item(item_id=row.id)
            except _READING_NONCRITICAL_EXCEPTIONS:
                logger.debug("reading_export_highlights_fetch_failed")
                highlights = []
            payload["highlights"] = [_serialize_highlight_row(h) for h in highlights]
        return payload

    export_rows = [_serialize_row(row) for row in rows]
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if format.lower() == "zip":
        import io
        import json as _json
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            payload = "".join(_json.dumps(row, ensure_ascii=False) + "\n" for row in export_rows)
            zf.writestr("reading_export.jsonl", payload)
        buffer.seek(0)
        headers = {"Content-Disposition": f"attachment; filename=reading_export_{timestamp}.zip"}
        return StreamingResponse(buffer, media_type="application/zip", headers=headers)

    if format.lower() != "jsonl":
        raise HTTPException(status_code=400, detail="reading_export_format_invalid")

    import json as _json

    def _iter_lines():
        for row in export_rows:
            yield _json.dumps(row, ensure_ascii=False) + "\n"

    headers = {"Content-Disposition": f"attachment; filename=reading_export_{timestamp}.jsonl"}
    return StreamingResponse(_iter_lines(), media_type="application/x-ndjson", headers=headers)


# -------------------------
# Reading Digest Schedules
# -------------------------
def _digest_schedule_response(row) -> ReadingDigestScheduleResponse:
    return ReadingDigestScheduleResponse(
        id=str(row.id),
        name=row.name,
        cron=row.cron,
        timezone=row.timezone,
        enabled=bool(row.enabled),
        require_online=bool(row.require_online),
        format=str(row.format or "md"),
        template_id=row.template_id,
        template_name=row.template_name,
        retention_days=row.retention_days,
        filters=_parse_digest_filters(getattr(row, "filters_json", None)),
        last_run_at=row.last_run_at,
        next_run_at=row.next_run_at,
        last_status=row.last_status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.post(
    "/digests/schedules",
    response_model=dict[str, str],
    status_code=201,
    dependencies=[Depends(rbac_rate_limit("reading.digests"))],
)
async def create_reading_digest_schedule(
    body: ReadingDigestScheduleCreateRequest,
    current_user: User = Depends(get_request_user),
) -> dict[str, str]:
    _validate_cron_or_422(body.cron, body.timezone)
    svc = get_reading_digest_scheduler()
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    filters = body.filters.model_dump(exclude_none=True) if body.filters else {}
    schedule_id = svc.create(
        tenant_id=tenant_id,
        user_id=str(current_user.id),
        name=body.name,
        cron=body.cron,
        timezone=body.timezone,
        enabled=body.enabled,
        require_online=body.require_online,
        filters=filters,
        template_id=body.template_id,
        template_name=body.template_name,
        format=body.format,
        retention_days=body.retention_days,
    )
    return {"id": schedule_id}


@router.get(
    "/digests/schedules",
    response_model=list[ReadingDigestScheduleResponse],
    dependencies=[Depends(rbac_rate_limit("reading.digests"))],
)
async def list_reading_digest_schedules(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
) -> list[ReadingDigestScheduleResponse]:
    svc = get_reading_digest_scheduler()
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    rows = svc.list(tenant_id=tenant_id, user_id=str(current_user.id), limit=limit, offset=offset)
    return [_digest_schedule_response(row) for row in rows]


@router.get(
    "/digests/schedules/{schedule_id}",
    response_model=ReadingDigestScheduleResponse,
    dependencies=[Depends(rbac_rate_limit("reading.digests"))],
)
async def get_reading_digest_schedule(
    schedule_id: str,
    current_user: User = Depends(get_request_user),
) -> ReadingDigestScheduleResponse:
    svc = get_reading_digest_scheduler()
    schedule = svc.get(schedule_id)
    if not schedule or str(schedule.user_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="reading_digest_schedule_not_found")
    return _digest_schedule_response(schedule)


@router.patch(
    "/digests/schedules/{schedule_id}",
    response_model=ReadingDigestScheduleResponse,
    dependencies=[Depends(rbac_rate_limit("reading.digests"))],
)
async def update_reading_digest_schedule(
    schedule_id: str,
    body: ReadingDigestScheduleUpdateRequest,
    current_user: User = Depends(get_request_user),
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingDigestScheduleResponse:
    svc = get_reading_digest_scheduler()
    schedule = svc.get(schedule_id)
    if not schedule or str(schedule.user_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="reading_digest_schedule_not_found")
    if body.cron:
        _validate_cron_or_422(body.cron, body.timezone or schedule.timezone)
    patch = body.model_dump(exclude_none=True)
    filters_payload = patch.get("filters")
    if isinstance(filters_payload, ReadingDigestScheduleFilters):
        patch["filters"] = filters_payload.model_dump(exclude_none=True)
    elif isinstance(filters_payload, dict):
        patch["filters"] = filters_payload
    if not patch:
        return _digest_schedule_response(schedule)
    svc.update(schedule_id, patch)
    updated = collections_db.get_reading_digest_schedule(schedule_id)
    return _digest_schedule_response(updated)


@router.delete(
    "/digests/schedules/{schedule_id}",
    response_model=dict[str, bool],
    dependencies=[Depends(rbac_rate_limit("reading.digests"))],
)
async def delete_reading_digest_schedule(
    schedule_id: str,
    current_user: User = Depends(get_request_user),
) -> dict[str, bool]:
    svc = get_reading_digest_scheduler()
    schedule = svc.get(schedule_id)
    if not schedule or str(schedule.user_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="reading_digest_schedule_not_found")
    ok = svc.delete(schedule_id)
    return {"ok": bool(ok)}


@router.get(
    "/digests/outputs",
    response_model=ReadingDigestOutputsListResponse,
    dependencies=[Depends(rbac_rate_limit("reading.digests"))],
)
async def list_reading_digest_outputs(
    schedule_id: str | None = Query(None, description="Optional schedule id filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _current_user: User = Depends(get_request_user),
    collections_db=Depends(get_collections_db_for_user),
) -> ReadingDigestOutputsListResponse:
    batch = 200
    matched: list[ReadingDigestOutput] = []
    scan_offset = 0
    total_seen = 0

    while True:
        rows, total = collections_db.list_output_artifacts(
            limit=batch,
            offset=scan_offset,
            type_="reading_digest",
            include_deleted=False,
        )
        total_seen = total
        if not rows:
            break
        for row in rows:
            meta: dict[str, object] = {}
            try:
                if row.metadata_json:
                    meta = json.loads(row.metadata_json)
            except (json.JSONDecodeError, TypeError, ValueError):
                meta = {}
            if schedule_id and str(meta.get("schedule_id")) != str(schedule_id):
                continue
            matched.append(
                ReadingDigestOutput(
                    output_id=row.id,
                    title=row.title,
                    format=row.format,
                    created_at=row.created_at,
                    download_url=f"/api/v1/outputs/{row.id}/download",
                    schedule_id=meta.get("schedule_id"),
                    schedule_name=meta.get("schedule_name"),
                    item_count=meta.get("item_count"),
                    metadata=meta or None,
                )
            )
        scan_offset += batch
        if scan_offset >= total_seen:
            break

    total_matches = len(matched)
    page = matched[offset : offset + limit]
    return ReadingDigestOutputsListResponse(
        items=page,
        total=total_matches,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total_matches,
            limit=limit,
            offset=offset,
            count=len(page),
        ),
    )
