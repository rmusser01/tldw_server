# audio_history.py
# Description: TTS history endpoints.
import base64
import binascii
import contextlib
import json
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from loguru import logger
from starlette import status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_cursor_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    TTSHistoryDetailResponse,
    TTSHistoryFavoriteUpdate,
    TTSHistoryListItem,
    TTSHistoryListResponse,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.TTS.utils import compute_tts_history_text_hash, parse_bool

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

_CURSOR_VERSION = 1
_TEXT_PREVIEW_LEN = 120
_TTS_HISTORY_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_TTS_HISTORY_CURSOR_EXCEPTIONS: tuple[type[BaseException], ...] = (
    binascii.Error,
    json.JSONDecodeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_TTS_HISTORY_JSON_PARSE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
)


def _tts_history_config() -> dict[str, Any]:
    return {
        "store_text": parse_bool(getattr(settings, "TTS_HISTORY_STORE_TEXT", True), default=True),
        "hash_key": getattr(settings, "TTS_HISTORY_HASH_KEY", None),
    }


def _encode_cursor(created_at: str, row_id: int) -> str:
    payload = {"v": _CURSOR_VERSION, "created_at": created_at, "id": int(row_id)}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _decode_cursor(token: str) -> tuple[str, int]:
    if not token:
        raise ValueError("empty cursor")
    pad = "=" * (-len(token) % 4)
    raw = base64.urlsafe_b64decode((token + pad).encode("utf-8")).decode("utf-8")
    payload = json.loads(raw)
    if int(payload.get("v", 0)) != _CURSOR_VERSION:
        raise ValueError("unsupported cursor version")
    created_at = payload.get("created_at")
    row_id = payload.get("id")
    if not created_at or row_id is None:
        raise ValueError("missing cursor fields")
    return str(created_at), int(row_id)


def _parse_json_field(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except _TTS_HISTORY_JSON_PARSE_EXCEPTIONS:
        return None


@router.get(
    "/history",
    summary="List TTS history entries.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.history", count_as="call")),
    ],
    response_model=TTSHistoryListResponse,
)
async def list_tts_history(
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
    q: Optional[str] = Query(default=None),
    text_exact: Optional[str] = Query(default=None),
    favorite: Optional[bool] = Query(default=None),
    provider: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    voice_id: Optional[str] = Query(default=None),
    voice_name: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    cursor: Optional[str] = Query(default=None),
    include_total: bool = Query(default=False),
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = Query(default=None, alias="to"),
):
    cfg = _tts_history_config()
    if q:
        if not cfg.get("store_text", True):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text search is disabled when TTS_HISTORY_STORE_TEXT=false",
            )
        q = str(q).strip()
        if not q:
            q = None

    text_hash = None
    if text_exact:
        try:
            text_hash = compute_tts_history_text_hash(text_exact, cfg.get("hash_key"))
        except _TTS_HISTORY_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("TTS history: failed to compute text_exact hash")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="TTS history hash key not configured",
            ) from exc

    cursor_created_at = None
    cursor_id = None
    if cursor:
        try:
            cursor_created_at, cursor_id = _decode_cursor(cursor)
            offset = 0
        except _TTS_HISTORY_CURSOR_EXCEPTIONS as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cursor",
            ) from exc

    list_start = None
    try:
        list_start = time.monotonic()
    except _TTS_HISTORY_NONCRITICAL_EXCEPTIONS:
        list_start = None
    try:
        rows = media_db.list_tts_history(
            user_id=str(request_user.id),
            q=q,
            text_hash=text_hash,
            favorite=favorite,
            provider=provider,
            model=model,
            voice_id=voice_id,
            voice_name=voice_name,
            created_from=from_,
            created_to=to,
            cursor_created_at=cursor_created_at,
            cursor_id=cursor_id,
            limit=limit + 1,
            offset=offset,
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to list TTS history") from exc
    with contextlib.suppress(_TTS_HISTORY_NONCRITICAL_EXCEPTIONS):
        log_counter(
            "tts_history_reads_total",
            labels={
                "favorite": "any" if favorite is None else str(bool(favorite)).lower(),
                "provider": provider or "any",
                "mode": "list",
            },
        )
    if list_start is not None:
        with contextlib.suppress(_TTS_HISTORY_NONCRITICAL_EXCEPTIONS):
            log_histogram(
                "tts_history_read_latency_ms",
                value=max(0.0, (time.monotonic() - list_start) * 1000),
                labels={"mode": "list"},
            )

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    items: list[TTSHistoryListItem] = []
    for row in rows:
        text_value = row.get("text")
        text_preview = text_value[:_TEXT_PREVIEW_LEN] if isinstance(text_value, str) else None
        items.append(
            TTSHistoryListItem(
                id=int(row.get("id")),
                created_at=row.get("created_at"),
                has_text=text_value is not None,
                text_preview=text_preview,
                provider=row.get("provider"),
                model=row.get("model"),
                voice_id=row.get("voice_id"),
                voice_name=row.get("voice_name"),
                voice_info=_parse_json_field(row.get("voice_info")),
                duration_ms=row.get("duration_ms"),
                format=row.get("format"),
                status=row.get("status"),
                favorite=bool(row.get("favorite")),
                job_id=row.get("job_id"),
                output_id=row.get("output_id"),
                artifact_deleted_at=row.get("artifact_deleted_at"),
            )
        )

    next_cursor = None
    if has_more and rows:
        last_row = rows[-1]
        try:
            next_cursor = _encode_cursor(last_row.get("created_at"), int(last_row.get("id")))
        except _TTS_HISTORY_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(
                "TTS history: failed to build next cursor for id={} created_at={}",
                last_row.get("id"),
                last_row.get("created_at"),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list TTS history",
            ) from exc

    total = None
    if include_total:
        total_start = None
        try:
            total_start = time.monotonic()
        except _TTS_HISTORY_NONCRITICAL_EXCEPTIONS:
            total_start = None
        try:
            total = media_db.count_tts_history(
                user_id=str(request_user.id),
                q=q,
                text_hash=text_hash,
                favorite=favorite,
                provider=provider,
                model=model,
                voice_id=voice_id,
                voice_name=voice_name,
                created_from=from_,
                created_to=to,
            )
        except DatabaseError as exc:
            raise map_db_error_to_http(exc, default_detail="Failed to list TTS history") from exc
        if total_start is not None:
            with contextlib.suppress(_TTS_HISTORY_NONCRITICAL_EXCEPTIONS):
                log_histogram(
                    "tts_history_read_latency_ms",
                    value=max(0.0, (time.monotonic() - total_start) * 1000),
                    labels={"mode": "count"},
                )

    return TTSHistoryListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_cursor=next_cursor,
        pagination=build_cursor_pagination_meta(
            limit=limit,
            cursor=cursor,
            next_cursor=next_cursor,
            has_more=has_more,
        ),
    )


@router.get(
    "/history/{history_id}",
    summary="Get TTS history entry details.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.history", count_as="call")),
    ],
    response_model=TTSHistoryDetailResponse,
)
async def get_tts_history_entry(
    history_id: int,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
):
    detail_start = None
    try:
        detail_start = time.monotonic()
    except _TTS_HISTORY_NONCRITICAL_EXCEPTIONS:
        detail_start = None
    try:
        row = media_db.get_tts_history_entry(
            user_id=str(request_user.id),
            history_id=int(history_id),
            include_deleted=False,
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch TTS history entry") from exc
    with contextlib.suppress(_TTS_HISTORY_NONCRITICAL_EXCEPTIONS):
        log_counter(
            "tts_history_reads_total",
            labels={"mode": "detail"},
        )
    if detail_start is not None:
        with contextlib.suppress(_TTS_HISTORY_NONCRITICAL_EXCEPTIONS):
            log_histogram(
                "tts_history_read_latency_ms",
                value=max(0.0, (time.monotonic() - detail_start) * 1000),
                labels={"mode": "detail"},
            )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History entry not found")

    text_value = row.get("text")
    return TTSHistoryDetailResponse(
        id=int(row.get("id")),
        created_at=row.get("created_at"),
        has_text=text_value is not None,
        text=text_value if isinstance(text_value, str) else None,
        text_length=row.get("text_length"),
        provider=row.get("provider"),
        model=row.get("model"),
        voice_id=row.get("voice_id"),
        voice_name=row.get("voice_name"),
        voice_info=_parse_json_field(row.get("voice_info")),
        format=row.get("format"),
        duration_ms=row.get("duration_ms"),
        generation_time_ms=row.get("generation_time_ms"),
        params_json=_parse_json_field(row.get("params_json")),
        status=row.get("status"),
        segments_json=_parse_json_field(row.get("segments_json")),
        favorite=bool(row.get("favorite")),
        job_id=row.get("job_id"),
        output_id=row.get("output_id"),
        artifact_ids=_parse_json_field(row.get("artifact_ids")),
        artifact_deleted_at=row.get("artifact_deleted_at"),
        error_message=row.get("error_message"),
    )


@router.patch(
    "/history/{history_id}",
    summary="Update TTS history entry fields.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.history", count_as="call")),
    ],
)
async def update_tts_history_entry(
    history_id: int,
    payload: TTSHistoryFavoriteUpdate,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
):
    try:
        updated = media_db.update_tts_history_favorite(
            user_id=str(request_user.id),
            history_id=int(history_id),
            favorite=bool(payload.favorite),
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update TTS history entry") from exc
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History entry not found")
    return {"id": int(history_id), "favorite": bool(payload.favorite)}


@router.delete(
    "/history/{history_id}",
    summary="Delete a TTS history entry (soft delete).",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.history", count_as="call")),
    ],
)
async def delete_tts_history_entry(
    history_id: int,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
):
    try:
        deleted = media_db.soft_delete_tts_history_entry(
            user_id=str(request_user.id),
            history_id=int(history_id),
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete TTS history entry") from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History entry not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
