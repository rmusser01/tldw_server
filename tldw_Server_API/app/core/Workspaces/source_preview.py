"""Reusable bounded source-preview projection for local and shared workspaces."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

_MAX_PREVIEW_CHARS = 12_000
_MAX_PREVIEW_CHUNKS = 10


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_get_media(media_db: Any | None, media_id: int) -> dict[str, Any] | None:
    if media_db is None:
        return None
    try:
        return media_db_api.get_media_by_id(media_db, media_id)
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError):
        return None


def _preview_mode_for_unavailable(source_status: dict[str, Any]) -> str:
    state = str(source_status.get("state") or "")
    reason = str(source_status.get("status_reason") or "")
    if state == "missing_media" or reason in {
        "media_not_found",
        "media_id_missing",
        "media_db_unavailable",
    }:
        return "missing_media"
    if state == "failed" or "failed" in reason:
        return "failed"
    if state in {"queued", "ingesting", "extracting", "chunking", "indexing", "retrying"}:
        return "pending"
    return "empty"


def _content_excerpt_snippet(
    *,
    source_id: str,
    media_id: int | None,
    text_preview: str,
) -> dict[str, Any]:
    return {
        "id": "content:0",
        "source_id": source_id,
        "media_id": media_id,
        "kind": "content_excerpt",
        "text": text_preview,
        "start_char": 0,
        "end_char": len(text_preview),
        "chunk_index": None,
        "chunk_uuid": None,
        "chunk_type": None,
    }


def _chunk_preview_snippets(
    *,
    media_db: Any | None,
    source_id: str,
    media_id: int | None,
    chunk_limit: int,
    focus_chunk_index: int | None,
) -> list[dict[str, Any]]:
    if media_db is None or media_id is None or chunk_limit <= 0:
        return []
    if focus_chunk_index is None:
        start_index = 0
    else:
        start_index = max(0, focus_chunk_index - (chunk_limit // 2))
    end_index = start_index + chunk_limit - 1
    try:
        chunks = media_db_api.get_unvectorized_chunks_in_range(
            media_db,
            media_id,
            start_index,
            end_index,
        )
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError):
        return []
    snippets: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks[:chunk_limit]):
        text = str(chunk.get("chunk_text") or "")
        if not text.strip():
            continue
        chunk_uuid = chunk.get("uuid")
        chunk_index = chunk.get("chunk_index")
        snippet_id = str(
            chunk_uuid or f"chunk:{chunk_index if chunk_index is not None else index}"
        )
        snippets.append(
            {
                "id": snippet_id,
                "source_id": source_id,
                "media_id": media_id,
                "kind": "chunk",
                "text": text,
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "chunk_index": chunk_index,
                "chunk_uuid": str(chunk_uuid) if chunk_uuid is not None else None,
                "chunk_type": chunk.get("chunk_type"),
            }
        )
    return snippets


def build_workspace_source_preview(
    *,
    workspace_id: str,
    source: dict[str, Any],
    source_status: dict[str, Any],
    media_db: Any | None,
    max_chars: int,
    chunk_limit: int,
    focus_chunk_index: int | None = None,
) -> dict[str, Any]:
    """Build the existing local preview shape with bounded optional focus."""
    if not 1 <= max_chars <= _MAX_PREVIEW_CHARS:
        raise ValueError(f"max_chars must be between 1 and {_MAX_PREVIEW_CHARS}")
    if not 0 <= chunk_limit <= _MAX_PREVIEW_CHUNKS:
        raise ValueError(f"chunk_limit must be between 0 and {_MAX_PREVIEW_CHUNKS}")
    if focus_chunk_index is not None and focus_chunk_index < 0:
        raise ValueError("focus_chunk_index must be non-negative")

    media_id_raw = source.get("media_id")
    try:
        media_id = int(media_id_raw) if media_id_raw is not None else None
    except (TypeError, ValueError):
        media_id = None

    media = _safe_get_media(media_db, media_id) if media_id is not None else None
    content = str((media or {}).get("content") or "")
    if not content.strip():
        reason = (
            "media_db_unavailable"
            if media_db is None
            else str(source_status.get("status_reason") or "content_unavailable")
        )
        return {
            "workspace_id": workspace_id,
            "source_id": source["id"],
            "media_id": media_id,
            "title": source.get("title") or "",
            "source_type": source.get("source_type") or "",
            "url": source.get("url"),
            "state": source_status.get("state") or "missing_media",
            "status_reason": reason,
            "readiness": source_status.get("readiness") or {},
            "content_available": False,
            "preview_mode": _preview_mode_for_unavailable(
                {**source_status, "status_reason": reason}
            ),
            "unavailable_reason": reason,
            "text_preview": None,
            "text_total_chars": None,
            "text_truncated": False,
            "snippets": [],
            "generated_at": _utc_now_iso(),
        }

    text_preview = content[:max_chars]
    snippets = [
        _content_excerpt_snippet(
            source_id=str(source["id"]),
            media_id=media_id,
            text_preview=text_preview,
        )
    ]
    snippets.extend(
        _chunk_preview_snippets(
            media_db=media_db,
            source_id=str(source["id"]),
            media_id=media_id,
            chunk_limit=chunk_limit,
            focus_chunk_index=focus_chunk_index,
        )
    )
    return {
        "workspace_id": workspace_id,
        "source_id": source["id"],
        "media_id": media_id,
        "title": source.get("title") or "",
        "source_type": source.get("source_type") or "",
        "url": source.get("url"),
        "state": source_status.get("state") or "queryable",
        "status_reason": source_status.get("status_reason") or "source_queryable",
        "readiness": source_status.get("readiness") or {},
        "content_available": True,
        "preview_mode": "available",
        "unavailable_reason": None,
        "text_preview": text_preview,
        "text_total_chars": len(content),
        "text_truncated": len(content) > len(text_preview),
        "snippets": snippets,
        "generated_at": _utc_now_iso(),
    }
