"""Read and bound text from the authenticated owner's saved media."""

from hashlib import sha256
from typing import Any

from fastapi import HTTPException

from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_source_projection
from tldw_Server_API.app.core.DB_Management.media_db.services.media_details_service import (
    _normalize_content_payload,
)

MAX_MEDIA_ANALYSIS_BYTES = 2 * 1024 * 1024


def read_saved_media_text(db: Any, media_id: int, *, owner_user_id: int, for_update: bool = False) -> str:
    """Use the bounded, explicitly owner-filtered source projection."""
    source = get_media_source_projection(
        db,
        media_id,
        max_chars=MAX_MEDIA_ANALYSIS_BYTES,
        owner_user_id=str(owner_user_id),
        current_content_only=True,
        for_update=for_update,
    )
    if source is None:
        raise HTTPException(404, "Media item not found.")
    if source["source_invalid"]:
        raise HTTPException(422, "Media item contains unsupported text.")
    raw_text = source["source_text"] or ""
    if len(raw_text) > MAX_MEDIA_ANALYSIS_BYTES:
        raise _too_large_error()
    _, text = _normalize_content_payload({"type": source.get("media_type"), "content": raw_text})
    return text


def prepare_analysis_text(prefix: str, text: str) -> str:
    """Bound the server-loaded prompt independently of the browser JSON limit."""
    if not text.strip():
        raise HTTPException(422, detail={
            "code": "media_analysis_no_content",
            "message": "No content available for analysis.",
        })
    combined = f"{prefix}\n\n{text}" if prefix else text
    if len(combined) > MAX_MEDIA_ANALYSIS_BYTES or len(combined.encode("utf-8")) > MAX_MEDIA_ANALYSIS_BYTES:
        raise _too_large_error()
    return combined


def _too_large_error() -> HTTPException:
    return HTTPException(413, detail={
        "code": "media_analysis_too_large",
        "message": "This item exceeds the 2 MiB analysis limit. Split it into smaller parts.",
    })


def media_source_sha256(text: str) -> str:
    """Identify the exact text supplied for an analysis."""
    return sha256(text.encode("utf-8")).hexdigest()
