from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.schemas.document_upload_processing import (
    ChatDocumentDraftCreateRequest,
    ChatDocumentDraftCreateResponse,
    ChatDocumentDraftReadResponse,
    DocumentModeCapability,
    DocumentProcessingMode,
    DocumentProcessingStatus,
    DocumentUploadPreflightFile,
    DocumentUploadPreflightItem,
    DocumentUploadPreflightRequest,
    DocumentUploadPreflightResponse,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import (
    list_backends as _list_backends,
)

router = APIRouter(tags=["Media Processing"])

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".docx",
    ".rtf",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".json",
}
SUPPORTED_EBOOK_EXTENSIONS = {".epub"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

DEFAULT_MAX_CHAT_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_CHAT_UPLOAD_PAGES = 200
DEFAULT_MAX_DIRECT_CHAT_TOKENS = 24_000
DRAFT_TTL_SECONDS = 15 * 60
MAX_DRAFT_PAYLOAD_BYTES = DEFAULT_MAX_CHAT_UPLOAD_BYTES

# ponytail: process-local handoff store; use shared cache/DB if drafts must survive restarts.
_DRAFTS: dict[str, dict[str, Any]] = {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _list_ocr_backends() -> dict[str, Any]:
    return _list_backends()


def _owner_key(current_user: Any) -> str:
    user_id = getattr(current_user, "id", None)
    if user_id is None:
        user_id = getattr(current_user, "id_str", "")
    return str(user_id)


def _cleanup_expired_drafts(now: datetime | None = None) -> None:
    current_time = now or _now_utc()
    expired_ids = [
        draft_id
        for draft_id, draft in _DRAFTS.items()
        if draft["expires_at"] <= current_time
    ]
    for draft_id in expired_ids:
        _DRAFTS.pop(draft_id, None)


MediaType = Literal["pdf", "document", "ebook", "unsupported"]


def _media_type_for(filename: str) -> MediaType:
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in SUPPORTED_EBOOK_EXTENSIONS:
        return "ebook"
    return "unsupported"


def _capability(
    available: bool,
    status_value: DocumentProcessingStatus,
    reason: str | None = None,
) -> DocumentModeCapability:
    return DocumentModeCapability(
        available=available,
        status=status_value,
        reason=reason,
    )


def _available(reason: str | None = None) -> DocumentModeCapability:
    return _capability(True, "available", reason)


def _unavailable(reason: str) -> DocumentModeCapability:
    return _capability(False, "unavailable", reason)


def _blocked(reason: str) -> DocumentModeCapability:
    return _capability(False, "blocked", reason)


def _ocr_available() -> bool:
    backends = _list_ocr_backends()
    return any(
        isinstance(details, dict) and details.get("available") is True
        for details in backends.values()
    )


def _size_limit_reason(file: DocumentUploadPreflightFile) -> str | None:
    if file.size_bytes <= DEFAULT_MAX_CHAT_UPLOAD_BYTES:
        return None
    size_mb = DEFAULT_MAX_CHAT_UPLOAD_BYTES // (1024 * 1024)
    return f"{file.filename} exceeds {size_mb} MB limit"


def _page_limit_reason(file: DocumentUploadPreflightFile) -> str | None:
    if file.page_count is None or file.page_count <= DEFAULT_MAX_CHAT_UPLOAD_PAGES:
        return None
    return f"{file.filename} exceeds {DEFAULT_MAX_CHAT_UPLOAD_PAGES} page limit"


def _token_limit_reason(file: DocumentUploadPreflightFile) -> str | None:
    if file.estimated_tokens is None or file.estimated_tokens <= DEFAULT_MAX_DIRECT_CHAT_TOKENS:
        return None
    return f"{file.filename} exceeds {DEFAULT_MAX_DIRECT_CHAT_TOKENS} token direct-chat limit"


def _unsupported_modes() -> dict[DocumentProcessingMode, DocumentModeCapability]:
    reason = "Unsupported document type"
    return {
        "add_to_chat": _unavailable(reason),
        "ocr_pages": _unavailable(reason),
        "ingest_to_library": _unavailable(reason),
    }


def _preflight_file(file: DocumentUploadPreflightFile) -> DocumentUploadPreflightItem:
    media_type = _media_type_for(file.filename)
    if media_type == "unsupported":
        return DocumentUploadPreflightItem(
            client_id=file.client_id,
            filename=file.filename,
            media_type="unsupported",
            default_mode=None,
            modes=_unsupported_modes(),
            max_size_bytes=DEFAULT_MAX_CHAT_UPLOAD_BYTES,
            max_pages=DEFAULT_MAX_CHAT_UPLOAD_PAGES,
            max_chat_tokens=DEFAULT_MAX_DIRECT_CHAT_TOKENS,
            estimated_pages=file.page_count,
            estimated_tokens=file.estimated_tokens,
            requires_send_time_estimate=False,
        )

    size_reason = _size_limit_reason(file)
    page_reason = _page_limit_reason(file)
    token_reason = _token_limit_reason(file)
    direct_reason = size_reason or page_reason or token_reason

    add_to_chat = _blocked(direct_reason) if direct_reason else _available()
    ingest = _blocked(size_reason) if size_reason else _available()
    ocr = _ocr_capability(file, media_type, size_reason or page_reason)

    modes: dict[DocumentProcessingMode, DocumentModeCapability] = {
        "add_to_chat": add_to_chat,
        "ocr_pages": ocr,
        "ingest_to_library": ingest,
    }
    default_mode: DocumentProcessingMode | None = (
        "add_to_chat" if modes["add_to_chat"].available else None
    )
    return DocumentUploadPreflightItem(
        client_id=file.client_id,
        filename=file.filename,
        media_type=media_type,
        default_mode=default_mode,
        modes=modes,
        max_size_bytes=DEFAULT_MAX_CHAT_UPLOAD_BYTES,
        max_pages=DEFAULT_MAX_CHAT_UPLOAD_PAGES,
        max_chat_tokens=DEFAULT_MAX_DIRECT_CHAT_TOKENS,
        estimated_pages=file.page_count,
        estimated_tokens=file.estimated_tokens,
        requires_send_time_estimate=file.page_count is None or file.estimated_tokens is None,
    )


def _ocr_capability(
    file: DocumentUploadPreflightFile,
    media_type: str,
    blocked_reason: str | None,
) -> DocumentModeCapability:
    if media_type != "pdf":
        suffix = Path(file.filename).suffix.upper() or "file"
        return _unavailable(f"OCR unavailable: server cannot render {suffix} pages")
    if blocked_reason:
        return _blocked(blocked_reason)
    if not _ocr_available():
        return _unavailable("OCR unavailable: no OCR backend configured")
    return _available()


def _draft_for(draft_id: str, current_user: Any) -> dict[str, Any]:
    _cleanup_expired_drafts()
    draft = _DRAFTS.get(draft_id)
    if draft is None or draft["owner"] != _owner_key(current_user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    return draft


@router.post(
    "/document-upload/preflight",
    response_model=DocumentUploadPreflightResponse,
)
def preflight_document_upload(
    request: DocumentUploadPreflightRequest,
) -> DocumentUploadPreflightResponse:
    return DocumentUploadPreflightResponse(
        files=[_preflight_file(file) for file in request.files],
    )


@router.post(
    "/document-upload/drafts",
    response_model=ChatDocumentDraftCreateResponse,
)
def create_document_upload_draft(
    request: ChatDocumentDraftCreateRequest,
    current_user: Any = Depends(get_request_user),
) -> ChatDocumentDraftCreateResponse:
    payload_json = json.dumps(request.payload, separators=(",", ":"))
    if len(payload_json.encode("utf-8")) > MAX_DRAFT_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Draft payload exceeds upload limit",
        )

    now = _now_utc()
    _cleanup_expired_drafts(now)
    draft_id = uuid4().hex
    expires_at = now + timedelta(seconds=DRAFT_TTL_SECONDS)
    _DRAFTS[draft_id] = {
        "owner": _owner_key(current_user),
        "created_at": now,
        "expires_at": expires_at,
        "payload": request.payload,
    }
    return ChatDocumentDraftCreateResponse(
        draft_id=draft_id,
        expires_at=expires_at.isoformat(),
    )


@router.get(
    "/document-upload/drafts/{draft_id}",
    response_model=ChatDocumentDraftReadResponse,
)
def read_document_upload_draft(
    draft_id: str,
    current_user: Any = Depends(get_request_user),
) -> ChatDocumentDraftReadResponse:
    draft = _draft_for(draft_id, current_user)
    return ChatDocumentDraftReadResponse(
        draft_id=draft_id,
        created_at=draft["created_at"].isoformat(),
        expires_at=draft["expires_at"].isoformat(),
        payload=draft["payload"],
    )


@router.delete(
    "/document-upload/drafts/{draft_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document_upload_draft(
    draft_id: str,
    response: Response,
    current_user: Any = Depends(get_request_user),
) -> Response:
    _draft_for(draft_id, current_user)
    _DRAFTS.pop(draft_id, None)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
