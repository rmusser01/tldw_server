"""API routes for chat document upload preflight and handoff drafts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.schemas.document_upload_processing import (
    ChatDocumentDraftCreateRequest,
    ChatDocumentDraftCreateResponse,
    ChatDocumentDraftReadResponse,
    DocumentUploadPreflightRequest,
    DocumentUploadPreflightResponse,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_preflight import (
    DEFAULT_MAX_CHAT_UPLOAD_BYTES,
    DEFAULT_MAX_CHAT_UPLOAD_PAGES,
    DEFAULT_MAX_DIRECT_CHAT_TOKENS,
    preflight_document_upload_files,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import (
    list_backends as _list_backends,
)

router = APIRouter(tags=["Media Processing"])

DRAFT_TTL_SECONDS = 15 * 60
MAX_DRAFT_PAYLOAD_BYTES = DEFAULT_MAX_CHAT_UPLOAD_BYTES
MAX_DRAFTS_TOTAL = 256
MAX_DRAFTS_PER_OWNER = 32

# ponytail: process-local handoff store; use shared cache/DB if drafts must survive restarts.
_DRAFTS: dict[str, dict[str, Any]] = {}
_DRAFTS_LOCK = RLock()


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
    with _DRAFTS_LOCK:
        expired_ids = [
            draft_id
            for draft_id, draft in _DRAFTS.items()
            if draft["expires_at"] <= current_time
        ]
        for draft_id in expired_ids:
            _DRAFTS.pop(draft_id, None)


def _enforce_draft_quota(owner: str) -> None:
    owner_count = sum(1 for draft in _DRAFTS.values() if draft["owner"] == owner)
    if owner_count >= MAX_DRAFTS_PER_OWNER:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many active document upload drafts for this user",
        )
    if len(_DRAFTS) >= MAX_DRAFTS_TOTAL:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many active document upload drafts",
        )


def _draft_for(draft_id: str, current_user: Any) -> dict[str, Any]:
    with _DRAFTS_LOCK:
        _cleanup_expired_drafts()
        draft = _DRAFTS.get(draft_id)
        if draft is None or draft["owner"] != _owner_key(current_user):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
        return dict(draft)


@router.post(
    "/document-upload/preflight",
    response_model=DocumentUploadPreflightResponse,
    dependencies=[Depends(rbac_rate_limit("media.read"))],
)
def preflight_document_upload(
    request: DocumentUploadPreflightRequest,
    _current_user: Any = Depends(get_request_user),
) -> DocumentUploadPreflightResponse:
    return DocumentUploadPreflightResponse(
        files=preflight_document_upload_files(
            request.files,
            list_ocr_backends=_list_ocr_backends,
        ),
    )


@router.post(
    "/document-upload/drafts",
    response_model=ChatDocumentDraftCreateResponse,
    dependencies=[Depends(rbac_rate_limit("media.write"))],
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
    draft_id = uuid4().hex
    expires_at = now + timedelta(seconds=DRAFT_TTL_SECONDS)
    owner = _owner_key(current_user)
    with _DRAFTS_LOCK:
        _cleanup_expired_drafts(now)
        _enforce_draft_quota(owner)
        _DRAFTS[draft_id] = {
            "owner": owner,
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
    dependencies=[Depends(rbac_rate_limit("media.read"))],
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
    dependencies=[Depends(rbac_rate_limit("media.write"))],
)
def delete_document_upload_draft(
    draft_id: str,
    response: Response,
    current_user: Any = Depends(get_request_user),
) -> Response:
    with _DRAFTS_LOCK:
        _draft_for(draft_id, current_user)
        _DRAFTS.pop(draft_id, None)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
