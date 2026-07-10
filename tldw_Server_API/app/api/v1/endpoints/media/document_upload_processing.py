"""API routes for chat document upload preflight and handoff drafts."""

from __future__ import annotations

from typing import Any

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
from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_drafts import (
    DRAFT_TTL_SECONDS,
    MAX_DRAFT_PAYLOAD_BYTES,
    MAX_DRAFTS_PER_OWNER,
    MAX_DRAFTS_TOTAL,
    DocumentUploadDraftPayloadTooLargeError,
    DocumentUploadDraftQuotaError,
    get_document_upload_draft_store,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import (
    list_backends as _list_backends,
)

router = APIRouter(tags=["Media Processing"])


def _list_ocr_backends() -> dict[str, Any]:
    """Return the currently registered OCR backend capabilities."""
    return _list_backends()


def _owner_key(current_user: Any) -> str:
    """Return the stable storage owner key for an authenticated user."""
    user_id = getattr(current_user, "id", None)
    if user_id is None:
        user_id = getattr(current_user, "id_str", "")
    return str(user_id)


@router.post(
    "/document-upload/preflight",
    response_model=DocumentUploadPreflightResponse,
    dependencies=[Depends(rbac_rate_limit("media.read"))],
)
def preflight_document_upload(
    request: DocumentUploadPreflightRequest,
    _current_user: Any = Depends(get_request_user),
) -> DocumentUploadPreflightResponse:
    """Return supported processing modes and limits for uploaded documents."""
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
    """Persist a short-lived upload handoff draft for the current user."""
    try:
        draft = get_document_upload_draft_store().create(
            owner=_owner_key(current_user),
            payload=request.payload,
        )
    except DocumentUploadDraftPayloadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DocumentUploadDraftQuotaError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    return ChatDocumentDraftCreateResponse(
        draft_id=draft.draft_id,
        expires_at=draft.expires_at.isoformat(),
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
    """Return an owned, unexpired document upload draft."""
    draft = get_document_upload_draft_store().get(
        owner=_owner_key(current_user),
        draft_id=draft_id,
    )
    if draft is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    return ChatDocumentDraftReadResponse(
        draft_id=draft_id,
        created_at=draft.created_at.isoformat(),
        expires_at=draft.expires_at.isoformat(),
        payload=draft.payload,
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
    """Delete an owned, unexpired document upload draft."""
    deleted = get_document_upload_draft_store().delete(
        owner=_owner_key(current_user),
        draft_id=draft_id,
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
