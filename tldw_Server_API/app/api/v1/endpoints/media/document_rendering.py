"""Stateless generated-document rendering and canonical annotated PDF export."""

from __future__ import annotations

import json
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Response

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireApiKeyScope,
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
    require_expected_user,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.document_rendering import (
    MAX_ANNOTATION_BYTES,
    MAX_PDF_BYTES,
    PDF_ERRORS,
    DocumentRenderRequest,
    PdfAnnotation,
)
from tldw_Server_API.app.core import Storage
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_READ
from tldw_Server_API.app.core.Chat.bounded_daemon import DaemonCapacityError
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)
from tldw_Server_API.app.core.Document_Workspace.pdf_service import (
    annotate_document_pdf,
    pdf_operation,
    render_document_pdf,
)
from tldw_Server_API.app.core.Storage.storage_interface import StorageError

router = APIRouter(tags=["Document Workspace"])


def _pdf_response(data: bytes, filename: str) -> Response:
    return Response(
        data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _processing_error(error: Exception) -> HTTPException:
    if isinstance(error, DaemonCapacityError):
        return HTTPException(503, "Document PDF processing is busy. Try again shortly.")
    if isinstance(error, TimeoutError):
        return HTTPException(504, "Document PDF processing timed out.")
    return HTTPException(422, str(error) if str(error) in PDF_ERRORS.values() else PDF_ERRORS["invalid_document"])


@router.post(
    "/documents/render",
    dependencies=[
        Depends(require_expected_user),
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(RequireApiKeyScope("write")),
        Depends(rbac_rate_limit("media.create")),
    ],
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
    summary="Render a reviewed plain-text document draft",
)
async def render_document(body: DocumentRenderRequest, _user: User = Depends(get_request_user)):
    """Render a new PDF; saving still uses normal Media ingestion and quotas."""
    try:
        with pdf_operation():
            data = await render_document_pdf(body)
    except (ValueError, TimeoutError, DaemonCapacityError) as exc:
        raise _processing_error(exc) from exc
    return _pdf_response(data, "document-draft.pdf")


@router.get(
    "/{media_id:int}/annotated-file",
    dependencies=[
        Depends(require_expected_user),
        Depends(RequirePermission(MEDIA_READ)),
        Depends(RequireApiKeyScope("read")),
        Depends(rbac_rate_limit("media.read")),
    ],
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
    summary="Export the preserved PDF with saved annotation comments",
)
async def get_annotated_document(
    media_id: int = Path(..., ge=1), db: Any = Depends(get_media_db_for_user), user: User = Depends(get_request_user)
):
    """Read only authorized saved content; never accept caller-supplied annotation rows."""
    try:
        with pdf_operation():
            return await _export_document(media_id, db, user)
    except DaemonCapacityError as exc:
        raise _processing_error(exc) from exc


async def _export_document(media_id: int, db: Any, user: User) -> Response:
    """Materialize canonical content only while holding request admission."""
    if not db.get_media_by_id(media_id, include_deleted=False, include_trash=False):
        raise HTTPException(404, "Media not found or is inactive/trashed")
    record = db.get_media_file(media_id, "original")
    if not record or not record.get("storage_path"):
        raise HTTPException(404, "The original PDF is not available. Re-upload it with its original file preserved.")
    if isinstance(record.get("file_size"), int) and record["file_size"] > MAX_PDF_BYTES:
        raise HTTPException(413, PDF_ERRORS["too_large"])
    rows = DocumentWorkspaceRepository(db).list_annotations_for_export(media_id=media_id, user_id=str(user.id))
    try:
        if len(rows) > 500 or any(row["invalid_text"] or len(str(row["location"])) > 64 for row in rows):
            raise ValueError(PDF_ERRORS["invalid_annotation"])
        annotations = [
            PdfAnnotation(
                page=int(row["location"]) if str(row["location"]).isdecimal() else 0,
                text=row["text"],
                note=row.get("note") or "",
                color=row["color"],
                kind=row["annotation_type"],
            )
            for row in rows
        ]
        if (
            len(json.dumps([item.model_dump() for item in annotations], ensure_ascii=False).encode("utf-8"))
            > MAX_ANNOTATION_BYTES
        ):
            raise ValueError(PDF_ERRORS["invalid_annotation"])
    except ValueError as exc:
        raise HTTPException(422, PDF_ERRORS["invalid_annotation"]) from exc
    storage = Storage.get_storage_backend()
    chunks: list[bytes] = []
    size = 0
    try:
        async with aclosing(storage.retrieve_stream(record["storage_path"])) as stream:
            async for chunk in stream:
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    raise HTTPException(413, PDF_ERRORS["too_large"])
                chunks.append(chunk)
    except FileNotFoundError as exc:
        raise HTTPException(404, "The original PDF is missing from storage.") from exc
    except StorageError as exc:
        raise HTTPException(500, "The original PDF could not be read from storage.") from exc
    try:
        data = await annotate_document_pdf(b"".join(chunks), annotations)
    except (ValueError, TimeoutError, DaemonCapacityError) as exc:
        raise _processing_error(exc) from exc
    return _pdf_response(data, f"document-{media_id}-annotated.pdf")
