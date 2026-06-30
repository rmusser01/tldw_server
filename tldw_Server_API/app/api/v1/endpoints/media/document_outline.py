# Document Outline/TOC Extraction Endpoint
# Extracts table of contents from PDF documents using PyMuPDF
#
from __future__ import annotations

from typing import Any, BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.document_outline import (
    DocumentOutlineResponse,
    OutlineEntry,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Storage import get_storage_backend
from tldw_Server_API.app.core.Storage.storage_interface import StorageError

router = APIRouter(tags=["Document Workspace"])

# Maximum file size for outline extraction (500 MB)
MAX_OUTLINE_FILE_SIZE = 500 * 1024 * 1024


def _check_pymupdf_available() -> bool:
    """Check if PyMuPDF is available for import."""
    try:
        import pymupdf  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_pdf_outline(pdf_data: bytes | BinaryIO) -> tuple[list[OutlineEntry], int]:
    """
    Extract table of contents from PDF using PyMuPDF.

    Args:
        pdf_data: Raw PDF file content as bytes, or a file-like object (BinaryIO).
                  If a BinaryIO is passed, its contents will be read into memory.

    Returns:
        Tuple of (list of outline entries, total page count)

    Raises:
        RuntimeError: If PyMuPDF is not installed
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed") from exc

    # Handle both bytes and file-like objects
    pdf_bytes = pdf_data.read() if hasattr(pdf_data, "read") else pdf_data

    entries: list[OutlineEntry] = []

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = doc.page_count

        # get_toc returns list of [level, title, page_number, ...]
        # page_number is 1-indexed
        toc = doc.get_toc()

        for item in toc:
            if len(item) >= 3:
                level, title, page = item[0], item[1], item[2]
                # Validate and clamp values
                if isinstance(level, int) and isinstance(title, str) and isinstance(page, int):
                    # Clamp level to 1-6 range
                    level = max(1, min(6, level))
                    # Ensure page is within valid range (some PDFs have invalid page refs)
                    page = max(1, min(total_pages, page)) if total_pages > 0 else max(1, page)
                    # Filter out entries with empty titles
                    title_stripped = title.strip()
                    if title_stripped:
                        entries.append(OutlineEntry(level=level, title=title_stripped, page=page))

        doc.close()
        logger.debug(
            "Extracted {} outline entries from PDF with {} pages",
            len(entries),
            total_pages,
        )
        return entries, total_pages

    except Exception:
        logger.error("Error extracting PDF outline")
        return [], 0


@router.get(
    "/{media_id:int}/outline",
    status_code=status.HTTP_200_OK,
    summary="Get Document Outline/Table of Contents",
    response_model=DocumentOutlineResponse,
    responses={
        200: {"description": "Outline retrieved (may be empty if document has no TOC or is not a PDF)"},
        404: {"description": "Media item not found"},
        413: {"description": "File too large for outline extraction (max 500MB)"},
        500: {"description": "Server error (database, storage, or extraction failure)"},
        501: {"description": "PyMuPDF dependency not installed on server"},
    },
)
async def get_document_outline(
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> DocumentOutlineResponse:
    """
    Extract and return the table of contents/outline from a document.

    Currently supports PDF files. Returns outline entries with heading level,
    title, and page number.

    ## Response Pattern (Graceful Degradation)

    This endpoint uses graceful degradation - it returns HTTP 200 with
    `has_outline=false` in cases where outline extraction is not possible
    but the request itself was valid:

    - **200 with outline**: PDF with embedded TOC
    - **200 with empty outline**: Any of:
      - PDF without embedded TOC
      - Non-PDF media type (video, audio, etc.)
      - File not stored or missing from storage
      - Non-PDF MIME type

    HTTP errors are reserved for actual failures:

    - **404**: Media ID does not exist
    - **413**: File exceeds 500MB size limit
    - **500**: Database error, storage error, or extraction crash
    - **501**: PyMuPDF library not installed on server

    ## Outline Entry Fields

    - `level`: Heading depth (1 = top-level chapter, 2-6 = sub-headings)
    - `title`: Text of the outline entry
    - `page`: 1-indexed page number the entry links to
    """
    logger.debug(
        "Fetching document outline for media_id={}, user_id={}",
        media_id,
        getattr(current_user, "id", "?"),
    )

    # 0. Check PyMuPDF availability early
    if not _check_pymupdf_available():
        logger.error("PyMuPDF not installed - cannot extract document outline")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="PDF outline extraction is not available. PyMuPDF is not installed.",
        )

    # 1. Verify media item exists
    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as e:
        logger.error("Database error fetching media item")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from e

    if not media:
        logger.warning(
            "Media not found for outline extraction: {} (user: {})",
            media_id,
            getattr(current_user, "id", "?"),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    # 2. Check media type - only PDFs supported for now
    media_type = media.get("type", "").lower()
    if media_type not in ("pdf", "document"):
        # For non-PDF types, return empty outline gracefully
        logger.debug(
            "Media type '{}' does not support outline extraction (media_id={})",
            media_type,
            media_id,
        )
        return DocumentOutlineResponse(
            media_id=media_id,
            has_outline=False,
            entries=[],
            total_pages=0,
        )

    # 3. Get original file from storage
    file_record = db.get_media_file(media_id, "original")
    if not file_record:
        logger.debug(
            "No original file found for outline extraction (media_id={})",
            media_id,
        )
        # Return empty outline if no file - document may have been processed without keeping original
        return DocumentOutlineResponse(
            media_id=media_id,
            has_outline=False,
            entries=[],
            total_pages=0,
        )

    storage_path = file_record.get("storage_path")
    mime_type = file_record.get("mime_type", "")

    if not storage_path:
        logger.error(
            "File record exists but storage_path is empty for media_id={}",
            media_id,
        )
        return DocumentOutlineResponse(
            media_id=media_id,
            has_outline=False,
            entries=[],
            total_pages=0,
        )

    # Verify it's a PDF - require application/pdf MIME type or .pdf extension
    is_pdf_mime = mime_type.lower() == "application/pdf"
    is_pdf_extension = storage_path.lower().endswith(".pdf")
    if not is_pdf_mime and not is_pdf_extension:
        logger.debug(
            "File is not a PDF for outline extraction (media_id={}, mime={})",
            media_id,
            mime_type,
        )
        return DocumentOutlineResponse(
            media_id=media_id,
            has_outline=False,
            entries=[],
            total_pages=0,
        )

    # 4. Retrieve file from storage
    storage = get_storage_backend()
    try:
        file_exists = await storage.exists(storage_path)
        if not file_exists:
            logger.warning(
                "File missing from storage for outline extraction: {} (media_id={})",
                storage_path,
                media_id,
            )
            return DocumentOutlineResponse(
                media_id=media_id,
                has_outline=False,
                entries=[],
                total_pages=0,
            )

        # Check file size before loading into memory
        try:
            file_size = await storage.get_size(storage_path)
            if file_size > MAX_OUTLINE_FILE_SIZE:
                logger.warning(
                    "File too large for outline extraction: {} bytes (max: {}) (media_id={})",
                    file_size,
                    MAX_OUTLINE_FILE_SIZE,
                    media_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large for outline extraction. Maximum size is {MAX_OUTLINE_FILE_SIZE // (1024 * 1024)} MB.",
                )
        except FileNotFoundError:
            # File was deleted between exists() and get_size() - rare race condition
            return DocumentOutlineResponse(
                media_id=media_id,
                has_outline=False,
                entries=[],
                total_pages=0,
            )

        # Read file into memory for PyMuPDF processing
        # storage.retrieve() returns a BinaryIO (file-like object)
        pdf_file = await storage.retrieve(storage_path)

    except StorageError as e:
        logger.error("Storage error retrieving file for outline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing file storage",
        ) from e

    # 5. Extract outline
    try:
        entries, total_pages = _extract_pdf_outline(pdf_file)

        return DocumentOutlineResponse(
            media_id=media_id,
            has_outline=len(entries) > 0,
            entries=entries,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error("Error extracting document outline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error extracting document outline",
        ) from e


__all__ = ["router"]
