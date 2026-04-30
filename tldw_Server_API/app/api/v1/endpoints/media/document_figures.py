# Document Figures/Image Extraction Endpoint
# Extracts images from PDF documents using PyMuPDF
#
from __future__ import annotations

import base64
from typing import Any, BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.document_figures import (
    DocumentFiguresResponse,
    Figure,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Storage import get_storage_backend
from tldw_Server_API.app.core.Storage.storage_interface import StorageError

router = APIRouter(tags=["Document Workspace"])

# Maximum file size for figure extraction (500 MB)
MAX_FIGURES_FILE_SIZE = 500 * 1024 * 1024


def _check_pymupdf_available() -> bool:
    """Check if PyMuPDF is available for import."""
    try:
        import pymupdf  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_pdf_figures(
    pdf_data: bytes | BinaryIO,
    min_size: int = 50,
) -> list[Figure]:
    """
    Extract images/figures from PDF using PyMuPDF.

    Args:
        pdf_data: Raw PDF file content as bytes, or a file-like object (BinaryIO).
        min_size: Minimum width/height in pixels for images to include (filters icons/bullets).

    Returns:
        List of Figure objects containing image data.

    Raises:
        RuntimeError: If PyMuPDF is not installed
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed") from exc

    # Handle both bytes and file-like objects
    pdf_bytes = pdf_data.read() if hasattr(pdf_data, "read") else pdf_data

    figures: list[Figure] = []

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                image = None
                try:
                    image = doc.extract_image(xref)
                except Exception:
                    logger.debug(
                        "Skipping unreadable extracted image xref {} on page {}",
                        xref,
                        page_num + 1,
                    )
                if image is None:
                    continue

                # Filter out small images (icons, bullets, etc.)
                if image["width"] < min_size or image["height"] < min_size:
                    continue

                img_bytes = image["image"]
                ext = image["ext"]

                # Convert to base64 data URL
                b64_data = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                data_url = f"data:{mime_type};base64,{b64_data}"

                figures.append(
                    Figure(
                        id=f"fig_{page_num + 1}_{img_index}",
                        page=page_num + 1,  # 1-indexed
                        width=image["width"],
                        height=image["height"],
                        format=ext,
                        data_url=data_url,
                    )
                )

        doc.close()
        logger.debug(
            "Extracted {} figures from PDF with {} pages",
            len(figures),
            doc.page_count,
        )
        return figures

    except Exception:
        logger.error("Error extracting PDF figures")
        return []


@router.get(
    "/{media_id:int}/figures",
    status_code=status.HTTP_200_OK,
    summary="Get Document Figures/Images",
    response_model=DocumentFiguresResponse,
    responses={
        200: {"description": "Figures retrieved (may be empty if document has no images or is not a PDF)"},
        404: {"description": "Media item not found"},
        413: {"description": "File too large for figure extraction (max 500MB)"},
        500: {"description": "Server error (database, storage, or extraction failure)"},
        501: {"description": "PyMuPDF dependency not installed on server"},
    },
)
async def get_document_figures(
    media_id: int = Path(..., description="The ID of the media item"),
    min_size: int = Query(50, ge=10, le=500, description="Minimum image size in pixels to include"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> DocumentFiguresResponse:
    """
    Extract and return figures/images from a PDF document.

    ## Features

    - Extracts embedded images from PDF pages
    - Filters out small icons and bullets (configurable min_size)
    - Returns images as base64 data URLs
    - Includes page number for navigation

    ## Response Pattern (Graceful Degradation)

    Returns HTTP 200 with `has_figures=false` when extraction is not possible
    but the request itself was valid:

    - **200 with figures**: PDF with extractable images
    - **200 with empty figures**: Any of:
      - PDF without embedded images
      - Non-PDF media type
      - File not stored or missing

    HTTP errors are reserved for actual failures:

    - **404**: Media ID does not exist
    - **413**: File exceeds 500MB size limit
    - **500**: Database error, storage error, or extraction crash
    - **501**: PyMuPDF library not installed on server
    """
    logger.debug(
        "Fetching document figures for media_id={}, user_id={}",
        media_id,
        getattr(current_user, "id", "?"),
    )

    # 0. Check PyMuPDF availability early
    if not _check_pymupdf_available():
        logger.error("PyMuPDF not installed - cannot extract document figures")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Figure extraction is not available. PyMuPDF is not installed.",
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
            "Media not found for figure extraction: {} (user: {})",
            media_id,
            getattr(current_user, "id", "?"),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    # 2. Check media type - only PDFs supported
    media_type = media.get("type", "").lower()
    if media_type not in ("pdf", "document"):
        # Non-PDF types return empty gracefully
        logger.debug(
            "Media type '{}' does not support figure extraction (media_id={})",
            media_type,
            media_id,
        )
        return DocumentFiguresResponse(
            media_id=media_id,
            has_figures=False,
            figures=[],
            total_count=0,
        )

    # 3. Get original file from storage
    file_record = db.get_media_file(media_id, "original")
    if not file_record:
        logger.debug(
            "No original file found for figure extraction (media_id={})",
            media_id,
        )
        return DocumentFiguresResponse(
            media_id=media_id,
            has_figures=False,
            figures=[],
            total_count=0,
        )

    storage_path = file_record.get("storage_path")
    mime_type = file_record.get("mime_type", "")

    if not storage_path:
        logger.error(
            "File record exists but storage_path is empty for media_id={}",
            media_id,
        )
        return DocumentFiguresResponse(
            media_id=media_id,
            has_figures=False,
            figures=[],
            total_count=0,
        )

    # Verify it's a PDF
    is_pdf_mime = mime_type.lower() == "application/pdf"
    is_pdf_extension = storage_path.lower().endswith(".pdf")
    if not is_pdf_mime and not is_pdf_extension:
        logger.debug(
            "File is not a PDF for figure extraction (media_id={}, mime={})",
            media_id,
            mime_type,
        )
        return DocumentFiguresResponse(
            media_id=media_id,
            has_figures=False,
            figures=[],
            total_count=0,
        )

    # 4. Retrieve file from storage
    storage = get_storage_backend()
    try:
        file_exists = await storage.exists(storage_path)
        if not file_exists:
            logger.warning(
                "File missing from storage for figure extraction: {} (media_id={})",
                storage_path,
                media_id,
            )
            return DocumentFiguresResponse(
                media_id=media_id,
                has_figures=False,
                figures=[],
                total_count=0,
            )

        # Check file size
        try:
            file_size = await storage.get_size(storage_path)
            if file_size > MAX_FIGURES_FILE_SIZE:
                logger.warning(
                    "File too large for figure extraction: {} bytes (max: {}) (media_id={})",
                    file_size,
                    MAX_FIGURES_FILE_SIZE,
                    media_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large for figure extraction. Maximum size is {MAX_FIGURES_FILE_SIZE // (1024 * 1024)} MB.",
                )
        except FileNotFoundError:
            # Race condition - file deleted between exists() and get_size()
            return DocumentFiguresResponse(
                media_id=media_id,
                has_figures=False,
                figures=[],
                total_count=0,
            )

        # Read file
        pdf_file = await storage.retrieve(storage_path)

    except StorageError as e:
        logger.error("Storage error retrieving file for figures")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing file storage",
        ) from e

    # 5. Extract figures
    try:
        figures = _extract_pdf_figures(pdf_file, min_size)

        return DocumentFiguresResponse(
            media_id=media_id,
            has_figures=len(figures) > 0,
            figures=figures,
            total_count=len(figures),
        )

    except Exception as e:
        logger.error("Error extracting document figures")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error extracting document figures",
        ) from e


__all__ = ["router"]
