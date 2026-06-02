# Reading Progress Endpoints
# Tracks and restores document reading position, zoom, and view mode
#
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.reading_progress import (
    ReadingProgressNotFound,
    ReadingProgressResponse,
    ReadingProgressUpdate,
    ViewMode,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)

router = APIRouter(tags=["Document Workspace"])


def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _log_missing_media_context(
    operation: str,
    media_id: int,
    user_id: str,
    db: Any,
) -> None:
    db_path = getattr(db, "db_path_str", getattr(db, "db_path", "<unknown>"))
    logger.warning(
        "Reading progress {} requested for missing media_id={} user_id={} db_path={}",
        operation,
        media_id,
        user_id,
        db_path,
    )


@router.get(
    "/{media_id:int}/progress",
    status_code=status.HTTP_200_OK,
    summary="Get Reading Progress",
    response_model=Union[ReadingProgressResponse, ReadingProgressNotFound],
    responses={
        200: {"description": "Reading progress retrieved (or none exists)"},
        404: {"description": "Media item not found"},
    },
)
async def get_reading_progress(
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> ReadingProgressResponse | ReadingProgressNotFound:
    """
    Get the reading progress for a document.

    Returns the saved reading position, zoom level, and view mode.
    If no progress exists, returns has_progress=false.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Getting reading progress for media_id={}, user_id={}",
        media_id,
        user_id,
    )

    # Verify media exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        _log_missing_media_context("get", media_id, user_id, db)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        row = repo.get_reading_progress(media_id=media_id, user_id=user_id)
    except Exception as e:
        logger.error("Error fetching reading progress")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch reading progress",
        ) from e

    if not row:
        return ReadingProgressNotFound(media_id=media_id, has_progress=False)

    try:
        row_dict = dict(row)
        current_page = row_dict["current_page"]
        total_pages = row_dict["total_pages"]
        # Use stored percentage if available (for EPUB), otherwise calculate from page
        stored_percentage = row_dict.get("percentage")
        percent_complete = (
            stored_percentage
            if stored_percentage is not None
            else (current_page / total_pages * 100) if total_pages > 0 else 0
        )

        return ReadingProgressResponse(
            media_id=media_id,
            current_page=current_page,
            total_pages=total_pages,
            zoom_level=row_dict["zoom_level"],
            view_mode=ViewMode(row_dict["view_mode"]),
            percent_complete=round(percent_complete, 1),
            cfi=row_dict.get("cfi"),
            last_read_at=datetime.fromisoformat(row_dict["last_read_at"]),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("Ignoring corrupt reading progress row")
        return ReadingProgressNotFound(media_id=media_id, has_progress=False)


@router.put(
    "/{media_id:int}/progress",
    status_code=status.HTTP_200_OK,
    summary="Update Reading Progress",
    response_model=ReadingProgressResponse,
    responses={
        200: {"description": "Reading progress updated successfully"},
        404: {"description": "Media item not found"},
    },
)
async def update_reading_progress(
    body: ReadingProgressUpdate,
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> ReadingProgressResponse:
    """
    Update reading progress for a document.

    Saves the current page, total pages, zoom level, and view mode.
    Creates a new record if none exists, otherwise updates the existing one.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Updating reading progress for media_id={}, user_id={}, page={}/{}",
        media_id,
        user_id,
        body.current_page,
        body.total_pages,
    )

    # Verify media exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        _log_missing_media_context("update", media_id, user_id, db)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    now = _now_iso()

    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        repo.upsert_reading_progress(
            media_id=media_id,
            user_id=user_id,
            current_page=body.current_page,
            total_pages=body.total_pages,
            zoom_level=body.zoom_level,
            view_mode=body.view_mode.value,
            cfi=body.cfi,
            percentage=body.percentage,
            last_read_at=now,
        )
    except Exception as e:
        logger.error("Error updating reading progress")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update reading progress",
        ) from e

    # Use stored percentage if provided (for EPUB), otherwise calculate from page
    percent_complete = (
        body.percentage
        if body.percentage is not None
        else (body.current_page / body.total_pages * 100) if body.total_pages > 0 else 0
    )

    return ReadingProgressResponse(
        media_id=media_id,
        current_page=body.current_page,
        total_pages=body.total_pages,
        zoom_level=body.zoom_level,
        view_mode=body.view_mode,
        percent_complete=round(percent_complete, 1),
        cfi=body.cfi,
        last_read_at=datetime.fromisoformat(now),
    )


@router.delete(
    "/{media_id:int}/progress",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete Reading Progress",
    responses={
        204: {"description": "Reading progress deleted successfully"},
        404: {"description": "Media item not found"},
    },
)
async def delete_reading_progress(
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> Response:
    """
    Delete reading progress for a document.

    Removes the saved reading position, allowing for a fresh start.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Deleting reading progress for media_id={}, user_id={}",
        media_id,
        user_id,
    )

    # Verify media exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        _log_missing_media_context("delete", media_id, user_id, db)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        repo.delete_reading_progress(media_id=media_id, user_id=user_id)
    except Exception as e:
        logger.error("Error deleting reading progress")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete reading progress",
        ) from e

    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
