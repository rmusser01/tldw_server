# Document Annotations CRUD Endpoints
# Manages annotations (highlights and notes) for PDF/EPUB documents
#
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.document_annotations import (
    AnnotationCreate,
    AnnotationListResponse,
    AnnotationResponse,
    AnnotationSyncRequest,
    AnnotationSyncResponse,
    AnnotationUpdate,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)

router = APIRouter(tags=["Document Workspace"])


def _generate_annotation_id() -> str:
    """Generate a unique annotation ID."""
    return f"ann_{uuid.uuid4().hex[:12]}"


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
        "Document annotations {} requested for missing media_id={} user_id={} db_path={}",
        operation,
        media_id,
        user_id,
        db_path,
    )


def _row_to_response(row: dict, media_id: int) -> AnnotationResponse:
    """Convert a database row to an AnnotationResponse."""
    return AnnotationResponse(
        id=row["id"],
        media_id=media_id,
        location=row["location"],
        text=row["text"],
        color=row["color"],
        note=row.get("note"),
        annotation_type=row.get("annotation_type", "highlight"),
        chapter_title=row.get("chapter_title"),
        percentage=row.get("percentage"),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


@router.get(
    "/{media_id:int}/annotations",
    status_code=status.HTTP_200_OK,
    summary="List Document Annotations",
    response_model=AnnotationListResponse,
    responses={
        200: {"description": "List of annotations for the document"},
        404: {"description": "Media item not found"},
    },
)
async def list_annotations(
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> AnnotationListResponse:
    """
    Retrieve all annotations for a document.

    Returns all highlights and notes created by the current user for the specified
    media item, sorted by creation date (newest first).
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Listing annotations for media_id={}, user_id={}",
        media_id,
        user_id,
    )

    # Verify media exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        _log_missing_media_context("list", media_id, user_id, db)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        rows = repo.list_annotations(media_id=media_id, user_id=user_id)
    except Exception as e:
        logger.error("Error fetching annotations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch annotations",
        ) from e

    annotations = [_row_to_response(dict(row), media_id) for row in rows]

    return AnnotationListResponse(
        media_id=media_id,
        annotations=annotations,
        total_count=len(annotations),
    )


@router.post(
    "/{media_id:int}/annotations",
    status_code=status.HTTP_201_CREATED,
    summary="Create Document Annotation",
    response_model=AnnotationResponse,
    responses={
        201: {"description": "Annotation created successfully"},
        404: {"description": "Media item not found"},
    },
)
async def create_annotation(
    body: AnnotationCreate,
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> AnnotationResponse:
    """
    Create a new annotation (highlight or page note) for a document.

    The annotation is associated with the specified media item and the current user.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Creating annotation for media_id={}, user_id={}, location={}",
        media_id,
        user_id,
        body.location,
    )

    # Verify media exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        _log_missing_media_context("create", media_id, user_id, db)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    annotation_id = _generate_annotation_id()
    now = _now_iso()

    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        row = repo.create_annotation(
            annotation_id=annotation_id,
            media_id=media_id,
            user_id=user_id,
            location=body.location,
            text=body.text,
            color=body.color.value,
            note=body.note,
            annotation_type=body.annotation_type.value,
            chapter_title=body.chapter_title,
            percentage=body.percentage,
            created_at=now,
            updated_at=now,
        )
    except Exception as e:
        logger.error("Error creating annotation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create annotation",
        ) from e

    return _row_to_response(row, media_id)


@router.put(
    "/{media_id:int}/annotations/{annotation_id}",
    status_code=status.HTTP_200_OK,
    summary="Update Document Annotation",
    response_model=AnnotationResponse,
    responses={
        200: {"description": "Annotation updated successfully"},
        404: {"description": "Annotation or media item not found"},
    },
)
async def update_annotation(
    body: AnnotationUpdate,
    media_id: int = Path(..., description="The ID of the media item"),
    annotation_id: str = Path(..., description="The ID of the annotation"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> AnnotationResponse:
    """
    Update an existing annotation.

    Only the fields provided in the request body will be updated.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Updating annotation {} for media_id={}, user_id={}",
        annotation_id,
        media_id,
        user_id,
    )

    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        row = repo.get_annotation(annotation_id=annotation_id, media_id=media_id, user_id=user_id)
    except Exception as e:
        logger.error("Error fetching annotation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch annotation",
        ) from e

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    row_dict = dict(row)
    if body.text is None and body.color is None and body.note is None:
        return _row_to_response(row_dict, media_id)

    now = _now_iso()
    try:
        updated = repo.update_annotation(
            annotation_id=annotation_id,
            media_id=media_id,
            user_id=user_id,
            text=body.text,
            color=body.color.value if body.color is not None else None,
            note=body.note,
            updated_at=now,
        )
    except Exception as e:
        logger.error("Error updating annotation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update annotation",
        ) from e

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return _row_to_response(updated, media_id)


@router.delete(
    "/{media_id:int}/annotations/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete Document Annotation",
    responses={
        204: {"description": "Annotation deleted successfully"},
        404: {"description": "Annotation not found"},
    },
)
async def delete_annotation(
    media_id: int = Path(..., description="The ID of the media item"),
    annotation_id: str = Path(..., description="The ID of the annotation"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> Response:
    """
    Delete an annotation (soft delete).

    The annotation is marked as deleted but retained in the database.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Deleting annotation {} for media_id={}, user_id={}",
        annotation_id,
        media_id,
        user_id,
    )

    now = _now_iso()
    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        deleted = repo.soft_delete_annotation(
            annotation_id=annotation_id,
            media_id=media_id,
            user_id=user_id,
            updated_at=now,
        )
    except Exception as e:
        logger.error("Error deleting annotation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete annotation",
        ) from e
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{media_id:int}/annotations/sync",
    status_code=status.HTTP_200_OK,
    summary="Sync Document Annotations",
    response_model=AnnotationSyncResponse,
    responses={
        200: {"description": "Annotations synced successfully"},
        404: {"description": "Media item not found"},
    },
)
async def sync_annotations(
    body: AnnotationSyncRequest,
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> AnnotationSyncResponse:
    """
    Batch sync annotations from client.

    This endpoint allows clients to send multiple annotations at once,
    useful for offline-first scenarios where changes are queued locally.

    If client_ids are provided, the response includes a mapping from
    client IDs to server-generated IDs.
    """
    user_id = str(getattr(current_user, "id", current_user))
    logger.debug(
        "Syncing {} annotations for media_id={}, user_id={}",
        len(body.annotations),
        media_id,
        user_id,
    )

    # Verify media exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        _log_missing_media_context("sync", media_id, user_id, db)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    annotation_rows: list[dict[str, Any]] = []
    id_mapping: dict = {}

    try:
        for i, annotation in enumerate(body.annotations):
            annotation_id = _generate_annotation_id()
            now = _now_iso()
            annotation_rows.append(
                {
                    "id": annotation_id,
                    "location": annotation.location,
                    "text": annotation.text,
                    "color": annotation.color.value,
                    "note": annotation.note,
                    "annotation_type": annotation.annotation_type.value,
                    "chapter_title": annotation.chapter_title,
                    "percentage": annotation.percentage,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            if body.client_ids and i < len(body.client_ids):
                id_mapping[body.client_ids[i]] = annotation_id

        repo = DocumentWorkspaceRepository.from_media_db(db)
        synced_rows = repo.sync_annotations(
            media_id=media_id,
            user_id=user_id,
            annotation_rows=annotation_rows,
        )

    except Exception as e:
        logger.error("Error syncing annotations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync annotations",
        ) from e

    synced_annotations = [_row_to_response(dict(row), media_id) for row in synced_rows]

    return AnnotationSyncResponse(
        media_id=media_id,
        synced_count=len(synced_annotations),
        annotations=synced_annotations,
        id_mapping=id_mapping if id_mapping else None,
    )


__all__ = ["router"]
