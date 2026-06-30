from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    MediaKeywordsUpdateRequest,
    MediaUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.media_response_models import (
    MediaDetailResponse,
    MediaKeywordsResponse,
)
from tldw_Server_API.app.api.v1.utils.cache import generate_etag, is_not_modified
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.utils.rag_cache import (
    delete_media_vectors,
    invalidate_rag_caches,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_DELETE, MEDIA_UPDATE
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    fetch_keywords_for_media,
    get_full_media_details_rich,
    get_media_by_id,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)

from .....core.DB_Management.media_db.legacy_maintenance import (
    permanently_delete_item,
)

router = APIRouter(tags=["Media Management"])


def _is_test_mode() -> bool:
    try:
        from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode_impl

        return bool(_is_test_mode_impl())
    except Exception:
        return False


@router.get(
    "/{media_id:int}",
    status_code=status.HTTP_200_OK,
    summary="Get Media Item Details",
    responses={
        status.HTTP_304_NOT_MODIFIED: {
            "description": "Media item not modified (ETag match).",
        },
    },
)
async def get_media_item(
    request: Request,
    response: Response,
    media_id: int = Path(..., description="The ID of the media item"),
    include_content: bool = Query(
        True,
        description="Include main content text in response",
    ),
    include_versions: bool = Query(
        True,
        description="Include versions list",
    ),
    include_version_content: bool = Query(
        False,
        description="Include content for each version in versions list",
    ),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
    if_none_match: str | None = Header(None),
) -> Any:
    """
    Retrieve Media Item by ID.

    Fetches the details for a specific active media item, including
    its associated keywords, latest prompt/analysis, and versions.
    """
    logger.debug(
        "Attempting to fetch rich details for media_id: {}",
        media_id,
    )

    # TEST_MODE diagnostics
    try:
        if _is_test_mode():
            db_path = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
            headers = getattr(request, "headers", {}) or {}
            logger.info(
                "TEST_MODE: get_media_item id={} db_path={} user_id={} "
                "auth_headers={{'X-API-KEY': {{'present': {}}}}, 'Authorization': {{'present': {}}}}}",
                media_id,
                db_path,
                getattr(current_user, "id", "?"),
                bool(headers.get("X-API-KEY")),
                bool(headers.get("authorization")),
            )
    except Exception:
        logger.debug("Failed to emit media item auth header diagnostics")

    try:
        details = get_full_media_details_rich(
            db,
            media_id=media_id,
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )
        if not details:
            logger.warning(
                "Media not found or not active for ID: {}",
                media_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or is inactive/trashed",
            )

        response_model = MediaDetailResponse(**details)
        payload = response_model.model_dump()

        etag = generate_etag(payload)
        response.headers["ETag"] = etag
        if is_not_modified(etag, if_none_match):
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return {}

        return payload
    except HTTPException:
        raise
    except (DatabaseError, InputError, ConflictError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Database error retrieving media details",
            input_detail="Invalid media identifier",
            conflict_detail="Conflict detected while retrieving media details",
            log_context=f"get_media_item media_id={media_id}",
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error fetching details for media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred retrieving media details",
        ) from exc


@router.delete(
    "/{media_id:int}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Move media item to trash",
    description="Soft-delete a media item by moving it to trash (is_trash=1). Use POST /{media_id}/restore to undo.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Media not found or already deleted"},
        status.HTTP_409_CONFLICT: {"description": "Media could not be moved to trash"},
    },
    dependencies=[
        Depends(RequirePermission(MEDIA_DELETE)),
        Depends(rbac_rate_limit("media.delete")),
    ],
)
async def delete_media_item(
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> Response:
    """
    Soft-delete a media item by moving it to trash (is_trash=1).
    """
    try:
        existing = get_media_by_id(
            db,
            media_id,
            include_deleted=False,
            include_trash=True,
        )
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or already deleted",
            )
        if existing.get("is_trash"):
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        success = db.mark_as_trash(media_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Media could not be moved to trash",
            )
        invalidate_rag_caches(current_user, media_id=media_id)
        await delete_media_vectors(current_user, media_id=media_id)
        logger.info(
            "User {} moved media {} to trash",
            getattr(current_user, "id", "?"),
            media_id,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Database error moving media to trash",
            input_detail="Invalid media identifier",
            conflict_detail="Media was modified concurrently",
            log_context=f"delete_media_item media_id={media_id}",
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error trashing media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error moving media to trash",
        ) from exc


@router.post(
    "/{media_id:int}/restore",
    status_code=status.HTTP_200_OK,
    summary="Restore a media item from trash",
    response_model=MediaDetailResponse,
    description="Restore a trashed media item (is_trash=0) and return its details.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Media not found or already deleted"},
        status.HTTP_409_CONFLICT: {"description": "Media could not be restored from trash"},
    },
    dependencies=[
        Depends(RequirePermission(MEDIA_DELETE)),
        Depends(rbac_rate_limit("media.delete")),
    ],
)
async def restore_media_item(
    media_id: int = Path(..., description="The ID of the media item"),
    include_content: bool = Query(
        True,
        description="Include main content text in response",
    ),
    include_versions: bool = Query(
        True,
        description="Include versions list",
    ),
    include_version_content: bool = Query(
        False,
        description="Include content for each version in versions list",
    ),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> Any:
    """
    Restore a trashed media item (is_trash=0) and return its details.
    """
    try:
        existing = get_media_by_id(
            db,
            media_id,
            include_deleted=False,
            include_trash=True,
        )
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or already deleted",
            )
        if existing.get("is_trash"):
            success = db.restore_from_trash(media_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Media could not be restored from trash",
                )
            logger.info(
                "User {} restored media {} from trash",
                getattr(current_user, "id", "?"),
                media_id,
            )
        details = get_full_media_details_rich(
            db,
            media_id=media_id,
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or is inactive/trashed",
            )
        return MediaDetailResponse(**details)
    except HTTPException:
        raise
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Database error restoring media from trash",
            input_detail="Invalid media identifier",
            conflict_detail="Media was modified concurrently",
            log_context=f"restore_media_item media_id={media_id}",
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error restoring media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error restoring media from trash",
        ) from exc


@router.delete(
    "/{media_id:int}/permanent",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Permanently delete a trashed media item",
    description="Hard-delete a trashed media item. This cannot be undone.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Media not found or already deleted"},
        status.HTTP_409_CONFLICT: {"description": "Media is not in trash"},
    },
    dependencies=[
        Depends(RequirePermission(MEDIA_DELETE)),
        Depends(rbac_rate_limit("media.delete")),
    ],
)
async def permanently_delete_media_item(
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> Response:
    """
    Permanently delete a trashed media item.
    """
    try:
        existing = get_media_by_id(
            db,
            media_id,
            include_deleted=False,
            include_trash=True,
        )
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or already deleted",
            )
        if not existing.get("is_trash"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Media must be in trash before permanent delete",
            )
        deleted = permanently_delete_item(db, media_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or already deleted",
            )
        invalidate_rag_caches(current_user, media_id=media_id)
        await delete_media_vectors(current_user, media_id=media_id)
        logger.warning(
            "User {} permanently deleted media {}",
            getattr(current_user, "id", "?"),
            media_id,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Database error permanently deleting media",
            input_detail="Invalid media identifier",
            conflict_detail="Media was modified concurrently",
            log_context=f"permanently_delete_media_item media_id={media_id}",
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error permanently deleting media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error permanently deleting media",
        ) from exc


@router.put(
    "/{media_id:int}",
    tags=["Media Management"],
    summary="Update Media Item",
    status_code=status.HTTP_200_OK,
    response_model=MediaDetailResponse,
)
async def update_media_item(
    payload: MediaUpdateRequest,
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MediaDetailResponse:
    """
    Update Media Item Details.

    Modifies attributes of an active main media item record (for example, title or author).

    When ``content`` is updated:
      - A new document version is created using the provided ``payload.content``.
      - ``payload.prompt`` and ``payload.analysis`` (when provided) are stored on the new version.
      - The main ``Media`` record's ``content``, ``content_hash``, ``last_modified``,
        and ``version`` fields are updated.
      - FTS index for the media item is updated.

    When only non-content fields are updated:
      - Only the main ``Media`` record is updated (including ``last_modified`` and ``version``).
      - FTS index is updated when the title changes.
    """
    logger.debug(
        "Received request to update media_id={} with payload: {}",
        media_id,
        payload.model_dump(exclude_unset=True),
    )

    update_fields: dict[str, Any] = payload.model_dump(
        exclude_unset=True,
        exclude={"prompt", "analysis", "keywords"},
    )

    # No-op update: return current representation if the item exists,
    # matching the legacy handler's behaviour.
    if not update_fields:
        logger.info(
            "Update request for media {} received with no fields to update.",
            media_id,
        )
        current_data = get_media_by_id(
            db,
            media_id,
            include_deleted=False,
            include_trash=False,
        )
        if not current_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media item not found or inactive.",
            )
        # Use the rich detail view for consistency with normal responses.
        details = get_full_media_details_rich(
            db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after update",
            )
        invalidate_rag_caches(current_user, media_id=media_id)
        return MediaDetailResponse(**details)

    try:
        effects = db.apply_media_item_update(
            media_id=media_id,
            fields=update_fields,
            prompt=payload.prompt,
            analysis_content=payload.analysis,
        )

        details = get_full_media_details_rich(
            db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after update",
            )
        if effects.get("invalidate_rag", True):
            invalidate_rag_caches(current_user, media_id=media_id)
        return MediaDetailResponse(**details)
    except HTTPException:
        raise
    except InputError as exc:
        raise map_db_error_to_http(
            exc,
            input_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            default_detail="Database error during update",
            input_detail="Database error during update",
            conflict_detail="Conflict detected during update",
            log_context=f"update_media_item media_id={media_id}",
            not_found_substrings=("not found or inactive/trashed",),
            not_found_detail="Media item not found or is inactive/trashed",
        ) from exc
    except (ConflictError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            input_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            default_detail="Database error during update",
            input_detail="Database error during update",
            conflict_detail="Conflict detected during update",
            log_context=f"update_media_item media_id={media_id}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error updating media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from exc


@router.patch(
    "/{media_id:int}/keywords",
    response_model=MediaKeywordsResponse,
    summary="Update media keywords (add/remove/set)",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("media.update")),
    ],
)
async def update_media_keywords(
    payload: MediaKeywordsUpdateRequest,
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    _current_user: User = Depends(get_request_user),
) -> MediaKeywordsResponse:
    """Update media keywords without altering other media fields."""
    mode = payload.mode
    target_keywords = [k.strip() for k in payload.keywords if k and k.strip()]
    try:
        current_keywords = fetch_keywords_for_media(db, media_id)
        if mode == "set":
            desired = target_keywords
        elif mode == "remove":
            to_remove = {k.lower() for k in target_keywords}
            desired = [k for k in current_keywords if k.lower() not in to_remove]
        else:
            # add (default)
            existing = {k.lower() for k in current_keywords}
            desired = current_keywords + [k for k in target_keywords if k.lower() not in existing]
        db.update_keywords_for_media(media_id=media_id, keywords=desired)
        updated_keywords = fetch_keywords_for_media(db, media_id)
        return MediaKeywordsResponse(media_id=media_id, keywords=updated_keywords)
    except HTTPException:
        raise
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            input_status=status.HTTP_404_NOT_FOUND,
            default_detail="Failed to update keywords",
            input_detail="Media not found or deleted",
            conflict_detail="Conflict detected updating keywords",
            log_context=f"update_media_keywords media_id={media_id}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error updating keywords for media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update keywords",
        ) from exc


__all__ = ["router"]
