from __future__ import annotations

import contextlib
import json
from datetime import datetime
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    AdvancedVersionUpsertRequest,
    MetadataPatchRequest,
    VersionCreateRequest,
    VersionRollbackRequest,
)
from tldw_Server_API.app.api.v1.schemas.media_response_models import (
    MediaDetailResponse,
    VersionDetailResponse,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    check_media_exists,
    get_document_version,
    get_full_media_details_rich,
    list_document_versions,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.Utils.metadata_utils import (
    normalize_safe_metadata,
    update_version_safe_metadata_in_transaction,
)

router = APIRouter(tags=["Media Versioning"])

_MEDIA_VERSIONS_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_MEDIA_VERSIONS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _is_test_mode() -> bool:
    try:
        from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode_impl

        return bool(_is_test_mode_impl())
    except _MEDIA_VERSIONS_NONCRITICAL_EXCEPTIONS:
        return False


@router.get(
    "/{media_id:int}/versions",
    summary="List Media Versions",
    response_model=list[VersionDetailResponse],
    response_model_exclude_none=True,
)
async def list_versions(
    media_id: int = Path(..., description="The ID of the media item"),
    include_content: bool = Query(
        False,
        description="Include full content in response",
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100,
        description="Results per page",
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number",
    ),
    db: Any = Depends(get_media_db_for_user),
) -> list[VersionDetailResponse]:
    """
    List active versions for an active media item.
    """
    logger.debug(
        "Listing versions for media_id: {} (Page: {}, Limit: {}, Content: {})",
        media_id,
        page,
        limit,
        include_content,
    )

    offset = (page - 1) * limit

    try:
        media_exists = check_media_exists(db, media_id=media_id)
        if not media_exists:
            logger.warning(
                "Cannot list versions: Media ID {} not found or deleted.",
                media_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media item not found or deleted",
            )

        raw_rows = list_document_versions(
            db,
            media_id,
            include_content=include_content,
            include_deleted=False,
            limit=limit,
            offset=offset,
        )

        versions: list[VersionDetailResponse] = []
        for rv in raw_rows:
            created_at_dt: datetime | None = rv.get("created_at")
            if isinstance(created_at_dt, str):
                with contextlib.suppress(_MEDIA_VERSIONS_COERCE_EXCEPTIONS):
                    created_at_dt = datetime.fromisoformat(created_at_dt.replace("Z", "+00:00"))
            safe_md = rv.get("safe_metadata")
            if isinstance(safe_md, str):
                import json as _json

                try:
                    safe_md = _json.loads(safe_md)
                except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
                    safe_md = None
            versions.append(
                VersionDetailResponse(
                    uuid=rv.get("uuid"),
                    media_id=rv.get("media_id"),
                    version_number=rv.get("version_number"),
                    created_at=created_at_dt,
                    prompt=rv.get("prompt"),
                    analysis_content=rv.get("analysis_content"),
                    safe_metadata=safe_md,
                    content=rv.get("content") if include_content else None,
                )
            )

        return versions
    except DatabaseError as exc:
        logger.error(
            "Database error listing versions for media {}",
            media_id,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error occurred",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error listing versions for media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error listing versions",
        ) from exc


@router.get(
    "/{media_id:int}/versions/{version_number:int}",
    summary="Get Specific Media Version",
    response_model=VersionDetailResponse,
    response_model_exclude_none=True,
)
async def get_version(
    media_id: int = Path(..., description="The ID of the media item"),
    version_number: int = Path(..., description="The version number"),
    include_content: bool = Query(
        True,
        description="Include full content in response",
    ),
    db: Any = Depends(get_media_db_for_user),
) -> VersionDetailResponse:
    """
    Get details of a specific active version for an active media item.
    """
    logger.debug(
        "Getting version {} for media_id: {} (Content: {})",
        version_number,
        media_id,
        include_content,
    )
    try:
        version_dict = get_document_version(
            db,
            media_id=media_id,
            version_number=version_number,
            include_content=include_content,
        )

        if version_dict is None:
            logger.warning(
                "Active version {} not found for active media {}",
                version_number,
                media_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found or media/version is inactive",
            )

        created_at_dt = version_dict.get("created_at")
        if isinstance(created_at_dt, str):
            with contextlib.suppress(_MEDIA_VERSIONS_COERCE_EXCEPTIONS):
                created_at_dt = datetime.fromisoformat(created_at_dt.replace("Z", "+00:00"))
        safe_md = version_dict.get("safe_metadata")
        if isinstance(safe_md, str):
            import json as _json

            try:
                safe_md = _json.loads(safe_md)
            except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
                safe_md = None

        return VersionDetailResponse(
            uuid=version_dict.get("uuid"),
            media_id=version_dict.get("media_id"),
            version_number=version_dict.get("version_number"),
            created_at=created_at_dt,
            prompt=version_dict.get("prompt"),
            analysis_content=version_dict.get("analysis_content"),
            safe_metadata=safe_md,
            content=version_dict.get("content") if include_content else None,
        )
    except ValueError as exc:
        logger.warning(
            "Invalid input for get_document_version: {}",
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request parameters",
        ) from exc
    except DatabaseError as exc:
        logger.error(
            "Database error getting version {} for media {}",
            version_number,
            media_id,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error occurred",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error getting version {} for media {}",
            version_number,
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting version",
        ) from exc


@router.post(
    "/{media_id:int}/versions",
    summary="Create Media Version",
    status_code=status.HTTP_201_CREATED,
    response_model=MediaDetailResponse,
)
async def create_version(
    media_id: int,
    request_body: VersionCreateRequest,
    request: Request,
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MediaDetailResponse:
    """
    Create a new document version for an active media item.
    """
    logger.debug("Attempting to create version for media_id: {}", media_id)

    # TEST_MODE diagnostics (best-effort; non-fatal if anything fails)
    try:
        if _is_test_mode():
            db_path = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
            headers: dict[str, Any] = getattr(request, "headers", {}) or {}
            logger.info(
                "TEST_MODE: create_version media_id={} db_path={} user_id={} "
                "auth_headers={{'X-API-KEY': {{'present': {}}}}, "
                "'Authorization': {{'present': {}}}}}",
                media_id,
                db_path,
                getattr(current_user, "id", "?"),
                bool(headers.get("X-API-KEY")),
                bool(headers.get("authorization")),
            )
    except _MEDIA_VERSIONS_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - diagnostics only
        pass

    try:
        import json as _json

        media_exists = check_media_exists(db, media_id=media_id)
        if not media_exists:
            logger.warning(
                "Cannot create version: Media ID {} not found or deleted.",
                media_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or deleted",
            )

        with db.transaction():
            safe_metadata_json: str | None = None
            if request_body.safe_metadata is not None:
                try:
                    safe_metadata_json = _json.dumps(
                        request_body.safe_metadata,
                        ensure_ascii=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "Invalid safe_metadata for media {} on create_version: {}",
                        media_id,
                        exc,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="safe_metadata is not JSON-serializable",
                    ) from exc

            result_dict = db.create_document_version(
                media_id=media_id,
                content=request_body.content,
                prompt=request_body.prompt,
                analysis_content=request_body.analysis_content,
                safe_metadata=safe_metadata_json,
            )

        logger.info(
            "Successfully created version {} (UUID: {}) for media_id: {}",
            result_dict.get("version_number"),
            result_dict.get("uuid"),
            media_id,
        )

        details = get_full_media_details_rich(
            db=db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after version creation",
            )
        return MediaDetailResponse(**details)
    except InputError as exc:
        logger.warning(
            "Cannot create version for media {}: {}",
            media_id,
            exc,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error occurred",
        ) from exc
    except (DatabaseError, ConflictError) as exc:
        logger.error(
            "Database error creating version for media {}",
            media_id,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error occurred",
            conflict_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            conflict_detail="Database error occurred",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Unexpected error creating version for media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during version creation",
        ) from exc


@router.delete(
    "/{media_id:int}/versions/{version_number:int}",
    summary="Soft Delete Media Version",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_version(
    media_id: int = Path(..., description="The ID of the media item"),
    version_number: int = Path(..., description="The version number"),
    db: Any = Depends(get_media_db_for_user),
) -> Response:
    """
    Soft delete a specific active version for an active media item.
    """
    logger.debug(
        "Attempting to soft delete version {} for media_id: {}",
        version_number,
        media_id,
    )
    try:
        query_uuid = """
            SELECT dv.uuid
            FROM DocumentVersions dv
            JOIN Media m ON dv.media_id = m.id
            WHERE dv.media_id = ?
              AND dv.version_number = ?
              AND dv.deleted = 0
              AND m.deleted = 0
              AND m.is_trash = 0
        """
        cursor = db.execute_query(query_uuid, (media_id, version_number))
        result_uuid = cursor.fetchone()

        if not result_uuid:
            logger.warning(
                "Active version {} for active media {} not found.",
                version_number,
                media_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Active media or specific active version not found.",
            )

        version_uuid = result_uuid["uuid"]
        logger.debug(
            "Found UUID {} for version {} of media {}",
            version_uuid,
            version_number,
            media_id,
        )

        with db.transaction():
            success = db.soft_delete_document_version(version_uuid=version_uuid)

        if success:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        logger.warning(
            "Failed to delete version {} (UUID: {}) for media {} - likely the last active version.",
            version_number,
            version_uuid,
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the only active version of the document.",
        )
    except ConflictError as exc:
        logger.error(
            "Conflict deleting version {} (UUID: {}) for media {}: {}",
            version_number,
            locals().get("version_uuid"),
            media_id,
            exc,
            exc_info=True,
        )
        raise map_db_error_to_http(
            exc,
            conflict_detail="Conflict during deletion",
        ) from exc
    except InputError as exc:
        logger.error(
            "Input error deleting version {} for media {}: {}",
            version_number,
            media_id,
            exc,
            exc_info=True,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error occurred",
        ) from exc
    except DatabaseError as exc:
        logger.error(
            "Database error deleting version {} for media {}",
            version_number,
            media_id,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error occurred",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Unexpected error deleting version {} for media {}",
            version_number,
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error deleting version",
        ) from exc


@router.post(
    "/{media_id:int}/versions/rollback",
    summary="Rollback to Media Version",
    response_model=MediaDetailResponse,
)
async def rollback_version(
    media_id: int,
    request_body: VersionRollbackRequest,
    db: Any = Depends(get_media_db_for_user),
) -> MediaDetailResponse:
    """
    Roll back an active media item to a specified previous version.
    """
    target_version_number = request_body.version_number
    logger.debug(
        "Attempting to rollback media_id {} to version {}",
        media_id,
        target_version_number,
    )
    try:
        with db.transaction():
            rollback_result = db.rollback_to_version(
                media_id=media_id,
                target_version_number=target_version_number,
            )

        if "error" in rollback_result:
            error_msg = rollback_result["error"]
            logger.warning(
                "Rollback failed for media {} to version {}: {}",
                media_id,
                target_version_number,
                error_msg,
            )
            lower_msg = error_msg.lower()
            if "not found" in lower_msg:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=error_msg,
                )
            if "cannot rollback to the current latest version" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )

        logger.info(
            "Rollback successful for media {} to version {}. New doc version: {}",
            media_id,
            target_version_number,
            rollback_result.get("new_document_version_number"),
        )
        details = get_full_media_details_rich(
            db=db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after rollback",
            )
        return MediaDetailResponse(**details)
    except InputError as exc:
        logger.warning(
            "Invalid DB input for rollback media {}: {}",
            media_id,
            exc,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error during rollback",
        ) from exc
    except ValueError as exc:
        logger.warning(
            "Invalid input for rollback media {}: {}",
            media_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request parameters",
        ) from exc
    except ConflictError as exc:
        logger.error(
            "Conflict rolling back media {} to version {}: {}",
            media_id,
            target_version_number,
            exc,
            exc_info=True,
        )
        raise map_db_error_to_http(
            exc,
            conflict_detail="Conflict during rollback",
        ) from exc
    except DatabaseError as exc:
        logger.error(
            "Database error rolling back media {} to version {}",
            media_id,
            target_version_number,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Database error during rollback",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Unexpected error rolling back media {} to version {}",
            media_id,
            target_version_number,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during rollback",
        ) from exc


@router.patch(
    "/{media_id:int}/metadata",
    tags=["Media Management"],
    summary="Update safe metadata for the latest version",
    response_model=MediaDetailResponse,
)
async def patch_metadata(
    media_id: int = Path(..., description="The ID of the media item"),
    body: MetadataPatchRequest = Body(...),
    db: Any = Depends(get_media_db_for_user),
) -> MediaDetailResponse:
    """
    Update safe_metadata on the latest active version or create a new version.
    """
    import json as _json

    try:
        try:
            normalized = normalize_safe_metadata(body.safe_metadata or {})
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        latest = get_document_version(
            db,
            media_id=media_id,
            version_number=None,
            include_content=True,
        )
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active version found for this media.",
            )

        existing = latest.get("safe_metadata")
        if isinstance(existing, str):
            try:
                existing = _json.loads(existing)
            except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
                existing = None
        if not isinstance(existing, dict):
            existing = {}

        new_meta: dict[str, Any] = dict(existing)
        if body.merge:
            new_meta.update(normalized)
        else:
            new_meta = dict(normalized)

        try:
            new_meta_json = _json.dumps(new_meta, ensure_ascii=False)
        except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="safe_metadata is not JSON-serializable",
            ) from None

        if body.new_version:
            with db.transaction():
                db.create_document_version(
                    media_id=media_id,
                    content=latest.get("content") or "",
                    prompt=latest.get("prompt"),
                    analysis_content=latest.get("analysis_content"),
                    safe_metadata=new_meta_json,
                )
        else:
            dv_id = latest.get("id")
            if not dv_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Latest version record missing identifier",
                )
            with db.transaction() as conn:
                update_version_safe_metadata_in_transaction(
                    db=db,
                    dv_id=dv_id,
                    safe_metadata_json=new_meta_json,
                    merged_metadata=new_meta,
                    connection=conn,
                )

        details = get_full_media_details_rich(
            db=db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after metadata update",
            )
        return MediaDetailResponse(**details)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Error patching safe metadata for media {}",
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update metadata",
        ) from exc


@router.put(
    "/{media_id:int}/versions/{version_number:int}/metadata",
    summary="Set safe metadata for a specific version",
    response_model=MediaDetailResponse,
)
async def put_version_metadata(
    media_id: int = Path(..., description="The ID of the media item"),
    version_number: int = Path(..., description="The version number"),
    body: MetadataPatchRequest = Body(...),
    db: Any = Depends(get_media_db_for_user),
) -> MediaDetailResponse:
    """
    Set or merge safe_metadata JSON on a specific active version.
    """
    import json as _json

    try:
        try:
            normalized = normalize_safe_metadata(body.safe_metadata or {})
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        version_dict = get_document_version(
            db,
            media_id=media_id,
            version_number=version_number,
            include_content=False,
        )
        if not version_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found",
            )

        dv_id = version_dict.get("id")
        if not dv_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version ID missing for metadata update",
            )
        existing = version_dict.get("safe_metadata")
        if isinstance(existing, str):
            try:
                existing = _json.loads(existing)
            except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
                existing = None
        if not isinstance(existing, dict):
            existing = {}

        new_meta: dict[str, Any] = dict(existing)
        if body.merge:
            new_meta.update(normalized)
        else:
            new_meta = dict(normalized)

        try:
            smj = _json.dumps(new_meta, ensure_ascii=False)
        except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="safe_metadata is not JSON-serializable",
            ) from None

        with db.transaction() as conn:
            update_version_safe_metadata_in_transaction(
                db=db,
                dv_id=dv_id,
                safe_metadata_json=smj,
                merged_metadata=new_meta,
                connection=conn,
            )

        details = get_full_media_details_rich(
            db=db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after metadata update",
            )
        return MediaDetailResponse(**details)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Error updating metadata for media {} v{}",
            media_id,
            version_number,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update version metadata",
        ) from exc


@router.post(
    "/{media_id:int}/versions/advanced",
    summary="Create or update version with content + safe metadata",
    response_model=MediaDetailResponse,
)
async def create_or_update_version_advanced(
    media_id: int,
    body: AdvancedVersionUpsertRequest,
    db: Any = Depends(get_media_db_for_user),
) -> MediaDetailResponse:
    """
    Convenience endpoint to create a new version or update latest metadata.
    """
    import json as _json

    try:
        normalized: dict[str, Any] | None = None
        if body.safe_metadata is not None:
            try:
                normalized = normalize_safe_metadata(body.safe_metadata)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc

        latest = get_document_version(
            db,
            media_id=media_id,
            version_number=None,
            include_content=True,
        )
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active version found for this media.",
            )

        if not body.new_version and (
            body.content is not None or body.prompt is not None or body.analysis_content is not None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="When new_version=false, only safe_metadata updates are allowed",
            )

        latest_sm = latest.get("safe_metadata")
        if isinstance(latest_sm, str):
            try:
                latest_sm = _json.loads(latest_sm)
            except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
                latest_sm = None
        if not isinstance(latest_sm, dict):
            latest_sm = {}

        if body.safe_metadata is not None:
            if normalized is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="safe_metadata normalization failed",
                )
            if body.merge:
                merged_sm = dict(latest_sm)
                merged_sm.update(normalized)
            else:
                merged_sm = dict(normalized)
        else:
            merged_sm = dict(latest_sm)

        try:
            smj = _json.dumps(merged_sm, ensure_ascii=False) if merged_sm else None
        except _MEDIA_VERSIONS_COERCE_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="safe_metadata is not JSON-serializable",
            ) from None

        if body.new_version:
            content = body.content if body.content is not None else (latest.get("content") or "")
            prompt = body.prompt if body.prompt is not None else latest.get("prompt")
            analysis = body.analysis_content if body.analysis_content is not None else latest.get("analysis_content")
            with db.transaction():
                db.create_document_version(
                    media_id=media_id,
                    content=content,
                    prompt=prompt,
                    analysis_content=analysis,
                    safe_metadata=smj,
                )
        else:
            dv_id = latest.get("id")
            if not dv_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Latest version record missing identifier",
                )
            with db.transaction() as conn:
                update_version_safe_metadata_in_transaction(
                    db=db,
                    dv_id=dv_id,
                    safe_metadata_json=smj,
                    merged_metadata=merged_sm,
                    connection=conn,
                )

        details = get_full_media_details_rich(
            db=db,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found after version upsert",
            )
        return MediaDetailResponse(**details)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Error in advanced version upsert for media {}: {}",
            media_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upsert version",
        ) from exc


__all__ = ["router"]
