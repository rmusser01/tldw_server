"""
RAG pipeline preset and cleanup endpoints extracted from evaluations_unified.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import TokenScopeGuard, User

from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    create_error_response,
    get_eval_request_user,
    sanitize_error_message,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    PipelineCleanupResponse,
    PipelinePresetCreate,
    PipelinePresetListResponse,
    PipelinePresetResponse,
)
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service_for_user,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores import (
    create_from_settings_for_user,
)

pipeline_router = APIRouter()

_TS_PARSE_EXCEPTIONS = (TypeError, ValueError)
_PIPELINE_ENDPOINT_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_PIPELINE_CLEANUP_ITEM_EXCEPTIONS = (
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@pipeline_router.post(
    "/rag/pipeline/presets",
    response_model=PipelinePresetResponse,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="evals.rag_pipeline.preset.create"))],
)
async def create_or_update_pipeline_preset(
    preset: PipelinePresetCreate,
    current_user: User = Depends(get_eval_request_user),
):
    try:
        svc = get_unified_evaluation_service_for_user(current_user.id)
        db = svc.db
        if db is None:
            raise ValueError("Database not available")
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        db.upsert_pipeline_preset(preset.name, preset.config, user_id=stable_user_id)
        row = db.get_pipeline_preset(preset.name, user_id=stable_user_id)

        def to_ts(x: str) -> Optional[int]:
            try:
                if not x:
                    return None
                if "T" in x:
                    return int(datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())
                return int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
            except _TS_PARSE_EXCEPTIONS:
                return None

        return PipelinePresetResponse(
            name=row["name"],
            config=row["config"],
            created_at=to_ts(row.get("created_at")),
            updated_at=to_ts(row.get("updated_at")),
        )
    except _PIPELINE_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to save preset")
        raise create_error_response(
            message=f"Failed to save preset: {sanitize_error_message(e, 'save_preset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@pipeline_router.get(
    "/rag/pipeline/presets",
    response_model=PipelinePresetListResponse,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="evals.rag_pipeline.preset.list"))],
)
async def list_pipeline_presets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        svc = get_unified_evaluation_service_for_user(current_user.id)
        db = svc.db
        if db is None:
            raise ValueError("Database not available")
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        items, total = db.list_pipeline_presets(limit=limit, offset=offset, user_id=stable_user_id)
        resp_items = []

        def to_ts(x: str) -> Optional[int]:
            try:
                if not x:
                    return None
                if "T" in x:
                    return int(datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())
                return int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
            except _TS_PARSE_EXCEPTIONS:
                return None

        for r in items:
            resp_items.append(PipelinePresetResponse(
                name=r["name"],
                config=r["config"],
                created_at=to_ts(r.get("created_at")),
                updated_at=to_ts(r.get("updated_at")),
            ))
        pagination = build_offset_pagination_meta(total=total, limit=limit, offset=offset, count=len(resp_items))
        return PipelinePresetListResponse(
            items=resp_items,
            total=total,
            limit=limit,
            offset=offset,
            pagination=pagination,
        )
    except _PIPELINE_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to list presets")
        raise create_error_response(
            message=f"Failed to list presets: {sanitize_error_message(e, 'list_presets')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@pipeline_router.get(
    "/rag/pipeline/presets/{name}",
    response_model=PipelinePresetResponse,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="evals.rag_pipeline.preset.get"))],
)
async def get_pipeline_preset(
    name: str,
    current_user: User = Depends(get_eval_request_user),
):
    try:
        svc = get_unified_evaluation_service_for_user(current_user.id)
        db = svc.db
        if db is None:
            raise ValueError("Database not available")
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        row = db.get_pipeline_preset(name, user_id=stable_user_id)
        if not row:
            raise create_error_response(
                message=f"Preset {name} not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        def to_ts(x: str) -> Optional[int]:
            try:
                if not x:
                    return None
                if "T" in x:
                    return int(datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())
                return int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
            except _TS_PARSE_EXCEPTIONS:
                return None

        return PipelinePresetResponse(
            name=row["name"],
            config=row["config"],
            created_at=to_ts(row.get("created_at")),
            updated_at=to_ts(row.get("updated_at")),
        )
    except HTTPException:
        raise
    except _PIPELINE_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to get preset")
        raise create_error_response(
            message=f"Failed to get preset: {sanitize_error_message(e, 'get_preset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@pipeline_router.delete(
    "/rag/pipeline/presets/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="evals.rag_pipeline.preset.delete"))],
)
async def delete_pipeline_preset(
    name: str,
    current_user: User = Depends(get_eval_request_user),
) -> Response:
    try:
        svc = get_unified_evaluation_service_for_user(current_user.id)
        db = svc.db
        if db is None:
            raise ValueError("Database not available")
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        ok = db.delete_pipeline_preset(name, user_id=stable_user_id)
        if not ok:
            raise create_error_response(
                message=f"Preset {name} not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except _PIPELINE_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to delete preset")
        raise create_error_response(
            message=f"Failed to delete preset: {sanitize_error_message(e, 'delete_preset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@pipeline_router.post(
    "/rag/pipeline/cleanup",
    response_model=PipelineCleanupResponse,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="evals.rag_pipeline.cleanup"))],
)
async def cleanup_ephemeral_collections(
    current_user: User = Depends(get_eval_request_user),
):
    """Delete expired ephemeral collections according to TTL registry."""
    try:
        svc = get_unified_evaluation_service_for_user(current_user.id)
        db = svc.db
        if db is None:
            raise ValueError("Database not available")
        expired = db.list_expired_ephemeral_collections()
        if not expired:
            return PipelineCleanupResponse(expired_count=0, deleted_count=0)
        from tldw_Server_API.app.core.config import settings as app_settings
        adapter = create_from_settings_for_user(app_settings, current_user.id_str)
        await adapter.initialize()
        deleted = 0
        errors: list[str] = []
        for name in expired:
            try:
                await adapter.delete_collection(name)
                db.mark_ephemeral_deleted(name)
                deleted += 1
            except _PIPELINE_CLEANUP_ITEM_EXCEPTIONS:
                logger.warning("Failed to delete expired collection")
                errors.append("Collection cleanup failed")
        return PipelineCleanupResponse(expired_count=len(expired), deleted_count=deleted, errors=errors or None)
    except _PIPELINE_ENDPOINT_EXCEPTIONS as e:
        logger.error("Cleanup failed")
        raise create_error_response(
            message=f"Cleanup failed: {sanitize_error_message(e, 'cleanup')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e
