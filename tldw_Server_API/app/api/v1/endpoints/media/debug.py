from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.media_response_models import DebugSchemaResponse
from tldw_Server_API.app.core.DB_Management.media_db.api import describe_media_schema

router = APIRouter()


@router.get(
    "/debug/schema",
    response_model=DebugSchemaResponse,
    summary="Debug DB schema for media service",
    tags=["Media Debug"],
)
async def debug_schema(
    db: Any = Depends(get_media_db_for_user),
) -> DebugSchemaResponse:
    """
    Return basic schema and row-count diagnostics for the media database.

    This endpoint is read-only and intended for debugging and integration
    tests; it mirrors the legacy `/debug/schema` behavior while routing
    through the modular `media` package.
    """
    try:
        return DebugSchemaResponse(**describe_media_schema(db))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("debug_schema failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while reading media schema.",
        ) from exc


__all__ = ["router"]
