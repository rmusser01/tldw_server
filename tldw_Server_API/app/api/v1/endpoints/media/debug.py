from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.media_response_models import DebugSchemaResponse

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
        with db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables: list[str] = [row[0] for row in cursor.fetchall()]

            def _table_columns(name: str) -> list[str]:
                try:
                    cursor.execute(f"PRAGMA table_info({name})")
                    return [col[1] for col in cursor.fetchall()]
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.warning(
                        "Failed to introspect columns for table {}: {}",
                        name,
                        exc,
                    )
                    return []

            media_columns = _table_columns("Media")
            media_mods_columns = _table_columns("MediaModifications")

            cursor.execute("SELECT COUNT(*) FROM Media")
            media_count_row = cursor.fetchone()
            media_count = int(media_count_row[0]) if media_count_row else 0

        return DebugSchemaResponse(
            tables=tables,
            media_columns=media_columns,
            media_mods_columns=media_mods_columns,
            media_count=media_count,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("debug_schema failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while reading media schema.",
        ) from exc


__all__ = ["router"]
