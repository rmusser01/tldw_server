"""FastAPI dependencies for Meetings database access (per-user Media DB)."""

from typing import Any

from fastapi import Depends, HTTPException, Query, WebSocket, status
from loguru import logger
from starlette.requests import Request as StarletteRequest

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Meetings_DB import (
    MeetingsDatabase,
    MeetingsDatabaseError,
    SchemaError,
)


async def get_meetings_db_for_user(
    current_user: User = Depends(get_request_user),
) -> MeetingsDatabase:
    if not current_user or current_user.id is None:
        logger.error("get_meetings_db_for_user called without a valid User")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User identification failed",
        )
    try:
        return MeetingsDatabase.for_user(user_id=current_user.id)
    except (MeetingsDatabaseError, SchemaError) as exc:
        logger.error(
            "Failed to init Meetings DB for user {}: {}",
            current_user.id,
            type(exc).__name__,
        )
        raise map_db_error_to_http(exc, default_detail="Meetings DB unavailable") from exc
    except Exception as exc:
        logger.error(
            "Failed to init Meetings DB for user {}: unexpected initialization error",
            current_user.id,
        )
        raise HTTPException(status_code=500, detail="Meetings DB unavailable") from exc


def _extract_websocket_credentials(
    websocket: WebSocket,
    token: str | None,
    api_key: str | None,
) -> tuple[str | None, str | None]:
    resolved_token = token
    resolved_api_key = api_key

    try:
        auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer ") and not resolved_token:
            resolved_token = auth_header.split(" ", 1)[1].strip()
    except Exception as exc:
        logger.debug("Unable to parse websocket authorization header: {}", type(exc).__name__)

    try:
        api_key_header = websocket.headers.get("x-api-key") or websocket.headers.get("X-API-KEY")
        if api_key_header and not resolved_api_key:
            resolved_api_key = api_key_header.strip()
    except Exception as exc:
        logger.debug("Unable to parse websocket API key header: {}", type(exc).__name__)

    try:
        proto_header = websocket.headers.get("sec-websocket-protocol") or websocket.headers.get("Sec-WebSocket-Protocol")
        if proto_header and not resolved_token:
            parts = [part.strip() for part in proto_header.split(",")]
            if len(parts) >= 2 and parts[0].lower() == "bearer" and parts[1]:
                resolved_token = parts[1]
    except Exception as exc:
        logger.debug("Unable to parse websocket subprotocol auth token: {}", type(exc).__name__)

    return resolved_token, resolved_api_key


def _build_request_from_websocket(websocket: WebSocket) -> StarletteRequest:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": websocket.url.path,
        "headers": [(key.encode("latin-1"), value.encode("latin-1")) for key, value in websocket.headers.items()],
    }
    query_string = websocket.scope.get("query_string")
    if isinstance(query_string, (bytes, bytearray)):
        scope["query_string"] = bytes(query_string)
    return StarletteRequest(scope)


async def get_meetings_db_for_websocket(
    websocket: WebSocket,
    token: str | None = Query(default=None),
    api_key: str | None = Query(default=None),
) -> MeetingsDatabase:
    resolved_token, resolved_api_key = _extract_websocket_credentials(
        websocket=websocket,
        token=token,
        api_key=api_key,
    )
    request = _build_request_from_websocket(websocket)
    current_user = await get_request_user(
        request=request,
        api_key=resolved_api_key,
        token=resolved_token,
    )
    if not current_user or current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized websocket client",
        )
    try:
        return MeetingsDatabase.for_user(user_id=current_user.id)
    except (MeetingsDatabaseError, SchemaError) as exc:
        logger.error(
            "Failed to init Meetings DB for websocket user {}: {}",
            current_user.id,
            type(exc).__name__,
        )
        raise map_db_error_to_http(exc, default_detail="Meetings DB unavailable") from exc
    except Exception as exc:
        logger.error(
            "Failed to init Meetings DB for websocket user {}: unexpected initialization error",
            current_user.id,
        )
        raise HTTPException(status_code=500, detail="Meetings DB unavailable") from exc
