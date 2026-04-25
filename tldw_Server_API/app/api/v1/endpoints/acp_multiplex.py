"""ACP Multiplex sub-module.

Provides a single WebSocket endpoint that multiplexes event streams for
multiple agent sessions over one connection.

``WS /acp/multiplex`` -- send ``STREAM_OPEN`` / ``STREAM_CLOSE`` frames to
subscribe / unsubscribe from session event buses.  Events arrive as
``STREAM_DATA`` frames.  30-second ``PING`` / ``PONG`` keepalive.
"""
from __future__ import annotations

import contextlib
import uuid

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import (
    _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS,
    _acp_ws_release_quota,
    _acp_ws_try_acquire_quota,
    _authenticate_ws,
    _require_session_access,
    _resolve_acp_session_persona_id,
    get_runner_client,
    get_session_event_bus,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.manager import MultiplexManager
from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.protocol import MultiplexMessage

router = APIRouter(prefix="/acp", tags=["acp-multiplex"])


@router.websocket("/multiplex")
async def acp_multiplex_ws(
    websocket: WebSocket,
    token: str | None = Query(None),
    api_key: str | None = Query(None),
) -> None:
    """Multi-session event multiplexer.

    Accepts a WebSocket, creates a :class:`MultiplexManager` backed by the
    shared session event-bus registry, and loops reading client frames until
    disconnect.
    """
    user_id = await _authenticate_ws(
        websocket,
        token=token,
        api_key=api_key,
        required_scope="write",
    )
    if user_id is None:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4401)
        return

    try:
        client = await get_runner_client()
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Multiplex WS runner client unavailable: {}", exc)
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4404)
        return

    await websocket.accept()
    connection_id = uuid.uuid4().hex[:12]
    logger.info("Multiplex WS connected: {}", connection_id)

    async def _authorize_stream(
        stream_id: str,
    ) -> tuple[bool, str | None, dict[str, str | None] | None]:
        try:
            await _require_session_access(client, session_id=stream_id, user_id=int(user_id))
        except HTTPException:
            return False, "Unknown session or access denied", None
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Multiplex WS {} access check failed for stream {}: {}",
                connection_id,
                stream_id,
                exc,
            )
            return False, "Unable to authorize stream", None

        persona_id = await _resolve_acp_session_persona_id(
            client,
            session_id=stream_id,
            user_id=int(user_id),
        )
        quota_token, _quota_reason = _acp_ws_try_acquire_quota(
            user_id=int(user_id),
            session_id=stream_id,
            persona_id=persona_id,
        )
        if quota_token is None:
            return False, "ACP WebSocket quota exceeded", None
        return True, None, quota_token

    def _release_stream(stream_id: str, quota_token: dict[str, str | None] | None) -> None:
        del stream_id
        _acp_ws_release_quota(quota_token)

    manager = MultiplexManager(
        connection_id=connection_id,
        send_fn=websocket.send_text,
        get_bus_fn=get_session_event_bus,
        authorize_stream_fn=_authorize_stream,
        release_stream_fn=_release_stream,
    )
    manager.start()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                await manager.handle_message(raw)
            except Exception:
                logger.exception("Multiplex WS {} handler failure", connection_id)
                with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                    await websocket.send_text(
                        MultiplexMessage.error("Internal multiplex error").to_json()
                    )
    except WebSocketDisconnect:
        logger.info("Multiplex WS disconnected: {}", connection_id)
    finally:
        await manager.stop()
