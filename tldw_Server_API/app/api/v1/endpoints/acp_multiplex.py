"""Multi-session WebSocket multiplexer endpoint."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(prefix="/acp", tags=["acp-multiplex"])


@router.websocket("/multiplex")
async def acp_multiplex_ws(websocket: WebSocket) -> None:
    """Multi-session event multiplexer.

    Subscribe to multiple sessions over a single WebSocket.
    Send STREAM_OPEN with stream_id=session_id to subscribe.
    Receive STREAM_DATA frames with events from all subscribed sessions.
    """
    await websocket.accept()

    from tldw_Server_API.app.core.Agent_Client_Protocol.multiplex.manager import MultiplexManager

    # Get bus accessor (reuse the SSE endpoint's bus registry)
    try:
        from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import get_session_event_bus
    except ImportError:
        get_session_event_bus = lambda sid: None  # noqa: E731

    manager = MultiplexManager(
        send_fn=websocket.send_text,
        get_bus_fn=get_session_event_bus,
    )

    await manager.start()
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(data)
    except WebSocketDisconnect:
        logger.debug("Multiplex WebSocket disconnected")
    except Exception as exc:
        logger.error("Multiplex WebSocket error: {}", exc)
    finally:
        await manager.stop()
