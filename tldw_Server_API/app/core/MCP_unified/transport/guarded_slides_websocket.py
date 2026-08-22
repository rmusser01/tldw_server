"""Uvicorn WebSocket protocol with receive-time Slides admission scanning."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from uvicorn.protocols.websockets.websockets_sansio_impl import (
    WebSocketsSansIOProtocol,
)
from websockets.frames import Frame, Opcode

from tldw_Server_API.app.core.Security.standalone_html_request_guard import (
    SCANNER_CHUNK_BYTES,
    ShallowStandaloneFieldScanner,
    StandaloneHtmlAdmissionError,
)

GUARDED_SLIDES_SCOPE_KEY = "_tldw_guarded_slides_websocket"
_MCP_WEBSOCKET_PATHS = frozenset({"/api/v1/mcp/ws", "/mcp/ws"})
_GUARDED_SLIDES_MARKER = object()


def _mark_guarded_slides_websocket_scope(scope: MutableMapping[str, Any]) -> None:
    """Attach the process-private marker; only the guarded protocol calls this."""

    scope[GUARDED_SLIDES_SCOPE_KEY] = _GUARDED_SLIDES_MARKER


def is_guarded_slides_websocket_scope(scope: Mapping[str, Any]) -> bool:
    """Return whether an ASGI scope carries the process-private guard marker."""

    return scope.get(GUARDED_SLIDES_SCOPE_KEY) is _GUARDED_SLIDES_MARKER


def guarded_slides_websocket_metadata(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only a valid scope marker into MCP connection metadata."""

    metadata: dict[str, Any] = {"mcp_transport": "websocket"}
    if is_guarded_slides_websocket_scope(scope):
        metadata[GUARDED_SLIDES_SCOPE_KEY] = _GUARDED_SLIDES_MARKER
    return metadata


def is_guarded_slides_websocket_metadata(metadata: Mapping[str, Any]) -> bool:
    """Return whether server-created request metadata has the trusted marker."""

    return metadata.get(GUARDED_SLIDES_SCOPE_KEY) is _GUARDED_SLIDES_MARKER


class GuardedSlidesWebSocketProtocol(WebSocketsSansIOProtocol):
    """Scan decoded MCP text frames before Uvicorn concatenates each message."""

    def __init__(self, config, server_state, app_state, _loop=None) -> None:
        if config.ws_per_message_deflate:
            raise RuntimeError("guarded_slides_websocket_requires_compression_disabled")
        super().__init__(config, server_state, app_state, _loop)
        self._guard_enabled = False
        self._guard_text_active = False
        self._guard_forward_started = False
        self._guard_pending = bytearray()
        self._guard_scanner: ShallowStandaloneFieldScanner | None = None

    @property
    def guard_retained_bytes(self) -> int:
        """Return transport-owned lookbehind and scanner bytes for tests/metrics."""

        scanner_bytes = self._guard_scanner.retained_bytes if self._guard_scanner else 0
        return len(self._guard_pending) + scanner_bytes

    def handle_connect(self, event) -> None:
        """Create the base scope, then mark only MCP scopes under this protocol."""

        super().handle_connect(event)
        path = event.path.partition("?")[0]
        self._guard_enabled = path in _MCP_WEBSOCKET_PATHS
        if self._guard_enabled and hasattr(self, "scope"):
            _mark_guarded_slides_websocket_scope(self.scope)

    def _finish_guard(self) -> None:
        scanner = self._guard_scanner
        self._guard_scanner = None
        if scanner is not None:
            scanner.finish()
        self._guard_pending.clear()
        self._guard_text_active = False
        self._guard_forward_started = False

    def _begin_guarded_text(self) -> None:
        self._finish_guard()
        self._guard_scanner = ShallowStandaloneFieldScanner(mode="mcp")
        self._guard_text_active = True

    def _forward_text_bytes(self, data: bytes, *, fin: bool) -> None:
        if not self._guard_forward_started:
            self._guard_forward_started = True
            super().handle_text(Frame(Opcode.TEXT, data, fin=fin))
        else:
            super().handle_cont(Frame(Opcode.CONT, data, fin=fin))

    def _reject_message(self, error: StandaloneHtmlAdmissionError) -> None:
        self._finish_guard()
        self.bytes = b""
        self.logger.warning("Rejected standalone HTML on guarded MCP WebSocket")
        if not self.close_sent and not self.transport.is_closing():
            self.queue.put_nowait(
                {
                    "type": "websocket.disconnect",
                    "code": 1008,
                    "reason": error.code,
                }
            )
            self.conn.send_close(1008, error.code)
            self.transport.write(b"".join(self.conn.data_to_send()))
            self.close_sent = True
            self.transport.close()

    def _scan_text_frame(self, event: Frame, *, start: bool) -> None:
        if start:
            self._begin_guarded_text()
        scanner = self._guard_scanner
        if scanner is None:
            return
        data = bytes(event.data)
        try:
            for offset in range(0, len(data), SCANNER_CHUNK_BYTES):
                piece = data[offset : offset + SCANNER_CHUNK_BYTES]
                cursor = 0
                while self._guard_pending and cursor < len(piece):
                    unit = piece[cursor : cursor + 1]
                    scanner.feed(unit)
                    self._guard_pending.extend(unit)
                    cursor += 1
                    if not scanner.requires_value_lookbehind:
                        self._forward_text_bytes(
                            bytes(self._guard_pending),
                            fin=False,
                        )
                        self._guard_pending.clear()
                        break
                if self._guard_pending:
                    continue
                remainder = piece[cursor:]
                if not remainder:
                    continue
                hold_from = scanner.feed(remainder)
                if scanner.requires_value_lookbehind:
                    safe = remainder[: hold_from or 0]
                    if safe:
                        self._forward_text_bytes(safe, fin=False)
                    self._guard_pending.extend(remainder[hold_from or 0 :])
                else:
                    self._forward_text_bytes(remainder, fin=False)
        except StandaloneHtmlAdmissionError as error:
            self._reject_message(error)
            return

        if event.fin:
            pending = bytes(self._guard_pending)
            self._guard_pending.clear()
            scanner.finish()
            self._guard_scanner = None
            self._forward_text_bytes(pending, fin=True)
            self._guard_text_active = False
            self._guard_forward_started = False

    def handle_text(self, event: Frame) -> None:
        if not self._guard_enabled:
            super().handle_text(event)
            return
        self._scan_text_frame(event, start=True)

    def handle_cont(self, event: Frame) -> None:
        if not self._guard_enabled or not self._guard_text_active:
            super().handle_cont(event)
            return
        self._scan_text_frame(event, start=False)

    def handle_bytes(self, event: Frame) -> None:
        self._finish_guard()
        super().handle_bytes(event)

    def handle_close(self, event: Frame) -> None:
        self._finish_guard()
        super().handle_close(event)

    def connection_lost(self, exc: Exception | None) -> None:
        self._finish_guard()
        super().connection_lost(exc)

    def shutdown(self) -> None:
        self._finish_guard()
        super().shutdown()


__all__ = [
    "GUARDED_SLIDES_SCOPE_KEY",
    "GuardedSlidesWebSocketProtocol",
    "guarded_slides_websocket_metadata",
    "is_guarded_slides_websocket_metadata",
    "is_guarded_slides_websocket_scope",
]
