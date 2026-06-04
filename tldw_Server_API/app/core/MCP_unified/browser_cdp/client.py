"""Bounded Chrome DevTools Protocol client for read-only browser inspection."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class CDPClientError(RuntimeError):
    """Reason-coded CDP client error safe to surface through MCP tools."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


@dataclass(frozen=True, slots=True)
class CDPClientConfig:
    """Operator-owned CDP connection and budget settings."""

    debugger_url: str | None
    request_timeout_seconds: float = 3.0
    observation_window_ms: int = 250
    max_events: int = 100
    max_snapshot_nodes: int = 200
    screenshot_max_bytes: int = 2_000_000
    allow_non_loopback: bool = False


@dataclass(frozen=True, slots=True)
class CDPPageTarget:
    """Inspectable CDP page target."""

    target_id: str
    title: str
    url: str
    type: str
    websocket_url: str
    attached: bool | None = None
    metadata: dict[str, Any] | None = None


class CDPBrowserClient:
    """Small async CDP client for target discovery, commands, and events."""

    def __init__(self, config: CDPClientConfig) -> None:
        self.config = config
        self._next_command_id = 1

    @property
    def debugger_base_url(self) -> str:
        """Return a normalized debugger HTTP base URL or raise a reason-coded error."""
        return _normalize_debugger_url(
            self.config.debugger_url,
            allow_non_loopback=self.config.allow_non_loopback,
        )

    async def get_version(self) -> dict[str, Any]:
        """Return normalized `/json/version` metadata."""
        try:
            payload = await self._get_json("/json/version")
        except CDPClientError:
            raise
        except Exception as exc:
            raise CDPClientError("cdp_unreachable", "Unable to reach CDP version endpoint") from exc
        if not isinstance(payload, dict):
            raise CDPClientError("cdp_protocol_error", "CDP version endpoint returned a non-object payload")
        return {
            "browser": _text(payload.get("Browser")),
            "protocol_version": _text(payload.get("Protocol-Version")),
            "websocket_url": _text(payload.get("webSocketDebuggerUrl")),
            "raw": dict(payload),
        }

    async def list_pages(self) -> list[CDPPageTarget]:
        """Return inspectable page targets from `/json/list`."""
        try:
            payload = await self._get_json("/json/list")
        except CDPClientError:
            raise
        except Exception as exc:
            raise CDPClientError("cdp_unreachable", "Unable to reach CDP target list endpoint") from exc
        if not isinstance(payload, list):
            raise CDPClientError("cdp_protocol_error", "CDP target list endpoint returned a non-array payload")
        pages: list[CDPPageTarget] = []
        for item in payload:
            if not isinstance(item, dict) or item.get("type") != "page":
                continue
            target_id = _text(item.get("id"))
            websocket_url = _text(item.get("webSocketDebuggerUrl"))
            if not target_id or not websocket_url:
                continue
            pages.append(
                CDPPageTarget(
                    target_id=target_id,
                    title=_text(item.get("title")) or "",
                    url=_text(item.get("url")) or "",
                    type="page",
                    websocket_url=websocket_url,
                    attached=item.get("attached") if isinstance(item.get("attached"), bool) else None,
                    metadata=dict(item),
                )
            )
        return pages

    async def send_command(
        self,
        page: CDPPageTarget,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one CDP command to a page target and return its matching result."""
        command_id = self._allocate_command_id()
        message = {
            "id": command_id,
            "method": method,
            "params": params or {},
        }
        timeout_seconds = max(float(self.config.request_timeout_seconds), 0.1)
        deadline = time.monotonic() + timeout_seconds
        try:
            async with self._connect_websocket(page.websocket_url) as websocket:
                await websocket.send(json.dumps(message))
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CDPClientError("cdp_command_timeout", "CDP command response timed out")
                    try:
                        raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                    except TimeoutError as exc:
                        raise CDPClientError("cdp_command_timeout", "CDP command response timed out") from exc
                    payload = _json_object(raw)
                    if payload.get("id") != command_id:
                        continue
                    if isinstance(payload.get("error"), dict):
                        error = payload["error"]
                        message_text = _text(error.get("message")) or "CDP command failed"
                        raise CDPClientError("cdp_protocol_error", message_text)
                    result = payload.get("result")
                    return result if isinstance(result, dict) else {}
        except CDPClientError:
            raise
        except Exception as exc:
            raise CDPClientError("cdp_unreachable", "Unable to communicate with CDP target") from exc

    async def observe_events(
        self,
        page: CDPPageTarget,
        *,
        enable_methods: list[str],
        event_names: set[str],
        window_ms: int,
        max_events: int,
    ) -> dict[str, Any]:
        """Collect selected CDP events during a bounded observation window."""
        started = time.monotonic()
        deadline = started + max(window_ms, 1) / 1000
        events: list[dict[str, Any]] = []
        truncated = False
        try:
            async with self._connect_websocket(page.websocket_url) as websocket:
                for method in enable_methods:
                    command_id = self._allocate_command_id()
                    await websocket.send(
                        json.dumps({"id": command_id, "method": method, "params": {}})
                    )

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                    except TimeoutError:
                        break
                    payload = _json_object(raw)
                    method = _text(payload.get("method"))
                    if method not in event_names:
                        continue
                    params = payload.get("params")
                    events.append(
                        {
                            "method": method,
                            "params": params if isinstance(params, dict) else {},
                        }
                    )
                    if len(events) >= max_events:
                        truncated = True
                        break
        except CDPClientError:
            raise
        except Exception as exc:
            raise CDPClientError("cdp_unreachable", "Unable to observe CDP target events") from exc

        return {
            "events": events,
            "truncated": truncated,
            "observed_for_ms": int((time.monotonic() - started) * 1000),
        }

    async def _get_json(self, path: str) -> Any:
        """Fetch JSON from the configured debugger endpoint."""
        url = f"{self.debugger_base_url}{path}"
        timeout = httpx.Timeout(max(float(self.config.request_timeout_seconds), 0.1))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    def _connect_websocket(self, websocket_url: str) -> Any:
        """Return a WebSocket async context manager for a CDP target."""
        import websockets

        return websockets.connect(
            websocket_url,
            open_timeout=max(float(self.config.request_timeout_seconds), 0.1),
            close_timeout=max(float(self.config.request_timeout_seconds), 0.1),
        )

    def _allocate_command_id(self) -> int:
        command_id = self._next_command_id
        self._next_command_id += 1
        return command_id


def _normalize_debugger_url(
    debugger_url: str | None,
    *,
    allow_non_loopback: bool,
) -> str:
    raw_url = str(debugger_url or "").strip()
    if not raw_url:
        raise CDPClientError("cdp_not_configured", "CDP debugger URL is not configured")
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CDPClientError("cdp_endpoint_invalid", "CDP debugger URL must be an HTTP(S) URL")
    host = parsed.hostname or ""
    if not allow_non_loopback and not _is_literal_loopback_host(host):
        raise CDPClientError(
            "cdp_endpoint_not_allowed",
            "CDP debugger URL must use a loopback host unless explicitly allowed",
        )
    path = parsed.path.rstrip("/")
    if path == "":
        path = ""
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_literal_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _json_object(raw: str | bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CDPClientError("cdp_protocol_error", "CDP returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise CDPClientError("cdp_protocol_error", "CDP returned a non-object message")
    return payload


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
