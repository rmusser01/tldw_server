"""Upstream Streamable HTTP and legacy SSE transports for external MCP servers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import ssl
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from loguru import logger

from mcp_unified.federation.models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
)
from mcp_unified.federation.resource_payloads import (
    normalize_external_resource_list,
    normalize_external_resource_read,
)
from mcp_unified.storage.models import ExternalServerDefinition

_MCP_PROTOCOL_VERSION = "2024-11-05"
_MCP_PROTOCOL_VERSION_STREAMABLE = "2025-03-26"
_CLIENT_INFO = {"name": "mcp_unified_external_federation", "version": "0.1.0"}
_DEFAULT_CONNECT_TIMEOUT_S = 30.0
_DEFAULT_REQUEST_TIMEOUT_S = 30.0
_DEFAULT_HEALTH_TIMEOUT_S = 5.0
_DEFAULT_CLOSE_TIMEOUT_S = 5.0
_SESSION_HEADER = "mcp-session-id"
_PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
_ACCEPT_BOTH = "application/json, text/event-stream"
_SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "cookie", "x-api-key", "api-key"}
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class HttpExternalTransportError(RuntimeError):
    """Raised when an HTTP transport operation fails without exposing secrets."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        server_id: str | None = None,
        method: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{message} (reason_code={reason_code})")
        self.reason_code = reason_code
        self.server_id = server_id
        self.method = method
        self.details = deepcopy(details or {})


def _reason_code_for_transport_error(exc: httpx.HTTPError) -> str:
    """Map an httpx transport failure to a stable, secret-free reason code.

    Args:
        exc: The httpx error raised while sending a request or reading a response.

    Returns:
        One of ``"tls_failed"``, ``"connect_failed"``, ``"request_timeout"``, or
        ``"connection_closed"``.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        cause: BaseException | None = exc
        while cause is not None:
            if isinstance(cause, ssl.SSLError):
                return "tls_failed"
            cause = cause.__cause__ or cause.__context__
        return "connect_failed"
    if isinstance(exc, httpx.TimeoutException):
        return "request_timeout"
    return "connection_closed"


def _normalize_tool_definitions(result: Any) -> list[ExternalToolDefinition]:
    """Normalize a ``tools/list`` result into external tool definitions.

    Args:
        result: The raw ``result`` member of a ``tools/list`` JSON-RPC response.

    Returns:
        Valid tool definitions; rows without a usable name are dropped, and
        malformed descriptions, schemas, and metadata fall back to safe defaults.
    """
    if isinstance(result, dict):
        raw_tools = result.get("tools") or []
    elif isinstance(result, list):
        raw_tools = result
    else:
        raw_tools = []

    tools: list[ExternalToolDefinition] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        description = item.get("description")
        if not isinstance(description, str):
            description = ""
        input_schema = item.get("inputSchema")
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object"}
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        tools.append(
            ExternalToolDefinition(
                name=name,
                description=description,
                input_schema=deepcopy(input_schema),
                metadata=deepcopy(metadata),
            )
        )
    return tools


async def _iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event, data)`` pairs from a text/event-stream line iterator.

    Args:
        lines: Decoded lines of an SSE body, without line terminators.

    Yields:
        Each dispatched event as an ``(event_name, joined_data)`` tuple.
    """
    event = "message"
    data_lines: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)


class _HttpExternalTransportBase:
    """Shared plumbing for URL-targeted external MCP transports."""

    transport_name = ""

    def __init__(
        self,
        server: ExternalServerDefinition,
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    ) -> None:
        self._server = server.model_copy(deep=True)
        self.server_id = self._server.id
        if self._server.transport != self.transport_name:
            raise self._error(
                "External HTTP transport received an unsupported server transport",
                reason_code="unsupported_transport",
            )
        url = (self._server.url or "").strip()
        scheme = urlsplit(url).scheme.lower()
        if not url or scheme not in ("http", "https"):
            raise self._error(
                "External HTTP transport requires an http(s) url",
                reason_code="invalid_url",
            )
        self._url = url
        self._static_headers = {
            str(k): str(v) for k, v in (self._server.headers or {}).items()
        }
        host = (urlsplit(url).hostname or "").lower()
        self._plain_http_non_loopback = scheme == "http" and host not in _LOOPBACK_HOSTS
        if self._plain_http_non_loopback:
            sensitive = {name.lower() for name in self._static_headers}
            if sensitive & _SENSITIVE_HEADER_NAMES:
                raise self._error(
                    "Credential headers over plain http are only allowed to loopback",
                    reason_code="insecure_url",
                )
        self._connect_timeout_s = self._positive_timeout(connect_timeout_s, "connect_timeout_s")
        self._request_timeout_s = self._positive_timeout(request_timeout_s, "request_timeout_s")
        self._health_timeout_s = self._positive_timeout(health_timeout_s, "health_timeout_s")
        self._connect_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._next_request_id = 1
        self._initialized = False

    def _take_request_id(self) -> int:
        """Return the next monotonically increasing JSON-RPC request id."""
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _error(
        self,
        message: str,
        *,
        reason_code: str,
        method: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> HttpExternalTransportError:
        """Build a reason-coded transport error bound to this server id."""
        return HttpExternalTransportError(
            message,
            reason_code=reason_code,
            server_id=getattr(self, "server_id", None),
            method=method,
            details=details,
        )

    def _status_error(self, status: int, method: str | None) -> HttpExternalTransportError:
        """Map an HTTP error status to a reason-coded transport error."""
        if status in (401, 403):
            return self._error(
                "External HTTP server requires authorization",
                reason_code="auth_required",
                method=method,
                details={"status": status},
            )
        return self._error(
            "External HTTP server returned an error status",
            reason_code="upstream_http_error",
            method=method,
            details={"status": status},
        )

    def _transport_error(
        self, exc: httpx.HTTPError, method: str | None
    ) -> HttpExternalTransportError:
        """Map an httpx transport failure to a reason-coded transport error."""
        return self._error(
            "External HTTP request failed",
            reason_code=_reason_code_for_transport_error(exc),
            method=method,
        )

    def _runtime_auth_headers(
        self,
        runtime_auth: BrokeredExternalCredential | None,
    ) -> dict[str, str]:
        """Return brokered per-call headers, refusing credentials over plain http.

        Args:
            runtime_auth: Optional brokered credential resolved for this call.

        Returns:
            Header names/values to merge into the outgoing request.

        Raises:
            HttpExternalTransportError: ``insecure_url`` when a credential
                header would be sent over plain http to a non-loopback host.
        """
        if runtime_auth is None or not runtime_auth.headers:
            return {}
        headers = {str(k): str(v) for k, v in runtime_auth.headers.items()}
        if self._plain_http_non_loopback:
            sensitive = {name.lower() for name in headers}
            if sensitive & _SENSITIVE_HEADER_NAMES:
                raise self._error(
                    "Brokered credential headers over plain http are only allowed to loopback",
                    reason_code="insecure_url",
                )
        return headers

    @classmethod
    def _positive_timeout(cls, value: float, field_name: str) -> float:
        """Validate that a timeout value is a finite positive number."""
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise HttpExternalTransportError(
                f"External HTTP transport {field_name} must be a finite positive number",
                reason_code="invalid_timeout",
            ) from exc
        if timeout <= 0 or not math.isfinite(timeout):
            raise HttpExternalTransportError(
                f"External HTTP transport {field_name} must be a finite positive number",
                reason_code="invalid_timeout",
            )
        return timeout

    @staticmethod
    def _encode_payload(payload: dict[str, Any], method: str) -> bytes:
        """Encode a JSON-RPC payload, rejecting non-serializable requests."""
        try:
            return json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise HttpExternalTransportError(
                "External HTTP request is not JSON serializable",
                reason_code="invalid_request",
                method=method,
            ) from exc

    def _tool_call_result(
        self, tool_name: str, response: dict[str, Any]
    ) -> ExternalToolCallResult:
        """Normalize a JSON-RPC tool response into an ExternalToolCallResult."""
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if not isinstance(message, str) or not message:
                message = "External MCP tool call failed"
            return ExternalToolCallResult(
                content=[{"type": "text", "text": message}],
                is_error=True,
                metadata={
                    "server_id": self.server_id,
                    "tool_name": tool_name,
                    "reason_code": "upstream_error",
                },
            )
        result = response.get("result")
        if isinstance(result, dict):
            content = result.get("content") if "content" in result else result
            is_error = bool(result.get("isError"))
        else:
            content = result
            is_error = False
        return ExternalToolCallResult(
            content=deepcopy(content),
            is_error=is_error,
            metadata={"server_id": self.server_id, "tool_name": tool_name},
        )


class StreamableHttpExternalTransport(_HttpExternalTransportBase):
    """MCP Streamable HTTP transport: JSON-RPC POSTs against a single endpoint."""

    transport_name = "streamable_http"

    def __init__(
        self,
        server: ExternalServerDefinition,
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    ) -> None:
        super().__init__(
            server,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
            health_timeout_s=health_timeout_s,
        )
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._protocol_version: str | None = None

    async def connect(self) -> None:
        """Initialize an MCP session against the configured endpoint."""
        if self._initialized and self._client is not None:
            return
        async with self._connect_lock:
            if self._initialized and self._client is not None:
                return
            await self._close_client()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._request_timeout_s, connect=self._connect_timeout_s
                )
            )
            try:
                response = await self._request(
                    "initialize",
                    {
                        "protocolVersion": _MCP_PROTOCOL_VERSION_STREAMABLE,
                        "capabilities": {},
                        "clientInfo": _CLIENT_INFO,
                    },
                    timeout_s=self._connect_timeout_s,
                )
                result = response.get("result")
                negotiated = (
                    result.get("protocolVersion") if isinstance(result, dict) else None
                )
                if isinstance(negotiated, str) and negotiated.strip():
                    self._protocol_version = negotiated.strip()
                await self._notify("notifications/initialized", {})
            except Exception:
                await self._close_client()
                raise
            self._initialized = True

    async def close(self) -> None:
        """Terminate the MCP session and release the HTTP client."""
        async with self._request_lock, self._connect_lock:
            await self._close_client()

    async def health_check(self) -> dict[str, bool]:
        """Return quick connectivity and initialization health indicators.

        Returns:
            Boolean checks: ``configured``, ``connected``, ``initialized``, and
            ``spawns_process`` (always ``False`` for HTTP transports).
        """
        checks = {
            "configured": True,
            "connected": self._client is not None,
            "initialized": self._initialized and self._client is not None,
            "spawns_process": False,
        }
        if not checks["initialized"]:
            return checks
        try:
            async with self._request_lock:
                await self._request("ping", {}, timeout_s=self._health_timeout_s)
        except HttpExternalTransportError:
            self._initialized = False
            checks["connected"] = False
            checks["initialized"] = False
        return checks

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Discover and normalize upstream MCP tool definitions."""
        async with self._request_lock:
            await self._ensure_connected()
            response = await self._request("tools/list", {})
        return _normalize_tool_definitions(response.get("result"))

    async def list_resources(self) -> list[dict[str, Any]]:
        """Discover and normalize upstream MCP resource descriptors."""
        async with self._request_lock:
            await self._ensure_connected()
            response = await self._request("resources/list", {})
        return normalize_external_resource_list(response.get("result"))

    async def read_resource(self, uri: str, *, context: Any = None) -> dict[str, Any]:
        """Read one upstream MCP resource.

        Args:
            uri: The upstream resource URI to read.
            context: Unused; accepted for transport-protocol compatibility.

        Returns:
            The normalized ``resources/read`` result payload.
        """
        del context
        async with self._request_lock:
            await self._ensure_connected()
            response = await self._request("resources/read", {"uri": uri})
        return normalize_external_resource_read(response.get("result"))

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Execute one upstream MCP tool call.

        Args:
            tool_name: Upstream tool name to invoke.
            arguments: JSON-serializable tool arguments.
            context: Unused; accepted for transport-protocol compatibility.
            runtime_auth: Optional brokered credential; its ``headers`` are merged
                into this call's HTTP headers (``env`` has no HTTP equivalent and
                is ignored).

        Returns:
            The normalized tool call result; upstream JSON-RPC errors are
            returned as ``is_error`` results rather than raised.
        """
        del context
        extra_headers = self._runtime_auth_headers(runtime_auth)
        async with self._request_lock:
            await self._ensure_connected()
            params: dict[str, Any] = {
                "name": tool_name,
                "arguments": deepcopy(arguments or {}),
            }
            response = await self._request(
                "tools/call",
                params,
                raise_on_error=False,
                extra_headers=extra_headers,
            )
        return self._tool_call_result(tool_name, response)

    async def _ensure_connected(self) -> None:
        """Connect (or reconnect) if the transport is not initialized."""
        if not self._initialized or self._client is None:
            await self.connect()

    def _base_headers(self, extra_headers: dict[str, str] | None) -> httpx.Headers:
        """Build request headers: static first, protocol values winning, extras last."""
        headers = httpx.Headers(self._static_headers)
        headers["accept"] = _ACCEPT_BOTH
        headers["content-type"] = "application/json"
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        if self._protocol_version:
            headers[_PROTOCOL_VERSION_HEADER] = self._protocol_version
        for name, value in (extra_headers or {}).items():
            headers[name] = value
        return headers

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        raise_on_error: bool = True,
        extra_headers: dict[str, str] | None = None,
        _retried: bool = False,
    ) -> dict[str, Any]:
        """Send one JSON-RPC request and return the correlated response payload."""
        client = self._require_client(method)
        request_id = self._take_request_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": deepcopy(params or {}),
        }
        encoded = self._encode_payload(payload, method)
        effective_timeout = timeout_s or self._request_timeout_s
        try:
            result = await asyncio.wait_for(
                self._send_request(
                    client, encoded, request_id, method, extra_headers, effective_timeout
                ),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise self._error(
                f"External HTTP request timed out for method '{method}'",
                reason_code="request_timeout",
                method=method,
            ) from exc
        except HttpExternalTransportError as exc:
            if exc.reason_code == "session_expired" and not _retried:
                self._initialized = False
                await self.connect()
                return await self._request(
                    method,
                    params,
                    timeout_s=timeout_s,
                    raise_on_error=raise_on_error,
                    extra_headers=extra_headers,
                    _retried=True,
                )
            raise
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, method) from exc

        if result.get("error") and raise_on_error:
            raise self._error(
                f"External HTTP request failed for method '{method}'",
                reason_code="upstream_error",
                method=method,
            )
        return result

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        encoded: bytes,
        request_id: int,
        method: str,
        extra_headers: dict[str, str] | None,
        effective_timeout: float,
    ) -> dict[str, Any]:
        """POST one encoded request and read its JSON or SSE-framed response."""
        async with client.stream(
            "POST",
            self._url,
            content=encoded,
            headers=self._base_headers(extra_headers),
            timeout=effective_timeout,
        ) as response:
            if response.status_code == 404 and self._session_id:
                self._session_id = None
                self._initialized = False
                raise self._error(
                    "External HTTP session expired",
                    reason_code="session_expired",
                    method=method,
                    details={"status": 404},
                )
            if response.status_code >= 400:
                raise self._status_error(response.status_code, method)
            session_id = response.headers.get(_SESSION_HEADER)
            if session_id:
                self._session_id = session_id
            content_type = response.headers.get("content-type", "")
            if content_type.startswith("text/event-stream"):
                return await self._read_sse_response(response, request_id, method)
            body = await response.aread()
            payload = self._decode_json_response(body, method)
            if payload.get("id") != request_id:
                raise self._error(
                    "External HTTP response id does not match the request",
                    reason_code="invalid_response",
                    method=method,
                )
            return payload

    async def _read_sse_response(
        self, response: httpx.Response, request_id: int, method: str
    ) -> dict[str, Any]:
        """Read an SSE-framed POST response until the matching id arrives."""
        async for _event, data in _iter_sse_events(response.aiter_lines()):
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("id") == request_id:
                return payload
        raise self._error(
            "External HTTP SSE response ended without a matching response",
            reason_code="connection_closed",
            method=method,
        )

    def _decode_json_response(self, body: bytes, method: str) -> dict[str, Any]:
        """Decode a JSON response body into a JSON-RPC object."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise self._error(
                "External HTTP response is not valid JSON",
                reason_code="invalid_response",
                method=method,
            ) from exc
        if not isinstance(payload, dict):
            raise self._error(
                "External HTTP response is not a JSON-RPC object",
                reason_code="invalid_response",
                method=method,
            )
        return payload

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """POST one JSON-RPC notification, accepting any 2xx response."""
        client = self._require_client(method)
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": deepcopy(params or {}),
        }
        encoded = self._encode_payload(payload, method)
        try:
            response = await client.post(
                self._url,
                content=encoded,
                headers=self._base_headers(None),
                timeout=self._request_timeout_s,
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, method) from exc
        if response.status_code >= 400:
            raise self._status_error(response.status_code, method)

    def _require_client(self, method: str) -> httpx.AsyncClient:
        """Return the live HTTP client or raise not_connected."""
        client = self._client
        if client is None:
            raise self._error(
                "External HTTP transport is not connected",
                reason_code="not_connected",
                method=method,
            )
        return client

    async def _close_client(self) -> None:
        """Send session DELETE (bounded), close the client, and reset state."""
        client = self._client
        self._client = None
        self._initialized = False
        if client is None:
            self._session_id = None
            return
        if self._session_id:
            with contextlib.suppress(Exception):
                await client.delete(
                    self._url,
                    headers={**self._static_headers, _SESSION_HEADER: self._session_id},
                    timeout=_DEFAULT_CLOSE_TIMEOUT_S,
                )
        self._session_id = None
        with contextlib.suppress(Exception):
            await client.aclose()


class SseExternalTransport(_HttpExternalTransportBase):
    """Legacy MCP HTTP+SSE transport: persistent GET stream plus POSTed messages."""

    transport_name = "sse"

    def __init__(
        self,
        server: ExternalServerDefinition,
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    ) -> None:
        super().__init__(
            server,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
            health_timeout_s=health_timeout_s,
        )
        self._client: httpx.AsyncClient | None = None
        self._stream_response: httpx.Response | None = None
        self._events: AsyncIterator[tuple[str, str]] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._endpoint_url: str | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}

    async def connect(self) -> None:
        """Open the SSE stream, resolve the message endpoint, and initialize."""
        if self._initialized and self._reader_alive():
            return
        async with self._connect_lock:
            if self._initialized and self._reader_alive():
                return
            await self._teardown()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._request_timeout_s,
                    connect=self._connect_timeout_s,
                    read=None,
                )
            )
            try:
                request = self._client.build_request(
                    "GET",
                    self._url,
                    headers={**self._static_headers, "accept": "text/event-stream"},
                )
                try:
                    response = await self._client.send(request, stream=True)
                except httpx.HTTPError as exc:
                    raise self._transport_error(exc, "connect") from exc
                if response.status_code >= 400:
                    status = response.status_code
                    await response.aclose()
                    raise self._status_error(status, "connect")
                self._stream_response = response
                self._events = _iter_sse_events(response.aiter_lines())
                endpoint = await asyncio.wait_for(
                    self._read_endpoint_event(),
                    timeout=self._connect_timeout_s,
                )
                endpoint_url = urljoin(str(response.url), endpoint)
                stream_parts = urlsplit(str(response.url))
                endpoint_parts = urlsplit(endpoint_url)
                if (
                    endpoint_parts.scheme != stream_parts.scheme
                    or endpoint_parts.netloc != stream_parts.netloc
                ):
                    raise self._error(
                        "External SSE endpoint event points outside the stream origin",
                        reason_code="invalid_endpoint",
                    )
                self._endpoint_url = endpoint_url
                self._reader_task = asyncio.create_task(self._pump_events())
                await self._request(
                    "initialize",
                    {
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": _CLIENT_INFO,
                    },
                    timeout_s=self._connect_timeout_s,
                )
                await self._post_notification("notifications/initialized", {})
            except HttpExternalTransportError:
                await self._teardown()
                raise
            except asyncio.TimeoutError as exc:
                await self._teardown()
                raise self._error(
                    "External SSE stream did not provide a message endpoint in time",
                    reason_code="connect_timeout",
                ) from exc
            except httpx.HTTPError as exc:
                await self._teardown()
                raise self._transport_error(exc, "connect") from exc
            except Exception:
                await self._teardown()
                raise
            self._initialized = True

    async def close(self) -> None:
        """Close the SSE stream, fail pending requests, and release the client."""
        async with self._request_lock, self._connect_lock:
            await self._teardown()

    async def health_check(self) -> dict[str, bool]:
        """Return quick connectivity and initialization health indicators.

        Returns:
            Boolean checks: ``configured``, ``connected``, ``initialized``, and
            ``spawns_process`` (always ``False`` for HTTP transports).
        """
        checks = {
            "configured": True,
            "connected": self._reader_alive(),
            "initialized": self._initialized and self._reader_alive(),
            "spawns_process": False,
        }
        if not checks["initialized"]:
            self._initialized = False
            return checks
        try:
            async with self._request_lock:
                await self._request("ping", {}, timeout_s=self._health_timeout_s)
        except HttpExternalTransportError:
            self._initialized = False
            checks["connected"] = False
            checks["initialized"] = False
        return checks

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Discover and normalize upstream MCP tool definitions."""
        async with self._request_lock:
            await self._ensure_connected()
            response = await self._request("tools/list", {})
        return _normalize_tool_definitions(response.get("result"))

    async def list_resources(self) -> list[dict[str, Any]]:
        """Discover and normalize upstream MCP resource descriptors."""
        async with self._request_lock:
            await self._ensure_connected()
            response = await self._request("resources/list", {})
        return normalize_external_resource_list(response.get("result"))

    async def read_resource(self, uri: str, *, context: Any = None) -> dict[str, Any]:
        """Read one upstream MCP resource.

        Args:
            uri: The upstream resource URI to read.
            context: Unused; accepted for transport-protocol compatibility.

        Returns:
            The normalized ``resources/read`` result payload.
        """
        del context
        async with self._request_lock:
            await self._ensure_connected()
            response = await self._request("resources/read", {"uri": uri})
        return normalize_external_resource_read(response.get("result"))

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Execute one upstream MCP tool call.

        Args:
            tool_name: Upstream tool name to invoke.
            arguments: JSON-serializable tool arguments.
            context: Unused; accepted for transport-protocol compatibility.
            runtime_auth: Optional brokered credential; its ``headers`` are merged
                into this call's POST headers (``env`` has no HTTP equivalent and
                is ignored).

        Returns:
            The normalized tool call result; upstream JSON-RPC errors are
            returned as ``is_error`` results rather than raised.
        """
        del context
        extra_headers = self._runtime_auth_headers(runtime_auth)
        async with self._request_lock:
            await self._ensure_connected()
            params: dict[str, Any] = {
                "name": tool_name,
                "arguments": deepcopy(arguments or {}),
            }
            response = await self._request(
                "tools/call",
                params,
                raise_on_error=False,
                extra_headers=extra_headers,
            )
        return self._tool_call_result(tool_name, response)

    async def _ensure_connected(self) -> None:
        """Connect (or reconnect) if the transport is not initialized."""
        if not self._initialized or not self._reader_alive():
            await self.connect()

    def _reader_alive(self) -> bool:
        """Return whether the background stream reader task is running."""
        return self._reader_task is not None and not self._reader_task.done()

    async def _read_endpoint_event(self) -> str:
        """Consume stream events until the endpoint event arrives."""
        events = self._events
        if events is None:
            raise self._error(
                "External SSE stream is not connected",
                reason_code="not_connected",
            )
        async for event, data in events:
            if event == "endpoint" and data.strip():
                return data.strip()
        raise self._error(
            "External SSE stream closed before providing a message endpoint",
            reason_code="connection_closed",
        )

    async def _pump_events(self) -> None:
        """Route streamed JSON-RPC responses to their pending futures."""
        events = self._events
        if events is None:
            return
        try:
            async for _event, data in events:
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                future = self._pending.pop(payload.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(payload)
        except Exception as exc:  # noqa: BLE001 - stream teardown races vary by backend.
            logger.debug(
                "External SSE stream reader ended for server {server_id}: {error_type}",
                server_id=self.server_id,
                error_type=type(exc).__name__,
            )
        finally:
            self._initialized = False
            self._fail_pending("connection_closed")

    def _fail_pending(self, reason_code: str) -> None:
        """Fail every pending request future with a reason-coded error."""
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(
                    self._error(
                        "External SSE stream closed before the response arrived",
                        reason_code=reason_code,
                    )
                )

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        raise_on_error: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send one JSON-RPC request and return the correlated response payload."""
        request_id = self._take_request_id()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": deepcopy(params or {}),
        }

        async def _round_trip() -> dict[str, Any]:
            """POST the payload, then await the stream-correlated response."""
            await self._post_payload(payload, method, extra_headers)
            return await future

        try:
            response = await asyncio.wait_for(
                _round_trip(), timeout=timeout_s or self._request_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise self._error(
                f"External SSE request timed out for method '{method}'",
                reason_code="request_timeout",
                method=method,
            ) from exc
        finally:
            self._pending.pop(request_id, None)

        if response.get("error") and raise_on_error:
            raise self._error(
                f"External SSE request failed for method '{method}'",
                reason_code="upstream_error",
                method=method,
            )
        return response

    async def _post_notification(self, method: str, params: dict[str, Any]) -> None:
        """POST one JSON-RPC notification to the message endpoint."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": deepcopy(params or {}),
        }
        await self._post_payload(payload, method, None)

    async def _post_payload(
        self,
        payload: dict[str, Any],
        method: str,
        extra_headers: dict[str, str] | None,
    ) -> None:
        """POST one JSON-RPC payload to the resolved message endpoint."""
        client = self._client
        endpoint = self._endpoint_url
        if client is None or endpoint is None:
            raise self._error(
                "External SSE transport is not connected",
                reason_code="not_connected",
                method=method,
            )
        headers = {
            "content-type": "application/json",
            **self._static_headers,
        }
        if extra_headers:
            headers.update(extra_headers)
        encoded = self._encode_payload(payload, method)
        try:
            response = await client.post(
                endpoint,
                content=encoded,
                headers=headers,
                timeout=self._request_timeout_s,
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, method) from exc
        if response.status_code >= 400:
            raise self._status_error(response.status_code, method)

    async def _teardown(self) -> None:
        """Cancel the reader, close stream and client, and fail pending requests."""
        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader_task
        stream_response = self._stream_response
        self._stream_response = None
        self._events = None
        if stream_response is not None:
            with contextlib.suppress(Exception):
                await stream_response.aclose()
        client = self._client
        self._client = None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()
        self._endpoint_url = None
        self._initialized = False
        self._fail_pending("connection_closed")


def create_http_external_transport(
    server: ExternalServerDefinition,
) -> StreamableHttpExternalTransport | SseExternalTransport:
    """Create a URL-targeted external transport for a supported server definition.

    Args:
        server: The stored external server definition to connect to.

    Returns:
        A Streamable HTTP or SSE transport matching ``server.transport``.

    Raises:
        HttpExternalTransportError: If the definition's transport is not a
            supported HTTP transport.
    """
    if server.transport == "streamable_http":
        return StreamableHttpExternalTransport(server)
    if server.transport == "sse":
        return SseExternalTransport(server)
    raise HttpExternalTransportError(
        "External server transport is not supported by the HTTP factory",
        reason_code="unsupported_transport",
        server_id=server.id,
    )


__all__ = [
    "HttpExternalTransportError",
    "SseExternalTransport",
    "StreamableHttpExternalTransport",
    "create_http_external_transport",
]
