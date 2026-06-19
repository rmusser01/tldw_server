"""Transport adapters for MCP smoke JSON-RPC flows."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Collection
from contextlib import suppress
from typing import Any

import httpx
import websockets

from mcp_unified.gateway.jsonrpc import (
    GATEWAY_RESPONSE_TYPES,
    GatewayNoResponse,
    handle_jsonrpc,
    response_to_json,
)
from mcp_unified.gateway.runtime import GatewayRuntime
from mcp_unified.smoke.exceptions import McpSmokeTransportError
from mcp_unified.smoke.reporting import redact_detail
from mcp_unified.smoke.types import JsonObject, JsonRpcPayload, McpSmokeTransport

_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
_RETRYABLE_HTTP_STATUS_CODES = frozenset({500, 502, 503, 504})
_DEFAULT_IDEMPOTENT_HTTP_METHODS = frozenset(
    {
        "initialize",
        "ping",
        "tools/list",
        "resources/list",
        "prompts/list",
    }
)
_PROFILE_HTTP_HEADER_NAMES = frozenset(
    {
        "x-mcp-profile",
        "x-mcp-profile-id",
    }
)
_DEFAULT_WEBSOCKET_TIMEOUT_SECONDS = 10.0
_DEFAULT_STDIO_TIMEOUT_SECONDS = 10.0
_DEFAULT_STDERR_MAX_BYTES = 8192
_DEFAULT_RESPONSE_MAX_BYTES = 1024 * 1024
_STDIO_CLOSE_TIMEOUT_SECONDS = 1.0
_IGNORE_STDIO_RESPONSE = object()


class InProcessGatewayTransport:
    """Call the standalone gateway JSON-RPC handler directly in-process."""

    def __init__(
        self,
        runtime: GatewayRuntime,
        *,
        path: str = "inprocess://mcp/request",
        client_host: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.path = path
        self.client_host = client_host
        self.metadata = dict(metadata or {})
        self._started = False

    async def start(self) -> None:
        """Mark the no-op in-process transport as started."""

        self._started = True

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Dispatch one JSON-RPC payload through `handle_jsonrpc`."""

        response = await handle_jsonrpc(
            self.runtime,
            payload,
            path=self.path,
            client_host=self.client_host,
            metadata=self.metadata,
        )
        return _gateway_response_to_plain_json(response)

    async def notify(self, payload: JsonObject) -> object | None:
        """Dispatch a notification payload and return any observed response."""

        return await self.request(payload)

    async def close(self) -> None:
        """Mark the no-op in-process transport as closed."""

        self._started = False


class InProcessFastApiTransport:
    """Call a FastAPI MCP JSON-RPC route through `httpx.ASGITransport`."""

    def __init__(
        self,
        app: Any,
        *,
        request_path: str = "/api/v1/mcp/request",
        base_url: str = "http://mcp-smoke.local",
        headers: dict[str, str] | None = None,
        response_max_bytes: int = _DEFAULT_RESPONSE_MAX_BYTES,
    ) -> None:
        self.app = app
        self.request_path = request_path
        self.base_url = base_url
        self.headers = dict(headers or {})
        self.response_max_bytes = max(1, int(response_max_bytes))
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the ASGI-backed async HTTP client."""

        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url=self.base_url,
            headers=self.headers,
        )

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """POST one JSON-RPC request or batch to the configured ASGI route."""

        client = await self._started_client()
        response = await client.post(self.request_path, json=payload)
        if response.status_code == 204:
            return None
        response.raise_for_status()
        if not response.content:
            return None
        _ensure_response_size(
            len(response.content),
            self.response_max_bytes,
            method=_single_payload_method(payload),
        )
        try:
            return response.json()
        except ValueError as exc:
            raise McpSmokeTransportError(
                "transport_invalid_json_response",
                "In-process FastAPI transport received a non-JSON response body",
                method=_single_payload_method(payload),
                status_code=response.status_code,
                cause=exc,
            ) from exc

    async def notify(self, payload: JsonObject) -> object | None:
        """POST one notification to the configured ASGI route."""

        return await self.request(payload)

    async def close(self) -> None:
        """Close the ASGI-backed async HTTP client."""

        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def _started_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        if self._client is None:  # pragma: no cover - defensive guard
            raise RuntimeError("In-process FastAPI transport failed to start")
        return self._client


class LiveHttpTransport:
    """POST JSON-RPC payloads to a live MCP HTTP endpoint."""

    def __init__(
        self,
        url: str,
        *,
        bearer_token: str | None = None,
        api_key: str | None = None,
        profile_id: str | None = None,
        timeout: float | httpx.Timeout | None = _DEFAULT_HTTP_TIMEOUT_SECONDS,
        max_retries: int = 0,
        retry_methods: Collection[str] | None = None,
        retry_idempotent_methods: bool = False,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        response_max_bytes: int = _DEFAULT_RESPONSE_MAX_BYTES,
    ) -> None:
        self.url = url
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.profile_id = profile_id
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_methods = set(retry_methods or ())
        if retry_idempotent_methods:
            self.retry_methods.update(_DEFAULT_IDEMPOTENT_HTTP_METHODS)
        self.headers = dict(headers or {})
        self._client = http_client
        self.response_max_bytes = max(1, int(response_max_bytes))

    async def start(self) -> None:
        """Create the async HTTP client when one was not injected."""

        if self._client is not None:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """POST one JSON-RPC request or batch and return the decoded response."""

        return await self._post(payload)

    async def notify(self, payload: JsonObject) -> object | None:
        """POST one JSON-RPC notification and accept HTTP 204 no-content success."""

        return await self._post(payload)

    async def close(self) -> None:
        """Close the async HTTP client."""

        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def _post(self, payload: JsonRpcPayload) -> object | None:
        attempt = 0
        while True:
            try:
                response = await (await self._started_client()).post(
                    self.url,
                    json=payload,
                    headers=self._request_headers(),
                    timeout=self.timeout,
                )
            except httpx.RequestError as exc:
                if self._should_retry_request_error(exc, payload, attempt=attempt):
                    attempt += 1
                    continue
                raise self._request_error(exc, payload, attempt=attempt) from exc

            if response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                if self._can_retry_payload(payload, attempt=attempt):
                    attempt += 1
                    continue
                if attempt < self.max_retries:
                    raise self._retry_skipped_error(
                        payload,
                        status_code=response.status_code,
                    )

            return self._decode_response(response, payload)

    async def _started_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        if self._client is None:  # pragma: no cover - defensive guard
            raise McpSmokeTransportError(
                "transport_start_failed",
                "HTTP transport failed to create an async client",
            )
        return self._client

    def _request_headers(self) -> httpx.Headers:
        headers = httpx.Headers({"accept": "application/json"})
        headers.update(self.headers)
        if self.profile_id:
            self._delete_headers(headers, _PROFILE_HTTP_HEADER_NAMES)
            headers["x-mcp-profile"] = self.profile_id
        if self.bearer_token:
            self._delete_headers(headers, ("authorization",))
            headers["Authorization"] = self._authorization_header(self.bearer_token)
        if self.api_key:
            self._delete_headers(headers, ("x-api-key",))
            headers["X-API-KEY"] = self.api_key
        return headers

    def _should_retry_request_error(
        self,
        exc: httpx.RequestError,
        payload: JsonRpcPayload,
        *,
        attempt: int,
    ) -> bool:
        if attempt >= self.max_retries:
            return False
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        return self._payload_is_retryable(payload)

    def _can_retry_payload(self, payload: JsonRpcPayload, *, attempt: int) -> bool:
        return attempt < self.max_retries and self._payload_is_retryable(payload)

    def _payload_is_retryable(self, payload: JsonRpcPayload) -> bool:
        methods = self._payload_methods(payload)
        return bool(methods) and all(method in self.retry_methods for method in methods)

    def _request_error(
        self,
        exc: httpx.RequestError,
        payload: JsonRpcPayload,
        *,
        attempt: int,
    ) -> McpSmokeTransportError:
        if attempt < self.max_retries and not isinstance(
            exc,
            (httpx.ConnectError, httpx.ConnectTimeout),
        ):
            return self._retry_skipped_error(payload, cause=exc)
        return McpSmokeTransportError(
            "transport_http_request_failed",
            "HTTP transport request failed",
            method=self._single_payload_method(payload),
            cause=exc,
        )

    def _retry_skipped_error(
        self,
        payload: JsonRpcPayload,
        *,
        status_code: int | None = None,
        cause: BaseException | None = None,
    ) -> McpSmokeTransportError:
        return McpSmokeTransportError(
            "transport_retry_skipped_non_idempotent",
            "retry suppressed because the transmitted JSON-RPC method is not "
            "configured as idempotent",
            method=self._single_payload_method(payload),
            status_code=status_code,
            cause=cause,
        )

    def _decode_response(
        self,
        response: httpx.Response,
        payload: JsonRpcPayload,
    ) -> object | None:
        if response.status_code == 204:
            return None
        if response.status_code >= 400:
            raise McpSmokeTransportError(
                "transport_http_status",
                "HTTP transport received an error status",
                method=self._single_payload_method(payload),
                status_code=response.status_code,
            )
        if not response.content:
            return None
        _ensure_response_size(
            len(response.content),
            self.response_max_bytes,
            method=self._single_payload_method(payload),
        )
        try:
            return response.json()
        except ValueError as exc:
            raise McpSmokeTransportError(
                "transport_invalid_json_response",
                "HTTP transport received a non-JSON response body",
                method=self._single_payload_method(payload),
                status_code=response.status_code,
                cause=exc,
            ) from exc

    @staticmethod
    def _authorization_header(token: str) -> str:
        stripped = token.strip()
        if stripped.lower().startswith("bearer "):
            return stripped
        return f"Bearer {stripped}"

    @staticmethod
    def _delete_headers(headers: httpx.Headers, names: Collection[str]) -> None:
        for name in names:
            if name in headers:
                del headers[name]

    @staticmethod
    def _payload_methods(payload: JsonRpcPayload) -> list[str]:
        if isinstance(payload, dict):
            method = payload.get("method")
            return [method] if isinstance(method, str) else []
        methods: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                return []
            method = item.get("method")
            if not isinstance(method, str):
                return []
            methods.append(method)
        return methods

    @classmethod
    def _single_payload_method(cls, payload: JsonRpcPayload) -> str | None:
        methods = cls._payload_methods(payload)
        if len(methods) == 1:
            return methods[0]
        if methods:
            return ",".join(methods)
        return None


class LiveWebSocketTransport:
    """Exchange JSON-RPC payloads with a live MCP WebSocket endpoint."""

    def __init__(
        self,
        url: str,
        *,
        bearer_token: str | None = None,
        api_key: str | None = None,
        profile_id: str | None = None,
        timeout: float | None = _DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
        headers: dict[str, str] | None = None,
        response_max_bytes: int = _DEFAULT_RESPONSE_MAX_BYTES,
    ) -> None:
        self.url = url
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.profile_id = profile_id
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.response_max_bytes = max(1, int(response_max_bytes))
        self._connection: Any | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._pending: dict[object, asyncio.Future[object]] = {}
        self._start_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._receive_error: McpSmokeTransportError | None = None
        self._closing = False

    async def start(self) -> None:
        """Open one WebSocket connection for the current smoke scenario."""

        async with self._start_lock:
            if self._connection is not None:
                if self._receive_error is not None:
                    raise self._receive_error
                return

            self._closing = False
            self._receive_error = None
            try:
                self._connection = await websockets.connect(
                    self.url,
                    **self._connect_kwargs(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise McpSmokeTransportError(
                    "transport_websocket_connect_failed",
                    "WebSocket transport failed to connect",
                    cause=exc,
                ) from exc

            self._receiver_task = asyncio.create_task(self._receive_loop())

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Send one JSON-RPC request or batch and await correlated responses."""

        connection = await self._started_connection()
        request_ids = self._payload_request_ids(payload)
        futures = self._register_pending(request_ids)
        try:
            await self._send_payload(connection, payload)
            if not futures:
                return None
            if len(futures) == 1:
                return await self._wait_for_response(
                    futures[0],
                    request_ids=request_ids,
                    payload=payload,
                )
            return await self._wait_for_batch_responses(
                futures,
                request_ids=request_ids,
                payload=payload,
            )
        except Exception:
            self._discard_pending(request_ids)
            raise

    async def notify(self, payload: JsonObject) -> object | None:
        """Send one JSON-RPC notification without awaiting a response."""

        connection = await self._started_connection()
        await self._send_payload(connection, payload)
        return None

    async def close(self) -> None:
        """Close the WebSocket connection and cancel receive processing."""

        self._closing = True
        pending_error = McpSmokeTransportError(
            "transport_websocket_closed",
            "WebSocket transport closed before all responses were received",
        )
        self._fail_pending(pending_error)

        connection = self._connection
        task = self._receiver_task
        self._connection = None
        self._receiver_task = None

        if connection is not None:
            with suppress(Exception):
                await connection.close()

        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self._closing = False

    async def _started_connection(self) -> Any:
        if self._receive_error is not None:
            raise self._receive_error
        if self._connection is None:
            await self.start()
        if self._connection is None:  # pragma: no cover - defensive guard
            raise McpSmokeTransportError(
                "transport_start_failed",
                "WebSocket transport failed to create a connection",
            )
        return self._connection

    async def _send_payload(self, connection: Any, payload: JsonRpcPayload) -> None:
        try:
            encoded = json.dumps(payload, separators=(",", ":"))
            async with self._send_lock:
                await connection.send(encoded)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert unexpected send failures.
            raise McpSmokeTransportError(
                "transport_websocket_send_failed",
                "WebSocket transport failed to send a JSON-RPC payload",
                method=self._single_payload_method(payload),
                cause=exc,
            ) from exc

    async def _wait_for_response(
        self,
        future: asyncio.Future[object],
        *,
        request_ids: list[object],
        payload: JsonRpcPayload,
    ) -> object:
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except TimeoutError as exc:
            raise self._timeout_error(request_ids, payload, cause=exc) from exc

    async def _wait_for_batch_responses(
        self,
        futures: list[asyncio.Future[object]],
        *,
        request_ids: list[object],
        payload: JsonRpcPayload,
    ) -> list[object]:
        try:
            return await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=self.timeout,
            )
        except TimeoutError as exc:
            raise self._timeout_error(request_ids, payload, cause=exc) from exc

    async def _receive_loop(self) -> None:
        try:
            async for message in self._connection:
                self._handle_frame(message)
        except asyncio.CancelledError:
            raise
        except McpSmokeTransportError as exc:
            self._receive_error = exc
            self._fail_pending(exc)
            await self._close_after_receive_error()
        except websockets.exceptions.ConnectionClosedError as exc:
            if _websocket_close_code(exc) == 1009:
                error = McpSmokeTransportError(
                    "response_too_large",
                    "WebSocket transport response exceeded max byte size",
                    cause=exc,
                )
            else:
                error = McpSmokeTransportError(
                    "transport_websocket_receive_failed",
                    "WebSocket transport failed while receiving frames",
                    cause=exc,
                )
            self._receive_error = error
            self._fail_pending(error)
            await self._close_after_receive_error()
        except Exception as exc:  # noqa: BLE001 - convert unexpected receive failures.
            error = McpSmokeTransportError(
                "transport_websocket_receive_failed",
                "WebSocket transport failed while receiving frames",
                cause=exc,
            )
            self._receive_error = error
            self._fail_pending(error)
            await self._close_after_receive_error()
        else:
            if self._pending and not self._closing:
                error = McpSmokeTransportError(
                    "transport_websocket_closed",
                    "WebSocket connection closed before pending responses completed",
                )
                self._receive_error = error
                self._fail_pending(error)
        finally:
            if not self._closing:
                self._connection = None
                self._receiver_task = None

    async def _close_after_receive_error(self) -> None:
        connection = self._connection
        if connection is not None:
            with suppress(Exception):
                await connection.close()

    def _handle_frame(self, message: object) -> None:
        if not isinstance(message, (str, bytes, bytearray)):
            raise McpSmokeTransportError(
                "transport_invalid_json_response",
                "WebSocket transport received a non-text JSON-RPC frame",
            )
        _ensure_response_size(
            _frame_byte_count(message),
            self.response_max_bytes,
            method=None,
        )
        try:
            decoded = json.loads(message)
        except (TypeError, ValueError) as exc:
            raise McpSmokeTransportError(
                "transport_invalid_json_response",
                "WebSocket transport received a non-JSON frame",
                cause=exc,
            ) from exc

        if isinstance(decoded, list):
            if not decoded:
                raise McpSmokeTransportError(
                    "transport_malformed_websocket_frame",
                    "WebSocket transport received an empty JSON-RPC batch frame",
                )
            for item in decoded:
                self._handle_message(item)
            return

        self._handle_message(decoded)

    def _handle_message(self, message: object) -> None:
        if not isinstance(message, dict):
            raise McpSmokeTransportError(
                "transport_malformed_websocket_frame",
                "WebSocket transport received a non-object JSON-RPC message",
            )
        if self._is_server_notification(message):
            return
        if not self._is_response_message(message):
            raise McpSmokeTransportError(
                "transport_malformed_websocket_frame",
                "WebSocket transport received a malformed JSON-RPC message",
            )

        request_id = message["id"]
        future = self._pending.pop(request_id, None)
        if future is None:
            raise McpSmokeTransportError(
                "transport_unexpected_websocket_response",
                "WebSocket transport received a response for an unknown request id",
            )
        if not future.done():
            future.set_result(message)

    def _register_pending(
        self,
        request_ids: list[object],
    ) -> list[asyncio.Future[object]]:
        seen_ids: set[object] = set()
        for request_id in request_ids:
            if request_id in seen_ids or request_id in self._pending:
                raise McpSmokeTransportError(
                    "transport_duplicate_request_id",
                    "WebSocket transport cannot send duplicate pending request ids",
                )
            seen_ids.add(request_id)

        futures: list[asyncio.Future[object]] = []
        loop = asyncio.get_running_loop()
        for request_id in request_ids:
            future: asyncio.Future[object] = loop.create_future()
            self._pending[request_id] = future
            futures.append(future)
        return futures

    def _discard_pending(self, request_ids: list[object]) -> None:
        for request_id in request_ids:
            future = self._pending.pop(request_id, None)
            if future is not None and not future.done():
                future.cancel()

    def _fail_pending(self, error: McpSmokeTransportError) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    def _timeout_error(
        self,
        request_ids: list[object],
        payload: JsonRpcPayload,
        *,
        cause: BaseException,
    ) -> McpSmokeTransportError:
        return McpSmokeTransportError(
            "transport_websocket_response_timeout",
            "Timed out waiting for WebSocket JSON-RPC response ids "
            f"{', '.join(map(str, request_ids))}",
            method=self._single_payload_method(payload),
            cause=cause,
        )

    def _connect_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        signature = inspect.signature(websockets.connect)
        parameters = signature.parameters

        headers = self._request_headers()
        if headers:
            if "additional_headers" in parameters:
                kwargs["additional_headers"] = headers
            elif "extra_headers" in parameters:
                kwargs["extra_headers"] = headers

        if "open_timeout" in parameters:
            kwargs["open_timeout"] = self.timeout
        if "max_size" in parameters:
            kwargs["max_size"] = self.response_max_bytes
        if "proxy" in parameters:
            kwargs["proxy"] = None
        return kwargs

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.profile_id:
            self._delete_headers(headers, _PROFILE_HTTP_HEADER_NAMES)
            headers["x-mcp-profile"] = self.profile_id
        if self.bearer_token:
            self._delete_header(headers, "authorization")
            headers["Authorization"] = LiveHttpTransport._authorization_header(
                self.bearer_token
            )
        if self.api_key:
            self._delete_header(headers, "x-api-key")
            headers["X-API-KEY"] = self.api_key
        return headers

    @staticmethod
    def _delete_header(headers: dict[str, str], target: str) -> None:
        for name in list(headers):
            if name.lower() == target:
                del headers[name]

    @classmethod
    def _delete_headers(cls, headers: dict[str, str], targets: Collection[str]) -> None:
        for target in targets:
            cls._delete_header(headers, target)

    @staticmethod
    def _payload_request_ids(payload: JsonRpcPayload) -> list[object]:
        if isinstance(payload, dict):
            return [payload["id"]] if "id" in payload else []

        request_ids: list[object] = []
        seen_ids: set[object] = set()
        for item in payload:
            if not isinstance(item, dict) or "id" not in item:
                continue
            request_id = item["id"]
            if request_id in seen_ids:
                raise McpSmokeTransportError(
                    "transport_duplicate_request_id",
                    "WebSocket transport cannot send duplicate request ids",
                )
            seen_ids.add(request_id)
            request_ids.append(request_id)
        return request_ids

    @staticmethod
    def _is_response_message(message: dict[str, object]) -> bool:
        if message.get("jsonrpc") != "2.0" or "id" not in message:
            return False
        has_result = "result" in message
        has_error = "error" in message
        return has_result != has_error

    @staticmethod
    def _is_server_notification(message: dict[str, object]) -> bool:
        if message.get("jsonrpc") != "2.0" or "id" in message:
            return False
        if "result" in message or "error" in message:
            return False
        return isinstance(message.get("method"), str)

    @staticmethod
    def _single_payload_method(payload: JsonRpcPayload) -> str | None:
        if isinstance(payload, dict):
            method = payload.get("method")
            return method if isinstance(method, str) else None
        methods = [
            item.get("method")
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("method"), str)
        ]
        if len(methods) == 1:
            return methods[0]
        if methods:
            return ",".join(methods)
        return None


class StdioSubprocessTransport:
    """Exchange newline-delimited JSON-RPC payloads with a subprocess over stdio."""

    def __init__(
        self,
        command: str,
        *,
        args: list[str] | tuple[str, ...] | None = None,
        cwd: str | None = None,
        env_allowlist: Collection[str] | None = None,
        startup_timeout: float | None = _DEFAULT_STDIO_TIMEOUT_SECONDS,
        request_timeout: float | None = _DEFAULT_STDIO_TIMEOUT_SECONDS,
        stderr_max_bytes: int = _DEFAULT_STDERR_MAX_BYTES,
        response_max_bytes: int = _DEFAULT_RESPONSE_MAX_BYTES,
    ) -> None:
        self.command = command
        self.args = tuple(args or ())
        self.cwd = cwd
        self.env_allowlist = tuple(env_allowlist or ())
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.stderr_max_bytes = max(0, int(stderr_max_bytes))
        self.response_max_bytes = max(1, int(response_max_bytes))
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_buffer = bytearray()
        self._stderr_truncated = False
        self._start_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the subprocess using argv execution."""

        async with self._start_lock:
            if self._process is not None and self._process.returncode is None:
                return
            self._stderr_buffer.clear()
            self._stderr_truncated = False
            try:
                self._process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        self.command,
                        *self.args,
                        cwd=self.cwd,
                        env=self._subprocess_env(),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=self.startup_timeout,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                raise McpSmokeTransportError(
                    "transport_stdio_start_timeout",
                    "Timed out starting stdio subprocess",
                    cause=exc,
                ) from exc
            except Exception as exc:
                raise McpSmokeTransportError(
                    "transport_stdio_start_failed",
                    "Failed to start stdio subprocess",
                    cause=exc,
                ) from exc

            self._stderr_task = asyncio.create_task(self._capture_stderr())

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Send one JSON-RPC line and parse one response line when required."""

        request_ids = self._payload_request_ids(payload)
        try:
            process = await self._started_process()
            async with self._request_lock:
                await self._send_payload(process, payload)
                if not request_ids:
                    return None
                return await self._read_response(process, request_ids, payload)
        except asyncio.CancelledError:
            await self.close()
            raise
        except McpSmokeTransportError:
            await self.close()
            raise
        except Exception as exc:
            error = self._stdio_error(
                "transport_stdio_request_failed",
                "Stdio subprocess request failed",
                payload,
                cause=exc,
            )
            await self.close()
            raise error from exc

    async def notify(self, payload: JsonObject) -> object | None:
        """Send one JSON-RPC notification line."""

        return await self.request(payload)

    async def close(self) -> None:
        """Close pipes and terminate the subprocess if it is still running."""

        process = self._process
        stderr_task = self._stderr_task
        self._process = None
        self._stderr_task = None

        if process is not None:
            await self._close_stdin(process)
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(
                        process.wait(),
                        timeout=_STDIO_CLOSE_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    process.kill()
                    with suppress(Exception):
                        await asyncio.wait_for(
                            process.wait(),
                            timeout=_STDIO_CLOSE_TIMEOUT_SECONDS,
                        )
            else:
                with suppress(Exception):
                    await process.wait()

        if stderr_task is not None:
            try:
                await asyncio.wait_for(
                    stderr_task,
                    timeout=_STDIO_CLOSE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                stderr_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stderr_task

    async def _started_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            await self.start()
        if self._process is None:  # pragma: no cover - defensive guard
            raise McpSmokeTransportError(
                "transport_start_failed",
                "Stdio transport failed to create a subprocess",
            )
        if self._process.returncode is not None:
            raise self._stdio_error(
                "transport_stdio_process_exited",
                f"Stdio subprocess exited with code {self._process.returncode}",
                None,
            )
        return self._process

    async def _send_payload(
        self,
        process: asyncio.subprocess.Process,
        payload: JsonRpcPayload,
    ) -> None:
        if process.stdin is None:
            raise self._stdio_error(
                "transport_stdio_pipe_unavailable",
                "Stdio subprocess stdin is unavailable",
                payload,
            )
        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            process.stdin.write(encoded + b"\n")
            await process.stdin.drain()
        except Exception as exc:
            raise self._stdio_error(
                "transport_stdio_send_failed",
                "Failed to write JSON-RPC payload to stdio subprocess",
                payload,
                cause=exc,
            ) from exc

    async def _read_response(
        self,
        process: asyncio.subprocess.Process,
        request_ids: list[object],
        payload: JsonRpcPayload,
    ) -> object:
        if process.stdout is None:
            raise self._stdio_error(
                "transport_stdio_pipe_unavailable",
                "Stdio subprocess stdout is unavailable",
                payload,
            )
        try:
            return await asyncio.wait_for(
                self._read_matching_response(process, request_ids, payload),
                timeout=self.request_timeout,
            )
        except TimeoutError as exc:
            raise self._stdio_error(
                "transport_stdio_response_timeout",
                "Timed out waiting for stdio JSON-RPC response",
                payload,
                cause=exc,
            ) from exc

    async def _read_matching_response(
        self,
        process: asyncio.subprocess.Process,
        request_ids: list[object],
        payload: JsonRpcPayload,
    ) -> object:
        if process.stdout is None:
            raise self._stdio_error(
                "transport_stdio_pipe_unavailable",
                "Stdio subprocess stdout is unavailable",
                payload,
            )
        expected_ids = set(request_ids)
        while True:
            line = await process.stdout.readline()
            if not line:
                raise self._stdio_error(
                    "transport_stdio_closed",
                    "Stdio subprocess closed stdout before a JSON-RPC response",
                    payload,
                )
            if len(line) > self.response_max_bytes:
                raise self._stdio_error(
                    "response_too_large",
                    "Stdio subprocess response exceeded max byte size "
                    f"({len(line)}>{self.response_max_bytes})",
                    payload,
                )
            response = self._decode_response_line(line, payload)
            matched = self._match_response(response, expected_ids, payload)
            if matched is _IGNORE_STDIO_RESPONSE:
                continue
            return matched

    def _match_response(
        self,
        response: object,
        expected_ids: set[object],
        payload: JsonRpcPayload,
    ) -> object:
        if isinstance(response, dict):
            if self._is_server_notification(response):
                return _IGNORE_STDIO_RESPONSE
            self._validate_response_message(response, payload)
            response_id = response["id"]
            if response_id not in expected_ids:
                raise self._stdio_error(
                    "transport_unexpected_stdio_response",
                    "Stdio subprocess emitted a response for an unexpected request id",
                    payload,
                )
            if len(expected_ids) != 1:
                raise self._stdio_error(
                    "transport_incomplete_stdio_batch_response",
                    "Stdio subprocess emitted a single response for a batch request",
                    payload,
                )
            return response

        if isinstance(response, list):
            if not response:
                raise self._stdio_error(
                    "transport_malformed_stdio_response",
                    "Stdio subprocess emitted an empty JSON-RPC batch response",
                    payload,
                )
            matched_responses: list[dict[str, object]] = []
            seen_ids: set[object] = set()
            for item in response:
                if not isinstance(item, dict):
                    raise self._stdio_error(
                        "transport_malformed_stdio_response",
                        "Stdio subprocess emitted a non-object batch response item",
                        payload,
                    )
                if self._is_server_notification(item):
                    continue
                self._validate_response_message(item, payload)
                response_id = item["id"]
                if response_id not in expected_ids:
                    raise self._stdio_error(
                        "transport_unexpected_stdio_response",
                        "Stdio subprocess emitted a batch response for an unexpected request id",
                        payload,
                    )
                if response_id in seen_ids:
                    raise self._stdio_error(
                        "transport_malformed_stdio_response",
                        "Stdio subprocess emitted duplicate batch response ids",
                        payload,
                    )
                seen_ids.add(response_id)
                matched_responses.append(item)
            if not matched_responses:
                return _IGNORE_STDIO_RESPONSE
            if seen_ids != expected_ids:
                raise self._stdio_error(
                    "transport_incomplete_stdio_batch_response",
                    "Stdio subprocess emitted an incomplete batch response",
                    payload,
                )
            return matched_responses

        raise self._stdio_error(
            "transport_malformed_stdio_response",
            "Stdio subprocess emitted a non-object JSON-RPC response",
            payload,
        )

    def _validate_response_message(
        self,
        response: dict[str, object],
        payload: JsonRpcPayload,
    ) -> None:
        if response.get("jsonrpc") != "2.0" or "id" not in response:
            raise self._stdio_error(
                "transport_malformed_stdio_response",
                "Stdio subprocess emitted a malformed JSON-RPC response",
                payload,
            )
        has_result = "result" in response
        has_error = "error" in response
        if has_result == has_error:
            raise self._stdio_error(
                "transport_malformed_stdio_response",
                "Stdio subprocess response must contain exactly one of result or error",
                payload,
            )

    def _decode_response_line(self, line: bytes, payload: JsonRpcPayload) -> object:
        try:
            return json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise self._stdio_error(
                "transport_invalid_json_response",
                "Stdio subprocess emitted a non-JSON response line",
                payload,
                cause=exc,
            ) from exc

    @staticmethod
    def _is_server_notification(message: dict[str, object]) -> bool:
        if message.get("jsonrpc") != "2.0" or "id" in message:
            return False
        if "result" in message or "error" in message:
            return False
        return isinstance(message.get("method"), str)

    async def _capture_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(1024)
            if not chunk:
                return
            remaining = self.stderr_max_bytes - len(self._stderr_buffer)
            if remaining > 0:
                self._stderr_buffer.extend(chunk[:remaining])
            if len(chunk) > max(0, remaining):
                self._stderr_truncated = True

    async def _close_stdin(self, process: asyncio.subprocess.Process) -> None:
        if process.stdin is None:
            return
        with suppress(Exception):
            process.stdin.close()
            await process.stdin.wait_closed()

    def _subprocess_env(self) -> dict[str, str]:
        return {name: os.environ[name] for name in self.env_allowlist if name in os.environ}

    def _stderr_detail(self) -> str | None:
        if not self._stderr_buffer and not self._stderr_truncated:
            return None
        text = self._stderr_buffer.decode("utf-8", errors="replace")
        if self._stderr_truncated:
            text = f"{text}...[stderr truncated]"
        return str(redact_detail(text))

    def _stdio_error(
        self,
        reason_code: str,
        message: str,
        payload: JsonRpcPayload | None,
        *,
        cause: BaseException | None = None,
    ) -> McpSmokeTransportError:
        stderr = self._stderr_detail()
        if stderr:
            message = f"{message} stderr={stderr}"
        return McpSmokeTransportError(
            reason_code,
            message,
            method=self._single_payload_method(payload),
            cause=cause,
        )

    @staticmethod
    def _payload_request_ids(payload: JsonRpcPayload) -> list[object]:
        if isinstance(payload, dict):
            return [payload["id"]] if "id" in payload else []

        request_ids: list[object] = []
        seen_ids: set[object] = set()
        for item in payload:
            if not isinstance(item, dict) or "id" not in item:
                continue
            request_id = item["id"]
            if request_id in seen_ids:
                raise McpSmokeTransportError(
                    "transport_duplicate_request_id",
                    "Stdio transport cannot send duplicate request ids",
                )
            seen_ids.add(request_id)
            request_ids.append(request_id)
        return request_ids

    @staticmethod
    def _single_payload_method(payload: JsonRpcPayload | None) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            method = payload.get("method")
            return method if isinstance(method, str) else None
        methods = [
            item.get("method")
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("method"), str)
        ]
        if len(methods) == 1:
            return methods[0]
        if methods:
            return ",".join(methods)
        return None


def _gateway_response_to_plain_json(response: object) -> object | None:
    if isinstance(response, GatewayNoResponse):
        return None
    if isinstance(response, list):
        return [_gateway_response_to_plain_json(item) for item in response]
    if isinstance(response, GATEWAY_RESPONSE_TYPES):
        return response_to_json(response)
    return response


def _ensure_response_size(
    byte_count: int,
    max_bytes: int,
    *,
    method: str | None,
) -> None:
    if byte_count <= max_bytes:
        return
    raise McpSmokeTransportError(
        "response_too_large",
        "JSON-RPC response exceeded max byte size "
        f"({byte_count}>{max_bytes})",
        method=method,
    )


def _frame_byte_count(message: str | bytes | bytearray) -> int:
    if isinstance(message, str):
        return len(message.encode("utf-8"))
    return len(message)


def _single_payload_method(payload: JsonRpcPayload) -> str | None:
    if isinstance(payload, dict):
        method = payload.get("method")
        return method if isinstance(method, str) else None
    methods = [
        item.get("method")
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("method"), str)
    ]
    if len(methods) == 1:
        return methods[0]
    if methods:
        return ",".join(methods)
    return None


def _websocket_close_code(exc: websockets.exceptions.ConnectionClosedError) -> int | None:
    """Return the WebSocket close code when the peer or client supplied one."""

    for close_frame in (exc.rcvd, exc.sent):
        if close_frame is not None:
            return close_frame.code
    return None


__all__ = [
    "InProcessFastApiTransport",
    "InProcessGatewayTransport",
    "JsonObject",
    "JsonRpcPayload",
    "LiveHttpTransport",
    "LiveWebSocketTransport",
    "McpSmokeTransport",
    "McpSmokeTransportError",
    "StdioSubprocessTransport",
]
