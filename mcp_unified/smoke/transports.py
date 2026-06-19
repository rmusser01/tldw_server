"""Transport adapters for MCP smoke JSON-RPC flows."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol

import httpx

from mcp_unified.gateway.jsonrpc import (
    GATEWAY_RESPONSE_TYPES,
    GatewayNoResponse,
    handle_jsonrpc,
    response_to_json,
)
from mcp_unified.gateway.runtime import GatewayRuntime

JsonObject = dict[str, object]
JsonRpcPayload = JsonObject | list[object]
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


class McpSmokeTransport(Protocol):
    """Async transport protocol consumed by MCP smoke scenarios."""

    async def start(self) -> None:
        """Open any resources needed by this transport."""

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Send a JSON-RPC request or batch payload and return the decoded response."""

    async def notify(self, payload: JsonObject) -> None:
        """Send a JSON-RPC notification payload."""

    async def close(self) -> None:
        """Release any resources opened by this transport."""


class McpSmokeTransportError(RuntimeError):
    """Raised when a smoke transport cannot exchange JSON-RPC payloads."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        method: str | None = None,
        status_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        parts = [reason_code, message]
        if method is not None:
            parts.append(f"method={method}")
        if status_code is not None:
            parts.append(f"status_code={status_code}")
        super().__init__(": ".join(parts))
        self.reason_code = reason_code
        self.method = method
        self.status_code = status_code
        self.__cause__ = cause


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

    async def notify(self, payload: JsonObject) -> None:
        """Dispatch a notification payload and ignore the no-response sentinel."""

        await self.request(payload)

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
    ) -> None:
        self.app = app
        self.request_path = request_path
        self.base_url = base_url
        self.headers = dict(headers or {})
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
        return response.json()

    async def notify(self, payload: JsonObject) -> None:
        """POST one notification to the configured ASGI route."""

        await self.request(payload)

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

    async def start(self) -> None:
        """Create the async HTTP client when one was not injected."""

        if self._client is not None:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """POST one JSON-RPC request or batch and return the decoded response."""

        return await self._post(payload)

    async def notify(self, payload: JsonObject) -> None:
        """POST one JSON-RPC notification and accept HTTP 204 no-content success."""

        await self._post(payload)

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

    def _request_headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        headers.update(self.headers)
        if self.profile_id:
            headers["x-mcp-profile"] = self.profile_id
        if self.bearer_token:
            headers["Authorization"] = self._authorization_header(self.bearer_token)
        if self.api_key:
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


def _gateway_response_to_plain_json(response: object) -> object | None:
    if isinstance(response, GatewayNoResponse):
        return None
    if isinstance(response, list):
        return [_gateway_response_to_plain_json(item) for item in response]
    if isinstance(response, GATEWAY_RESPONSE_TYPES):
        return response_to_json(response)
    return response


__all__ = [
    "InProcessFastApiTransport",
    "InProcessGatewayTransport",
    "JsonObject",
    "JsonRpcPayload",
    "LiveHttpTransport",
    "McpSmokeTransport",
    "McpSmokeTransportError",
]
