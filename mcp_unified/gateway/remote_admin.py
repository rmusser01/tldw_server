"""Remote administration client for a running standalone MCP gateway."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RemoteGatewayAdminConfig:
    """Configuration for remote standalone gateway admin requests."""

    gateway_url: str
    admin_header_name: str = "X-MCP-Gateway-Admin-Key"
    admin_key: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate and normalize remote gateway client settings."""

        gateway_url = self.gateway_url.strip().rstrip("/")
        if not gateway_url:
            raise ValueError("gateway_url is required")

        parsed = urllib.parse.urlparse(gateway_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("gateway_url must be an http or https URL")

        admin_header_name = self.admin_header_name.strip()
        if not admin_header_name:
            raise ValueError("admin_header_name is required")
        if "\r" in admin_header_name or "\n" in admin_header_name:
            raise ValueError("admin_header_name cannot contain line breaks")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        admin_key = self.admin_key.strip() if self.admin_key is not None else None
        object.__setattr__(self, "gateway_url", gateway_url)
        object.__setattr__(self, "admin_header_name", admin_header_name)
        object.__setattr__(self, "admin_key", admin_key or None)


class RemoteGatewayAdminError(RuntimeError):
    """Raised when the remote gateway returns an error or invalid response."""

    def __init__(
        self,
        message: str,
        *,
        payload: Mapping[str, Any] | None = None,
        reason_code: str = "remote_gateway_error",
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> None:
        """Store a machine-readable error payload for CLI emission."""

        super().__init__(message)
        if payload is None:
            normalized_payload: dict[str, Any] = {
                "error": message,
                "ok": False,
                "reason_code": reason_code,
            }
            if error_type is not None:
                normalized_payload["error_type"] = error_type
        else:
            normalized_payload = dict(payload)
            normalized_payload["ok"] = False
            normalized_payload.setdefault("error", message)
            normalized_payload.setdefault("reason_code", reason_code)
            if error_type is not None:
                normalized_payload.setdefault("error_type", error_type)

        if status_code is not None:
            normalized_payload.setdefault("status_code", status_code)
        self._payload = normalized_payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        status_code: int | None = None,
    ) -> RemoteGatewayAdminError:
        """Build an error from a public gateway JSON error payload."""

        error = payload.get("error")
        message = error if isinstance(error, str) and error else "Remote gateway error"
        reason_code = payload.get("reason_code")
        return cls(
            message,
            payload=payload,
            reason_code=(
                reason_code
                if isinstance(reason_code, str) and reason_code
                else "remote_gateway_error"
            ),
            status_code=status_code,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable error payload."""

        return dict(self._payload)


class RemoteGatewayAdminClient:
    """Small stdlib HTTP client for running gateway runtime admin routes."""

    def __init__(
        self,
        config: RemoteGatewayAdminConfig,
        *,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        """Create a client using a `urllib.request.urlopen`-compatible opener."""

        self.config = config
        self._opener = opener or urllib.request.urlopen

    def endpoint_url(self, path: str) -> str:
        """Resolve an endpoint path against the configured mounted base URL."""

        return f"{self.config.gateway_url}/{path.lstrip('/')}"

    def list_runtime_servers(self) -> dict[str, Any]:
        """List runtime state for managed external servers."""

        return self._request_json("GET", "/external-servers/runtime")

    def start_server(self, server_id: str) -> dict[str, Any]:
        """Start one managed external server through the running gateway."""

        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/start",
        )

    def stop_server(self, server_id: str) -> dict[str, Any]:
        """Stop one managed external server through the running gateway."""

        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/stop",
        )

    def restart_server(self, server_id: str) -> dict[str, Any]:
        """Restart one managed external server through the running gateway."""

        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/restart",
        )

    def refresh_server(self, server_id: str | None = None) -> dict[str, Any]:
        """Refresh one external runtime, or all runtimes when no id is supplied."""

        if server_id is None:
            return self._request_json("POST", "/external-servers/refresh")
        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/refresh",
        )

    def reconcile(self, server_id: str | None = None) -> dict[str, Any]:
        """Reconcile one external runtime, or all runtimes when no id is supplied."""

        if server_id is None:
            return self._request_json("POST", "/external-servers/reconcile")
        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/reconcile",
        )

    def install_server(self, server_id: str) -> dict[str, Any]:
        """Run the configured install flow for one external server."""

        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/install",
        )

    def update_server(self, server_id: str) -> dict[str, Any]:
        """Run the configured update flow for one external server."""

        return self._request_json(
            "POST",
            f"/external-servers/{_quote_server_id(server_id)}/update",
        )

    def _request_json(self, method: str, path: str) -> dict[str, Any]:
        """Send one request and return a JSON object response."""

        request = urllib.request.Request(
            self.endpoint_url(path),
            data=b"{}" if method == "POST" else None,
            headers=self._headers(method),
            method=method,
        )
        try:
            response = self._opener(request, timeout=self.config.timeout_seconds)
            with response:
                return _parse_json_object(response.read())
        except urllib.error.HTTPError as exc:
            raise _error_from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RemoteGatewayAdminError(
                "Unable to reach gateway",
                reason_code="gateway_connection_failed",
                error_type=exc.__class__.__name__,
            ) from exc

    def _headers(self, method: str) -> dict[str, str]:
        """Build request headers without adding an empty admin credential."""

        headers = {"Accept": "application/json"}
        if method == "POST":
            headers["Content-Type"] = "application/json"
        if self.config.admin_key is not None:
            headers[self.config.admin_header_name] = self.config.admin_key
        return headers


def _quote_server_id(server_id: str) -> str:
    """Quote a server id for safe use in one path segment."""

    normalized = server_id.strip()
    if not normalized:
        raise ValueError("server_id is required")
    return urllib.parse.quote(normalized, safe="")


def _error_from_http_error(exc: urllib.error.HTTPError) -> RemoteGatewayAdminError:
    """Preserve public JSON HTTP error payloads when the gateway provides them."""

    try:
        payload = _parse_json_object(exc.read())
    except RemoteGatewayAdminError:
        return RemoteGatewayAdminError(
            "Gateway returned an HTTP error",
            reason_code="gateway_http_error",
            status_code=exc.code,
            error_type=exc.__class__.__name__,
        )
    return RemoteGatewayAdminError.from_payload(payload, status_code=exc.code)


def _parse_json_object(response_body: bytes) -> dict[str, Any]:
    """Parse a UTF-8 JSON object from a gateway response body."""

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteGatewayAdminError(
            "Gateway returned an invalid JSON object",
            reason_code="gateway_invalid_response",
        ) from exc

    if not isinstance(payload, dict):
        raise RemoteGatewayAdminError(
            "Gateway returned an invalid JSON object",
            reason_code="gateway_invalid_response",
        )
    return payload
