"""
HTTP/WebSocket guard helpers for MCP Unified.
"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import HTTPException, Request
from loguru import logger
from starlette.requests import ClientDisconnect

from ..config import get_config
from ..environment import is_test_mode
from .ip_filter import enforce_ip_allowlist as _enforce_ip_allowlist
from .ip_filter import get_ip_access_controller

_BODYLESS_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


async def enforce_request_body_limit(request: Request) -> None:
    """
    Ensure the request payload stays within configured bounds.

    Reads the body once and reuses FastAPI's internal caching so downstream
    handlers still see the payload.
    """
    cfg = get_config()
    limit = int(cfg.http_max_body_bytes or 0)
    if limit <= 0:
        return
    if request.method.upper() in _BODYLESS_METHODS:
        return

    try:
        body = await request.body()
    except ClientDisconnect:
        logger.debug(
            "Client disconnected before MCP request body could be read",
            extra={"audit": True, "path": request.url.path, "method": request.method},
        )
        return

    if len(body) > limit:
        logger.warning(
            "Rejecting MCP request exceeding size limit",
            extra={"audit": True, "path": request.url.path, "size": len(body), "limit": limit},
        )
        raise HTTPException(status_code=413, detail="Payload too large")


def enforce_client_certificate(request: Request) -> None:
    """
    Enforce client certificate headers for HTTP requests when configured.
    """
    cfg = get_config()
    if not cfg.client_cert_required:
        return

    headers = request.headers
    # Only accept client-certificate headers when the immediate peer is a trusted proxy
    controller = get_ip_access_controller()
    peer_ip = getattr(request.client, "host", None) if request.client else None
    # Allow test harness peer during TEST_MODE, otherwise require trusted proxy
    if not controller._is_trusted_proxy(peer_ip):  # type: ignore[attr-defined]
        if not (is_test_mode() and peer_ip in {"testclient", "127.0.0.1"}):
            logger.warning(
                "Rejected client-cert header from untrusted peer",
                extra={"audit": True, "peer_ip": peer_ip, "path": request.url.path},
            )
            raise HTTPException(status_code=403, detail="Client certificate required")

    if not _is_client_certificate_valid(headers, remote_addr=peer_ip):
        logger.warning(
            "Client certificate validation failed for MCP request",
            extra={"audit": True, "path": request.url.path},
        )
        raise HTTPException(status_code=403, detail="Client certificate required")


def enforce_client_certificate_headers(headers: Mapping[str, str], remote_addr: str | None = None) -> None:
    """
    Enforce client certificate headers for WebSocket connections.
    """
    cfg = get_config()
    if not cfg.client_cert_required:
        return
    # Only accept client-certificate headers when the immediate peer is a trusted proxy
    controller = get_ip_access_controller()
    if not controller._is_trusted_proxy(remote_addr):  # type: ignore[attr-defined]
        if not (is_test_mode() and remote_addr in {"testclient", "127.0.0.1"}):
            raise HTTPException(status_code=403, detail="Client certificate required")

    if not _is_client_certificate_valid(headers, remote_addr=remote_addr):
        raise HTTPException(status_code=403, detail="Client certificate required")


def _is_client_certificate_valid(headers: Mapping[str, str], remote_addr: str | None = None) -> bool:
    """
    Inspect headers for client certificate verification results.

    Supports two common patterns:
    - Header containing the literal PEM/base64 certificate (non-empty check)
    - Header containing verify status (SUCCESS/OK)
    """
    cfg = get_config()
    header_name = (cfg.client_cert_header or "x-ssl-client-verify").lower()
    expected_value = (cfg.client_cert_header_value or "").strip().lower()

    header_value: str | None = None
    for key, value in headers.items():
        if key.lower() == header_name:
            header_value = value
            break

    if header_value is None:
        return False
    # Require an explicit expected value when client certs are required
    # This prevents accepting arbitrary non-empty blobs in production.
    if not expected_value:
        return False

    return header_value.strip().lower() == expected_value


async def enforce_http_security(request: Request) -> None:
    """Run all HTTP-layer guards in sequence."""
    await _enforce_ip_allowlist(request)
    await enforce_request_body_limit(request)
    enforce_client_certificate(request)
    # No further checks here; reserved for future guards
    return None
