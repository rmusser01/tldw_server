from __future__ import annotations

import ipaddress
import json
import os
import re
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


ACP_DEFAULT_RPC_TIMEOUT_SECONDS = _float_env("ACP_RPC_TIMEOUT_SECONDS", 60.0)
ACP_SESSION_UPDATE_QUEUE_MAXLEN = max(1, _int_env("ACP_SESSION_UPDATE_QUEUE_MAXLEN", 1000))
ACP_SESSION_UPDATE_MAX_BYTES = max(1024, _int_env("ACP_SESSION_UPDATE_MAX_BYTES", 262_144))
ACP_STREAM_BUFFER_MAX_BYTES = max(1024, _int_env("ACP_STREAM_BUFFER_MAX_BYTES", 1_048_576))

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_AUTH_HEADER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+")
_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,;]+"
)
_OPENAI_STYLE_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def make_session_update_queue() -> deque[dict[str, Any]]:
    return deque(maxlen=ACP_SESSION_UPDATE_QUEUE_MAXLEN)


def bounded_session_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        rendered = json.dumps(payload, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(payload)
    if len(rendered.encode("utf-8", errors="ignore")) <= ACP_SESSION_UPDATE_MAX_BYTES:
        return payload
    return {
        "sessionId": payload.get("sessionId"),
        "type": payload.get("type", "truncated"),
        "truncated": True,
        "message": "ACP session update exceeded queue payload limit",
    }


def redact_agent_output(value: str | bytes, *, max_chars: int = 1000) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)

    text = _PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", text)
    text = _AUTH_HEADER_RE.sub(r"\1[REDACTED]", text)
    text = _KEY_VALUE_SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _OPENAI_STYLE_TOKEN_RE.sub("[REDACTED]", text)

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "...[truncated]"
    keep = max(0, max_chars - len(marker))
    return f"{text[:keep]}{marker}"


def _normalized_origin(parsed) -> tuple[str, str, int | None]:
    port = parsed.port
    if port is None:
        if parsed.scheme == "http":
            port = 80
        elif parsed.scheme == "https":
            port = 443
    return parsed.scheme, str(parsed.hostname or "").lower(), port


def _is_blocked_host(hostname: str) -> bool:
    host = hostname.strip("[]").strip().lower().rstrip(".")
    if not host:
        return True
    if host in {"localhost", "ip6-localhost", "metadata.google.internal"}:
        return True
    if host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(
        (
            ip.is_loopback,
            ip.is_private,
            ip.is_link_local,
            ip.is_reserved,
            ip.is_multicast,
            ip.is_unspecified,
        )
    )


def validate_mcp_http_url(
    url: str,
    *,
    allow_private_network: bool | None = None,
    label: str = "MCP endpoint",
) -> str:
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{label} must use http or https")
    if not parsed.hostname:
        raise ValueError(f"{label} must include a hostname")
    allow_private = _truthy(os.getenv("ACP_ALLOW_PRIVATE_MCP_HTTP")) if allow_private_network is None else allow_private_network
    if not allow_private and _is_blocked_host(parsed.hostname):
        raise ValueError(f"{label} targets a local, private, or otherwise unsafe host")
    return str(url)


def validate_sse_post_url(
    sse_url: str,
    post_url: str,
    *,
    allow_private_network: bool | None = None,
    allow_cross_origin: bool | None = None,
) -> str:
    validate_mcp_http_url(sse_url, allow_private_network=allow_private_network, label="MCP SSE URL")
    validate_mcp_http_url(post_url, allow_private_network=allow_private_network, label="MCP SSE post URL")

    allow_cross = (
        _truthy(os.getenv("ACP_ALLOW_CROSS_ORIGIN_SSE_POST"))
        if allow_cross_origin is None
        else allow_cross_origin
    )
    if not allow_cross and _normalized_origin(urlparse(sse_url)) != _normalized_origin(urlparse(post_url)):
        raise ValueError("MCP SSE post URL must have the same origin as the SSE URL")
    return str(post_url)


def _resolve_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve(strict=False)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mcp_server_url_fields(server: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("url", "endpoint", "sse_url", "post_url"):
        value = server.get(key)
        if isinstance(value, str) and value.strip():
            urls.append(value.strip())
    return urls


def _is_stdio_mcp_server(server: dict[str, Any]) -> bool:
    transport = str(
        server.get("type")
        or server.get("transport")
        or server.get("mcp_transport")
        or ""
    ).strip().lower()
    if transport == "stdio":
        return True
    return bool(server.get("command")) and not _mcp_server_url_fields(server)


def validate_acp_session_launch_inputs(
    *,
    cwd: str,
    allowed_cwd_roots: list[str],
    runner_cwd: str | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    session_env: dict[str, str] | None = None,
    allow_session_env: bool = False,
    allow_inline_stdio_mcp_servers: bool = False,
    allow_private_mcp_http: bool = False,
) -> None:
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("ACP session cwd is required")
    cwd_path = Path(cwd).expanduser()
    if not cwd_path.is_absolute():
        raise ValueError("ACP session cwd must be absolute")

    roots = list(allowed_cwd_roots or [])
    if not roots and runner_cwd:
        roots = [runner_cwd]
    if not roots:
        roots = [os.getcwd()]

    resolved_cwd = _resolve_path(str(cwd_path))
    resolved_roots = [_resolve_path(str(root)) for root in roots if str(root or "").strip()]
    if not resolved_roots or not any(_is_within_root(resolved_cwd, root) for root in resolved_roots):
        raise ValueError("ACP session cwd is outside the allowed runner roots")

    if session_env and not allow_session_env:
        raise ValueError("ACP session env forwarding is disabled by default")

    for server in mcp_servers or []:
        if not isinstance(server, dict):
            raise ValueError("ACP mcpServers entries must be objects")
        if _is_stdio_mcp_server(server) and not allow_inline_stdio_mcp_servers:
            raise ValueError("ACP inline stdio MCP servers are disabled by default")
        for url in _mcp_server_url_fields(server):
            validate_mcp_http_url(
                url,
                allow_private_network=allow_private_mcp_http,
                label="ACP mcpServers URL",
            )
