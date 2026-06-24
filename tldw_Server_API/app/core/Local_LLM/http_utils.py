"""HTTP utilities for Local_LLM handlers.

Provides a shared httpx AsyncClient factory, simple retry helpers,
readiness polling, and command redaction for safer logging.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger

from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.http_client import (
    RetryPolicy,
    adownload,
    afetch,
    afetch_json,
)
from tldw_Server_API.app.core.http_client import (
    create_async_client as _create_async_client,
)

DEFAULT_TIMEOUT: float = 120.0
DEFAULT_RETRIES: int = 2
DEFAULT_BACKOFF: float = 0.75
_SECRET_QUERY_KEYS = {
    "access_token",
    "apikey",
    "api_key",
    "auth",
    "authorization",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "token",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-signature",
}


def _is_httpx_async_client(client: Any) -> bool:
    module = getattr(client.__class__, "__module__", "")
    return module.startswith("httpx") and client.__class__.__name__ == "AsyncClient"


def get_http_status_from_exception(exc: Exception) -> int | None:
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            status = getattr(resp, attr, None)
            if status is not None:
                try:
                    return int(status)
                except (TypeError, ValueError):
                    pass
    for attr in ("status_code", "status"):
        status = getattr(exc, attr, None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                pass
    if isinstance(exc, NetworkError):
        match = re.search(r"HTTP\s+(\d{3})", str(exc))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def get_http_error_text(exc: Exception) -> str:
    resp = getattr(exc, "response", None)
    if resp is not None:
        text = getattr(resp, "text", None)
        if text is not None:
            return str(text)
    response_text = getattr(exc, "response_text", None)
    if response_text:
        return str(response_text)
    return str(exc)


def is_network_error(exc: Exception) -> bool:
    if isinstance(exc, (NetworkError, RetryExhaustedError)):
        return True
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    if module.startswith("httpx"):
        return "RequestError" in name or "Timeout" in name or "Connect" in name
    if module.startswith("requests"):
        return "RequestException" in name or "Timeout" in name or "ConnectionError" in name
    return False


def redacted_url(url: str) -> str:
    """Return a log-safe URL string without credentials or sensitive query values."""
    value = str(url or "").strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid-url>"
    if not parsed.scheme or not parsed.netloc:
        return value

    host = parsed.hostname or ""
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query = urlencode(
        [
            (key, val)
            for key, val in parse_qsl(parsed.query, keep_blank_values=True)
            if key.strip().lower() not in _SECRET_QUERY_KEYS
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def redact_cmd_args(
    args: Iterable[str],
    sensitive_flags: tuple[str, ...] = (
        "--api-key",
        "--hf-token",
        "--token",
        "--openai-api-key",
        "--anthropic-api-key",
        "--password",
        "--secret",
        "--auth",
        "--bearer",
        "--credential",
        "--credentials",
        "--access-token",
        "--refresh-token",
        "--client-secret",
    ),
) -> list[str]:
    """Redact values for flags like --api-key in a command list.

    Handles both space-separated and equals-separated formats:
    - ["--api-key", "secret"] -> ["--api-key", "REDACTED"]
    - ["--api-key=secret"] -> ["--api-key=REDACTED"]

    Example: ["llamafile", "--api-key", "secret", "-m", "model"] ->
             ["llamafile", "--api-key", "REDACTED", "-m", "model"]
    """
    redacted: list[str] = []
    itr = iter(args)
    for token in itr:
        # Check for --flag=value format
        if "=" in token:
            flag_part = token.split("=", 1)[0]
            if flag_part in sensitive_flags:
                redacted.append(f"{flag_part}=REDACTED")
                continue
        redacted.append(token)
        if token in sensitive_flags:
            # Replace the next token with REDACTED if present
            try:
                _ = next(itr)
                redacted.append("REDACTED")
            except StopIteration:
                # Nothing to redact
                pass
    return redacted


def create_async_client(timeout: float | None = None) -> Any:
    """Create a configured AsyncClient via central factory.

    Uses centralized defaults (trust_env=False, HTTP/2 if available).
    """
    return _create_async_client(timeout=timeout or DEFAULT_TIMEOUT)


async def request_json(
    client: Any,
    method: str,
    url: str,
    *,
    json: dict | None = None,
    headers: dict | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
):
    """Perform an HTTP request and return parsed JSON via central helpers.

    Uses centralized egress enforcement, retries, and backoff.
    """
    if not _is_httpx_async_client(client):
        raise TypeError("request_json requires an httpx.AsyncClient instance")

    # Map legacy semantics (attempts = 1 + retries) for real httpx clients
    retry_count = max(0, int(retries))
    attempts = max(1, retry_count + 1)
    # Convert legacy backoff seconds to ms base with a reasonable cap
    backoff_ms = max(50, int(backoff * 1000))
    policy = RetryPolicy(attempts=attempts, backoff_base_ms=backoff_ms)
    return await afetch_json(
        method=method,
        url=url,
        client=client,
        headers=headers,
        json=json,
        retry=policy,
    )


async def wait_for_http_ready(
    base_url: str,
    *,
    paths: tuple[str, ...] = ("/health", "/v1/models"),
    timeout_total: float = 30.0,
    interval: float = 0.5,
    accept_any_non_5xx: bool = False,
) -> bool:
    """Poll service until one of the candidate paths responds with a success status.

    Args:
        base_url: Base URL of the service to poll.
        paths: Tuple of paths to try (in order).
        timeout_total: Maximum time to wait for readiness.
        interval: Time between polling attempts.
        accept_any_non_5xx: If True, accept any non-5xx status (legacy behavior).
                           If False (default), only accept 2xx success codes.

    Returns True if ready within timeout; otherwise False.
    """
    async with create_async_client(timeout=5.0) as client:
        deadline = asyncio.get_event_loop().time() + timeout_total
        while asyncio.get_event_loop().time() < deadline:
            for path in paths:
                url = base_url.rstrip("/") + "/" + path.lstrip("/")
                try:
                    resp = await afetch(method="GET", url=url, client=client)
                    # Default: only accept 2xx success codes
                    # Legacy mode: accept any non-5xx (for backward compatibility)
                    if accept_any_non_5xx:
                        if resp.status_code < 500:
                            return True
                    else:
                        if 200 <= resp.status_code < 300:
                            return True
                except Exception as healthcheck_error:
                    # Not ready yet
                    logger.debug("Local LLM readiness probe failed; retrying", exc_info=healthcheck_error)
            await asyncio.sleep(interval)
    return False


async def wait_for_ollama_ready(base_url: str, timeout_total: float = 30.0, interval: float = 0.5) -> bool:
    """Poll Ollama until ready by checking common endpoints."""
    return await wait_for_http_ready(
        base_url,
        paths=("/api/version", "/api/tags"),
        timeout_total=timeout_total,
        interval=interval,
    )


async def async_stream_download(
    url: str,
    dest_path: str,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    chunk_size: int = 8192,
) -> None:
    """Download a file via centralized downloader with safety checks.

    Uses atomic rename and optional resume (disabled). On failure, partial
    files are removed by the downloader.
    """
    retry_count = max(0, int(retries))
    attempts = max(1, retry_count + 1)
    backoff_ms = max(50, int(backoff * 1000))
    policy = RetryPolicy(attempts=attempts, backoff_base_ms=backoff_ms)
    try:
        await adownload(url=url, dest=dest_path, retry=policy)
    except Exception as e:
        logger.error(f"Download failed for {redacted_url(url)}: {e}")
        raise
