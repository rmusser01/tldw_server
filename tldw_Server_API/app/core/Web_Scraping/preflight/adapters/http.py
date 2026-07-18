"""Governed async HTTP probe adapter with explicit redirect handling."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

from loguru import logger

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Web_Scraping.preflight.context import (
    PreflightDeadlineExceeded,
    PreflightRuntimeControls,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import (
    ProbeError,
    ProbeHttpRequest,
    ProbeHttpResponse,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import (
    ProbeEgressDecision,
    ProbeEgressGuard,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import (
    RuntimeRequestContext,
)

try:
    from curl_cffi.requests import AsyncSession as _CurlAsyncSession
except ImportError:  # pragma: no cover - optional dependency
    _CurlAsyncSession = None


_DEFAULT_SESSION_FACTORY = object()
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_TIMEOUT_CLASS_NAMES = frozenset({"Timeout", "TimeoutError", "TimeoutException"})


class _HttpTransport(Protocol):
    async def send(self, request: ProbeHttpRequest) -> Any:
        raise NotImplementedError


def _subrequest_context(context: RuntimeRequestContext) -> RuntimeRequestContext:
    return replace(context, stage="preflight_subrequest")


def _denied_error(reason: str) -> ProbeError:
    code = "policy_error" if reason == "policy_error" else "policy_denied"
    return ProbeError(code, "Probe destination was denied.")


async def _fresh_decision(
    guard: ProbeEgressGuard,
    url: str,
    *,
    context: RuntimeRequestContext,
) -> ProbeEgressDecision:
    try:
        return await guard.decide(url, context=context)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - policy failures must fail closed
        raise ProbeError(
            "policy_error",
            "Probe destination was denied.",
        ) from None


def _mutable_proxies(
    proxies: Mapping[str, str] | str | None,
) -> dict[str, str] | str | None:
    if proxies is None:
        return None
    if isinstance(proxies, Mapping):
        return dict(proxies)
    return str(proxies)


async def _invoke_close(resource: Any) -> None:
    close = getattr(resource, "aclose", None)
    if not callable(close):
        close = getattr(resource, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _close_resource(resource: Any, *, label: str) -> None:
    """Finish one acquired-resource close while preserving caller cancellation."""
    cleanup_task = asyncio.create_task(
        _invoke_close(resource),
        name=f"preflight-http-close-{label}",
    )
    pending_cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            caller_was_cancelled = current_task is not None and current_task.cancelling() > 0
            if caller_was_cancelled and pending_cancellation is None:
                pending_cancellation = exc
        except Exception:  # noqa: BLE001 - inspected below without raw details
            break

    try:
        cleanup_task.result()
    except asyncio.CancelledError:
        if pending_cancellation is None:
            logger.warning(f"HTTP probe {label} cleanup failed.")
    except Exception:  # noqa: BLE001 - cleanup failure is deliberately secondary
        logger.warning(f"HTTP probe {label} cleanup failed.")
    if pending_cancellation is not None:
        raise pending_cancellation


async def _close_response(raw: Any) -> None:
    await _close_resource(raw, label="response")


async def _snapshot_response(
    raw: Any,
    *,
    fallback_url: str,
) -> ProbeHttpResponse:
    status = getattr(raw, "status_code", None)
    if status is None:
        status = getattr(raw, "status", 0)
    raw_headers = getattr(raw, "headers", None)
    headers = dict(raw_headers or {})
    text = getattr(raw, "text", "")
    raw_url = getattr(raw, "url", None)
    return ProbeHttpResponse(
        url=str(raw_url) if raw_url else fallback_url,
        status=int(status or 0),
        headers=headers,
        text=str(text or ""),
    )


def _redirect_location(response: ProbeHttpResponse) -> str | None:
    if response.status not in _REDIRECT_STATUS_CODES:
        return None
    for name, value in response.headers.items():
        if name.lower() == "location":
            if not value or not value.strip():
                raise ProbeError("invalid_redirect", "Redirect target is invalid.")
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ProbeError("invalid_redirect", "Redirect target is invalid.")
            return value.strip()
    return None


def _canonical_host(host: str) -> str | None:
    if not host or "%" in host:
        return None
    normalized = host.lower()
    try:
        return ipaddress.ip_address(normalized).compressed
    except ValueError:
        pass
    if normalized.endswith("."):
        normalized = normalized[:-1]
    if not normalized or normalized.endswith("."):
        return None
    try:
        normalized = normalized.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if len(normalized) > 253:
        return None
    labels = normalized.split(".")
    if not all(
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    ):
        return None
    return normalized


def _normalized_origin(url: str) -> tuple[str, str, int] | None:
    if not isinstance(url, str):
        return None
    if "\\" in url or any(ord(character) <= 32 or ord(character) == 127 for character in url):
        return None
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
            return None
        if "@" in parsed.netloc:
            return None
        host = _canonical_host(parsed.hostname or "")
        if host is None:
            return None
        port = parsed.port
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, int(port)


def _resolve_redirect(current_url: str, location: str) -> str:
    try:
        location_parts = urlsplit(location)
    except (TypeError, UnicodeError, ValueError):
        raise ProbeError("invalid_redirect", "Redirect target is invalid.") from None
    if location_parts.scheme and not location_parts.netloc:
        raise ProbeError("invalid_redirect", "Redirect target is invalid.")
    if location.startswith("//") and not location_parts.netloc:
        raise ProbeError("invalid_redirect", "Redirect target is invalid.")
    try:
        resolved = urljoin(current_url, location)
        parsed = urlsplit(resolved)
        resolved = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    except (TypeError, UnicodeError, ValueError):
        raise ProbeError("invalid_redirect", "Redirect target is invalid.") from None
    current_origin = _normalized_origin(current_url)
    target_origin = _normalized_origin(resolved)
    if target_origin is None:
        raise ProbeError("invalid_redirect", "Redirect target is invalid.")
    if current_origin is not None and current_origin[0] == "https" and target_origin[0] == "http":
        raise ProbeError("invalid_redirect", "Redirect target is invalid.")
    return resolved


def _credentials_for_hop(
    headers: Mapping[str, str],
    cookies: Mapping[str, str],
    *,
    original_url: str,
    target_url: str,
) -> tuple[dict[str, str], dict[str, str]]:
    current_headers = dict(headers)
    current_cookies = dict(cookies)
    original_origin = _normalized_origin(original_url)
    target_origin = _normalized_origin(target_url)
    if original_origin is not None and target_origin is not None and original_origin == target_origin:
        return current_headers, current_cookies
    return (
        {
            name: value
            for name, value in current_headers.items()
            if name.lower() not in http_client.SENSITIVE_REDIRECT_HEADERS
        },
        {},
    )


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    httpx_module = getattr(http_client, "httpx", None)
    httpx_timeout = getattr(httpx_module, "TimeoutException", None)
    if isinstance(httpx_timeout, type) and isinstance(exc, httpx_timeout):
        return True
    return type(exc).__name__ in _TIMEOUT_CLASS_NAMES


class HttpxProbeTransport:
    """Single-attempt transport over the central DNS-pinned async boundary."""

    async def send(self, request: ProbeHttpRequest) -> Any:
        return await http_client.afetch(
            method="GET",
            url=request.url,
            headers=dict(request.headers),
            cookies=dict(request.cookies),
            timeout=request.timeout_s,
            allow_redirects=False,
            proxies=_mutable_proxies(request.proxies),
            retry=http_client.RetryPolicy(attempts=1),
        )


class _CurlResponse:
    """Bind a curl response to the async session that owns it."""

    def __init__(self, response: Any, session: Any) -> None:
        self._response = response
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    async def aclose(self) -> None:
        response_error: BaseException | None = None
        try:
            await _invoke_close(self._response)
        except asyncio.CancelledError as exc:
            response_error = exc
        except Exception as exc:  # noqa: BLE001 - cleanup must continue to session
            response_error = exc
        try:
            await _invoke_close(self._session)
        except asyncio.CancelledError as exc:
            if response_error is None:
                response_error = exc
        except Exception as exc:  # noqa: BLE001 - cleanup failures stay secondary
            if response_error is None:
                response_error = exc
        if response_error is not None:
            raise response_error


class CurlCffiProbeTransport:
    """Native async impersonating transport with a second fresh egress check."""

    def __init__(
        self,
        *,
        egress_guard: ProbeEgressGuard,
        request_context: RuntimeRequestContext,
        session_factory: Any = _DEFAULT_SESSION_FACTORY,
    ) -> None:
        self._egress_guard = egress_guard
        self._request_context = _subrequest_context(request_context)
        self._session_factory = _CurlAsyncSession if session_factory is _DEFAULT_SESSION_FACTORY else session_factory

    async def send(self, request: ProbeHttpRequest) -> Any:
        if self._session_factory is None:
            raise ProbeUnavailable(error_code="missing_dependency")

        session = self._session_factory(impersonate=request.impersonate)
        try:
            decision = await _fresh_decision(
                self._egress_guard,
                request.url,
                context=self._request_context,
            )
            if not decision.allowed:
                raise _denied_error(decision.reason)
            response = await session.get(
                request.url,
                headers=dict(request.headers),
                cookies=dict(request.cookies),
                timeout=request.timeout_s,
                allow_redirects=False,
                proxies=_mutable_proxies(request.proxies),
            )
        except asyncio.CancelledError:
            await _close_resource(session, label="session")
            raise
        except Exception:
            await _close_resource(session, label="session")
            raise
        return _CurlResponse(response, session)


class GuardedHttpProbe:
    """Apply per-hop budgets, policy, deadlines, redirects, and cleanup."""

    def __init__(
        self,
        *,
        controls: PreflightRuntimeControls,
        egress_guard: ProbeEgressGuard,
        transport: _HttpTransport | None = None,
        curl_transport: _HttpTransport | None = None,
    ) -> None:
        self._controls = controls
        self._egress_guard = egress_guard
        self._transport = transport if transport is not None else HttpxProbeTransport()
        self._curl_transport = (
            curl_transport
            if curl_transport is not None
            else CurlCffiProbeTransport(
                egress_guard=egress_guard,
                request_context=controls.request_context,
            )
        )

    def _subrequest_context(self) -> RuntimeRequestContext:
        return _subrequest_context(self._controls.request_context)

    def _transport_for(self, request: ProbeHttpRequest) -> _HttpTransport:
        if request.impersonate is not None:
            return self._curl_transport
        return self._transport

    async def _dispatch(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
        try:
            raw = await self._transport_for(request).send(request)
            try:
                return await _snapshot_response(raw, fallback_url=request.url)
            finally:
                await _close_response(raw)
        except asyncio.CancelledError:
            raise
        except (ProbeError, PreflightDeadlineExceeded):
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize the transport boundary
            if _is_timeout_error(exc):
                if self._controls.deadline_exhausted():
                    raise PreflightDeadlineExceeded() from None
                raise ProbeTimeout() from None
            raise ProbeError("probe_error", "HTTP probe failed.") from None

    async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
        original_url = request.url
        current_url = request.url
        current_headers = dict(request.headers)
        current_cookies = dict(request.cookies)
        visited: set[str] = set()

        for hop in range(http_client.DEFAULT_MAX_REDIRECTS + 1):
            if current_url in visited:
                raise ProbeError("redirect_loop", "Redirect loop detected.")
            visited.add(current_url)
            await self._controls.reserve("request")
            decision = await _fresh_decision(
                self._egress_guard,
                current_url,
                context=self._subrequest_context(),
            )
            if not decision.allowed:
                raise _denied_error(decision.reason)
            timeout_s = self._controls.cap_timeout(request.timeout_s)
            response = await self._dispatch(
                replace(
                    request,
                    url=current_url,
                    headers=current_headers,
                    cookies=current_cookies,
                    timeout_s=timeout_s,
                    allow_redirects=False,
                )
            )
            if not request.allow_redirects:
                return response
            location = _redirect_location(response)
            if location is None:
                return response
            if hop == http_client.DEFAULT_MAX_REDIRECTS:
                raise ProbeError(
                    "too_many_redirects",
                    "Redirect limit exceeded.",
                )
            next_url = _resolve_redirect(current_url, location)
            current_headers, current_cookies = _credentials_for_hop(
                current_headers,
                current_cookies,
                original_url=original_url,
                target_url=next_url,
            )
            current_url = next_url
        raise ProbeError("too_many_redirects", "Redirect limit exceeded.")
