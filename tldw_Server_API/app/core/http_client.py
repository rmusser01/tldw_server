from __future__ import annotations

"""
Centralized HTTP client factories and helpers with safe defaults.

Implements:
- Client factories (sync/async) with HTTP/2 by default and trust_env=False
- Egress policy enforcement for original URL, redirects, and proxies
- Retry policy with decorrelated jitter and Retry-After handling
- JSON helpers with content-type validation and max_bytes guard
- Streaming helpers: bytes and SSE, with retry handling for SSE disconnects
- Download helpers with atomic rename and optional checksum/length validation
- Structured logging and metrics hooks; optional trace header injection
"""

import asyncio  # noqa: E402
import contextvars  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import ssl  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping  # noqa: E402
from contextlib import asynccontextmanager, contextmanager, nullcontext, suppress  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from email.utils import parsedate_to_datetime  # noqa: E402
from functools import wraps  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Protocol, TypedDict, TypeVar  # noqa: E402
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit  # noqa: E402

from loguru import logger  # noqa: E402

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore

try:
    import aiohttp
except ImportError:  # pragma: no cover - optional dependency
    aiohttp = None  # type: ignore

try:
    from asyncio import timeout as _stream_timeout
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from async_timeout import timeout as _stream_timeout

try:  # Optional OpenTelemetry traceparent injection
    from opentelemetry import trace as _otel_trace  # type: ignore
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False
    _otel_trace = None  # type: ignore

try:  # Optional OpenTelemetry HTTP auto-instrumentation suppression
    from opentelemetry import context as _otel_context  # type: ignore
except ImportError:  # pragma: no cover
    _otel_context = None  # type: ignore

_OTEL_HTTP_SUPPRESSION_KEY = getattr(
    _otel_context,
    "_SUPPRESS_HTTP_INSTRUMENTATION_KEY",
    None,
)

_StreamItem = TypeVar("_StreamItem")

try:  # Optional public OpenTelemetry suppression helper
    from opentelemetry.instrumentation.utils import (  # type: ignore
        suppress_http_instrumentation as _suppress_http_instrumentation,
    )
except ImportError:  # pragma: no cover - optional dependency
    _suppress_http_instrumentation = nullcontext

from tldw_Server_API.app.core.exceptions import (  # noqa: E402
    DownloadError,
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
    StreamingProtocolError,
    raise_detached_error,
)
from tldw_Server_API.app.core.Metrics import (  # noqa: E402
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)
from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager  # noqa: E402
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope  # noqa: E402
from tldw_Server_API.app.core.stt_observability_context import (
    OPAQUE_STT_ENDPOINT_ID as _OPAQUE_STT_ENDPOINT_ID,
)
from tldw_Server_API.app.core.stt_observability_context import (
    get_opaque_stt_endpoint_id,
)
from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_explicit_pytest_runtime,
    is_test_mode,
    is_truthy,
)


class _TerminalHTTPStatusError(Exception):
    """Carry an already-classified terminal HTTP status past transport retry logic."""

    def __init__(self, status_code: int) -> None:
        self.status_code = int(status_code)
        super().__init__(f"HTTP {self.status_code}")


def _terminal_status_network_error(error: _TerminalHTTPStatusError) -> NetworkError:
    """Return a detached-safe error without propagating an invalid HTTP status."""
    status_code = error.status_code if 100 <= error.status_code <= 599 else None
    return NetworkError(str(error), status_code=status_code)


ResponseHeadersCallback = Callable[[int, Mapping[str, str]], Awaitable[None] | None]

_HTTPCLIENT_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    DownloadError,
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
    StreamingProtocolError,
    TimeoutError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
)
_HTTPX_REQUEST_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if httpx is not None:
    _HTTPX_REQUEST_EXCEPTIONS = tuple(
        cls
        for cls in (
            getattr(httpx, "HTTPError", None),
            getattr(httpx, "TransportError", None),
            getattr(httpx, "TimeoutException", None),
        )
        if isinstance(cls, type)
    )
_AIOHTTP_REQUEST_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if aiohttp is not None:
    _AIOHTTP_REQUEST_EXCEPTIONS = tuple(
        cls
        for cls in (getattr(aiohttp, "ClientError", None),)
        if isinstance(cls, type)
    )
_HTTPCLIENT_REQUEST_EXCEPTIONS = (
    _HTTPCLIENT_NONCRITICAL_EXCEPTIONS
    + _HTTPX_REQUEST_EXCEPTIONS
    + _AIOHTTP_REQUEST_EXCEPTIONS
)
_STREAM_CLEANUP_TRANSPORT_EXCEPTIONS = (
    NetworkError,
    OSError,
) + _HTTPX_REQUEST_EXCEPTIONS + _AIOHTTP_REQUEST_EXCEPTIONS

try:
    # Python 3.8+/backport safe import
    from importlib import metadata as _importlib_metadata  # type: ignore
except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    _importlib_metadata = None  # type: ignore


def _resolve_httpx():
    """Return the current `httpx` module, honoring test stubs in sys.modules.

    Falls back to the module imported at file import time if dynamic import fails.
    """
    try:
        import importlib
        return importlib.import_module("httpx")
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return httpx  # type: ignore


def _resolve_curl_session():
    """Return curl_cffi Session class if available, otherwise None."""
    try:
        import importlib
        mod = importlib.import_module("curl_cffi.requests")
        return getattr(mod, "Session", None)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None


# --------------------------------------------------------------------------------------
# Defaults & env config
# --------------------------------------------------------------------------------------

DEFAULT_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "5"))
DEFAULT_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "30"))
DEFAULT_WRITE_TIMEOUT = float(os.getenv("HTTP_WRITE_TIMEOUT", "30"))
DEFAULT_POOL_TIMEOUT = float(os.getenv("HTTP_POOL_TIMEOUT", "30"))
DEFAULT_ATTEMPTS = int(os.getenv("HTTP_RETRY_ATTEMPTS", "3"))
DEFAULT_BACKOFF_BASE_MS = int(os.getenv("HTTP_BACKOFF_BASE_MS", "250"))
DEFAULT_BACKOFF_CAP_S = int(os.getenv("HTTP_BACKOFF_CAP_S", "30"))
DEFAULT_MAX_REDIRECTS = int(os.getenv("HTTP_MAX_REDIRECTS", "5"))
DEFAULT_TRUST_ENV = is_truthy(os.getenv("HTTP_TRUST_ENV", "false"))
DEFAULT_USER_AGENT = os.getenv("HTTP_DEFAULT_USER_AGENT", "tldw_server httpx")
PROXY_ALLOWLIST = {h.strip().lower() for h in (os.getenv("PROXY_ALLOWLIST", "").split(",")) if h.strip()}
ENFORCE_TLS_MIN = (
    os.getenv("HTTP_ENFORCE_TLS_MIN")
    or os.getenv("TLS_ENFORCE_MIN_VERSION")
    or "false"
)
ENFORCE_TLS_MIN = is_truthy(str(ENFORCE_TLS_MIN))
TLS_MIN_VERSION = (os.getenv("HTTP_TLS_MIN_VERSION") or os.getenv("TLS_MIN_VERSION") or "1.2").strip()
_SENSITIVE_OBSERVABILITY_URL = "https://sensitive-endpoint.invalid"
_SENSITIVE_HTTP_LOG_CONTEXT = contextvars.ContextVar(
    "tldw_sensitive_http_log_context",
    default=False,
)


def _effective_sensitive_observability(explicit: bool = False) -> bool:
    """Honor an active sensitive scope even when a call site omits the flag."""
    return bool(explicit or _SENSITIVE_HTTP_LOG_CONTEXT.get())


class _SensitiveHTTPLogFilter(logging.Filter):
    """Suppress third-party URL logs only in a sensitive request context."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not _SENSITIVE_HTTP_LOG_CONTEXT.get():
            return True
        return not record.name.startswith(("httpx", "httpcore", "aiohttp"))


_SENSITIVE_HTTP_LOG_FILTER = _SensitiveHTTPLogFilter()


def _install_sensitive_http_log_filter() -> None:
    """Install the context-aware filter on active stdlib logging handlers."""
    loggers = [
        logging.getLogger(),
        logging.getLogger("httpx"),
        logging.getLogger("httpcore"),
        logging.getLogger("aiohttp"),
    ]
    for name, candidate in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(candidate, logging.Logger) and name.startswith(
            ("httpx", "httpcore", "aiohttp")
        ):
            loggers.append(candidate)
    for log in loggers:
        if _SENSITIVE_HTTP_LOG_FILTER not in log.filters:
            log.addFilter(_SENSITIVE_HTTP_LOG_FILTER)
        for handler in log.handlers:
            if _SENSITIVE_HTTP_LOG_FILTER not in handler.filters:
                handler.addFilter(_SENSITIVE_HTTP_LOG_FILTER)


@contextmanager
def _suppress_otel_http_auto_instrumentation() -> Iterator[None]:
    """Suppress standard OTel HTTP client spans in the current request context."""
    if _otel_context is None or _OTEL_HTTP_SUPPRESSION_KEY is None:
        yield
        return
    token = _otel_context.attach(
        _otel_context.set_value(_OTEL_HTTP_SUPPRESSION_KEY, True)
    )
    try:
        yield
    finally:
        _otel_context.detach(token)


@contextmanager
def sensitive_http_observability_context() -> Iterator[None]:
    """Suppress transport URL logs and HTTP auto-instrumentation in this context."""
    _install_sensitive_http_log_filter()
    token = _SENSITIVE_HTTP_LOG_CONTEXT.set(True)
    try:
        with _suppress_otel_http_auto_instrumentation():
            yield
    finally:
        _SENSITIVE_HTTP_LOG_CONTEXT.reset(token)


def _with_sensitive_http_log_context(
    function: Callable[..., httpx.Response],
) -> Callable[..., httpx.Response]:
    """Activate sensitive logging before any request validation runs."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> httpx.Response:
        if kwargs.get("sensitive_observability") is not True:
            return function(*args, **kwargs)
        with sensitive_http_observability_context():
            return function(*args, **kwargs)

    return wrapped


def _with_sensitive_http_log_context_async(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep sensitive third-party log filtering active across an awaited request."""
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("sensitive_observability") is not True:
            return await function(*args, **kwargs)
        with sensitive_http_observability_context():
            return await function(*args, **kwargs)

    return wrapped


@asynccontextmanager
async def _sensitive_http_stream_context(enabled: bool) -> AsyncIterator[None]:
    """Toggle request-local stream protection without rescanning log handlers."""
    if _SENSITIVE_HTTP_LOG_CONTEXT.get():
        yield
        return
    enabled = _effective_sensitive_observability(enabled)
    if not enabled:
        yield
        return
    token = _SENSITIVE_HTTP_LOG_CONTEXT.set(True)
    try:
        with _suppress_otel_http_auto_instrumentation():
            yield
    finally:
        _SENSITIVE_HTTP_LOG_CONTEXT.reset(token)


@contextmanager
def _http_client_observability_span(
    tracing_manager: Any,
    name: str,
    *,
    attributes: dict[str, Any],
    sensitive_observability: bool,
) -> Iterator[Any]:
    """Keep raw sensitive exceptions outside the application-owned span."""
    protected = (
        _effective_sensitive_observability(sensitive_observability)
        or _opaque_stt_endpoint_id() is not None
    )
    if not protected:
        with tracing_manager.span(name, attributes=attributes) as span:
            yield span
        return

    error: BaseException | None = None
    with tracing_manager.span(name, attributes=attributes) as span:
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001
            error = exc
    if error is not None:
        raise error.with_traceback(error.__traceback__)


@asynccontextmanager
async def _async_http_client_observability_span(
    tracing_manager: Any,
    name: str,
    *,
    attributes: dict[str, Any],
    sensitive_observability: bool,
) -> AsyncIterator[Any]:
    """Async counterpart that re-raises sensitive failures after span exit."""
    protected = (
        _effective_sensitive_observability(sensitive_observability)
        or _opaque_stt_endpoint_id() is not None
    )
    if not protected:
        async with tracing_manager.async_span(name, attributes=attributes) as span:
            yield span
        return

    error: BaseException | None = None
    async with tracing_manager.async_span(name, attributes=attributes) as span:
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001
            error = exc
    if error is not None:
        raise error.with_traceback(error.__traceback__)


async def _iterate_sensitive_http_items(
    stream: AsyncIterator[_StreamItem],
    *,
    enabled: bool,
) -> AsyncIterator[_StreamItem]:
    """Advance a sensitive stream inside request-local scopes, never across yield."""
    iterator = stream.__aiter__()
    enabled = _effective_sensitive_observability(enabled)
    if enabled:
        _install_sensitive_http_log_filter()
    try:
        while True:
            try:
                async with _sensitive_http_stream_context(enabled):
                    chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            async with _sensitive_http_stream_context(enabled):
                await close()

_OPAQUE_STT_ENDPOINT_ID_RE = re.compile(
    r"^sha256:[0-9a-f]{64}$"
)
_OPAQUE_STT_LOG_FACTORY_LOCK = threading.Lock()


def _install_opaque_stt_log_record_factory() -> None:
    """Chain a factory that redacts HTTP client logs in STT context."""
    current = logging.getLogRecordFactory()
    if (
        getattr(current, "_tldw_opaque_stt_context", None)
        is _OPAQUE_STT_ENDPOINT_ID
    ):
        return
    with _OPAQUE_STT_LOG_FACTORY_LOCK:
        current = logging.getLogRecordFactory()
        if (
            getattr(current, "_tldw_opaque_stt_context", None)
            is _OPAQUE_STT_ENDPOINT_ID
        ):
            return
        previous_factory = current

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = previous_factory(*args, **kwargs)
            endpoint_id = _OPAQUE_STT_ENDPOINT_ID.get()
            if endpoint_id is None or not (
                record.name in {"httpx", "httpcore", "aiohttp"}
                or record.name.startswith(
                    ("httpx.", "httpcore.", "aiohttp.")
                )
            ):
                return record
            status_code = next(
                (
                    value
                    for value in (
                        record.args
                        if isinstance(record.args, tuple)
                        else ()
                    )
                    if isinstance(value, int)
                    and 100 <= value <= 599
                ),
                None,
            )
            if status_code is None:
                record.msg = (
                    "planned STT HTTP endpoint_id=%s"
                )
                record.args = (endpoint_id,)
            else:
                record.msg = (
                    "planned STT HTTP endpoint_id=%s status_code=%s"
                )
                record.args = (endpoint_id, status_code)
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            return record

        record_factory._tldw_opaque_stt_context = (  # type: ignore[attr-defined]
            _OPAQUE_STT_ENDPOINT_ID
        )
        logging.setLogRecordFactory(record_factory)


@contextmanager
def opaque_stt_http_observability(
    endpoint_id: str,
) -> Iterator[None]:
    """Expose only an opaque endpoint identifier for a planned STT call."""
    if (
        not isinstance(endpoint_id, str)
        or _OPAQUE_STT_ENDPOINT_ID_RE.fullmatch(endpoint_id) is None
    ):
        raise ValueError("Invalid opaque STT endpoint identifier")
    _install_opaque_stt_log_record_factory()
    token = _OPAQUE_STT_ENDPOINT_ID.set(endpoint_id)
    try:
        with _suppress_http_instrumentation():
            yield
    finally:
        _OPAQUE_STT_ENDPOINT_ID.reset(token)


def _opaque_stt_endpoint_id() -> str | None:
    """Return the opaque endpoint identifier for the active planned STT call."""
    return get_opaque_stt_endpoint_id()


def _http_trace_attributes(
    method: str,
    url: str,
) -> dict[str, Any]:
    """Build privacy-aware trace attributes for an HTTP request."""
    endpoint_id = _opaque_stt_endpoint_id()
    if endpoint_id is not None:
        return {"stt.endpoint_id": endpoint_id}
    if _effective_sensitive_observability():
        url = _SENSITIVE_OBSERVABILITY_URL
    return {
        "http.method": method.upper(),
        "net.host.name": _parse_host_from_url(url),
        "url.full": _sanitize_url_for_logs(url),
    }


def _http_metric_labels(
    method: str,
    url: str,
    *,
    status_code: int | None = None,
) -> dict[str, str]:
    """Build privacy-aware metric labels for an HTTP request."""
    endpoint_id = _opaque_stt_endpoint_id()
    if endpoint_id is not None:
        labels = {"endpoint_id": endpoint_id}
    else:
        if _effective_sensitive_observability():
            url = _SENSITIVE_OBSERVABILITY_URL
        labels = {
            "method": method.upper(),
            "host": _parse_host_from_url(url),
        }
    if status_code is not None:
        labels["status"] = str(status_code)
    return labels


def _httpx_timeout_from_defaults() -> httpx.Timeout:
    return httpx.Timeout(
        connect=DEFAULT_CONNECT_TIMEOUT,
        read=DEFAULT_READ_TIMEOUT,
        write=DEFAULT_WRITE_TIMEOUT,
        pool=DEFAULT_POOL_TIMEOUT,
    )


def build_limits(
    *,
    max_connections: int | None = None,
    max_keepalive_connections: int | None = None,
    keepalive_expiry: float | None = None,
) -> httpx.Limits | None:
    """Create an httpx.Limits instance when httpx is available."""
    _hx = _resolve_httpx()
    if _hx is None or not hasattr(_hx, "Limits"):
        return None
    kwargs: dict[str, Any] = {}
    if max_connections is not None:
        kwargs["max_connections"] = max_connections
    if max_keepalive_connections is not None:
        kwargs["max_keepalive_connections"] = max_keepalive_connections
    if keepalive_expiry is not None:
        kwargs["keepalive_expiry"] = keepalive_expiry
    if not kwargs:
        return None
    try:
        return _hx.Limits(**kwargs)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None


_CACHED_VERSION: str | None = None


def _get_project_version() -> str:
    global _CACHED_VERSION
    if _CACHED_VERSION:
        return _CACHED_VERSION
    # 1) Env override
    v = os.getenv("TLDW_VERSION")
    if v:
        _CACHED_VERSION = v.strip()
        return _CACHED_VERSION
    # 2) Try package metadata if installed
    try:
        if _importlib_metadata is not None:
            _CACHED_VERSION = _importlib_metadata.version("tldw-server")  # type: ignore[attr-defined]
            if _CACHED_VERSION:
                return _CACHED_VERSION
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    # 3) Fallback: parse pyproject.toml in repo root
    try:
        root = Path(__file__).resolve().parents[3]
        pp = root / "pyproject.toml"
        if pp.exists():
            text = pp.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
            if m:
                _CACHED_VERSION = m.group(1).strip()
                return _CACHED_VERSION
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    _CACHED_VERSION = "0.0.0"
    return _CACHED_VERSION


_BOUNDED_RESPONSE_EXTENSION = "tldw_bounded_response"


def _is_bounded_httpx_response(response: httpx.Response) -> bool:
    request = getattr(response, "request", None)
    extensions = getattr(request, "extensions", {})
    return bool(
        extensions.get(_BOUNDED_RESPONSE_EXTENSION)
        or extensions.get("tldw_stream_response")
    )


def _capture_error_body_hook(response: httpx.Response) -> None:
    try:
        if response.status_code >= 400 and not _is_bounded_httpx_response(response):
            response.read()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass


async def _capture_error_body_hook_async(response: httpx.Response) -> None:
    try:
        if response.status_code >= 400 and not _is_bounded_httpx_response(response):
            try:
                await response.aread()
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    response.read()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass


_HTTP_CLIENT_METRICS_REGISTERED = False


def _register_http_client_metrics_once() -> None:
    global _HTTP_CLIENT_METRICS_REGISTERED
    if _HTTP_CLIENT_METRICS_REGISTERED:
        return
    try:
        reg = get_metrics_registry()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return
    # Register http-client-specific metrics if not present
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        reg.register_metric(
            MetricDefinition(
                name="http_client_requests_total",
                type=MetricType.COUNTER,
                description="Total number of outbound HTTP client requests",
                labels=["method", "host", "status"],
            )
        )
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        reg.register_metric(
            MetricDefinition(
                name="http_client_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Outbound HTTP client request duration (seconds)",
                unit="s",
                labels=["method", "host"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
            )
        )
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        reg.register_metric(
            MetricDefinition(
                name="http_client_retries_total",
                type=MetricType.COUNTER,
                description="Total retries attempted by HTTP client",
                labels=["reason"],
            )
        )
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        reg.register_metric(
            MetricDefinition(
                name="http_client_egress_denials_total",
                type=MetricType.COUNTER,
                description="Total egress policy denials for outbound HTTP",
                labels=["reason"],
            )
        )
    _HTTP_CLIENT_METRICS_REGISTERED = True

# Ensure metrics are registered on import
with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
    _register_http_client_metrics_once()


# --------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------

class HttpResponse(TypedDict):
    status: int
    headers: dict[str, str]
    text: str
    url: str
    backend: str  # 'curl' or 'httpx'


@dataclass
class RetryPolicy:
    attempts: int = DEFAULT_ATTEMPTS
    backoff_base_ms: int = DEFAULT_BACKOFF_BASE_MS
    backoff_cap_s: int = DEFAULT_BACKOFF_CAP_S
    retry_on_status: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    retry_on_methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    respect_retry_after: bool = True
    retry_on_unsafe: bool = False

    @property
    def enabled(self) -> bool:
        return self.attempts > 1


@dataclass
class SSEEvent:
    event: str = "message"
    data: str = ""
    id: str | None = None
    retry: int | None = None


class _AiohttpResponse:
    def __init__(self, response: Any, body: bytes) -> None:
        self._response = response
        self.status_code = int(getattr(response, "status", 0))
        self.headers = getattr(response, "headers", {}) or {}
        self.url = str(getattr(response, "url", ""))
        self.request = SimpleNamespace(url=self.url)
        self._body = body or b""
        try:
            encoding = getattr(response, "charset", None) or "utf-8"
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            encoding = "utf-8"
        try:
            self._text = self._body.decode(encoding, errors="replace")
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            self._text = self._body.decode("utf-8", errors="replace")

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._body

    def json(self) -> Any:
        return json.loads(self._text)

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        if httpx is not None:
            try:
                req = httpx.Request("GET", self.url)
                raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=req, response=self)  # type: ignore[arg-type]  # noqa: TRY003
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                raise NetworkError(f"HTTP {self.status_code}")  # noqa: B904, TRY003
        raise NetworkError(f"HTTP {self.status_code}")  # noqa: TRY003

    async def aclose(self) -> None:
        try:
            if self._response is not None:
                self._response.release()
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            pass

    async def aiter_bytes(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        if not self._body:
            return
        for idx in range(0, len(self._body), chunk_size):
            yield self._body[idx : idx + chunk_size]


# --------------------------------------------------------------------------------------
# Adapter scaffolding (Stage 2)
# --------------------------------------------------------------------------------------

class SyncResponseLike(Protocol):
    status_code: int
    headers: dict[str, Any]
    url: str
    text: str

    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...
    def close(self) -> None: ...


class AsyncResponseLike(Protocol):
    status_code: int
    headers: dict[str, Any]
    url: str
    text: str

    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...
    async def aclose(self) -> None: ...


class TransportAdapter(Protocol):
    name: str

    def request(
        self,
        *,
        method: str,
        url: str,
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: Any | None = None,
        allow_redirects: bool = True,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        configured_endpoint: ConfiguredEndpointScope | None = None,
        sensitive_observability: bool = False,
    ) -> SyncResponseLike: ...

    async def arequest(
        self,
        *,
        method: str,
        url: str,
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: Any | None = None,
        allow_redirects: bool = True,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        verify: bool | str | ssl.SSLContext | None = None,
        configured_endpoint: ConfiguredEndpointScope | None = None,
        sensitive_observability: bool = False,
        max_response_bytes: int | None = None,
    ) -> AsyncResponseLike: ...

    async def stream_bytes(
        self,
        *,
        method: str,
        url: str,
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: Any | None = None,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        chunk_size: int = 65536,
        cert_pinning: dict[str, set[str]] | None = None,
        sensitive_observability: bool = False,
        on_response: ResponseHeadersCallback | None = None,
        raise_for_status: bool = True,
        verify: bool | str | ssl.SSLContext | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def stream_sse(
        self,
        *,
        url: str,
        method: str = "GET",
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        timeout: Any | None = None,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        sensitive_observability: bool = False,
    ) -> AsyncIterator[SSEEvent]: ...


class HttpxAdapter:
    name = "httpx"

    @_with_sensitive_http_log_context
    def request(
        self,
        *,
        method: str,
        url: str,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: float | httpx.Timeout | None = None,
        allow_redirects: bool = True,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        configured_endpoint: ConfiguredEndpointScope | None = None,
        sensitive_observability: bool = False,
    ) -> httpx.Response:
        return _fetch_httpx_response(
            method=method,
            url=url,
            client=client,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            allow_redirects=allow_redirects,
            proxies=proxies,
            retry=retry,
            cert_pinning=cert_pinning,
            configured_endpoint=configured_endpoint,
            sensitive_observability=sensitive_observability,
        )

    @_with_sensitive_http_log_context_async
    async def arequest(
        self,
        *,
        method: str,
        url: str,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: float | httpx.Timeout | None = None,
        allow_redirects: bool = True,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        verify: bool | str | ssl.SSLContext | None = None,
        configured_endpoint: ConfiguredEndpointScope | None = None,
        sensitive_observability: bool = False,
        max_response_bytes: int | None = None,
    ) -> httpx.Response:
        return await _afetch_httpx(
            method=method,
            url=url,
            client=client,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            allow_redirects=allow_redirects,
            proxies=proxies,
            retry=retry,
            cert_pinning=cert_pinning,
            verify=verify,
            configured_endpoint=configured_endpoint,
            sensitive_observability=sensitive_observability,
            max_response_bytes=max_response_bytes,
        )

    async def stream_bytes(
        self,
        *,
        method: str,
        url: str,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: float | httpx.Timeout | None = None,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        chunk_size: int = 65536,
        cert_pinning: dict[str, set[str]] | None = None,
        sensitive_observability: bool = False,
        on_response: ResponseHeadersCallback | None = None,
        raise_for_status: bool = True,
        verify: bool | str | ssl.SSLContext | None = None,
    ) -> AsyncIterator[bytes]:
        stream = _astream_bytes_httpx(
            method=method,
            url=url,
            client=client,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            proxies=proxies,
            retry=retry,
            chunk_size=chunk_size,
            cert_pinning=cert_pinning,
            sensitive_observability=sensitive_observability,
            on_response=on_response,
            raise_for_status=raise_for_status,
            verify=verify,
        )
        scoped_stream = _iterate_sensitive_http_items(
            stream,
            enabled=sensitive_observability,
        )
        try:
            async for chunk in scoped_stream:
                yield chunk
        finally:
            await scoped_stream.aclose()

    async def stream_sse(
        self,
        *,
        url: str,
        method: str = "GET",
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        timeout: float | httpx.Timeout | None = None,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        sensitive_observability: bool = False,
    ) -> AsyncIterator[SSEEvent]:
        stream = _astream_sse_httpx(
            url=url,
            method=method,
            client=client,
            headers=headers,
            params=params,
            json=json,
            data=data,
            timeout=timeout,
            proxies=proxies,
            retry=retry,
            cert_pinning=cert_pinning,
            sensitive_observability=sensitive_observability,
        )
        scoped_stream = _iterate_sensitive_http_items(
            stream,
            enabled=sensitive_observability,
        )
        try:
            async for event in scoped_stream:
                yield event
        finally:
            await scoped_stream.aclose()


class AiohttpAdapter:
    name = "aiohttp"

    def request(self, **_: Any) -> SyncResponseLike:
        raise NotImplementedError("AiohttpAdapter does not support sync requests")

    @_with_sensitive_http_log_context_async
    async def arequest(
        self,
        *,
        method: str,
        url: str,
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: Any | None = None,
        allow_redirects: bool = True,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        verify: bool | str | ssl.SSLContext | None = None,
        configured_endpoint: ConfiguredEndpointScope | None = None,
        sensitive_observability: bool = False,
        max_response_bytes: int | None = None,
    ) -> AsyncResponseLike:
        return await _afetch_aiohttp(
            method=method,
            url=url,
            client=client,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            allow_redirects=allow_redirects,
            proxies=proxies,
            retry=retry,
            cert_pinning=cert_pinning,
            verify=verify,
            configured_endpoint=configured_endpoint,
            sensitive_observability=sensitive_observability,
            max_response_bytes=max_response_bytes,
        )

    async def stream_bytes(
        self,
        *,
        method: str,
        url: str,
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        timeout: Any | None = None,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        chunk_size: int = 65536,
        cert_pinning: dict[str, set[str]] | None = None,
        sensitive_observability: bool = False,
        on_response: ResponseHeadersCallback | None = None,
        raise_for_status: bool = True,
        verify: bool | str | ssl.SSLContext | None = None,
    ) -> AsyncIterator[bytes]:
        stream = _astream_bytes_aiohttp(
            method=method,
            url=url,
            client=client,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            proxies=proxies,
            retry=retry,
            chunk_size=chunk_size,
            cert_pinning=cert_pinning,
            sensitive_observability=sensitive_observability,
            on_response=on_response,
            raise_for_status=raise_for_status,
            verify=verify,
        )
        scoped_stream = _iterate_sensitive_http_items(
            stream,
            enabled=sensitive_observability,
        )
        try:
            async for chunk in scoped_stream:
                yield chunk
        finally:
            await scoped_stream.aclose()

    async def stream_sse(
        self,
        *,
        url: str,
        method: str = "GET",
        client: Any | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        timeout: Any | None = None,
        proxies: str | dict[str, str] | None = None,
        retry: RetryPolicy | None = None,
        cert_pinning: dict[str, set[str]] | None = None,
        sensitive_observability: bool = False,
    ) -> AsyncIterator[SSEEvent]:
        stream = _astream_sse_aiohttp(
            url=url,
            method=method,
            client=client,
            headers=headers,
            params=params,
            json=json,
            data=data,
            timeout=timeout,
            proxies=proxies,
            retry=retry,
            cert_pinning=cert_pinning,
            sensitive_observability=sensitive_observability,
        )
        scoped_stream = _iterate_sensitive_http_items(
            stream,
            enabled=sensitive_observability,
        )
        try:
            async for event in scoped_stream:
                yield event
        finally:
            await scoped_stream.aclose()


_HTTPX_ADAPTER = HttpxAdapter()
_AIOHTTP_ADAPTER = AiohttpAdapter()


def _get_transport_adapter(name: str) -> TransportAdapter:
    adapter_key = str(name).lower().strip()
    if adapter_key == "aiohttp":
        return _AIOHTTP_ADAPTER
    return _HTTPX_ADAPTER


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
}
_TELEGRAM_BOT_PATH_PATTERN = re.compile(r"^/bot[^/]+(?P<suffix>/.*)?$")


def _redact_headers(h: dict[str, str] | None) -> dict[str, str]:
    safe: dict[str, str] = {}
    if not h:
        return safe
    for k, v in h.items():
        if k.lower() in _SENSITIVE_HEADER_KEYS:
            safe[k] = "<redacted>"
        else:
            safe[k] = v
    return safe


def _sanitize_accept_encoding_for_backend(headers: dict[str, str] | None, backend: str) -> dict[str, str]:
    """Return a copy of headers with backend-specific Accept-Encoding tweaks.

    - Case-insensitively reads/removes existing Accept-Encoding headers.
    - For 'httpx' and 'aiohttp' backends, removes any 'zstd' codings (with or without parameters, e.g. 'zstd;q=0.9').
    - For 'requests' and 'urllib3' backends, removes 'zstd' and 'br' codings to avoid unsupported decoders.
    - Writes back a single canonical 'Accept-Encoding' header if tokens remain; otherwise removes it.
    - Best-effort: on any parsing error, leaves headers unchanged.
    """
    hdrs: dict[str, str] = dict(headers or {})
    backend_norm = str(backend).lower()
    if backend_norm not in {"httpx", "aiohttp", "requests", "urllib3"}:
        return hdrs
    try:
        # Find all Accept-Encoding header keys regardless of case
        ae_keys = [k for k in hdrs if k.lower() == "accept-encoding"]
        if not ae_keys:
            return hdrs

        # Combine values from all variants
        raw_vals: list[str] = []
        for k in ae_keys:
            v = hdrs.get(k)
            if v is None:
                continue
            raw_vals.append(str(v))
        combined = ",".join(raw_vals)

        drop_cod = {"zstd"}
        if backend_norm in {"requests", "urllib3"}:
            drop_cod = {"zstd", "br"}

        # Parse tokens, dropping any disallowed codings regardless of parameters
        filtered: list[str] = []
        for part in combined.split(','):
            token = part.strip()
            if not token:
                continue
            coding = token.split(';', 1)[0].strip().lower()
            if coding in drop_cod:
                continue
            filtered.append(token)

        # Commit: remove all original variants
        for k in ae_keys:
            hdrs.pop(k, None)
        # Write canonical header back if any tokens remain
        if filtered:
            hdrs["Accept-Encoding"] = ", ".join(filtered)
        # else: if nothing remains, header stays removed
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        # Best-effort: return original headers unchanged
        return dict(headers or {})
    return hdrs


def _force_identity_accept_encoding(
    headers: dict[str, str] | None,
) -> dict[str, str]:
    """Return copied headers that request only an identity-encoded response."""
    identity_headers = dict(headers or {})
    for key in tuple(identity_headers):
        if key.lower() == "accept-encoding":
            identity_headers.pop(key)
    identity_headers["Accept-Encoding"] = "identity"
    return identity_headers


def _uses_compressed_content_encoding(headers: Any) -> bool:
    """Return whether headers declare any non-identity content encoding."""
    try:
        values = [
            str(value)
            for key, value in headers.items()
            if str(key).lower() == "content-encoding"
        ]
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return False
    return any(
        token.strip().lower() not in {"", "identity"}
        for value in values
        for token in value.split(",")
    )


def _read_bounded_chunks(chunks: Iterable[bytes], max_response_bytes: int) -> bytes:
    """Read an iterator without retaining more than the configured byte limit."""
    body = bytearray()
    for chunk in chunks:
        if len(body) + len(chunk) > max_response_bytes:
            raise ValueError("Response exceeds max_response_bytes limit")
        body.extend(chunk)
    return bytes(body)


class _BoundedBodyCollector:
    """Collect callback-delivered bytes without retaining data past the limit."""

    def __init__(self, max_response_bytes: int) -> None:
        self._max_response_bytes = max_response_bytes
        self._body = bytearray()
        self.overflow = False

    @property
    def body(self) -> bytes:
        return bytes(self._body)

    def __call__(self, chunk: bytes) -> int:
        remaining = self._max_response_bytes - len(self._body)
        if len(chunk) > remaining:
            if remaining > 0:
                self._body.extend(chunk[:remaining])
            self.overflow = True
        else:
            self._body.extend(chunk)
        return len(chunk)


def _decode_bounded_response_body(response: Any, body: bytes) -> str:
    """Decode raw bounded response bytes using the response encoding when known."""
    encoding = str(getattr(response, "encoding", "") or "utf-8")
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _close_simple_response(response: Any) -> None:
    """Best-effort close for simple fetch streaming responses."""
    close = getattr(response, "close", None)
    if callable(close):
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            close()


def _redact_path_for_logging(host: str, path: str) -> str:
    if host == "api.telegram.org":
        match = _TELEGRAM_BOT_PATH_PATTERN.match(path or "/")
        if match:
            suffix = match.group("suffix") or ""
            return f"/bot<redacted>{suffix}"
    return path or "/"


def _sanitize_url_for_logs(u: str | Any) -> str:
    endpoint_id = _opaque_stt_endpoint_id()
    if endpoint_id is not None:
        return endpoint_id
    try:
        s = str(u)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return ""
    try:
        parsed = urlparse(s)
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()
        if not scheme or not host:
            return "" if _effective_sensitive_observability() else s
        if _effective_sensitive_observability():
            try:
                host = host.encode("idna").decode("ascii")
                port = parsed.port
            except (UnicodeError, ValueError):
                return ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            default_port = 443 if scheme == "https" else 80 if scheme == "http" else None
            authority = host if port is None or port == default_port else f"{host}:{port}"
            return f"{scheme}://{authority}"
        port = parsed.port
        path = _redact_path_for_logging(host, parsed.path or "/")
        authority = host
        if port is not None:
            authority = f"{host}:{port}"
        return f"{scheme}://{authority}{path}"
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return "" if _effective_sensitive_observability() else s


def _raise_if_json_response_exceeds_limit(
    *,
    headers: Any,
    content: bytes,
    max_bytes: int | None,
) -> None:
    """Raise when JSON response metadata or actual body exceeds the configured limit."""
    if max_bytes is None:
        return

    content_length = headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as exc:
            raise JSONDecodeError("Invalid content-length header") from exc
        if declared_length > max_bytes:
            raise JSONDecodeError("Response exceeds max_bytes limit")  # noqa: TRY003

    if len(content) > max_bytes:
        raise JSONDecodeError("Response exceeds max_bytes limit")  # noqa: TRY003


def _url_parts(u: str | Any) -> tuple[str, str, str]:
    """Return (scheme, host, path) for logging; redacts query by omission."""
    endpoint_id = _opaque_stt_endpoint_id()
    if endpoint_id is not None:
        return "", endpoint_id, ""
    try:
        s = str(u)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        s = ""
    try:
        p = urlparse(s)
        scheme = (p.scheme or "").lower()
        host = (p.hostname or "").lower()
        if _effective_sensitive_observability():
            try:
                host = host.encode("idna").decode("ascii")
            except UnicodeError:
                return "", "", ""
            return scheme, host, ""
        path = _redact_path_for_logging(host, p.path or "/")
        return scheme, host, path  # noqa: TRY300
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return "", "", ""


def _log_outbound_request(
    *,
    method: str,
    url: str | Any,
    status_code: int,
    start_time: float,
    attempt: int,
    last_retry_delay_s: float = 0.0,
    exception_class: str = "",
) -> None:
    """Emit a single structured log line for an outbound HTTP call.

    Fields: request_id (from global log context), method, scheme, host, path,
    status_code, duration_ms, attempt, retry_delay_ms, exception_class.
    """
    if _effective_sensitive_observability():
        url = _SENSITIVE_OBSERVABILITY_URL
    try:
        duration_ms = int(max(0.0, time.time() - start_time) * 1000)
        endpoint_id = _opaque_stt_endpoint_id()
        if endpoint_id is not None:
            lvl = (
                "warning"
                if status_code >= 400 or exception_class
                else "info"
            )
            logger.bind(
                endpoint_id=endpoint_id,
                status_code=int(status_code),
                duration_ms=duration_ms,
            ).log(lvl, "planned STT HTTP outbound")
            return
        retry_delay_ms = int(max(0.0, last_retry_delay_s) * 1000)
        scheme, host, path = _url_parts(url)
        lvl = "warning" if (status_code >= 400 or exception_class) else "info"
        logger.bind(
            method=method.upper(),
            scheme=scheme,
            host=host,
            path=path,
            status_code=int(status_code),
            duration_ms=duration_ms,
            attempt=int(attempt),
            retry_delay_ms=retry_delay_ms,
            exception_class=exception_class,
        ).log(lvl, "http.client outbound")
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        # Never raise on logging failures
        pass

def _parse_host_from_url(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return ""


def _normalize_dns_pin_host(host: str) -> str:
    from tldw_Server_API.app.core.Security.egress import (
        _canonical_ip_literal,
        _normalize_hostname,
    )

    out = _normalize_hostname(host or "")
    return _canonical_ip_literal(out) or out


def _accepted_ips_for_url(
    dns_pin_cache: dict[str, tuple[str, ...]],
    url: str,
) -> tuple[str, ...]:
    return dns_pin_cache.get(_normalize_dns_pin_host(_parse_host_from_url(url)), ())


def _prepare_pinned_transport_target(
    url: str,
    headers: dict[str, str] | None,
    accepted_resolved_ips: tuple[str, ...],
) -> tuple[str, dict[str, str] | None, str | None]:
    """Bind transport I/O to a vetted IP while preserving HTTP and TLS identity."""
    if not accepted_resolved_ips:
        return url, headers, None

    parsed = urlsplit(url)
    original_host = parsed.hostname or ""
    if not original_host:
        raise EgressPolicyError(
            "Validated URL has no host",
            reason_code="invalid_url",
        )

    accepted_ip = accepted_resolved_ips[0]
    try:
        import ipaddress

        ip_value = ipaddress.ip_address(accepted_ip)
    except ValueError as exc:
        raise EgressPolicyError(
            "Validated URL resolved to an invalid address",
            reason_code="address_invalid",
        ) from exc

    try:
        original_ip = ipaddress.ip_address(original_host)
    except ValueError:
        try:
            identity_host = original_host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise EgressPolicyError(
                "Validated URL has an invalid host",
                reason_code="invalid_url",
            ) from exc
    else:
        identity_host = original_ip.compressed

    transport_host = f"[{ip_value.compressed}]" if ip_value.version == 6 else ip_value.compressed
    if parsed.port is not None:
        transport_host = f"{transport_host}:{parsed.port}"
    if "@" in parsed.netloc:
        transport_host = f"{parsed.netloc.rsplit('@', 1)[0]}@{transport_host}"
    transport_url = urlunsplit(
        (parsed.scheme, transport_host, parsed.path, parsed.query, parsed.fragment)
    )

    original_authority = f"[{identity_host}]" if ":" in identity_host else identity_host
    if parsed.port is not None:
        original_authority = f"{original_authority}:{parsed.port}"
    prepared_headers = {
        key: value
        for key, value in (headers or {}).items()
        if key.lower() != "host"
    }
    prepared_headers["Host"] = original_authority
    return transport_url, prepared_headers, identity_host


def _restore_httpx_response_url(response: Any, original_url: str) -> None:
    """Keep the caller-visible request URL and redirect base on the logical origin."""
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        response.request.url = httpx.URL(original_url)


def _pins_for_host(
    pins_map: dict[str, set[str]] | None,
    host: str,
) -> set[str] | None:
    if not pins_map:
        return None
    normalized_host = _normalize_dns_pin_host(host)
    for pin_host, pins in pins_map.items():
        if _normalize_dns_pin_host(pin_host) == normalized_host:
            return pins
    return None


def _inject_trace_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out = dict(headers or {})
    if _OTEL_AVAILABLE:
        try:  # best-effort injection
            span = _otel_trace.get_current_span()
            if span is not None:
                ctx = span.get_span_context()
                if ctx and getattr(ctx, "trace_id", 0):
                    trace_id = format(ctx.trace_id, "032x")
                    span_id = format(ctx.span_id, "016x")
                    out.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            pass
    # Also propagate X-Request-Id from tracing baggage when available
    try:
        tm = get_tracing_manager()
        req_id = tm.get_baggage("request_id")
        if req_id:
            out.setdefault("X-Request-Id", str(req_id))
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return out


def _validate_egress_or_raise(
    url: str,
    *,
    dns_pin_cache: dict[str, tuple[str, ...]] | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
) -> None:
    from urllib.parse import urlparse as _urlparse

    from tldw_Server_API.app.core.Security.egress import evaluate_url_policy

    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )

    # In test environments, avoid DNS-based private IP checks for hostnames
    # (they can hang or be environment-dependent). For literal IPs we still
    # enforce the default private IP policy so security-focused tests remain
    # accurate.
    block_override: bool | None = None
    host = ""
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        host = ((_urlparse(url).hostname or "").strip())
    if is_explicit_pytest_runtime() or env_flag_enabled("TESTING") or is_test_mode():
        try:
            parsed = _urlparse(url)
            host = (parsed.hostname or "").strip()
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            host = ""
        is_ip = False
        if host:
            try:
                import ipaddress as _ipaddr
                _ipaddr.ip_address(host)
                is_ip = True
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                is_ip = False
        if not is_ip:
            block_override = False

    cache_host = _normalize_dns_pin_host(host)
    pinned_ips: tuple[str, ...] | None = None
    if dns_pin_cache is not None and cache_host:
        pinned_ips = dns_pin_cache.get(cache_host)

    policy_kwargs: dict[str, Any] = {"block_private_override": block_override}
    if configured_endpoint is not None:
        policy_kwargs["configured_endpoint"] = configured_endpoint
    if sensitive_observability:
        policy_kwargs["sensitive_observability"] = True
    if pinned_ips:
        policy_kwargs["pinned_resolved_ips"] = pinned_ips
    res = evaluate_url_policy(url, **policy_kwargs)
    if not pinned_ips:
        if dns_pin_cache is not None and cache_host:
            resolved_ips = tuple(
                str(ip).strip()
                for ip in (getattr(res, "resolved_ips", ()) or ())
                if str(ip).strip()
            )
            if resolved_ips:
                dns_pin_cache[cache_host] = resolved_ips

    if not getattr(res, "allowed", False):
        reason = res.reason or "URL not allowed by egress policy"
        # metrics
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            get_metrics_registry().increment(
                "http_client_egress_denials_total",
                1,
                labels={
                    "reason": "sensitive_endpoint_denied"
                    if sensitive_observability
                    else (reason or "denied")
                },
            )
        if sensitive_observability:
            raise EgressPolicyError(
                "Sensitive endpoint denied by egress policy",
                reason_code=getattr(res, "reason_code", None),
            ) from None
        raise EgressPolicyError(reason, reason_code=getattr(res, "reason_code", None))


async def _avalidate_egress_or_raise(
    url: str,
    *,
    dns_pin_cache: dict[str, tuple[str, ...]] | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
) -> None:
    await asyncio.to_thread(
        _validate_egress_or_raise,
        url,
        dns_pin_cache=dns_pin_cache,
        configured_endpoint=configured_endpoint,
        sensitive_observability=sensitive_observability,
    )


def _is_url_allowed(url: str) -> bool:
    """Lightweight policy check used by simple fetch path (tests monkeypatch this).

    Delegates to the central egress policy evaluator and returns a boolean.
    """
    try:
        from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
        policy_kwargs = (
            {"sensitive_observability": True}
            if _effective_sensitive_observability()
            else {}
        )
        res = evaluate_url_policy(url, **policy_kwargs)
        return bool(getattr(res, "allowed", False))
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        # Fail closed in strict paths; the simple path's callers expect explicit
        # ValueError on denial and do not rely on exceptions from here.
        return False


def _validate_proxies_or_raise(proxies: str | dict[str, str] | None) -> None:
    if not proxies:
        return
    hosts: set[str] = set()
    if isinstance(proxies, str):
        hosts.add(_parse_host_from_url(proxies))
    elif isinstance(proxies, dict):
        for v in proxies.values():
            hosts.add(_parse_host_from_url(v))
    # Deny by default: if allowlist is empty, proxies are disabled
    if not PROXY_ALLOWLIST:
        raise EgressPolicyError("Proxies not allowed (no allowlist configured)")  # noqa: TRY003
    for h in hosts:
        if not h:
            continue
        if h not in PROXY_ALLOWLIST:
            raise EgressPolicyError(f"Proxy host not in allowlist: {h}")  # noqa: TRY003


def validate_proxies_or_raise(proxies: str | dict[str, str] | None) -> None:
    """Validate outbound proxies through the central egress policy."""
    _validate_proxies_or_raise(proxies)


def _resolve_proxy_for_url(url: str, proxies: str | dict[str, str] | None) -> str | None:
    if not proxies:
        return None
    if isinstance(proxies, str):
        return proxies
    try:
        scheme = (urlparse(url).scheme or "").lower()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        scheme = ""
    if scheme and scheme in proxies:
        return proxies.get(scheme)
    return proxies.get("http") or proxies.get("https")


def _resolve_redirect_url(base_url: str, location: str) -> str | None:
    try:
        if httpx is not None:
            try:
                return str(httpx.URL(base_url).join(httpx.URL(location)))
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                return str(httpx.URL(location))
        return str(urljoin(base_url, location))
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None


# Headers that carry credentials and must not accompany a redirect hop to a
# different origin (scheme/host/port) — the redirect target would otherwise
# receive the caller's secrets.
SENSITIVE_REDIRECT_HEADERS: frozenset[str] = frozenset({
    "authorization",
    "proxy-authorization",
    "cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
})


def _url_origin(url: str) -> tuple[str, str, int] | None:
    """Return ``(scheme, host, port)`` for *url*, normalizing default ports.

    Returns ``None`` when the URL has no parseable scheme/host.
    """
    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()
        if not scheme or not host:
            return None
        port = parsed.port or (443 if scheme == "https" else 80)
        return scheme, host, int(port)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None


def _is_cross_origin(original_url: str, target_url: str) -> bool:
    """Return True when *target_url* has a different origin than *original_url*.

    Unparseable URLs are treated as cross-origin (fail closed).
    """
    original = _url_origin(original_url)
    target = _url_origin(target_url)
    if original is None or target is None:
        return True
    return original != target


def _strip_sensitive_headers_for_cross_origin(
    headers: dict[str, str] | None,
    *,
    original_url: str,
    target_url: str,
) -> dict[str, str] | None:
    """Drop credential-bearing headers when a request hop leaves the original origin.

    Args:
        headers: Prepared per-hop request headers (may be ``None``).
        original_url: URL of the caller's original request.
        target_url: URL of the hop about to be issued (redirect target).

    Returns:
        *headers* unchanged for same-origin hops; otherwise a copy with the
        :data:`SENSITIVE_REDIRECT_HEADERS` removed (case-insensitive).
    """
    if not headers or not _is_cross_origin(original_url, target_url):
        return headers
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in headers.items():
        if key.lower() in SENSITIVE_REDIRECT_HEADERS:
            dropped.append(key)
        else:
            kept[key] = value
    dropped.sort()
    if dropped:
        observed_target = (
            _SENSITIVE_OBSERVABILITY_URL
            if _effective_sensitive_observability()
            else target_url
        )
        logger.bind(target_host=_parse_host_from_url(observed_target)).debug(
            "Stripped sensitive headers on cross-origin redirect: {}", ", ".join(dropped)
        )
    return kept


def _cookies_for_hop(
    cookies: dict[str, str] | None, *, original_url: str, target_url: str
) -> dict[str, str] | None:
    """Return *cookies* for same-origin hops, ``None`` for cross-origin hops."""
    if cookies and _is_cross_origin(original_url, target_url):
        return None
    return cookies


def _clear_client_cookie_jar(client: Any) -> None:
    """Best-effort clear of a client/session cookie jar at a cross-origin redirect boundary.

    Ambient jar state (cookies accumulated on the httpx client / aiohttp
    session) would otherwise ride along to the redirect target; mirrors the
    boundary behavior of the simple-fetch redirect loop's cookie clearing.

    Args:
        client: httpx ``Client``/``AsyncClient`` or aiohttp ``ClientSession``
            (any object exposing ``cookies``/``cookie_jar``/``jar``).
    """
    for attr_name in ("cookies", "cookie_jar", "jar"):
        store = getattr(client, attr_name, None)
        if store is None:
            continue
        clear = getattr(store, "clear", None)
        if callable(clear):
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                clear()
            continue
        if isinstance(store, dict):
            store.clear()


def _get_response_url(resp: Any, fallback: str) -> str:
    try:
        req = getattr(resp, "request", None)
        url = getattr(req, "url", None)
        if url:
            return str(url)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        url = getattr(resp, "url", None)
        if url:
            return str(url)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return str(fallback)


def _is_dns_resolution_error(exc: Exception) -> bool:
    """Best-effort detection of DNS resolution / unknown-host failures.

    Looks for socket.gaierror in the exception chain and for common
    platform-specific substrings in the message, including the explicit
    sentinel used by this module ("DNSResolutionError").
    """
    try:
        if getattr(exc, "_tldw_dns_resolution", False):
            return True
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        import socket as _socket

        markers = (
            "nodename nor servname provided",
            "Name or service not known",
            "Temporary failure in name resolution",
            "Host could not be resolved",
            "DNSResolutionError",
        )
        seen_ids: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen_ids:
            seen_ids.add(id(cur))
            if isinstance(cur, _socket.gaierror):
                return True
            msg = str(cur)
            if any(m in msg for m in markers):
                return True
            next_exc = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
            if not isinstance(next_exc, BaseException):
                break
            cur = next_exc
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return False
    return False


def _is_aiohttp_client(client: Any) -> bool:
    if aiohttp is None:
        return False
    try:
        return isinstance(client, aiohttp.ClientSession)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return False


def _network_error_classification(exc: BaseException) -> str | None:
    """Return stable transport-error metadata without parsing exception text."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    try:
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            return "timeout"
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        if aiohttp is not None and isinstance(exc, aiohttp.ServerTimeoutError):
            return "timeout"
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return None


def _aiohttp_timeout_from_defaults() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=None,
        connect=DEFAULT_CONNECT_TIMEOUT,
        sock_connect=DEFAULT_CONNECT_TIMEOUT,
        sock_read=DEFAULT_READ_TIMEOUT,
    )


def _aiohttp_timeout_from_value(timeout: Any | None) -> aiohttp.ClientTimeout | None:
    if aiohttp is None:  # pragma: no cover
        return None
    if timeout is None:
        return _aiohttp_timeout_from_defaults()
    if isinstance(timeout, aiohttp.ClientTimeout):
        return timeout
    if isinstance(timeout, (int, float)):
        return aiohttp.ClientTimeout(total=float(timeout))
    # httpx.Timeout or similar object with connect/read attrs
    connect = getattr(timeout, "connect", None)
    read = getattr(timeout, "read", None)
    return aiohttp.ClientTimeout(
        total=None,
        connect=connect if connect is not None else DEFAULT_CONNECT_TIMEOUT,
        sock_connect=connect if connect is not None else DEFAULT_CONNECT_TIMEOUT,
        sock_read=read if read is not None else DEFAULT_READ_TIMEOUT,
    )


def _aiohttp_ssl_from_verify(verify: Any | None) -> Any | None:
    if aiohttp is None:  # pragma: no cover
        return None
    if verify is None or verify is True:
        return _build_ssl_context(ENFORCE_TLS_MIN, TLS_MIN_VERSION)
    if verify is False:
        return False
    if isinstance(verify, ssl.SSLContext):
        return verify
    if isinstance(verify, str):
        try:
            ctx = ssl.create_default_context(cafile=verify)
            if ENFORCE_TLS_MIN:
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    ctx.minimum_version = _tls_min_version_from_str(TLS_MIN_VERSION)
            return ctx  # noqa: TRY300
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            return _build_ssl_context(ENFORCE_TLS_MIN, TLS_MIN_VERSION)
    return _build_ssl_context(ENFORCE_TLS_MIN, TLS_MIN_VERSION)


_AIOHTTP_SESSION_CACHE: dict[int, Any] = {}
_AIOHTTP_SESSION_LOCK = threading.Lock()
_HTTPX_ASYNC_CLIENT_CACHE: dict[tuple[int, Any], Any] = {}
_HTTPX_CLIENT_CACHE: dict[Any, Any] = {}
_HTTPX_CLIENT_LOCK = threading.Lock()


def _normalize_httpx_cache_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            return tuple(sorted((str(k), str(v)) for k, v in value.items()))
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            return repr(value)
    if isinstance(value, ssl.SSLContext):
        return ("sslcontext", id(value))
    return str(value)


def _httpx_cache_key(
    proxies: Any | None,
    verify: Any | None,
    *,
    factory: Any | None = None,
) -> tuple[Any, Any, Any]:
    factory_key: Any = factory
    if factory is not None:
        try:
            hash(factory)
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            factory_key = id(factory)
    return (
        _normalize_httpx_cache_value(proxies),
        _normalize_httpx_cache_value(verify),
        factory_key,
    )


def _get_httpx_async_client(
    *,
    proxies: str | dict[str, str] | None = None,
    verify: bool | str | ssl.SSLContext | None = None,
) -> httpx.AsyncClient:
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    loop = asyncio.get_running_loop()
    key = (id(loop), _httpx_cache_key(proxies, verify, factory=create_async_client))
    with _HTTPX_CLIENT_LOCK:
        client = _HTTPX_ASYNC_CLIENT_CACHE.get(key)
        if client is not None and not getattr(client, "is_closed", False):
            return client
        client = create_async_client(proxies=proxies, verify=verify)
        _HTTPX_ASYNC_CLIENT_CACHE[key] = client
        return client


def _get_httpx_client(
    *,
    proxies: str | dict[str, str] | None = None,
    verify: bool | str | ssl.SSLContext | None = None,
) -> httpx.Client:
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    key = _httpx_cache_key(proxies, verify, factory=create_client)
    with _HTTPX_CLIENT_LOCK:
        client = _HTTPX_CLIENT_CACHE.get(key)
        if client is not None and not getattr(client, "is_closed", False):
            return client
        client = create_client(proxies=proxies, verify=verify)
        _HTTPX_CLIENT_CACHE[key] = client
        return client


def _get_aiohttp_session() -> aiohttp.ClientSession:
    if aiohttp is None:  # pragma: no cover
        raise RuntimeError("aiohttp is not available")  # noqa: TRY003
    loop = asyncio.get_running_loop()
    key = id(loop)
    with _AIOHTTP_SESSION_LOCK:
        session = _AIOHTTP_SESSION_CACHE.get(key)
        if session is not None and not getattr(session, "closed", False):
            return session
        connector = None
        try:
            ssl_ctx = _build_ssl_context(ENFORCE_TLS_MIN, TLS_MIN_VERSION)
            connector = aiohttp.TCPConnector(
                limit=int(os.getenv("HTTP_MAX_CONNECTIONS", "100")),
                limit_per_host=int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20")),
                ssl=ssl_ctx,
            )
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            connector = None
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=_aiohttp_timeout_from_defaults(),
            trust_env=DEFAULT_TRUST_ENV,
            headers=_build_default_headers(),
        )
        try:
            env_pins = _parse_pins_from_env()
            if env_pins:
                session._tldw_cert_pinning = env_pins
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            pass
        _AIOHTTP_SESSION_CACHE[key] = session
        return session


async def shutdown_http_client() -> None:
    if aiohttp is None:
        sessions: list[Any] = []
    else:
        sessions = []
        with _AIOHTTP_SESSION_LOCK:
            sessions = list(_AIOHTTP_SESSION_CACHE.values())
            _AIOHTTP_SESSION_CACHE.clear()
    for session in sessions:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            await session.close()
    async_clients: list[Any] = []
    sync_clients: list[Any] = []
    with _HTTPX_CLIENT_LOCK:
        async_clients = list(_HTTPX_ASYNC_CLIENT_CACHE.values())
        _HTTPX_ASYNC_CLIENT_CACHE.clear()
        sync_clients = list(_HTTPX_CLIENT_CACHE.values())
        _HTTPX_CLIENT_CACHE.clear()
    for client in async_clients:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            await client.aclose()
    for client in sync_clients:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            client.close()


def _iter_file_items(files: Any) -> Iterable[tuple[str, Any]]:
    if files is None:
        return []
    if isinstance(files, dict):
        return list(files.items())
    return list(files)


def _rewind_files(files: Any) -> None:
    for _, spec in _iter_file_items(files):
        try:
            file_obj = spec[1] if isinstance(spec, (tuple, list)) and len(spec) >= 2 else spec
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            continue


def _validate_retry_files_seekable(files: Any, retry: RetryPolicy | None) -> None:
    if files is None or retry is None:
        return
    if not retry.enabled:
        return
    for _, spec in _iter_file_items(files):
        file_obj = spec[1] if isinstance(spec, (tuple, list)) and len(spec) >= 2 else spec
        if hasattr(file_obj, "seekable"):
            try:
                is_seekable = file_obj.seekable()
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as exc:
                raise ValueError(  # noqa: TRY003
                    "File-like object must be seekable when retries are enabled. "
                    "Either disable retries or use a seekable stream."
                ) from exc
            if is_seekable is False:
                raise ValueError(  # noqa: TRY003
                    "File-like object must be seekable when retries are enabled. "
                    "Either disable retries or use a seekable stream."
                )


def _build_aiohttp_form(data: Any | None, files: Any | None) -> aiohttp.FormData | None:
    if aiohttp is None:  # pragma: no cover
        return None
    if files is None:
        return None
    form = aiohttp.FormData()
    if isinstance(data, dict):
        for key, val in data.items():
            form.add_field(str(key), str(val))
    elif isinstance(data, (list, tuple)):
        for key, val in data:
            form.add_field(str(key), str(val))
    elif data is not None:
        form.add_field("data", str(data))
    for name, spec in _iter_file_items(files):
        filename = None
        file_obj = None
        content_type = None
        if isinstance(spec, (tuple, list)):
            if len(spec) >= 1:
                filename = spec[0]
            if len(spec) >= 2:
                file_obj = spec[1]
            if len(spec) >= 3:
                content_type = spec[2]
        else:
            file_obj = spec
        if filename is None:
            filename = getattr(file_obj, "name", None) or "file"
        form.add_field(
            str(name),
            file_obj,
            filename=str(filename),
            content_type=content_type,
        )
    return form


def _decorrelated_jitter_sleep(prev: float, base_ms: int, cap_s: int) -> float:
    base = max(0.001, base_ms / 1000.0)
    cap = max(base, float(cap_s))
    sleep = base if prev <= 0 else min(cap, random.uniform(base, prev * 3))  # nosec B311
    return sleep


def _parse_retry_after_delay_seconds(
    retry_after: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse Retry-After as delta-seconds or HTTP-date and return seconds to wait."""
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after.strip()))
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        parsed = parsedate_to_datetime(retry_after.strip())
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (parsed - current).total_seconds())
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None


def _should_retry(method: str, status: int | None, exc: Exception | None, policy: RetryPolicy) -> tuple[bool, str]:
    m = method.upper()
    if exc is not None:
        if m not in policy.retry_on_methods and not policy.retry_on_unsafe:
            return False, "method_not_retriable"
        # Treat DNS resolution / unknown-host failures as permanent.
        try:
            if _is_dns_resolution_error(exc):
                return False, exc.__class__.__name__
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            pass
        # Other network-level exceptions remain retriable.
        return True, exc.__class__.__name__
    if status is None:
        return False, "no_status"
    if status in policy.retry_on_status and (m in policy.retry_on_methods or policy.retry_on_unsafe):
        return True, f"{status}"
    return False, "status_not_retriable"


def _build_default_headers(component: str | None = None) -> dict[str, str]:
    # Standardize UA: tldw_server/<version> (<component>)
    version = _get_project_version()
    ua = f"tldw_server/{version} ({component})" if component else f"tldw_server/{version}"
    # Allow env to override completely if provided
    if os.getenv("HTTP_DEFAULT_USER_AGENT"):
        ua = os.getenv("HTTP_DEFAULT_USER_AGENT") or ua
    return {"User-Agent": ua}


def _httpx_limits_default():  # Optional["httpx.Limits"]
    """Return a default httpx.Limits if available; otherwise None.

    Some tests stub `httpx` with minimal objects lacking `Limits`. In that case,
    skip providing limits entirely so the client factory can succeed.
    """
    try:
        _hx = _resolve_httpx()
        if _hx is not None and hasattr(_hx, "Limits"):
            return _hx.Limits(
                max_connections=int(os.getenv("HTTP_MAX_CONNECTIONS", "100")),
                max_keepalive_connections=int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20")),
            )
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return None


def _tls_min_version_from_str(ver: str | None) -> ssl.TLSVersion:
    try:
        v = (ver or "1.2").strip().lower()
        if v in {"1.3", "tls1.3", "tlsv1.3"}:
            return ssl.TLSVersion.TLSv1_3
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return ssl.TLSVersion.TLSv1_2


def _build_ssl_context(enforce_min: bool, min_ver: str | None) -> ssl.SSLContext | None:
    if not enforce_min:
        return None
    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    try:
        ctx.minimum_version = _tls_min_version_from_str(min_ver)
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def _get_client_cert_pins(client: Any) -> dict[str, set[str]] | None:
    try:
        pins = getattr(client, "_tldw_cert_pinning", None)
        if pins is None:
            return None
        out: dict[str, set[str]] = {}
        for host, vals in pins.items():
            out[_normalize_dns_pin_host(str(host))] = {
                str(v).lower() for v in (vals or set())
            }
        return out  # noqa: TRY300
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None


def _parse_pins_from_env() -> dict[str, set[str]] | None:
    """Parse env-driven certificate pins: HTTP_CERT_PINS="hostA=pinA|pinB,hostB=pinC".

    Returns a mapping host -> set of lowercase sha256 hex pins.
    """
    raw = os.getenv("HTTP_CERT_PINS", "").strip()
    if not raw:
        return None
    out: dict[str, set[str]] = {}
    try:
        parts = [p for p in re.split(r"[,;]", raw) if p]
        for part in parts:
            if "=" not in part:
                continue
            host, pins_str = part.split("=", 1)
            host = _normalize_dns_pin_host(host)
            pins = {p.strip().lower() for p in pins_str.split("|") if p.strip()}
            if host and pins:
                out[host] = pins
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        return None
    return out or None


def _check_cert_pinning(
    host: str,
    port: int,
    pins: set[str],
    min_ver: str | None,
    *,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    accepted_resolved_ips: tuple[str, ...] = (),
    sensitive_observability: bool = False,
) -> None:
    if not host or not pins:
        return
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    try:
        # Enforce egress policy for the pinning connection itself. This guards
        # against any future callers that might invoke pinning without having
        # already passed through the main egress checks.
        try:
            normalized_host = _normalize_dns_pin_host(host)
            url_host = normalized_host
            with suppress(ValueError):
                import ipaddress

                if ipaddress.ip_address(normalized_host).version == 6:
                    url_host = f"[{normalized_host}]"
            url = f"https://{url_host}"
            if port != 443:
                url = f"https://{url_host}:{port}"
            validation_kwargs: dict[str, Any] = {}
            if configured_endpoint is not None:
                validation_kwargs["configured_endpoint"] = configured_endpoint
            if accepted_resolved_ips:
                validation_kwargs["dns_pin_cache"] = {
                    normalized_host: tuple(accepted_resolved_ips)
                }
            if sensitive_observability:
                validation_kwargs["sensitive_observability"] = True
            _validate_egress_or_raise(url, **validation_kwargs)
        except EgressPolicyError:
            raise
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
            raise EgressPolicyError(
                f"TLS pinning egress check failed: {e}",
                reason_code="tls_pin_error",
            ) from e

        ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        try:
            ctx.minimum_version = _tls_min_version_from_str(min_ver)
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        connect_host = accepted_resolved_ips[0] if accepted_resolved_ips else host
        with socket.create_connection((connect_host, port), timeout=DEFAULT_CONNECT_TIMEOUT) as sock, \
             ctx.wrap_socket(sock, server_hostname=host) as ssock:
            der = ssock.getpeercert(binary_form=True)
        if not der:
            raise EgressPolicyError(
                "TLS pinning: no certificate presented",
                reason_code="tls_pin_missing",
            )
        fp = hashlib.sha256(der).hexdigest().lower()
        if fp not in pins:
            raise EgressPolicyError(
                "TLS pinning mismatch for host",
                reason_code="tls_pin_mismatch",
            )
    except EgressPolicyError:
        raise
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
        raise EgressPolicyError(
            f"TLS pinning verification failed: {e}",
            reason_code="tls_pin_error",
        ) from e


def _check_cert_pins_for_url(
    url: str,
    pins_map: dict[str, set[str]] | None,
    *,
    configured_endpoint: ConfiguredEndpointScope | None,
    dns_pin_cache: dict[str, tuple[str, ...]],
    sensitive_observability: bool = False,
) -> None:
    """Apply configured certificate pins to one already-validated request hop."""
    if not pins_map:
        return
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or not host:
        return
    pins = _pins_for_host(pins_map, host)
    if not pins:
        return
    _check_cert_pinning(
        host,
        parsed.port or 443,
        pins,
        TLS_MIN_VERSION,
        configured_endpoint=configured_endpoint,
        accepted_resolved_ips=_accepted_ips_for_url(dns_pin_cache, url),
        sensitive_observability=sensitive_observability,
    )


# --------------------------------------------------------------------------------------
# Client factories
# --------------------------------------------------------------------------------------

def _instantiate_client(factory, kwargs: dict[str, Any]):  # type: ignore[no-untyped-def]
    """Instantiate httpx client tolerating version differences in kwargs.

    On TypeError mentioning an unexpected keyword argument, remove that kwarg
    and retry. This also supports tests that stub `httpx.Client` with a minimal
    constructor accepting only a subset (e.g., just `timeout`).
    """
    import re as _re
    while True:
        try:
            return factory(**kwargs)
        except TypeError as e:
            msg = str(e)
            # Look for patterns like: "unexpected keyword argument 'foo'"
            m = _re.search(r"unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", msg)
            key = m.group(1) if m else None
            if key and key in kwargs:
                kwargs.pop(key, None)
                continue
            # Fallback: remove commonly problematic kwargs if present, then retry once
            removed_any = False
            for k in ["limits", "http2", "proxies", "headers", "transport", "trust_env", "base_url", "verify"]:
                if k in kwargs:
                    kwargs.pop(k, None)
                    removed_any = True
            if removed_any:
                continue
            raise
        except ImportError as e:
            # Gracefully disable http2 if h2 not installed
            if "Using http2=True" in str(e) and kwargs.get("http2"):
                kwargs["http2"] = False
                continue
            raise


def create_async_client(
    *,
    timeout: float | httpx.Timeout | None = None,
    limits: httpx.Limits | None = None,
    base_url: str | None = None,
    proxies: str | dict[str, str] | None = None,
    trust_env: bool = DEFAULT_TRUST_ENV,
    http2: bool = True,
    http3: bool = False,  # placeholder for future
    headers: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    enforce_tls_min_version: bool = ENFORCE_TLS_MIN,
    tls_min_version: str = TLS_MIN_VERSION,
    cert_pinning: dict[str, set[str]] | None = None,
    verify: bool | str | ssl.SSLContext | None = None,
) -> httpx.AsyncClient:
    _hx = _resolve_httpx()
    if _hx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    _validate_proxies_or_raise(proxies)
    # Build a timeout value tolerant of stubbed httpx without Timeout class
    if hasattr(_hx, "Timeout"):
        try:
            to = (
                timeout
                if isinstance(timeout, _hx.Timeout)
                else (timeout if timeout is not None else _httpx_timeout_from_defaults())
            )
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            to = timeout if timeout is not None else DEFAULT_READ_TIMEOUT
        if not isinstance(to, getattr(_hx, "Timeout", object)):
            try:
                to = _hx.Timeout(float(to))
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                to = float(timeout) if timeout is not None else DEFAULT_READ_TIMEOUT
    else:
        to = float(timeout) if timeout is not None else DEFAULT_READ_TIMEOUT
    hdrs = _build_default_headers()
    if headers:
        hdrs.update(headers)
    verify_ctx = _build_ssl_context(enforce_tls_min_version, tls_min_version)
    kwargs: dict[str, Any] = {
        "timeout": to,
        "trust_env": trust_env,
        "http2": http2,
        "proxies": proxies,
        "headers": hdrs,
        "transport": transport,
    }
    kwargs["event_hooks"] = {"response": [_capture_error_body_hook_async]}
    lim = limits or _httpx_limits_default()
    if lim is not None:
        kwargs["limits"] = lim
    if verify is not None:
        kwargs["verify"] = verify
    elif verify_ctx is not None:
        kwargs["verify"] = verify_ctx
    if base_url is not None:
        kwargs["base_url"] = base_url
    client = _instantiate_client(getattr(_hx, "AsyncClient", object), kwargs)
    try:
        if cert_pinning:
            client._tldw_cert_pinning = cert_pinning
        else:
            env_pins = _parse_pins_from_env()
            if env_pins:
                client._tldw_cert_pinning = env_pins
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return client


def create_client(
    *,
    timeout: float | httpx.Timeout | None = None,
    limits: httpx.Limits | None = None,
    base_url: str | None = None,
    proxies: str | dict[str, str] | None = None,
    trust_env: bool = DEFAULT_TRUST_ENV,
    http2: bool = True,
    http3: bool = False,  # placeholder for future
    headers: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    follow_redirects: bool = False,
    enforce_tls_min_version: bool = ENFORCE_TLS_MIN,
    tls_min_version: str = TLS_MIN_VERSION,
    cert_pinning: dict[str, set[str]] | None = None,
    verify: bool | str | ssl.SSLContext | None = None,
) -> httpx.Client:
    _hx = _resolve_httpx()
    if _hx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    _validate_proxies_or_raise(proxies)
    # Build a timeout value tolerant of stubbed httpx without Timeout class
    if hasattr(_hx, "Timeout"):
        try:
            to = (
                timeout
                if isinstance(timeout, _hx.Timeout)
                else (timeout if timeout is not None else _httpx_timeout_from_defaults())
            )
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            to = timeout if timeout is not None else DEFAULT_READ_TIMEOUT
        if not isinstance(to, getattr(_hx, "Timeout", object)):
            try:
                to = _hx.Timeout(float(to))
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                to = float(timeout) if timeout is not None else DEFAULT_READ_TIMEOUT
    else:
        to = float(timeout) if timeout is not None else DEFAULT_READ_TIMEOUT
    hdrs = _build_default_headers()
    if headers:
        hdrs.update(headers)
    verify_ctx = _build_ssl_context(enforce_tls_min_version, tls_min_version)
    kwargs: dict[str, Any] = {
        "timeout": to,
        "trust_env": trust_env,
        "http2": http2,
        "proxies": proxies,
        "headers": hdrs,
        "transport": transport,
        "follow_redirects": follow_redirects,
    }
    kwargs["event_hooks"] = {"response": [_capture_error_body_hook]}
    lim = limits or _httpx_limits_default()
    if lim is not None:
        kwargs["limits"] = lim
    if verify is not None:
        kwargs["verify"] = verify
    elif verify_ctx is not None:
        kwargs["verify"] = verify_ctx
    if base_url is not None:
        kwargs["base_url"] = base_url
    # Debug which factory is being used, to verify test monkeypatches
    try:
        from loguru import logger as _logger  # local import to avoid global cost
        _logger.debug("http_client.create_client: httpx.Client factory={} kwargs_keys={}", getattr(_hx, "Client", None), list(kwargs.keys()))
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    client = _instantiate_client(getattr(_hx, "Client", object), kwargs)
    try:
        if cert_pinning:
            client._tldw_cert_pinning = cert_pinning
        else:
            env_pins = _parse_pins_from_env()
            if env_pins:
                client._tldw_cert_pinning = env_pins
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
        pass
    return client


# --------------------------------------------------------------------------------------
# Transport-only IO helpers (no policy enforcement)
# --------------------------------------------------------------------------------------

def _httpx_request_io(
    *,
    client: httpx.Client,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    follow_redirects: bool = False,
    accepted_resolved_ips: tuple[str, ...] = (),
) -> httpx.Response:
    method_upper = str(method).upper()
    transport_url, headers, sni_hostname = _prepare_pinned_transport_target(
        url,
        headers,
        accepted_resolved_ips,
    )
    if accepted_resolved_ips:
        response = client.request(
            method_upper,
            transport_url,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            follow_redirects=follow_redirects,
            extensions={"sni_hostname": sni_hostname},
        )
        _restore_httpx_response_url(response, url)
        return response
    if method_upper == "POST" and hasattr(client, "post"):
        return client.post(
            url,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )
    if method_upper == "GET" and hasattr(client, "get"):
        return client.get(
            url,
            headers=headers,
            cookies=cookies,
            params=params,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )
    req_kwargs: dict[str, Any] = {
        "headers": headers,
        "cookies": cookies,
        "params": params,
        "json": json,
        "data": data,
        "files": files,
        "timeout": timeout,
        "follow_redirects": follow_redirects,
    }
    return client.request(method_upper, url, **req_kwargs)


async def _httpx_arequest_io(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    follow_redirects: bool = False,
    verify: bool | str | ssl.SSLContext | None = None,
    accepted_resolved_ips: tuple[str, ...] = (),
    max_response_bytes: int | None = None,
) -> httpx.Response:
    method_upper = str(method).upper()
    transport_url, headers, sni_hostname = _prepare_pinned_transport_target(
        url,
        headers,
        accepted_resolved_ips,
    )
    if max_response_bytes is not None:
        if type(max_response_bytes) is not int or max_response_bytes < 0:
            raise ValueError("max_response_bytes must be a non-negative integer")
        headers = _force_identity_accept_encoding(headers)
        stream_kwargs: dict[str, Any] = {
            "headers": headers,
            "cookies": cookies,
            "params": params,
            "json": json,
            "data": data,
            "files": files,
            "timeout": timeout,
            "follow_redirects": follow_redirects,
        }
        stream_kwargs["extensions"] = {
            _BOUNDED_RESPONSE_EXTENSION: True,
        }
        if accepted_resolved_ips:
            stream_kwargs["extensions"]["sni_hostname"] = sni_hostname
        async with client.stream(
            method_upper,
            transport_url,
            **stream_kwargs,
        ) as streamed:
            body = bytearray()
            if 200 <= int(streamed.status_code) < 300:
                if _uses_compressed_content_encoding(streamed.headers):
                    raise NetworkError(
                        "Compressed responses are not allowed with "
                        "max_response_bytes"
                    )
                async for chunk in streamed.aiter_raw():
                    if len(body) + len(chunk) > max_response_bytes:
                        raise NetworkError(
                            "Response exceeds max_response_bytes limit"
                        )
                    body.extend(chunk)
            copied_headers = httpx.Headers(streamed.headers)
            response = httpx.Response(
                status_code=streamed.status_code,
                content=bytes(body),
                request=streamed.request,
                extensions=streamed.extensions,
                history=streamed.history,
            )
            response.headers = copied_headers
        if accepted_resolved_ips:
            _restore_httpx_response_url(response, url)
        return response
    if accepted_resolved_ips:
        response = await client.request(
            method_upper,
            transport_url,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            follow_redirects=follow_redirects,
            extensions={"sni_hostname": sni_hostname},
        )
        _restore_httpx_response_url(response, url)
        return response
    if method_upper == "POST" and hasattr(client, "post") and verify is None:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            logger.debug("afetch io: using AsyncClient.post")
        return await client.post(
            url,
            headers=headers,
            cookies=cookies,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )
    if method_upper == "GET" and hasattr(client, "get") and verify is None:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            logger.debug("afetch io: using AsyncClient.get")
        return await client.get(
            url,
            headers=headers,
            cookies=cookies,
            params=params,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )
    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
        logger.debug("afetch io: using AsyncClient.request")
    req_kwargs: dict[str, Any] = {
        "headers": headers,
        "cookies": cookies,
        "params": params,
        "json": json,
        "data": data,
        "files": files,
        "timeout": timeout,
        "follow_redirects": follow_redirects,
    }
    return await client.request(method_upper, url, **req_kwargs)


async def _aiohttp_request_io(
    *,
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    proxies: str | dict[str, str] | None = None,
    ssl_override: Any | None = None,
    accepted_resolved_ips: tuple[str, ...] = (),
    max_response_bytes: int | None = None,
) -> _AiohttpResponse:
    original_url = url
    url, headers, server_hostname = _prepare_pinned_transport_target(
        url,
        headers,
        accepted_resolved_ips,
    )
    if max_response_bytes is not None:
        if type(max_response_bytes) is not int or max_response_bytes < 0:
            raise ValueError("max_response_bytes must be a non-negative integer")
        headers = _force_identity_accept_encoding(headers)
    req_timeout = _aiohttp_timeout_from_value(timeout)
    proxy = _resolve_proxy_for_url(url, proxies)
    req_kwargs: dict[str, Any] = {
        "headers": headers,
        "cookies": cookies,
        "params": params,
        "timeout": req_timeout,
        "allow_redirects": False,
    }
    if max_response_bytes is not None:
        req_kwargs["auto_decompress"] = False
    if proxy:
        req_kwargs["proxy"] = proxy
    if ssl_override is not None:
        req_kwargs["ssl"] = ssl_override
    if server_hostname and urlsplit(original_url).scheme.lower() == "https":
        req_kwargs["server_hostname"] = server_hostname
    if files is not None:
        _rewind_files(files)
        req_kwargs["data"] = _build_aiohttp_form(data, files)
    else:
        if json is not None:
            req_kwargs["json"] = json
        if data is not None:
            req_kwargs["data"] = data
    async with session.request(str(method).upper(), url, **req_kwargs) as resp:
        if max_response_bytes is None:
            body = await resp.read()
        else:
            bounded_body = bytearray()
            if 200 <= int(getattr(resp, "status", 0)) < 300:
                if _uses_compressed_content_encoding(resp.headers):
                    raise NetworkError(
                        "Compressed responses are not allowed with "
                        "max_response_bytes"
                    )
                while True:
                    remaining = max_response_bytes - len(bounded_body)
                    chunk = await resp.content.read(remaining + 1)
                    if not chunk:
                        break
                    if len(bounded_body) + len(chunk) > max_response_bytes:
                        raise NetworkError(
                            "Response exceeds max_response_bytes limit"
                        )
                    bounded_body.extend(chunk)
            body = bytes(bounded_body)
        wrapped = _AiohttpResponse(resp, body)
        if accepted_resolved_ips:
            wrapped.url = original_url
            wrapped.request.url = original_url
        return wrapped


@asynccontextmanager
async def _httpx_stream_io(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    chunk_size: int | None = None,
    accepted_resolved_ips: tuple[str, ...] = (),
) -> AsyncIterator[tuple[httpx.Response, AsyncIterator[bytes]]]:
    original_url = url
    url, headers, sni_hostname = _prepare_pinned_transport_target(
        url,
        headers,
        accepted_resolved_ips,
    )
    extensions: dict[str, Any] = {"tldw_stream_response": True}
    if sni_hostname:
        extensions["sni_hostname"] = sni_hostname
    async with client.stream(
        str(method).upper(),
        url,
        headers=headers,
        cookies=cookies,
        params=params,
        json=json,
        data=data,
        files=files,
        timeout=timeout,
        follow_redirects=False,
        extensions=extensions,
    ) as resp:
        if accepted_resolved_ips:
            _restore_httpx_response_url(resp, original_url)
        if chunk_size is None:
            yield resp, resp.aiter_bytes()
        else:
            yield resp, resp.aiter_bytes(chunk_size)


@asynccontextmanager
async def _aiohttp_stream_io(
    *,
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    proxies: str | dict[str, str] | None = None,
    ssl_override: Any | None = None,
    chunk_size: int | None = None,
    accepted_resolved_ips: tuple[str, ...] = (),
) -> AsyncIterator[tuple[Any, AsyncIterator[bytes]]]:
    original_url = url
    url, headers, server_hostname = _prepare_pinned_transport_target(
        url,
        headers,
        accepted_resolved_ips,
    )
    req_timeout = _aiohttp_timeout_from_value(timeout)
    proxy = _resolve_proxy_for_url(url, proxies)
    req_kwargs: dict[str, Any] = {
        "headers": headers,
        "cookies": cookies,
        "params": params,
        "timeout": req_timeout,
        "allow_redirects": False,
    }
    if proxy:
        req_kwargs["proxy"] = proxy
    if ssl_override is not None:
        req_kwargs["ssl"] = ssl_override
    if server_hostname and urlsplit(original_url).scheme.lower() == "https":
        req_kwargs["server_hostname"] = server_hostname
    if files is not None:
        _rewind_files(files)
        req_kwargs["data"] = _build_aiohttp_form(data, files)
    else:
        if json is not None:
            req_kwargs["json"] = json
        if data is not None:
            req_kwargs["data"] = data
    async with session.request(str(method).upper(), url, **req_kwargs) as resp:
        if chunk_size is None:
            yield resp, resp.content.iter_any()
        else:
            yield resp, resp.content.iter_chunked(chunk_size)


@asynccontextmanager
async def _preserve_terminal_status_during_stream_cleanup(
    stream_context: Any,
) -> AsyncIterator[Any]:
    """Keep an observed terminal HTTP status authoritative over cleanup errors."""
    terminal_error: _TerminalHTTPStatusError | None = None
    try:
        async with stream_context as stream:
            try:
                yield stream
            except _TerminalHTTPStatusError as exc:
                terminal_error = exc
    except asyncio.CancelledError:
        raise
    except _STREAM_CLEANUP_TRANSPORT_EXCEPTIONS:
        if terminal_error is None:
            raise
    if terminal_error is not None:
        raise terminal_error from None


async def _iter_sse_events_from_bytes(
    byte_iter: AsyncIterator[bytes],
    *,
    on_body_started: Callable[[], None] | None = None,
) -> AsyncIterator[SSEEvent]:
    """Parse SSE events and report the first non-empty response body chunk."""
    buffer = ""
    iterator = byte_iter.__aiter__()
    try:
        async for chunk in iterator:
            if not chunk:
                continue
            if on_body_started is not None:
                on_body_started()
                on_body_started = None
            try:
                text = chunk.decode("utf-8", errors="replace")
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                raise StreamingProtocolError(f"Failed to decode SSE chunk: {e}")  # noqa: B904, TRY003
            buffer += text
            while "\n\n" in buffer or "\r\n\r\n" in buffer:
                if "\r\n\r\n" in buffer and (
                    "\n\n" not in buffer
                    or buffer.index("\r\n\r\n") < buffer.index("\n\n")
                ):
                    raw, buffer = buffer.split("\r\n\r\n", 1)
                else:
                    raw, buffer = buffer.split("\n\n", 1)
                event = _parse_sse_event(raw)
                if event is not None:
                    yield event
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


def _stream_timeout_values(timeout: Any | None) -> tuple[float, float]:
    first_byte_timeout = DEFAULT_CONNECT_TIMEOUT
    idle_timeout = DEFAULT_READ_TIMEOUT
    if timeout is None:
        return first_byte_timeout, idle_timeout
    if isinstance(timeout, (int, float)):
        idle_timeout = float(timeout)
        return first_byte_timeout, idle_timeout
    connect = getattr(timeout, "connect", None)
    if connect is None:
        connect = getattr(timeout, "sock_connect", None)
    read = getattr(timeout, "read", None)
    if read is None:
        read = getattr(timeout, "sock_read", None)
    if connect is not None:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            first_byte_timeout = float(connect)
    if read is not None:
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            idle_timeout = float(read)
    return first_byte_timeout, idle_timeout


async def _iter_bytes_with_timeouts(
    byte_iter: AsyncIterator[bytes],
    timeout: Any | None,
) -> AsyncIterator[bytes]:
    first_timeout, idle_timeout = _stream_timeout_values(timeout)
    first_timeout = max(0.001, float(first_timeout))
    idle_timeout = max(0.001, float(idle_timeout))
    first = True
    iterator = byte_iter.__aiter__()
    try:
        while True:
            timeout_s = first_timeout if first else idle_timeout
            try:
                async with _stream_timeout(timeout_s):
                    chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                phase = "first_byte" if first else "idle"
                raise NetworkError(f"StreamTimeout:{phase}") from exc
            first = False
            yield chunk
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


_SAFE_SENSITIVE_SSE_ERRORS = frozenset(
    {
        "Invalid redirect Location header",
        "Redirect without Location header",
        "StreamTimeout:first_byte",
        "StreamTimeout:idle",
        "Too many redirects",
    }
)


def _sse_network_error(exc: BaseException, *, sensitive_observability: bool) -> NetworkError:
    """Normalize SSE failures without exposing endpoint-bearing transport details."""
    message = str(exc)
    if not sensitive_observability or message in _SAFE_SENSITIVE_SSE_ERRORS:
        return NetworkError(message)
    return NetworkError(exc.__class__.__name__)


# --------------------------------------------------------------------------------------
# Core request helpers (sync/async) with retries + redirects + egress checks
# --------------------------------------------------------------------------------------

async def _afetch_httpx(
    *,
    method: str,
    url: str,
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    allow_redirects: bool = True,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    verify: bool | str | ssl.SSLContext | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
    max_response_bytes: int | None = None,
) -> httpx.Response:
    """Async httpx request with retries and egress enforcement.

    ``max_response_bytes`` requests identity encoding and bounds application
    response-body accumulation while streaming successful responses. Encoded
    successful responses are rejected; bounded non-2xx responses are returned
    with an empty body.

    Raises ValueError when retries are enabled and a file-like object in `files`
    is not seekable: "File-like object must be seekable when retries are enabled.
    Either disable retries or use a seekable stream."
    """
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    retry = retry or RetryPolicy()
    _validate_retry_files_seekable(files, retry)
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        configured_endpoint=configured_endpoint,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)

    attempts = max(1, retry.attempts)
    sleep_s = 0.0
    t0 = time.time()
    last_exc: Exception | None = None
    tm = get_tracing_manager()
    observability_url = _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else url
    method_upper = str(method).upper()
    _head_disable_h2_tried = False
    _head_get_range_tried = False

    async def _do_once(ac: httpx.AsyncClient, target_url: str) -> tuple[httpx.Response | None, str, str | None]:
        req_headers = _inject_trace_headers(headers)
        req_headers = _strip_sensitive_headers_for_cross_origin(
            req_headers, original_url=url, target_url=target_url
        )
        req_cookies = _cookies_for_hop(cookies, original_url=url, target_url=target_url)
        # Parity with other paths: drop 'zstd' from Accept-Encoding for httpx
        try:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                logger.debug(
                    f"afetch _do_once: method={method_upper} "
                    f"url={_sanitize_url_for_logs(observability_url)}"
                )
            req_headers = _sanitize_accept_encoding_for_backend(req_headers, "httpx")
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            # Optional cert pinning per host
            try:
                pins_map = cert_pinning or _get_client_cert_pins(ac)
                if pins_map:
                    u = httpx.URL(target_url)
                    if u.scheme.lower() == "https":
                        host = (u.host or "").lower()
                        pins = _pins_for_host(pins_map, host)
                        if pins:
                            _check_cert_pinning(
                                host,
                                int(u.port or 443),
                                pins,
                                TLS_MIN_VERSION,
                                configured_endpoint=configured_endpoint,
                                accepted_resolved_ips=_accepted_ips_for_url(
                                    dns_pin_cache, target_url
                                ),
                                sensitive_observability=sensitive_observability,
                            )
            except EgressPolicyError:
                raise
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                return None, e.__class__.__name__, _network_error_classification(e)
            r = await _httpx_arequest_io(
                client=ac,
                method=method_upper,
                url=target_url,
                headers=req_headers,
                cookies=req_cookies,
                params=params,
                json=json,
                data=data,
                files=files,
                timeout=timeout,
                follow_redirects=False,
                verify=verify,
                accepted_resolved_ips=_accepted_ips_for_url(
                    dns_pin_cache,
                    target_url,
                ),
                max_response_bytes=max_response_bytes,
            )
            return r, "ok", None  # noqa: TRY300
        except EgressPolicyError:
            raise
        except _HTTPCLIENT_REQUEST_EXCEPTIONS as e:
            # Let callers see HTTPStatusError directly so that adapters/tests
            # can distinguish 4xx/5xx responses from transport failures. All
            # other exceptions are normalized into a NetworkError reason.
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                if _opaque_stt_endpoint_id() is not None:
                    logger.debug(
                        "planned STT HTTP transport exception"
                    )
                else:
                    logger.debug(
                        "afetch _do_once: caught exception {}",
                        type(e).__name__,
                    )
            try:
                _hx = _resolve_httpx()
                if _hx is not None and isinstance(e, getattr(_hx, "HTTPStatusError", Exception)):
                    raise
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                # If httpx cannot be resolved for some reason, fall back to
                # treating the error as a generic network failure.
                pass
            # Classify DNS resolution errors explicitly so that retry logic
            # can treat them as permanent failures.
            try:
                if _is_dns_resolution_error(e):
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        e._tldw_dns_resolution = True
                    return None, "DNSResolutionError", _network_error_classification(e)
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                pass
            return None, e.__class__.__name__, _network_error_classification(e)

    # Create ephemeral client if none provided
    need_close = False
    ac = client
    if ac is None:
        ac = _get_httpx_async_client(proxies=proxies, verify=verify)
        need_close = False

    try:
        async with _async_http_client_observability_span(
            tm,
            "http.client",
            sensitive_observability=sensitive_observability,
            attributes=_http_trace_attributes(method, url),
        ):
            for attempt in range(1, attempts + 1):
                last_exc = None
                cur_url = url
                redirects = 0

                # Manual redirect handling inside each attempt
                while True:
                    await _avalidate_egress_or_raise(
                        cur_url,
                        dns_pin_cache=dns_pin_cache,
                        configured_endpoint=configured_endpoint,
                        sensitive_observability=sensitive_observability,
                    )
                    resp, reason, classification = await _do_once(ac, cur_url)
                    if resp is None:
                        # Special HEAD fallbacks: disable HTTP/2, then GET with Range 0-0
                        if method_upper == "HEAD":
                            if not _head_disable_h2_tried:
                                _head_disable_h2_tried = True
                                try:
                                    if need_close:
                                        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                            await ac.aclose()
                                    ac = create_async_client(proxies=proxies, http2=False)
                                    need_close = True
                                    # Retry immediately inside same attempt
                                    continue
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    pass
                            if not _head_get_range_tried:
                                _head_get_range_tried = True
                                try:
                                    await _avalidate_egress_or_raise(
                                        cur_url,
                                        dns_pin_cache=dns_pin_cache,
                                        configured_endpoint=configured_endpoint,
                                        sensitive_observability=sensitive_observability,
                                    )
                                    _check_cert_pins_for_url(
                                        cur_url,
                                        cert_pinning or _get_client_cert_pins(ac),
                                        configured_endpoint=configured_endpoint,
                                        dns_pin_cache=dns_pin_cache,
                                        sensitive_observability=sensitive_observability,
                                    )
                                    req_headers = _inject_trace_headers(headers)
                                    req_headers = _strip_sensitive_headers_for_cross_origin(
                                        req_headers, original_url=url, target_url=cur_url
                                    )
                                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                        req_headers = _sanitize_accept_encoding_for_backend(req_headers, "httpx")
                                    req_headers.setdefault("Range", "bytes=0-0")
                                    # Use a small per-request timeout specifically for this fallback
                                    try:
                                        _head_fb_to = float(os.getenv("HTTP_HEAD_RANGE_FALLBACK_TIMEOUT", "5"))
                                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                        _head_fb_to = 5.0
                                    r2 = await _httpx_arequest_io(
                                        client=ac,
                                        method="GET",
                                        url=cur_url,
                                        headers=req_headers,
                                        cookies=_cookies_for_hop(cookies, original_url=url, target_url=cur_url),
                                        params=params,
                                        json=json,
                                        data=data,
                                        files=files,
                                        timeout=_head_fb_to,
                                        follow_redirects=False,
                                        verify=verify,
                                        accepted_resolved_ips=_accepted_ips_for_url(
                                            dns_pin_cache,
                                            cur_url,
                                        ),
                                        max_response_bytes=max_response_bytes,
                                    )
                                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                        tm.set_attributes({"http.status_code": int(r2.status_code)})
                                    _log_outbound_request(
                                        method="GET",
                                        url=observability_url,
                                        status_code=int(r2.status_code),
                                        start_time=t0,
                                        attempt=attempt,
                                        last_retry_delay_s=sleep_s,
                                    )
                                    return r2  # noqa: TRY300
                                except EgressPolicyError:
                                    raise
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    pass
                        # network exception occurred (no HEAD fallback succeeded)
                        last_exc = NetworkError(reason, classification=classification)
                        try:
                            if reason == "DNSResolutionError":
                                last_exc._tldw_dns_resolution = True
                        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                            pass
                        # Exit redirect loop; retry/backoff handled after loop
                        break
                    else:
                        # Handle redirects explicitly to enforce per-hop egress
                        if allow_redirects and resp.status_code in (301, 302, 303, 307, 308):
                            location = resp.headers.get("location")
                            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                await resp.aclose()
                            if not location:
                                last_exc = NetworkError("Redirect without Location header")
                                break
                            else:
                                try:
                                    next_url = str(resp.request.url.join(httpx.URL(location)))
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    try:
                                        next_url = str(httpx.URL(location))
                                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                        last_exc = NetworkError("Invalid redirect Location header")
                                        break
                                redirects += 1
                                if redirects > DEFAULT_MAX_REDIRECTS:
                                    last_exc = NetworkError("Too many redirects")
                                    break
                                else:
                                    if _is_cross_origin(url, next_url):
                                        _clear_client_cookie_jar(ac)
                                    cur_url = next_url
                                    continue
                        else:
                            # final response
                            if resp.status_code < 400:
                                # metrics for success
                                try:
                                    response_url = str(resp.request.url)
                                    response_observability_url = (
                                        observability_url
                                        if sensitive_observability
                                        else response_url
                                    )
                                    get_metrics_registry().increment(
                                        "http_client_requests_total",
                                        1,
                                        labels=_http_metric_labels(
                                            method,
                                            response_url,
                                            status_code=int(
                                                resp.status_code
                                            ),
                                        ),
                                    )
                                    get_metrics_registry().observe(
                                        "http_client_request_duration_seconds",
                                        time.time() - t0,
                                        labels=_http_metric_labels(
                                            method,
                                            response_url,
                                        ),
                                    )
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    pass
                                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                    tm.set_attributes({"http.status_code": int(resp.status_code)})
                                _log_outbound_request(
                                    method=method,
                                    url=response_observability_url,
                                    status_code=int(resp.status_code),
                                    start_time=t0,
                                    attempt=attempt,
                                    last_retry_delay_s=sleep_s,
                                )
                                return resp
                            # candidate for retry
                            should, rsn = _should_retry(method, resp.status_code, None, retry)
                            if not should or attempt == attempts:
                                _log_outbound_request(
                                    method=method,
                                    url=observability_url,
                                    status_code=int(resp.status_code),
                                    start_time=t0,
                                    attempt=attempt,
                                    last_retry_delay_s=sleep_s,
                                )
                                return resp
                            reason = rsn
                            try:
                                get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": reason})
                            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                pass
                            # Honor Retry-After
                            delay = 0.0
                            if retry.respect_retry_after:
                                delay = _parse_retry_after_delay_seconds(resp.headers.get("retry-after")) or 0.0
                            if delay <= 0:
                                delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                            logger.debug(
                                f"afetch retry attempt={attempt} reason={reason} delay={delay:.3f}s "
                                f"url={_sanitize_url_for_logs(observability_url)}"
                            )
                            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                tm.add_event("http.retry", {"attempt": attempt, "reason": reason})
                            await asyncio.sleep(delay)
                            sleep_s = delay
                            # Restart outer attempt
                            break

                # network error path
                if last_exc is not None:
                    should, rsn = _should_retry(method, None, last_exc, retry)
                    if not should or attempt == attempts:
                        _log_outbound_request(
                            method=method,
                            url=observability_url,
                            status_code=0,
                            start_time=t0,
                            attempt=attempt,
                            last_retry_delay_s=sleep_s,
                            exception_class=last_exc.__class__.__name__,
                        )
                        raise last_exc
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": rsn})
                    delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                    logger.debug(
                        f"afetch network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                        f"url={_sanitize_url_for_logs(observability_url)}"
                    )
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        tm.add_event("http.retry", {"attempt": attempt, "reason": rsn})
                    await asyncio.sleep(delay)
                    sleep_s = delay
                    continue

        # If we exit loop without return, attempts exhausted
        _log_outbound_request(
            method=method,
            url=observability_url,
            status_code=0,
            start_time=t0,
            attempt=attempts,
            last_retry_delay_s=sleep_s,
            exception_class="RetryExhaustedError",
        )
        raise RetryExhaustedError("All retry attempts exhausted")  # noqa: TRY003
    finally:
        if need_close:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                await ac.aclose()


async def _afetch_aiohttp(
    *,
    method: str,
    url: str,
    client: Any | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    allow_redirects: bool = True,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    verify: Any | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
    max_response_bytes: int | None = None,
) -> _AiohttpResponse:
    """Async aiohttp request with retries and egress enforcement.

    ``max_response_bytes`` requests identity encoding and bounds application
    response-body accumulation while streaming successful responses. Encoded
    successful responses are rejected; bounded non-2xx responses are returned
    with an empty body.

    Raises ValueError when retries are enabled and a file-like object in `files`
    is not seekable: "File-like object must be seekable when retries are enabled.
    Either disable retries or use a seekable stream."
    """
    if aiohttp is None:  # pragma: no cover
        raise RuntimeError("aiohttp is not available")  # noqa: TRY003
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    retry = retry or RetryPolicy()
    _validate_retry_files_seekable(files, retry)
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        configured_endpoint=configured_endpoint,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)

    attempts = max(1, retry.attempts)
    sleep_s = 0.0
    t0 = time.time()
    last_exc: Exception | None = None
    tm = get_tracing_manager()
    observability_url = _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else url
    method_upper = str(method).upper()
    _head_get_range_tried = False

    ssl_override = _aiohttp_ssl_from_verify(verify)

    async def _do_once(
        session: aiohttp.ClientSession, target_url: str
    ) -> tuple[_AiohttpResponse | None, str, str | None]:
        req_headers = _inject_trace_headers(headers)
        req_headers = _strip_sensitive_headers_for_cross_origin(
            req_headers, original_url=url, target_url=target_url
        )
        req_cookies = _cookies_for_hop(cookies, original_url=url, target_url=target_url)
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            req_headers = _sanitize_accept_encoding_for_backend(req_headers, "aiohttp")
        try:
            # Optional cert pinning per host
            try:
                pins_map = cert_pinning or _get_client_cert_pins(session)
                if pins_map:
                    if httpx is not None:
                        u = httpx.URL(target_url)
                        host = (u.host or "").lower()
                        scheme = (u.scheme or "").lower()
                        port = int(u.port or (443 if scheme == "https" else 80))
                    else:
                        parsed = urlparse(target_url)
                        host = (parsed.hostname or "").lower()
                        scheme = (parsed.scheme or "").lower()
                        port = parsed.port or (443 if scheme == "https" else 80)
                    pins = _pins_for_host(pins_map, host) if scheme == "https" else None
                    if pins:
                        _check_cert_pinning(
                            host,
                            port,
                            pins,
                            TLS_MIN_VERSION,
                            configured_endpoint=configured_endpoint,
                            accepted_resolved_ips=_accepted_ips_for_url(
                                dns_pin_cache, target_url
                            ),
                            sensitive_observability=sensitive_observability,
                        )
            except EgressPolicyError:
                raise
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                return None, e.__class__.__name__, _network_error_classification(e)
            resp = await _aiohttp_request_io(
                session=session,
                method=method_upper,
                url=target_url,
                headers=req_headers,
                cookies=req_cookies,
                params=params,
                json=json,
                data=data,
                files=files,
                timeout=timeout,
                proxies=proxies,
                ssl_override=ssl_override,
                accepted_resolved_ips=_accepted_ips_for_url(
                    dns_pin_cache,
                    target_url,
                ),
                max_response_bytes=max_response_bytes,
            )
            return resp, "ok", None  # noqa: TRY300
        except EgressPolicyError:
            raise
        except _HTTPCLIENT_REQUEST_EXCEPTIONS as e:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                logger.debug(
                    "afetch _do_once: caught exception {}",
                    type(e).__name__,
                )
            try:
                if _is_dns_resolution_error(e):
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        e._tldw_dns_resolution = True
                    return None, "DNSResolutionError", _network_error_classification(e)
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                pass
            return None, e.__class__.__name__, _network_error_classification(e)

    session = client or _get_aiohttp_session()

    async with _async_http_client_observability_span(
        tm,
        "http.client",
        sensitive_observability=sensitive_observability,
        attributes=_http_trace_attributes(method, url),
    ):
        for attempt in range(1, attempts + 1):
            last_exc = None
            cur_url = url
            redirects = 0

            while True:
                await _avalidate_egress_or_raise(
                    cur_url,
                    dns_pin_cache=dns_pin_cache,
                    configured_endpoint=configured_endpoint,
                    sensitive_observability=sensitive_observability,
                )
                resp, reason, classification = await _do_once(session, cur_url)
                if resp is None:
                    if method_upper == "HEAD" and not _head_get_range_tried:
                        _head_get_range_tried = True
                        try:
                            await _avalidate_egress_or_raise(
                                cur_url,
                                dns_pin_cache=dns_pin_cache,
                                configured_endpoint=configured_endpoint,
                                sensitive_observability=sensitive_observability,
                            )
                            _check_cert_pins_for_url(
                                cur_url,
                                cert_pinning or _get_client_cert_pins(session),
                                configured_endpoint=configured_endpoint,
                                dns_pin_cache=dns_pin_cache,
                                sensitive_observability=sensitive_observability,
                            )
                            req_headers = _inject_trace_headers(headers)
                            req_headers = _strip_sensitive_headers_for_cross_origin(
                                req_headers, original_url=url, target_url=cur_url
                            )
                            req_headers.setdefault("Range", "bytes=0-0")
                            try:
                                _head_fb_to = float(os.getenv("HTTP_HEAD_RANGE_FALLBACK_TIMEOUT", "5"))
                            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                _head_fb_to = 5.0
                            r2_wrap = await _aiohttp_request_io(
                                session=session,
                                method="GET",
                                url=cur_url,
                                headers=req_headers,
                                cookies=_cookies_for_hop(cookies, original_url=url, target_url=cur_url),
                                params=params,
                                timeout=_head_fb_to,
                                proxies=proxies,
                                ssl_override=ssl_override,
                                accepted_resolved_ips=_accepted_ips_for_url(
                                    dns_pin_cache,
                                    cur_url,
                                ),
                                max_response_bytes=max_response_bytes,
                            )
                            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                tm.set_attributes({"http.status_code": int(r2_wrap.status_code)})
                            _log_outbound_request(
                                method="GET",
                                url=observability_url,
                                status_code=int(r2_wrap.status_code),
                                start_time=t0,
                                attempt=attempt,
                                last_retry_delay_s=sleep_s,
                            )
                            return r2_wrap  # noqa: TRY300
                        except EgressPolicyError:
                            raise
                        except _HTTPCLIENT_REQUEST_EXCEPTIONS:
                            pass
                    last_exc = NetworkError(reason, classification=classification)
                    try:
                        if reason == "DNSResolutionError":
                            last_exc._tldw_dns_resolution = True
                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                        pass
                    break

                if allow_redirects and resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        last_exc = NetworkError("Redirect without Location header")
                        break
                    base_url = _get_response_url(resp, cur_url)
                    next_url = _resolve_redirect_url(base_url, location)
                    if not next_url:
                        last_exc = NetworkError("Invalid redirect Location header")
                        break
                    redirects += 1
                    if redirects > DEFAULT_MAX_REDIRECTS:
                        last_exc = NetworkError("Too many redirects")
                        break
                    if _is_cross_origin(url, next_url):
                        _clear_client_cookie_jar(session)
                    cur_url = next_url
                    continue

                # final response
                if resp.status_code < 400:
                    try:
                        response_url = _get_response_url(resp, cur_url)
                        response_observability_url = (
                            observability_url
                            if sensitive_observability
                            else response_url
                        )
                        get_metrics_registry().increment(
                            "http_client_requests_total",
                            1,
                            labels=_http_metric_labels(
                                method,
                                response_url,
                                status_code=int(
                                    resp.status_code
                                ),
                            ),
                        )
                        get_metrics_registry().observe(
                            "http_client_request_duration_seconds",
                            time.time() - t0,
                            labels=_http_metric_labels(
                                method,
                                response_url,
                            ),
                        )
                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                        pass
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        tm.set_attributes({"http.status_code": int(resp.status_code)})
                    _log_outbound_request(
                        method=method,
                        url=response_observability_url,
                        status_code=int(resp.status_code),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                    )
                    return resp

                should, rsn = _should_retry(method, resp.status_code, None, retry)
                if not should or attempt == attempts:
                    _log_outbound_request(
                        method=method,
                        url=observability_url,
                        status_code=int(resp.status_code),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                    )
                    return resp
                reason = rsn
                try:
                    get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": reason})
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                    pass
                delay = 0.0
                if retry.respect_retry_after:
                    delay = _parse_retry_after_delay_seconds(resp.headers.get("retry-after")) or 0.0
                if delay <= 0:
                    delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                logger.debug(
                    f"afetch retry attempt={attempt} reason={reason} delay={delay:.3f}s "
                    f"url={_sanitize_url_for_logs(observability_url)}"
                )
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    tm.add_event("http.retry", {"attempt": attempt, "reason": reason})
                await asyncio.sleep(delay)
                sleep_s = delay
                break

            if last_exc is not None:
                should, rsn = _should_retry(method, None, last_exc, retry)
                if not should or attempt == attempts:
                    _log_outbound_request(
                        method=method,
                        url=observability_url,
                        status_code=0,
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=last_exc.__class__.__name__,
                    )
                    raise last_exc
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": rsn})
                delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                logger.debug(
                    f"afetch network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                    f"url={_sanitize_url_for_logs(observability_url)}"
                )
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    tm.add_event("http.retry", {"attempt": attempt, "reason": rsn})
                await asyncio.sleep(delay)
                sleep_s = delay
                continue

    _log_outbound_request(
        method=method,
        url=observability_url,
        status_code=0,
        start_time=t0,
        attempt=attempts,
        last_retry_delay_s=sleep_s,
        exception_class="RetryExhaustedError",
    )
    raise RetryExhaustedError("All retry attempts exhausted")  # noqa: TRY003


def resolve_afetch_transport(
    transport: str | None = None,
) -> str:
    """Resolve and validate the async transport selected by ``afetch``."""
    selected = (
        transport
        if transport is not None
        else ("aiohttp" if aiohttp is not None else "httpx")
    )
    normalized = str(selected).strip().lower()
    if normalized == "aiohttp":
        if aiohttp is None:
            raise RuntimeError("aiohttp is not available")  # noqa: TRY003
        return normalized
    if normalized == "httpx":
        if _resolve_httpx() is None:
            raise RuntimeError("httpx is not available")  # noqa: TRY003
        return normalized
    raise ValueError("Unsupported async HTTP transport")


@_with_sensitive_http_log_context_async
async def afetch(
    *,
    method: str,
    url: str,
    client: Any | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    allow_redirects: bool = True,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    verify: bool | str | ssl.SSLContext | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
    transport: str | None = None,
    max_response_bytes: int | None = None,
) -> Any:
    """Issue one async request, optionally bounding the decoded success body."""
    if client is not None:
        adapter_name = "aiohttp" if _is_aiohttp_client(client) else "httpx"
        if (
            transport is not None
            and resolve_afetch_transport(transport) != adapter_name
        ):
            raise ValueError(
                "Explicit async HTTP transport does not match client"
            )
    else:
        adapter_name = resolve_afetch_transport(transport)
    adapter = _get_transport_adapter(adapter_name)
    return await adapter.arequest(
        method=method,
        url=url,
        client=client,
        headers=headers,
        cookies=cookies,
        params=params,
        json=json,
        data=data,
        files=files,
        timeout=timeout,
        allow_redirects=allow_redirects,
        proxies=proxies,
        retry=retry,
        cert_pinning=cert_pinning,
        verify=verify,
        configured_endpoint=configured_endpoint,
        sensitive_observability=sensitive_observability,
        max_response_bytes=max_response_bytes,
    )


@_with_sensitive_http_log_context_async
async def apost(
    *,
    url: str,
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    proxies: str | dict[str, str] | None = None,
    sensitive_observability: bool = False,
) -> httpx.Response:
    """
    Minimal async POST helper that enforces egress policy.

    This is intentionally lightweight (no retries/redirect handling) so that
    adapters and unit tests can patch `apost` (or the async client factory)
    to intercept calls while centralizing the egress check.
    """
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)

    need_close = False
    ac = client
    if ac is None:
        ac = create_async_client(proxies=proxies, timeout=timeout)
        need_close = True

    try:
        resp = await _httpx_arequest_io(
            client=ac,
            method="POST",
            url=url,
            headers=headers,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
            follow_redirects=False,
            accepted_resolved_ips=_accepted_ips_for_url(dns_pin_cache, url),
        )
        return resp
    finally:
        if need_close:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                await ac.aclose()


@_with_sensitive_http_log_context
def _fetch_httpx_response(
    *,
    method: str,
    url: str,
    client: httpx.Client | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    allow_redirects: bool = True,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
) -> httpx.Response:
    """Sync httpx request with retries and egress enforcement.

    Raises ValueError when retries are enabled and a file-like object in `files`
    is not seekable: "File-like object must be seekable when retries are enabled.
    Either disable retries or use a seekable stream."
    """
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    retry = retry or RetryPolicy()
    _validate_retry_files_seekable(files, retry)
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    _validate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        configured_endpoint=configured_endpoint,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)

    attempts = max(1, retry.attempts)
    sleep_s = 0.0
    t0 = time.time()
    tm = get_tracing_manager()
    observability_url = _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else url
    method_upper = str(method).upper()
    _head_disable_h2_tried = False
    _head_get_range_tried = False

    def _do_once(sc: httpx.Client, target_url: str) -> tuple[httpx.Response | None, str]:
        req_headers = _inject_trace_headers(headers)
        req_headers = _strip_sensitive_headers_for_cross_origin(
            req_headers, original_url=url, target_url=target_url
        )
        req_cookies = _cookies_for_hop(cookies, original_url=url, target_url=target_url)
        # Parity with simple fetch: drop 'zstd' from Accept-Encoding for httpx
        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
            req_headers = _sanitize_accept_encoding_for_backend(req_headers, "httpx")
        try:
            # Optional cert pinning per host
            try:
                pins_map = cert_pinning or _get_client_cert_pins(sc)
                if pins_map:
                    u = httpx.URL(target_url)
                    if u.scheme.lower() == "https":
                        host = (u.host or "").lower()
                        pins = _pins_for_host(pins_map, host)
                        if pins:
                            _check_cert_pinning(
                                host,
                                int(u.port or 443),
                                pins,
                                TLS_MIN_VERSION,
                                configured_endpoint=configured_endpoint,
                                accepted_resolved_ips=_accepted_ips_for_url(
                                    dns_pin_cache, target_url
                                ),
                                sensitive_observability=sensitive_observability,
                            )
            except EgressPolicyError:
                raise
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                return None, e.__class__.__name__
            r = _httpx_request_io(
                client=sc,
                method=method_upper,
                url=target_url,
                headers=req_headers,
                cookies=req_cookies,
                params=params,
                json=json,
                data=data,
                files=files,
                timeout=timeout,
                follow_redirects=False,
                accepted_resolved_ips=_accepted_ips_for_url(
                    dns_pin_cache,
                    target_url,
                ),
            )
            return r, "ok"  # noqa: TRY300
        except EgressPolicyError:
            raise
        except _HTTPCLIENT_REQUEST_EXCEPTIONS as e:
            # Classify DNS resolution errors explicitly so that retry logic
            # can treat them as permanent failures.
            try:
                if _is_dns_resolution_error(e):
                    return None, "DNSResolutionError"
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                pass
            return None, e.__class__.__name__

    need_close = False
    sc = client
    if sc is None:
        sc = _get_httpx_client(proxies=proxies)
        need_close = False

    try:
        with _http_client_observability_span(
            tm,
            "http.client",
            sensitive_observability=sensitive_observability,
            attributes=_http_trace_attributes(method, url),
        ):
            for attempt in range(1, attempts + 1):
                cur_url = url
                redirects = 0
                while True:
                    _validate_egress_or_raise(
                        cur_url,
                        dns_pin_cache=dns_pin_cache,
                        configured_endpoint=configured_endpoint,
                        sensitive_observability=sensitive_observability,
                    )
                    resp, reason = _do_once(sc, cur_url)
                    if resp is None:
                        if method_upper == "HEAD":
                            if not _head_disable_h2_tried:
                                _head_disable_h2_tried = True
                                try:
                                    if need_close:
                                        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                            sc.close()
                                    sc = create_client(proxies=proxies, http2=False)
                                    need_close = True
                                    # Retry immediately within same attempt
                                    continue
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    pass
                            if not _head_get_range_tried:
                                _head_get_range_tried = True
                                try:
                                    _validate_egress_or_raise(
                                        cur_url,
                                        dns_pin_cache=dns_pin_cache,
                                        configured_endpoint=configured_endpoint,
                                        sensitive_observability=sensitive_observability,
                                    )
                                    _check_cert_pins_for_url(
                                        cur_url,
                                        cert_pinning or _get_client_cert_pins(sc),
                                        configured_endpoint=configured_endpoint,
                                        dns_pin_cache=dns_pin_cache,
                                        sensitive_observability=sensitive_observability,
                                    )
                                    req_headers = _inject_trace_headers(headers)
                                    req_headers = _strip_sensitive_headers_for_cross_origin(
                                        req_headers, original_url=url, target_url=cur_url
                                    )
                                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                        req_headers = _sanitize_accept_encoding_for_backend(req_headers, "httpx")
                                    req_headers.setdefault("Range", "bytes=0-0")
                                    # Use a small per-request timeout specifically for this fallback
                                    try:
                                        _head_fb_to = float(os.getenv("HTTP_HEAD_RANGE_FALLBACK_TIMEOUT", "5"))
                                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                        _head_fb_to = 5.0
                                    r2 = _httpx_request_io(
                                        client=sc,
                                        method="GET",
                                        url=cur_url,
                                        headers=req_headers,
                                        cookies=_cookies_for_hop(cookies, original_url=url, target_url=cur_url),
                                        params=params,
                                        json=json,
                                        data=data,
                                        files=files,
                                        timeout=_head_fb_to,
                                        follow_redirects=False,
                                        accepted_resolved_ips=_accepted_ips_for_url(
                                            dns_pin_cache,
                                            cur_url,
                                        ),
                                    )
                                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                        tm.set_attributes({"http.status_code": int(r2.status_code)})
                                    _log_outbound_request(
                                        method="GET",
                                        url=observability_url,
                                        status_code=int(r2.status_code),
                                        start_time=t0,
                                        attempt=attempt,
                                        last_retry_delay_s=sleep_s,
                                    )
                                    return r2  # noqa: TRY300
                                except EgressPolicyError:
                                    raise
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    pass
                        should, rsn = _should_retry(method, None, NetworkError(reason), retry)
                        if not should or attempt == attempts:
                            raise NetworkError(reason)
                        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                            get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": rsn})
                        delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                        logger.debug(
                            f"fetch network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                            f"url={_sanitize_url_for_logs(observability_url)}"
                        )
                        time.sleep(delay)
                        sleep_s = delay
                        break
                    # redirect handling
                    if allow_redirects and resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("location")
                        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                            resp.close()
                        if not location:
                            if attempt == attempts:
                                raise NetworkError("Redirect without Location header")  # noqa: TRY003
                            delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                            time.sleep(delay)
                            sleep_s = delay
                            break
                        try:
                            next_url = str(resp.request.url.join(httpx.URL(location)))
                        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                            try:
                                next_url = str(httpx.URL(location))
                            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                raise NetworkError("Invalid redirect Location header")  # noqa: B904, TRY003
                        redirects += 1
                        if redirects > DEFAULT_MAX_REDIRECTS:
                            raise NetworkError("Too many redirects")  # noqa: TRY003
                        if _is_cross_origin(url, next_url):
                            _clear_client_cookie_jar(sc)
                        cur_url = next_url
                        continue
                    if resp.status_code < 400:
                        try:
                            response_url = str(resp.request.url)
                            response_observability_url = (
                                observability_url
                                if sensitive_observability
                                else response_url
                            )
                            get_metrics_registry().increment(
                                "http_client_requests_total",
                                1,
                                labels=_http_metric_labels(
                                    method,
                                    response_url,
                                    status_code=int(
                                        resp.status_code
                                    ),
                                ),
                            )
                            get_metrics_registry().observe(
                                "http_client_request_duration_seconds",
                                time.time() - t0,
                                labels=_http_metric_labels(
                                    method,
                                    response_url,
                                ),
                            )
                        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                            pass
                        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                            tm.set_attributes({"http.status_code": int(resp.status_code)})
                        _log_outbound_request(
                            method=method,
                            url=response_observability_url,
                            status_code=int(resp.status_code),
                            start_time=t0,
                            attempt=attempt,
                            last_retry_delay_s=sleep_s,
                        )
                        return resp
                    should, rsn = _should_retry(method, resp.status_code, None, retry)
                    if not should or attempt == attempts:
                        _log_outbound_request(
                            method=method,
                            url=observability_url,
                            status_code=int(resp.status_code),
                            start_time=t0,
                            attempt=attempt,
                            last_retry_delay_s=sleep_s,
                        )
                        return resp
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": rsn})
                    delay = 0.0
                    if retry.respect_retry_after:
                        delay = _parse_retry_after_delay_seconds(resp.headers.get("retry-after")) or 0.0
                    if delay <= 0:
                        delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                    logger.debug(
                        f"fetch retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                        f"url={_sanitize_url_for_logs(observability_url)}"
                    )
                    with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                        tm.add_event("http.retry", {"attempt": attempt, "reason": rsn})
                    time.sleep(delay)
                    sleep_s = delay
                    break
        _log_outbound_request(
            method=method,
            url=observability_url,
            status_code=0,
            start_time=t0,
            attempt=attempts,
            last_retry_delay_s=sleep_s,
            exception_class="RetryExhaustedError",
        )
        raise RetryExhaustedError("All retry attempts exhausted")  # noqa: TRY003
    finally:
        if need_close:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                sc.close()


@contextmanager
def stream_response(
    *,
    method: str,
    url: str,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> Iterator[httpx.Response]:
    """Open one checked synchronous response stream without following redirects."""
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003

    options = dict(kwargs)
    proxies = options.pop("proxies", None)
    verify = options.pop("verify", None)
    cert_pinning = options.pop("cert_pinning", None)
    trust_env = options.pop("trust_env", DEFAULT_TRUST_ENV)
    options.pop("allow_redirects", None)
    options["follow_redirects"] = False
    options["headers"] = _sanitize_accept_encoding_for_backend(
        _inject_trace_headers(options.get("headers")),
        "httpx",
    )

    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    _validate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        configured_endpoint=configured_endpoint,
    )
    _validate_proxies_or_raise(proxies)

    owns_client = client is None
    sc = client
    if sc is None:
        sc = create_client(
            timeout=options.get("timeout"),
            proxies=proxies,
            trust_env=is_truthy(trust_env),
            cert_pinning=cert_pinning,
            verify=verify,
        )
    try:
        pins_map = cert_pinning or _get_client_cert_pins(sc)
        parsed = httpx.URL(url)
        host = parsed.host or ""
        pins = _pins_for_host(pins_map, host)
        if parsed.scheme.lower() == "https" and pins:
            _check_cert_pinning(
                host,
                int(parsed.port or 443),
                pins,
                TLS_MIN_VERSION,
                configured_endpoint=configured_endpoint,
                accepted_resolved_ips=_accepted_ips_for_url(dns_pin_cache, url),
            )
        transport_url, transport_headers, sni_hostname = _prepare_pinned_transport_target(
            url,
            options.get("headers"),
            _accepted_ips_for_url(dns_pin_cache, url),
        )
        options["headers"] = transport_headers
        if sni_hostname:
            options["extensions"] = {"sni_hostname": sni_hostname}
        with sc.stream(str(method).upper(), transport_url, **options) as response:
            if transport_url != url:
                _restore_httpx_response_url(response, url)
            yield response
    finally:
        if owns_client:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                sc.close()


def _fetch_curl_simple(
    *,
    url: str,
    headers: dict[str, str],
    cookies: dict[str, str] | None,
    timeout: float | None,
    impersonate: str | None,
    proxies: dict[str, str] | None,
    session: Any | None = None,
) -> HttpResponse:
    CurlSession = None
    if session is None:
        CurlSession = _resolve_curl_session()
        if CurlSession is None:
            raise RuntimeError("curl_cffi is not installed")  # noqa: TRY003
    if proxies:
        _validate_proxies_or_raise(proxies)

    req_kwargs: dict[str, Any] = {
        "headers": headers,
        "cookies": cookies,
        "allow_redirects": False,
    }
    if timeout is not None:
        req_kwargs["timeout"] = timeout
    if proxies:
        req_kwargs["proxies"] = proxies

    if session is not None:
        resp = session.get(url, **req_kwargs)
        return HttpResponse(
            status=int(getattr(resp, "status_code", 0)),
            headers=dict(getattr(resp, "headers", {}) or {}),
            text=str(getattr(resp, "text", "")),
            url=str(getattr(resp, "url", url)),
            backend="curl",
        )

    with CurlSession(impersonate=impersonate) as session:
        resp = session.get(url, **req_kwargs)
        return HttpResponse(
            status=int(getattr(resp, "status_code", 0)),
            headers=dict(getattr(resp, "headers", {}) or {}),
            text=str(getattr(resp, "text", "")),
            url=str(getattr(resp, "url", url)),
            backend="curl",
        )


def fetch(*args, **kwargs):
    """Dual-mode fetch helper.

    - If called with keyword 'method', delegates to the HTTPX response API and
      returns an httpx.Response (backward compatible with existing callers).
      Set `sensitive_observability=True` when the URL is request-derived and
      must be redacted from logs, metrics, and tracing attributes.
      When retries are enabled and `files` are provided, file-like objects must
      be seekable; otherwise a ValueError is raised with the message:
      "File-like object must be seekable when retries are enabled. Either disable
      retries or use a seekable stream."
    - Otherwise, provides a simplified fetch suitable for Web_Scraping tests:
      accepts `url` (positional), optional `headers`, `backend` (default 'httpx'),
      and returns a mapping with keys: status, headers, text, url, backend.
    """
    # Response-based API path (existing callers/tests)
    if "method" in kwargs:
        client = kwargs.get("client")
        adapter_name = "aiohttp" if (client is not None and _is_aiohttp_client(client)) else "httpx"
        adapter = _get_transport_adapter(adapter_name)
        return adapter.request(**kwargs)
    # Simple path (tests: Web_Scraping/test_http_client_fetch.py)
    if not args and "url" not in kwargs:
        raise TypeError("fetch() missing required argument: 'url'")  # noqa: TRY003
    url = str(args[0] if args else kwargs.get("url"))
    backend = str(kwargs.get("backend", "httpx"))
    headers = kwargs.get("headers") or {}
    cookies = kwargs.get("cookies")
    impersonate = kwargs.get("impersonate")
    follow_redirects_cfg = kwargs.get("follow_redirects")
    trust_env = kwargs.get("trust_env")
    proxies = kwargs.get("proxies")
    timeout = kwargs.get("timeout")
    max_response_bytes = kwargs.get("max_response_bytes")

    if max_response_bytes is not None:
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")

    # Enforce egress via stubbed policy helper (tests monkeypatch this).
    # This remains intentionally lightweight so tests can override without
    # triggering full DNS lookups in the central policy during unit runs.
    if not _is_url_allowed(url):
        raise ValueError("Egress denied for URL")  # noqa: TRY003

    # Validate proxies against allowlist even in simple mode
    _validate_proxies_or_raise(proxies)

    backend_norm = str(backend).lower().strip()
    if backend_norm == "auto":
        backend_norm = "curl" if _resolve_curl_session() is not None else "httpx"
    _hx = _resolve_httpx()
    if _hx is None and backend_norm != "curl":  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003

    # Sanitize Accept-Encoding as per backend expectations
    req_headers = _sanitize_accept_encoding_for_backend(headers, backend_norm)
    if max_response_bytes is not None:
        req_headers = _force_identity_accept_encoding(req_headers)

    # Determine redirect behavior, honoring env/Config_Files when caller did not
    # explicitly supply follow_redirects.
    if follow_redirects_cfg is None:
        env_allow_redirects = os.getenv("HTTP_ALLOW_REDIRECTS")
        if env_allow_redirects is not None:
            follow_redirects = is_truthy(str(env_allow_redirects).strip())
        else:
            follow_redirects = True
    else:
        follow_redirects = bool(follow_redirects_cfg)

    allow_cross_host = is_truthy(os.getenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", ""))
    allow_downgrade = is_truthy(os.getenv("HTTP_ALLOW_SCHEME_DOWNGRADE", ""))

    def _extract_redirect_host(redirect_url: str) -> str:
        try:
            parsed = urlparse(redirect_url)
            return (parsed.hostname or "").lower()
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            return ""

    def _extract_redirect_port(redirect_url: str) -> int | None:
        try:
            parsed = urlparse(redirect_url)
            if parsed.port is not None:
                return int(parsed.port)
            scheme = (parsed.scheme or "").lower()
            if scheme == "https":
                return 443
            if scheme == "http":
                return 80
            return None
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            return None

    def _extract_redirect_scheme(redirect_url: str) -> str:
        try:
            parsed = urlparse(redirect_url)
            return (parsed.scheme or "").lower()
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            return ""

    def _is_cross_host_redirect(prev: str, nxt: str) -> bool:
        prev_host = _extract_redirect_host(prev)
        next_host = _extract_redirect_host(nxt)
        prev_port = _extract_redirect_port(prev)
        next_port = _extract_redirect_port(nxt)
        return bool(
            prev_host
            and next_host
            and ((prev_host, prev_port) != (next_host, next_port))
        )

    def _is_scheme_downgrade(prev: str, nxt: str) -> bool:
        return (
            _extract_redirect_scheme(prev) == "https"
            and _extract_redirect_scheme(nxt) == "http"
        )

    def _redirect_boundary_headers(current_headers: dict[str, str]) -> dict[str, str]:
        # Preserve only safe transport headers across origin boundaries.
        safe_headers: dict[str, str] = {}
        for header_key, header_value in (current_headers or {}).items():
            key_lc = str(header_key).lower()
            if key_lc == "user-agent":
                safe_headers["User-Agent"] = str(header_value)
            elif key_lc == "accept-encoding":
                safe_headers["Accept-Encoding"] = str(header_value)
        return safe_headers

    def _clear_session_cookie_state(session: Any) -> None:
        for attr_name in ("cookies", "cookie_jar", "jar"):
            store = getattr(session, attr_name, None)
            if store is None:
                continue
            clear = getattr(store, "clear", None)
            if callable(clear):
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    clear()
                continue
            if isinstance(store, dict):
                store.clear()

    def _redirect_allowed(prev: str, nxt: str) -> bool:
        try:
            prev_scheme = _extract_redirect_scheme(prev)
            next_scheme = _extract_redirect_scheme(nxt)
            prev_host = _extract_redirect_host(prev)
            next_host = _extract_redirect_host(nxt)
            prev_port = _extract_redirect_port(prev)
            next_port = _extract_redirect_port(nxt)
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
            return False
        # Disallow scheme downgrade unless explicitly allowed
        if not allow_downgrade and prev_scheme == "https" and next_scheme == "http":
            return False
        same_host = bool(prev_host and next_host and prev_host == next_host)
        # Same-host scheme changes are governed separately by allow_downgrade.
        if same_host and prev_scheme != next_scheme:
            return True
        # Same-host/same-port redirects are always allowed (subject to egress checks)
        if (prev_host, prev_port) == (next_host, next_port):
            return True
        # Cross-host redirects configurable (default disabled)
        return bool(allow_cross_host)

    client_kwargs: dict[str, Any] = {}
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if trust_env is not None:
        client_kwargs["trust_env"] = trust_env
    if proxies is not None:
        client_kwargs["proxies"] = proxies

    if backend_norm == "curl":
        CurlSession = _resolve_curl_session()
        if CurlSession is None:
            raise RuntimeError("curl_cffi is not installed")  # noqa: TRY003

        cur_url = url
        hop_headers = req_headers
        hop_cookies = cookies
        redirects = 0
        with CurlSession(impersonate=impersonate) as curl_session:
            while True:
                if not _is_url_allowed(cur_url):
                    raise ValueError("Egress denied for URL")  # noqa: TRY003

                if max_response_bytes is None:
                    resp = _fetch_curl_simple(
                        url=cur_url,
                        headers=hop_headers,
                        cookies=hop_cookies,
                        timeout=timeout,
                        impersonate=impersonate,
                        proxies=proxies,
                        session=curl_session,
                    )
                else:
                    request_stream = getattr(curl_session, "get", None)
                    if not callable(request_stream):
                        raise RuntimeError("Selected backend does not support bounded response streaming")
                    body_collector = _BoundedBodyCollector(max_response_bytes)

                    req_kwargs: dict[str, Any] = {
                        "headers": hop_headers,
                        "cookies": hop_cookies,
                        "allow_redirects": False,
                        "accept_encoding": None,
                        "content_callback": body_collector,
                    }
                    if timeout is not None:
                        req_kwargs["timeout"] = timeout
                    if proxies:
                        req_kwargs["proxies"] = proxies
                    streamed = request_stream(cur_url, **req_kwargs)
                    try:
                        response_headers = dict(getattr(streamed, "headers", {}) or {})
                        status_code = int(getattr(streamed, "status_code", 0))
                        response_url = str(getattr(streamed, "url", cur_url))
                        is_followed_redirect = follow_redirects and status_code in (301, 302, 303, 307, 308)
                        if is_followed_redirect:
                            body = b""
                        else:
                            if _uses_compressed_content_encoding(response_headers):
                                raise ValueError("Compressed responses are not allowed with " "max_response_bytes")
                            if body_collector.overflow:
                                raise ValueError("Response exceeds max_response_bytes limit")
                            body = body_collector.body
                        resp = HttpResponse(
                            status=status_code,
                            headers=response_headers,
                            text=_decode_bounded_response_body(streamed, body),
                            url=response_url,
                            backend="curl",
                        )
                    finally:
                        _close_simple_response(streamed)
                status = int(resp["status"])

                if not follow_redirects or status not in (301, 302, 303, 307, 308):
                    return resp

                location = (resp["headers"] or {}).get("location") or (resp["headers"] or {}).get("Location")
                if not location:
                    return resp

                base_url = str(resp["url"] or cur_url)
                next_url = _resolve_redirect_url(base_url, str(location))
                if not next_url:
                    return resp

                if not _redirect_allowed(cur_url, next_url):
                    return resp

                if _is_cross_host_redirect(cur_url, next_url) or _is_scheme_downgrade(cur_url, next_url):
                    hop_headers = _redirect_boundary_headers(hop_headers)
                    hop_cookies = None
                    _clear_session_cookie_state(curl_session)

                redirects += 1
                if redirects > DEFAULT_MAX_REDIRECTS:
                    return resp

                cur_url = next_url

    if max_response_bytes is not None:
        client_cls = getattr(_hx, "Client", object) if _hx is not None else object
        sc = _instantiate_client(client_cls, client_kwargs)
        with sc as sc:
            cur_url = url
            hop_headers = req_headers
            hop_cookies = cookies
            redirects = 0

            while True:
                if not _is_url_allowed(cur_url):
                    raise ValueError("Egress denied for URL")  # noqa: TRY003

                request_stream = getattr(sc, "stream", None)
                if not callable(request_stream):
                    raise RuntimeError("Selected backend does not support bounded response streaming")
                with request_stream(
                    "GET",
                    cur_url,
                    headers=hop_headers,
                    cookies=hop_cookies,
                    follow_redirects=False,
                ) as streamed:
                    status = int(getattr(streamed, "status_code", 0))
                    response_headers = dict(getattr(streamed, "headers", {}) or {})
                    response_url = str(getattr(streamed, "url", cur_url))

                    if not follow_redirects or status not in (301, 302, 303, 307, 308):
                        if _uses_compressed_content_encoding(response_headers):
                            raise ValueError("Compressed responses are not allowed with " "max_response_bytes")
                        iter_raw = getattr(streamed, "iter_raw", None)
                        if not callable(iter_raw):
                            raise RuntimeError("Selected backend does not support bounded response streaming")
                        body = _read_bounded_chunks(iter_raw(), max_response_bytes)
                        return HttpResponse(
                            status=status,
                            headers=response_headers,
                            text=_decode_bounded_response_body(streamed, body),
                            url=response_url,
                            backend="httpx",
                        )

                location = response_headers.get("location") or response_headers.get("Location")
                if not location:
                    return HttpResponse(
                        status=status,
                        headers=response_headers,
                        text="",
                        url=response_url,
                        backend="httpx",
                    )

                next_url = _resolve_redirect_url(response_url, str(location))
                if not next_url or not _redirect_allowed(cur_url, next_url):
                    return HttpResponse(
                        status=status,
                        headers=response_headers,
                        text="",
                        url=response_url,
                        backend="httpx",
                    )

                if _is_cross_host_redirect(cur_url, next_url) or _is_scheme_downgrade(cur_url, next_url):
                    hop_headers = _redirect_boundary_headers(hop_headers)
                    hop_cookies = None
                    _clear_session_cookie_state(sc)

                redirects += 1
                if redirects > DEFAULT_MAX_REDIRECTS:
                    return HttpResponse(
                        status=status,
                        headers=response_headers,
                        text="",
                        url=response_url,
                        backend="httpx",
                    )

                cur_url = next_url

    # Minimal client lifecycle for unbounded simple fetch with explicit redirect handling
    client_cls = getattr(_hx, "Client", object) if _hx is not None else object
    sc = _instantiate_client(client_cls, client_kwargs)
    with sc as sc:
        cur_url = url
        hop_headers = req_headers
        hop_cookies = cookies
        redirects = 0

        while True:
            # Re-enforce lightweight egress guard on each hop
            if not _is_url_allowed(cur_url):
                raise ValueError("Egress denied for URL")  # noqa: TRY003

            r = sc.request("GET", cur_url, headers=hop_headers, cookies=hop_cookies, follow_redirects=False)
            status = int(getattr(r, "status_code", 0))

            if not follow_redirects or status not in (301, 302, 303, 307, 308):
                break

            location = getattr(r, "headers", {}) or {}
            location = location.get("location") or location.get("Location")
            if not location:
                break

            base_url = str(getattr(r, "url", cur_url))
            next_url = _resolve_redirect_url(base_url, str(location))
            if not next_url:
                break

            if not _redirect_allowed(cur_url, next_url):
                break

            if _is_cross_host_redirect(cur_url, next_url) or _is_scheme_downgrade(cur_url, next_url):
                hop_headers = _redirect_boundary_headers(hop_headers)
                hop_cookies = None
                _clear_session_cookie_state(sc)

            redirects += 1
            if redirects > DEFAULT_MAX_REDIRECTS:
                break

            cur_url = next_url

        return HttpResponse(
            status=status,
            headers=dict(getattr(r, "headers", {}) or {}),
            text=str(getattr(r, "text", "")),
            url=str(getattr(r, "url", cur_url)),
            backend=str(backend_norm),
        )


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------

class _BoundedJSONRedirect(Exception):
    def __init__(self, location: str) -> None:
        self.location = location


def _redirect_crosses_authority(previous_url: str, next_url: str) -> bool:
    previous = urlparse(previous_url)
    following = urlparse(next_url)

    def _origin(parsed: Any) -> tuple[str, str, int | None]:
        scheme = (parsed.scheme or "").lower()
        port = parsed.port
        if port is None:
            port = 443 if scheme == "https" else 80 if scheme == "http" else None
        return scheme, (parsed.hostname or "").lower(), port

    previous_origin = _origin(previous)
    following_origin = _origin(following)
    return previous_origin != following_origin


def _redirect_boundary_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Retain only non-authority transport headers across redirect origins."""
    safe_headers: dict[str, str] = {}
    for key, value in (headers or {}).items():
        normalized = str(key).casefold()
        if normalized == "user-agent":
            safe_headers["User-Agent"] = str(value)
        elif normalized == "accept-encoding":
            safe_headers["Accept-Encoding"] = str(value)
    return safe_headers


async def _afetch_json_bounded(
    *,
    method: str,
    url: str,
    client: Any | None,
    require_json_ct: bool,
    max_bytes: int,
    on_response: ResponseHeadersCallback | None,
    **kwargs: Any,
) -> Any:
    allow_redirects = bool(kwargs.pop("allow_redirects", True))
    hop_headers = dict(kwargs.pop("headers", None) or {})
    hop_cookies = kwargs.pop("cookies", None)
    current_url = url
    redirects = 0

    while True:
        async def _validate_response_headers(
            response_status: int,
            response_headers: Mapping[str, str],
        ) -> None:
            location = response_headers.get("location")
            if allow_redirects and response_status in {301, 302, 303, 307, 308}:
                if not location:
                    raise NetworkError("Redirect without Location header")  # noqa: TRY003
                raise _BoundedJSONRedirect(str(location))
            await _invoke_response_callback(
                on_response,
                response_status,
                response_headers,
            )
            content_type = response_headers.get("content-type", "").lower()
            if require_json_ct and "application/json" not in content_type:
                raise JSONDecodeError("Response is not application/json")  # noqa: TRY003
            _raise_if_json_response_exceeds_limit(
                headers=response_headers,
                content=b"",
                max_bytes=max_bytes,
            )

        body = bytearray()
        stream = astream_bytes(
            method=method,
            url=current_url,
            client=client,
            headers=hop_headers,
            cookies=hop_cookies,
            chunk_size=max(1, min(65536, max_bytes + 1)),
            on_response=_validate_response_headers,
            raise_for_status=False,
            **kwargs,
        )
        try:
            async for chunk in stream:
                if len(body) + len(chunk) > max_bytes:
                    raise JSONDecodeError("Response exceeds max_bytes limit")  # noqa: TRY003
                body.extend(chunk)
        except _BoundedJSONRedirect as redirect:
            next_url = _resolve_redirect_url(current_url, redirect.location)
            if not next_url:
                raise NetworkError("Invalid redirect Location header") from None  # noqa: TRY003
            redirects += 1
            if redirects > DEFAULT_MAX_REDIRECTS:
                raise NetworkError("Too many redirects") from None  # noqa: TRY003
            if _redirect_crosses_authority(current_url, next_url):
                hop_headers = _redirect_boundary_headers(hop_headers)
                hop_cookies = None
            current_url = next_url
            continue
        finally:
            await stream.aclose()

        try:
            return json.loads(body)
        except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as exc:
            raise JSONDecodeError(str(exc)) from exc


async def afetch_json(
    *,
    method: str,
    url: str,
    client: httpx.AsyncClient | None = None,
    require_json_ct: bool = True,
    max_bytes: int | None = None,
    on_response: ResponseHeadersCallback | None = None,
    **kwargs: Any,
) -> Any:
    if max_bytes is not None:
        return await _afetch_json_bounded(
            method=method,
            url=url,
            client=client,
            require_json_ct=require_json_ct,
            max_bytes=max_bytes,
            on_response=on_response,
            **kwargs,
        )

    r = await afetch(method=method, url=url, client=client, **kwargs)
    if on_response is not None:
        try:
            await _invoke_response_callback(
                on_response,
                int(getattr(r, "status_code", getattr(r, "status", 0)) or 0),
                getattr(r, "headers", {}) or {},
            )
        except asyncio.CancelledError:
            await r.aclose()
            raise
        except Exception:
            await r.aclose()
            raise
    ctype = r.headers.get("content-type", "").lower()
    if require_json_ct and "application/json" not in ctype:
        await r.aclose()
        raise JSONDecodeError("Response is not application/json")  # noqa: TRY003
    try:
        data = r.json()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
        await r.aclose()
        raise JSONDecodeError(str(e))  # noqa: B904
    return data


def fetch_json(
    *,
    method: str,
    url: str,
    client: httpx.Client | None = None,
    require_json_ct: bool = True,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> Any:
    r = fetch(method=method, url=url, client=client, **kwargs)
    ctype = r.headers.get("content-type", "").lower()
    if require_json_ct and "application/json" not in ctype:
        r.close()
        raise JSONDecodeError("Response is not application/json")  # noqa: TRY003
    if max_bytes is not None:
        try:
            _raise_if_json_response_exceeds_limit(
                headers=r.headers,
                content=r.content,
                max_bytes=max_bytes,
            )
        except JSONDecodeError:
            r.close()
            raise
    try:
        data = r.json()
    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
        r.close()
        raise JSONDecodeError(str(e))  # noqa: B904
    return data


# --------------------------------------------------------------------------------------
# Streaming helpers
# --------------------------------------------------------------------------------------

class _CaseInsensitiveHeaders(Mapping[str, str]):
    """Read-only response headers with case-insensitive lookup.

    Repeated fields are comma-joined in wire order for transport parity and to
    keep ambiguous singleton headers, such as Content-Type, visible to callers.
    """

    def __init__(self, headers: Mapping[str, Any]) -> None:
        grouped: dict[str, list[str]] = {}
        for key, value in _iter_response_header_items(headers):
            grouped.setdefault(str(key).lower(), []).append(str(value))
        self._headers = {key: ", ".join(values) for key, values in grouped.items()}

    def __getitem__(self, key: str) -> str:
        return self._headers[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._headers)

    def __len__(self) -> int:
        return len(self._headers)


def _iter_response_header_items(headers: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield repeated header fields without transport-specific value loss."""
    multi_items = getattr(headers, "multi_items", None)
    if callable(multi_items):
        yield from multi_items()
        return

    getall = getattr(headers, "getall", None)
    get_list = getattr(headers, "get_list", None)
    repeated_getter = getall if callable(getall) else get_list if callable(get_list) else None
    if repeated_getter is not None:
        seen: set[str] = set()
        for key in headers:
            normalized = str(key).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            for value in repeated_getter(key):
                yield str(key), value
        return

    yield from headers.items()


async def _invoke_response_callback(
    callback: ResponseHeadersCallback | None,
    status: int,
    headers: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    result = callback(int(status), _CaseInsensitiveHeaders(headers))
    if inspect.isawaitable(result):
        await result


async def _astream_bytes_httpx(
    *,
    method: str,
    url: str,
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    chunk_size: int = 65536,
    cert_pinning: dict[str, set[str]] | None = None,
    sensitive_observability: bool = False,
    on_response: ResponseHeadersCallback | None = None,
    raise_for_status: bool = True,
    verify: bool | str | ssl.SSLContext | None = None,
) -> AsyncIterator[bytes]:
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    retry = retry or RetryPolicy()
    _validate_retry_files_seekable(files, retry)
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)
    observability_url = _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else url

    def _observed_url(candidate: Any) -> Any:
        return _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else candidate

    need_close = False
    ac = client
    if ac is None:
        ac = _get_httpx_async_client(proxies=proxies, verify=verify)
        need_close = False

    attempts = max(1, retry.attempts)
    t0 = time.time()
    try:
        sleep_s = 0.0
        for attempt in range(1, attempts + 1):
            await _avalidate_egress_or_raise(
                url,
                dns_pin_cache=dns_pin_cache,
                sensitive_observability=sensitive_observability,
            )
            yielded_any = False
            callback_error: BaseException | None = None
            response_committed = False
            req_headers = _inject_trace_headers(headers)
            resp = None
            try:
                try:
                    pins_map = cert_pinning or _get_client_cert_pins(ac)
                    if pins_map:
                        u = httpx.URL(url)
                        if u.scheme.lower() == "https":
                            host = (u.host or "").lower()
                            if host in pins_map:
                                _check_cert_pinning(
                                    host,
                                    int(u.port or 443),
                                    pins_map[host],
                                    TLS_MIN_VERSION,
                                    accepted_resolved_ips=_accepted_ips_for_url(
                                        dns_pin_cache,
                                        url,
                                    ),
                                    sensitive_observability=sensitive_observability,
                                )
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                    raise NetworkError(e.__class__.__name__) from e

                async with _preserve_terminal_status_during_stream_cleanup(
                    _httpx_stream_io(
                        client=ac,
                        method=method.upper(),
                        url=url,
                        headers=req_headers,
                        cookies=cookies,
                        params=params,
                        json=json,
                        data=data,
                        files=files,
                        timeout=timeout,
                        chunk_size=chunk_size,
                        accepted_resolved_ips=_accepted_ips_for_url(
                            dns_pin_cache,
                            url,
                        ),
                    )
                ) as (resp, byte_iter):
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError:
                        should, rsn = _should_retry(method, resp.status_code, None, retry)
                        if should and attempt < attempts:
                            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                get_metrics_registry().increment(
                                    "http_client_retries_total",
                                    1,
                                    labels={"reason": rsn},
                                )
                            delay = 0.0
                            if retry.respect_retry_after:
                                delay = _parse_retry_after_delay_seconds(resp.headers.get("retry-after")) or 0.0
                            if delay <= 0:
                                delay = _decorrelated_jitter_sleep(
                                    sleep_s,
                                    retry.backoff_base_ms,
                                    retry.backoff_cap_s,
                                )
                            logger.debug(
                                f"astream_bytes retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                                f"url={observability_url}"
                            )
                            await asyncio.sleep(delay)
                            sleep_s = delay
                            continue
                        if on_response is not None:
                            try:
                                await _invoke_response_callback(on_response, resp.status_code, resp.headers)
                                response_committed = True
                            except Exception as exc:
                                callback_error = exc
                                raise
                        if raise_for_status:
                            _log_outbound_request(
                                method=method,
                                url=_observed_url(resp.request.url),
                                status_code=int(resp.status_code),
                                start_time=t0,
                                attempt=attempt,
                                last_retry_delay_s=sleep_s,
                            )
                            raise _TerminalHTTPStatusError(resp.status_code) from None

                    if on_response is not None and not response_committed:
                        try:
                            await _invoke_response_callback(on_response, resp.status_code, resp.headers)
                            response_committed = True
                        except Exception as exc:
                            callback_error = exc
                            raise
                    timed_iter = _iter_bytes_with_timeouts(byte_iter, timeout)
                    try:
                        async for chunk in timed_iter:
                            if chunk:
                                yielded_any = True
                            yield chunk
                    finally:
                        await timed_iter.aclose()
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(resp.request.url),
                        status_code=int(resp.status_code),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                    )
                    return
            except _TerminalHTTPStatusError as e:
                raise_detached_error(
                    _terminal_status_network_error(e)
                )
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as e:
                if e is callback_error:
                    raise
                raise
            except httpx.HTTPError as e:
                if e is callback_error:
                    raise
                network_exc = NetworkError(e.__class__.__name__)
                if response_committed:
                    raise network_exc from e
                if yielded_any:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(url),
                        status_code=int(getattr(resp, "status_code", 0) or 0),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=network_exc.__class__.__name__,
                    )
                    if sensitive_observability:
                        raise network_exc from None
                    raise network_exc from e
                should, rsn = _should_retry(method, None, e, retry)
                if not should or attempt == attempts:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(url),
                        status_code=0,
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=network_exc.__class__.__name__,
                    )
                    if sensitive_observability:
                        raise network_exc from None
                    raise network_exc from e
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    get_metrics_registry().increment(
                        "http_client_retries_total",
                        1,
                        labels={"reason": rsn},
                    )
                delay = _decorrelated_jitter_sleep(
                    sleep_s,
                    retry.backoff_base_ms,
                    retry.backoff_cap_s,
                )
                logger.debug(
                    f"astream_bytes network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                    f"url={observability_url}"
                )
                await asyncio.sleep(delay)
                sleep_s = delay
            except NetworkError as e:
                if e is callback_error:
                    raise
                if response_committed:
                    raise
                if yielded_any:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(url),
                        status_code=int(getattr(resp, "status_code", 0) or 0),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=e.__class__.__name__,
                    )
                    raise
                should, rsn = _should_retry(method, None, e, retry)
                if not should or attempt == attempts:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(url),
                        status_code=0,
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=e.__class__.__name__,
                    )
                    raise
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    get_metrics_registry().increment(
                        "http_client_retries_total",
                        1,
                        labels={"reason": rsn},
                    )
                delay = _decorrelated_jitter_sleep(
                    sleep_s,
                    retry.backoff_base_ms,
                    retry.backoff_cap_s,
                )
                logger.debug(
                    f"astream_bytes network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                    f"url={observability_url}"
                )
                await asyncio.sleep(delay)
                sleep_s = delay
    except asyncio.CancelledError:
        # propagate cancellations cleanly
        raise
    finally:
        if need_close:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                await ac.aclose()


async def _astream_bytes_aiohttp(
    *,
    method: str,
    url: str,
    client: Any | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    chunk_size: int = 65536,
    cert_pinning: dict[str, set[str]] | None = None,
    sensitive_observability: bool = False,
    on_response: ResponseHeadersCallback | None = None,
    raise_for_status: bool = True,
    verify: bool | str | ssl.SSLContext | None = None,
) -> AsyncIterator[bytes]:
    if aiohttp is None:  # pragma: no cover
        raise RuntimeError("aiohttp is not available")  # noqa: TRY003
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    retry = retry or RetryPolicy()
    _validate_retry_files_seekable(files, retry)
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)
    observability_url = _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else url

    def _observed_url(candidate: Any) -> Any:
        return _SENSITIVE_OBSERVABILITY_URL if sensitive_observability else candidate

    session = client or _get_aiohttp_session()
    attempts = max(1, retry.attempts)
    t0 = time.time()
    try:
        sleep_s = 0.0
        for attempt in range(1, attempts + 1):
            await _avalidate_egress_or_raise(
                url,
                dns_pin_cache=dns_pin_cache,
                sensitive_observability=sensitive_observability,
            )
            yielded_any = False
            callback_error: BaseException | None = None
            response_committed = False
            terminal_status_error = False
            resp = None
            req_headers = _inject_trace_headers(headers)
            try:
                try:
                    pins_map = cert_pinning or _get_client_cert_pins(session)
                    if pins_map:
                        if httpx is not None:
                            u = httpx.URL(url)
                            host = (u.host or "").lower()
                            port = int(u.port or (443 if (u.scheme or "").lower() == "https" else 80))
                        else:
                            parsed = urlparse(url)
                            host = (parsed.hostname or "").lower()
                            port = parsed.port or (443 if (parsed.scheme or "").lower() == "https" else 80)
                        if host in pins_map:
                            _check_cert_pinning(
                                host,
                                port,
                                pins_map[host],
                                TLS_MIN_VERSION,
                                accepted_resolved_ips=_accepted_ips_for_url(
                                    dns_pin_cache,
                                    url,
                                ),
                                sensitive_observability=sensitive_observability,
                            )
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                    raise NetworkError(e.__class__.__name__) from e

                ssl_ctx = _aiohttp_ssl_from_verify(verify)
                async with _preserve_terminal_status_during_stream_cleanup(
                    _aiohttp_stream_io(
                        session=session,
                        method=method.upper(),
                        url=url,
                        headers=req_headers,
                        cookies=cookies,
                        params=params,
                        json=json,
                        data=data,
                        files=files,
                        timeout=timeout,
                        proxies=proxies,
                        ssl_override=ssl_ctx,
                        chunk_size=chunk_size,
                        accepted_resolved_ips=_accepted_ips_for_url(
                            dns_pin_cache,
                            url,
                        ),
                    )
                ) as (resp, byte_iter):
                    if resp.status >= 400:
                        should, rsn = _should_retry(method, resp.status, None, retry)
                        if should and attempt < attempts:
                            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                                get_metrics_registry().increment(
                                    "http_client_retries_total",
                                    1,
                                    labels={"reason": rsn},
                                )
                            delay = 0.0
                            if retry.respect_retry_after:
                                delay = _parse_retry_after_delay_seconds(resp.headers.get("retry-after")) or 0.0
                            if delay <= 0:
                                delay = _decorrelated_jitter_sleep(
                                    sleep_s,
                                    retry.backoff_base_ms,
                                    retry.backoff_cap_s,
                                )
                            logger.debug(
                                f"astream_bytes retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                                f"url={observability_url}"
                            )
                            await asyncio.sleep(delay)
                            sleep_s = delay
                            continue
                        if on_response is not None:
                            try:
                                await _invoke_response_callback(on_response, resp.status, resp.headers)
                                response_committed = True
                            except Exception as exc:
                                callback_error = exc
                                raise
                        if raise_for_status:
                            terminal_status_error = True
                            _log_outbound_request(
                                method=method,
                                url=_observed_url(str(getattr(resp, "url", url))),
                                status_code=int(resp.status),
                                start_time=t0,
                                attempt=attempt,
                                last_retry_delay_s=sleep_s,
                            )
                            raise _TerminalHTTPStatusError(resp.status)

                    if on_response is not None and not response_committed:
                        try:
                            await _invoke_response_callback(on_response, resp.status, resp.headers)
                            response_committed = True
                        except Exception as exc:
                            callback_error = exc
                            raise
                    timed_iter = _iter_bytes_with_timeouts(byte_iter, timeout)
                    try:
                        async for chunk in timed_iter:
                            if not chunk:
                                continue
                            yielded_any = True
                            yield chunk
                    finally:
                        await timed_iter.aclose()
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(str(getattr(resp, "url", url))),
                        status_code=int(resp.status),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                    )
                    return
            except _TerminalHTTPStatusError as e:
                raise_detached_error(
                    _terminal_status_network_error(e)
                )
            except asyncio.CancelledError:
                raise
            except NetworkError as e:
                if e is callback_error or terminal_status_error or response_committed:
                    raise
                if yielded_any:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(str(getattr(resp, "url", url))),
                        status_code=int(getattr(resp, "status", 0) or 0),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=e.__class__.__name__,
                    )
                    raise
                should, rsn = _should_retry(method, None, e, retry)
                if not should or attempt == attempts:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(str(getattr(resp, "url", url))),
                        status_code=0,
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=e.__class__.__name__,
                    )
                    raise
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    get_metrics_registry().increment(
                        "http_client_retries_total",
                        1,
                        labels={"reason": rsn},
                    )
                delay = _decorrelated_jitter_sleep(
                    sleep_s,
                    retry.backoff_base_ms,
                    retry.backoff_cap_s,
                )
                logger.debug(
                    f"astream_bytes network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                    f"url={observability_url}"
                )
                await asyncio.sleep(delay)
                sleep_s = delay
            except _HTTPCLIENT_REQUEST_EXCEPTIONS as e:
                if e is callback_error:
                    raise
                network_exc = NetworkError(e.__class__.__name__)
                if response_committed:
                    raise network_exc from e
                if yielded_any:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(str(getattr(resp, "url", url))),
                        status_code=int(getattr(resp, "status", 0) or 0),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=network_exc.__class__.__name__,
                    )
                    if sensitive_observability:
                        raise network_exc from None
                    raise network_exc from e
                should, rsn = _should_retry(method, None, e, retry)
                if not should or attempt == attempts:
                    _log_outbound_request(
                        method=method,
                        url=_observed_url(str(getattr(resp, "url", url))),
                        status_code=0,
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=network_exc.__class__.__name__,
                    )
                    if sensitive_observability:
                        raise network_exc from None
                    raise network_exc from e
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    get_metrics_registry().increment(
                        "http_client_retries_total",
                        1,
                        labels={"reason": rsn},
                    )
                delay = _decorrelated_jitter_sleep(
                    sleep_s,
                    retry.backoff_base_ms,
                    retry.backoff_cap_s,
                )
                logger.debug(
                    f"astream_bytes network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                    f"url={observability_url}"
                )
                await asyncio.sleep(delay)
                sleep_s = delay
    except asyncio.CancelledError:
        raise


async def astream_bytes(
    *,
    method: str,
    url: str,
    client: Any | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    chunk_size: int = 65536,
    cert_pinning: dict[str, set[str]] | None = None,
    sensitive_observability: bool = False,
    on_response: ResponseHeadersCallback | None = None,
    raise_for_status: bool = True,
    verify: bool | str | ssl.SSLContext | None = None,
) -> AsyncIterator[bytes]:
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    if client is not None:
        adapter_name = "aiohttp" if _is_aiohttp_client(client) else "httpx"
    else:
        adapter_name = "aiohttp" if aiohttp is not None else "httpx"
    adapter = _get_transport_adapter(adapter_name)
    stream = adapter.stream_bytes(
        method=method,
        url=url,
        client=client,
        headers=headers,
        cookies=cookies,
        params=params,
        json=json,
        data=data,
        files=files,
        timeout=timeout,
        proxies=proxies,
        retry=retry,
        chunk_size=chunk_size,
        cert_pinning=cert_pinning,
        sensitive_observability=sensitive_observability,
        on_response=on_response,
        raise_for_status=raise_for_status,
        verify=verify,
    )
    try:
        async for chunk in stream:
            yield chunk
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


async def _astream_sse_httpx(
    *,
    url: str,
    method: str = "GET",
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    timeout: float | httpx.Timeout | None = None,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    sensitive_observability: bool = False,
) -> AsyncIterator[SSEEvent]:
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    hdrs = {"Accept": "text/event-stream"}
    if headers:
        hdrs.update(headers)
    retry = retry or RetryPolicy()
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)

    need_close = False
    ac = client
    if ac is None:
        ac = _get_httpx_async_client(proxies=proxies)
        need_close = False

    attempts = max(1, retry.attempts)
    sleep_s = 0.0
    cur_url = url
    redirects = 0
    t0 = time.time()

    try:
        for attempt in range(1, attempts + 1):
            body_started = False

            def mark_body_started() -> None:
                nonlocal body_started
                body_started = True

            # manual redirect handling before starting to read body
            while True:
                await _avalidate_egress_or_raise(
                    cur_url,
                    dns_pin_cache=dns_pin_cache,
                    sensitive_observability=sensitive_observability,
                )
                try:
                    # Optional cert pinning
                    try:
                        pins_map = cert_pinning or _get_client_cert_pins(ac)
                        if pins_map:
                            u = httpx.URL(cur_url)
                            if u.scheme.lower() == "https":
                                host = (u.host or "").lower()
                                if host in pins_map:
                                    _check_cert_pinning(
                                        host,
                                        int(u.port or 443),
                                        pins_map[host],
                                        TLS_MIN_VERSION,
                                        accepted_resolved_ips=_accepted_ips_for_url(
                                            dns_pin_cache,
                                            cur_url,
                                        ),
                                        sensitive_observability=sensitive_observability,
                                    )
                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                        raise NetworkError(e.__class__.__name__) from e

                    async with _preserve_terminal_status_during_stream_cleanup(
                        _httpx_stream_io(
                            client=ac,
                            method=method.upper(),
                            url=cur_url,
                            headers=_strip_sensitive_headers_for_cross_origin(
                                _inject_trace_headers(hdrs), original_url=url, target_url=cur_url
                            ),
                            params=params,
                            json=json,
                            data=data,
                            timeout=timeout,
                            chunk_size=None,
                            accepted_resolved_ips=_accepted_ips_for_url(
                                dns_pin_cache,
                                cur_url,
                            ),
                        )
                    ) as (resp, byte_iter):
                        # Handle redirect response codes before reading any bytes
                        if resp.status_code in (301, 302, 303, 307, 308):
                            if redirects >= DEFAULT_MAX_REDIRECTS:
                                raise NetworkError("Too many redirects")  # noqa: TRY003
                            location = resp.headers.get("location")
                            if not location:
                                raise NetworkError("Redirect without Location header")  # noqa: TRY003
                            try:
                                next_url = str(resp.request.url.join(httpx.URL(location)))
                            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                try:
                                    next_url = str(httpx.URL(location))
                                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                    raise NetworkError("Invalid redirect Location header")  # noqa: B904, TRY003
                            redirects += 1
                            if _is_cross_origin(url, next_url):
                                _clear_client_cookie_jar(ac)
                            cur_url = next_url
                            continue  # loop to re-validate egress and attempt again
                        # Raise for non-OK statuses pre-body if not retriable
                        if resp.status_code >= 400:
                            should, rsn = _should_retry(method, resp.status_code, None, retry)
                            if not should or attempt == attempts:
                                # escalate as NetworkError; caller handles as appropriate
                                raise _TerminalHTTPStatusError(resp.status_code)
                            delay = 0.0
                            if retry.respect_retry_after:
                                delay = _parse_retry_after_delay_seconds(
                                    resp.headers.get("retry-after")
                                ) or 0.0
                            if delay <= 0:
                                delay = _decorrelated_jitter_sleep(
                                    sleep_s,
                                    retry.backoff_base_ms,
                                    retry.backoff_cap_s,
                                )
                            await asyncio.sleep(delay)
                            sleep_s = delay
                            break  # exit redirect loop to outer attempt

                        # Successful response; iterate SSE bytes and yield events
                        timed_iter = _iter_bytes_with_timeouts(byte_iter, timeout)
                        event_stream = _iter_sse_events_from_bytes(
                            timed_iter,
                            on_body_started=mark_body_started,
                        )
                        try:
                            async for event in event_stream:
                                yield event
                        finally:
                            await event_stream.aclose()
                        # per-request structured log on successful end of stream
                        _log_outbound_request(
                            method=method,
                            url=(
                                _SENSITIVE_OBSERVABILITY_URL
                                if sensitive_observability
                                else resp.request.url
                            ),
                            status_code=int(resp.status_code),
                            start_time=t0,
                            attempt=attempt,
                            last_retry_delay_s=sleep_s,
                        )
                        return  # finished streaming without error
                except _TerminalHTTPStatusError as e:
                    raise_detached_error(
                        _terminal_status_network_error(e)
                    )
                except asyncio.CancelledError:
                    raise
                except _HTTPCLIENT_REQUEST_EXCEPTIONS as e:
                    # network or early error before bytes consumed
                    network_error = _sse_network_error(
                        e,
                        sensitive_observability=sensitive_observability,
                    )
                    if body_started:
                        if sensitive_observability:
                            raise network_error from None
                        raise
                    should, rsn = _should_retry(method, None, e, retry)
                    if not should or attempt == attempts:
                        if sensitive_observability:
                            raise network_error from None
                        raise
                    delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                    await asyncio.sleep(delay)
                    sleep_s = delay
                    break  # next outer attempt
        # exhausted attempts
        _log_outbound_request(
            method=method,
            url=(
                _SENSITIVE_OBSERVABILITY_URL
                if sensitive_observability
                else cur_url
            ),
            status_code=0,
            start_time=t0,
            attempt=attempts,
            last_retry_delay_s=sleep_s,
            exception_class="RetryExhaustedError",
        )
        raise RetryExhaustedError("All retry attempts exhausted (astream_sse)")  # noqa: TRY003
    finally:
        if need_close:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                await ac.aclose()


async def _astream_sse_aiohttp(
    *,
    url: str,
    method: str = "GET",
    client: Any | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    files: Any | None = None,
    timeout: Any | None = None,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    sensitive_observability: bool = False,
) -> AsyncIterator[SSEEvent]:
    if aiohttp is None:  # pragma: no cover
        raise RuntimeError("aiohttp is not available")  # noqa: TRY003
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    hdrs = {"Accept": "text/event-stream"}
    if headers:
        hdrs.update(headers)
    retry = retry or RetryPolicy()
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    await _avalidate_egress_or_raise(
        url,
        dns_pin_cache=dns_pin_cache,
        sensitive_observability=sensitive_observability,
    )
    _validate_proxies_or_raise(proxies)

    session = client or _get_aiohttp_session()
    attempts = max(1, retry.attempts)
    sleep_s = 0.0
    cur_url = url
    redirects = 0
    t0 = time.time()

    for attempt in range(1, attempts + 1):
        body_started = False

        def mark_body_started() -> None:
            nonlocal body_started
            body_started = True

        while True:
            await _avalidate_egress_or_raise(
                cur_url,
                dns_pin_cache=dns_pin_cache,
                sensitive_observability=sensitive_observability,
            )
            try:
                # Optional cert pinning
                try:
                    pins_map = cert_pinning or _get_client_cert_pins(session)
                    if pins_map:
                        if httpx is not None:
                            u = httpx.URL(cur_url)
                            host = (u.host or "").lower()
                            port = int(u.port or (443 if (u.scheme or "").lower() == "https" else 80))
                        else:
                            parsed = urlparse(cur_url)
                            host = (parsed.hostname or "").lower()
                            port = parsed.port or (443 if (parsed.scheme or "").lower() == "https" else 80)
                        if host in pins_map:
                            _check_cert_pinning(
                                host,
                                port,
                                pins_map[host],
                                TLS_MIN_VERSION,
                                accepted_resolved_ips=_accepted_ips_for_url(
                                    dns_pin_cache,
                                    cur_url,
                                ),
                                sensitive_observability=sensitive_observability,
                            )
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                    raise NetworkError(e.__class__.__name__) from e

                ssl_ctx = _build_ssl_context(ENFORCE_TLS_MIN, TLS_MIN_VERSION)
                async with _preserve_terminal_status_during_stream_cleanup(
                    _aiohttp_stream_io(
                        session=session,
                        method=method.upper(),
                        url=cur_url,
                        headers=_strip_sensitive_headers_for_cross_origin(
                            _inject_trace_headers(hdrs), original_url=url, target_url=cur_url
                        ),
                        params=params,
                        json=json,
                        data=data,
                        files=files,
                        timeout=timeout,
                        proxies=proxies,
                        ssl_override=ssl_ctx,
                        chunk_size=None,
                        accepted_resolved_ips=_accepted_ips_for_url(
                            dns_pin_cache,
                            cur_url,
                        ),
                    )
                ) as (resp, byte_iter):
                    if resp.status in (301, 302, 303, 307, 308):
                        if redirects >= DEFAULT_MAX_REDIRECTS:
                            raise NetworkError("Too many redirects")  # noqa: TRY003
                        location = resp.headers.get("location")
                        if not location:
                            raise NetworkError("Redirect without Location header")  # noqa: TRY003
                        next_url = _resolve_redirect_url(cur_url, location)
                        if not next_url:
                            raise NetworkError("Invalid redirect Location header")  # noqa: TRY003
                        redirects += 1
                        if _is_cross_origin(url, next_url):
                            _clear_client_cookie_jar(session)
                        cur_url = next_url
                        continue

                    if resp.status >= 400:
                        should, rsn = _should_retry(method, resp.status, None, retry)
                        if not should or attempt == attempts:
                            raise _TerminalHTTPStatusError(resp.status)
                        delay = 0.0
                        if retry.respect_retry_after:
                            delay = _parse_retry_after_delay_seconds(
                                resp.headers.get("retry-after")
                            ) or 0.0
                        if delay <= 0:
                            delay = _decorrelated_jitter_sleep(
                                sleep_s,
                                retry.backoff_base_ms,
                                retry.backoff_cap_s,
                            )
                        await asyncio.sleep(delay)
                        sleep_s = delay
                        break

                    timed_iter = _iter_bytes_with_timeouts(byte_iter, timeout)
                    event_stream = _iter_sse_events_from_bytes(
                        timed_iter,
                        on_body_started=mark_body_started,
                    )
                    try:
                        async for event in event_stream:
                            yield event
                    finally:
                        await event_stream.aclose()
                    _log_outbound_request(
                        method=method,
                        url=(
                            _SENSITIVE_OBSERVABILITY_URL
                            if sensitive_observability
                            else str(getattr(resp, "url", cur_url))
                        ),
                        status_code=int(resp.status),
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                    )
                    return
            except _TerminalHTTPStatusError as e:
                raise_detached_error(
                    _terminal_status_network_error(e)
                )
            except asyncio.CancelledError:
                raise
            except _HTTPCLIENT_REQUEST_EXCEPTIONS as e:
                network_error = _sse_network_error(
                    e,
                    sensitive_observability=sensitive_observability,
                )
                if body_started:
                    if sensitive_observability:
                        raise network_error from None
                    raise
                should, rsn = _should_retry(method, None, e, retry)
                if not should or attempt == attempts:
                    if sensitive_observability:
                        raise network_error from None
                    raise
                delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                await asyncio.sleep(delay)
                sleep_s = delay
                break

    _log_outbound_request(
        method=method,
        url=(
            _SENSITIVE_OBSERVABILITY_URL
            if sensitive_observability
            else cur_url
        ),
        status_code=0,
        start_time=t0,
        attempt=attempts,
        last_retry_delay_s=sleep_s,
        exception_class="RetryExhaustedError",
    )
    raise RetryExhaustedError("All retry attempts exhausted (astream_sse)")  # noqa: TRY003


async def astream_sse(
    *,
    url: str,
    method: str = "GET",
    client: Any | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any | None = None,
    data: Any | None = None,
    timeout: Any | None = None,
    proxies: str | dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    sensitive_observability: bool = False,
) -> AsyncIterator[SSEEvent]:
    sensitive_observability = _effective_sensitive_observability(
        sensitive_observability
    )
    if client is not None:
        adapter_name = "aiohttp" if _is_aiohttp_client(client) else "httpx"
    else:
        adapter_name = "aiohttp" if aiohttp is not None else "httpx"
    adapter = _get_transport_adapter(adapter_name)
    stream = adapter.stream_sse(
        url=url,
        method=method,
        client=client,
        headers=headers,
        params=params,
        json=json,
        data=data,
        timeout=timeout,
        proxies=proxies,
        retry=retry,
        cert_pinning=cert_pinning,
        sensitive_observability=sensitive_observability,
    )
    try:
        async for event in stream:
            yield event
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


def _parse_sse_event(raw: str) -> SSEEvent | None:
    event = SSEEvent()
    data_lines: list[str] = []
    saw_event = False
    saw_id = False
    saw_retry = False
    for line in raw.splitlines():
        if not line or line.startswith(":"):
            continue
        if ":" in line:
            field, val = line.split(":", 1)
            val = val[1:] if val.startswith(" ") else val
        else:
            field, val = line, ""
        if field == "event":
            event.event = val
            saw_event = True
        elif field == "data":
            data_lines.append(val)
        elif field == "id":
            event.id = val
            saw_id = True
        elif field == "retry":
            try:
                event.retry = int(val)
                saw_retry = True
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                pass
    event.data = "\n".join(data_lines)
    if not data_lines and not saw_event and not saw_id and not saw_retry:
        return None
    return event


# --------------------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------------------

def download(
    *,
    url: str,
    dest: str | Path,
    client: httpx.Client | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    proxies: str | dict[str, str] | None = None,
    checksum: str | None = None,
    checksum_alg: str = "sha256",
    resume: bool = False,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    # Optional safety checks
    max_bytes_total: int | None = None,
    require_content_type: str | None = None,
) -> Path:
    if httpx is None:  # pragma: no cover
        raise RuntimeError("httpx is not available")  # noqa: TRY003
    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    _validate_egress_or_raise(url, dns_pin_cache=dns_pin_cache)
    _validate_proxies_or_raise(proxies)
    t0 = time.time()
    observability_url = (
        _SENSITIVE_OBSERVABILITY_URL
        if _effective_sensitive_observability()
        else url
    )
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    retry = retry or RetryPolicy()

    need_close = False
    sc = client
    if sc is None:
        sc = _get_httpx_client(proxies=proxies)
        need_close = False

    attempts = max(1, retry.attempts)
    sleep_s = 0.0

    try:
        for attempt in range(1, attempts + 1):
            _validate_egress_or_raise(url, dns_pin_cache=dns_pin_cache)
            req_headers = _inject_trace_headers(headers)
            # Basic resume support
            existing = 0
            if resume and tmp_path.exists():
                try:
                    existing = tmp_path.stat().st_size
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                    existing = 0
                if existing > 0:
                    req_headers = dict(req_headers)
                    req_headers["Range"] = f"bytes={existing}-"
            # Enforce disk quota if resuming
            if max_bytes_total is not None and existing > max_bytes_total:
                raise DownloadError("Disk quota exceeded before download")  # noqa: TRY003

            last_exc: Exception | None = None
            try:
                # Optional cert pinning
                try:
                    pins_map = cert_pinning or _get_client_cert_pins(sc)
                    if pins_map:
                        u = httpx.URL(url)
                        if u.scheme.lower() == "https":
                            host = (u.host or "").lower()
                            if host in pins_map:
                                _check_cert_pinning(
                                    host,
                                    int(u.port or 443),
                                    pins_map[host],
                                    TLS_MIN_VERSION,
                                    accepted_resolved_ips=_accepted_ips_for_url(
                                        dns_pin_cache,
                                        url,
                                    ),
                                )
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                    raise DownloadError(str(e))  # noqa: B904
                transport_url, transport_headers, sni_hostname = _prepare_pinned_transport_target(
                    url,
                    req_headers,
                    _accepted_ips_for_url(dns_pin_cache, url),
                )
                stream_options: dict[str, Any] = {
                    "headers": transport_headers,
                    "params": params,
                    "timeout": timeout,
                }
                if sni_hostname:
                    stream_options["extensions"] = {
                        "sni_hostname": sni_hostname
                    }
                with sc.stream("GET", transport_url, **stream_options) as resp:
                    if transport_url != url:
                        _restore_httpx_response_url(resp, url)
                    if resp.status_code in (200, 206):
                        # Optional content-type enforcement
                        if require_content_type:
                            ctype = (resp.headers.get("content-type") or "").lower()
                            if require_content_type.lower() not in ctype:
                                raise DownloadError("Unexpected Content-Type")  # noqa: TRY003
                        hasher = hashlib.new(checksum_alg) if checksum else None
                        mode = "ab" if (resume and existing > 0 and resp.status_code == 206) else "wb"
                        with open(tmp_path, mode) as f:
                            written = existing if (resume and mode == "ab") else 0
                            for chunk in resp.iter_bytes():
                                if not chunk:
                                    continue
                                if max_bytes_total is not None and written + len(chunk) > max_bytes_total:
                                    raise DownloadError("Disk quota exceeded")  # noqa: TRY003
                                f.write(chunk)
                                if hasher is not None:
                                    hasher.update(chunk)
                                written += len(chunk)
                        # Validate checksum
                        if checksum and hasher is not None:
                            hex_val = hasher.hexdigest()
                            if hex_val.lower() != checksum.lower():
                                raise DownloadError("Checksum validation failed")  # noqa: TRY003
                        # Validate content-length if present (when not resuming)
                        clen = resp.headers.get("content-length")
                        if clen and not resume:
                            try:
                                if tmp_path.stat().st_size != int(clen):
                                    raise DownloadError("Content-Length mismatch")  # noqa: TRY003
                            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:  # noqa: TRY203
                                raise
                        tmp_path.replace(dest_path)
                        # per-request structured log
                        _log_outbound_request(
                            method="GET",
                            url=resp.request.url if hasattr(resp.request, "url") else url,
                            status_code=int(resp.status_code),
                            start_time=t0,
                            attempt=attempt,
                            last_retry_delay_s=sleep_s,
                        )
                        return dest_path
                    else:
                        should, rsn = _should_retry("GET", resp.status_code, None, retry)
                        if not should or attempt == attempts:
                            # terminal error response
                            _log_outbound_request(
                                method="GET",
                                url=resp.request.url if hasattr(resp.request, "url") else url,
                                status_code=int(resp.status_code),
                                start_time=t0,
                                attempt=attempt,
                                last_retry_delay_s=sleep_s,
                            )
                            raise DownloadError(f"Download failed with status {resp.status_code}")  # noqa: TRY003
                        with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                            get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": rsn})
                        delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                        logger.debug(
                            f"download retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                            f"url={_sanitize_url_for_logs(observability_url)}"
                        )
                        time.sleep(delay)
                        sleep_s = delay
                        if not resume:
                            try:
                                if tmp_path.exists():
                                    tmp_path.unlink()
                            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                                pass
                        continue
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS as e:
                last_exc = e

            if last_exc is not None:
                should, rsn = _should_retry("GET", None, last_exc, retry)
                if not should or attempt == attempts:
                    try:
                        if tmp_path.exists() and (not resume or attempt == attempts):
                            tmp_path.unlink()
                    except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                        pass
                    # terminal network error
                    _log_outbound_request(
                        method="GET",
                        url=url,
                        status_code=0,
                        start_time=t0,
                        attempt=attempt,
                        last_retry_delay_s=sleep_s,
                        exception_class=(last_exc.__class__.__name__ if last_exc else "DownloadError"),
                    )
                    if isinstance(last_exc, DownloadError):
                        raise last_exc
                    raise DownloadError(str(last_exc))
                with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                    get_metrics_registry().increment("http_client_retries_total", 1, labels={"reason": rsn})
                delay = _decorrelated_jitter_sleep(sleep_s, retry.backoff_base_ms, retry.backoff_cap_s)
                logger.debug(
                    f"download network retry attempt={attempt} reason={rsn} delay={delay:.3f}s "
                    f"url={_sanitize_url_for_logs(observability_url)}"
                )
                time.sleep(delay)
                sleep_s = delay
                continue
        _log_outbound_request(
            method="GET",
            url=url,
            status_code=0,
            start_time=t0,
            attempt=attempts,
            last_retry_delay_s=sleep_s,
            exception_class="RetryExhaustedError",
        )
        raise RetryExhaustedError("All retry attempts exhausted (download)")  # noqa: TRY003
    finally:
        if need_close:
            with suppress(_HTTPCLIENT_NONCRITICAL_EXCEPTIONS):
                sc.close()


async def adownload(
    *,
    url: str,
    dest: str | Path,
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    proxies: str | dict[str, str] | None = None,
    checksum: str | None = None,
    checksum_alg: str = "sha256",
    resume: bool = False,
    retry: RetryPolicy | None = None,
    cert_pinning: dict[str, set[str]] | None = None,
    max_bytes_total: int | None = None,
    require_content_type: str | None = None,
) -> Path:
    # Reuse sync downloader in a thread to avoid blocking event loop on file I/O
    return await asyncio.to_thread(
        download,
        url=url,
        dest=dest,
        client=None,  # create own sync client
        headers=headers,
        params=params,
        timeout=timeout,
        proxies=proxies,
        checksum=checksum,
        checksum_alg=checksum_alg,
        resume=resume,
        retry=retry,
        cert_pinning=cert_pinning,
        max_bytes_total=max_bytes_total,
        require_content_type=require_content_type,
    )


__all__ = [
    "HttpResponse",
    "ResponseHeadersCallback",
    "RetryPolicy",
    "SSEEvent",
    "build_limits",
    "validate_proxies_or_raise",
    "create_async_client",
    "create_client",
    "afetch",
    "fetch",
    "afetch_json",
    "fetch_json",
    "stream_response",
    "astream_bytes",
    "astream_sse",
    "download",
    "adownload",
    "shutdown_http_client",
]
