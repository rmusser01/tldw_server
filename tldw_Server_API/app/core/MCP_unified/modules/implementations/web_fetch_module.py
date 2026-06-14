"""Read-only ``web.fetch`` MCP tool with centralized outbound policy controls.

The tool retrieves a single user-specified URL, enforces the always-on
SSRF/egress outbound policy, and returns bounded extracted page content. Because
the tool takes a ``url`` argument, the gateway runtime automatically governs it
with the existing domain permission subjects (Claude-style ``WebFetch(<domain>)``
allow/deny/ask rules) — this module does not re-implement domain policy.
"""

from __future__ import annotations

import codecs
import html as html_lib
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

from loguru import logger

from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_tool_eval_metadata,
)
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    decide_web_outbound_policy,
)

from ..base import ModuleConfig, create_tool_definition
from .web_cache import ResponseCache, make_cache_key
from .web_rate_limit import DomainRateLimiter
from .web_tool_base import CONTROL_CHARS_RE, WebToolBase, WebToolError

_TOOL_FETCH = "web.fetch"
_TOOL_PROMPT_VERSION = "2026.06.12"

_ALLOWED_ARGS = {"url", "format", "max_bytes", "timeout_seconds", "respect_robots"}
_FORMATS = {"markdown", "text", "html"}

_DEFAULT_MAX_BYTES = 1_000_000
_MAX_MAX_BYTES = 5_000_000
_DEFAULT_TIMEOUT_SECONDS = 15
_MAX_TIMEOUT_SECONDS = 30
_DEFAULT_USER_AGENT = "tldw-mcp-web-fetch/1.0"

# Redirects are followed manually so the outbound policy is re-checked before
# each hop (auto-following would let a permitted URL redirect into denied
# address space and bypass SSRF/egress protection).
_REDIRECT_STATUS = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 5

# Content types we will surface as text. Anything else is rejected so the tool
# never returns binary blobs through the JSON tool contract.
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_PLAIN_TYPES = {
    "text/plain",
    "text/markdown",
    "application/json",
    "application/xml",
    "text/xml",
    "application/ld+json",
}

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t ]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class WebFetchResponse:
    """Normalized HTTP response consumed by :class:`WebFetchModule`.

    ``location`` carries the absolute redirect target when ``status_code`` is a
    redirect; the module re-checks the outbound policy against it before
    following, so the client never follows redirects on its own.
    """

    final_url: str
    status_code: int
    content_type: str
    body: bytes
    truncated: bool
    location: str | None = None


class WebFetchHttpClient(Protocol):
    """Injectable fetcher so the module is unit-testable without the network."""

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_bytes: int,
        user_agent: str,
    ) -> WebFetchResponse: ...


def _is_supported_content_type(content_type: str) -> bool:
    main_type = content_type.split(";", 1)[0].strip().lower()
    return (not main_type) or main_type in _HTML_TYPES or main_type in _PLAIN_TYPES


def _safe_host(url: str) -> str:
    """Return just the host for log context, never the path/query (may hold secrets)."""
    try:
        return urlsplit(url).hostname or "unknown"
    except ValueError:
        return "unknown"


class HttpxWebFetchClient:
    """Default :class:`WebFetchHttpClient` backed by ``httpx`` with a byte cap.

    Redirects are NOT auto-followed: a redirect response is returned with its
    resolved ``location`` so the caller can re-apply the outbound policy. The
    body is only downloaded for terminal responses with a supported content
    type, avoiding fetching binary payloads (images/video/PDF) just to reject
    them later.
    """

    def __init__(self, *, transport: Any | None = None) -> None:
        # ``transport`` is an injection seam for tests (e.g. httpx.MockTransport).
        self._transport = transport

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_bytes: int,
        user_agent: str,
    ) -> WebFetchResponse:
        import httpx  # Local import: keeps module import cheap and httpx optional at import time.

        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout_seconds,
            headers={"User-Agent": user_agent},
            transport=self._transport,
        ) as client:
            async with client.stream("GET", url) as response:
                content_type = response.headers.get("content-type", "")

                if response.status_code in _REDIRECT_STATUS:
                    location_header = response.headers.get("location")
                    location = urljoin(url, location_header) if location_header else None
                    return WebFetchResponse(
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=content_type,
                        body=b"",
                        truncated=False,
                        location=location,
                    )

                if not _is_supported_content_type(content_type):
                    # Reject unsupported (binary) payloads without downloading them.
                    return WebFetchResponse(
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=content_type,
                        body=b"",
                        truncated=False,
                    )

                chunks: list[bytes] = []
                downloaded = 0
                truncated = False
                async for chunk in response.aiter_bytes():
                    remaining = max_bytes - downloaded
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(chunk) > remaining:
                        chunks.append(chunk[:remaining])
                        downloaded += remaining
                        truncated = True
                        break
                    chunks.append(chunk)
                    downloaded += len(chunk)
                return WebFetchResponse(
                    final_url=str(response.url),
                    status_code=response.status_code,
                    content_type=content_type,
                    body=b"".join(chunks),
                    truncated=truncated,
                )


class WebFetchModule(WebToolBase):
    """Single read-only ``web.fetch`` tool governed by outbound + domain policy."""

    _ACTION_FAMILY = "web_fetch"
    _RESULT_KIND = "bounded_web_document"
    _TOOL_PROMPT_VERSION = _TOOL_PROMPT_VERSION

    def __init__(
        self,
        config: ModuleConfig,
        *,
        client: WebFetchHttpClient | None = None,
        rate_limiter: DomainRateLimiter | None = None,
        response_cache: ResponseCache | None = None,
    ) -> None:
        super().__init__(config)
        self._client: WebFetchHttpClient = client or HttpxWebFetchClient()
        # A default, generously-bounded per-domain limiter is on unless the caller
        # supplies one (pass DomainRateLimiter(max_requests=0) to disable).
        self._rate_limiter: DomainRateLimiter = rate_limiter or DomainRateLimiter()
        # Response caching is opt-in (None = no caching) since it trades freshness.
        self._response_cache: ResponseCache | None = response_cache

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True, "client": self._client is not None}

    async def get_tools(self) -> list[dict[str, Any]]:
        tool = create_tool_definition(
            name=_TOOL_FETCH,
            description=(
                "Fetch a single http(s) URL and return bounded, extracted page "
                "content. Subject to outbound (SSRF/egress) policy and domain "
                "permission rules."
            ),
            parameters={
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The http(s) URL to fetch.",
                    },
                    "format": {
                        "type": "string",
                        "enum": sorted(_FORMATS),
                        "description": "Extraction format for HTML content.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_MAX_BYTES,
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_TIMEOUT_SECONDS,
                    },
                    "respect_robots": {"type": "boolean"},
                },
                "required": ["url"],
            },
            metadata={
                "category": "web",
                "readOnlyHint": True,
                "uses_network": True,
                "capabilities": ["research.web", "external.network"],
                **build_tool_eval_metadata(
                    tool_prompt_id=f"mcp.{_TOOL_FETCH}.v1",
                    tool_prompt_version=_TOOL_PROMPT_VERSION,
                    task_families=["web_research", "citation_collection"],
                    expected_result_kind="bounded_web_document",
                    success_signals=[
                        "enforced_outbound_policy",
                        "bounded_response",
                        "extracted_readable_content",
                    ],
                ),
            },
        )
        tool["inputSchema"]["additionalProperties"] = False
        return [tool]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if tool_name != _TOOL_FETCH:
            return self._structured_error(tool_name, "unknown_tool", "Unknown web tool.", context=context)

        # Note: the inherited SQL-oriented sanitize_input() is intentionally NOT
        # applied to the raw URL — legitimate URLs commonly contain substrings
        # such as ``--`` or ``/*`` that it would reject. Inputs are validated
        # explicitly below instead.
        try:
            requested_url, fmt, max_bytes, timeout_seconds, respect_robots = self._validate(
                arguments or {}
            )
        except WebToolError as exc:
            return self._structured_error(tool_name, exc.reason_code, exc.message, context=context)

        # Cache hit: return the stored result (the gateway already enforced the
        # call's permission rules, and a cache hit performs no network request).
        cache_key = make_cache_key(requested_url, fmt, max_bytes)
        if self._response_cache is not None:
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}

        # Follow redirects manually, re-applying the outbound policy to every hop
        # so a permitted URL cannot redirect into denied/private address space.
        current_url = requested_url
        for _hop in range(_MAX_REDIRECTS + 1):
            decision = await decide_web_outbound_policy(
                current_url,
                respect_robots=respect_robots,
                user_agent=_DEFAULT_USER_AGENT,
                source="mcp.web_fetch",
                stage="web.fetch",
            )
            if not getattr(decision, "allowed", False):
                return self._structured_error(
                    tool_name,
                    "outbound_policy_denied",
                    f"Outbound policy denied the request ({getattr(decision, 'reason', 'denied')}).",
                    context=context,
                )

            host = _safe_host(current_url)
            if not self._rate_limiter.try_acquire(host):
                logger.bind(stage="web.fetch", host=host).warning("web.fetch per-domain rate limit exceeded")
                return self._structured_error(
                    tool_name,
                    "rate_limited",
                    "Per-domain request rate limit exceeded.",
                    context=context,
                )

            try:
                response = await self._client.fetch(
                    current_url,
                    timeout_seconds=timeout_seconds,
                    max_bytes=max_bytes,
                    user_agent=_DEFAULT_USER_AGENT,
                )
            except Exception as exc:  # noqa: BLE001 - network/client errors are expected and mapped.
                logger.bind(stage="web.fetch", host=_safe_host(current_url)).opt(exception=exc).warning(
                    "web.fetch client error"
                )
                return self._structured_error(tool_name, "fetch_failed", "Failed to fetch the URL.", context=context)

            if response.status_code in _REDIRECT_STATUS:
                if not response.location:
                    logger.bind(stage="web.fetch", host=_safe_host(current_url)).warning(
                        "web.fetch redirect without a Location header"
                    )
                    return self._structured_error(
                        tool_name,
                        "fetch_failed",
                        "Upstream returned a redirect without a Location header.",
                        status_code=response.status_code,
                        context=context,
                    )
                current_url = response.location
                continue
            break
        else:
            logger.bind(stage="web.fetch", host=_safe_host(requested_url)).warning(
                "web.fetch exceeded the redirect limit"
            )
            return self._structured_error(
                tool_name,
                "fetch_failed",
                f"Exceeded the redirect limit ({_MAX_REDIRECTS}).",
                context=context,
            )

        if response.status_code >= 400:
            return self._structured_error(
                tool_name,
                "fetch_failed",
                f"Upstream returned status {response.status_code}.",
                status_code=response.status_code,
                context=context,
            )

        try:
            content, title = self._extract(response, fmt)
        except WebToolError as exc:
            logger.bind(
                stage="web.fetch",
                host=_safe_host(response.final_url),
                reason_code=exc.reason_code,
                content_type=response.content_type,
            ).debug("web.fetch extraction rejected the response")
            return self._structured_error(
                tool_name,
                exc.reason_code,
                exc.message,
                status_code=response.status_code,
                context=context,
            )

        result = {
            "ok": True,
            "url": requested_url,
            "final_url": response.final_url,
            "status_code": response.status_code,
            "content_type": response.content_type,
            "title": title,
            "format": fmt,
            "content": content,
            "bytes_fetched": len(response.body),
            "truncated": bool(response.truncated),
            "cached": False,
            "eval": self._eval_metadata(
                _TOOL_FETCH,
                reason_code=None,
                truncated=bool(response.truncated),
                context=context,
            ),
        }
        if self._response_cache is not None:
            self._response_cache.put(cache_key, result)
        return result

    # ---- validation ----------------------------------------------------

    def _validate(self, args: dict[str, Any]) -> tuple[str, str, int, int, bool]:
        unknown = sorted(set(args) - _ALLOWED_ARGS)
        if unknown:
            raise WebToolError("invalid_arguments", f"unknown arguments: {', '.join(unknown)}")

        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise WebToolError("invalid_arguments", "url is required")
        url = url.strip()
        if CONTROL_CHARS_RE.search(url):
            raise WebToolError("invalid_url", "url must not contain control characters")
        if not re.match(r"^https?://", url, re.IGNORECASE):
            raise WebToolError("invalid_url", "url must use the http or https scheme")

        fmt = args.get("format", "markdown")
        if fmt not in _FORMATS:
            raise WebToolError("invalid_arguments", "format must be one of markdown, text, html")

        max_bytes = self._bounded_int(
            args, "max_bytes", default=_DEFAULT_MAX_BYTES, maximum=_MAX_MAX_BYTES
        )
        timeout_seconds = self._bounded_int(
            args, "timeout_seconds", default=_DEFAULT_TIMEOUT_SECONDS, maximum=_MAX_TIMEOUT_SECONDS
        )

        respect_robots = args.get("respect_robots", False)
        if not isinstance(respect_robots, bool):
            raise WebToolError("invalid_arguments", "respect_robots must be a boolean")

        return url, fmt, max_bytes, timeout_seconds, respect_robots

    @staticmethod
    def _bounded_int(args: dict[str, Any], name: str, *, default: int, maximum: int) -> int:
        value = args.get(name)
        if value is None:
            return default
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WebToolError("invalid_arguments", f"{name} must be a positive integer")
        if value > maximum:
            raise WebToolError("invalid_arguments", f"{name} exceeds maximum ({maximum})")
        return value

    # ---- extraction ----------------------------------------------------

    def _extract(self, response: WebFetchResponse, fmt: str) -> tuple[str, str | None]:
        main_type = response.content_type.split(";", 1)[0].strip().lower()
        text = self._decode_body(response.body, response.content_type)

        if main_type in _HTML_TYPES or (not main_type and "<html" in text.lower()):
            if fmt == "html":
                return text, self._html_title(text)
            content = self._extract_html(text, fmt)
            if not content:
                raise WebToolError("empty_content", "No readable content could be extracted.")
            return content, self._html_title(text)

        # Plain text, or an absent/misconfigured content type that did not look
        # like HTML: surface the decoded body directly.
        if main_type in _PLAIN_TYPES or not main_type:
            cleaned = text.strip()
            if not cleaned:
                raise WebToolError("empty_content", "Response body was empty.")
            return cleaned, None

        raise WebToolError(
            "empty_content", f"Unsupported content type for extraction: {main_type}"
        )

    @staticmethod
    def _decode_body(body: bytes, content_type: str) -> str:
        """Decode the response body using the charset advertised in the header."""
        charset = "utf-8"
        if "charset=" in content_type.lower():
            candidate = content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip().strip('"')
            try:
                codecs.lookup(candidate)
                charset = candidate
            except (LookupError, ValueError):
                charset = "utf-8"
        return body.decode(charset, errors="replace")

    def _extract_html(self, html: str, fmt: str) -> str:
        output_format = "markdown" if fmt == "markdown" else "txt"
        try:
            import trafilatura

            extracted = trafilatura.extract(html, output_format=output_format)
            if extracted and extracted.strip():
                return extracted.strip()
        except Exception as exc:  # noqa: BLE001 - extractor failures fall back to tag strip.
            logger.bind(stage="web.fetch").opt(exception=exc).debug(
                "trafilatura extraction failed; falling back to tag strip"
            )
        return self._strip_tags(html)

    @staticmethod
    def _strip_tags(html: str) -> str:
        without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
        text = _TAG_RE.sub(" ", without_scripts)
        text = html_lib.unescape(text)
        text = _WS_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = _BLANKLINES_RE.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def _html_title(html: str) -> str | None:
        match = _TITLE_RE.search(html)
        if not match:
            return None
        title = html_lib.unescape(_WS_RE.sub(" ", _TAG_RE.sub(" ", match.group(1)))).strip()
        return title or None
