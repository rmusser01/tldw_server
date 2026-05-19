from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from loguru import logger

try:
    # Local HTTP client used across the project; enforces egress policy internally
    from tldw_Server_API.app.core.http_client import fetch as http_fetch
except ImportError:  # pragma: no cover - defensive import to avoid runtime breakage
    http_fetch = None  # type: ignore[assignment]

try:
    # Centralized egress policy evaluator
    from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
except ImportError:  # pragma: no cover - defensive import
    evaluate_url_policy = None  # type: ignore[assignment]

_WEB_FILTER_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class URLFilter:
    """Synchronous URL filter base class."""

    def apply(self, url: str) -> bool:  # pragma: no cover - simple interface
        raise NotImplementedError


class FilterChain:
    """Apply a sequence of filters (AND semantics)."""

    def __init__(self, filters: list[URLFilter] | None = None) -> None:
        self.filters = list(filters or [])

    def add(self, f: URLFilter) -> FilterChain:
        self.filters.append(f)
        return self

    def apply(self, url: str) -> bool:
        return all(f.apply(url) for f in self.filters)


@dataclass
class DomainFilter(URLFilter):
    """Allow/deny domains (optionally including subdomains)."""

    allowed: set[str] | None = None
    blocked: set[str] | None = None
    include_subdomains: bool = True

    def _host(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except (TypeError, ValueError):
            return ""

    def _is_sub(self, host: str, root: str) -> bool:
        if not host or not root:
            return False
        return host == root or host.endswith("." + root)

    def apply(self, url: str) -> bool:
        host = self._host(url)
        if not host:
            return False
        # Blocked wins
        if self.blocked:
            for d in self.blocked:
                if (self._is_sub(host, d) if self.include_subdomains else host == d):
                    return False
        # If no allowed set, pass
        if not self.allowed:
            return True
        return any(self._is_sub(host, d) if self.include_subdomains else host == d for d in self.allowed)


class ContentTypeFilter(URLFilter):
    """Fast extension-based content filter.

    Allows typical HTML-like pages, rejects common binary/media/document types.
    """

    # Extensions to reject outright
    _REJECT_EXT = {
        # images
        "jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "tiff", "ico",
        # audio/video
        "mp3", "m4a", "wav", "ogg", "mp4", "mpeg", "webm", "avi", "mov", "flv", "wmv", "mkv",
        # archives/binaries
        "zip", "gz", "tgz", "bz2", "7z", "rar", "exe", "msi", "apk", "dmg", "iso",
        # docs
        "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf",
        # code/packages
        "tar", "jar", "swf",
    }

    # Extensions commonly used for HTML-like dynamic pages that we allow
    _ALLOW_EXT = {"", "html", "htm", "xhtml", "php", "asp", "aspx", "phtml"}

    @staticmethod
    def _ext(url: str) -> str:
        try:
            path = urlparse(url).path
            if not path:
                return ""
            dot = path.rfind(".")
            if dot == -1:
                return ""
            # stop at non-alnum chars
            end = len(path)
            for i in range(dot + 1, len(path)):
                c = path[i]
                if not c.isalnum():
                    end = i
                    break
            return path[dot + 1:end].lower()
        except (TypeError, ValueError):
            return ""

    def apply(self, url: str) -> bool:
        ext = self._ext(url)
        if ext in self._REJECT_EXT:
            return False
        # Allow empty/no extension and known HTML-like extensions
        return ext in self._ALLOW_EXT


@dataclass
class URLPatternFilter(URLFilter):
    """Include/exclude substring patterns.

    - Exclude patterns take precedence.
    - If `include_patterns` is provided (non-empty), require at least one include match.
    - If `include_patterns` is empty, default allow (subject to excludes).
    """

    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None

    def apply(self, url: str) -> bool:
        s = (url or "").lower()
        # Exclude wins
        if self.exclude_patterns:
            for p in self.exclude_patterns:
                if p and p.lower() in s:
                    return False
        # Include gating (optional)
        if self.include_patterns:
            return any(p and p.lower() in s for p in self.include_patterns)
        return True


_RobotsStatus = Literal["allowed", "disallowed", "egress_denied", "unreachable"]


@dataclass(slots=True, frozen=True)
class _RobotsFetchResult:
    """Internal representation of a robots.txt fetch attempt."""

    parser: RobotFileParser | None
    status: Literal["ok", "unreachable"]


@dataclass(slots=True, frozen=True)
class RobotsCheckResult:
    """Structured result for a robots.txt permission check."""

    allowed: bool
    status: _RobotsStatus


class RobotsFilter:
    """Asynchronous robots.txt gate with per-domain cache.

    Notes:
    - This filter is not part of the synchronous FilterChain. Call `await check(...)`
      or `await allowed(...)` explicitly.
    - Compatibility callers can fail open on robots fetch errors; strict callers can
      request fail-closed behavior without depending on exceptions.
    - Honors centralized egress guard unless the caller explicitly reuses a prior
      outbound decision via `skip_egress_check=True`.
    """

    def __init__(
        self,
        user_agent: str,
        *,
        ttl_seconds: int = 1800,
        backend: str = "httpx",
        timeout: float = 5.0,
    ) -> None:
        self.user_agent = user_agent
        self.ttl_seconds = int(ttl_seconds)
        self.backend = backend
        self.timeout = float(timeout)
        # host -> (RobotFileParser|None, fetched_at_epoch_seconds)
        self._cache: dict[str, tuple[RobotFileParser | None, float]] = {}
        # Lock to avoid stampede on first fetch per host
        self._locks: dict[str, asyncio.Lock] = {}

    def _robots_url_for(self, target_url: str) -> str:
        p = urlparse(target_url)
        return f"{p.scheme}://{p.netloc}/robots.txt"

    def _host(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except (TypeError, ValueError):
            return ""

    async def _fetch_parser(self, url: str) -> _RobotsFetchResult:
        host = self._host(url)
        if not host:
            return _RobotsFetchResult(parser=None, status="unreachable")

        # TTL-based cache
        now = time.time()
        cached = self._cache.get(host)
        if cached is not None:
            parser, ts = cached
            if (now - ts) < self.ttl_seconds:
                status: Literal["ok", "unreachable"] = "ok" if parser is not None else "unreachable"
                return _RobotsFetchResult(parser=parser, status=status)

        # Ensure only one fetch in-flight per host
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock
            cached = self._cache.get(host)
            if cached is not None:
                parser, ts = cached
                if (time.time() - ts) < self.ttl_seconds:
                    status: Literal["ok", "unreachable"] = "ok" if parser is not None else "unreachable"
                    return _RobotsFetchResult(parser=parser, status=status)

            robots_url = self._robots_url_for(url)
            try:
                if http_fetch is None:
                    logger.debug("http_fetch not available; allowing by default (no robots fetch)")
                    self._cache[host] = (None, time.time())
                    return _RobotsFetchResult(parser=None, status="unreachable")

                # Use thread offload to keep interface consistent with other code paths
                resp = await asyncio.to_thread(
                    http_fetch,
                    method="GET",
                    url=robots_url,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                if isinstance(resp, dict):
                    text = resp.get("text")
                    status = resp.get("status")
                else:
                    text = getattr(resp, "text", None)
                    status = getattr(resp, "status_code", None)
                if not text or (isinstance(status, int) and status >= 400):
                    # Treat missing/unreadable robots as unreachable and let the caller
                    # decide whether to fail open or closed.
                    self._cache[host] = (None, time.time())
                    return _RobotsFetchResult(parser=None, status="unreachable")
                rp = RobotFileParser()
                rp.parse(text.splitlines())
                self._cache[host] = (rp, time.time())
                return _RobotsFetchResult(parser=rp, status="ok")
            except _WEB_FILTER_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - network/parse errors fail open
                logger.debug(f"Robots fetch failed for host={host}: {e}")
                self._cache[host] = (None, time.time())
                return _RobotsFetchResult(parser=None, status="unreachable")

    async def check(
        self,
        url: str,
        *,
        skip_egress_check: bool = False,
        fail_open: bool = True,
    ) -> RobotsCheckResult:
        """Return a structured robots.txt decision for the target URL.

        Parameters
        ----------
        skip_egress_check:
            Reuse a prior egress decision instead of evaluating policy again.
        fail_open:
            When `True`, unreachable or unreadable robots.txt is treated as allowed.
            When `False`, the same condition is treated as blocked.
        """
        if not skip_egress_check:
            # Import lazily so test monkeypatches on egress.evaluate_url_policy are honored.
            eval_fn = None
            try:
                from tldw_Server_API.app.core.Security import egress as _egress  # local import to honor monkeypatch

                eval_fn = getattr(_egress, "evaluate_url_policy", None)
            except ImportError:
                eval_fn = evaluate_url_policy
            try:
                if eval_fn is not None:
                    pol = eval_fn(url)
                    if not getattr(pol, "allowed", False):
                        return RobotsCheckResult(allowed=False, status="egress_denied")
            except _WEB_FILTER_NONCRITICAL_EXCEPTIONS:
                # On egress evaluation error, fail closed and preserve that distinction for observability.
                return RobotsCheckResult(allowed=False, status="egress_error")

        fetch_result = await self._fetch_parser(url)
        if fetch_result.status != "ok" or fetch_result.parser is None:
            return RobotsCheckResult(allowed=fail_open, status="unreachable")

        try:
            allowed = bool(fetch_result.parser.can_fetch(self.user_agent, url))
        except _WEB_FILTER_NONCRITICAL_EXCEPTIONS:
            return RobotsCheckResult(allowed=fail_open, status="unreachable")
        return RobotsCheckResult(
            allowed=allowed,
            status="allowed" if allowed else "disallowed",
        )

    async def allowed(
        self,
        url: str,
        *,
        skip_egress_check: bool = False,
        fail_open: bool = True,
    ) -> bool:
        """Return `True` when the URL passes robots policy for the requested mode."""
        result = await self.check(
            url,
            skip_egress_check=skip_egress_check,
            fail_open=fail_open,
        )
        return result.allowed
