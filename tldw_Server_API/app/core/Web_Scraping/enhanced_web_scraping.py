# enhanced_web_scraping.py - Production Web Scraping Pipeline
"""
Enhanced web scraping pipeline with production features:
- Concurrent scraping with rate limiting (local and optional ResourceGovernor)
- Job queue management with priority
- Cookie/session management
- Progress tracking and resumability
- Content deduplication
- Robust error handling and retries
- Support for multiple scraping strategies

Persistent caches:
- Cookies and deduplication hashes are stored under `Databases/webscraper/`
  within the project root (created on demand) for durability and easier ops.
"""

import asyncio
import atexit
import contextlib
import hashlib
import json
import os
import re
import time
import uuid
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import aiofiles
import httpx
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import Browser, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.http_client import afetch
from tldw_Server_API.app.core.Ingestion_Media_Processing.logging_safety import (
    exception_type_for_log,
    redact_url_for_log,
)

#
# Import existing components
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
from tldw_Server_API.app.core.Metrics.metrics_logger import (
    log_counter,
)
from tldw_Server_API.app.core.Metrics.metrics_logger import log_gauge as log_gauge
from tldw_Server_API.app.core.Metrics.metrics_logger import log_histogram as log_histogram
from tldw_Server_API.app.core.Security.safe_pickle import safe_pickle_loads
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.Utils import get_database_dir
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    browser_transport_failure_result,
    default_browser_transport_decision,
    resolve_browser_transport_decision,
)
from tldw_Server_API.app.core.Web_Scraping.content import convert_html_to_markdown
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_with_pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction_async import run_extraction_in_thread
from tldw_Server_API.app.core.Web_Scraping.filters import (
    ContentTypeFilter as ContentTypeFilter,
)
from tldw_Server_API.app.core.Web_Scraping.filters import (
    DomainFilter as DomainFilter,
)
from tldw_Server_API.app.core.Web_Scraping.filters import (
    FilterChain as FilterChain,
)
from tldw_Server_API.app.core.Web_Scraping.filters import (
    RobotsFilter as RobotsFilter,
)
from tldw_Server_API.app.core.Web_Scraping.filters import (
    URLPatternFilter as URLPatternFilter,
)
from tldw_Server_API.app.core.Web_Scraping.handlers import resolve_handler
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    WebOutboundPolicyDecision,
)
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    decide_web_outbound_policy as decide_web_outbound_policy,
)
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    decide_web_outbound_policy_sync as decide_web_outbound_policy_sync,
)
from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.scoring import (
    CompositeScorer as CompositeScorer,
)
from tldw_Server_API.app.core.Web_Scraping.scoring import (
    ContentTypeScorer as CTScorer,  # noqa: F401 - historical compatibility export
)
from tldw_Server_API.app.core.Web_Scraping.scoring import (
    DomainAuthorityScorer as DomainAuthorityScorer,
)
from tldw_Server_API.app.core.Web_Scraping.scoring import (
    FreshnessScorer as FreshnessScorer,
)
from tldw_Server_API.app.core.Web_Scraping.scoring import (
    KeywordRelevanceScorer as KeywordRelevanceScorer,
)
from tldw_Server_API.app.core.Web_Scraping.scoring import (
    PathDepthScorer as PathDepthScorer,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_router import DEFAULT_HANDLER, ScraperRouter
from tldw_Server_API.app.core.Web_Scraping.ua_profiles import build_browser_headers, profile_to_impersonate
from tldw_Server_API.app.core.Web_Scraping.url_utils import normalize_for_crawl as normalize_for_crawl


class ArticleRetrievalError(ValueError):
    """A terminal HTTP response that must not be retried using another transport."""


_WEBSCRAPE_TIMEOUT_EXCEPTIONS = (TimeoutError, httpx.TimeoutException, PlaywrightTimeoutError)


def _failed_scrape_result(url: str, error: Exception) -> dict[str, Any]:
    """Preserve typed timeout metadata even when the transport message is empty."""
    result = {"url": url, "error": str(error), "extraction_successful": False}
    if isinstance(error, _WEBSCRAPE_TIMEOUT_EXCEPTIONS) or (
        isinstance(error, NetworkError) and error.classification == "timeout"
    ):
        result["error_code"] = "extraction_timeout"
    return result


_WEBSCRAPE_NONCRITICAL_EXCEPTIONS = (
    *_WEBSCRAPE_TIMEOUT_EXCEPTIONS,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    NetworkError,
    OSError,
    PermissionError,
    RuntimeError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)

# Optional Resource Governor integration (gated by global RG_ENABLED/config)
try:  # pragma: no cover - RG is optional
    from tldw_Server_API.app.core.config import rg_enabled  # type: ignore
    from tldw_Server_API.app.core.Resource_Governance import (  # type: ignore
        MemoryResourceGovernor,
        RedisResourceGovernor,
        RGRequest,
    )
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import (  # type: ignore
        PolicyLoader,
        PolicyReloadConfig,
        default_policy_loader,
    )
except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - safe fallback when RG not installed
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    RGRequest = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    default_policy_loader = None  # type: ignore
    rg_enabled = None  # type: ignore

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BEST_FIRST_BATCH_SIZE = 10


_ENHANCED_POLICY_CHECKER = DefaultWebOutboundPolicyChecker()

# Stable skip reasons for crawl observability. Keep this list small and
# explicit to prevent accidental metric cardinality growth.
CRAWL_SKIP_REASON_VISITED = "visited"
CRAWL_SKIP_REASON_MAX_DEPTH = "max_depth"
CRAWL_SKIP_REASON_DUPLICATE = "duplicate"
CRAWL_SKIP_REASON_FILTER_CHAIN = "filter_chain"
CRAWL_SKIP_REASON_PATH_DEPTH = "path_depth"
CRAWL_SKIP_REASON_ROBOTS = "robots"
CRAWL_SKIP_REASON_POLICY = "policy"
CRAWL_SKIP_REASON_BELOW_THRESHOLD = "below_threshold"
CRAWL_SKIP_REASON_CUSTOM_FILTER = "custom_filter"
CRAWL_SKIP_REASON_OTHER = "other"
CRAWL_SKIP_REASON_LABELS = {
    CRAWL_SKIP_REASON_VISITED,
    CRAWL_SKIP_REASON_MAX_DEPTH,
    CRAWL_SKIP_REASON_DUPLICATE,
    CRAWL_SKIP_REASON_FILTER_CHAIN,
    CRAWL_SKIP_REASON_PATH_DEPTH,
    CRAWL_SKIP_REASON_ROBOTS,
    CRAWL_SKIP_REASON_POLICY,
    CRAWL_SKIP_REASON_BELOW_THRESHOLD,
    CRAWL_SKIP_REASON_CUSTOM_FILTER,
}


def _record_robot_policy_block(url: str, reason: str) -> None:
    if not reason.startswith("robots_"):
        return
    try:
        parsed = urlparse(url)
        increment_counter("scrape_blocked_by_robots_total", labels={"domain": parsed.netloc})
    except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
        increment_counter("scrape_blocked_by_robots_total", labels={})


def _blocked_scrape_result(url: str, decision: WebOutboundPolicyDecision) -> dict[str, Any]:
    _record_robot_policy_block(url, decision.reason)
    if decision.reason.startswith("robots_"):
        error = "Blocked by outbound policy"
    else:
        error = f"Egress denied: {decision.reason}"
    return {
        "url": url,
        "error": error,
        "extraction_successful": False,
        "policy_reason": decision.reason,
        "policy_mode": decision.mode,
        "policy_stage": decision.stage,
        "policy_source": decision.source,
    }


def _policy_skip_reason(decision: WebOutboundPolicyDecision) -> str:
    if decision.reason.startswith("robots_"):
        return CRAWL_SKIP_REASON_ROBOTS
    return CRAWL_SKIP_REASON_POLICY


def _default_rules_path() -> str:
    here = Path(__file__).resolve()
    # Resolve project root: .../tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py -> repo root
    project_root = here.parents[4]  # tldw_Server_API/ at index 3; repo root at 4
    return str(project_root / "tldw_Server_API" / "Config_Files" / "custom_scrapers.yaml")


from .contracts.execution import CrawlReport, DiscoveryResult, PageData
from .jobs.models import JobPriority, ScrapingJob
from .jobs.models import JobStatus as JobStatus
from .jobs.queue import ProcessLocalJobQueue
from .orchestration.crawl import execute_crawl
from .runtime.controls import PhysicalWork, WorkHandle, owned_to_thread, physical_work

_WEB_SCRAPING_DEPRECATION_WARNED = False


def _emit_web_scraping_legacy_deprecation(context: str) -> None:
    global _WEB_SCRAPING_DEPRECATION_WARNED
    if _WEB_SCRAPING_DEPRECATION_WARNED:
        return
    _WEB_SCRAPING_DEPRECATION_WARNED = True
    msg = (
        "Web scraping legacy rate limiter is deprecated (Phase 2). "
        f"Context: {context}. Enable RG_ENABLED=true for enforcement. "
        "This shim will be removed in a future release."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)


class RateLimiter:
    """Rate limiting for scraping requests.

    **Phase 2 Deprecation Notice**:
    Primary enforcement is handled by ``RGSimpleMiddleware`` and the per-module
    ``_maybe_enforce_with_rg_web_scraping``. When RG is unavailable or disabled,
    this shim fails open (no sleeps or counters). This shim will be removed in a
    future release.
    """

    def __init__(
        self, max_requests_per_second: float = 2.0, max_requests_per_minute: int = 60, max_requests_per_hour: int = 1000
    ):
        # Parameters preserved for API compatibility; no internal state.
        self.max_rps = max_requests_per_second
        self.max_rpm = max_requests_per_minute
        self.max_rph = max_requests_per_hour
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            rg_active = _rg_web_scraping_enabled()
            rg_decision = await _maybe_enforce_with_rg_web_scraping() if rg_active else None
            if rg_active:
                if rg_decision is None:
                    _log_rg_web_fallback("rg_decision_unavailable")
                    _emit_web_scraping_legacy_deprecation("rg_decision_unavailable")
                elif not rg_decision["allowed"]:
                    retry_after = rg_decision.get("retry_after") or 1
                    logger.info(
                        "Web scraping request delayed by ResourceGovernor: retry_after={}s",
                        retry_after,
                    )
                    # For scraping, we model RG denials as backoff rather than hard 429s.
                    await asyncio.sleep(retry_after)
                return

            # RG disabled: fail-open (no sleeps, no counters).
            _emit_web_scraping_legacy_deprecation("rg_disabled")


_rg_web_governor = None
_rg_web_loader = None
_rg_web_lock = asyncio.Lock()
_rg_web_fallback_logged = False


def _log_rg_web_fallback(reason: str) -> None:
    global _rg_web_fallback_logged
    if _rg_web_fallback_logged:
        return
    _rg_web_fallback_logged = True
    logger.warning(
        "Web scraping ResourceGovernor unavailable; using diagnostics-only compatibility shim (no sleeps/counters). "
        "reason={}",
        reason,
    )


def _rg_web_scraping_enabled() -> bool:
    """Return True when RG should gate web scraping requests."""
    if rg_enabled is not None:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
            return False
    return False


async def _get_web_scraping_rg_governor():
    """Lazily initialize a ResourceGovernor instance for web scraping."""
    global _rg_web_governor, _rg_web_loader
    if not _rg_web_scraping_enabled():
        return None
    if RGRequest is None or PolicyLoader is None:
        return None
    if _rg_web_governor is not None:
        return _rg_web_governor
    async with _rg_web_lock:
        if _rg_web_governor is not None:
            return _rg_web_governor
        try:
            loader = (
                default_policy_loader()
                if default_policy_loader
                else PolicyLoader(
                    os.getenv(
                        "RG_POLICY_PATH",
                        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
                    ),
                    PolicyReloadConfig(
                        enabled=True,
                        interval_sec=int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10"),
                    ),
                )
            )
            await loader.load_once()
            _rg_web_loader = loader
            backend = os.getenv("RG_BACKEND", "memory").lower()
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_web_governor = gov
            return gov
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - optional path
            logger.debug(
                "Web scraping RG governor init failed; using diagnostics-only compatibility shim: {}",
                exc,
            )
            return None


async def _maybe_enforce_with_rg_web_scraping() -> Optional[dict[str, object]]:
    """
    Optionally enforce web scraping request limits via ResourceGovernor.

    Returns a decision dict when RG is used, or None when RG is
    unavailable or disabled.
    """
    gov = await _get_web_scraping_rg_governor()
    if gov is None:
        return None
    policy_id = os.getenv("RG_WEB_SCRAPING_POLICY_ID", "web_scraping.default")
    op_id = f"web-scrape-{time.time_ns()}"
    try:
        decision, handle = await gov.reserve(
            RGRequest(
                entity="service:web_scraping",
                categories={"requests": {"units": 1}},
                tags={"policy_id": policy_id, "module": "web_scraping"},
            ),
            op_id=op_id,
        )
        if decision.allowed:
            if handle:
                try:
                    await gov.commit(handle, None, op_id=op_id)
                except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Web scraping RG commit failed")
            return {"allowed": True, "retry_after": None, "policy_id": policy_id}
        return {
            "allowed": False,
            "retry_after": decision.retry_after or 1,
            "policy_id": policy_id,
        }
    except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Web scraping RG reserve failed: {}", exc)
        return None


class CookieManager:
    """Manages cookies for scraping."""

    def __init__(self, storage_path: Optional[Path] = None, *, connector_limit: int = 10, per_host_limit: int = 2):
        if storage_path is None:
            base = Path(get_database_dir()) / "webscraper"
            base.mkdir(parents=True, exist_ok=True)
            storage_path = base / "cookies.json"
        self.storage_path = storage_path
        self._cookies: dict[str, list[dict[str, Any]]] = {}
        self._connector_limit = int(connector_limit)
        self._per_host_limit = int(per_host_limit)
        self._load_cookies()

    def _load_cookies(self):
        """Load cookies from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    self._cookies = json.load(f)
                logger.info(f"Loaded cookies for {len(self._cookies)} domains")
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
                logger.error("Failed to load cookies ({})", exception_type_for_log(e))

    def _save_cookies(self):
        """Save cookies to storage"""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self._cookies, f, indent=2)
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to save cookies ({})", exception_type_for_log(e))

    def add_cookies(self, domain: str, cookies: list[dict[str, Any]]):
        """Add cookies for a domain"""
        self._cookies[domain] = cookies
        self._save_cookies()

    def get_cookies(self, url: str) -> Optional[list[dict[str, Any]]]:
        """Get cookies for a URL"""
        domain = urlparse(url).netloc
        return self._cookies.get(domain)

    def _snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Detach the trusted jar once; never serialize it into a job or report."""
        return deepcopy(self._cookies)


class ContentDeduplicator:
    """Handles content deduplication"""

    _LEGACY_PICKLE_ENV_VAR = "WEBSCRAPER_ALLOW_LEGACY_PICKLE_HASHES"

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            base = Path(get_database_dir()) / "webscraper"
            base.mkdir(parents=True, exist_ok=True)
            storage_path = base / "content_hashes.json"
        self.storage_path = Path(storage_path)
        self._legacy_storage_path = self.storage_path.with_suffix(".pkl")
        self._allow_legacy_pickle_hashes = is_truthy(os.getenv(self._LEGACY_PICKLE_ENV_VAR, "false"))
        self._hashes: dict[str, dict[str, Any]] = {}
        self._load_hashes()

    def _load_hashes(self):
        """Load content hashes from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                self._hashes = self._normalize_hash_data(loaded)
                logger.info(f"Loaded {len(self._hashes)} content hashes")
                return
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
                logger.error("Failed to load hashes ({})", exception_type_for_log(e))
                # If the configured path itself is the legacy .pkl file, continue
                # to optional compatibility loading below.
                if self.storage_path != self._legacy_storage_path:
                    return

        if self._legacy_storage_path.exists():
            if not self._allow_legacy_pickle_hashes:
                logger.warning(
                    "Legacy dedupe hash file detected at {} but loading is disabled. Set {}=true to migrate it.",
                    self._legacy_storage_path,
                    self._LEGACY_PICKLE_ENV_VAR,
                )
                return

            try:
                with open(self._legacy_storage_path, "rb") as f:
                    loaded = safe_pickle_loads(f.read())
                self._hashes = self._normalize_hash_data(loaded)
                self._save_hashes()
                logger.info(
                    "Loaded and migrated {} legacy content hashes from {}",
                    len(self._hashes),
                    self._legacy_storage_path,
                )
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
                logger.error("Failed to load legacy hashes ({})", exception_type_for_log(e))

    def _save_hashes(self):
        """Save content hashes to storage"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.storage_path.with_name(f"{self.storage_path.name}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._hashes, f, ensure_ascii=True, indent=2)
            tmp_path.replace(self.storage_path)
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to save hashes ({})", exception_type_for_log(e))

    def _normalize_hash_data(self, data: Any) -> dict[str, dict[str, Any]]:
        """Normalize persisted hash data into the expected dictionary shape."""
        if not isinstance(data, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue

            normalized[key] = {
                "url": str(value.get("url", "")),
                "title": str(value.get("title", "")),
                "first_seen": str(value.get("first_seen", "")),
                "last_seen": str(value.get("last_seen", "")),
            }

        return normalized

    def compute_hash(self, content: str) -> str:
        """Compute hash for content"""
        # Normalize content before hashing
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, url: str, content: str) -> bool:
        """Check if content is duplicate"""
        content_hash = self.compute_hash(content)

        # Check if exact hash exists
        if content_hash in self._hashes:
            existing = self._hashes[content_hash]
            if existing["url"] != url:
                logger.info(
                    "Duplicate content found: {} duplicates {}",
                    redact_url_for_log(url),
                    redact_url_for_log(existing["url"]),
                )
                return True

        return False

    def add_content(self, url: str, content: str, title: str = ""):
        """Add content hash to deduplication store"""
        content_hash = self.compute_hash(content)
        self._hashes[content_hash] = {
            "url": url,
            "title": title,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        }
        self._save_hashes()

    def update_seen(self, content_hash: str):
        """Update last seen time for content"""
        if content_hash in self._hashes:
            self._hashes[content_hash]["last_seen"] = datetime.now().isoformat()
            self._save_hashes()

    def flush(self):
        """Force a save to disk"""
        try:
            self._save_hashes()
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to flush content hashes ({})", exception_type_for_log(e))


def _owned_adapter_page(operation: Callable[[], Awaitable[PageData]]) -> WorkHandle[PageData]:
    """Track compatibility callbacks and every physical operation they register."""
    loop = asyncio.get_running_loop()
    owner = PhysicalWork()

    async def run():
        token = physical_work.set(owner)
        try:
            return await operation()
        finally:
            physical_work.reset(token)

    task = loop.create_task(run())
    task.add_done_callback(lambda _done: owner.finish())
    return WorkHandle(task, owner.drained, lambda: loop.call_soon_threadsafe(task.cancel))


class ScrapingJobQueue(ProcessLocalJobQueue):
    """Signature-preserving adapter over the single bounded process-local queue."""

    def __init__(self, max_concurrent: int = 5, parent_scraper=None, *, completed_retention: int = 1000):
        self.parent_scraper = parent_scraper
        super().__init__(
            execute=self._start_job,
            max_concurrent=max_concurrent,
            completed_retention=completed_retention,
        )

    def _start_job(self, payload, controls):
        """Call the supported live callback using the admitted private snapshot."""
        # Capture a replaced callback at execution; no public job retains credentials.
        callback = self._execute_job
        parent = self._canonical_parent(callback)
        if controls is not None and parent is not None:
            return parent._start_controlled_job(payload, controls)

        async def run():
            if controls is not None:
                controls.check()
                reservation = controls.reserve("page_attempts")
                try:
                    reservation.commit()
                finally:
                    reservation.close()
            job = ScrapingJob(
                payload.job_id,
                payload.url,
                payload.operation,
                user_id=payload.owner,
                metadata=payload.options,
            )
            result = await callback(job)
            return PageData(result, DiscoveryResult(payload.url, payload.url, ()))

        return _owned_adapter_page(run)

    def _canonical_parent(self, callback=None):
        """Recognize only the unreplaced default pair; opaque callbacks stay opaque."""
        parent = self.parent_scraper
        execute = callback if callback is not None else self._execute_job
        scrape = getattr(parent, "scrape_article", None)
        if (
            getattr(execute, "__self__", None) is self
            and getattr(execute, "__func__", None) is _DEFAULT_QUEUE_EXECUTE
            and getattr(scrape, "__self__", None) is parent
            and getattr(scrape, "__func__", None) is _DEFAULT_ENHANCED_SCRAPE
        ):
            return parent
        return None

    async def submit_page(self, job, *, controls, execute=None):
        """Keep instance rate waits inside accepted ownership before page dispatch."""
        parent = self._canonical_parent() if execute is None else None
        if parent is not None:
            job = replace(job, metadata=parent._controlled_job_options(job.url, job.metadata))
        if execute is None or self.parent_scraper is None:
            return await super().submit_page(job, controls=controls, execute=execute)
        limiter = getattr(self.parent_scraper, "rate_limiter", None)
        if limiter is None:
            return await super().submit_page(job, controls=controls, execute=execute)

        def limited(payload, parent):
            async def run():
                await limiter.acquire()
                parent.check()
                handle = execute(payload, parent)
                physical_work.get().observe(handle.drained)
                try:
                    return await asyncio.shield(handle.result)
                except BaseException:
                    handle.cancel()
                    raise

            return _owned_adapter_page(run)

        return await super().submit_page(job, controls=controls, execute=limited)

    async def _execute_job(self, job: ScrapingJob) -> dict[str, Any]:
        """Preserve duck-typed parent callbacks, or use standalone canonical scraping."""
        if self.parent_scraper is not None:
            return await self.parent_scraper.scrape_article(
                job.url,
                job.method,
                custom_cookies=job.metadata.get("custom_cookies"),
                user_agent=job.metadata.get("user_agent"),
                custom_headers=job.metadata.get("custom_headers"),
                allow_llm_extraction=job.metadata.get("allow_llm_extraction", True),
            )
        else:
            from tldw_Server_API.app.core.Web_Scraping.orchestration import scrape_article

            return await scrape_article(
                job.url,
                custom_cookies=job.metadata.get("custom_cookies"),
                allow_llm_extraction=job.metadata.get("allow_llm_extraction", True),
            )

    async def _drain(self) -> None:
        """After closing admission, retain instance resources through actual drain."""
        await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._tasks)), return_exceptions=True)

    def _record_completed_job(self, job: ScrapingJob) -> None:
        """Forward the historical projection helper to canonical bounded history."""
        job.context = None
        self._history.record(job, job.result)


class EnhancedWebScraper:
    """Main enhanced web scraping class"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        # Respect an explicitly provided (even empty) config; only fall back to
        # on-disk config when config is None.
        raw_cfg = config if config is not None else load_and_log_configs().get("web_scraper", {})

        # Normalize config values
        def _as_float(v, d):
            try:
                return float(v)
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                return float(d)

        def _as_int(v, d):
            try:
                return int(v)
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                return int(d)

        self.config = raw_cfg

        # Initialize components
        self.rate_limiter = RateLimiter(
            max_requests_per_second=_as_float(self.config.get("max_rps", 2.0), 2.0),
            max_requests_per_minute=_as_int(self.config.get("max_rpm", 60), 60),
            max_requests_per_hour=_as_int(self.config.get("max_rph", 1000), 1000),
        )
        connector_limit = _as_int(self.config.get("connector_limit", 10), 10)
        per_host_limit = _as_int(self.config.get("connector_limit_per_host", 2), 2)
        self.cookie_manager = CookieManager(connector_limit=connector_limit, per_host_limit=per_host_limit)
        self.deduplicator = ContentDeduplicator()
        max_completed_jobs = _as_int(self.config.get("max_completed_jobs", 1000), 1000)
        self.job_queue = ScrapingJobQueue(
            max_concurrent=_as_int(self.config.get("max_concurrent", 5), 5),
            parent_scraper=self,  # Pass self reference to job queue
            completed_retention=max_completed_jobs,
        )

        # Playwright browser
        self._browser: Optional[Browser] = None
        self._playwright = None

        # Progress tracking
        self._progress: dict[str, Any] = defaultdict(dict)

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return is_truthy(value)
        return default

    @staticmethod
    def _normalize_cookie_map(custom_cookies: Optional[list[dict[str, Any]]]) -> dict[str, str]:
        cookies_map: dict[str, str] = {}
        if not custom_cookies:
            return cookies_map
        for cookie in custom_cookies:
            if not isinstance(cookie, dict):
                continue
            if "name" in cookie and "value" in cookie:
                cookies_map[str(cookie["name"])] = str(cookie["value"])
            else:
                for key, value in cookie.items():
                    cookies_map[str(key)] = str(value)
        return cookies_map

    @staticmethod
    def _merge_cookie_maps(
        custom_cookies: Optional[list[dict[str, Any]]],
        plan_cookies: Optional[dict[str, str]],
    ) -> list[dict[str, Any]]:
        merged = EnhancedWebScraper._normalize_cookie_map(custom_cookies)
        if plan_cookies:
            merged.update({str(k): str(v) for k, v in plan_cookies.items()})
        return [{"name": k, "value": v} for k, v in merged.items()]

    @staticmethod
    def _normalize_playwright_cookies(
        url: str,
        custom_cookies: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        cookies_list: list[dict[str, Any]] = []
        if not custom_cookies:
            return cookies_list
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else url
        for cookie in custom_cookies:
            if not isinstance(cookie, dict):
                continue
            if "name" in cookie and "value" in cookie:
                cookies_list.append(
                    {
                        "name": str(cookie["name"]),
                        "value": str(cookie["value"]),
                        "url": base_url,
                    }
                )
            else:
                for key, value in cookie.items():
                    cookies_list.append(
                        {
                            "name": str(key),
                            "value": str(value),
                            "url": base_url,
                        }
                    )
        return cookies_list

    @staticmethod
    def _parse_proxy_for_playwright(proxy_url: str) -> Optional[dict[str, str]]:
        if not proxy_url:
            return None
        pattern = (
            r"^(?:(?P<scheme>\\w+)://)?(?:(?P<username>[^:@]+):(?P<password>[^@]+)@)?(?P<host>[^:]+):(?P<port>\\d+)$"
        )
        match = re.match(pattern, proxy_url)
        if not match:
            return None
        scheme = match.group("scheme") or "http"
        host = match.group("host")
        port = match.group("port")
        if not host or not port:
            return None
        result: dict[str, str] = {"server": f"{scheme}://{host}:{port}"}
        username = match.group("username")
        password = match.group("password")
        if username and password:
            result["username"] = username
            result["password"] = password
        return result

    @staticmethod
    def _build_request_headers(
        user_agent: Optional[str],
        custom_headers: Optional[dict[str, str]],
    ) -> dict[str, str]:
        header_copy = dict(custom_headers) if custom_headers else {}
        effective_user_agent = user_agent or header_copy.pop("User-Agent", None) or DEFAULT_USER_AGENT
        headers = {"User-Agent": effective_user_agent}
        if header_copy:
            headers.update(header_copy)
        return headers

    @staticmethod
    def _build_plan_headers(
        ua_profile: str,
        plan_headers: Optional[dict[str, str]],
        user_agent: Optional[str],
        custom_headers: Optional[dict[str, str]],
    ) -> dict[str, str]:
        headers = build_browser_headers(ua_profile, accept_lang="en-US,en;q=0.9")
        if plan_headers:
            headers.update({str(k): str(v) for k, v in plan_headers.items()})
        if custom_headers:
            headers.update({str(k): str(v) for k, v in custom_headers.items()})
        if user_agent:
            headers["User-Agent"] = user_agent
        return headers

    def _build_cookie_map(
        self,
        url: str,
        custom_cookies: Optional[list[dict[str, Any]]],
    ) -> Optional[dict[str, str]]:
        cookies: dict[str, str] = {}
        stored = self.cookie_manager.get_cookies(url)
        if stored:
            cookies.update(self._normalize_cookie_map(stored))
        custom_map = self._normalize_cookie_map(custom_cookies)
        if custom_map:
            cookies.update(custom_map)
        return cookies or None

    def _resolve_scrape_plan(
        self, url: str, *, configuration: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], str, str]:
        ws_cfg = (self.config if configuration is None else configuration) or {}
        rules_path = ws_cfg.get("custom_scrapers_yaml_path", _default_rules_path())
        rules = ScraperRouter.load_rules_from_yaml(rules_path)
        ua_mode = str(ws_cfg.get("web_scraper_ua_mode", "fixed") or "fixed")
        respect_robots_default = ws_cfg.get("web_scraper_respect_robots", True)
        if isinstance(respect_robots_default, str):
            respect_robots_default = is_truthy(respect_robots_default)
        router = ScraperRouter(rules, ua_mode=ua_mode, default_respect_robots=bool(respect_robots_default))
        plan = router.resolve(url)

        backend_choice = str(getattr(plan, "backend", "auto") or "auto").lower().strip()
        if backend_choice not in {"auto", "curl", "httpx", "playwright"}:
            backend_choice = "auto"
        if backend_choice == "auto":
            default_backend = ws_cfg.get("web_scraper_default_backend")
            if isinstance(default_backend, str):
                backend_choice = default_backend.lower().strip() or "auto"
                if backend_choice not in {"auto", "curl", "httpx", "playwright"}:
                    backend_choice = "auto"

        handler_path = str(getattr(plan, "handler", "") or "")
        return plan, backend_choice, handler_path

    def _apply_dedup(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("extraction_successful"):
            return data
        content = data.get("content", "") or ""
        title = data.get("title", "") or ""
        if self.deduplicator.is_duplicate(url, content):
            return {
                "url": url,
                "error": "Duplicate content",
                "extraction_successful": False,
                "is_duplicate": True,
            }
        self.deduplicator.add_content(url, content, title)
        return data

    def _emit_scrape_metrics(
        self,
        *,
        backend: str,
        outcome: str,
        elapsed_s: Optional[float] = None,
        content: Optional[str] = None,
    ) -> None:
        try:
            if elapsed_s is not None:
                observe_histogram("scrape_fetch_latency_seconds", elapsed_s, labels={"backend": backend})
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
            pass
        with contextlib.suppress(_WEBSCRAPE_NONCRITICAL_EXCEPTIONS):
            increment_counter("scrape_fetch_total", labels={"backend": backend, "outcome": outcome})
        if outcome == "success" and content:
            with contextlib.suppress(_WEBSCRAPE_NONCRITICAL_EXCEPTIONS):
                observe_histogram(
                    "scrape_content_length_bytes",
                    len(content.encode("utf-8", errors="ignore")),
                    labels={"backend": backend},
                )

    def _extract_from_html_with_pipeline(
        self,
        html: str,
        url: str,
        *,
        strategy_order: Optional[list[str]] = None,
        handler: Optional[Any] = None,
        postprocess_markdown: bool = False,
        method_label: str = "trafilatura",
        fallback_extractor: Optional[Callable[[str, str], dict[str, Any]]] = None,
        schema_rules: Optional[dict[str, Any]] = None,
        llm_settings: Optional[dict[str, Any]] = None,
        regex_settings: Optional[dict[str, Any]] = None,
        cluster_settings: Optional[dict[str, Any]] = None,
        allow_llm_extraction: bool = True,
    ) -> dict[str, Any]:
        data = extract_article_with_pipeline(
            html,
            url,
            strategy_order=strategy_order,
            handler=handler,
            fallback_extractor=fallback_extractor,
            schema_rules=schema_rules,
            llm_settings=llm_settings,
            regex_settings=regex_settings,
            cluster_settings=cluster_settings,
            allow_llm_extraction=allow_llm_extraction,
        )
        if postprocess_markdown and data.get("extraction_successful") and data.get("content"):
            data["content"] = convert_html_to_markdown(data["content"])
        data.setdefault("method", method_label)
        return data

    async def _fetch_html(
        self,
        url: str,
        *,
        headers: dict[str, str],
        cookies: Optional[dict[str, str]],
        backend: str,
        impersonate: Optional[str],
        proxies: Optional[dict[str, str]],
    ) -> tuple[str, str, float]:
        if backend == "curl":
            try:
                t0 = time.time()
                html = await owned_to_thread(
                    self._fetch_html_curl,
                    url,
                    headers=headers,
                    cookies=cookies,
                    timeout=15.0,
                    impersonate=impersonate,
                    proxies=proxies,
                )
                elapsed = max(0.0, time.time() - t0)
                return html, "curl", elapsed
            except ArticleRetrievalError:
                raise
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("curl backend failed; falling back to httpx ({})", exception_type_for_log(exc))
        t0 = time.time()
        resp = None
        try:
            resp = await afetch(
                method="GET",
                url=url,
                headers=headers,
                cookies=cookies,
                proxies=proxies,
            )
            response_status = int(getattr(resp, "status_code", getattr(resp, "status", 0)))
            if not 200 <= response_status < 300:
                raise ArticleRetrievalError(
                    f"Article retrieval failed: HTTP {response_status} (expected terminal 2xx response)"
                )
        finally:
            await self._close_response(resp)
        backend_used = "httpx"
        try:
            module_name = getattr(resp, "__class__", type(resp)).__module__ or ""
            if module_name.startswith("aiohttp"):
                backend_used = "aiohttp"
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
            backend_used = "httpx"
        return resp.text, backend_used, max(0.0, time.time() - t0)

    @staticmethod
    def _fetch_html_curl(
        url: str,
        *,
        headers: dict[str, str],
        cookies: Optional[dict[str, str]],
        timeout: float,
        impersonate: Optional[str],
        proxies: Optional[dict[str, str]],
    ) -> str:
        from tldw_Server_API.app.core import http_client as _http_client

        resp = _http_client.fetch(
            url,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            proxies=proxies,
            backend="curl",
            impersonate=impersonate,
            follow_redirects=True,
        )
        status = int(resp.get("status", 0))
        if 200 <= status < 300:
            return resp["text"]
        raise ArticleRetrievalError(f"curl fetch did not reach a terminal 2xx response (status={status})")

    @staticmethod
    async def _close_response(resp: Any) -> None:
        if resp is None:
            return
        close = getattr(resp, "aclose", None)
        if callable(close):
            await close()
            return
        close = getattr(resp, "close", None)
        if callable(close):
            close()

    def _set_progress(self, task_id: Optional[str], **updates: Any) -> None:
        if not task_id:
            return
        entry = dict(self._progress.get(task_id, {}))
        entry.update(updates)
        entry["updated_at"] = datetime.now().isoformat()
        self._progress[task_id] = entry

    async def start(self):
        """Start the scraper"""
        await self.job_queue.start()

        transport = resolve_browser_transport_decision(
            default_browser_transport_decision,
            component="enhanced_web_scraper",
        )
        if not transport.allowed:
            logger.bind(
                component="enhanced_web_scraper",
                operation="start_browser",
                reason=transport.reason,
            ).warning("Playwright browser startup skipped by transport policy.")
        else:
            try:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"]
                )
                logger.info("Playwright browser initialized successfully")
            except ImportError:
                logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")
                logger.warning("Web scraping will proceed without JavaScript rendering support")
                self._playwright = None
                self._browser = None
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
                logger.error("Failed to initialize Playwright browser ({})", exception_type_for_log(e))
                logger.warning("Web scraping will proceed without JavaScript rendering support")
                self._playwright = None
                self._browser = None

        # Ensure dedup flush on process exit
        atexit.register(lambda: self.deduplicator.flush())
        logger.info("Enhanced web scraper started")

    async def stop(self):
        """Stop the scraper"""
        await self.job_queue.stop()
        await self.job_queue._drain()

        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        # Final flush
        self.deduplicator.flush()

        logger.info("Enhanced web scraper stopped")

    async def scrape_article(
        self,
        url: str,
        method: str = "auto",
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        allow_llm_extraction: bool = True,
    ) -> dict[str, Any]:
        """Scrape a single article with specified method"""
        # Apply rate limiting
        await self.rate_limiter.acquire()

        preflight_payload = None

        def _attach_preflight(result: dict[str, Any]) -> dict[str, Any]:
            if preflight_payload and isinstance(result, dict):
                result.setdefault("preflight_analysis", preflight_payload)
            return result

        try:
            plan, backend_choice, handler_path = await owned_to_thread(
                self._resolve_scrape_plan,
                url,
            )
            handler_path = str(handler_path or "")
            is_default_handler = (not handler_path) or handler_path == DEFAULT_HANDLER
            handler_func = resolve_handler(handler_path) if not is_default_handler else None
            strategy_order = getattr(plan, "strategy_order", None)
            schema_rules = getattr(plan, "schema_rules", None)
            llm_settings = getattr(plan, "llm_settings", None)
            regex_settings = getattr(plan, "regex_settings", None)
            cluster_settings = getattr(plan, "cluster_settings", None)
            postprocess_markdown = is_default_handler

            merged_cookies = self._merge_cookie_maps(custom_cookies, getattr(plan, "cookies", {}))
            headers = self._build_plan_headers(
                getattr(plan, "ua_profile", "chrome_120_win"),
                getattr(plan, "extra_headers", {}),
                user_agent,
                custom_headers,
            )
            if backend_choice in {"httpx", "auto"}:
                try:
                    from tldw_Server_API.app.core import http_client as _http_client

                    headers = _http_client._sanitize_accept_encoding_for_backend(headers, "httpx")  # type: ignore[attr-defined]
                except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                    pass

            options = preflight_facade.PreflightOptions.from_mapping(self.config or {})
            try:
                target = await preflight_facade.evaluate_target(
                    url,
                    respect_robots=bool(getattr(plan, "respect_robots", True)),
                    user_agent=headers.get("User-Agent", DEFAULT_USER_AGENT),
                    request_context=RuntimeRequestContext(source="enhanced_scrape", stage="pre_fetch"),
                    config={"web_scraper": self.config or {}},
                    policy_checker=_ENHANCED_POLICY_CHECKER,
                )
            except asyncio.CancelledError:
                raise
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                logger.error("Outbound policy evaluation failed.")
                return _attach_preflight(
                    {
                        "url": url,
                        "error": "Outbound policy evaluation failed. Please contact system administrator.",
                        "extraction_successful": False,
                    }
                )

            if not target.decision.allowed:
                return _attach_preflight(_blocked_scrape_result(url, target.decision))

            preflight_result = None
            if options.enabled:
                context = preflight_facade.build_execution_context(
                    target,
                    options,
                    policy_checker=_ENHANCED_POLICY_CHECKER,
                )
                preflight_result = await preflight_facade.run_preflight(target, options, context)
            backend_setting = str(getattr(plan, "backend", "auto") or "auto").lower().strip()
            backend_choice, method, preflight_result = preflight_facade.apply_preflight_advice(
                preflight_result,
                backend=backend_choice,
                method=method,
                backend_setting=backend_setting,
            )
            preflight_notes = list(preflight_result.advice.notes) if preflight_result is not None else []
            if preflight_notes:
                logger.debug(f"Preflight advice: {preflight_notes}")
            preflight_payload = preflight_facade.public_preflight_payload(
                preflight_result,
                options.include_results,
            )

            effective_method = method
            if backend_choice == "playwright":
                effective_method = "playwright"
            elif effective_method == "auto":
                effective_method = "trafilatura"

            if effective_method == "trafilatura":
                result = await self._scrape_with_trafilatura(
                    url,
                    merged_cookies,
                    user_agent=None,
                    custom_headers=headers,
                    backend=backend_choice,
                    impersonate=getattr(plan, "impersonate", profile_to_impersonate(getattr(plan, "ua_profile", ""))),
                    proxies=getattr(plan, "proxies", None),
                    handler=handler_func,
                    strategy_order=strategy_order,
                    postprocess_markdown=postprocess_markdown,
                    schema_rules=schema_rules,
                    llm_settings=llm_settings,
                    regex_settings=regex_settings,
                    cluster_settings=cluster_settings,
                    allow_llm_extraction=allow_llm_extraction,
                )
                return _attach_preflight(result)
            elif effective_method == "playwright":
                result = await self._scrape_with_playwright(
                    url,
                    merged_cookies,
                    user_agent=None,
                    custom_headers=headers,
                    proxies=getattr(plan, "proxies", None),
                    handler=handler_func,
                    strategy_order=strategy_order,
                    postprocess_markdown=postprocess_markdown,
                    schema_rules=schema_rules,
                    llm_settings=llm_settings,
                    regex_settings=regex_settings,
                    cluster_settings=cluster_settings,
                    allow_llm_extraction=allow_llm_extraction,
                )
                return _attach_preflight(result)
            elif effective_method == "beautifulsoup":
                result = await self._scrape_with_beautifulsoup(
                    url,
                    merged_cookies,
                    user_agent=None,
                    custom_headers=headers,
                    backend=backend_choice,
                    impersonate=getattr(plan, "impersonate", profile_to_impersonate(getattr(plan, "ua_profile", ""))),
                    proxies=getattr(plan, "proxies", None),
                    handler=handler_func,
                    strategy_order=strategy_order,
                    postprocess_markdown=postprocess_markdown,
                    schema_rules=schema_rules,
                    llm_settings=llm_settings,
                    regex_settings=regex_settings,
                    cluster_settings=cluster_settings,
                    allow_llm_extraction=allow_llm_extraction,
                )
                return _attach_preflight(result)
            else:
                raise ValueError(f"Unknown scraping method: {effective_method}")

        except asyncio.CancelledError:
            raise
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to scrape {} ({})", redact_url_for_log(url), exception_type_for_log(e))
            return _attach_preflight(_failed_scrape_result(url, e))

    async def _scrape_with_trafilatura(
        self,
        url: str,
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        backend: str = "httpx",
        impersonate: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
        handler: Optional[Any] = None,
        strategy_order: Optional[list[str]] = None,
        postprocess_markdown: bool = False,
        schema_rules: Optional[dict[str, Any]] = None,
        llm_settings: Optional[dict[str, Any]] = None,
        regex_settings: Optional[dict[str, Any]] = None,
        cluster_settings: Optional[dict[str, Any]] = None,
        allow_llm_extraction: bool = True,
    ) -> dict[str, Any]:
        """Scrape using trafilatura"""
        headers = self._build_request_headers(user_agent, custom_headers)
        cookies = self._build_cookie_map(url, custom_cookies)
        effective_backend = backend if backend in {"curl", "httpx"} else "httpx"
        try:
            from tldw_Server_API.app.core import http_client as _http_client

            headers = _http_client._sanitize_accept_encoding_for_backend(headers, effective_backend)  # type: ignore[attr-defined]
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
            pass
        t0 = time.time()
        try:
            html, backend_used, elapsed = await self._fetch_html(
                url,
                headers=headers,
                cookies=cookies,
                backend=effective_backend,
                impersonate=impersonate,
                proxies=proxies,
            )
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:
            elapsed = max(0.0, time.time() - t0)
            self._emit_scrape_metrics(
                backend=effective_backend,
                outcome="error",
                elapsed_s=elapsed,
            )
            return _failed_scrape_result(url, exc)

        data = await run_extraction_in_thread(
            self._extract_from_html_with_pipeline,
            html,
            url,
            strategy_order=strategy_order,
            handler=handler,
            postprocess_markdown=postprocess_markdown,
            method_label="trafilatura",
            fallback_extractor=self._extract_trafilatura_json,
            schema_rules=schema_rules,
            llm_settings=llm_settings,
            regex_settings=regex_settings,
            cluster_settings=cluster_settings,
            allow_llm_extraction=allow_llm_extraction,
        )
        outcome = "success" if data.get("extraction_successful") else "no_extract"
        self._emit_scrape_metrics(
            backend=backend_used,
            outcome=outcome,
            elapsed_s=elapsed,
            content=data.get("content") if outcome == "success" else None,
        )
        return self._apply_dedup(url, data)

    def _extract_trafilatura_json(self, html: str, url: str) -> dict[str, Any]:
        """Extract using trafilatura JSON output to preserve legacy enhanced behavior."""
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_images=False,
            output_format="json",
        )

        if content:
            try:
                content_dict = json.loads(content)
            except json.JSONDecodeError as exc:
                return {
                    "url": url,
                    "error": f"Invalid trafilatura JSON: {exc}",
                    "extraction_successful": False,
                }

            return {
                "url": url,
                "title": content_dict.get("title", "Untitled"),
                "author": content_dict.get("author", "Unknown"),
                "date": content_dict.get("date", ""),
                "content": content_dict.get("text", ""),
                "extraction_successful": True,
                "method": "trafilatura",
            }

        return {
            "url": url,
            "error": "No content extracted",
            "extraction_successful": False,
        }

    async def _scrape_with_playwright(
        self,
        url: str,
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        proxies: Optional[dict[str, str]] = None,
        handler: Optional[Any] = None,
        strategy_order: Optional[list[str]] = None,
        postprocess_markdown: bool = False,
        schema_rules: Optional[dict[str, Any]] = None,
        llm_settings: Optional[dict[str, Any]] = None,
        regex_settings: Optional[dict[str, Any]] = None,
        cluster_settings: Optional[dict[str, Any]] = None,
        allow_llm_extraction: bool = True,
    ) -> dict[str, Any]:
        """Scrape using Playwright for JavaScript-heavy sites"""
        transport = resolve_browser_transport_decision(
            default_browser_transport_decision,
            component="enhanced_web_scraper",
        )
        if not transport.allowed:
            return browser_transport_failure_result(url, transport)
        # Fallback gracefully if browser isn't initialized
        if not self._browser:
            return await self._scrape_with_trafilatura(
                url,
                custom_cookies=custom_cookies,
                user_agent=user_agent,
                custom_headers=custom_headers,
                handler=handler,
                strategy_order=strategy_order,
                postprocess_markdown=postprocess_markdown,
                schema_rules=schema_rules,
                llm_settings=llm_settings,
                regex_settings=regex_settings,
                cluster_settings=cluster_settings,
                allow_llm_extraction=allow_llm_extraction,
            )
        headers_copy = dict(custom_headers) if custom_headers else {}
        effective_user_agent = user_agent or headers_copy.pop("User-Agent", None) or DEFAULT_USER_AGENT

        proxy_cfg = None
        if proxies:
            try:
                from tldw_Server_API.app.core import http_client as _http_client

                _http_client._validate_proxies_or_raise(proxies)  # type: ignore[attr-defined]
                proxy_server = _http_client._resolve_proxy_for_url(url, proxies)  # type: ignore[attr-defined]
                if proxy_server:
                    proxy_cfg = self._parse_proxy_for_playwright(proxy_server)
                    if proxy_cfg is None:
                        if "://" in proxy_server:
                            proxy_cfg = {"server": proxy_server}
                        else:
                            proxy_cfg = {"server": f"http://{proxy_server}"}
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Playwright proxy validation failed ({})", exception_type_for_log(exc))

        if proxy_cfg:
            context = await self._browser.new_context(user_agent=effective_user_agent, proxy=proxy_cfg)
        else:
            context = await self._browser.new_context(user_agent=effective_user_agent)

        if headers_copy:
            await context.set_extra_http_headers(headers_copy)

        pw_cookies = self._normalize_playwright_cookies(url, custom_cookies)
        if pw_cookies:
            await context.add_cookies(pw_cookies)

        page = await context.new_page()
        t0 = time.time()

        try:
            # Navigate with timeout
            response = await page.goto(url, wait_until="networkidle", timeout=30000)
            if response is None:
                raise ArticleRetrievalError(
                    "Article retrieval failed: no navigation response (expected terminal 2xx response)"
                )
            if not 200 <= response.status < 300:
                raise ArticleRetrievalError(
                    f"Article retrieval failed: HTTP {response.status} (expected terminal 2xx response)"
                )

            # Wait for content to load
            await page.wait_for_load_state("domcontentloaded")

            html = await page.content()
            data = await run_extraction_in_thread(
                self._extract_from_html_with_pipeline,
                html,
                url,
                strategy_order=strategy_order,
                handler=handler,
                postprocess_markdown=postprocess_markdown,
                method_label="playwright",
                fallback_extractor=self._extract_trafilatura_json,
                schema_rules=schema_rules,
                llm_settings=llm_settings,
                regex_settings=regex_settings,
                cluster_settings=cluster_settings,
                allow_llm_extraction=allow_llm_extraction,
            )
            if data.get("extraction_successful") or handler is not None:
                outcome = "success" if data.get("extraction_successful") else "no_extract"
                elapsed = max(0.0, time.time() - t0)
                self._emit_scrape_metrics(
                    backend="playwright",
                    outcome=outcome,
                    elapsed_s=elapsed,
                    content=data.get("content") if outcome == "success" else None,
                )
                return self._apply_dedup(url, data)
            pipeline_trace = data.get("extraction_trace") if isinstance(data.get("extraction_trace"), list) else []

            # Extract content (Playwright fallback)
            title = await page.title()

            # Try to find main content
            content = ""
            for selector in ["main", "article", '[role="main"]', "#content", ".content"]:
                elements = await page.query_selector_all(selector)
                if elements:
                    for element in elements:
                        text = await element.inner_text()
                        if len(text) > len(content):
                            content = text

            # Fallback to body if no content found
            if not content:
                content = await page.inner_text("body")

            # Extract metadata
            author = await page.evaluate("""() => {
                const meta = document.querySelector('meta[name="author"]');
                return meta ? meta.content : "Unknown";
            }""")

            date = await page.evaluate("""() => {
                const meta = document.querySelector('meta[property="article:published_time"]');
                return meta ? meta.content : "";
            }""")

            data = {
                "url": url,
                "title": title,
                "author": author,
                "date": date,
                "content": content,
                "extraction_successful": True,
                "method": "playwright",
            }
            trace = list(pipeline_trace)
            trace.append({"strategy": "playwright", "status": "success", "reason": "fallback_extracted"})
            data["extraction_trace"] = trace
            data["extraction_strategy"] = "playwright"
            data.setdefault("extraction_strategy_order", ["playwright"])
            log_counter("extraction_strategy_total", labels={"strategy": "playwright", "status": "success"})
            elapsed = max(0.0, time.time() - t0)
            self._emit_scrape_metrics(
                backend="playwright",
                outcome="success",
                elapsed_s=elapsed,
                content=content,
            )
            return self._apply_dedup(url, data)

        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:
            elapsed = max(0.0, time.time() - t0)
            self._emit_scrape_metrics(
                backend="playwright",
                outcome="error",
                elapsed_s=elapsed,
            )
            return _failed_scrape_result(url, exc)

        finally:
            await page.close()
            await context.close()

    async def _scrape_with_beautifulsoup(
        self,
        url: str,
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        backend: str = "httpx",
        impersonate: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
        handler: Optional[Any] = None,
        strategy_order: Optional[list[str]] = None,
        postprocess_markdown: bool = False,
        schema_rules: Optional[dict[str, Any]] = None,
        llm_settings: Optional[dict[str, Any]] = None,
        regex_settings: Optional[dict[str, Any]] = None,
        cluster_settings: Optional[dict[str, Any]] = None,
        allow_llm_extraction: bool = True,
    ) -> dict[str, Any]:
        """Scrape using BeautifulSoup for simple HTML parsing"""
        headers = self._build_request_headers(user_agent, custom_headers)
        cookies = self._build_cookie_map(url, custom_cookies)
        effective_backend = backend if backend in {"curl", "httpx"} else "httpx"
        try:
            from tldw_Server_API.app.core import http_client as _http_client

            headers = _http_client._sanitize_accept_encoding_for_backend(headers, effective_backend)  # type: ignore[attr-defined]
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
            pass
        t0 = time.time()
        try:
            html, backend_used, elapsed = await self._fetch_html(
                url,
                headers=headers,
                cookies=cookies,
                backend=effective_backend,
                impersonate=impersonate,
                proxies=proxies,
            )
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as exc:
            elapsed = max(0.0, time.time() - t0)
            self._emit_scrape_metrics(
                backend=effective_backend,
                outcome="error",
                elapsed_s=elapsed,
            )
            return _failed_scrape_result(url, exc)

        data = await run_extraction_in_thread(
            self._extract_from_html_with_pipeline,
            html,
            url,
            strategy_order=strategy_order,
            handler=handler,
            postprocess_markdown=postprocess_markdown,
            method_label="beautifulsoup",
            fallback_extractor=self._extract_trafilatura_json,
            schema_rules=schema_rules,
            llm_settings=llm_settings,
            regex_settings=regex_settings,
            cluster_settings=cluster_settings,
            allow_llm_extraction=allow_llm_extraction,
        )
        if data.get("extraction_successful") or handler is not None:
            outcome = "success" if data.get("extraction_successful") else "no_extract"
            self._emit_scrape_metrics(
                backend=backend_used,
                outcome=outcome,
                elapsed_s=elapsed,
                content=data.get("content") if outcome == "success" else None,
            )
            return self._apply_dedup(url, data)
        data = self._extract_beautifulsoup_html(html, url, data.get("extraction_trace"))
        self._emit_scrape_metrics(
            backend=backend_used,
            outcome="success",
            elapsed_s=elapsed,
            content=data["content"],
        )
        return self._apply_dedup(url, data)

    @staticmethod
    def _extract_beautifulsoup_html(html: str, url: str, pipeline_trace=None) -> dict[str, Any]:
        """Keep the enhanced fallback pure so canonical acquisition can reuse it."""
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = (title_tag.get_text(strip=True) if title_tag else "Untitled") or "Untitled"
        for script in soup(["script", "style"]):
            script.decompose()
        content = ""
        for tag in ["main", "article", "div"]:
            elements = soup.find_all(
                tag,
                class_=lambda x: x and any(keyword in x.lower() for keyword in ["content", "article", "post", "entry"]),
            )
            if elements:
                content = "\n\n".join(elem.get_text(strip=True) for elem in elements)
                break
        if not content:
            content = soup.get_text(strip=True)
        author_meta = soup.find("meta", {"name": "author"})
        date_meta = soup.find("meta", {"property": "article:published_time"})
        trace = list(pipeline_trace) if isinstance(pipeline_trace, list) else []
        trace.append({"strategy": "beautifulsoup", "status": "success", "reason": "fallback_extracted"})
        log_counter("extraction_strategy_total", labels={"strategy": "beautifulsoup", "status": "success"})
        return {
            "url": url,
            "title": title,
            "author": author_meta.get("content", "Unknown") if author_meta else "Unknown",
            "date": date_meta.get("content", "") if date_meta else "",
            "content": content,
            "extraction_successful": True,
            "method": "beautifulsoup",
            "extraction_trace": trace,
            "extraction_strategy": "beautifulsoup",
            "extraction_strategy_order": ["beautifulsoup"],
        }

    def _start_controlled_job(self, payload, controls) -> WorkHandle[PageData]:
        """Bind this instance's capabilities to one canonical, controlled attempt."""
        from .orchestration import article

        config = payload.options.get("configuration", self.config)
        config = article._snapshot_config(config)
        cookies = article._snapshot_cookies(payload.options["selected_cookies"])
        method = payload.operation
        resolver = self._resolve_scrape_plan
        state = {"method": method, "route_backend": "auto", "default_handler": True}
        dependencies = article._build_default_dependencies(cookies)
        advice = dependencies.apply_preflight_advice
        handler_resolver = dependencies.resolve_handler

        def resolve(url, _config):
            route, backend, handler = (
                resolver(url, configuration=config)
                if getattr(resolver, "__func__", None) is _DEFAULT_ENHANCED_RESOLVE
                else resolver(url)
            )
            plan = article.ArticlePlan.from_routing_plan(route, config, cookies)
            headers = self._build_plan_headers(
                plan.ua_profile,
                dict(plan.headers),
                payload.options.get("user_agent"),
                payload.options.get("custom_headers"),
            )
            plan = replace(
                plan,
                backend=backend,
                handler=handler or DEFAULT_HANDLER,
                headers=headers,
                browser=replace(plan.browser, user_agent=headers["User-Agent"]),
            ).for_enhanced_browser(cookies)
            state["route_backend"] = str(getattr(route, "backend", "auto") or "auto")
            state["default_handler"] = not handler or handler == DEFAULT_HANDLER
            return plan

        def apply_advice(*args, **kwargs):
            backend, advised_method, result = advice(*args, **kwargs)
            state["method"] = (
                "playwright"
                if backend == "playwright"
                else ("trafilatura" if advised_method == "auto" else advised_method)
            )
            return backend, advised_method, result

        def extract(html, url, **kwargs):
            data = self._extract_from_html_with_pipeline(
                html,
                url,
                **kwargs,
                method_label=state["method"],
                postprocess_markdown=state["default_handler"],
                fallback_extractor=self._extract_trafilatura_json,
            )
            if (
                state["method"] == "beautifulsoup"
                and not data.get("extraction_successful")
                and kwargs.get("handler") is None
            ):
                return self._extract_beautifulsoup_html(html, url, data.get("extraction_trace"))
            snapshot = getattr(dependencies.browser, "_rendered_fallback", None)
            if (
                state["method"] == "playwright"
                and not data.get("extraction_successful")
                and kwargs.get("handler") is None
                and snapshot is not None
            ):
                trace = list(data.get("extraction_trace") or ())
                trace.append({"strategy": "playwright", "status": "success", "reason": "fallback_extracted"})
                log_counter("extraction_strategy_total", labels={"strategy": "playwright", "status": "success"})
                return {
                    "url": url,
                    **snapshot,
                    "extraction_successful": True,
                    "method": "playwright",
                    "extraction_trace": trace,
                    "extraction_strategy": "playwright",
                    "extraction_strategy_order": ["playwright"],
                }
            return data

        dependencies = replace(
            dependencies,
            resolve_plan=resolve,
            extract=extract,
            apply_preflight_advice=apply_advice,
            backend_setting=lambda _plan: state["route_backend"],
            convert_content=lambda content: content,
            resolve_handler=lambda path: None if path == DEFAULT_HANDLER else handler_resolver(path),
            http_only=True,
            http_failure=_failed_scrape_result,
        )

        async def run():
            await self.rate_limiter.acquire()
            controls.check()
            if method not in {"auto", "trafilatura", "beautifulsoup", "playwright"}:
                raise ValueError("Unknown scraping method")
            handle = article.start_page(
                payload.url,
                list(cookies),
                controls=controls,
                request_context=RuntimeRequestContext(source="enhanced_batch", stage="fetch", user_id=payload.owner),
                discover=lambda _html, request, final: DiscoveryResult(request, final, ()),
                configuration={"web_scraper": config},
                dependencies=dependencies,
                allow_llm_extraction=payload.options.get("allow_llm_extraction", True),
                discovery_gate=lambda _result: False,
                transform=self._apply_dedup,
                method_intent=method,
            )
            physical_work.get().observe(handle.drained)
            handle.result.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            try:
                await asyncio.wait({handle.result})
                return handle.result.result()
            except BaseException:
                handle.cancel()
                raise

        return _owned_adapter_page(run)

    def _controlled_job_options(self, url: str, options: Mapping[str, Any]) -> dict[str, Any]:
        """Byte-admit selected execution settings, keeping opaque callback inputs intact."""
        from .orchestration.crawl import _selected_cookies

        copied = deepcopy(dict(options))
        if "configuration" not in copied:
            copied["configuration"] = deepcopy(self.config)
        if "selected_cookies" not in copied:
            copied["selected_cookies"] = list(
                _selected_cookies(
                    url,
                    tuple(copied.get("custom_cookies") or ()),
                    {urlparse(url).netloc: self.cookie_manager.get_cookies(url)},
                )
            )
        return copied

    async def _batch_report(
        self,
        urls: Iterable[str],
        *,
        method="auto",
        priority=JobPriority.NORMAL,
        user_id=None,
        allow_llm_extraction=True,
        options: Mapping[str, Any] | None = None,
    ) -> CrawlReport:
        """Own a bounded producer; public conversion and summaries happen later."""
        from .contracts.requests import CrawlRequest, ScrapeContext
        from .crawl.models import CrawlLimits
        from .crawl.profiles import resolve_crawl_plan
        from .orchestration.batch import execute_batch
        from .orchestration.crawl_context import CrawlContext

        limits = CrawlLimits()
        plan = resolve_crawl_plan(
            CrawlRequest("https://batch.invalid/", context=ScrapeContext(user_id=user_id)),
            profile="enhanced_batch",
            options={"method": method, "priority": priority.name, "allow_llm_extraction": allow_llm_extraction},
            supplied=frozenset({"method", "priority", "allow_llm_extraction"}),
            config=self.config,
            limits=limits,
        )
        metadata = deepcopy({**(options or {}), "allow_llm_extraction": allow_llm_extraction})
        canonical = isinstance(self.job_queue, ScrapingJobQueue) and self.job_queue._canonical_parent() is self
        if canonical:
            metadata["configuration"] = deepcopy(self.config)
        stored = self.cookie_manager._snapshot() if canonical else {}
        from .orchestration.crawl import _selected_cookies

        owner = str(user_id) if user_id is not None else None

        def job(url):
            return ScrapingJob(
                f"job_{uuid.uuid4().hex}",
                url,
                method,
                user_id=owner,
                priority=priority,
                metadata={
                    **metadata,
                    "selected_cookies": list(
                        _selected_cookies(
                            url,
                            tuple(metadata.get("custom_cookies") or ()),
                            stored,
                        )
                    ),
                }
                if canonical
                else metadata,
            )

        try:
            await self.job_queue.start()
            return await execute_batch(
                urls,
                plan=plan,
                context=CrawlContext(limits, clock=time.monotonic),
                queue=self.job_queue,
                make_job=job,
            )
        finally:
            # Only admitted handles retain private capabilities through actual drain.
            self = urls = options = metadata = stored = job = None

    async def scrape_multiple(
        self,
        urls: list[str],
        method: str = "auto",
        priority: JobPriority = JobPriority.NORMAL,
        summarize: bool = False,
        user_id: Optional[str] = None,
        allow_llm_extraction: bool = True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert the bounded batch report before optional service postprocessing."""
        from .contracts.conversion import to_legacy_crawl

        summary_prompt_overrides = kwargs.pop("summary_prompt_overrides", None)
        report = await self._batch_report(
            urls,
            method=method,
            priority=priority,
            user_id=user_id,
            allow_llm_extraction=allow_llm_extraction,
            options=kwargs,
        )
        results = to_legacy_crawl(report, profile="enhanced_batch")
        if summarize:
            for result in results:
                if result.get("extraction_successful") and result.get("content"):
                    result["summary"] = await self._summarize_content(
                        result["content"],
                        summary_prompt_overrides=summary_prompt_overrides,
                        **kwargs,
                    )
        return results

    async def _summarize_content(self, content: str, **kwargs) -> str:
        """Summarize content using LLM"""
        try:
            overrides = kwargs.get("summary_prompt_overrides") or {}
            system_default = kwargs.get("system_prompt")
            if system_default is None:
                system_default = "Summarize this article concisely."
            summary = await asyncio.to_thread(
                analyze,
                input_data=content,
                custom_prompt_arg=overrides.get("user", kwargs.get("custom_prompt", "")),
                api_name=kwargs.get("api_name", "openai"),
                api_key=kwargs.get("api_key"),
                temp=kwargs.get("temperature", 0.7),
                system_message=overrides.get("system", kwargs.get("system_message", system_default)),
            )
            return summary
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Summarization failed ({})", exception_type_for_log(e))
            return "Summary generation failed"

    async def _crawl_report(
        self,
        base_url: str,
        *,
        profile: str,
        options: Mapping[str, Any],
        custom_cookies=None,
        user_agent=None,
        custom_headers=None,
        url_filter=None,
        task_id=None,
        user_id=None,
    ) -> CrawlReport:
        """Compose instance capabilities privately and expose the canonical report."""
        from .contracts.requests import CrawlRequest, ScrapeContext
        from .crawl.models import CrawlHooks

        sitemap = profile == "enhanced_sitemap"
        total = (
            {"total_urls": 0, "processed_urls": 0}
            if sitemap
            else {
                "total_pages": options.get("max_pages", 100),
                "pages_scraped": 0,
            }
        )
        if task_id and task_id not in self._progress:
            self._progress[task_id] = {
                **total,
                "status": "in_progress",
                "current_url": base_url,
                "started_at": datetime.now().isoformat(),
                "owner_user_id": None if user_id is None else str(user_id),
            }
        if task_id:
            self._progress[task_id].pop("completed_at", None)
            self._set_progress(task_id, status="in_progress")

        def progress(snapshot):
            updates = dict(snapshot)
            if sitemap:
                updates["processed_urls"] = snapshot["pages_scraped"]
                updates["total_urls"] = snapshot["pages_scraped"] + snapshot["queue_size"]
            self._set_progress(task_id, **updates)

        await self.job_queue.start()
        try:
            report = await execute_crawl(
                CrawlRequest(base_url, context=ScrapeContext(user_id=user_id)),
                profile=profile,
                options=options,
                supplied=frozenset(options),
                config=self.config,
                credentials={
                    "custom_cookies": custom_cookies,
                    "user_agent": user_agent,
                    "custom_headers": custom_headers,
                    "stored_cookies_by_netloc": self.cookie_manager._snapshot(),
                },
                hooks=CrawlHooks(url_filter=url_filter, on_progress=progress),
                queue=self.job_queue,
                transform=self._apply_dedup,
            )
        except asyncio.CancelledError:
            self._set_progress(task_id, status="cancelled")
            raise
        self._set_progress(task_id, status=report.outcome.state)
        if report.outcome.state == "completed":
            self._set_progress(task_id, completed_at=datetime.now().isoformat())
        return report

    async def scrape_sitemap(
        self,
        sitemap_url: str,
        filter_func: Optional[Callable[[str], bool]] = None,
        max_urls: Optional[int] = None,
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        allow_llm_extraction: bool = True,
    ) -> list[dict[str, Any]]:
        """Adapt live sitemap filtering without serializing executable callbacks."""
        from .contracts.conversion import to_legacy_crawl

        report = await self._crawl_report(
            sitemap_url,
            profile="enhanced_sitemap",
            options={"max_urls": max_urls, "allow_llm_extraction": allow_llm_extraction},
            custom_cookies=custom_cookies,
            user_agent=user_agent,
            custom_headers=custom_headers,
            url_filter=filter_func,
            task_id=task_id,
            user_id=user_id,
        )
        return to_legacy_crawl(report, profile="enhanced_sitemap")

    async def recursive_scrape(
        self,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        url_filter: Optional[Callable[[str], bool]] = None,
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        include_external_override: Optional[bool] = None,
        score_threshold_override: Optional[float] = None,
        crawl_strategy: Optional[str] = None,
        allow_llm_extraction: bool = True,
    ) -> list[dict[str, Any]]:
        """Preserve the public signature while using the sole canonical crawl."""
        from .contracts.conversion import to_legacy_crawl

        report = await self._crawl_report(
            base_url,
            profile="enhanced_recursive",
            options={
                "max_pages": max_pages,
                "max_depth": max_depth,
                "include_external_override": include_external_override,
                "score_threshold_override": score_threshold_override,
                "crawl_strategy": crawl_strategy,
                "allow_llm_extraction": allow_llm_extraction,
            },
            custom_cookies=custom_cookies,
            user_agent=user_agent,
            custom_headers=custom_headers,
            url_filter=url_filter,
            task_id=task_id,
            user_id=user_id,
        )
        return to_legacy_crawl(report, profile="enhanced_recursive")

    async def _extract_links(self, base_url: str, content: str) -> list[str]:
        """Extract links. If provided content looks like plain text, fetch HTML first."""
        html_text = content or ""
        # Heuristic: if content lacks HTML tags, fetch the page HTML
        if "<a" not in html_text and "<html" not in html_text:
            try:
                headers = self._build_request_headers(None, None)
                cookies = self._build_cookie_map(base_url, None)
                resp = await afetch(
                    method="GET",
                    url=base_url,
                    headers=headers,
                    cookies=cookies,
                )
                try:
                    html_text = resp.text
                finally:
                    await self._close_response(resp)
            except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
                logger.warning("Failed to fetch HTML for link extraction ({})", exception_type_for_log(e))
                return []

        try:
            soup = BeautifulSoup(html_text, "html.parser")
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href and not href.startswith("#"):
                    links.append(href)
            return links
        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning("Error parsing links from content ({})", exception_type_for_log(e))
            return []

    def get_progress(self, task_name: str, *, owner_user_id: Optional[str] = None) -> dict[str, Any]:
        """Get progress for a specific task.

        ``self._progress`` is process-global and shared by every caller, so a
        task id alone is not an authorization claim. Request handlers must pass
        ``owner_user_id``; a task belonging to somebody else reads as absent.
        """
        entry = self._progress.get(task_name, {})
        if owner_user_id is None or not entry:
            return entry
        if str(entry.get("owner_user_id") or "") != str(owner_user_id):
            return {}
        return entry

    async def save_progress(self, task_name: str, filepath: Path):
        """Save progress to file for resumability"""
        progress_data = self._progress.get(task_name, {})
        async with aiofiles.open(filepath, "w") as f:
            await f.write(json.dumps(progress_data, indent=2))

    async def load_progress(self, task_name: str, filepath: Path):
        """Load progress from file"""
        if filepath.exists():
            async with aiofiles.open(filepath) as f:
                content = await f.read()
                self._progress[task_name] = json.loads(content)


_DEFAULT_QUEUE_EXECUTE = ScrapingJobQueue._execute_job
_DEFAULT_ENHANCED_SCRAPE = EnhancedWebScraper.scrape_article
_DEFAULT_ENHANCED_RESOLVE = EnhancedWebScraper._resolve_scrape_plan


# Integration with existing system
async def create_enhanced_scraper() -> EnhancedWebScraper:
    """Create and initialize enhanced scraper"""
    scraper = EnhancedWebScraper()
    await scraper.start()
    return scraper


# Example usage
if __name__ == "__main__":

    async def test_enhanced_scraper():
        """Test the enhanced scraper"""
        scraper = await create_enhanced_scraper()

        try:
            # Test single article scraping
            print("Testing single article scraping...")
            result = await scraper.scrape_article("https://example.com/article", method="trafilatura")
            print(f"Result: {result['title'] if result.get('extraction_successful') else result['error']}")

            # Test multiple URL scraping
            print("\nTesting multiple URL scraping...")
            urls = ["https://example.com/article1", "https://example.com/article2", "https://example.com/article3"]
            results = await scraper.scrape_multiple(urls, priority=JobPriority.HIGH, summarize=True)
            print(f"Scraped {len(results)} articles")

            # Test recursive scraping
            print("\nTesting recursive scraping...")
            results = await scraper.recursive_scrape("https://example.com", max_pages=10, max_depth=2)
            print(f"Recursively scraped {len(results)} pages")

            # Get queue status
            status = scraper.job_queue.get_status()
            print(f"\nQueue status: {status}")

        finally:
            await scraper.stop()

    asyncio.run(test_enhanced_scraper())
