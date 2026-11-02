"""Private composition of crawl policy, credentials and canonical page handles."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Ingestion_Media_Processing.logging_safety import redact_url_for_log
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_gauge, log_histogram
from tldw_Server_API.app.core.testing import is_truthy

from ..browser_transport import (
    browser_transport_failure_result,
    default_browser_transport_decision,
    resolve_browser_transport_decision,
)
from ..contracts.conversion import _to_plain
from ..contracts.execution import CrawlReport, DiscoveryResult, PageData
from ..contracts.requests import CrawlRequest
from ..crawl.discovery import discover, normalize_reference, resolve_reference, url_bytes
from ..crawl.engine import run_crawl
from ..crawl.models import CheckpointPort, CrawlHooks, CrawlLimits, CrawlPlan, FrontierEntry, TraversalDecision
from ..crawl.profiles import resolve_crawl_plan
from ..filters import ContentTypeFilter, DomainFilter, FilterChain, RobotsFilter, URLPatternFilter
from ..jobs.models import ScrapingJob
from ..jobs.queue import ProcessLocalJobQueue
from ..outbound_policy import decide_web_outbound_policy
from ..runtime import DefaultFetchClient, FetchRequest, RuntimeRequestContext
from ..runtime.controls import PhysicalWork, WorkHandle, physical_work
from ..scoring import (
    CompositeScorer,
    ContentTypeScorer,
    DomainAuthorityScorer,
    FreshnessScorer,
    KeywordRelevanceScorer,
    PathDepthScorer,
)
from . import article
from .batch import _wait_owned, queued_page
from .crawl_context import CrawlContext

_EXCLUDES = [
    "/tag/",
    "/category/",
    "/author/",
    "/search/",
    "/page/",
    "wp-content",
    "wp-includes",
    "wp-json",
    "wp-admin",
    "login",
    "register",
    "cart",
    "checkout",
    "account",
    ".jpg",
    ".png",
    ".gif",
    ".pdf",
    ".zip",
]


def _truthy(value: Any) -> bool:
    """Use the same string normalization as the characterized enhanced adapter."""
    return is_truthy(value) if isinstance(value, str) else bool(value)


def _segments(url: str) -> int:
    """Count nonempty path segments for seed-relative depth and score metadata."""
    return len([part for part in urlsplit(url).path.split("/") if part])


def _observe(function: Callable, *args: Any, **kwargs: Any) -> None:
    """Keep optional legacy crawl telemetry from changing traversal outcomes."""
    with contextlib.suppress(AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError):
        function(*args, **kwargs)


def _skip(reason: str, strategy: str, entry: FrontierEntry) -> TraversalDecision:
    """Emit stable bounded reasons without placing targets in metric labels."""
    _observe(log_counter, "webscraping.crawl.urls_skipped", labels={"reason": reason})
    logger.debug("Skip URL [{}:{} depth={}]: {}", strategy, reason, entry.depth, redact_url_for_log(entry.url))
    return TraversalDecision(False, reason="filtered")


class ProfileTraversalPolicy:
    """Compose existing filters/scorers and governed robots outside the engine."""

    def __init__(
        self,
        plan: CrawlPlan,
        *,
        config: Mapping[str, Any],
        headers: Mapping[str, str],
        fetch_document: Callable[[str], Awaitable[Mapping[str, Any]]],
        hooks: CrawlHooks,
        controls: CrawlContext,
    ) -> None:
        self.plan, self.config, self.headers = plan, dict(config), dict(headers)
        self.hooks, self.controls = hooks, controls
        self._fetch_document = fetch_document
        self._prepared = False
        self.active_entry = None

    def _prepare(self) -> None:
        """Prepare only after engine admission has bounded the request seed."""
        if self._prepared:
            return
        plan, config, headers = self.plan, self.config, self.headers
        url_bytes(plan.request.base_url, plan)
        self.enhanced = plan.profile in {"enhanced_recursive", "enhanced_url_level"}
        self.base = normalize_reference(plan.request.base_url, plan.request.base_url, plan)
        self.base_depth = _segments(self.base)
        options = plan.options
        allowed = {
            part.strip().lower() for part in str(options.get("allowed_domains") or "").split(",") if part.strip()
        }
        blocked = {
            part.strip().lower() for part in str(options.get("blocked_domains") or "").split(",") if part.strip()
        }
        external = options.get("include_external", False)
        domains = (allowed or None) if external else (allowed | {urlsplit(self.base).netloc.lower()})
        self.filters = FilterChain(
            [
                DomainFilter(allowed=domains, blocked=blocked or None),
                ContentTypeFilter(),
                URLPatternFilter(exclude_patterns=_EXCLUDES),
            ]
        )
        self.legacy_filters = FilterChain(
            [
                ContentTypeFilter(),
                URLPatternFilter(include_patterns=["/article/", "/post/", "/blog/"], exclude_patterns=_EXCLUDES),
            ]
        )
        scorers = [PathDepthScorer(), ContentTypeScorer(), FreshnessScorer()]
        if bool(options.get("enable_keyword_scorer")):
            keywords = [part.strip() for part in str(options.get("keywords") or "").split(",") if part.strip()]
            if keywords:
                scorers.append(KeywordRelevanceScorer(keywords))
        if bool(options.get("enable_domain_map")):
            raw = options.get("domain_map") or ""
            try:
                weights = (
                    dict(raw)
                    if isinstance(raw, Mapping)
                    else json.loads(raw)
                    if str(raw).strip().startswith("{")
                    else dict(part.split(":", 1) for part in str(raw).split(",") if ":" in part)
                )
                weights = {str(key).lower(): float(value) for key, value in weights.items()}
            except (TypeError, ValueError):
                weights = {}
            if weights:
                scorers.append(DomainAuthorityScorer(weights))
        self.scorer, self.order = CompositeScorer(scorers), PathDepthScorer()
        self.respect = _truthy(options.get("respect_robots", config.get("web_scraper_respect_robots", True)))
        self.robots = RobotsFilter(headers.get("User-Agent", "*"), fetch_robots=self._fetch_document)
        self._prepared = True

    async def _policy(self, url: str, *, robots: bool, source: str, stage: str) -> TraversalDecision:
        """Fail closed on local policy errors, never on parent interruption."""
        self.controls.check()
        try:
            decision = await decide_web_outbound_policy(
                url,
                respect_robots=robots,
                user_agent=self.headers.get("User-Agent"),
                source=source,
                stage=stage,
                config={"web_scraper": self.config},
                robots_filter=self.robots,
            )
            if decision.allowed:
                return TraversalDecision()
            failure = article._policy_blocked_result(url, decision)
            if self.plan.profile.startswith("enhanced_"):
                # The enhanced collector's established denial payload omits the
                # legacy placeholder article fields, but keeps policy details.
                failure = {
                    key: value for key, value in failure.items() if key not in {"title", "author", "date", "content"}
                }
            return TraversalDecision(False, metadata=failure, reason="policy_denied")
        except (AttributeError, ImportError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            logger.warning("crawl_policy_evaluation_failed")
            if self.plan.profile.startswith("enhanced_"):
                failure = {
                    "url": url,
                    "error": "Outbound policy evaluation failed. Please contact system administrator.",
                    "extraction_successful": False,
                }
            else:
                failure = {**article._blocking_failure_result(url), "error": "Outbound policy evaluation failed"}
            return TraversalDecision(False, metadata=failure, reason="policy_denied")

    async def evaluate(
        self, entry: FrontierEntry, *, phase: Literal["seed", "discovery", "visit"]
    ) -> TraversalDecision:
        """Preserve seed admission versus ordinary filtered-candidate semantics."""
        self.controls.check()
        self._prepare()
        if phase == "seed":
            if self.plan.profile == "legacy_recursive":
                transport = resolve_browser_transport_decision(
                    default_browser_transport_decision, component="legacy_article_extractor"
                )
                if not transport.allowed:
                    return TraversalDecision(
                        False,
                        metadata=browser_transport_failure_result(entry.url, transport),
                        reason="transport_unavailable",
                    )
            if self.plan.profile in {
                "enhanced_recursive",
                "enhanced_url_level",
                "legacy_sitemap",
                "enhanced_sitemap",
                "legacy_url_level",
            }:
                decision = await self._policy(
                    entry.url,
                    robots=self.respect if self.plan.profile == "enhanced_sitemap" else False,
                    source="enhanced_sitemap"
                    if self.plan.profile == "enhanced_sitemap"
                    else "sitemap_scrape"
                    if self.plan.profile == "legacy_sitemap"
                    else "recursive_scrape",
                    stage="pre_fetch",
                )
                if not decision.allowed:
                    return decision
            return TraversalDecision(ordering_score=self.order.score(entry.url) if self.enhanced else 0)
        if phase == "discovery":
            if self.enhanced:
                strategy = str(self.plan.options["crawl_strategy"])
                if not self.filters.apply(entry.url):
                    return _skip("filter_chain", strategy, entry)
                if max(0, _segments(entry.url) - self.base_depth) > self.plan.options["max_depth"]:
                    return _skip("path_depth", strategy, entry)
                decision = await self._policy(
                    entry.url, robots=self.respect, source="recursive_scrape_candidate", stage="discovery"
                )
                if not decision.allowed:
                    reason = (
                        "robots" if str(decision.metadata.get("policy_reason", "")).startswith("robots_") else "policy"
                    )
                    return _skip(reason, strategy, entry)
                relevance = self.scorer.score(entry.url)
                if relevance < self.plan.options["score_threshold"]:
                    return _skip("below_threshold", strategy, entry)
                if self.plan.profile == "enhanced_url_level":
                    # Historical URL Level counts '/' as one split component.
                    if len(urlsplit(entry.url).path.strip("/").split("/")) > self.plan.options.get("level", 3):
                        return TraversalDecision(False, reason="filtered")
                if self.hooks.url_filter and not self.hooks.url_filter(entry.url):
                    return _skip("custom_filter", strategy, entry)
                return TraversalDecision(relevance_score=relevance, ordering_score=self.order.score(entry.url))
            if self.plan.profile == "legacy_recursive":
                allowed = (
                    entry.url.startswith(self.plan.request.base_url)
                    and entry.url.startswith("http")
                    and self.legacy_filters.apply(entry.url)
                )
                return TraversalDecision(allowed, reason="allowed" if allowed else "filtered")
            if self.plan.profile == "legacy_url_level":
                allowed = urlsplit(entry.url).netloc == urlsplit(self.base).netloc
                return TraversalDecision(allowed, reason="allowed" if allowed else "filtered")
        if phase == "visit":
            decision = await self._policy(entry.url, robots=False, source=self.plan.profile, stage="visit")
            if not decision.allowed:
                return decision
        if phase == "visit" and self.enhanced:
            self.active_entry = entry
            relevance, ordering = self.scorer.score(entry.url), self.order.score(entry.url)
            strategy = str(self.plan.options["crawl_strategy"])
            if strategy in {"best_first", "best-first", "bestfirst"}:
                strategy, ordering = "best_first", entry.score
            _observe(log_gauge, "webscraping.crawl.depth", float(entry.depth))
            _observe(log_histogram, "webscraping.crawl.score", relevance, labels={"stage": "visit"})
            return TraversalDecision(
                relevance_score=relevance,
                ordering_score=ordering,
                metadata={
                    "depth": entry.depth,
                    "parent_url": entry.parent_url,
                    "score": relevance,
                    "ordering_score": ordering,
                    "score_type": "composite_relevance",
                    "ordering_score_type": "path_depth",
                    "crawl_strategy": strategy,
                },
            )
        return TraversalDecision()


def _selected_cookies(
    url: str, explicit: tuple[Mapping[str, Any], ...], stored: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    """Use the existing exact netloc lookup, never flatten or normalize jar scope."""
    merged: dict[str, dict[str, Any]] = {}
    for source in (stored.get(urlsplit(url).netloc) or (), explicit):
        for item in source:
            if not isinstance(item, Mapping):
                continue
            if "name" in item and "value" in item:
                name, value = str(item["name"]), str(item["value"])
                merged[name] = {**item, "name": name, "value": value}
            else:
                for name, value in item.items():
                    merged[str(name)] = {"name": str(name), "value": str(value)}
    return tuple(merged.values())


def _queued_crawl_page(
    handle: WorkHandle[PageData], *, entry: FrontierEntry, context: CrawlContext
) -> WorkHandle[PageData]:
    """Translate only child-local cancellation, preserving owner stop and drain."""

    async def result() -> PageData:
        # Parent check after notification wins over a concurrent child cancellation.
        # Owning-task cancellation is still handled by the engine's shielded await.
        await _wait_owned({handle.result}, context)
        if handle.result.cancelled():
            return PageData(
                {"url": entry.url, "error": "Job cancelled", "extraction_successful": False},
                DiscoveryResult(entry.url, entry.url, ()),
            )
        return handle.result.result()

    task = asyncio.create_task(result())
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    return WorkHandle(task, handle.drained, handle.cancel)


def _prepare_page_plan(
    page_plan: article.ArticlePlan,
    *,
    headers: Mapping[str, str],
    legacy: bool,
    method: Any,
    explicit_agent: Any,
    enhanced: bool = False,
    custom_cookies=(),
) -> article.ArticlePlan:
    """Compose request headers without changing extraction-method intent."""
    agent = str(
        explicit_agent
        or (headers["User-Agent"] if legacy else page_plan.headers.get("User-Agent") or headers["User-Agent"])
    )
    prepared = replace(
        page_plan,
        headers={**dict(page_plan.headers), **headers, "User-Agent": agent},
        browser=replace(page_plan.browser, user_agent=agent),
    )
    if legacy:
        return replace(prepared, backend="playwright", respect_robots=False)
    if enhanced:
        prepared = prepared.for_enhanced_browser(custom_cookies)
    return replace(prepared, backend="playwright") if method == "playwright" else prepared


async def execute_crawl(
    request: CrawlRequest,
    *,
    profile: str,
    options: Mapping[str, Any],
    supplied: frozenset[str],
    config: Mapping[str, Any],
    credentials: Mapping[str, Any],
    limits: CrawlLimits = CrawlLimits(),
    checkpoint: CheckpointPort | None = None,
    hooks: CrawlHooks | None = None,
    seeds: Iterable[str] | None = None,
    queue: ProcessLocalJobQueue | None = None,
    transform: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    browser_http_fallback: Callable[[], bool] | None = None,
) -> CrawlReport:
    """Snapshot private inputs and compose the sole canonical engine."""
    config = article._snapshot_config(config)
    private = deepcopy(dict(credentials))
    plan = resolve_crawl_plan(
        request, profile=profile, options=options, supplied=supplied, config=config, limits=limits
    )
    context = CrawlContext(limits, clock=time.monotonic)
    hooks = hooks or CrawlHooks()
    cookies = article._snapshot_cookies(private.get("custom_cookies"))
    stored = private.get("stored_cookies_by_netloc") or {}
    requested_agent = private.get("user_agent") or (private.get("custom_headers") or {}).get("User-Agent")
    header_copy = dict(private.get("custom_headers") or {})
    default_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        if profile == "legacy_recursive"
        else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    headers = {
        "User-Agent": str(private.get("user_agent") or header_copy.pop("User-Agent", None) or default_agent),
        **header_copy,
    }
    committed = 0
    queued_handles: list[WorkHandle[PageData]] = []

    async def fetch_document(url: str) -> dict[str, Any]:
        """Register transport ownership before any policy/HTTP dispatch."""
        context.check()
        owner = PhysicalWork()
        registered = asyncio.Event()

        async def execute() -> dict[str, Any]:
            # Also gate eager task factories: registration must precede dispatch,
            # not merely precede the caller's next await.
            await registered.wait()
            token = physical_work.set(owner)
            try:
                return await fetch_owned_document(url)
            finally:
                physical_work.reset(token)

        def completed(done: asyncio.Task) -> None:
            owner.finish()
            if not done.cancelled():
                done.exception()

        task = asyncio.create_task(execute())
        task.add_done_callback(completed)
        context.attach(WorkHandle(task, owner.drained, task.cancel))
        registered.set()
        try:
            return await asyncio.shield(task)
        except BaseException:
            task.cancel()
            raise

    async def fetch_owned_document(url: str) -> dict[str, Any]:
        """Govern each discovery/robots redirect without an extracted-page debit."""
        dependencies = replace(
            article._build_default_dependencies(cookies),
            parent=context,
            fetch_client=DefaultFetchClient(context),
            request_context=RuntimeRequestContext(source=profile, stage="discovery", user_id=request.context.user_id),
        )
        current_headers = dict(headers)
        current_cookies = {
            str(item["name"]): str(item.get("value", ""))
            for item in _selected_cookies(url, cookies, stored)
            if item.get("name")
            and (
                not item.get("domain")
                or urlsplit(url).hostname == str(item["domain"]).lstrip(".")
                or (urlsplit(url).hostname or "").endswith("." + str(item["domain"]).lstrip("."))
            )
        }
        for _ in range(11):
            context.check()
            url_bytes(url, plan)
            decision = await decide_web_outbound_policy(
                url,
                respect_robots=False,
                user_agent=current_headers.get("User-Agent"),
                source=profile,
                stage="discovery_fetch",
                config={"web_scraper": dict(config)},
            )
            if not decision.allowed:
                raise article.ArticleFailure("policy_error", "discovery")
            response = await article._fetch_response(
                dependencies,
                FetchRequest(
                    url=url,
                    headers=current_headers,
                    cookies=current_cookies,
                    timeout=context.cap_timeout(10),
                    allow_redirects=False,
                    max_response_bytes=article.ArticleLimits.from_mapping(config).max_article_bytes,
                ),
            )
            url_bytes(response.url, plan)
            location = article._redirect_location(response)
            if response.status not in {301, 302, 303, 307, 308} or not location:
                return {"text": response.text, "status": response.status, "url": response.url}
            target = resolve_reference(response.url or url, location, plan)
            if urlsplit(target).scheme not in {"http", "https"}:
                raise article.ArticleFailure("fetch_error", "redirect")
            if article._redirect_crosses_credential_boundary(url, target):
                current_headers, current_cookies = article._redirect_boundary_headers(current_headers), {}
            url = target
        raise article.ArticleFailure("fetch_error", "redirect")

    policy = ProfileTraversalPolicy(
        plan, config=config, headers=headers, fetch_document=fetch_document, hooks=hooks, controls=context
    )

    def progress(snapshot: Mapping[str, Any]) -> None:
        nonlocal committed
        if profile in {"enhanced_recursive", "enhanced_url_level"}:
            _observe(log_gauge, "webscraping.crawl.queue_size", float(snapshot["queue_size"]))
            if snapshot["pages_scraped"] > committed and policy.active_entry is not None:
                entry = policy.active_entry
                logger.debug(
                    "Accept URL [{} depth={}]: {}",
                    plan.options["crawl_strategy"],
                    entry.depth,
                    redact_url_for_log(entry.url),
                )
                _observe(log_counter, "webscraping.crawl.pages_crawled")
        committed = snapshot["pages_scraped"]
        if hooks.on_progress:
            hooks.on_progress(snapshot)

    def start_entry(entry: FrontierEntry, execution: Mapping[str, Any] | None = None) -> WorkHandle[PageData]:
        """Return ownership before canonical routing, acquisition or extraction."""
        page_config, page_headers, page_cookies = (
            (config, headers, _selected_cookies(entry.url, cookies, stored))
            if execution is None
            else (execution["configuration"], execution["custom_headers"], execution["custom_cookies"])
        )
        root_sitemap = profile.endswith("sitemap") and entry.parent_url is None
        level_excluded = profile == "legacy_url_level" and len(
            urlsplit(entry.url).path.strip("/").split("/")
        ) > plan.options.get("level", 3)
        if root_sitemap or level_excluded:
            owner = PhysicalWork()

            async def document() -> PageData:
                token = physical_work.set(owner)
                try:
                    response = await fetch_document(entry.url)
                    if response["status"] >= 400:
                        raise RuntimeError("discovery transport failed")
                    return PageData(
                        {},
                        discover(
                            response["text"],
                            request_url=entry.url,
                            final_url=response["url"],
                            plan=plan,
                            controls=context,
                            hooks=hooks,
                        ),
                    )
                finally:
                    physical_work.reset(token)

            task = asyncio.create_task(document())
            task.add_done_callback(lambda _done: owner.finish())
            return WorkHandle(task, owner.drained, task.cancel)

        def prepare_plan(
            page_plan: article.ArticlePlan,
            snapshot_headers: Mapping[str, str] = MappingProxyType(dict(page_headers)),
            legacy: bool = profile == "legacy_recursive",
            method: Any = plan.options.get("method"),
            explicit_agent: Any = requested_agent if execution is None else execution.get("user_agent"),
        ) -> article.ArticlePlan:
            return _prepare_page_plan(
                page_plan,
                headers=snapshot_headers,
                legacy=legacy,
                method=method,
                explicit_agent=explicit_agent,
                enhanced=profile.startswith("enhanced_"),
                custom_cookies=page_cookies,
            )

        def discovery_gate(result: Mapping[str, Any]) -> bool:
            recursive = profile in {"legacy_recursive", "enhanced_recursive", "enhanced_url_level"}
            if profile in {"enhanced_batch", "legacy_sitemap", "enhanced_sitemap"}:
                return False
            if recursive:
                depth = plan.options.get("max_depth")
                maximum = plan.options.get("max_pages")
                return (depth is None or entry.depth < depth) and (
                    maximum is None or committed + int(bool(result.get("extraction_successful"))) < maximum
                )
            return True

        def links(document: str, request_url: str, final_url: str) -> DiscoveryResult:
            found = discover(document, request_url=request_url, final_url=final_url, plan=plan, controls=context)
            if profile in {"enhanced_recursive", "enhanced_url_level"}:
                identities = set()
                for link in found.links:
                    if link.url in identities:
                        _skip("duplicate", str(plan.options["crawl_strategy"]), entry)
                    identities.add(link.url)
            return found

        return article.start_page(
            entry.url,
            list(page_cookies),
            controls=context,
            request_context=RuntimeRequestContext(source=profile, stage="fetch", user_id=request.context.user_id),
            discover=links,
            prepare_plan=prepare_plan,
            configuration=page_config if "web_scraper" in page_config else {"web_scraper": page_config},
            allow_llm_extraction=plan.options["allow_llm_extraction"],
            discovery_on_success=profile in {"enhanced_recursive", "enhanced_url_level"},
            discovery_gate=discovery_gate,
            transform=transform,
            method_intent="trafilatura" if profile == "enhanced_recursive" else "auto",
            browser_http_fallback=browser_http_fallback,
        )

    async def acquire(entry: FrontierEntry) -> WorkHandle[PageData]:
        """Keep direct traversal default; queued attempts reuse the same conversion."""
        if queue is None:
            return start_entry(entry)

        def make_job(url: str) -> ScrapingJob:
            # Charge normalized execution inputs even though the trusted callable
            # is a separate, live capability rather than JSON metadata.
            return ScrapingJob(
                str(uuid4()),
                url,
                "auto",
                user_id=str(request.context.user_id) if request.context.user_id is not None else None,
                metadata=_to_plain(
                    {
                        "configuration": config,
                        "custom_headers": headers,
                        "user_agent": requested_agent,
                        "custom_cookies": _selected_cookies(url, cookies, stored),
                        "crawl_options": plan.options,
                    }
                ),
            )

        def execute(payload, controls):
            if controls is not context:
                raise RuntimeError("page execution owner mismatch")
            return start_entry(replace(entry, url=payload.url), payload.options)

        queued_handles[:] = [handle for handle in queued_handles if not handle.drained.done()]
        handle = await queued_page(
            entry, context=context, queue=queue, make_job=make_job, execute=execute, owned=queued_handles
        )
        queued_handles.append(handle)
        return _queued_crawl_page(handle, entry=entry, context=context)

    try:
        return await run_crawl(
            plan,
            context=context,
            acquire=acquire,
            checkpoint=checkpoint,
            hooks=replace(hooks, on_progress=progress),
            seeds=seeds,
            policy=policy,
        )
    finally:
        # Exception tracebacks can outlive this owner. Physical handles and the
        # engine's drain retainer, not completed coroutine frames, own live state.
        private.clear()
        transform = None
        stored = None
        requested_agent = None
        config = headers = header_copy = credentials = options = cookies = hooks = policy = acquire = fetch_document = (
            None
        )
