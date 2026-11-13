"""Deterministic boundaries for characterization of the current crawl adapters."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import pytest

PREFIX = "tldw_Server_API.app.core.Web_Scraping"
LEGACY = f"{PREFIX}.Article_Extractor_Lib"
ENHANCED = f"{PREFIX}.enhanced_web_scraping.EnhancedWebScraper"
SERVICE = "tldw_Server_API.app.services.enhanced_web_scraping_service.WebScrapingService"


def bind_enhanced_capabilities(scraper, dependencies):
    """Bind fixture routing/extraction at the instance boundary now used by crawls."""

    def route(url):
        plan = dependencies.resolve_plan(url, {"web_scraper": scraper.config})
        raw_backend = dependencies.backend_setting(plan) if dependencies.backend_setting else plan.backend
        return (
            SimpleNamespace(
                **{
                    **{field.name: getattr(plan, field.name) for field in fields(plan)},
                    "backend": raw_backend,
                    "extra_headers": plan.headers,
                }
            ),
            plan.backend,
            plan.handler,
        )

    def extract(html, url, **kwargs):
        data = dict(dependencies.extract(html, url, **kwargs))
        if data.get("content"):
            data["content"] = dependencies.convert_content(data["content"])
        return data

    scraper._resolve_scrape_plan = route
    scraper._extract_from_html_with_pipeline = extract


def resolve(path: str) -> Any:
    """Resolve both module exports and class members without copying adapters."""
    module, _, name = path.rpartition(".")
    try:
        owner = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        owner = resolve(module)
    return getattr(owner, name)


@dataclass(frozen=True)
class ProfileManifest:
    """Entry points whose baseline is exercised by each named profile."""

    profiles: dict[str, tuple[str, ...]]

    @property
    def entrypoints(self) -> set[str]:
        return {path for paths in self.profiles.values() for path in paths}


PROFILE_MANIFEST = ProfileManifest(
    {
        "legacy_recursive": (f"{LEGACY}.recursive_scrape", f"{LEGACY}.sync_recursive_scrape"),
        "enhanced_recursive": (f"{ENHANCED}.recursive_scrape",),
        "legacy_sitemap": (f"{LEGACY}.scrape_from_sitemap",),
        "legacy_url_level": (f"{LEGACY}.scrape_by_url_level",),
        "enhanced_sitemap": (f"{ENHANCED}.scrape_sitemap",),
        "enhanced_url_level": (f"{SERVICE}._scrape_by_url_level",),
        "enhanced_batch": (f"{ENHANCED}.scrape_multiple",),
    }
)


@dataclass(frozen=True)
class CrawlImportInventory:
    """Current import consumers expanded to their callable crawl surface."""

    crawl_entrypoints: set[str]
    consumers: dict[str, set[str]]


@lru_cache(maxsize=1)
def current_crawl_inventory() -> CrawlImportInventory:
    """Scan actual imports, expanding module aliases and class imports to methods.

    The older JSON artifact inventories imported classes/modules, not their crawl
    methods. Keep it unchanged; expand the current scanner's records here instead.
    Scope is the seven Phase 5 profiles, not unused legacy conversion utilities.
    """
    from Helper_Scripts.web_scraping_refactor_inventory import TARGET_MODULES, inventory_for_roots

    root = Path(__file__).resolve().parents[3]
    service_module = SERVICE.rpartition(".")[0]
    targets = {**TARGET_MODULES, "enhanced_service": service_module}
    records = inventory_for_roots(root, [root / "tldw_Server_API/app", root / "tldw_Server_API/tests"], targets=targets)
    scopes = {
        "Article_Extractor_Lib": (
            LEGACY,
            ("recursive_scrape", "sync_recursive_scrape", "scrape_from_sitemap", "scrape_by_url_level"),
        ),
        "enhanced_web_scraping": (ENHANCED, ("recursive_scrape", "scrape_sitemap", "scrape_multiple")),
        "enhanced_service": (SERVICE, ("_scrape_by_url_level",)),
    }
    consumers: dict[str, set[str]] = {}
    for target, (owner, names) in scopes.items():
        namespace_names = {None, target, owner.rpartition(".")[2]}
        for record in records.get(target, []):
            imported = record.imported_name
            members = names if imported in namespace_names else (imported,)
            for name in members:
                if name not in names:
                    continue
                path = f"{owner}.{name}"
                if not callable(resolve(path)):
                    raise AssertionError(f"Imported crawl entry is not callable: {path}")
                consumers.setdefault(path, set()).add(record.path)
    return CrawlImportInventory(set(consumers), consumers)


def current_crawl_entrypoints() -> set[str]:
    """Return the public crawl import inventory independently of the manifest."""
    return current_crawl_inventory().crawl_entrypoints


class FakeClock:
    """Advance crawl delays without wall-clock sleeps or global clock patches."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


class FakeResponse(dict):
    """Shared HTTP boundary shape with observable synchronous/asynchronous close."""

    def __init__(self, text: str) -> None:
        super().__init__(status=200, text=text)
        self.text = text
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


class Phase5Harness:
    """Run real traversal, parser, queue and service adapters with inert pages.

    Legacy recursion also runs the real guarded-session and async article adapter;
    only browser acquisition, extraction execution and delays are replaced. Other
    profiles replace per-page execution, not traversal or batch implementations.
    Queue workers/retries are real and always drained. Files live in a temporary
    directory, never the default scrape_progress.json in the shared worktree.
    """

    def __init__(self, *, profile: str, outcomes: Sequence[dict[str, Any]]) -> None:
        if profile not in PROFILE_MANIFEST.profiles:
            raise ValueError(f"Unknown profile: {profile}")
        if not outcomes:
            raise ValueError("At least one URL outcome is required")
        self.profile = profile
        self.outcomes = {item["url"]: deepcopy(item) for item in outcomes}
        self.seed = next(iter(self.outcomes))
        self.clock = FakeClock()
        self.transport_started = asyncio.Event()
        self.transport_release = asyncio.Event()
        self.transport_release.set()
        self.executor_started = asyncio.Event()
        self.executor_release = asyncio.Event()
        self.executor_release.set()
        self.calls: list[dict[str, Any]] = []
        self.unexpected_urls: list[str] = []
        self.transport_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.policy_calls: list[dict[str, Any]] = []
        self.callbacks: list[str] = []
        self.jobs: list[Any] = []
        self.responses: list[FakeResponse] = []
        self.credentials = {"custom_cookies": [{"name": "session", "value": "test-cookie", "domain": "example.com"}]}
        self.config: dict[str, Any] = {}
        self.transport_allowed = True
        self.denied_urls: set[str] = set()
        self.sitemap_url = "https://example.com/sitemap.xml"
        self.sitemap_kind = "namespaced"
        self.scraper: Any = None
        self.service: Any = None

    @property
    def attempted_urls(self) -> list[str]:
        """Logical crawl submissions; physical queue retries remain in calls."""
        if self.profile.startswith("enhanced"):
            return [job.url for job in self.jobs]
        if self.profile == "legacy_recursive":
            return [call["url"] for call in self.transport_calls]
        return [call["url"] for call in self.calls]

    def article(self, url: str) -> dict[str, Any]:
        """Return a fresh canonical result; nested fixtures expose lossy adapters."""
        outcome = self.outcomes.get(url)
        if outcome is None and url != "https://example.com/child-sitemap.xml":
            self.unexpected_urls.append(url)
            raise AssertionError(f"Unexpected page execution: {url}")
        outcome = outcome or {"url": url}
        return {
            "url": url,
            "title": "Fixture title",
            "author": "Fixture author",
            "date": "2026-11-01",
            "content": "kept article",
            "extraction_successful": True,
            "metadata": {"nested": {"labels": ["baseline"]}},
            "custom_extra": {"rank": 7},
            "preflight_analysis": {
                "analysis": {"results": {"js": {"js_required": True}}},
                "advice": {"backend": "curl", "method": "playwright", "notes": ["js_required"]},
            },
            **deepcopy({k: v for k, v in outcome.items() if k not in {"links", "transport_error"}}),
        }

    def _html(self, url: str) -> str:
        outcome = {} if url == "https://example.com/child-sitemap.xml" else self.outcomes[url]
        links = "".join(f'<a href="{escape(link, quote=True)}">link</a>' for link in outcome.get("links", []))
        return f"<html><title>Fixture title</title><body>{links}</body></html>"

    def _response(self, url: str, **kwargs: Any) -> FakeResponse:
        self.fetch_calls.append({"url": url, **deepcopy(kwargs)})
        if url == self.sitemap_url:
            locs = "".join(f"<url><loc>{escape(page)}</loc></url>" for page in self.outcomes)
            ns = ' xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
            text = {
                "namespaced": f"<urlset{ns}>{locs}</urlset>",
                "plain": f"<urlset>{locs}</urlset>",
                "index": f"<sitemapindex{ns}><sitemap><loc>https://example.com/child-sitemap.xml</loc></sitemap></sitemapindex>",
                "malformed": f"<urlset{ns}><url><loc>{escape(self.seed)}</loc>",
            }[self.sitemap_kind]
        else:
            text = self._html(url)
        response = FakeResponse(text)
        self.responses.append(response)
        return response

    def _policy(self, url: str, **kwargs: Any) -> Any:
        self.policy_calls.append({"url": url, **kwargs})
        decision = resolve(f"{PREFIX}.outbound_policy.WebOutboundPolicyDecision")
        return decision(
            allowed=url not in self.denied_urls,
            mode="strict",
            reason="robots_unreachable" if url in self.denied_urls else "allowed",
            source=kwargs.get("source", "test"),
            stage=kwargs.get("stage", "pre_fetch"),
        )

    def _page_sync(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"url": url, "kwargs": deepcopy(kwargs)})
        return self.article(url)

    async def _page_async(self, url: str, method: str = "auto", **kwargs: Any) -> dict[str, Any]:
        self.executor_started.set()
        await self.executor_release.wait()
        return self._page_sync(url, method=method, **kwargs)

    @contextmanager
    def boundaries(self) -> Iterator[None]:
        """Patch only external/page boundaries; do not stub traversal results."""
        legacy = importlib.import_module(LEGACY)
        enhanced = importlib.import_module(f"{PREFIX}.enhanced_web_scraping")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(legacy, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": self.clock.sleep}))
            decision_type = resolve(f"{PREFIX}.browser_transport.BrowserTransportDecision")
            patch.setattr(
                legacy,
                "resolve_browser_transport_decision",
                lambda *_a, **_kw: decision_type(
                    allowed=self.transport_allowed,
                    configured_mode="url_guarded" if self.transport_allowed else "disabled",
                    effective_mode="url_guarded" if self.transport_allowed else "disabled",
                    dns_peer_attested=False,
                    reason=(
                        "browser_transport_allowed_legacy" if self.transport_allowed else "browser_transport_disabled"
                    ),
                ),
            )

            async def acquire(session: Any, url: str) -> str:
                self.transport_calls.append(
                    {"url": url, "cookies": session._profile.custom_cookies, "user_agent": session._profile.user_agent}
                )
                self.transport_started.set()
                await self.transport_release.wait()
                if url not in self.outcomes:
                    self.unexpected_urls.append(url)
                if self.outcomes[url].get("transport_error"):
                    failure = resolve(f"{PREFIX}.orchestration.article_models.ArticleFailure")
                    raise failure("browser_transport_unavailable", "acquire")
                return self._html(url)

            async def execute(func: Any, *args: Any, **kwargs: Any) -> Any:
                self.executor_started.set()
                await self.executor_release.wait()
                return func(*args, **kwargs)

            async def async_fetch(**kwargs: Any) -> FakeResponse:
                return self._response(**kwargs)

            async def async_policy(url: str, **kwargs: Any) -> Any:
                return self._policy(url, **kwargs)

            patch.setattr(legacy._LegacyGuardedBrowserSession, "acquire", acquire)
            patch.setattr(legacy, "run_extraction_in_thread", execute)
            patch.setattr(legacy, "extract_article_with_pipeline", lambda _html, url, **kw: self._page_sync(url, **kw))
            patch.setattr(legacy, "scrape_article_blocking", self._page_sync)
            patch.setattr(legacy, "http_fetch", self._response)
            patch.setattr(legacy, "decide_web_outbound_policy_sync", self._policy)
            patch.setattr(enhanced, "afetch", async_fetch)
            patch.setattr(enhanced, "decide_web_outbound_policy", async_policy)
            patch.setattr(enhanced, "decide_web_outbound_policy_sync", self._policy)
            from tldw_Server_API.app.core.Web_Scraping.orchestration import article, crawl
            from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse
            from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import (
                _allowed_target,
                _harness,
                _plan,
            )

            patch.setattr(crawl, "decide_web_outbound_policy", async_policy)
            patch.setattr(crawl, "resolve_browser_transport_decision", legacy.resolve_browser_transport_decision)

            class Browser:
                async def acquire(inner, url, profile, limits):
                    return await acquire(SimpleNamespace(_profile=profile), url)

            class Fetch:
                def fetch(inner, request):
                    response = self._response(request.url, headers=dict(request.headers), cookies=dict(request.cookies))
                    response.close()
                    return FetchResponse(
                        url=request.url,
                        status=200,
                        text=response.text,
                        headers={},
                        backend="httpx",
                    )

            class Executor:
                async def run(inner, func, *args, **kwargs):
                    return await execute(func, *args, **kwargs)

            async def target(url, **kwargs):
                context = kwargs.get("request_context")
                return replace(
                    _allowed_target(url),
                    decision=self._policy(
                        url,
                        source=getattr(context, "source", "article"),
                        stage=getattr(context, "stage", "pre_fetch"),
                        **kwargs,
                    ),
                )

            def dependencies(cookies):
                fixture = _harness()

                def plan(*_args):
                    value = _plan()
                    return replace(value, browser=replace(value.browser, custom_cookies=cookies))

                return replace(
                    fixture.dependencies,
                    browser=Browser(),
                    fetch_client=Fetch(),
                    executor=Executor(),
                    evaluate_target=target,
                    resolve_plan=plan,
                    convert_content=lambda value: value,
                    extract=lambda _html, url, **kwargs: self._page_sync(
                        url,
                        custom_cookies=[dict(item) for item in cookies],
                        **kwargs,
                    ),
                )

            patch.setattr(article, "_build_default_dependencies", dependencies)

            async def response(_dependencies, request):
                value = self._response(request.url, headers=dict(request.headers), cookies=dict(request.cookies))
                value.close()
                return FetchResponse(url=request.url, status=200, text=value.text, headers={}, backend="httpx")

            patch.setattr(article, "_fetch_response", response)
            if self.profile.startswith("enhanced"):
                self.scraper = enhanced.EnhancedWebScraper(config={"max_concurrent": 1, **self.config})
                patch.setattr(self.scraper, "scrape_article", self._page_async)
                patch.setattr(
                    self.scraper,
                    "_extract_from_html_with_pipeline",
                    lambda _html, url, **kwargs: self._page_sync(
                        url,
                        custom_cookies=self.credentials["custom_cookies"],
                        **{key: value for key, value in kwargs.items() if key != "fallback_extractor"},
                    ),
                )
                add_job = self.scraper.job_queue.add_job

                async def submit(job: Any) -> Any:
                    self.jobs.append(job)
                    return await add_job(job)

                patch.setattr(self.scraper.job_queue, "add_job", submit)
                submit_page = self.scraper.job_queue.submit_page

                async def submit_owned(job: Any, **kwargs: Any) -> Any:
                    if not (self.profile.endswith("sitemap") and job.url == self.sitemap_url):
                        self.jobs.append(job)
                    return await submit_page(job, **kwargs)

                patch.setattr(self.scraper.job_queue, "submit_page", submit_owned)
                patch.setattr(self.scraper, "_apply_dedup", lambda _url, data: data)
            try:
                yield
            finally:
                assert not self.unexpected_urls, f"Unexpected executed URLs: {self.unexpected_urls}"

    def _legacy(self, directory: str, *, sync: bool = False, **options: Any) -> Any:
        legacy = importlib.import_module(LEGACY)
        base = options.pop("base_url", self.seed)
        if self.profile == "legacy_recursive":
            frontier = options.pop("resume_frontier", None)
            progress = str(Path(directory) / "legacy-progress.json")
            if frontier is not None:
                Path(progress).write_text(
                    json.dumps(
                        {
                            "visited": [],
                            "to_visit": frontier,
                            "scraped_articles": [],
                            "pages_scraped": 0,
                        }
                    ),
                    encoding="utf-8",
                )
            if sync:
                return legacy.sync_recursive_scrape(
                    base,
                    options.pop("max_pages", 10),
                    options.pop("max_depth", 3),
                    delay=0,
                    **options,
                )
            return legacy.recursive_scrape(
                base,
                options.pop("max_pages", 10),
                options.pop("max_depth", 3),
                delay=0,
                resume_file=progress,
                progress_callback=self.callbacks.append,
                **options,
            )
        if self.profile == "legacy_sitemap":
            return legacy.scrape_from_sitemap(self.sitemap_url, **options)
        return legacy.scrape_by_url_level(base, options.pop("level", 3), **options)

    async def run(self, **options: Any) -> Any:
        """Call the current entry adapter, with no synthesized traversal result."""
        if self.profile in {"legacy_sitemap", "legacy_url_level"}:
            return await asyncio.to_thread(self.run_sync, **options)
        with self.boundaries(), TemporaryDirectory(prefix="phase5-crawl-") as directory:
            if self.profile == "legacy_recursive":
                return await self._legacy(directory, **options)
            await self.scraper.job_queue.start()
            try:
                if self.profile == "enhanced_batch":
                    return await self.scraper.scrape_multiple(list(self.outcomes), **options)
                if self.profile == "enhanced_sitemap":
                    return await self.scraper.scrape_sitemap(self.sitemap_url, **options)
                base = options.pop("base_url", self.seed)
                if self.profile == "enhanced_recursive":
                    return await self.scraper.recursive_scrape(base, **options)
                self.service = resolve(SERVICE)()
                self.service.scraper = self.scraper
                return await self.service._scrape_by_url_level(
                    base,
                    options.pop("level", 3),
                    options.pop("max_pages", 10),
                    options.pop("summarize", False),
                    None,
                    None,
                    None,
                    "fixture",
                    None,
                    0.7,
                    resolve(f"{PREFIX}.enhanced_web_scraping.JobPriority").NORMAL,
                    options.pop("custom_cookies", None),
                    options.pop("user_agent", None),
                    options.pop("custom_headers", None),
                    **options,
                )
            finally:
                await self.scraper.job_queue.stop()

    def run_sync(self, **options: Any) -> list[dict[str, Any]]:
        """Exercise synchronous legacy adapters without returning awaitables."""
        with self.boundaries(), TemporaryDirectory(prefix="phase5-crawl-") as directory:
            if self.profile == "legacy_recursive":
                # The sync adapter has no resume_file argument: isolate its default path.
                legacy = importlib.import_module(LEGACY)
                recursive = legacy.recursive_scrape

                async def isolated(*args: Any, **kwargs: Any) -> Any:
                    return await recursive(*args, resume_file=str(Path(directory) / "progress.json"), **kwargs)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(legacy, "recursive_scrape", isolated)
                    return self._legacy(directory, sync=True, **options)
            return self._legacy(directory, sync=True, **options)


def make_harness(*, profile: str, outcomes: Sequence[dict[str, Any]]) -> Phase5Harness:
    """Build an isolated characterization harness for one public entry profile."""
    return Phase5Harness(profile=profile, outcomes=outcomes)
