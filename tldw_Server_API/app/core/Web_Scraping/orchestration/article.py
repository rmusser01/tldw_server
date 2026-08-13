"""Canonical governed orchestration for standard asynchronous article scraping."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from loguru import logger

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.content import convert_html_to_markdown
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_with_pipeline
from tldw_Server_API.app.core.Web_Scraping.handlers import resolve_handler
from tldw_Server_API.app.core.Web_Scraping.observability import sanitized_host
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import GuardedArticleBrowser
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
    ArticlePlan,
    article_failure_result,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import (
    DEFAULT_EXTRACTION_EXECUTOR,
    ExtractionExecutorManager,
)
from tldw_Server_API.app.core.Web_Scraping.policy import (
    DefaultProbeEgressGuard,
    DefaultWebOutboundPolicyChecker,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    DefaultFetchClient,
    FetchClient,
    FetchRequest,
    FetchResponse,
    RuntimeRequestContext,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter

_FETCH_TIMEOUT_SECONDS = 15.0
_AUTO_BACKEND = "auto"
_PLAYWRIGHT = "playwright"
_HTTP_BACKENDS = frozenset({_AUTO_BACKEND, "curl", "httpx"})
_SAFE_POLICY_FAILURE = "policy_error"
_RESPONSE_TOO_LARGE_MESSAGE = "Response exceeds max_response_bytes limit"
_JS_REQUIRED_DOMAINS = frozenset(
    {
        "medium.com",
        "substack.com",
        "notion.site",
        "notion.so",
        "webflow.io",
        "squarespace.com",
        "wixsite.com",
        "x.com",
        "twitter.com",
        "tiktok.com",
        "instagram.com",
        "facebook.com",
        "linkedin.com",
    }
)


@dataclass(frozen=True, slots=True)
class ArticleDependencies:
    """Injectable collaborators for a single standard article request."""

    load_config: Callable[[], Mapping[str, Any]]
    resolve_plan: Callable[[str, Mapping[str, Any]], ArticlePlan]
    evaluate_target: Callable[..., Awaitable[Any]]
    run_preflight: Callable[..., Awaitable[Any]]
    apply_preflight_advice: Callable[..., tuple[str, str, Any]]
    fetch_client: FetchClient
    browser: GuardedArticleBrowser
    executor: ExtractionExecutorManager
    extract: Callable[..., dict[str, Any]]
    build_preflight_context: Callable[..., Any]
    preflight_options: Callable[[Mapping[str, Any]], Any]
    public_preflight_payload: Callable[[Any, bool], dict[str, Any] | None]
    resolve_handler: Callable[[str], Callable[[str, str], dict[str, Any]] | None]
    js_required: Callable[[str, Mapping[str, Any], str | None], bool]
    convert_content: Callable[[str], str]
    increment_counter: Callable[..., None]
    observe_histogram: Callable[..., None]
    clock: Callable[[], float]
    log: Callable[..., None]
    policy_checker: Any = None
    backend_setting: Callable[[ArticlePlan], str] | None = None


def _snapshot_config(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Capture the configuration map once without retaining caller-owned maps."""
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return _snapshot_mapping(value)


def _snapshot_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _snapshot_value(item) for key, item in value.items()})


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _snapshot_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_snapshot_value(item) for item in value)
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return value


def _web_scraper_values(config: Mapping[str, Any]) -> Mapping[str, Any]:
    values = config.get("web_scraper")
    if isinstance(values, Mapping):
        return values
    return MappingProxyType({})


def _fallback_plan(
    url: str, config: Mapping[str, Any], custom_cookies: Sequence[Mapping[str, Any]] | None
) -> ArticlePlan:
    """Create the same safe generic route used when router construction fails."""
    parsed = urlsplit(url)
    routing = SimpleNamespace(
        url=url,
        domain=parsed.netloc,
        backend="auto",
        handler="tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
        ua_profile="chrome_120_win",
        impersonate=None,
        extra_headers={},
        cookies={},
        proxies={},
        respect_robots=True,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
    )
    return ArticlePlan.from_routing_plan(routing, config, custom_cookies)


def _policy_blocked_result(url: str, decision: Any) -> dict[str, Any]:
    reason = str(getattr(decision, "reason", "policy_denied") or "policy_denied")
    return {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "Blocked by outbound policy" if reason.startswith("robots_") else f"Egress denied: {reason}",
        "policy_reason": reason,
        "policy_mode": str(getattr(decision, "mode", "compat") or "compat"),
        "policy_stage": str(getattr(decision, "stage", "pre_fetch") or "pre_fetch"),
        "policy_source": str(getattr(decision, "source", "article_extract") or "article_extract"),
    }


def _failure_result(url: str, code: str) -> dict[str, Any]:
    return {"url": url, **article_failure_result(ArticleFailure(code, "article"))}


def _attach_preflight(result: Mapping[str, Any], payload: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(result)
    if payload is not None:
        copied.setdefault("preflight_analysis", dict(payload))
    return copied


def _bounded_backend(value: str) -> str:
    normalized = str(value or _AUTO_BACKEND).lower().strip()
    return normalized if normalized in _HTTP_BACKENDS | {_PLAYWRIGHT} else _AUTO_BACKEND


def _js_required(html: str, headers: Mapping[str, Any], url: str | None = None) -> bool:
    """Return whether lightweight HTML should be retried through Playwright."""
    del headers
    try:
        domain = urlsplit(url).netloc.lower() if url else ""
        text = html.lower()
        if not text.strip():
            return True
        js_phrases = (
            "enable javascript",
            "please enable javascript",
            "requires javascript",
            "enable your javascript",
            "javascript is disabled",
            "please turn on javascript",
            "please turn on js",
        )
        if any(phrase in text for phrase in js_phrases):
            return True
        if "<noscript" in text and any(
            phrase in text for phrase in ("enable javascript", "javascript is disabled", "requires javascript")
        ):
            return True
        bot_phrases = (
            "cf-browser-verification",
            "cf-chl-bypass",
            "cloudflare ray id",
            "attention required",
            "checking your browser",
            "verify you are human",
            "hcaptcha",
            "recaptcha",
            "turnstile",
            "just a moment",
        )
        if any(phrase in text for phrase in bot_phrases):
            return True
        if ('http-equiv="refresh"' in text or "http-equiv='refresh'" in text) and len(text) < 1_500:
            return True
        soup = BeautifulSoup(html, "html.parser")
        visible_text = soup.get_text(" ", strip=True)
        visible_len = len(visible_text)
        script_count = len(soup.find_all("script"))
        if script_count >= 25 and visible_len < 800:
            return True
        if (
            script_count >= 10
            and visible_len < 400
            and ("__next" in text or "__nuxt" in text or "data-reactroot" in text)
        ):
            return True
        if script_count >= 1 and visible_len < 600:
            for shell_id in ("__next", "__nuxt", "root", "app", "app-root"):
                if f'id="{shell_id}"' in text or f"id='{shell_id}'" in text:
                    return True
        if ("data-reactroot" in text or "data-reactid" in text) and visible_len < 600:
            return True
        if domain and any(
            domain == candidate or domain.endswith(f".{candidate}") for candidate in _JS_REQUIRED_DOMAINS
        ):
            return visible_len < 1_200 or (script_count >= 15 and visible_len < 2_500)
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return False
    return False


def _record_counter(dependencies: ArticleDependencies, name: str, labels: Mapping[str, str]) -> None:
    try:
        dependencies.increment_counter(name, labels=dict(labels))
    except Exception:  # noqa: BLE001 - metrics must not change article outcomes
        return


def _record_histogram(
    dependencies: ArticleDependencies,
    name: str,
    value: float,
    labels: Mapping[str, str],
) -> None:
    try:
        dependencies.observe_histogram(name, value, labels=dict(labels))
    except Exception:  # noqa: BLE001 - metrics must not change article outcomes
        return


def _log_failure(
    dependencies: ArticleDependencies,
    exc: BaseException,
    *,
    code: str,
    stage: str,
    url: str,
) -> None:
    try:
        dependencies.log(
            "Article orchestration failure.",
            exception_type=type(exc).__name__[:80],
            code=code,
            stage=stage,
            host=sanitized_host(url),
        )
    except Exception:  # noqa: BLE001 - logging must not change article outcomes
        return


async def _fetch_response(dependencies: ArticleDependencies, request: FetchRequest) -> FetchResponse:
    return await asyncio.to_thread(dependencies.fetch_client.fetch, request)


def _response_too_large_error(exc: ValueError) -> ArticleFailure | None:
    if type(exc) is ValueError and str(exc) == _RESPONSE_TOO_LARGE_MESSAGE:
        return ArticleFailure("response_too_large", "fetch")
    return None


async def _fetch_lightweight(
    dependencies: ArticleDependencies,
    plan: ArticlePlan,
    *,
    url: str,
    cookies: Mapping[str, str],
    backend: str,
) -> tuple[FetchResponse, str]:
    def request_for(selected_backend: str) -> FetchRequest:
        return FetchRequest(
            url=url,
            method="GET",
            headers=plan.headers,
            cookies=cookies,
            timeout=_FETCH_TIMEOUT_SECONDS,
            backend=selected_backend,
            allow_redirects=True,
            impersonate=plan.impersonate,
            proxies=plan.proxies,
            context=RuntimeRequestContext(source="article_extract", stage="fetch"),
            max_response_bytes=plan.limits.max_article_bytes,
        )

    async def fetch_backend(selected_backend: str) -> tuple[FetchResponse, str]:
        try:
            response = await _fetch_response(dependencies, request_for(selected_backend))
        except ValueError as exc:
            overflow = _response_too_large_error(exc)
            if overflow is not None:
                raise overflow from None
            raise
        return response, _bounded_backend(response.backend)

    if backend == "curl":
        try:
            return await fetch_backend("curl")
        except asyncio.CancelledError:
            raise
        except ArticleFailure:
            raise
        except Exception:  # noqa: BLE001 - curl transport falls back to HTTPX
            return await fetch_backend("httpx")

    return await fetch_backend("httpx")


async def _extract(
    dependencies: ArticleDependencies,
    plan: ArticlePlan,
    html: str,
    url: str,
    *,
    allow_llm_extraction: bool,
) -> dict[str, Any]:
    handler = dependencies.resolve_handler(plan.handler) if plan.handler else None
    result = await dependencies.executor.run(
        dependencies.extract,
        html,
        url,
        strategy_order=list(plan.strategy_order) if plan.strategy_order is not None else None,
        handler=handler,
        schema_rules=dict(plan.schema_rules) if plan.schema_rules is not None else None,
        llm_settings=dict(plan.llm_settings) if plan.llm_settings is not None else None,
        regex_settings=dict(plan.regex_settings) if plan.regex_settings is not None else None,
        cluster_settings=dict(plan.cluster_settings) if plan.cluster_settings is not None else None,
        allow_llm_extraction=allow_llm_extraction,
    )
    if not isinstance(result, Mapping):
        raise ArticleFailure("extraction_error", "result")
    copied = dict(result)
    if copied.get("extraction_successful") and handler is None and copied.get("content"):
        copied["content"] = dependencies.convert_content(str(copied["content"]))
    return copied


async def _run_article(
    url: str,
    custom_cookies: list[dict[str, Any]] | None,
    allow_llm_extraction: bool,
    *,
    dependencies: ArticleDependencies,
) -> dict[str, Any]:
    """Run one request through its immutable route and governed dependencies."""
    config: Mapping[str, Any]
    try:
        config = _snapshot_config(dependencies.load_config())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - config loading has one safe empty fallback
        _log_failure(dependencies, exc, code="fetch_error", stage="config", url=url)
        config = MappingProxyType({})

    try:
        plan = await asyncio.to_thread(dependencies.resolve_plan, url, config)
        if not isinstance(plan, ArticlePlan):
            raise TypeError("resolve_plan must return ArticlePlan")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - route loading preserves the config snapshot
        _log_failure(dependencies, exc, code="fetch_error", stage="plan", url=url)
        plan = _fallback_plan(url, config, custom_cookies)

    values = _web_scraper_values(config)
    effective_user_agent = plan.headers.get("User-Agent") or plan.browser.user_agent
    request_context = RuntimeRequestContext(source="article_extract", stage="pre_fetch")
    preflight_payload: dict[str, Any] | None = None
    try:
        target = await dependencies.evaluate_target(
            url,
            respect_robots=plan.respect_robots,
            user_agent=effective_user_agent,
            request_context=request_context,
            config={"web_scraper": values},
            policy_checker=dependencies.policy_checker,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - public policy failures are stable
        _log_failure(dependencies, exc, code=_SAFE_POLICY_FAILURE, stage="pre_fetch", url=url)
        return _failure_result(url, _SAFE_POLICY_FAILURE)

    if not bool(getattr(getattr(target, "decision", None), "allowed", False)):
        if str(getattr(target.decision, "reason", "")).startswith("robots_"):
            _record_counter(dependencies, "scrape_blocked_by_robots_total", {})
        return _policy_blocked_result(url, target.decision)

    options = dependencies.preflight_options(values)
    preflight_result = None
    if bool(getattr(options, "enabled", False)):
        try:
            context = dependencies.build_preflight_context(
                target,
                options,
                policy_checker=dependencies.policy_checker,
            )
            preflight_result = await dependencies.run_preflight(target, options, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the analyzer is intentionally fail-open
            _log_failure(dependencies, exc, code="fetch_error", stage="preflight", url=url)
            preflight_result = None

    backend = _bounded_backend(plan.backend)
    route_backend = _bounded_backend(
        dependencies.backend_setting(plan) if dependencies.backend_setting is not None else plan.backend
    )
    configured_backend = _bounded_backend(values.get("web_scraper_default_backend", _AUTO_BACKEND))
    automatic_backend = route_backend == _AUTO_BACKEND and configured_backend == _AUTO_BACKEND
    backend_setting = _AUTO_BACKEND if automatic_backend else backend
    try:
        advised_backend, advised_method, advised_result = dependencies.apply_preflight_advice(
            preflight_result,
            backend=backend,
            method=_AUTO_BACKEND,
            backend_setting=backend_setting,
        )
    except Exception as exc:  # noqa: BLE001 - failed advice retains the snapshotted route
        _log_failure(dependencies, exc, code="fetch_error", stage="preflight_advice", url=url)
        advised_backend, advised_method, advised_result = backend, _AUTO_BACKEND, None

    if not automatic_backend:
        advised_backend = backend
    try:
        preflight_payload = dependencies.public_preflight_payload(
            advised_result,
            bool(getattr(options, "include_results", False)),
        )
    except Exception as exc:  # noqa: BLE001 - payload creation is optional observability
        _log_failure(dependencies, exc, code="fetch_error", stage="preflight_payload", url=url)

    cookies: dict[str, str] = {}
    for cookie in custom_cookies or ():
        if isinstance(cookie, Mapping) and "name" in cookie and "value" in cookie:
            cookies[str(cookie["name"])] = str(cookie["value"])
    cookies.update({str(key): str(value) for key, value in plan.cookies.items()})

    if advised_backend != _PLAYWRIGHT and advised_method != _PLAYWRIGHT:
        started = dependencies.clock()
        js_required = False
        try:
            response, backend_used = await _fetch_lightweight(
                dependencies,
                plan,
                url=url,
                cookies=cookies,
                backend=_bounded_backend(advised_backend),
            )
            _record_histogram(
                dependencies,
                "scrape_fetch_latency_seconds",
                max(0.0, dependencies.clock() - started),
                {"backend": backend_used},
            )
            if response.status < 400 and response.text:
                if dependencies.js_required(response.text, response.headers, url):
                    js_required = True
                    _record_counter(
                        dependencies,
                        "scrape_playwright_fallback_total",
                        {"reason": "js_required"},
                    )
                else:
                    final_extraction = await _extract(
                        dependencies,
                        plan,
                        response.text,
                        url,
                        allow_llm_extraction=allow_llm_extraction,
                    )
                    if final_extraction.get("extraction_successful"):
                        content = str(final_extraction.get("content", "") or "")
                        _record_histogram(
                            dependencies,
                            "scrape_content_length_bytes",
                            float(len(content.encode("utf-8", errors="ignore"))),
                            {"backend": backend_used},
                        )
                        _record_counter(
                            dependencies, "scrape_fetch_total", {"backend": backend_used, "outcome": "success"}
                        )
                        return _attach_preflight(final_extraction, preflight_payload)
                    _record_counter(
                        dependencies, "scrape_fetch_total", {"backend": backend_used, "outcome": "no_extract"}
                    )
            else:
                _record_counter(dependencies, "scrape_fetch_total", {"backend": backend_used, "outcome": "no_extract"})
            if not js_required:
                _record_counter(dependencies, "scrape_playwright_fallback_total", {"reason": "no_extract"})
        except asyncio.CancelledError:
            raise
        except ArticleFailure as exc:
            if exc.code == "response_too_large":
                _log_failure(dependencies, exc, code=exc.code, stage=exc.stage, url=url)
                return _attach_preflight(_failure_result(url, exc.code), preflight_payload)
            _log_failure(dependencies, exc, code=exc.code, stage=exc.stage, url=url)
            _record_counter(dependencies, "scrape_playwright_fallback_total", {"reason": "error"})
        except Exception as exc:  # noqa: BLE001 - browser fallback remains eligible
            _log_failure(dependencies, exc, code="fetch_error", stage="fetch", url=url)
            _record_counter(dependencies, "scrape_playwright_fallback_total", {"reason": "error"})

    browser_started = dependencies.clock()
    try:
        html = await dependencies.browser.acquire(url, plan.direct_browser, plan.limits)
    except asyncio.CancelledError:
        raise
    except ArticleFailure as exc:
        _record_histogram(
            dependencies,
            "scrape_fetch_latency_seconds",
            max(0.0, dependencies.clock() - browser_started),
            {"backend": _PLAYWRIGHT},
        )
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "error"})
        _log_failure(dependencies, exc, code=exc.code, stage=exc.stage, url=url)
        return _attach_preflight(_failure_result(url, exc.code), preflight_payload)
    except Exception as exc:  # noqa: BLE001 - guard boundary must be stable
        _record_histogram(
            dependencies,
            "scrape_fetch_latency_seconds",
            max(0.0, dependencies.clock() - browser_started),
            {"backend": _PLAYWRIGHT},
        )
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "error"})
        _log_failure(dependencies, exc, code="browser_error", stage="acquire", url=url)
        return _attach_preflight(_failure_result(url, "browser_error"), preflight_payload)

    _record_histogram(
        dependencies,
        "scrape_fetch_latency_seconds",
        max(0.0, dependencies.clock() - browser_started),
        {"backend": _PLAYWRIGHT},
    )

    try:
        result = await _extract(
            dependencies,
            plan,
            html,
            url,
            allow_llm_extraction=allow_llm_extraction,
        )
    except asyncio.CancelledError:
        raise
    except ArticleFailure as exc:
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "error"})
        _log_failure(dependencies, exc, code=exc.code, stage=exc.stage, url=url)
        return _attach_preflight(_failure_result(url, exc.code), preflight_payload)
    except Exception as exc:  # noqa: BLE001 - executor/extraction boundary is stable
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "error"})
        _log_failure(dependencies, exc, code="extraction_error", stage="extract", url=url)
        return _attach_preflight(_failure_result(url, "extraction_error"), preflight_payload)

    if result.get("extraction_successful"):
        content = str(result.get("content", "") or "")
        _record_histogram(
            dependencies,
            "scrape_content_length_bytes",
            float(len(content.encode("utf-8", errors="ignore"))),
            {"backend": _PLAYWRIGHT},
        )
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "success"})
    else:
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "no_extract"})
    return _attach_preflight(result, preflight_payload)


def _default_rules_path() -> str:
    return str(Path(__file__).resolve().parents[4] / "Config_Files" / "custom_scrapers.yaml")


def _build_default_dependencies(custom_cookies: Sequence[Mapping[str, Any]] | None) -> ArticleDependencies:
    """Build the live standard-article dependencies at request time."""
    resolved_backend_settings: dict[int, str] = {}

    def resolve_plan(url: str, config: Mapping[str, Any]) -> ArticlePlan:
        values = _web_scraper_values(config)
        rules_path = values.get("custom_scrapers_yaml_path", _default_rules_path())
        rules = ScraperRouter.load_rules_from_yaml(rules_path)
        ua_mode = str(values.get("web_scraper_ua_mode", "fixed") or "fixed")
        respect_robots = values.get("web_scraper_respect_robots", True)
        if isinstance(respect_robots, str):
            respect_robots = is_truthy(respect_robots.strip())
        route = ScraperRouter(
            rules,
            ua_mode=ua_mode,
            default_respect_robots=bool(respect_robots),
        ).resolve(url)
        plan = ArticlePlan.from_routing_plan(route, config, custom_cookies)
        resolved_backend_settings[id(plan)] = _bounded_backend(getattr(route, "backend", _AUTO_BACKEND))
        return plan

    def log(message: str, **fields: str) -> None:
        logger.bind(**fields).warning(message)

    browser = GuardedArticleBrowser(
        egress_guard=DefaultProbeEgressGuard(),
        context=RuntimeRequestContext(source="article_extract", stage="fetch"),
    )
    return ArticleDependencies(
        load_config=load_and_log_configs,
        resolve_plan=resolve_plan,
        evaluate_target=preflight_facade.evaluate_target,
        run_preflight=preflight_facade.run_preflight,
        apply_preflight_advice=preflight_facade.apply_preflight_advice,
        fetch_client=DefaultFetchClient(),
        browser=browser,
        executor=DEFAULT_EXTRACTION_EXECUTOR,
        extract=extract_article_with_pipeline,
        build_preflight_context=preflight_facade.build_execution_context,
        preflight_options=preflight_facade.PreflightOptions.from_mapping,
        public_preflight_payload=preflight_facade.public_preflight_payload,
        resolve_handler=resolve_handler,
        js_required=_js_required,
        convert_content=convert_html_to_markdown,
        increment_counter=increment_counter,
        observe_histogram=observe_histogram,
        clock=time.monotonic,
        log=log,
        policy_checker=DefaultWebOutboundPolicyChecker(),
        backend_setting=lambda plan: resolved_backend_settings.get(id(plan), plan.backend),
    )


async def scrape_article(
    url: str,
    custom_cookies: list[dict[str, Any]] | None = None,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    """Scrape one article through canonical governed orchestration."""
    dependencies = _build_default_dependencies(custom_cookies)
    return await _run_article(
        url,
        custom_cookies,
        allow_llm_extraction,
        dependencies=dependencies,
    )


__all__ = ["ArticleDependencies", "scrape_article"]
