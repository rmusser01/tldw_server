"""Canonical governed orchestration for standard asynchronous article scraping."""

import asyncio
import time
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Optional, Protocol, TypeVar
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from loguru import logger

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.http_client import DEFAULT_MAX_REDIRECTS
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
from tldw_Server_API.app.core.Security.egress import public_url_policy_scope
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.content import convert_html_to_markdown
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_with_pipeline
from tldw_Server_API.app.core.Web_Scraping.handlers import resolve_handler
from tldw_Server_API.app.core.Web_Scraping.observability import sanitized_host
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import GuardedArticleBrowser
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    PUBLIC_FAILURE_CODES,
    ArticleFailure,
    ArticleLimits,
    ArticlePlan,
    DirectBrowserProfile,
    article_failure_result,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import (
    DEFAULT_EXTRACTION_EXECUTOR,
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
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.app.core.Web_Scraping.scraper_router import DEFAULT_HANDLER, ScraperRouter

from ..contracts.execution import DiscoveryResult, PageData
from ..runtime.controls import PageControls, ParentControls, PhysicalWork, WorkHandle, owned_to_thread, physical_work

_FETCH_TIMEOUT_SECONDS = 15.0
_BLOCKING_FETCH_TIMEOUT_SECONDS = 30.0
_AUTO_BACKEND = "auto"
_PLAYWRIGHT = "playwright"
_HTTP_BACKENDS = frozenset({_AUTO_BACKEND, "curl", "httpx"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_REDIRECT_SAFE_HEADERS = frozenset({"accept-encoding", "user-agent"})
_SAFE_POLICY_FAILURE = "policy_error"
_RESPONSE_TOO_LARGE_MESSAGE = "Response exceeds max_response_bytes limit"
ACTIVE_EVENT_LOOP_ERROR = "Synchronous article scraping cannot run while an event loop is active in this thread"
BLOCKING_ARTICLE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_ARTICLE_LOG_STAGES = frozenset(
    {
        "acquire",
        "article",
        "browser",
        "browser_transfer",
        "callback",
        "callback_drain",
        "capability",
        "capacity",
        "cleanup",
        "config",
        "content",
        "context",
        "egress",
        "extract",
        "fetch",
        "http_route",
        "launch",
        "navigation",
        "page",
        "plan",
        "pre_fetch",
        "preflight",
        "preflight_advice",
        "preflight_payload",
        "redirect",
        "rendered_html",
        "result",
        "routing",
        "stealth",
        "wait",
        "websocket_route",
    }
)
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


_T = TypeVar("_T")


class _ArticleBrowser(Protocol):
    async def acquire(self, url: str, profile: DirectBrowserProfile, limits: ArticleLimits) -> str: ...


class _ArticleExecutor(Protocol):
    async def run(self, func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T: ...


@dataclass(frozen=True, slots=True)
class ArticleDependencies:
    """Injectable collaborators for a single standard article request."""

    load_config: Callable[[], Mapping[str, Any] | None]
    resolve_plan: Callable[[str, Mapping[str, Any]], ArticlePlan]
    evaluate_target: Callable[..., Awaitable[Any]]
    run_preflight: Callable[..., Awaitable[Any]]
    apply_preflight_advice: Callable[..., tuple[str, str, Any]]
    fetch_client: FetchClient
    browser: _ArticleBrowser
    executor: _ArticleExecutor
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
    parent: ParentControls | None = None
    request_context: RuntimeRequestContext | None = None
    supplied_plan: ArticlePlan | None = None
    acquired: Callable[[str, str, str], None] | None = None
    discovery_on_success: bool = False
    discovery_gate: Callable[[Mapping[str, Any]], bool] | None = None
    method_intent: str = "auto"
    http_only: bool = False
    http_failure: Callable[[str, Exception], dict[str, Any]] | None = None
    browser_http_fallback: Callable[[], bool] | None = None


@dataclass(frozen=True, slots=True)
class _PreparedArticle:
    """One policy-admitted immutable article request ready for acquisition."""

    plan: ArticlePlan
    advised_backend: str
    advised_method: str
    cookies: Mapping[str, str]
    policy_config: Mapping[str, Any]
    preflight_payload: dict[str, Any] | None
    http_only: bool = False


class _RedirectPolicyBlocked(Exception):
    """Carry a denied redirect decision across the fetch boundary."""

    def __init__(self, decision: Any) -> None:
        super().__init__("Redirect target denied by outbound policy")
        self.decision = decision


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


def _snapshot_cookies(
    value: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    """Capture caller cookies before the request can suspend."""
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(_snapshot_mapping(cookie) for cookie in value if isinstance(cookie, Mapping))


def _web_scraper_values(config: Mapping[str, Any]) -> Mapping[str, Any]:
    values = config.get("web_scraper")
    if isinstance(values, Mapping):
        return values
    return MappingProxyType({})


def _fallback_plan(
    url: str, config: Mapping[str, Any], custom_cookies: Sequence[Mapping[str, Any]] | None
) -> ArticlePlan:
    """Create the same safe generic route used when router construction fails."""
    try:
        domain = sanitized_host(url)
    except (AttributeError, TypeError, UnicodeError, ValueError):
        domain = ""
    if domain == "unknown":
        domain = ""
    routing = SimpleNamespace(
        url=url,
        domain=domain,
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


def _failure_result(url: str, failure: ArticleFailure | str) -> dict[str, Any]:
    captured = failure if isinstance(failure, ArticleFailure) else ArticleFailure(failure, "article")
    return {"url": url, **article_failure_result(captured)}


def _http_failure_result(dependencies: ArticleDependencies, url: str, error: Exception) -> dict[str, Any]:
    """Keep HTTP-only compatibility failures within their original result contract."""
    if dependencies.http_failure is not None:
        return dependencies.http_failure(url, error)
    return _failure_result(url, "fetch_error")


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


def _fallback_warning(message: str, **fields: str) -> None:
    """Emit a sanitized last-resort warning without affecting scrape outcomes."""
    try:
        logger.warning(message, **fields)
    except Exception:  # noqa: BLE001 - no observability backend may break scraping
        return


def _record_counter(dependencies: ArticleDependencies, name: str, labels: Mapping[str, str]) -> None:
    try:
        dependencies.increment_counter(name, labels=dict(labels))
    except Exception as exc:  # noqa: BLE001 - metrics must not change article outcomes
        _fallback_warning(
            "Article metric recording failed.",
            metric_type="counter",
            exception_type=type(exc).__name__[:80],
        )


def _record_histogram(
    dependencies: ArticleDependencies,
    name: str,
    value: float,
    labels: Mapping[str, str],
) -> None:
    try:
        dependencies.observe_histogram(name, value, labels=dict(labels))
    except Exception as exc:  # noqa: BLE001 - metrics must not change article outcomes
        _fallback_warning(
            "Article metric recording failed.",
            metric_type="histogram",
            exception_type=type(exc).__name__[:80],
        )


def _log_failure(
    dependencies: ArticleDependencies,
    exc: BaseException,
    *,
    code: str,
    stage: str,
    url: str,
) -> None:
    safe_code = code if type(code) is str and code in PUBLIC_FAILURE_CODES else "extraction_error"
    safe_stage = stage if type(stage) is str and stage in _ARTICLE_LOG_STAGES else "article"
    try:
        dependencies.log(
            "Article orchestration failure.",
            exception_type=type(exc).__name__[:80],
            code=safe_code,
            stage=safe_stage,
            host=sanitized_host(url),
        )
    except Exception as fallback_exc:  # noqa: BLE001 - logging must not change article outcomes
        _fallback_warning(
            "Article failure logging failed.",
            exception_type=type(fallback_exc).__name__[:80],
            code=safe_code,
            stage=safe_stage,
            host=sanitized_host(url),
        )


async def _fetch_response(dependencies: ArticleDependencies, request: FetchRequest) -> FetchResponse:
    if dependencies.parent is not None:
        dependencies.parent.check()
        request = replace(
            request,
            timeout=dependencies.parent.cap_timeout(request.timeout),
            context=replace(dependencies.request_context, stage="fetch"),
        )
    return await owned_to_thread(dependencies.fetch_client.fetch, request)


def _response_too_large_error(exc: ValueError) -> ArticleFailure | None:
    if type(exc) is ValueError and str(exc) == _RESPONSE_TOO_LARGE_MESSAGE:
        return ArticleFailure("response_too_large", "fetch")
    return None


def _redirect_location(response: FetchResponse) -> str | None:
    for key, value in response.headers.items():
        if str(key).lower() == "location":
            location = str(value).strip()
            return location or None
    return None


def _redirect_target(current_url: str, response: FetchResponse, location: str) -> str:
    try:
        target = urljoin(response.url or current_url, location)
        parsed = urlsplit(target)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        _ = parsed.port
    except (TypeError, UnicodeError, ValueError):
        raise ArticleFailure("fetch_error", "redirect") from None
    return target


def _redirect_crosses_credential_boundary(current_url: str, target_url: str) -> bool:
    try:
        current = urlsplit(current_url)
        target = urlsplit(target_url)
        current_port = current.port or (443 if current.scheme.lower() == "https" else 80)
        target_port = target.port or (443 if target.scheme.lower() == "https" else 80)
        cross_host = (current.hostname, current_port) != (target.hostname, target_port)
        downgrade = current.scheme.lower() == "https" and target.scheme.lower() == "http"
        return cross_host or downgrade
    except (TypeError, UnicodeError, ValueError):
        return True


def _redirect_boundary_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(value) for key, value in headers.items() if str(key).lower() in _REDIRECT_SAFE_HEADERS}


async def _fetch_lightweight(
    dependencies: ArticleDependencies,
    plan: ArticlePlan,
    *,
    url: str,
    cookies: Mapping[str, str],
    policy_config: Mapping[str, Any],
    backend: str,
    timeout: float = _FETCH_TIMEOUT_SECONDS,
    source: str = "article_extract",
) -> tuple[FetchResponse, str]:
    def request_for(
        selected_backend: str,
        request_url: str,
        request_headers: Mapping[str, str],
        request_cookies: Mapping[str, str],
    ) -> FetchRequest:
        return FetchRequest(
            url=request_url,
            method="GET",
            headers=request_headers,
            cookies=request_cookies,
            timeout=timeout,
            backend=selected_backend,
            allow_redirects=False,
            impersonate=plan.impersonate,
            proxies=plan.proxies,
            context=RuntimeRequestContext(source=source, stage="fetch"),
            max_response_bytes=plan.limits.max_article_bytes,
        )

    async def fetch_backend(selected_backend: str) -> tuple[FetchResponse, str]:
        current_url = url
        current_headers = dict(plan.headers)
        current_cookies = dict(cookies)
        seen_urls = {current_url}

        for redirect_count in range(DEFAULT_MAX_REDIRECTS + 1):
            try:
                response = await _fetch_response(
                    dependencies,
                    request_for(
                        selected_backend,
                        current_url,
                        current_headers,
                        current_cookies,
                    ),
                )
            except ValueError as exc:
                overflow = _response_too_large_error(exc)
                if overflow is not None:
                    raise overflow from None
                raise

            location = _redirect_location(response)
            if response.status not in _REDIRECT_STATUSES or location is None:
                return response, _bounded_backend(response.backend)
            if redirect_count == DEFAULT_MAX_REDIRECTS:
                raise ArticleFailure("fetch_error", "redirect")

            target_url = _redirect_target(current_url, response, location)
            if target_url in seen_urls:
                raise ArticleFailure("fetch_error", "redirect")
            seen_urls.add(target_url)

            try:
                target = await dependencies.evaluate_target(
                    target_url,
                    respect_robots=plan.respect_robots,
                    user_agent=current_headers.get("User-Agent") or plan.browser.user_agent,
                    request_context=(
                        replace(dependencies.request_context, stage="redirect")
                        if dependencies.request_context
                        else RuntimeRequestContext(source=source, stage="redirect")
                    ),
                    config=policy_config,
                    policy_checker=dependencies.policy_checker,
                )
            except asyncio.CancelledError:
                raise
            except ParentStop:
                raise
            except Exception as exc:  # noqa: BLE001 - redirect policy failures are fail-closed
                raise ArticleFailure(_SAFE_POLICY_FAILURE, "redirect") from exc

            decision = getattr(target, "decision", None)
            if not bool(getattr(decision, "allowed", False)):
                raise _RedirectPolicyBlocked(decision)

            if _redirect_crosses_credential_boundary(current_url, target_url):
                current_headers = _redirect_boundary_headers(current_headers)
                current_cookies = {}
            current_url = target_url

        raise ArticleFailure("fetch_error", "redirect")

    if backend == "curl":
        try:
            return await fetch_backend("curl")
        except asyncio.CancelledError:
            raise
        except ParentStop:
            raise
        except _RedirectPolicyBlocked:
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
    use_route_handler: bool = True,
    final_url: str | None = None,
) -> dict[str, Any]:
    if dependencies.parent is not None:
        dependencies.parent.check()
    deferred_discovery = dependencies.discovery_on_success or dependencies.discovery_gate is not None
    if dependencies.acquired is not None and not deferred_discovery:
        dependencies.acquired(html, url, final_url or url)
    handler = dependencies.resolve_handler(plan.handler) if use_route_handler and plan.handler else None
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
    if (
        deferred_discovery
        and dependencies.acquired is not None
        and (not dependencies.discovery_on_success or copied.get("extraction_successful"))
        and (dependencies.discovery_gate is None or dependencies.discovery_gate(copied))
    ):
        if dependencies.parent is not None:
            dependencies.parent.check()
        dependencies.acquired(html, url, final_url or url)
    if copied.get("extraction_successful") and handler is None and copied.get("content"):
        copied["content"] = dependencies.convert_content(str(copied["content"]))
    return copied


async def _run_article(
    url: str,
    custom_cookies: Sequence[Mapping[str, Any]] | None,
    allow_llm_extraction: bool,
    *,
    dependencies: ArticleDependencies,
    credential_free: bool = False,
) -> dict[str, Any]:
    """Run one request through its immutable route and governed dependencies."""
    prepared = await _prepare_article(
        url,
        custom_cookies,
        dependencies=dependencies,
        source="article_extract",
        plan_modifier=ArticlePlan.for_public_capture if credential_free else None,
    )
    if isinstance(prepared, dict):
        return prepared

    plan = prepared.plan
    advised_backend = prepared.advised_backend
    advised_method = prepared.advised_method
    cookies = prepared.cookies
    policy_config = prepared.policy_config
    preflight_payload = prepared.preflight_payload
    http_only = dependencies.http_only or prepared.http_only

    if advised_backend != _PLAYWRIGHT and advised_method != _PLAYWRIGHT:
        started = dependencies.clock()
        requested_backend = _bounded_backend(advised_backend)
        fetch_metric_backend = "curl" if requested_backend == "curl" else "httpx"
        try:
            response, backend_used = await _fetch_lightweight(
                dependencies,
                plan,
                url=url,
                cookies=cookies,
                policy_config=policy_config,
                backend=requested_backend,
            )
        except asyncio.CancelledError:
            raise
        except ParentStop:
            raise
        except _RedirectPolicyBlocked as exc:
            decision = exc.decision
            if str(getattr(decision, "reason", "")).startswith("robots_"):
                _record_counter(dependencies, "scrape_blocked_by_robots_total", {})
            return _attach_preflight(_policy_blocked_result(url, decision), preflight_payload)
        except ArticleFailure as exc:
            if exc.code in {"policy_error", "response_too_large"} or exc.stage == "redirect":
                _log_failure(dependencies, exc, code=exc.code, stage=exc.stage, url=url)
                return _attach_preflight(_failure_result(url, exc.code), preflight_payload)
            _log_failure(dependencies, exc, code="fetch_error", stage="fetch", url=url)
            _record_counter(
                dependencies,
                "scrape_fetch_total",
                {"backend": fetch_metric_backend, "outcome": "error"},
            )
            if http_only:
                return _attach_preflight(_http_failure_result(dependencies, url, exc), preflight_payload)
            _record_counter(dependencies, "scrape_playwright_fallback_total", {"reason": "error"})
        except Exception as exc:  # noqa: BLE001 - browser fallback remains eligible
            _log_failure(dependencies, exc, code="fetch_error", stage="fetch", url=url)
            _record_counter(
                dependencies,
                "scrape_fetch_total",
                {"backend": fetch_metric_backend, "outcome": "error"},
            )
            if http_only:
                return _attach_preflight(_http_failure_result(dependencies, url, exc), preflight_payload)
            _record_counter(dependencies, "scrape_playwright_fallback_total", {"reason": "error"})
        else:
            _record_histogram(
                dependencies,
                "scrape_fetch_latency_seconds",
                max(0.0, dependencies.clock() - started),
                {"backend": backend_used},
            )
            if http_only and not 200 <= response.status < 300:
                error = ValueError(f"Article retrieval failed: HTTP {response.status} (expected terminal 2xx response)")
                _record_counter(dependencies, "scrape_fetch_total", {"backend": backend_used, "outcome": "error"})
                return _attach_preflight(_http_failure_result(dependencies, url, error), preflight_payload)
            if response.status < 400 and (response.text or http_only):
                if not http_only and dependencies.js_required(response.text, response.headers, url):
                    _record_counter(
                        dependencies,
                        "scrape_playwright_fallback_total",
                        {"reason": "js_required"},
                    )
                else:
                    try:
                        final_extraction = await _extract(
                            dependencies,
                            plan,
                            response.text,
                            url,
                            allow_llm_extraction=allow_llm_extraction,
                            final_url=response.url or url,
                        )
                    except asyncio.CancelledError:
                        raise
                    except ParentStop:
                        raise
                    except Exception as exc:  # noqa: BLE001 - browser fallback remains eligible
                        _log_failure(dependencies, exc, code="extraction_error", stage="extract", url=url)
                        _record_counter(
                            dependencies,
                            "scrape_fetch_total",
                            {"backend": backend_used, "outcome": "error"},
                        )
                        if http_only:
                            return _attach_preflight(_http_failure_result(dependencies, url, exc), preflight_payload)
                        _record_counter(
                            dependencies,
                            "scrape_playwright_fallback_total",
                            {"reason": "error"},
                        )
                    else:
                        if final_extraction.get("extraction_successful"):
                            content = str(final_extraction.get("content", "") or "")
                            _record_histogram(
                                dependencies,
                                "scrape_content_length_bytes",
                                len(content.encode("utf-8", errors="ignore")),
                                {"backend": backend_used},
                            )
                            _record_counter(
                                dependencies,
                                "scrape_fetch_total",
                                {"backend": backend_used, "outcome": "success"},
                            )
                            return _attach_preflight(final_extraction, preflight_payload)
                        _record_counter(
                            dependencies,
                            "scrape_fetch_total",
                            {"backend": backend_used, "outcome": "no_extract"},
                        )
                        if http_only:
                            return _attach_preflight(final_extraction, preflight_payload)
                        _record_counter(
                            dependencies,
                            "scrape_playwright_fallback_total",
                            {"reason": "no_extract"},
                        )
            else:
                _record_counter(
                    dependencies,
                    "scrape_fetch_total",
                    {"backend": backend_used, "outcome": "no_extract"},
                )
                _record_counter(
                    dependencies,
                    "scrape_playwright_fallback_total",
                    {"reason": "no_extract"},
                )

    browser_started = dependencies.clock()
    try:
        html = await dependencies.browser.acquire(url, plan.direct_browser, plan.limits)
    except asyncio.CancelledError:
        raise
    except ParentStop:
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
        return _attach_preflight(_failure_result(url, exc), preflight_payload)
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
            final_url=getattr(dependencies.browser, "final_url", None) or url,
        )
    except asyncio.CancelledError:
        raise
    except ParentStop:
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
            len(content.encode("utf-8", errors="ignore")),
            {"backend": _PLAYWRIGHT},
        )
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "success"})
    else:
        _record_counter(dependencies, "scrape_fetch_total", {"backend": _PLAYWRIGHT, "outcome": "no_extract"})
    return _attach_preflight(result, preflight_payload)


async def _prepare_article(
    url: str,
    custom_cookies: Sequence[Mapping[str, Any]] | None,
    *,
    dependencies: ArticleDependencies,
    source: str,
    plan_modifier: Callable[[ArticlePlan], ArticlePlan] | None = None,
) -> _PreparedArticle | dict[str, Any]:
    """Load, snapshot, admit, and preflight one request before acquisition."""
    config: Mapping[str, Any]
    try:
        config = _snapshot_config(await owned_to_thread(dependencies.load_config))
    except asyncio.CancelledError:
        raise
    except ParentStop:
        raise
    except Exception as exc:  # noqa: BLE001 - config loading has one safe empty fallback
        _log_failure(dependencies, exc, code="fetch_error", stage="config", url=url)
        config = MappingProxyType({})

    try:
        plan = dependencies.supplied_plan or await owned_to_thread(dependencies.resolve_plan, url, config)
        if not isinstance(plan, ArticlePlan):
            raise TypeError("resolve_plan must return ArticlePlan")
    except asyncio.CancelledError:
        raise
    except ParentStop:
        raise
    except Exception as exc:  # noqa: BLE001 - route loading preserves the config snapshot
        _log_failure(dependencies, exc, code="fetch_error", stage="plan", url=url)
        plan = _fallback_plan(url, config, custom_cookies)

    if plan_modifier is not None:
        plan = plan_modifier(plan)

    values = _web_scraper_values(config)
    policy_config = MappingProxyType({"web_scraper": values})
    effective_user_agent = plan.headers.get("User-Agent") or plan.browser.user_agent
    request_context = (
        replace(dependencies.request_context, stage="pre_fetch")
        if dependencies.request_context
        else RuntimeRequestContext(source=source, stage="pre_fetch")
    )
    preflight_payload: dict[str, Any] | None = None
    try:
        target = await dependencies.evaluate_target(
            url,
            respect_robots=plan.respect_robots,
            user_agent=effective_user_agent,
            request_context=request_context,
            config=policy_config,
            policy_checker=dependencies.policy_checker,
        )
    except asyncio.CancelledError:
        raise
    except ParentStop:
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
                **({"parent": dependencies.parent} if dependencies.parent is not None else {}),
            )
            preflight_result = await dependencies.run_preflight(target, options, context)
        except asyncio.CancelledError:
            raise
        except ParentStop:
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
            method=dependencies.method_intent,
            backend_setting=backend_setting,
        )
    except asyncio.CancelledError:
        raise
    except ParentStop:
        raise
    except Exception as exc:  # noqa: BLE001 - failed advice retains the snapshotted route
        _log_failure(dependencies, exc, code="fetch_error", stage="preflight_advice", url=url)
        advised_backend, advised_method, advised_result = backend, dependencies.method_intent, None

    if not automatic_backend:
        advised_backend = backend
    try:
        preflight_payload = dependencies.public_preflight_payload(
            advised_result,
            bool(getattr(options, "include_results", False)),
        )
    except asyncio.CancelledError:
        raise
    except ParentStop:
        raise
    except Exception as exc:  # noqa: BLE001 - payload creation is optional observability
        _log_failure(dependencies, exc, code="fetch_error", stage="preflight_payload", url=url)

    fallback = False
    if (advised_backend == _PLAYWRIGHT or advised_method == _PLAYWRIGHT) and dependencies.browser_http_fallback:
        if dependencies.parent is not None:
            dependencies.parent.check()
        try:
            fallback = dependencies.browser_http_fallback()
        except ArticleFailure as failure:
            return _attach_preflight(_failure_result(url, failure), preflight_payload)
        if fallback:
            advised_backend, advised_method = "httpx", "trafilatura"
            plan = replace(plan, proxies={}, impersonate=None)

    cookies: dict[str, str] = {}
    for cookie in custom_cookies or ():
        if isinstance(cookie, Mapping) and "name" in cookie and "value" in cookie:
            cookies[str(cookie["name"])] = str(cookie["value"])
    cookies.update({str(key): str(value) for key, value in plan.cookies.items()})
    return _PreparedArticle(
        plan=plan,
        advised_backend=advised_backend,
        advised_method=advised_method,
        cookies=MappingProxyType(cookies),
        policy_config=policy_config,
        preflight_payload=preflight_payload,
        http_only=fallback,
    )


def _blocking_plan(plan: ArticlePlan) -> ArticlePlan:
    """Apply the legacy blocking transport profile to an immutable plan."""
    headers = dict(plan.headers)
    headers["User-Agent"] = BLOCKING_ARTICLE_USER_AGENT
    return replace(
        plan,
        headers=headers,
        handler=DEFAULT_HANDLER,
        respect_robots=False,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
        browser=replace(
            plan.browser,
            user_agent=BLOCKING_ARTICLE_USER_AGENT,
        ),
    )


def _blocking_failure_result(url: str) -> dict[str, Any]:
    return {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
    }


async def _run_blocking_article(
    url: str,
    custom_cookies: Sequence[Mapping[str, Any]] | None,
    allow_llm_extraction: bool,
    *,
    dependencies: ArticleDependencies,
) -> dict[str, Any]:
    """Run the legacy blocking profile with governed async collaborators."""
    prepared = await _prepare_article(
        url,
        custom_cookies,
        dependencies=dependencies,
        source="article_extract_blocking",
        plan_modifier=_blocking_plan,
    )
    if isinstance(prepared, dict):
        if prepared.get("error") == _SAFE_POLICY_FAILURE:
            result = _blocking_failure_result(url)
            result["error"] = "Outbound policy evaluation failed"
            return result
        return prepared

    use_browser = prepared.advised_backend == _PLAYWRIGHT or prepared.advised_method == _PLAYWRIGHT
    if not use_browser:
        try:
            response, _backend = await _fetch_lightweight(
                dependencies,
                prepared.plan,
                url=url,
                cookies=prepared.cookies,
                policy_config=prepared.policy_config,
                backend=_bounded_backend(prepared.advised_backend),
                timeout=_BLOCKING_FETCH_TIMEOUT_SECONDS,
                source="article_extract_blocking",
            )
        except asyncio.CancelledError:
            raise
        except ParentStop:
            raise
        except _RedirectPolicyBlocked as exc:
            decision = exc.decision
            if str(getattr(decision, "reason", "")).startswith("robots_"):
                _record_counter(dependencies, "scrape_blocked_by_robots_total", {})
            return _attach_preflight(
                _policy_blocked_result(url, decision),
                prepared.preflight_payload,
            )
        except ArticleFailure as exc:
            if exc.code in {"policy_error", "response_too_large"}:
                return _attach_preflight(_failure_result(url, exc.code), prepared.preflight_payload)
            return _attach_preflight(_blocking_failure_result(url), prepared.preflight_payload)
        except Exception:  # noqa: BLE001 - historical blocking fetch failures stay compact
            return _attach_preflight(_blocking_failure_result(url), prepared.preflight_payload)

        if response.status != 200:
            return _attach_preflight(_blocking_failure_result(url), prepared.preflight_payload)
        html = response.text
    else:
        try:
            html = await dependencies.browser.acquire(
                url,
                prepared.plan.direct_browser,
                prepared.plan.limits,
            )
        except asyncio.CancelledError:
            raise
        except ParentStop:
            raise
        except ArticleFailure as exc:
            return _attach_preflight(_failure_result(url, exc), prepared.preflight_payload)
        except Exception:  # noqa: BLE001 - guarded browser failures are stable
            return _attach_preflight(_failure_result(url, "browser_error"), prepared.preflight_payload)

    try:
        return _attach_preflight(
            await _extract(
                dependencies,
                prepared.plan,
                html,
                url,
                allow_llm_extraction=allow_llm_extraction,
                use_route_handler=False,
            ),
            prepared.preflight_payload,
        )
    except asyncio.CancelledError:
        raise
    except ParentStop:
        raise
    except Exception:  # noqa: BLE001 - preserve the compact historical blocking failure shape
        return _attach_preflight(_blocking_failure_result(url), prepared.preflight_payload)


def _raw_failure_result(url: str, failure: ArticleFailure | str) -> dict[str, Any]:
    code = failure.code if isinstance(failure, ArticleFailure) else failure
    safe_code = code if type(code) is str and code in PUBLIC_FAILURE_CODES else "browser_error"
    result: dict[str, Any] = {"url": url, "extraction_successful": False, "error": safe_code}
    if safe_code == "browser_transport_unavailable":
        captured = failure if isinstance(failure, ArticleFailure) else ArticleFailure(failure, "article")
        result["capability"] = article_failure_result(captured)["capability"]
    return result


async def _run_raw_browser_article(
    url: str,
    *,
    dependencies: ArticleDependencies,
) -> dict[str, Any]:
    """Acquire bounded rendered HTML without extraction for the raw sync helper."""
    prepared = await _prepare_article(
        url,
        (),
        dependencies=dependencies,
        source="article_extract_sync",
    )
    if isinstance(prepared, dict):
        return _raw_failure_result(url, "policy_error")

    try:
        html = await dependencies.browser.acquire(
            url,
            prepared.plan.direct_browser,
            prepared.plan.limits,
        )
    except asyncio.CancelledError:
        raise
    except ParentStop:
        raise
    except ArticleFailure as exc:
        return _raw_failure_result(url, exc)
    except Exception:  # noqa: BLE001 - direct browser details must not escape this API
        return _raw_failure_result(url, "browser_error")

    if not isinstance(html, str):
        return _raw_failure_result(url, "browser_error")
    try:
        title_tag = BeautifulSoup(html, "html.parser").find("title")
        title = title_tag.get_text(strip=True) if title_tag is not None else ""
    except Exception:  # noqa: BLE001 - malformed markup still has stable public output
        return _raw_failure_result(url, "browser_error")
    return {
        "url": url,
        "title": title,
        "content": html,
        "extraction_successful": True,
    }


def _default_rules_path() -> str:
    return str(Path(__file__).resolve().parents[4] / "Config_Files" / "custom_scrapers.yaml")


def _build_default_dependencies(
    custom_cookies: Sequence[Mapping[str, Any]] | None, *, credential_free: bool = False
) -> ArticleDependencies:
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
        policy_checker=DefaultWebOutboundPolicyChecker(credential_free=credential_free),
        backend_setting=lambda plan: resolved_backend_settings.get(id(plan), plan.backend),
    )


def start_page(
    url: str,
    custom_cookies: list[dict[str, Any]] | None,
    *,
    controls: ParentControls,
    request_context: RuntimeRequestContext,
    discover: Callable[[str, str, str], DiscoveryResult],
    plan: ArticlePlan | None = None,
    allow_llm_extraction: bool = True,
    discovery_on_success: bool = False,
    configuration: Mapping[str, Any] | None = None,
    discovery_gate: Callable[[Mapping[str, Any]], bool] | None = None,
    prepare_plan: Callable[[ArticlePlan], ArticlePlan] | None = None,
    transform: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    method_intent: str = "auto",
    dependencies: ArticleDependencies | None = None,
    browser_http_fallback: Callable[[], bool] | None = None,
) -> WorkHandle[PageData]:
    """Start one canonical page attempt with private discovery and physical drain."""
    loop = asyncio.get_running_loop()
    owner = PhysicalWork()
    if transform is not None:
        owner.drained.add_done_callback(lambda _done, callback=transform: None)
    parent = PageControls(controls)
    cookies = _snapshot_cookies(custom_cookies)
    configuration = _snapshot_config(configuration) if configuration is not None else None
    supplied_dependencies = dependencies

    async def run() -> PageData:
        token = physical_work.set(owner)
        try:
            reservation = parent.reserve("page_attempts")
            try:
                reservation.commit()
            finally:
                reservation.close()
            dependencies = supplied_dependencies or _build_default_dependencies(cookies)
            if browser_http_fallback is not None:
                dependencies = replace(dependencies, browser_http_fallback=browser_http_fallback)
            if configuration is not None:
                dependencies = replace(dependencies, load_config=lambda: configuration)
            if prepare_plan is not None:
                original_resolve = dependencies.resolve_plan
                original_backend = dependencies.backend_setting
                resolved_settings: dict[int, str] = {}

                def resolve_page(target: str, config: Mapping[str, Any]) -> ArticlePlan:
                    resolved = original_resolve(target, config)
                    prepared = prepare_plan(resolved)
                    # Header-only preparation must not erase raw-route advice
                    # precedence. An explicit backend change remains explicit.
                    resolved_settings[id(prepared)] = (
                        prepared.backend
                        if prepared.backend != resolved.backend
                        else original_backend(resolved)
                        if original_backend
                        else resolved.backend
                    )
                    return prepared

                dependencies = replace(
                    dependencies,
                    resolve_plan=resolve_page,
                    backend_setting=lambda value: resolved_settings.get(id(value), value.backend),
                )
            # Bind live controls only to this private dependency graph, never request metadata.
            if isinstance(dependencies.fetch_client, DefaultFetchClient):
                dependencies = replace(dependencies, fetch_client=DefaultFetchClient(parent))
            if isinstance(dependencies.browser, GuardedArticleBrowser):
                dependencies.browser.bind_parent(parent, request_context)
            if isinstance(dependencies.policy_checker, DefaultWebOutboundPolicyChecker):
                dependencies = replace(dependencies, policy_checker=DefaultWebOutboundPolicyChecker(parent=parent))
            sidecar = DiscoveryResult(url, url, ())
            document = None

            def acquired(document: str, request_url: str, final_url: str) -> None:
                nonlocal sidecar
                parent.check()
                sidecar = discover(document, request_url, final_url)
                parent.check()

            def deferred_acquired(html: str, request_url: str, final_url: str) -> None:
                nonlocal document
                document = (html, request_url, final_url)

            dependencies = replace(
                dependencies,
                parent=parent,
                request_context=request_context,
                supplied_plan=plan,
                acquired=acquired if transform is None else deferred_acquired,
                discovery_on_success=discovery_on_success if transform is None else False,
                discovery_gate=discovery_gate if transform is None else None,
                method_intent=method_intent,
            )
            timeout = preflight_facade._execution_timeout(parent.cap_timeout(None))
            try:
                async with timeout:
                    result = await _run_article(url, cookies, allow_llm_extraction, dependencies=dependencies)
                    parent.check()
                    if transform is not None:
                        original = result
                        result = transform(url, result)
                        if "preflight_analysis" in original:
                            result.setdefault("preflight_analysis", original["preflight_analysis"])
                        parent.check()
                        if (
                            document is not None
                            and (not discovery_on_success or result.get("extraction_successful"))
                            and (discovery_gate is None or discovery_gate(result))
                        ):
                            acquired(*document)
                    return PageData(result, sidecar)
            except TimeoutError:
                if timeout.expired():
                    parent.check()
                    raise ParentStop("deadline_exhausted") from None
                raise
        finally:
            parent.cancel()
            physical_work.reset(token)

    task = loop.create_task(run(), name="controlled-article-page")
    task.add_done_callback(lambda _done: owner.finish())

    def cancel() -> None:
        parent.cancel()
        loop.call_soon_threadsafe(task.cancel)

    return WorkHandle(task, owner.drained, cancel)


async def scrape_article(
    url: str,
    custom_cookies: list[dict[str, Any]] | None = None,
    *,
    allow_llm_extraction: bool = True,
    credential_free: bool = False,
) -> dict[str, typing.Any]:
    """Scrape one article through canonical governed orchestration."""
    cookie_snapshot = () if credential_free else _snapshot_cookies(custom_cookies)
    with public_url_policy_scope(enabled=credential_free):
        dependencies = (
            _build_default_dependencies(cookie_snapshot, credential_free=True)
            if credential_free
            else _build_default_dependencies(cookie_snapshot)
        )
        return await _run_article(
            url,
            cookie_snapshot,
            allow_llm_extraction and not credential_free,
            dependencies=dependencies,
            credential_free=credential_free,
        )


def _reject_active_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(ACTIVE_EVENT_LOOP_ERROR)


def scrape_article_blocking(
    url: str,
    custom_cookies: Optional[list[dict[str, Any]]] = None,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    """Synchronously scrape an article through the governed blocking profile."""
    _reject_active_event_loop()
    cookie_snapshot = _snapshot_cookies(custom_cookies)
    dependencies = _build_default_dependencies(cookie_snapshot)
    return asyncio.run(
        _run_blocking_article(
            url,
            cookie_snapshot,
            allow_llm_extraction,
            dependencies=dependencies,
        )
    )


def scrape_article_sync(url: str) -> dict[str, Any]:
    """Synchronously return governed rendered HTML without article extraction."""
    _reject_active_event_loop()
    dependencies = _build_default_dependencies(())
    return asyncio.run(_run_raw_browser_article(url, dependencies=dependencies))


__all__ = [
    "ACTIVE_EVENT_LOOP_ERROR",
    "ArticleDependencies",
    "scrape_article",
    "scrape_article_blocking",
    "scrape_article_sync",
]
