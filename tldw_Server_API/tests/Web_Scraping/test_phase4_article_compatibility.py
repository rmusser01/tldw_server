import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
    ArticleLimits,
    ArticlePlan,
    DirectBrowserProfile,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightOptions, PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)

URL = "https://example.com/article"
ACTIVE_EVENT_LOOP_ERROR = "Synchronous article scraping cannot run while an event loop is active in this thread"


class FakeFetchClient:
    def __init__(self, outcomes: Sequence[FetchResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeBrowser:
    def __init__(self, outcomes: Sequence[str | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, DirectBrowserProfile, ArticleLimits]] = []

    async def acquire(
        self,
        url: str,
        profile: DirectBrowserProfile,
        limits: ArticleLimits,
    ) -> str:
        self.calls.append((url, profile, limits))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)


def _canonical() -> Any:
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article

    return article


def _blocking() -> Any:
    canonical = _canonical()
    assert hasattr(canonical, "scrape_article_blocking")
    return canonical.scrape_article_blocking


def _raw_sync() -> Any:
    canonical = _canonical()
    assert hasattr(canonical, "scrape_article_sync")
    return canonical.scrape_article_sync


def _allowed_target(url: str = URL) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=PolicyDecision(
            allowed=True,
            mode="test",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )


def _plan(
    cookies: Sequence[Mapping[str, Any]] = (),
    *,
    plan_cookies: Mapping[str, str] | None = None,
    backend: str = "httpx",
    limits: ArticleLimits | None = None,
) -> ArticlePlan:
    return ArticlePlan(
        url=URL,
        domain="example.com",
        backend=backend,
        headers={"User-Agent": "route-agent"},
        browser=DirectBrowserProfile(
            user_agent="route-agent",
            custom_cookies=tuple(cookies),
            retries=1,
            timeout_ms=1_000,
            stealth_enabled=False,
            stealth_wait_ms=0,
        ),
        cookies=plan_cookies or {},
        limits=limits or ArticleLimits(123, 456),
    )


def _article(_html: str, url: str, **_kwargs: Any) -> dict[str, Any]:
    return {
        "url": url,
        "title": "Article",
        "author": "Author",
        "date": "2026-08-13",
        "content": "article body",
        "extraction_successful": True,
    }


def _dependencies(
    *,
    config: Mapping[str, Any] | None = None,
    fetch: FakeFetchClient | None = None,
    browser: FakeBrowser | None = None,
    target: PreflightTarget | BaseException | None = None,
    preflight: Any = None,
    extract: Any = _article,
) -> tuple[Any, dict[str, Any]]:
    canonical = _canonical()
    observations: dict[str, Any] = {"target_calls": [], "preflight_calls": [], "advice_calls": []}
    source_config = config or {"web_scraper": {"web_scraper_preflight_analyzers": False}}
    fake_fetch = fetch or FakeFetchClient([FetchResponse(URL, 200, {}, "<html><body>article</body></html>", "httpx")])
    fake_browser = browser or FakeBrowser(["<html><title>Browser</title><body>article</body></html>"])
    executor = FakeExecutor()

    async def evaluate_target(*_args: Any, **kwargs: Any) -> PreflightTarget:
        observations["target_calls"].append(kwargs)
        if isinstance(target, BaseException):
            raise target
        return target or _allowed_target()

    async def run_preflight(*args: Any, **kwargs: Any) -> Any:
        observations["preflight_calls"].append((args, kwargs))
        return preflight

    def apply_advice(result: Any, **kwargs: Any) -> tuple[str, str, Any]:
        observations["advice_calls"].append((result, kwargs))
        return kwargs["backend"], kwargs["method"], result

    dependencies = canonical.ArticleDependencies(
        load_config=lambda: source_config,
        resolve_plan=lambda _url, _config: _plan(),
        evaluate_target=evaluate_target,
        run_preflight=run_preflight,
        apply_preflight_advice=apply_advice,
        fetch_client=fake_fetch,
        browser=fake_browser,
        executor=executor,
        extract=extract,
        build_preflight_context=lambda *_args, **_kwargs: object(),
        preflight_options=lambda values: PreflightOptions.from_mapping(values),
        public_preflight_payload=lambda result, include: {"analysis": result} if include and result else None,
        resolve_handler=lambda _path: None,
        js_required=lambda *_args, **_kwargs: False,
        convert_content=lambda content: f"converted:{content}",
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        clock=lambda: 0.0,
        log=lambda *_args, **_kwargs: None,
    )
    observations.update(fetch=fake_fetch, browser=fake_browser, executor=executor)
    return dependencies, observations


def test_sync_entry_points_are_direct_canonical_exports_with_concrete_signatures() -> None:
    from tldw_Server_API.app.core.Web_Scraping import orchestration

    blocking = _blocking()
    raw_sync = _raw_sync()

    assert legacy.scrape_article_blocking is blocking
    assert legacy.scrape_article_sync is raw_sync
    assert orchestration.scrape_article_blocking is blocking
    assert orchestration.scrape_article_sync is raw_sync

    blocking_signature = inspect.signature(blocking)
    assert blocking_signature.parameters["url"].annotation is str
    assert blocking_signature.parameters["custom_cookies"].annotation == Optional[list[dict[str, Any]]]
    assert blocking_signature.parameters["allow_llm_extraction"].annotation is bool
    assert blocking_signature.return_annotation == dict[str, Any]
    assert blocking_signature.parameters["allow_llm_extraction"].kind is inspect.Parameter.KEYWORD_ONLY

    raw_signature = inspect.signature(raw_sync)
    assert raw_signature.parameters["url"].annotation is str
    assert raw_signature.return_annotation == dict[str, Any]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_name", ["scrape_article_blocking", "scrape_article_sync"])
async def test_sync_entry_points_reject_active_event_loops_before_constructing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    entry_name: str,
) -> None:
    canonical = _canonical()
    calls: list[str] = []

    def fail_dependency_construction(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("dependencies")
        raise AssertionError("active-loop guard must run before dependencies")

    monkeypatch.setattr(canonical, "_build_default_dependencies", fail_dependency_construction)

    entry = getattr(canonical, entry_name)
    with pytest.raises(RuntimeError, match=f"^{ACTIVE_EVENT_LOOP_ERROR}$"):
        entry(URL)

    assert calls == []


def test_blocking_profile_uses_governed_admission_reduced_cookies_timeout_and_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    source_config = {
        "web_scraper": {
            "web_scraper_preflight_analyzers": False,
            "nested": {"value": "original"},
        }
    }
    supplied_cookies = [
        {
            "name": "session",
            "value": "original",
            "domain": "example.com",
            "path": "/",
        }
    ]
    dependencies, observations = _dependencies(config=source_config)

    def resolve_plan(_url: str, config: Mapping[str, Any]) -> ArticlePlan:
        assert config["web_scraper"]["nested"]["value"] == "original"
        return _plan(tuple(supplied_cookies))

    async def evaluate_target(*_args: Any, **kwargs: Any) -> PreflightTarget:
        source_config["web_scraper"]["nested"]["value"] = "changed"
        supplied_cookies[0]["value"] = "changed"
        observations["target_calls"].append(kwargs)
        assert kwargs["respect_robots"] is False
        assert kwargs["config"]["web_scraper"]["nested"]["value"] == "original"
        return _allowed_target()

    dependencies = dataclasses.replace(
        dependencies,
        resolve_plan=resolve_plan,
        evaluate_target=evaluate_target,
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = _blocking()(URL, supplied_cookies)

    assert result == {
        "url": URL,
        "title": "Article",
        "author": "Author",
        "date": "2026-08-13",
        "content": "converted:article body",
        "extraction_successful": True,
    }
    request = observations["fetch"].requests[0]
    assert request.timeout == 30.0
    assert request.cookies == {"session": "original"}
    assert request.max_response_bytes == 123
    assert request.context == RuntimeRequestContext(source="article_extract_blocking", stage="fetch")
    assert request.headers["User-Agent"] == canonical.BLOCKING_ARTICLE_USER_AGENT
    assert observations["browser"].calls == []


def test_blocking_profile_accepts_only_exact_http_success_without_browser_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    fetch = FakeFetchClient([FetchResponse(URL, 204, {}, "ignored", "httpx")])
    browser = FakeBrowser([AssertionError("browser should not run")])
    dependencies, observations = _dependencies(fetch=fetch, browser=browser)
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = _blocking()(URL)

    assert result == {
        "url": URL,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
    }
    assert observations["executor"].calls == []
    assert observations["browser"].calls == []


def test_blocking_profile_keeps_lightweight_overflow_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    fetch = FakeFetchClient([ValueError("Response exceeds max_response_bytes limit")])
    browser = FakeBrowser([AssertionError("overflow must not fall back to the browser")])
    dependencies, observations = _dependencies(fetch=fetch, browser=browser)
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = _blocking()(URL)

    assert result["error"] == "response_too_large"
    assert result["extraction_successful"] is False
    assert observations["browser"].calls == []


def test_blocking_profile_honors_preflight_browser_advice_and_attaches_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    browser = FakeBrowser(["<html><title>Advised browser</title><body>article</body></html>"])
    dependencies, observations = _dependencies(
        config={
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
                "web_scraper_preflight_include_results": True,
            }
        },
        browser=browser,
        preflight={"advice": "use_browser"},
    )
    dependencies = dataclasses.replace(
        dependencies,
        resolve_plan=lambda _url, _config: _plan(backend="auto"),
        apply_preflight_advice=lambda result, **_kwargs: ("playwright", "auto", result),
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = _blocking()(URL)

    assert result["extraction_successful"] is True
    assert result["preflight_analysis"] == {"analysis": {"advice": "use_browser"}}
    assert observations["preflight_calls"]
    assert observations["fetch"].requests == []
    assert observations["browser"].calls == [(URL, _blocking_plan_for_assertion(canonical), ArticleLimits(123, 456))]


def test_blocking_browser_advice_preserves_scoped_cookies_without_lightweight_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    source_cookie = {
        "name": "session",
        "value": "secret",
        "domain": "example.com",
        "path": "/account",
        "httpOnly": True,
        "secure": True,
    }
    expected_cookie = dict(source_cookie)
    browser = FakeBrowser(["<article><p>browser content</p></article>"])
    dependencies, observations = _dependencies(
        config={
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
            }
        },
        browser=browser,
        preflight={"advice": "use_browser"},
    )

    async def evaluate_target(*_args: Any, **_kwargs: Any) -> PreflightTarget:
        source_cookie["domain"] = "mutated.example"
        source_cookie["path"] = "/mutated"
        return _allowed_target()

    def build_dependencies(cookie_snapshot: tuple[Mapping[str, Any], ...]) -> Any:
        plan = _plan(
            cookie_snapshot,
            plan_cookies={"route_cookie": "lightweight-only"},
            backend="auto",
        )
        return dataclasses.replace(
            dependencies,
            evaluate_target=evaluate_target,
            resolve_plan=lambda _url, _config: plan,
            apply_preflight_advice=lambda result, **_kwargs: ("playwright", "auto", result),
        )

    monkeypatch.setattr(canonical, "_build_default_dependencies", build_dependencies)

    result = _blocking()(URL, [source_cookie])

    assert result["extraction_successful"] is True
    profile = observations["browser"].calls[0][1]
    assert [dict(cookie) for cookie in profile.custom_cookies] == [expected_cookie]
    assert observations["fetch"].requests == []


def _blocking_plan_for_assertion(canonical: Any) -> DirectBrowserProfile:
    return DirectBrowserProfile(
        user_agent=canonical.BLOCKING_ARTICLE_USER_AGENT,
        custom_cookies=(),
        retries=1,
        timeout_ms=1_000,
        stealth_enabled=False,
        stealth_wait_ms=0,
    )


def test_blocking_profile_fails_closed_without_exposing_policy_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    dependencies, observations = _dependencies(target=RuntimeError("secret-token"))
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = _blocking()(URL)

    assert result["extraction_successful"] is False
    assert result["error"] == "Outbound policy evaluation failed"
    assert "secret" not in str(result)
    assert observations["fetch"].requests == []
    assert observations["browser"].calls == []


def test_raw_sync_uses_governed_browser_and_preserves_legacy_success_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    browser = FakeBrowser(["<html><head><title>Raw Title</title></head><body>raw</body></html>"])
    dependencies, observations = _dependencies(browser=browser)
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = _raw_sync()(URL)

    assert result == {
        "url": URL,
        "title": "Raw Title",
        "content": "<html><head><title>Raw Title</title></head><body>raw</body></html>",
        "extraction_successful": True,
    }
    assert observations["target_calls"][0]["respect_robots"] is True
    assert observations["browser"].calls == [(URL, _plan().direct_browser, ArticleLimits(123, 456))]
    assert observations["executor"].calls == []


def test_raw_sync_sanitizes_policy_and_browser_failures_without_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical()
    denied = PreflightTarget(
        url=URL,
        decision=PolicyDecision(False, "strict", "deny_test", "pre_fetch", "article_extract"),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )
    dependencies, observations = _dependencies(target=denied)
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    policy_result = _raw_sync()(URL)

    assert policy_result == {"url": URL, "extraction_successful": False, "error": "policy_error"}
    assert observations["browser"].calls == []

    browser = FakeBrowser([ArticleFailure("response_too_large", "rendered_html")])
    dependencies, observations = _dependencies(browser=browser)
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    browser_result = _raw_sync()(URL)

    assert browser_result == {"url": URL, "extraction_successful": False, "error": "response_too_large"}
    assert observations["executor"].calls == []
