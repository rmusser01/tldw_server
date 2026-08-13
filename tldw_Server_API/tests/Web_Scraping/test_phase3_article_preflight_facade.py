from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.contracts import (
    PreflightAdvice,
    PreflightResult,
    RuntimeFailure,
    WebScrapingStatus,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleLimits,
    ArticlePlan,
    DirectBrowserProfile,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)

URL = "https://example.com/article"
ANALYSIS = {"results": {"headers": {"status": "success"}}}


class FakePolicyChecker:
    def __init__(
        self,
        decision: PolicyDecision,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def decide(self, url: str, **kwargs: Any) -> PolicyDecision:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.decision


class FakeFetchClient:
    def __init__(self, responses: Sequence[FetchResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeCanonicalBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, DirectBrowserProfile, ArticleLimits]] = []

    async def acquire(
        self,
        url: str,
        profile: DirectBrowserProfile,
        limits: ArticleLimits,
    ) -> str:
        self.calls.append((url, profile, limits))
        return "<html><body>rendered</body></html>"


class FakeExecutor:
    async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def allowed_decision() -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        mode="compat",
        reason="allowed",
        stage="pre_fetch",
        source="article_extract",
    )


def denied_decision() -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        mode="strict",
        reason="robots_disallowed",
        stage="pre_fetch",
        source="article_extract",
    )


def target(decision: PolicyDecision | None = None, *, url: str = URL) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=decision or allowed_decision(),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )


def successful_preflight(analysis: Mapping[str, Any] | None = None) -> PreflightResult:
    return PreflightResult(analysis=analysis or ANALYSIS)


def successful_article(url: str, *, content: str = "article content") -> dict[str, Any]:
    return {
        "url": url,
        "title": "Article",
        "author": "Author",
        "date": "2026-07-18",
        "content": content,
        "extraction_successful": True,
    }


def failed_article(url: str) -> dict[str, Any]:
    return {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "No content extracted",
    }


@dataclass
class ArticleHarness:
    article: ModuleType
    policy_checker: FakePolicyChecker
    fetch_client: FakeFetchClient
    evaluate_target: AsyncMock
    build_context: Mock
    run_preflight: AsyncMock
    apply_advice: Mock
    public_payload: Mock
    extractor: Mock
    logs: list[dict[str, str]]


def install_article_defaults(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "auto",
    preflight: bool = True,
    include_results: bool = True,
    decision: PolicyDecision | None = None,
    preflight_result: PreflightResult | None = None,
    fetch_responses: Sequence[FetchResponse | BaseException] | None = None,
    extraction_results: Sequence[dict[str, Any]] | None = None,
) -> ArticleHarness:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    selected_decision = decision or allowed_decision()
    selected_result = preflight_result or successful_preflight()
    config = {
        "web_scraper_preflight_analyzers": preflight,
        "web_scraper_preflight_include_results": include_results,
        "web_scraper_preflight_scan_depth": "default",
        "web_scraper_preflight_timeout_s": 0,
        "web_scraper_default_backend": "auto",
        "web_scraper_retry_count": 1,
        "web_scraper_retry_timeout": 1,
        "web_scraper_stealth_playwright": False,
    }
    policy_checker = FakePolicyChecker(denied_decision())
    fetch_client = FakeFetchClient(
        fetch_responses
        if fetch_responses is not None
        else [
            FetchResponse(
                url=URL,
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>article</body></html>",
                backend="httpx",
            )
        ]
    )
    extractor = Mock(
        side_effect=list(extraction_results or [successful_article(URL)]),
    )
    evaluate = AsyncMock(return_value=target(selected_decision))
    build_context = Mock(return_value=object())
    run = AsyncMock(return_value=selected_result)
    real_apply = preflight_facade.apply_preflight_advice
    real_payload = preflight_facade.public_preflight_payload
    apply_advice = Mock(wraps=real_apply)
    public_payload = Mock(wraps=real_payload)

    browser = FakeCanonicalBrowser()
    logs: list[dict[str, str]] = []

    def resolve_plan(_url: str, _loaded_config: Mapping[str, Any]) -> ArticlePlan:
        headers = {"User-Agent": "article-test-agent"}
        return ArticlePlan(
            url=URL,
            domain="example.com",
            browser=DirectBrowserProfile(
                user_agent=headers["User-Agent"],
                custom_cookies=(),
                retries=1,
                timeout_ms=1_000,
                stealth_enabled=False,
                stealth_wait_ms=0,
            ),
            backend=backend,
            handler="",
            headers=headers,
            respect_robots=True,
            limits=ArticleLimits(max_article_bytes=4_096, max_browser_transfer_bytes=8_192),
        )

    dependencies = canonical.ArticleDependencies(
        load_config=lambda: {"web_scraper": config},
        resolve_plan=resolve_plan,
        evaluate_target=evaluate,
        run_preflight=run,
        apply_preflight_advice=apply_advice,
        fetch_client=fetch_client,
        browser=browser,
        executor=FakeExecutor(),
        extract=extractor,
        build_preflight_context=build_context,
        preflight_options=preflight_facade.PreflightOptions.from_mapping,
        public_preflight_payload=public_payload,
        resolve_handler=lambda _path: None,
        js_required=lambda *_args, **_kwargs: False,
        convert_content=lambda content: content,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        clock=lambda: 0.0,
        log=lambda _message, **fields: logs.append(fields),
        policy_checker=policy_checker,
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)
    return ArticleHarness(
        article=article,
        policy_checker=policy_checker,
        fetch_client=fetch_client,
        evaluate_target=evaluate,
        build_context=build_context,
        run_preflight=run,
        apply_advice=apply_advice,
        public_payload=public_payload,
        extractor=extractor,
        logs=logs,
    )


@pytest.mark.unit
async def test_article_denied_target_stops_before_context_preflight_or_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch, decision=denied_decision())

    result = await harness.article.scrape_article(URL)

    assert result == {
        "url": URL,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "Blocked by outbound policy",
        "policy_reason": "robots_disallowed",
        "policy_mode": "strict",
        "policy_stage": "pre_fetch",
        "policy_source": "article_extract",
    }
    harness.evaluate_target.assert_awaited_once()
    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()
    assert harness.fetch_client.requests == []
    harness.extractor.assert_not_called()


@pytest.mark.unit
async def test_article_policy_failure_is_structural_and_logs_no_sensitive_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_url = "https://user:password@example.com/article?token=secret-value"
    harness = install_article_defaults(monkeypatch)
    error = RuntimeError(f"credential leaked for {sensitive_url}")
    harness.evaluate_target.side_effect = error
    harness.policy_checker.error = error
    result = await harness.article.scrape_article(sensitive_url)

    assert result == {
        "url": sensitive_url,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "policy_error",
    }
    assert harness.logs == [
        {
            "exception_type": "RuntimeError",
            "code": "policy_error",
            "stage": "pre_fetch",
            "host": "example.com",
        }
    ]
    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()
    assert harness.fetch_client.requests == []


@pytest.mark.unit
async def test_article_disabled_still_admits_but_creates_no_preflight_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch, preflight=False, include_results=True)

    result = await harness.article.scrape_article(URL)

    assert result == successful_article(URL)
    harness.evaluate_target.assert_awaited_once()
    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()
    harness.apply_advice.assert_called_once_with(
        None,
        backend="auto",
        method="auto",
        backend_setting="auto",
    )
    harness.public_payload.assert_called_once_with(None, True)
    assert "preflight_analysis" not in result


@pytest.mark.unit
async def test_article_evaluates_primary_policy_once_with_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch)

    await harness.article.scrape_article(URL)

    harness.evaluate_target.assert_awaited_once()
    call = harness.evaluate_target.await_args
    assert call.args == (URL,)
    assert call.kwargs["respect_robots"] is True
    assert call.kwargs["user_agent"] == "article-test-agent"
    assert call.kwargs["policy_checker"] is harness.policy_checker
    assert call.kwargs["config"]["web_scraper"]["web_scraper_preflight_analyzers"] is True
    context = call.kwargs["request_context"]
    assert context.source == "article_extract"
    assert context.stage == "pre_fetch"
    harness.build_context.assert_called_once()
    assert harness.build_context.call_args.kwargs["policy_checker"] is harness.policy_checker
    harness.run_preflight.assert_awaited_once()
    assert harness.policy_checker.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("analysis", "expect_playwright"),
    [
        ({"results": {"js": {"status": "success", "js_required": True}}}, True),
        ({"results": {"js": {"status": "error", "js_required": True}}}, False),
        ({"results": {"js": {"status": "success", "js_required": False}}}, False),
    ],
)
async def test_article_js_routing_uses_only_successful_facade_signals(
    monkeypatch: pytest.MonkeyPatch,
    analysis: dict[str, Any],
    expect_playwright: bool,
) -> None:
    harness = install_article_defaults(
        monkeypatch,
        preflight_result=successful_preflight(analysis),
    )

    result = await harness.article.scrape_article(URL)

    assert result["extraction_successful"] is True
    assert (harness.fetch_client.requests == []) is expect_playwright
    expected_method = "playwright" if expect_playwright else "auto"
    assert result["preflight_analysis"]["advice"]["method"] == expected_method


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend_setting", "expected_request_backend", "expected_advice_backend"),
    [("auto", "curl", "curl"), ("httpx", "httpx", "httpx")],
)
async def test_article_tls_routing_changes_only_automatic_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_setting: str,
    expected_request_backend: str,
    expected_advice_backend: str,
) -> None:
    harness = install_article_defaults(
        monkeypatch,
        backend=backend_setting,
        preflight_result=successful_preflight(
            {"results": {"tls": {"status": "active"}}},
        ),
        fetch_responses=[
            FetchResponse(
                url=URL,
                status=200,
                text="<html><body>article</body></html>",
                backend=expected_request_backend,
            )
        ],
    )

    result = await harness.article.scrape_article(URL)

    assert harness.fetch_client.requests[0].backend == expected_request_backend
    assert result["preflight_analysis"]["advice"]["backend"] == expected_advice_backend


@pytest.mark.unit
@pytest.mark.parametrize("status", [WebScrapingStatus.TIMEOUT, WebScrapingStatus.ERROR])
async def test_article_non_ok_preflight_preserves_route_and_omits_payload(
    monkeypatch: pytest.MonkeyPatch,
    status: WebScrapingStatus,
) -> None:
    result_contract = PreflightResult(
        analysis={"results": {"js": {"status": "success", "js_required": True}}},
        advice=PreflightAdvice(backend="curl", method="playwright"),
        status=status,
        failure=RuntimeFailure(status=status, public_message="safe preflight failure"),
    )
    harness = install_article_defaults(
        monkeypatch,
        backend="httpx",
        preflight_result=result_contract,
    )

    result = await harness.article.scrape_article(URL)

    assert harness.fetch_client.requests[0].backend == "httpx"
    assert "preflight_analysis" not in result
    harness.public_payload.assert_called_once()


@pytest.mark.unit
async def test_article_analyzer_scoped_error_retains_ok_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {
        "results": {
            "headers": {"status": "error", "message": "probe failed"},
            "js": {"status": "success", "js_required": False},
        }
    }
    harness = install_article_defaults(
        monkeypatch,
        preflight_result=successful_preflight(analysis),
    )

    result = await harness.article.scrape_article(URL)

    assert result["preflight_analysis"] == {
        "analysis": analysis,
        "advice": {"backend": "auto", "method": "auto", "notes": []},
    }


@pytest.mark.unit
async def test_article_payload_attaches_to_lightweight_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch)

    result = await harness.article.scrape_article(URL)

    assert result == {
        **successful_article(URL),
        "preflight_analysis": {
            "analysis": ANALYSIS,
            "advice": {"backend": "auto", "method": "auto", "notes": []},
        },
    }


@pytest.mark.unit
async def test_article_payload_attaches_to_direct_playwright_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {"results": {"js": {"status": "success", "js_required": True}}}
    harness = install_article_defaults(
        monkeypatch,
        preflight_result=successful_preflight(analysis),
        extraction_results=[successful_article(URL, content="rendered")],
    )

    result = await harness.article.scrape_article(URL)

    assert harness.fetch_client.requests == []
    assert result["content"] == "rendered"
    assert result["preflight_analysis"]["advice"]["method"] == "playwright"


@pytest.mark.unit
async def test_article_payload_attaches_to_playwright_fallback_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(
        monkeypatch,
        extraction_results=[failed_article(URL), successful_article(URL, content="fallback")],
    )

    result = await harness.article.scrape_article(URL)

    assert len(harness.fetch_client.requests) == 1
    assert result["content"] == "fallback"
    assert "preflight_analysis" in result


@pytest.mark.unit
async def test_article_payload_attaches_to_final_extraction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(
        monkeypatch,
        extraction_results=[failed_article(URL), failed_article(URL)],
    )

    result = await harness.article.scrape_article(URL)

    assert result["extraction_successful"] is False
    assert result["error"] == "No content extracted"
    assert "preflight_analysis" in result


@pytest.mark.unit
async def test_article_evaluate_target_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch)
    harness.evaluate_target.side_effect = asyncio.CancelledError
    harness.policy_checker.error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await harness.article.scrape_article(URL)

    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()


@pytest.mark.unit
async def test_article_run_preflight_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch)
    harness.run_preflight.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await harness.article.scrape_article(URL)


def test_article_public_signature_is_unchanged() -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    signature = inspect.signature(article.scrape_article)
    assert list(signature.parameters) == ["url", "custom_cookies", "allow_llm_extraction"]
    assert signature.parameters["url"].default is inspect.Signature.empty
    assert signature.parameters["custom_cookies"].default is None
    assert signature.parameters["allow_llm_extraction"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["allow_llm_extraction"].default is True
    assert get_type_hints(article.scrape_article) == {
        "url": str,
        "custom_cookies": list[dict[str, Any]] | None,
        "allow_llm_extraction": bool,
        "return": dict[str, Any],
    }


def test_article_consumer_has_one_package_facade_and_no_private_policy_helper() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article

    source = inspect.getsource(article)
    assert not hasattr(article, "_decide_article_pre_fetch_policy")
    assert "Web_Scraping.scraper_analyzers" not in source
    assert source.count("Web_Scraping import preflight as preflight_facade") == 1
