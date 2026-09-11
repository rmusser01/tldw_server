from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.contracts import PreflightResult
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


class FakePolicyChecker:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str,
        context: RuntimeRequestContext,
        config: Mapping[str, Any],
    ) -> PolicyDecision:
        self.calls.append(
            {
                "url": url,
                "respect_robots": respect_robots,
                "user_agent": user_agent,
                "context": context,
                "config": config,
            }
        )
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


def _allowed_target(url: str) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )


def _install_article_defaults(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy_checker: FakePolicyChecker,
    fetch_client: FakeFetchClient,
    backend: str = "httpx",
    web_scraper_config: Mapping[str, Any] | None = None,
) -> ModuleType:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    config = {"web_scraper": web_scraper_config or {}}

    def fake_handler(html: str, url: str) -> dict[str, object]:
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    def resolve_plan(url: str, _config: Mapping[str, Any]) -> ArticlePlan:
        return ArticlePlan(
            url=url,
            domain="example.com",
            backend=backend,
            handler="tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
            headers={"User-Agent": "test-agent"},
            browser=DirectBrowserProfile("test-agent", (), 1, 1_000, False, 0),
            limits=ArticleLimits(4_096, 8_192),
        )

    def extract(html: str, url: str, **kwargs: Any) -> dict[str, object]:
        handler = kwargs["handler"]
        assert handler is not None
        return handler(html, url)

    class Browser:
        async def acquire(self, *_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("browser fallback is not expected")

    class Executor:
        async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    dependencies = canonical.ArticleDependencies(
        load_config=lambda: config,
        resolve_plan=resolve_plan,
        evaluate_target=lambda *args, **kwargs: preflight_facade.evaluate_target(*args, **kwargs),
        run_preflight=lambda *args, **kwargs: preflight_facade.run_preflight(*args, **kwargs),
        apply_preflight_advice=lambda *args, **kwargs: preflight_facade.apply_preflight_advice(*args, **kwargs),
        fetch_client=fetch_client,
        browser=Browser(),
        executor=Executor(),
        extract=extract,
        build_preflight_context=lambda *args, **kwargs: preflight_facade.build_execution_context(*args, **kwargs),
        preflight_options=preflight_facade.PreflightOptions.from_mapping,
        public_preflight_payload=lambda *args, **kwargs: preflight_facade.public_preflight_payload(*args, **kwargs),
        resolve_handler=lambda _: fake_handler,
        js_required=lambda *args, **kwargs: False,
        convert_content=lambda content: content,
        increment_counter=lambda *args, **kwargs: None,
        observe_histogram=lambda *args, **kwargs: None,
        clock=lambda: 0.0,
        log=lambda *args, **kwargs: None,
        policy_checker=policy_checker,
        backend_setting=lambda _plan: backend,
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)
    return ael


@pytest.mark.unit
async def test_scrape_article_uses_runtime_policy_before_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=False,
            mode="strict",
            reason="robots_disallowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient([])
    ael = _install_article_defaults(
        monkeypatch,
        policy_checker=policy_checker,
        fetch_client=fetch_client,
        backend="httpx",
        web_scraper_config={"web_scraper_preflight_analyzers": True},
    )
    build_context = Mock()
    run_preflight = AsyncMock()

    monkeypatch.setattr(preflight_facade, "build_execution_context", build_context)
    monkeypatch.setattr(preflight_facade, "run_preflight", run_preflight)

    result = await ael.scrape_article("https://example.com/path")

    assert result["extraction_successful"] is False
    assert result["policy_reason"] == "robots_disallowed"
    assert policy_checker.calls[0]["url"] == "https://example.com/path"
    assert policy_checker.calls[0]["context"].source == "article_extract"
    assert policy_checker.calls[0]["context"].stage == "pre_fetch"
    assert fetch_client.requests == []
    build_context.assert_not_called()
    run_preflight.assert_not_awaited()


@pytest.mark.unit
async def test_scrape_article_uses_runtime_fetch_client_for_httpx_success(monkeypatch: pytest.MonkeyPatch) -> None:
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient(
        [
            FetchResponse(
                url="https://example.com/path",
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>ok</body></html>",
                backend="httpx",
            )
        ]
    )
    ael = _install_article_defaults(
        monkeypatch,
        policy_checker=policy_checker,
        fetch_client=fetch_client,
        backend="httpx",
    )

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert len(fetch_client.requests) == 1
    request = fetch_client.requests[0]
    assert request.url == "https://example.com/path"
    assert request.method == "GET"
    assert request.backend == "httpx"
    assert request.allow_redirects is False


@pytest.mark.unit
async def test_scrape_article_preserves_curl_to_httpx_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient(
        [
            RuntimeError("curl unavailable"),
            FetchResponse(
                url="https://example.com/path",
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>ok</body></html>",
                backend="httpx",
            ),
        ]
    )
    ael = _install_article_defaults(
        monkeypatch,
        policy_checker=policy_checker,
        fetch_client=fetch_client,
        backend="curl",
    )

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert [request.backend for request in fetch_client.requests] == ["curl", "httpx"]


@pytest.mark.unit
async def test_scrape_article_preflight_tls_advice_still_selects_curl(monkeypatch: pytest.MonkeyPatch) -> None:
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=False,
            mode="strict",
            reason="deny_legacy_path",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient(
        [
            FetchResponse(
                url="https://example.com/path",
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>ok</body></html>",
                backend="curl",
            )
        ]
    )
    ael = _install_article_defaults(
        monkeypatch,
        policy_checker=policy_checker,
        fetch_client=fetch_client,
        backend="auto",
        web_scraper_config={
            "web_scraper_preflight_analyzers": True,
            "web_scraper_preflight_include_results": True,
        },
    )
    monkeypatch.setattr(
        preflight_facade,
        "evaluate_target",
        AsyncMock(return_value=_allowed_target("https://example.com/path")),
    )
    monkeypatch.setattr(
        preflight_facade,
        "run_preflight",
        AsyncMock(
            return_value=PreflightResult(
                analysis={"results": {"tls": {"status": "active"}, "js": {"status": "success"}}}
            )
        ),
    )

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert fetch_client.requests[0].backend == "curl"
    assert result["preflight_analysis"]["advice"]["backend"] == "curl"
    assert "tls_active" in result["preflight_analysis"]["advice"]["notes"]
