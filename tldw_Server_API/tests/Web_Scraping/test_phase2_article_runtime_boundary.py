from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.contracts import PreflightResult
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
    backend: str = "httpx",
    web_scraper_config: Mapping[str, Any] | None = None,
) -> ModuleType:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael

    config = {"web_scraper": web_scraper_config or {}}
    monkeypatch.setattr(ael, "load_and_log_configs", lambda: config)
    monkeypatch.setattr(ael, "_js_required", lambda *args, **kwargs: False)

    rules = {
        "domains": {
            "example.com": {
                "backend": backend,
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
            }
        }
    }
    monkeypatch.setattr(ael.ScraperRouter, "load_rules_from_yaml", lambda path: rules)

    def fake_handler(html: str, url: str) -> dict[str, object]:
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(ael, "resolve_handler", lambda _: fake_handler)
    monkeypatch.setattr(ael, "observe_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(ael, "increment_counter", lambda *args, **kwargs: None)
    return ael


@pytest.mark.unit
async def test_scrape_article_uses_runtime_policy_before_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    ael = _install_article_defaults(
        monkeypatch,
        backend="httpx",
        web_scraper_config={"web_scraper_preflight_analyzers": True},
    )
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
    build_context = Mock()
    run_preflight = AsyncMock()

    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(ael, "preflight_facade", preflight_facade, raising=False)
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
    ael = _install_article_defaults(
        monkeypatch,
        backend="httpx",
    )
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
    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert len(fetch_client.requests) == 1
    request = fetch_client.requests[0]
    assert request.url == "https://example.com/path"
    assert request.method == "GET"
    assert request.backend == "httpx"
    assert request.allow_redirects is True


@pytest.mark.unit
async def test_scrape_article_preserves_curl_to_httpx_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    ael = _install_article_defaults(monkeypatch, backend="curl")
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
    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert [request.backend for request in fetch_client.requests] == ["curl", "httpx"]


@pytest.mark.unit
async def test_scrape_article_preflight_tls_advice_still_selects_curl(monkeypatch: pytest.MonkeyPatch) -> None:
    ael = _install_article_defaults(
        monkeypatch,
        backend="auto",
        web_scraper_config={
            "web_scraper_preflight_analyzers": True,
            "web_scraper_preflight_include_results": True,
        },
    )
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

    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(ael, "preflight_facade", preflight_facade, raising=False)
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
