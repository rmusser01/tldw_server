from typing import Any

import pytest


@pytest.mark.asyncio
async def test_js_required_emits_one_bounded_fallback_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
        ArticleLimits,
        ArticlePlan,
        DirectBrowserProfile,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
    from tldw_Server_API.app.core.Web_Scraping.runtime import (
        FetchResponse,
        PolicyDecision,
        RuntimeRequestContext,
    )

    calls: list[tuple[str, dict[str, str]]] = []

    def _increment_counter(name, value=1, labels=None):
        calls.append((name, dict(labels or {})))

    class FetchClient:
        def fetch(self, request: Any) -> FetchResponse:
            return FetchResponse(
                url=request.url,
                status=200,
                headers={},
                text="Please enable JavaScript to continue",
                backend="httpx",
            )

    class Browser:
        async def acquire(self, *_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("playwright disabled in test")

    class Executor:
        async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    plan = ArticlePlan(
        url="https://example.com",
        domain="example.com",
        backend="httpx",
        headers={"User-Agent": "test-agent"},
        browser=DirectBrowserProfile("test-agent", (), 1, 1000, False, 0),
        limits=ArticleLimits(4096, 8192),
    )

    async def evaluate_target(url: str, **_kwargs: Any) -> PreflightTarget:
        return PreflightTarget(
            url=url,
            decision=PolicyDecision(True, "test", "allowed", "pre_fetch", "article_extract"),
            request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        )

    dependencies = canonical.ArticleDependencies(
        load_config=lambda: {"web_scraper": {"web_scraper_preflight_analyzers": False}},
        resolve_plan=lambda _url, _config: plan,
        evaluate_target=evaluate_target,
        run_preflight=lambda *_args, **_kwargs: None,
        apply_preflight_advice=lambda result, **kwargs: (kwargs["backend"], kwargs["method"], result),
        fetch_client=FetchClient(),
        browser=Browser(),
        executor=Executor(),
        extract=lambda *_args, **_kwargs: {"extraction_successful": False},
        build_preflight_context=lambda *_args, **_kwargs: object(),
        preflight_options=canonical.preflight_facade.PreflightOptions.from_mapping,
        public_preflight_payload=lambda *_args, **_kwargs: None,
        resolve_handler=lambda _path: None,
        js_required=canonical._js_required,
        convert_content=lambda content: content,
        increment_counter=_increment_counter,
        observe_histogram=lambda *_args, **_kwargs: None,
        clock=lambda: 0.0,
        log=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    res = await legacy.scrape_article("https://example.com")
    assert res["extraction_successful"] is False
    assert [
        metric for metric in calls if metric == ("scrape_playwright_fallback_total", {"reason": "js_required"})
    ] == [("scrape_playwright_fallback_total", {"reason": "js_required"})]
    assert ("scrape_playwright_fallback_total", {"reason": "no_extract"}) not in calls
