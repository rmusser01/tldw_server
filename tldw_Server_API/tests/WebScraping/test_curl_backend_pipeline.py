import dataclasses
import types

import pytest

from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse


@pytest.mark.asyncio
async def test_scrape_article_uses_curl_backend(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
        ArticleLimits,
        ArticlePlan,
        DirectBrowserProfile,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_router import DEFAULT_HANDLER

    html = """
    <html>
      <body>
        <article>
          <h1>Title</h1>
          <p>Body text</p>
        </article>
      </body>
    </html>
    """

    class FakeFetchClient:
        def __init__(self):
            self.requests = []

        def fetch(self, request):
            self.requests.append(request)
            return FetchResponse(
                url=request.url,
                status=200,
                text=html,
                backend=request.backend,
            )

    fetch_client = FakeFetchClient()
    default_dependencies = canonical._build_default_dependencies

    async def evaluate_target(*_args, **_kwargs):
        return types.SimpleNamespace(decision=types.SimpleNamespace(allowed=True))

    def build_dependencies(cookies):
        plan = ArticlePlan(
            url="https://example.com/path",
            domain="example.com",
            backend="curl",
            handler=DEFAULT_HANDLER,
            browser=DirectBrowserProfile("test", tuple(cookies), 1, 1_000, False, 0),
            respect_robots=False,
            strategy_order=("schema",),
            schema_rules={
                "baseSelector": "//article",
                "fields": [
                    {"name": "title", "selector": ".//h1", "type": "text"},
                    {"name": "content", "selector": ".//p", "type": "text"},
                ],
            },
            limits=ArticleLimits(),
        )
        dependencies = default_dependencies(cookies)
        return dataclasses.replace(
            dependencies,
            load_config=lambda: {"web_scraper": {"web_scraper_preflight_analyzers": False}},
            resolve_plan=lambda _url, _config: plan,
            evaluate_target=evaluate_target,
            fetch_client=fetch_client,
            extract=lambda _html, url, **_kwargs: {
                "url": url,
                "title": "Title",
                "content": "Body text",
                "extraction_successful": True,
            },
            js_required=lambda *_args, **_kwargs: False,
        )

    monkeypatch.setattr(canonical, "_build_default_dependencies", build_dependencies)

    result = await canonical.scrape_article("https://example.com/path")

    assert fetch_client.requests[0].backend == "curl"
    assert result["extraction_successful"] is True
    assert result["title"] == "Title"
