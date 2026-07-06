import types
import pytest

from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter


@pytest.mark.asyncio
async def test_scrape_article_uses_curl_backend(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
    from tldw_Server_API.app.core.Security import egress as eg

    pol = types.SimpleNamespace(allowed=True, reason="ok")
    monkeypatch.setattr(eg, "evaluate_url_policy", lambda _url: pol)

    monkeypatch.setattr(
        ael,
        "load_and_log_configs",
        lambda: {
            "web_scraper": {
                "custom_scrapers_yaml_path": "unused",
                "web_scraper_ua_mode": "fixed",
                "web_scraper_respect_robots": False,
            }
        },
    )

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
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)

    rules = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "strategy_order": ["schema"],
                    "schema_rules": {
                        "baseSelector": "//article",
                        "fields": [
                            {"name": "title", "selector": ".//h1", "type": "text"},
                            {"name": "content", "selector": ".//p", "type": "text"},
                        ],
                    },
                    "respect_robots": False,
                }
            }
        }
    )
    monkeypatch.setattr(ael.ScraperRouter, "load_rules_from_yaml", lambda _path: rules)

    result = await ael.scrape_article("https://example.com/path")

    assert fetch_client.requests[0].backend == "curl"
    assert result["extraction_successful"] is True
    assert result["title"] == "Title"
