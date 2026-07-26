import types  # noqa: I001
import pytest


def test_resolve_handler_fallback():
    from tldw_Server_API.app.core.Web_Scraping import handlers

    handler = handlers.resolve_handler("not-a-module")
    assert handler is handlers.handle_generic_html


def test_resolve_handler_valid_path():
    from tldw_Server_API.app.core.Web_Scraping import handlers

    handler = handlers.resolve_handler(
        "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
    )
    assert handler is handlers.handle_generic_html


@pytest.mark.asyncio
async def test_scrape_article_uses_handler(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael  # noqa: I001
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision

    policy_decide = AsyncMock(
        return_value=PolicyDecision(
            allowed=True,
            reason="allowed",
            mode="test",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    policy_checker = types.SimpleNamespace(decide=policy_decide)
    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "load_and_log_configs", lambda: {"web_scraper": {}})
    monkeypatch.setattr(ael, "_js_required", lambda *args, **kwargs: False)

    def fake_http_fetch(*args, **kwargs):
        return (
            types.SimpleNamespace(
                status=200,
                text="<html><body><p>ok</p></body></html>",
                headers={},
            ),
            "httpx",
        )

    def fake_handler(html, url):
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(ael, "_fetch_article_lightweight", fake_http_fetch)
    monkeypatch.setattr(ael, "resolve_handler", lambda _: fake_handler)

    result = await ael.scrape_article("https://example.com")

    policy_decide.assert_awaited_once()
    assert result["title"] == "handled"
    assert result["content"] == "handled-content"
