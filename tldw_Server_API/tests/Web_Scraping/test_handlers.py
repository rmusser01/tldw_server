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


def test_generic_html_handler_uses_canonical_extraction_and_converts_successful_content(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import content, extraction, handlers

    result = {"content": "<p>body</p>", "extraction_successful": True}
    calls = []

    def extract(html, url):
        calls.append(("extract", html, url))
        return result

    def convert(content):
        calls.append(("convert", content))
        return "body"

    monkeypatch.setattr(extraction, "extract_article_data_from_html", extract)
    monkeypatch.setattr(content, "convert_html_to_markdown", convert)

    actual = handlers.handle_generic_html("<html>body</html>", "https://example.com/article")

    assert actual is result
    assert actual["content"] == "body"
    assert calls == [
        ("extract", "<html>body</html>", "https://example.com/article"),
        ("convert", "<p>body</p>"),
    ]


@pytest.mark.parametrize(
    ("result", "expected_calls"),
    [
        ({"content": "<p>body</p>", "extraction_successful": False}, [("extract",)]),
        ({"content": "", "extraction_successful": True}, [("extract",)]),
    ],
)
def test_generic_html_handler_preserves_failure_or_empty_content_without_markdown(monkeypatch, result, expected_calls):
    from tldw_Server_API.app.core.Web_Scraping import content, extraction, handlers

    calls = []

    def extract(html, url):
        calls.append(("extract", html, url))
        return result

    def convert(_content):
        calls.append(("convert",))
        return "unexpected"

    monkeypatch.setattr(extraction, "extract_article_data_from_html", extract)
    monkeypatch.setattr(content, "convert_html_to_markdown", convert)

    actual = handlers.handle_generic_html("<html>body</html>", "https://example.com/article")

    assert actual is result
    assert [call[:1] for call in calls] == expected_calls


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
