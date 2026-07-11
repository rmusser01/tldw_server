import pytest

from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    DEFAULT_EXTRACTION_STRATEGY_ORDER,
    extract_article_with_pipeline,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter


def test_pipeline_trace_default_order(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service

    def _fake_llm_call(**_kwargs):
        return {"choices": [{"message": {"content": ""}}], "usage": {}}

    monkeypatch.setattr(chat_service, "perform_chat_api_call", _fake_llm_call)

    def fake_extractor(html: str, url: str):  # noqa: ANN001
        return {
            "url": url,
            "title": "Test",
            "author": "N/A",
            "date": "N/A",
            "content": "Hello",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        "<html><body><p>hello</p></body></html>",
        "https://example.com",
        fallback_extractor=fake_extractor,
    )

    assert result["extraction_successful"] is True
    assert result["extraction_strategy"] == "cluster"
    expected = list(DEFAULT_EXTRACTION_STRATEGY_ORDER)
    stop_at = expected.index(result["extraction_strategy"]) + 1
    assert [entry["strategy"] for entry in result["extraction_trace"]] == expected[:stop_at]


def test_default_pipeline_does_not_call_llm_provider(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service

    llm_calls = []

    def _fake_llm_call(**kwargs):
        llm_calls.append(kwargs)
        return {"choices": [{"message": {"content": ""}}], "usage": {}}

    monkeypatch.setattr(chat_service, "perform_chat_api_call", _fake_llm_call)

    result = extract_article_with_pipeline(
        """
        <html>
          <body>
            <p>Plain article content without structured metadata.</p>
            <p>Another paragraph for fallback extraction.</p>
          </body>
        </html>
        """,
        "https://example.com/plain",
    )

    assert result["extraction_successful"] is True
    assert llm_calls == []
    assert "llm" not in result["extraction_strategy_order"]
    assert all(entry["strategy"] != "llm" for entry in result["extraction_trace"])


def test_pipeline_strategy_order_override_from_router():
    rules = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "strategy_order": ["schema", "trafilatura"],
                }
            }
        }
    )
    router = ScraperRouter(rules)
    plan = router.resolve("https://example.com/page")

    assert plan.strategy_order == ["schema", "trafilatura"]

    def fake_extractor(html: str, url: str):  # noqa: ANN001
        return {
            "url": url,
            "title": "Test",
            "author": "N/A",
            "date": "N/A",
            "content": "Hello",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        "<html><body><p>hello</p></body></html>",
        "https://example.com/page",
        strategy_order=plan.strategy_order,
        fallback_extractor=fake_extractor,
    )
    assert [entry["strategy"] for entry in result["extraction_trace"]] == ["schema", "trafilatura"]


def test_pipeline_handler_stage_short_circuits():
    def handler(html: str, url: str):  # noqa: ANN001
        return {
            "url": url,
            "title": "Handled",
            "author": "N/A",
            "date": "N/A",
            "content": "Handled",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        "<html><body><p>hello</p></body></html>",
        "https://example.com/handled",
        strategy_order=["schema", "trafilatura"],
        handler=handler,
    )
    assert result["extraction_successful"] is True
    assert result["extraction_strategy"] == "schema"
    assert [entry["strategy"] for entry in result["extraction_trace"]] == ["schema"]
