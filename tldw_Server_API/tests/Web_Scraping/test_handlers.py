import dataclasses
import threading
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

    def extract(html, url, *, allow_llm_extraction=True):
        calls.append(("extract", html, url, allow_llm_extraction))
        return result

    def convert(content):
        calls.append(("convert", content))
        return "body"

    monkeypatch.setattr(extraction, "extract_article_data_from_html", extract)
    monkeypatch.setattr(content, "convert_html_to_markdown", convert)

    actual = handlers.handle_generic_html(
        "<html>body</html>",
        "https://example.com/article",
        allow_llm_extraction=False,
    )

    assert actual is result
    assert actual["content"] == "body"
    assert calls == [
        ("extract", "<html>body</html>", "https://example.com/article", False),
        ("convert", "<p>body</p>"),
    ]


@pytest.mark.parametrize("allow_llm_extraction", [False, True])
def test_pipeline_forwards_llm_policy_to_generic_handler(
    monkeypatch: pytest.MonkeyPatch,
    allow_llm_extraction: bool,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import extraction, handlers

    received: list[bool] = []

    def extract(_html: str, url: str, *, allow_llm_extraction: bool) -> dict[str, object]:
        received.append(allow_llm_extraction)
        return {
            "url": url,
            "content": "nested content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(extraction, "extract_article_data_from_html", extract)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.content.convert_html_to_markdown",
        lambda content: content,
    )

    result = extraction.extract_article_with_pipeline(
        "<html><body>body</body></html>",
        "https://example.com/article",
        strategy_order=["schema"],
        handler=handlers.handle_generic_html,
        allow_llm_extraction=allow_llm_extraction,
    )

    assert result["extraction_successful"] is True
    assert received == [allow_llm_extraction]


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

    def extract(html, url, *, allow_llm_extraction=True):
        del allow_llm_extraction
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
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
        ArticleLimits,
        ArticlePlan,
        DirectBrowserProfile,
    )
    from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, PolicyDecision

    policy_decide = AsyncMock(
        return_value=PolicyDecision(
            allowed=True,
            reason="allowed",
            mode="test",
            stage="pre_fetch",
            source="article_extract",
        )
    )

    event_loop_thread = threading.get_ident()
    extraction_threads = []
    default_dependencies = canonical._build_default_dependencies

    def fake_handler(html, url):
        extraction_threads.append(threading.get_ident())
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    class FetchClient:
        def fetch(self, request):
            return FetchResponse(
                url=request.url,
                status=200,
                text="<html><body><p>ok</p></body></html>",
                headers={},
                backend="httpx",
            )

    async def evaluate_target(*_args, **_kwargs):
        return types.SimpleNamespace(decision=await policy_decide())

    def build_dependencies(cookies):
        plan = ArticlePlan(
            url="https://example.com",
            domain="example.com",
            backend="httpx",
            browser=DirectBrowserProfile("test", tuple(cookies), 1, 1_000, False, 0),
            limits=ArticleLimits(),
        )
        dependencies = default_dependencies(cookies)
        return dataclasses.replace(
            dependencies,
            load_config=lambda: {"web_scraper": {"web_scraper_preflight_analyzers": False}},
            resolve_plan=lambda _url, _config: plan,
            evaluate_target=evaluate_target,
            fetch_client=FetchClient(),
            resolve_handler=lambda _path: fake_handler,
            js_required=lambda *_args, **_kwargs: False,
        )

    monkeypatch.setattr(canonical, "_build_default_dependencies", build_dependencies)

    result = await canonical.scrape_article("https://example.com")

    policy_decide.assert_awaited_once()
    assert result["title"] == "handled"
    assert result["content"] == "handled-content"
    assert extraction_threads and extraction_threads[0] != event_loop_thread
