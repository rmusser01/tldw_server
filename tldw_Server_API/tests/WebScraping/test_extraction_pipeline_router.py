"""Tests for extraction strategy routing and pipeline overrides."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    DEFAULT_EXTRACTION_STRATEGY_ORDER,
    extract_article_with_pipeline,
)
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter


def _install_llm_provider(monkeypatch: pytest.MonkeyPatch, provider: Callable[..., Any]) -> None:
    """Install a deterministic LLM provider for a pipeline routing test."""

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=provider,
    )
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)


def test_default_extraction_strategy_order_includes_llm_after_regex():
    assert DEFAULT_EXTRACTION_STRATEGY_ORDER == [
        "jsonld",
        "schema",
        "regex",
        "llm",
        "cluster",
        "trafilatura",
    ]


def test_pipeline_trace_default_order(monkeypatch):
    def _fake_llm_call(**_kwargs):
        return {"choices": [{"message": {"content": ""}}], "usage": {}}

    _install_llm_provider(monkeypatch, _fake_llm_call)

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


def test_default_pipeline_omits_only_llm_when_disallowed(monkeypatch):
    llm_calls = []

    def _fake_llm_call(**kwargs):
        llm_calls.append(kwargs)
        return {"choices": [{"message": {"content": ""}}], "usage": {}}

    _install_llm_provider(monkeypatch, _fake_llm_call)

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
        allow_llm_extraction=False,
    )

    assert result["extraction_successful"] is True
    assert llm_calls == []
    assert result["extraction_strategy_order"] == [
        strategy for strategy in DEFAULT_EXTRACTION_STRATEGY_ORDER if strategy != "llm"
    ]
    assert all(entry["strategy"] != "llm" for entry in result["extraction_trace"])


def test_default_pipeline_preserves_llm_when_allowed(monkeypatch):
    _install_llm_provider(
        monkeypatch,
        lambda **_kwargs: {"choices": [{"message": {"content": ""}}], "usage": {}},
    )

    result = extract_article_with_pipeline(
        "<html><body><p>Plain article content.</p></body></html>",
        "https://example.com/default-allowed",
        allow_llm_extraction=True,
    )

    assert result["extraction_strategy_order"] == DEFAULT_EXTRACTION_STRATEGY_ORDER


@pytest.mark.parametrize("allow_llm_extraction", [False, True])
def test_custom_pipeline_filters_only_llm_when_disallowed(
    monkeypatch,
    allow_llm_extraction,
):
    llm_calls = []

    def fake_llm_call(**kwargs):
        llm_calls.append(kwargs)
        return {"choices": [{"message": {"content": ""}}], "usage": {}}

    _install_llm_provider(monkeypatch, fake_llm_call)
    custom_order = ["llm", "trafilatura"]

    def fake_extractor(_html: str, url: str):
        return {
            "url": url,
            "title": "Test",
            "author": "N/A",
            "date": "N/A",
            "content": "Fallback content",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        "<html><body><p>Custom ordered article content.</p></body></html>",
        "https://example.com/custom-order",
        strategy_order=custom_order,
        allow_llm_extraction=allow_llm_extraction,
        fallback_extractor=fake_extractor,
    )

    expected_order = custom_order if allow_llm_extraction else ["trafilatura"]
    expected_trace = expected_order
    assert len(llm_calls) == (1 if allow_llm_extraction else 0)
    assert [entry["strategy"] for entry in result["extraction_trace"]] == expected_trace
    assert result["extraction_strategy_order"] == expected_order


def test_pipeline_strategy_order_override_from_router():
    rules = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "strategy_order": ["schema", "llm", "trafilatura"],
                }
            }
        }
    )
    router = ScraperRouter(rules)
    plan = router.resolve("https://example.com/page")

    assert plan.strategy_order == ["schema", "llm", "trafilatura"]

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
        allow_llm_extraction=False,
    )
    assert [entry["strategy"] for entry in result["extraction_trace"]] == ["schema", "trafilatura"]
    assert result["extraction_strategy_order"] == ["schema", "trafilatura"]


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
