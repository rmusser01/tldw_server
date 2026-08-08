import dataclasses
import types

import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm as llm_strategy

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_llm_pipeline_uses_plan_settings(monkeypatch):
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url: types.SimpleNamespace(allowed=True),
    )

    async def allow_robots(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        allow_robots,
    )

    rules = {
        "domains": {
            "example.com": {
                "strategy_order": ["llm"],
                "llm_settings": {
                    "provider": "openai",
                    "mode": "blocks",
                },
            }
        }
    }
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.scraper_router.ScraperRouter.load_rules_from_yaml",
        lambda path: rules,
    )

    calls = {"count": 0}

    def _fake_call(**_kwargs):
        calls["count"] += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"title": "LLM Title", "content": "LLM Body"}\n```',
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            "model": "gpt-test",
        }

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=_fake_call,
    )
    monkeypatch.setattr(llm_strategy, "build_default_dependencies", lambda: dependencies)

    scraper = EnhancedWebScraper(config={"custom_scrapers_yaml_path": "unused"})

    async def fake_fetch_html(*_args, **_kwargs):
        return "<html><body><p>LLM Body</p></body></html>", "httpx", 0.0

    monkeypatch.setattr(scraper, "_fetch_html", fake_fetch_html)

    result = await scraper.scrape_article("https://example.com/path")

    assert calls["count"] == 1
    assert result["extraction_successful"] is True
    assert result["title"] == "LLM Title"
    assert result["content"] == "LLM Body"
    assert result["llm_provider"] == "openai"
    assert result["llm_mode"] == "blocks"
    assert result.get("llm_usage", {}).get("total_tokens") == 8
