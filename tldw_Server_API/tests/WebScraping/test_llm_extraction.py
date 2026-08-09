import dataclasses

from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import extract_article_with_pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies


def _install_provider(monkeypatch, provider):
    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=provider,
    )
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)


def test_llm_extraction_parses_code_fenced_json(monkeypatch):
    def _fake_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": 'Here you go:\n```json\n{"title": "Hello", "content": "Body"}\n```',
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            "model": "gpt-test",
        }

    _install_provider(monkeypatch, _fake_call)

    result = extract_article_with_pipeline(
        "<html><body><p>Body</p></body></html>",
        "https://example.com",
        strategy_order=["llm"],
        llm_settings={"provider": "openai", "mode": "blocks"},
    )

    assert result["extraction_successful"] is True
    assert result["title"] == "Hello"
    assert result["content"] == "Body"
    assert result.get("llm_usage", {}).get("prompt_tokens") == 3


def test_llm_extraction_strict_json_rejects_extras(monkeypatch):
    def _fake_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": 'Prefix {"title": "Hello", "content": "Body"} Suffix',
                    }
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            "model": "gpt-test",
        }

    _install_provider(monkeypatch, _fake_call)

    result = extract_article_with_pipeline(
        "<html><body><p>Body</p></body></html>",
        "https://example.com",
        strategy_order=["llm"],
        llm_settings={"provider": "openai", "mode": "blocks", "strict_json": True},
    )

    assert result["extraction_successful"] is False
    assert result.get("llm_error") == "strict_json_failed"
