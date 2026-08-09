import dataclasses

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline, throttles
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies


def _install_dependencies(monkeypatch, provider, *, sleep, wall_time):
    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=provider,
        sleep=sleep,
        wall_time=wall_time,
    )
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)


def test_llm_throttling_applies_delay(monkeypatch):
    monkeypatch.setenv("LLM_DELAY_MS", "50")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")

    sleeps = []

    def _fake_call(**_kwargs):
        return {
            "choices": [{"message": {"content": '{"title": "T", "content": "C"}'}}],
            "usage": {},
            "model": "gpt-test",
        }

    throttles.clear_throttle_state()
    _install_dependencies(
        monkeypatch,
        _fake_call,
        sleep=sleeps.append,
        wall_time=lambda: 1000.0,
    )

    html = "<html><body>" + " ".join(["word"] * 80) + "</body></html>"
    result = ael.extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["llm"],
        llm_settings={
            "provider": "openai",
            "mode": "blocks",
            "chunk_token_threshold": 5,
            "word_token_rate": 1.0,
        },
    )

    assert result["extraction_successful"] is True
    assert any(value >= 0.05 for value in sleeps)


def test_llm_throttling_uses_env_concurrency(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("LLM_DELAY_MS", "0")

    calls = {"max": None, "acquire": 0, "release": 0}

    class DummySemaphore:
        def acquire(self):
            calls["acquire"] += 1
            return True

        def release(self):
            calls["release"] += 1

    def fake_get(provider, max_concurrency):
        calls["max"] = max_concurrency
        return DummySemaphore()

    monkeypatch.setattr(throttles, "get_llm_semaphore", fake_get)

    def _fake_call(**_kwargs):
        return {
            "choices": [{"message": {"content": '{"title": "T", "content": "C"}'}}],
            "usage": {},
            "model": "gpt-test",
        }

    throttles.clear_throttle_state()
    _install_dependencies(
        monkeypatch,
        _fake_call,
        sleep=lambda _value: None,
        wall_time=lambda: 1000.0,
    )

    html = "<html><body>" + " ".join(["word"] * 80) + "</body></html>"
    result = ael.extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["llm"],
        llm_settings={
            "provider": "openai",
            "mode": "blocks",
            "chunk_token_threshold": 5,
            "word_token_rate": 1.0,
        },
    )

    assert result["extraction_successful"] is True
    assert calls["max"] == 3
    assert calls["acquire"] >= 1
    assert calls["release"] >= 1
