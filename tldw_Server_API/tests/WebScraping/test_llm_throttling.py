"""Tests for canonical LLM extraction throttling."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline, throttles
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies


def _install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    provider: Callable[..., Any],
    *,
    sleep: Callable[[float], None],
    wall_time: Callable[[], float],
) -> None:
    """Install deterministic LLM dependencies for a pipeline test."""

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=provider,
        sleep=sleep,
        wall_time=wall_time,
    )
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)


def test_llm_throttling_applies_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DELAY_MS", "50")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")

    sleeps: list[float] = []

    def _fake_call(**_kwargs: Any) -> dict[str, Any]:
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


def test_llm_throttling_uses_env_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("LLM_DELAY_MS", "0")

    calls: dict[str, int | None] = {"max": None, "acquire": 0, "release": 0}

    class DummySemaphore:
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            del blocking, timeout
            calls["acquire"] = int(calls["acquire"] or 0) + 1
            return True

        def release(self) -> None:
            calls["release"] = int(calls["release"] or 0) + 1

    def fake_get(provider: str | None, max_concurrency: int) -> DummySemaphore:
        del provider
        calls["max"] = max_concurrency
        return DummySemaphore()

    monkeypatch.setattr(throttles, "get_llm_semaphore", fake_get)

    def _fake_call(**_kwargs: Any) -> dict[str, Any]:
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
