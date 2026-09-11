"""Tests for canonical LLM extraction throttling."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction import throttles


@pytest.fixture(autouse=True)
def _isolate_throttle_state() -> Iterator[None]:
    """Keep process-global throttle state isolated between tests."""

    throttles.clear_throttle_state()
    try:
        yield
    finally:
        throttles.clear_throttle_state()


def test_llm_throttling_applies_delay(
    monkeypatch: pytest.MonkeyPatch,
    install_extraction_dependencies,
) -> None:
    monkeypatch.setenv("LLM_DELAY_MS", "50")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")

    sleeps: list[float] = []

    def _fake_call(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"title": "T", "content": "C"}'}}],
            "usage": {},
            "model": "gpt-test",
        }

    install_extraction_dependencies(
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
    assert any(value == pytest.approx(0.05) for value in sleeps)


def test_llm_throttling_uses_env_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    install_extraction_dependencies,
) -> None:
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

    install_extraction_dependencies(
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


def test_llm_delay_reserves_sequential_provider_slots() -> None:
    sleeps: list[float] = []

    throttles.apply_llm_delay(
        "openai",
        50.0,
        0.0,
        wall_time=lambda: 1000.0,
        sleep=sleeps.append,
    )
    throttles.apply_llm_delay(
        "openai",
        50.0,
        0.0,
        wall_time=lambda: 1000.0,
        sleep=sleeps.append,
    )
    throttles.apply_llm_delay(
        "openai",
        50.0,
        0.0,
        wall_time=lambda: 1000.0,
        sleep=sleeps.append,
    )
    throttles.apply_llm_delay(
        "anthropic",
        50.0,
        0.0,
        wall_time=lambda: 1000.0,
        sleep=sleeps.append,
    )

    assert sleeps == pytest.approx([0.05, 0.1])
