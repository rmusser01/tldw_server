import asyncio
import time

import pytest

from tldw_Server_API.app.core.RAG.rag_service.suggestion_generator import (
    generate_suggestions,
)
import tldw_Server_API.app.core.RAG.rag_service.suggestion_generator as suggestion_generator


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_generate_suggestions_pads_llm_output_to_exact_requested_count(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def _fake_chat_call_async(**_kwargs):  # noqa: ANN001
        return '["What are the key risks?", "what are the key risks?"]'

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _fake_chat_call_async)

    suggestions = await generate_suggestions(
        query="RAG pipelines",
        response_text="Here is a response",
        num_suggestions=5,
        llm_timeout_sec=1.0,
    )

    assert len(suggestions) == 5
    assert suggestions[0] == "What are the key risks?"
    assert len({s.lower() for s in suggestions}) == 5


@pytest.mark.asyncio
async def test_generate_suggestions_timeout_falls_back_quickly_and_deterministically(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def _slow_chat_call_async(**_kwargs):  # noqa: ANN001
        await asyncio.sleep(1.0)
        return '["This should timeout"]'

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _slow_chat_call_async)

    started = time.monotonic()
    first = await generate_suggestions(
        query="How to evaluate retrieval quality?",
        response_text="Long answer text",
        num_suggestions=7,
        llm_timeout_sec=0.02,
    )
    elapsed = time.monotonic() - started

    second = await generate_suggestions(
        query="How to evaluate retrieval quality?",
        response_text="Long answer text",
        num_suggestions=7,
        llm_timeout_sec=0.02,
    )

    assert elapsed < 0.5
    assert len(first) == 7
    assert len(second) == 7
    assert first == second


@pytest.mark.asyncio
async def test_generate_suggestions_fallback_returns_exact_count_for_large_requests(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def _failing_chat_call_async(**_kwargs):  # noqa: ANN001
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _failing_chat_call_async)

    suggestions = await generate_suggestions(
        query="streaming transcription systems",
        response_text="Answer text",
        num_suggestions=10,
        llm_timeout_sec=0.1,
    )

    assert len(suggestions) == 10
    assert len({s.lower() for s in suggestions}) == 10


@pytest.mark.asyncio
async def test_generate_suggestions_sanitizes_llm_exception_details_while_falling_back(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    leaked_secret = "sk-test-secret-value"
    leaked_path = "/Users/alice/private/project/config.env"
    debug_messages: list[str] = []

    class SecretPathError(RuntimeError):
        def __repr__(self) -> str:
            return f"SecretPathError(api_key='{leaked_secret}', path='{leaked_path}')"

    async def _failing_chat_call_async(**_kwargs):  # noqa: ANN001
        raise SecretPathError("LLM unavailable")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _failing_chat_call_async)
    monkeypatch.setattr(suggestion_generator.logger, "debug", debug_messages.append)

    suggestions = await generate_suggestions(
        query="streaming transcription systems",
        response_text="Answer text",
        num_suggestions=6,
        llm_timeout_sec=0.1,
    )

    assert len(suggestions) == 6
    assert len({s.lower() for s in suggestions}) == 6
    assert len(debug_messages) == 1
    assert "Suggestion generation LLM call failed" in debug_messages[0]
    assert leaked_secret not in debug_messages[0]
    assert leaked_path not in debug_messages[0]


@pytest.mark.asyncio
async def test_generate_suggestions_parses_fenced_json_with_think_tags(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def _fake_chat_call_async(**_kwargs):  # noqa: ANN001
        return (
            "<think>reasoning</think>\n"
            "```json\n"
            '["What are the prerequisites?", "How do I benchmark this?"]\n'
            "```"
        )

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _fake_chat_call_async)

    suggestions = await generate_suggestions(
        query="RAG pipelines",
        response_text="Here is a response",
        num_suggestions=2,
        llm_timeout_sec=1.0,
    )

    assert suggestions == [
        "What are the prerequisites?",
        "How do I benchmark this?",
    ]
