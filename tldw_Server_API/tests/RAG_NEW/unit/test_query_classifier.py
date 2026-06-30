import io

import pytest
from loguru import logger

from tldw_Server_API.app.core.RAG.rag_service.query_classifier import (
    _parse_classification_response,
    classify_query,
    reformulate_query,
)


pytestmark = pytest.mark.unit


def test_parse_classification_response_parses_fenced_json_with_think_tags():
    raw = (
        "<think>reasoning</think>\n"
        "```json\n"
        "{"
        '"skip_search": false,'
        '"search_local_db": true,'
        '"search_web": true,'
        '"search_academic": false,'
        '"search_discussions": false,'
        '"standalone_query": "what is rag",'
        '"detected_intent": "definitional",'
        '"confidence": 0.9,'
        '"reasoning": "needs retrieval"'
        "}\n"
        "```"
    )

    parsed = _parse_classification_response(raw)
    assert parsed["search_web"] is True
    assert parsed["standalone_query"] == "what is rag"


def test_parse_classification_response_accepts_list_wrapped_object():
    raw = '[{"skip_search": true, "search_local_db": false, "search_web": false}]'
    parsed = _parse_classification_response(raw)
    assert parsed["skip_search"] is True
    assert parsed["search_local_db"] is False


@pytest.mark.asyncio
async def test_classify_query_fallback_log_sanitizes_llm_exception(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service

    async def fail_chat_call(**_kwargs):
        raise RuntimeError(
            "provider rejected sk-classify-secret at "
            "/tmp/private/user_databases/42/Media_DB_v2.db"
        )

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fail_chat_call)
    log_stream = io.StringIO()
    sink_id = logger.add(log_stream, level="WARNING", format="{message}")
    try:
        result = await classify_query("latest research news", llm_provider="openai")
    finally:
        logger.remove(sink_id)

    log_output = log_stream.getvalue()

    assert result.standalone_query == "latest research news"
    assert result.search_local_db is True
    assert result.search_web is True
    assert result.reasoning == "Heuristic classification (no LLM)"
    assert "LLM query classification failed" in log_output
    assert "RuntimeError" in log_output
    assert "sk-classify-secret" not in log_output
    assert "/tmp/private/user_databases/42/Media_DB_v2.db" not in log_output
    assert "provider rejected" not in log_output


@pytest.mark.asyncio
async def test_reformulate_query_fallback_log_sanitizes_llm_exception(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service

    async def fail_chat_call(**_kwargs):
        raise ValueError(
            "reformulation failed with token sk-reformulate-secret from "
            "/Users/example/private/prompts/history.txt"
        )

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fail_chat_call)
    log_stream = io.StringIO()
    sink_id = logger.add(log_stream, level="WARNING", format="{message}")
    try:
        result = await reformulate_query(
            "what about it?",
            [{"role": "user", "content": "Tell me about retrieval augmented generation."}],
        )
    finally:
        logger.remove(sink_id)

    log_output = log_stream.getvalue()

    assert result == "what about it?"
    assert "Query reformulation failed" in log_output
    assert "ValueError" in log_output
    assert "sk-reformulate-secret" not in log_output
    assert "/Users/example/private/prompts/history.txt" not in log_output
    assert "reformulation failed with token" not in log_output
