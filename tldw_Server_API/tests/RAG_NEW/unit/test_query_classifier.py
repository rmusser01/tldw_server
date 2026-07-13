import io

import pytest
from loguru import logger

from tldw_Server_API.app.core.RAG.rag_service.query_classifier import (
    _parse_classification_response,
    classify_and_reformulate,
    classify_query,
    reformulate_query,
)
from tldw_Server_API.tests.RAG_NEW.unit.test_generation_executor import (
    _RecordingCredentialRuntime,
    _install_explicit_chat_capture,
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


@pytest.mark.asyncio
async def test_classify_query_uses_explicit_runtime_credentials(monkeypatch):
    runtime = _RecordingCredentialRuntime()
    stage_metadata: dict[str, object] = {}
    captured = _install_explicit_chat_capture(
        monkeypatch,
        (
            '{"skip_search":false,"search_local_db":true,"search_web":false,'
            '"search_academic":false,"search_discussions":false,'
            '"standalone_query":"credential runtime research",'
            '"detected_intent":"factual","confidence":0.9,'
            '"reasoning":"requires retrieval"}'
        ),
    )

    result = await classify_query(
        "credential runtime research",
        llm_provider="anthropic",
        llm_model="claude-test",
        credential_runtime=runtime,
        stage_metadata=stage_metadata,
    )

    assert result.reasoning == "requires retrieval"
    assert runtime.resolved == ["anthropic"]
    assert runtime.marked == [runtime.handle]
    assert captured["kwargs"]["api_key"] == "runtime-only-key"
    assert captured["kwargs"]["app_config"] == runtime.handle.app_config
    assert captured["kwargs"]["credentials_resolved"] is True
    assert stage_metadata == {"verification_available": True}


@pytest.mark.asyncio
async def test_classify_query_runtime_failure_lowers_trust_without_detail():
    class FailingRuntime:
        async def resolve(self, _provider):
            raise RuntimeError("secret-key /private/credential-store.db")

    stage_metadata: dict[str, object] = {}
    result = await classify_query(
        "latest credential runtime research",
        llm_provider="anthropic",
        llm_model="claude-test",
        credential_runtime=FailingRuntime(),
        stage_metadata=stage_metadata,
    )

    assert result.confidence <= 0.5
    assert result.reasoning == "provider_unavailable"
    assert stage_metadata == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert "secret-key" not in str(result)
    assert "/private/" not in str(result)


@pytest.mark.asyncio
async def test_reformulate_query_runtime_failure_records_bounded_trust():
    class FailingRuntime:
        async def resolve(self, _provider):
            raise RuntimeError("secret-key /private/credential-store.db")

    stage_metadata: dict[str, object] = {}
    result = await reformulate_query(
        "what about it?",
        [{"role": "user", "content": "Explain credential runtimes."}],
        llm_provider="anthropic",
        llm_model="claude-test",
        credential_runtime=FailingRuntime(),
        stage_metadata=stage_metadata,
    )

    assert result == "what about it?"
    assert stage_metadata == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert "secret-key" not in str(stage_metadata)
    assert "/private/" not in str(stage_metadata)


@pytest.mark.asyncio
async def test_reformulate_query_runtime_success_records_verified(monkeypatch):
    runtime = _RecordingCredentialRuntime()
    _install_explicit_chat_capture(monkeypatch, "standalone credential runtime query")
    stage_metadata: dict[str, object] = {}

    result = await reformulate_query(
        "what about it?",
        [{"role": "user", "content": "Explain credential runtimes."}],
        llm_provider="anthropic",
        llm_model="claude-test",
        credential_runtime=runtime,
        stage_metadata=stage_metadata,
    )

    assert result == "standalone credential runtime query"
    assert stage_metadata == {"verification_available": True}


@pytest.mark.asyncio
async def test_combined_reformulation_does_not_erase_classification_unavailability(
    monkeypatch,
):
    from tldw_Server_API.app.core.Chat import chat_service

    runtime = _RecordingCredentialRuntime()
    responses = iter(
        [
            RuntimeError("secret-key /private/credential-store.db"),
            "standalone credential runtime question",
        ]
    )

    async def fake_chat_call(**_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    stage_metadata: dict[str, object] = {}

    result = await classify_and_reformulate(
        "what about it",
        [{"role": "user", "content": "Explain credential runtimes."}],
        llm_provider="anthropic",
        llm_model="claude-test",
        credential_runtime=runtime,
        stage_metadata=stage_metadata,
    )

    assert result.standalone_query == "standalone credential runtime question"
    assert result.confidence <= 0.5
    assert result.reasoning == "provider_unavailable"
    assert stage_metadata == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert "secret-key" not in str(result)
    assert "/private/" not in str(result)


@pytest.mark.asyncio
async def test_classify_query_without_runtime_keeps_legacy_call_shape(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service

    captured: dict[str, object] = {}

    async def fake_chat_call(**kwargs):
        captured.update(kwargs)
        return (
            '{"skip_search":false,"search_local_db":true,"search_web":false,'
            '"standalone_query":"legacy config","confidence":0.8}'
        )

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)

    await classify_query(
        "legacy config",
        llm_provider="openai",
        llm_model="gpt-test",
    )

    assert "api_key" not in captured
    assert "app_config" not in captured
    assert "credentials_resolved" not in captured
