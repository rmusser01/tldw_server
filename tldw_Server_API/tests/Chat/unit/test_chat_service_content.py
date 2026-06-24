import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_service import execute_non_stream_call


class _DummyMetrics:
    def track_llm_call(self, *_args, **_kwargs):
        return None

    def track_provider_fallback_success(self, *_args, **_kwargs):
        return None

    def track_tokens(self, *_args, **_kwargs):
        return None

    def track_moderation_output(self, *_args, **_kwargs):
        return None


class _CapturingMetrics(_DummyMetrics):
    def __init__(self):
        self.tokens: dict[str, object] = {}

    def track_tokens(self, **kwargs):
        self.tokens.update(kwargs)


class _RedactingModeration:
    class _Policy:
        enabled = True
        output_enabled = True
        output_action = "redact"

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy()

    def evaluate_action_with_match(self, *_args, **_kwargs):
        return ("redact", None, None, None, None)

    def check_text(self, *_args, **_kwargs):
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return f"REDACTED:{text}"


class _NoModeration:
    class _Policy:
        enabled = False
        output_enabled = False
        output_action = "block"

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy()

    def evaluate_action_with_match(self, *_args, **_kwargs):
        return ("pass", None, None, None, None)

    def check_text(self, *_args, **_kwargs):
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return text


class _KeywordModeration:
    class _Policy:
        enabled = True
        output_enabled = True

        def __init__(self, action: str):
            self.output_action = action

    def __init__(self, *, keyword: str, action: str):
        self.keyword = keyword
        self.action = action

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy(self.action)

    def evaluate_action_with_match(self, text, *_args, **_kwargs):
        if self.keyword in str(text):
            return (self.action, str(text).replace(self.keyword, "[redacted]"), "keyword", "default", (0, len(self.keyword)))
        return ("pass", None, None, None, None)

    def check_text(self, text, *_args, **_kwargs):
        if self.keyword in str(text):
            return (True, "keyword")
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return str(text).replace(self.keyword, "[redacted]")


async def _run_non_stream_content_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_response: dict,
    moderation,
    should_persist: bool = False,
    response_format: dict | None = None,
    metrics: object | None = None,
    save_calls: list[dict[str, object]] | None = None,
) -> tuple[dict, list[dict[str, object]], dict[str, object]]:
    logged_usage: dict[str, object] = {}

    async def fake_log_llm_usage(**kwargs):
        logged_usage.update(kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    if save_calls is None:
        save_calls = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_calls.append(payload)
        return f"message-{len(save_calls)}"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )
    cleaned_args = {
        "api_endpoint": "openai",
        "api_key": "test-key",
        "messages_payload": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "streaming": False,
    }
    if response_format is not None:
        cleaned_args["response_format"] = response_format

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args=cleaned_args,
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics or _DummyMetrics(),
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=should_persist,
        final_conversation_id="conv-multi-choice",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: llm_response,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: moderation,
    )
    return response, save_calls, logged_usage


@pytest.mark.asyncio
async def test_execute_non_stream_call_redacts_list_content(monkeypatch):
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    def llm_call_func():
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "secret"},
                            "loose secret",
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAA"},
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    async def save_message_fn(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "streaming": False,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=_DummyMetrics(),
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=False,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _RedactingModeration(),
    )

    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, list)
    assert content[0]["text"].startswith("REDACTED:")
    assert content[1] == "REDACTED:loose secret"
    assert content[2]["type"] == "image_url"


@pytest.mark.asyncio
async def test_execute_non_stream_call_normalizes_gemini_usage_for_logging(monkeypatch):
    logged_usage: dict[str, object] = {}

    async def fake_log_llm_usage(**kwargs):
        logged_usage.update(kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    def llm_call_func():
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "promptTokenCount": 80,
                "candidatesTokenCount": 30,
                "totalTokenCount": 115,
                "cachedContentTokenCount": 50,
                "thoughtsTokenCount": 5,
            },
        }

    async def save_message_fn(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )
    metrics = _CapturingMetrics()

    await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "google",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gemini-2.5-pro",
            "streaming": False,
        },
        selected_provider="google",
        provider="google",
        model="gemini-2.5-pro",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=False,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _RedactingModeration(),
    )

    assert logged_usage["prompt_tokens"] == 80
    assert logged_usage["completion_tokens"] == 30
    assert logged_usage["total_tokens"] == 115
    assert logged_usage["usage_metadata"] == {
        "promptTokenCount": 80,
        "candidatesTokenCount": 30,
        "totalTokenCount": 115,
        "cachedContentTokenCount": 50,
        "thoughtsTokenCount": 5,
    }
    assert logged_usage["estimate_source"] == "provider_usage"
    assert logged_usage["choice_count"] == 1
    assert metrics.tokens["prompt_tokens"] == 80
    assert metrics.tokens["completion_tokens"] == 30


@pytest.mark.asyncio
async def test_execute_non_stream_call_rejects_invalid_structured_output_before_persist(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    save_calls: list[dict[str, object]] = []

    def llm_call_func():
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"answer":123}'},
                    "finish_reason": "stop",
                }
            ]
        }

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_calls.append(payload)
        return "message-1"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    with pytest.raises(HTTPException) as exc_info:
        await execute_non_stream_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "api_key": "test-key",
                "messages_payload": [{"role": "user", "content": "hi"}],
                "model": "gpt-4o-mini",
                "streaming": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    },
                },
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-4o-mini",
            request_json="{}",
            request=request,
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[{"role": "user", "content": "hi"}],
            should_persist=True,
            final_conversation_id="conv-structured-invalid",
            character_card_for_context={"name": "Test"},
            chat_db=None,
            save_message_fn=save_message_fn,
            audit_service=None,
            audit_context=None,
            client_id="client",
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=llm_call_func,
            refresh_provider_params=lambda *_args, **_kwargs: None,
            moderation_getter=lambda: _RedactingModeration(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "structured_output_schema_error",
        "message": "Model output did not match the requested JSON schema.",
    }
    assert save_calls == []


@pytest.mark.asyncio
async def test_execute_non_stream_call_redacts_all_returned_choices(monkeypatch):
    response, _save_calls, _logged_usage = await _run_non_stream_content_test(
        monkeypatch,
        llm_response={
            "choices": [
                {"message": {"role": "assistant", "content": "first secret"}, "finish_reason": "stop"},
                {"message": {"role": "assistant", "content": "second secret"}, "finish_reason": "stop"},
            ]
        },
        moderation=_RedactingModeration(),
    )

    assert response["choices"][0]["message"]["content"] == "REDACTED:first secret"
    assert response["choices"][1]["message"]["content"] == "REDACTED:second secret"


@pytest.mark.asyncio
async def test_execute_non_stream_call_blocks_when_later_choice_violates(monkeypatch):
    save_calls: list[dict[str, object]] = []

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_calls.append(payload)
        return "message-1"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    with pytest.raises(HTTPException) as exc_info:
        await execute_non_stream_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "api_key": "test-key",
                "messages_payload": [{"role": "user", "content": "hi"}],
                "model": "gpt-4o-mini",
                "streaming": False,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-4o-mini",
            request_json="{}",
            request=request,
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[{"role": "user", "content": "hi"}],
            should_persist=True,
            final_conversation_id="conv-block-later-choice",
            character_card_for_context={"name": "Test"},
            chat_db=None,
            save_message_fn=save_message_fn,
            audit_service=None,
            audit_context=None,
            client_id="client",
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=lambda: {
                "choices": [
                    {"message": {"role": "assistant", "content": "safe"}, "finish_reason": "stop"},
                    {"message": {"role": "assistant", "content": "unsafe-token"}, "finish_reason": "stop"},
                ]
            },
            refresh_provider_params=lambda *_args, **_kwargs: None,
            moderation_getter=lambda: _KeywordModeration(keyword="unsafe-token", action="block"),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Output violates moderation policy"
    assert save_calls == []


@pytest.mark.asyncio
async def test_execute_non_stream_call_validates_all_structured_choices_before_persist(monkeypatch):
    save_calls: list[dict[str, object]] = []
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        await _run_non_stream_content_test(
            monkeypatch,
            llm_response={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer":"ok"}'}, "finish_reason": "stop"},
                    {"message": {"role": "assistant", "content": '{"answer":123}'}, "finish_reason": "stop"},
                ]
            },
            moderation=_NoModeration(),
            should_persist=True,
            response_format=response_format,
            save_calls=save_calls,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "structured_output_schema_error",
        "message": "Model output did not match the requested JSON schema.",
    }
    assert save_calls == []


@pytest.mark.asyncio
async def test_execute_non_stream_call_missing_usage_estimates_all_returned_choices(monkeypatch):
    metrics = _CapturingMetrics()
    response, _save_calls, logged_usage = await _run_non_stream_content_test(
        monkeypatch,
        llm_response={
            "choices": [
                {"message": {"role": "assistant", "content": "abcd"}, "finish_reason": "stop"},
                {"message": {"role": "assistant", "content": "abcdefgh"}, "finish_reason": "stop"},
            ]
        },
        moderation=_NoModeration(),
        metrics=metrics,
    )

    assert len(response["choices"]) == 2
    assert logged_usage["completion_tokens"] == 3
    assert logged_usage["total_tokens"] == logged_usage["prompt_tokens"] + 3
    assert logged_usage["estimate_source"] == "missing_usage"
