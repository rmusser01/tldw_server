from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_service import execute_streaming_call


class _DummyStreamTracker:
    def add_heartbeat(self):
        return None

    def add_chunk(self):
        return None


class _DummyMetrics:
    def track_llm_call(self, *_args, **_kwargs):
        return None

    def track_provider_fallback_success(self, *_args, **_kwargs):
        return None

    def track_rate_limit(self, *_args, **_kwargs):
        return None

    def track_moderation_output(self, *_args, **_kwargs):
        return None

    def track_moderation_stream_block(self, *_args, **_kwargs):
        return None

    @asynccontextmanager
    async def track_streaming(self, *_args, **_kwargs):
        yield _DummyStreamTracker()


class _NoModeration:
    class _Policy:
        enabled = False
        output_enabled = False

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy()

    def evaluate_action(self, *_args, **_kwargs):
        return None

    def check_text(self, *_args, **_kwargs):
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return text


def _stream_with_text(text: str):
    def _stream():
        yield text

    return _stream()


async def _collect_sse_chunks(response: StreamingResponse) -> list[str]:
    chunks: list[str] = []
    agen = response.body_iterator
    try:
        async for chunk in agen:
            if isinstance(chunk, (bytes, bytearray)):
                chunks.append(chunk.decode("utf-8", errors="replace"))
            else:
                chunks.append(str(chunk))
    finally:
        with contextlib.suppress(Exception):
            await agen.aclose()
    return chunks


def _json_payload_chunks(chunks: list[str]) -> list[tuple[int, dict[str, Any]]]:
    payloads: list[tuple[int, dict[str, Any]]] = []
    for chunk_idx, chunk in enumerate(chunks):
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            raw_payload = line[6:].strip()
            if not raw_payload or raw_payload == "[DONE]":
                continue
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append((chunk_idx, payload))
    return payloads


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_emits_structured_result_before_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)

    async def save_message_fn(_db, _conv_id, _payload, use_transaction=True):
        return "m-structured-1"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=9, api_key_id=None, team_ids=None, org_ids=None),
    )

    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "return structured"}],
            "model": "gpt-4o-mini",
            "streaming": True,
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
        templated_llm_payload=[{"role": "user", "content": "return structured"}],
        should_persist=True,
        final_conversation_id="conv-structured-stream-1",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-structured-1",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: _stream_with_text('{"answer":"ok"}'),
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=lambda: _NoModeration(),
    )

    assert isinstance(response, StreamingResponse)
    events = await _collect_sse_chunks(response)

    done_indices = [idx for idx, chunk in enumerate(events) if "data: [DONE]" in chunk]
    assert done_indices, "missing DONE marker"
    done_idx = done_indices[-1]

    structured_idx, payload = next(
        (idx, payload)
        for idx, payload in _json_payload_chunks(events)
        if payload.get("validated_payload") == {"answer": "ok"}
    )
    assert structured_idx < done_idx
    assert payload["validated"] is True
    assert payload["validated_payload"] == {"answer": "ok"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_emits_structured_error_and_done_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)

    async def save_message_fn(_db, _conv_id, _payload, use_transaction=True):
        return "m-structured-2"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=10, api_key_id=None, team_ids=None, org_ids=None),
    )

    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "return structured"}],
            "model": "gpt-4o-mini",
            "streaming": True,
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
        templated_llm_payload=[{"role": "user", "content": "return structured"}],
        should_persist=True,
        final_conversation_id="conv-structured-stream-2",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-structured-2",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: _stream_with_text('{"answer":123}'),
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=lambda: _NoModeration(),
    )

    assert isinstance(response, StreamingResponse)
    events = await _collect_sse_chunks(response)

    assert any("data: [DONE]" in chunk for chunk in events)
    payload = next(
        payload
        for _idx, payload in _json_payload_chunks(events)
        if payload.get("code") == "structured_output_schema_error"
    )
    assert payload["code"] == "structured_output_schema_error"
    assert payload["message"] == "Model output did not match the requested JSON schema."
