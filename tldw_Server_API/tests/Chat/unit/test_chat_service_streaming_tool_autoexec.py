from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_service import execute_streaming_call
from tldw_Server_API.app.core.Chat.tool_auto_exec import (
    ToolExecutionBatchResult,
    ToolExecutionRecord,
)


class _DummyStreamTracker:
    def add_heartbeat(self):
        return None

    def add_chunk(self):
        return None


class _DummyMetrics:
    def track_llm_call(self, *_args, **_kwargs):
        return None

    def track_run_first_rollout(self, *_args, **_kwargs):
        return None

    def track_run_first_first_tool(self, *_args, **_kwargs):
        return None

    def track_run_first_fallback_after_run(self, *_args, **_kwargs):
        return None

    def track_run_first_completion_proxy(self, *_args, **_kwargs):
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


def _tool_call_stream() -> Any:
    def _stream() -> Any:
        tool_delta = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "notes.search",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        yield f"data: {json.dumps(tool_delta)}\n\n"
        yield "data: [DONE]\n\n"

    return _stream()


def _run_then_notes_stream() -> Any:
    def _stream() -> Any:
        tool_delta = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "run",
                                    "arguments": "{\"command\":\"ls\"}",
                                },
                            },
                            {
                                "index": 1,
                                "id": "c2",
                                "type": "function",
                                "function": {
                                    "name": "notes.search",
                                    "arguments": "{\"query\":\"todo\"}",
                                },
                            },
                        ]
                    }
                }
            ]
        }
        yield f"data: {json.dumps(tool_delta)}\n\n"
        yield "data: [DONE]\n\n"

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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_streaming_autoexec_enabled_persists_tool_messages_and_emits_tool_results_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 4321)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)

    captured_kwargs: dict[str, Any] = {}

    async def fake_autoexec(**kwargs):
        captured_kwargs.update(kwargs)
        rec = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"ok": True},
            module="notes",
            content='{"ok":true}',
        )
        return ToolExecutionBatchResult(
            requested_calls=1,
            processed_calls=1,
            execution_attempts=1,
            executed_calls=1,
            truncated=False,
            results=[rec],
        )

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    save_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_payloads.append(payload)
        return f"m-{len(save_payloads)}"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=7, api_key_id=None, team_ids=None, org_ids=None),
    )

    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "streaming": True,
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
        final_conversation_id="conv-stream-1",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-1",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=_tool_call_stream,
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=lambda: _NoModeration(),
        assistant_parent_message_id="anchor-stream-1",
        continuation_metadata={
            "applied": True,
            "mode": "branch",
            "from_message_id": "anchor-stream-1",
        },
    )

    assert isinstance(response, StreamingResponse)
    chunks = await _collect_sse_chunks(response)

    assert captured_kwargs["max_tool_calls"] == 2
    assert captured_kwargs["timeout_ms"] == 4321
    assert captured_kwargs["allow_catalog"] == ["notes.*"]
    assert captured_kwargs["attach_idempotency"] is True
    assert len(save_payloads) == 2
    assert save_payloads[0]["role"] == "assistant"
    assert save_payloads[0]["parent_message_id"] == "anchor-stream-1"
    assert save_payloads[1]["role"] == "tool"
    assert save_payloads[1]["tool_call_id"] == "c1"

    tool_event_idx = next(i for i, msg in enumerate(chunks) if msg.startswith("event: tool_results"))
    finish_idx = next(i for i, msg in enumerate(chunks) if '"finish_reason": "stop"' in msg)
    end_idx = next(i for i, msg in enumerate(chunks) if msg.startswith("event: stream_end"))
    done_idx = next(i for i, msg in enumerate(chunks) if "data: [DONE]" in msg)
    assert tool_event_idx < finish_idx < end_idx < done_idx

    tool_data_line = next(line for line in chunks[tool_event_idx].splitlines() if line.startswith("data: "))
    payload = json.loads(tool_data_line[6:])
    assert payload["tool_results"][0]["tool_call_id"] == "c1"
    assert payload["tldw_conversation_id"] == "conv-stream-1"
    assert payload["tldw_message_id"] == "m-1"
    assert payload["tldw_continuation"]["mode"] == "branch"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_streaming_autoexec_disabled_does_not_emit_tool_results_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)

    called = {"autoexec": 0}

    async def fake_autoexec(**_kwargs):
        called["autoexec"] += 1
        return ToolExecutionBatchResult(0, 0, 0, 0, False, [])

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    save_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_payloads.append(payload)
        return f"m-{len(save_payloads)}"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=8, api_key_id=None, team_ids=None, org_ids=None),
    )

    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "streaming": True,
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
        final_conversation_id="conv-stream-2",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-2",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=_tool_call_stream,
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=lambda: _NoModeration(),
    )

    assert isinstance(response, StreamingResponse)
    chunks = await _collect_sse_chunks(response)

    assert called["autoexec"] == 0
    assert len(save_payloads) == 1
    assert save_payloads[0]["role"] == "assistant"
    assert not any(msg.startswith("event: tool_results") for msg in chunks)


class _RunFirstMetrics(_DummyMetrics):
    def __init__(self) -> None:
        self.rollout_calls: list[dict[str, Any]] = []
        self.first_tool_calls: list[dict[str, Any]] = []
        self.fallback_calls: list[dict[str, Any]] = []
        self.completion_calls: list[dict[str, Any]] = []

    def track_run_first_rollout(self, **kwargs):
        self.rollout_calls.append(kwargs)

    def track_run_first_first_tool(self, **kwargs):
        self.first_tool_calls.append(kwargs)

    def track_run_first_fallback_after_run(self, **kwargs):
        self.fallback_calls.append(kwargs)

    def track_run_first_completion_proxy(self, **kwargs):
        self.completion_calls.append(kwargs)

class _ProviderManagerStub:
    def __init__(self, fallback_provider: str) -> None:
        self.fallback_provider = fallback_provider
        self.failures: list[tuple[str, str]] = []
        self.successes: list[tuple[str, float]] = []

    def record_failure(self, provider: str, error: Exception) -> None:
        self.failures.append((provider, type(error).__name__))

    def record_success(self, provider: str, latency: float) -> None:
        self.successes.append((provider, latency))

    def get_available_provider(self, *, exclude: list[str] | None = None) -> str | None:
        if exclude and self.fallback_provider in exclude:
            return None
        return self.fallback_provider
@pytest.mark.asyncio
@pytest.mark.unit
async def test_streaming_autoexec_records_run_first_rollout_and_tool_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)
    monkeypatch.setattr(chat_service, "resolve_chat_run_first_rollout_mode", lambda raw_mode=None, default="off": "default_on")
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2b_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: ["openai:gpt-4o-mini"],
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes.*"])

    run_tool = {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Execute shell commands.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
    notes_tool = {
        "type": "function",
        "function": {
            "name": "notes.search",
            "description": "Search notes for relevant passages.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
    request_data = SimpleNamespace(
        model="gpt-4o-mini",
        stream=True,
        tools=[run_tool, notes_tool],
        tool_choice=None,
        temperature=0.2,
    )

    def _model_dump(*, exclude_none=True, exclude=None):
        payload = {
            "model": request_data.model,
            "stream": request_data.stream,
            "tools": request_data.tools,
            "temperature": request_data.temperature,
        }
        if exclude:
            payload = {k: v for k, v in payload.items() if k not in exclude}
        if exclude_none:
            payload = {k: v for k, v in payload.items() if v is not None}
        return payload

    request_data.model_dump = _model_dump  # type: ignore[attr-defined]

    cleaned_args = chat_service.build_call_params_from_request(
        request_data=request_data,
        target_api_provider="openai",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        final_system_message="Base system prompt.",
        app_config=None,
        grammar_record=None,
    )

    metrics = _RunFirstMetrics()
    save_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_payloads.append(payload)
        return f"m-{len(save_payloads)}"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=9, api_key_id=None, team_ids=None, org_ids=None),
    )

    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            **cleaned_args,
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=True,
        final_conversation_id="conv-stream-run-first",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-3",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=_run_then_notes_stream,
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=lambda: _NoModeration(),
    )

    assert isinstance(response, StreamingResponse)
    await _collect_sse_chunks(response)

    assert metrics.rollout_calls[0]["presentation_variant"] == "chat_phase2b_v1"
    assert metrics.rollout_calls[0]["cohort"] == "default_on"
    assert metrics.rollout_calls[0]["streaming"] is True
    assert metrics.rollout_calls[0]["eligible"] is True
    assert metrics.first_tool_calls[0]["first_tool"] == "run"
    assert metrics.fallback_calls[0]["fallback_tool"] == "notes.search"
    assert metrics.completion_calls[0]["outcome"] == "success"
    assert save_payloads[0]["role"] == "assistant"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_streaming_provider_fallback_refreshes_run_first_metric_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "prepare_structured_response_request", lambda **_kwargs: None)
    monkeypatch.setattr(chat_service, "apply_structured_response_request", lambda **_kwargs: None)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", lambda **_kwargs: _run_then_notes_stream())

    metrics = _RunFirstMetrics()
    provider_manager = _ProviderManagerStub("anthropic")
    save_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_payloads.append(payload)
        return f"m-{len(save_payloads)}"

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
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "streaming": True,
            "_chat_run_first_presentation_variant": "chat_phase2b_v1",
            "_chat_run_first_cohort": "default_on",
            "_chat_run_first_eligible": True,
            "_chat_effective_tool_names": ["run", "notes.search"],
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=True,
        final_conversation_id="conv-stream-fallback",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-4",
        queue_execution_enabled=False,
        enable_provider_fallback=True,
        llm_call_func=lambda: (_ for _ in ()).throw(chat_service.ChatAPIError("primary failed")),
        refresh_provider_params=lambda fallback_provider: (
            {
                "api_endpoint": fallback_provider,
                "api_key": "fallback-key",
                "messages_payload": [{"role": "user", "content": "hi"}],
                "model": "claude-3-7-sonnet",
                "streaming": True,
                "_chat_run_first_presentation_variant": "chat_phase2b_v1",
                "_chat_run_first_cohort": "out_of_cohort",
                "_chat_run_first_eligible": False,
                "_chat_run_first_ineligible_reason": "provider_not_in_rollout_allowlist",
                "_chat_effective_tool_names": ["run", "notes.search"],
            },
            "claude-3-7-sonnet",
        ),
        moderation_getter=lambda: _NoModeration(),
    )

    assert isinstance(response, StreamingResponse)
    await _collect_sse_chunks(response)

    assert len(metrics.rollout_calls) == 2
    assert metrics.rollout_calls[-1]["provider"] == "anthropic"
    assert metrics.rollout_calls[-1]["model"] == "claude-3-7-sonnet"
    assert metrics.rollout_calls[-1]["cohort"] == "out_of_cohort"
    assert metrics.completion_calls[0]["provider"] == "anthropic"
    assert metrics.completion_calls[0]["model"] == "claude-3-7-sonnet"
    assert metrics.completion_calls[0]["cohort"] == "out_of_cohort"


@pytest.mark.unit
def test_emit_chat_run_first_rollout_metrics_logs_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = SimpleNamespace(track_run_first_rollout=Mock(side_effect=RuntimeError("metrics down")))
    warning = Mock()
    monkeypatch.setattr(chat_service, "logger", SimpleNamespace(warning=warning))

    context = chat_service._emit_chat_run_first_rollout_metrics(
        metrics,
        cleaned_args={
            "_chat_run_first_presentation_variant": "chat_phase2b_v1",
            "_chat_run_first_cohort": "default_on",
            "_chat_run_first_eligible": True,
        },
        provider="openai",
        model="gpt-4o-mini",
        streaming=False,
    )

    assert context is not None
    assert context["cohort"] == "default_on"
    warning.assert_called_once()


@pytest.mark.unit
def test_emit_chat_run_first_rollout_metrics_propagates_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = SimpleNamespace(track_run_first_rollout=Mock(side_effect=AssertionError("unexpected")))
    warning = Mock()
    monkeypatch.setattr(chat_service, "logger", SimpleNamespace(warning=warning))

    with pytest.raises(AssertionError, match="unexpected"):
        chat_service._emit_chat_run_first_rollout_metrics(
            metrics,
            cleaned_args={
                "_chat_run_first_presentation_variant": "chat_phase2b_v1",
                "_chat_run_first_cohort": "default_on",
                "_chat_run_first_eligible": True,
            },
            provider="openai",
            model="gpt-4o-mini",
            streaming=False,
        )

    warning.assert_not_called()
