from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_service import (
    execute_non_stream_call,
    execute_streaming_call,
)
from tldw_Server_API.app.core.Chat.prompt_cost_envelope import build_prompt_cost_envelope
from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import (
    PromptCostGuardrailConfig,
    evaluate_prompt_cost_guardrails,
)


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

    def track_tokens(self, **_kwargs):
        return None

    def track_moderation_output(self, *_args, **_kwargs):
        return None

    def track_moderation_stream_block(self, *_args, **_kwargs):
        return None

    @contextlib.asynccontextmanager
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


def _warning_codes(decision) -> set[str]:
    return {warning["code"] for warning in decision.to_response_metadata()["warnings"]}


def _chat_request() -> SimpleNamespace:
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )


async def _save_message_fn(*_args, **_kwargs):
    return None


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


def test_default_config_is_disabled_and_warn_only():
    envelope = build_prompt_cost_envelope(
        [{"role": "system", "content": "rules " * 1000}],
        world_book_text="world " * 1000,
    )

    decision = evaluate_prompt_cost_guardrails(envelope)

    assert PromptCostGuardrailConfig().enabled is False
    assert PromptCostGuardrailConfig().default_action == "warn"
    assert decision.action == "allow"
    assert decision.to_response_metadata()["warnings"] == []


def test_warn_only_thresholds_return_bounded_metadata_without_prompt_text():
    secret_text = "secret-" + ("x" * 600)
    envelope = build_prompt_cost_envelope(
        [{"role": "system", "content": secret_text}],
        world_book_text=secret_text,
    )

    decision = evaluate_prompt_cost_guardrails(
        envelope,
        config=PromptCostGuardrailConfig(
            enabled=True,
            warn_total_estimated_tokens=1,
            warn_static_segment_tokens=1,
            warn_world_book_tokens=1,
        ),
    )
    metadata = decision.to_response_metadata()

    assert decision.action == "warn"
    assert {
        "large_prompt_estimate",
        "large_static_segment",
        "large_world_book_segment",
    }.issubset(_warning_codes(decision))
    assert secret_text not in repr(metadata)
    assert "secret-" not in repr(metadata)


def test_hard_block_threshold_overrides_warn_only_defaults():
    envelope = build_prompt_cost_envelope([{"role": "user", "content": "hello world"}])

    decision = evaluate_prompt_cost_guardrails(
        envelope,
        config=PromptCostGuardrailConfig(
            enabled=True,
            block_total_estimated_tokens=1,
        ),
    )

    assert decision.action == "block"
    assert "prompt_estimate_exceeds_hard_cap" in _warning_codes(decision)


def test_fingerprint_churn_warns_for_adjacent_turn_cache_risk():
    envelope = build_prompt_cost_envelope([{"role": "system", "content": "new stable rules"}])

    decision = evaluate_prompt_cost_guardrails(
        envelope,
        previous_fingerprints={
            "aggregate": "prompt-v1:sha256:old",
            "static": "prompt-v1:sha256:older-static",
        },
        config=PromptCostGuardrailConfig(enabled=True),
    )

    assert decision.action == "warn"
    assert {"prompt_fingerprint_churn", "static_fingerprint_churn"}.issubset(
        _warning_codes(decision)
    )


def test_output_choice_and_reasoning_risks_warn_from_request_parameters():
    envelope = build_prompt_cost_envelope([{"role": "user", "content": "hi"}])

    decision = evaluate_prompt_cost_guardrails(
        envelope,
        request_options={
            "max_completion_tokens": 4096,
            "n": 4,
            "reasoning_effort": "high",
        },
        config=PromptCostGuardrailConfig(
            enabled=True,
            warn_max_output_tokens=1024,
            warn_choice_count=1,
            warn_reasoning_efforts=("high", "xhigh"),
        ),
    )

    assert decision.action == "warn"
    assert {
        "high_output_token_cap",
        "high_choice_count",
        "reasoning_effort_risk",
    }.issubset(_warning_codes(decision))


@pytest.mark.asyncio
async def test_execute_non_stream_call_attaches_prompt_guardrail_warnings(monkeypatch):
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(
        chat_service,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=True, warn_total_estimated_tokens=1),
    )

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)

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
        request=_chat_request(),
        metrics=_DummyMetrics(),
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hello world"}],
        should_persist=False,
        final_conversation_id="conv-guardrail-warn",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=_save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _NoModeration(),
    )

    assert response["tldw_prompt_guardrails"]["action"] == "warn"
    assert "large_prompt_estimate" in {
        warning["code"] for warning in response["tldw_prompt_guardrails"]["warnings"]
    }


@pytest.mark.asyncio
async def test_execute_non_stream_call_blocks_before_provider_dispatch(monkeypatch):
    called = False
    monkeypatch.setattr(
        chat_service,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=True, block_total_estimated_tokens=1),
    )

    def llm_call_func():
        nonlocal called
        called = True
        return {"choices": [{"message": {"role": "assistant", "content": "late"}}]}

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
            request=_chat_request(),
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[{"role": "user", "content": "hello world"}],
            should_persist=False,
            final_conversation_id="conv-guardrail-block",
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_save_message_fn,
            audit_service=None,
            audit_context=None,
            client_id="client",
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=llm_call_func,
            refresh_provider_params=lambda *_args, **_kwargs: None,
            moderation_getter=lambda: _NoModeration(),
        )

    assert called is False
    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["type"] == "prompt_cost_guardrail_block"


@pytest.mark.asyncio
async def test_execute_streaming_call_blocks_before_provider_dispatch(monkeypatch):
    called = False
    monkeypatch.setattr(
        chat_service,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=True, block_total_estimated_tokens=1),
    )

    def llm_call_func():
        nonlocal called
        called = True
        return iter(["late"])

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
        request=_chat_request(),
        metrics=_DummyMetrics(),
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hello world"}],
        should_persist=False,
        final_conversation_id="conv-guardrail-stream-block",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=_save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=lambda: _NoModeration(),
    )
    chunks = await _collect_sse_chunks(response)

    assert called is False
    assert "prompt cost guardrail" in "".join(chunks).lower()
