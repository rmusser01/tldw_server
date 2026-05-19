from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_service import execute_non_stream_call
from tldw_Server_API.app.core.Chat.tool_auto_exec import (
    ToolExecutionBatchResult,
    ToolExecutionRecord,
)
from tldw_Server_API.app.core.LLM_Calls.structured_generation import (
    StructuredGenerationParseError,
)
from tldw_Server_API.tests.run_first_constants import PHASE2C_RUN_FIRST_COHORT


class _DummyMetrics:
    def track_llm_call(self, *_args, **_kwargs):
        return None

    def track_provider_fallback_success(self, *_args, **_kwargs):
        return None

    def track_tokens(self, *_args, **_kwargs):
        return None

    def track_moderation_output(self, *_args, **_kwargs):
        return None


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


def _build_llm_response_with_tool_calls() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Calling tool",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "notes.search", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


async def _run_execute_non_stream_call(
    *,
    llm_call_func,
    save_message_fn,
    cleaned_args_overrides: dict[str, Any] | None = None,
    metrics: Any | None = None,
    provider_manager: Any | None = None,
    enable_provider_fallback: bool = False,
    refresh_provider_params=None,
) -> dict[str, Any]:
    cleaned_args = {
        "api_endpoint": "openai",
        "api_key": "test-key",
        "messages_payload": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "streaming": False,
    }
    if cleaned_args_overrides:
        cleaned_args.update(cleaned_args_overrides)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=11, api_key_id=None),
    )
    return await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args=cleaned_args,
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics or _DummyMetrics(),
        provider_manager=provider_manager,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=True,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-1",
        queue_execution_enabled=False,
        enable_provider_fallback=enable_provider_fallback,
        llm_call_func=llm_call_func,
        refresh_provider_params=refresh_provider_params or (lambda *_args, **_kwargs: None),
        moderation_getter=lambda: _NoModeration(),
    )


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


class _StrictRunFirstMetrics:
    def __init__(self) -> None:
        self.first_tool_calls: list[dict[str, Any]] = []
        self.fallback_calls: list[dict[str, Any]] = []
        self.completion_calls: list[dict[str, Any]] = []

    def track_run_first_first_tool(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        first_tool: str,
    ) -> None:
        self.first_tool_calls.append(
            {
                "presentation_variant": presentation_variant,
                "cohort": cohort,
                "provider": provider,
                "model": model,
                "streaming": streaming,
                "eligible": eligible,
                "first_tool": first_tool,
            }
        )

    def track_run_first_fallback_after_run(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        fallback_tool: str,
    ) -> None:
        self.fallback_calls.append(
            {
                "presentation_variant": presentation_variant,
                "cohort": cohort,
                "provider": provider,
                "model": model,
                "streaming": streaming,
                "eligible": eligible,
                "fallback_tool": fallback_tool,
            }
        )

    def track_run_first_completion_proxy(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        outcome: str,
    ) -> None:
        self.completion_calls.append(
            {
                "presentation_variant": presentation_variant,
                "cohort": cohort,
                "provider": provider,
                "model": model,
                "streaming": streaming,
                "eligible": eligible,
                "outcome": outcome,
            }
        )


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
async def test_non_stream_autoexec_disabled_keeps_existing_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
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

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
    )

    assert called["autoexec"] == 0
    assert len(saved_payloads) == 1
    assert saved_payloads[0]["role"] == "assistant"
    assert "tldw_tool_results" not in response


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_loop_mode_disables_legacy_autoexec(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)

    called = {"autoexec": 0}

    async def fake_autoexec(**_kwargs):
        called["autoexec"] += 1
        return ToolExecutionBatchResult(0, 0, 0, 0, False, [])

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
        cleaned_args_overrides={"chat_loop_mode": "enabled"},
    )

    assert called["autoexec"] == 0
    assert len(saved_payloads) == 1
    assert saved_payloads[0]["role"] == "assistant"
    assert "tldw_tool_results" not in response


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_first_presented_tools_drive_autoexec_allow_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3210)
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: False)
    monkeypatch.setattr(chat_service, "resolve_chat_run_first_rollout_mode", lambda raw_mode=None, default="off": "gated")
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2a_v1",
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
        stream=False,
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

    assert [tool["function"]["name"] for tool in cleaned_args["tools"]] == ["run", "notes.search"]
    assert cleaned_args["_chat_effective_tool_names"] == ["run", "notes.search"]
    assert "run(command)" in cleaned_args["system_message"]
    assert cleaned_args.get("tool_choice") is None

    captured = {"allow_catalog": None}

    async def fake_autoexec(**kwargs):
        captured["allow_catalog"] = kwargs.get("allow_catalog")
        rec = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"echo": {"q": "hello"}},
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

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
        cleaned_args_overrides=cleaned_args,
    )

    assert captured["allow_catalog"] == ["run", "notes.search"]
    assert response["tldw_tool_results"][0]["tool_call_id"] == "c1"
    assert [payload["role"] for payload in saved_payloads] == ["assistant", "tool"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_build_call_params_marks_run_first_ineligible_when_provider_not_in_rollout_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_rollout_mode",
        lambda raw_mode=None, default="off": "default_on",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2b_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: ["anthropic:claude-3-7-sonnet"],
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
        stream=False,
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

    assert cleaned_args["_chat_run_first_eligible"] is False
    assert cleaned_args["_chat_run_first_ineligible_reason"] == "provider_not_in_rollout_allowlist"
    assert cleaned_args["_chat_run_first_cohort"] == "out_of_cohort"
    assert "run(command)" not in cleaned_args["system_message"]
    assert cleaned_args["_chat_effective_tool_names"] == ["run", "notes.search"]
    assert [tool["function"]["name"] for tool in cleaned_args["tools"]] == ["run", "notes.search"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_build_call_params_marks_default_on_cohort_when_provider_is_in_rollout_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_rollout_mode",
        lambda raw_mode=None, default="off": "default_on",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2b_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: PHASE2C_RUN_FIRST_COHORT,
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes.*"])

    request_data = SimpleNamespace(
        model="gpt-4o",
        stream=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Execute shell commands.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notes.search",
                    "description": "Search notes for relevant passages.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
        ],
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

    assert cleaned_args["_chat_run_first_eligible"] is True
    assert cleaned_args["_chat_run_first_cohort"] == "default_on"
    assert "run(command)" in cleaned_args["system_message"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_build_call_params_marks_google_gemini_flash_default_on_when_in_rollout_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_rollout_mode",
        lambda raw_mode=None, default="off": "default_on",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2b_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: PHASE2C_RUN_FIRST_COHORT,
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes.*"])

    request_data = SimpleNamespace(
        model="gemini-2.5-flash",
        stream=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Execute shell commands.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notes.search",
                    "description": "Search notes for relevant passages.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
        ],
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
        target_api_provider="google",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        final_system_message="Base system prompt.",
        app_config=None,
        grammar_record=None,
    )

    assert cleaned_args["_chat_run_first_eligible"] is True
    assert cleaned_args["_chat_run_first_cohort"] == "default_on"
    assert cleaned_args["_chat_effective_tool_names"] == ["run", "notes.search"]
    assert "run(command)" in cleaned_args["system_message"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_build_call_params_removes_pinned_tool_choice_when_tool_is_filtered_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_service, "resolve_chat_run_first_rollout_mode", lambda raw_mode=None, default="off": "gated")
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2a_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: ["openai:gpt-4o-mini"],
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run"])

    request_data = SimpleNamespace(
        model="gpt-4o-mini",
        stream=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Execute shell commands.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notes.search",
                    "description": "Search notes for relevant passages.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
        ],
        tool_choice={"type": "function", "function": {"name": "notes.search"}},
        temperature=0.2,
    )

    def _model_dump(*, exclude_none=True, exclude=None):
        payload = {
            "model": request_data.model,
            "stream": request_data.stream,
            "tools": request_data.tools,
            "tool_choice": request_data.tool_choice,
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

    assert [tool["function"]["name"] for tool in cleaned_args["tools"]] == ["run"]
    assert cleaned_args["_chat_effective_tool_names"] == ["run"]
    assert "tool_choice" not in cleaned_args


@pytest.mark.unit
def test_build_call_params_uses_resolved_model_for_default_on_run_first_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_rollout_mode",
        lambda raw_mode=None, default="off": "default_on",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2b_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: PHASE2C_RUN_FIRST_COHORT,
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes.*"])

    request_data = SimpleNamespace(
        model=None,
        stream=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Execute shell commands.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notes.search",
                    "description": "Search notes for relevant passages.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
        ],
        tool_choice=None,
        temperature=0.2,
    )

    def _model_dump(*, exclude_none=True, exclude=None):
        payload = {
            "model": request_data.model,
            "stream": request_data.stream,
            "tools": request_data.tools,
            "tool_choice": request_data.tool_choice,
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
        resolved_model="gpt-4o-mini",
    )

    assert cleaned_args["model"] == "gpt-4o-mini"
    assert cleaned_args["_chat_run_first_eligible"] is True
    assert cleaned_args["_chat_run_first_cohort"] == "default_on"
    assert [tool["function"]["name"] for tool in cleaned_args["tools"]] == ["run", "notes.search"]
    assert "run(command)" in cleaned_args["system_message"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_build_call_params_tracks_all_gemini_native_tool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_service, "resolve_chat_run_first_rollout_mode", lambda raw_mode=None, default="off": "gated")
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2a_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: ["openai:gpt-4o-mini"],
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes.*"])

    gemini_tools = {
        "function_declarations": [
            {
                "name": "notes.search",
                "description": "Search notes for relevant passages.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "name": "run",
                "description": "Execute shell commands.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        ]
    }

    request_data = SimpleNamespace(
        model="gpt-4o-mini",
        stream=False,
        tools=[gemini_tools],
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

    assert cleaned_args["_chat_effective_tool_names"] == ["run", "notes.search"]
    assert [decl["name"] for decl in cleaned_args["tools"][0]["function_declarations"]] == [
        "run",
        "notes.search",
    ]
    assert "run(command)" in cleaned_args["system_message"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_autoexec_enabled_persists_tool_messages_and_response_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3210)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)

    captured_kwargs: dict[str, Any] = {}

    async def fake_autoexec(**kwargs):
        captured_kwargs.update(kwargs)
        rec = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"echo": {"q": "hello"}},
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

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
    )

    assert captured_kwargs["max_tool_calls"] == 2
    assert captured_kwargs["timeout_ms"] == 3210
    assert captured_kwargs["allow_catalog"] == ["notes.*"]
    assert captured_kwargs["attach_idempotency"] is True
    assert len(saved_payloads) == 2
    assert saved_payloads[0]["role"] == "assistant"
    assert saved_payloads[1]["role"] == "tool"
    assert saved_payloads[1]["tool_call_id"] == "c1"
    assert response["tldw_tool_results"][0]["ok"] is True
    assert response["tldw_tool_results"][0]["tool_call_id"] == "c1"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_autoexec_enabled_handles_mixed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 3)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 5000)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: False)

    async def fake_autoexec(**_kwargs):
        rec_ok = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"ok": 1},
            module="notes",
            content='{"ok":true}',
        )
        rec_fail = ToolExecutionRecord(
            tool_call_id="c2",
            tool_name="notes.forbidden",
            ok=False,
            error="Permission denied",
            content='{"ok":false}',
        )
        return ToolExecutionBatchResult(
            requested_calls=2,
            processed_calls=2,
            execution_attempts=2,
            executed_calls=1,
            truncated=False,
            results=[rec_ok, rec_fail],
        )

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
    )

    assert len(saved_payloads) == 3
    assert [p["role"] for p in saved_payloads] == ["assistant", "tool", "tool"]
    assert len(response["tldw_tool_results"]) == 2
    assert response["tldw_tool_results"][0]["ok"] is True
    assert response["tldw_tool_results"][1]["ok"] is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_autoexec_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 3)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 5000)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: False)

    async def fake_autoexec(**_kwargs):
        raise RuntimeError("autoexec failed")

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
    )

    assert len(saved_payloads) == 1
    assert saved_payloads[0]["role"] == "assistant"
    assert "tldw_tool_results" not in response


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_auto_continue_runs_once_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)

    autoexec_called = {"count": 0}

    async def fake_autoexec(**_kwargs):
        autoexec_called["count"] += 1
        rec = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"ok": 1},
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

    continuation_calls: list[dict[str, Any]] = []

    async def fake_followup_call(**kwargs):
        continuation_calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer from continuation",
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "notes.other", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_followup_call)

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
    )

    assert autoexec_called["count"] == 1
    assert len(continuation_calls) == 1
    continuation_messages = continuation_calls[0]["messages_payload"]
    assert continuation_messages[-2]["role"] == "assistant"
    assert continuation_messages[-2]["tool_calls"][0]["id"] == "c1"
    assert continuation_messages[-1]["role"] == "tool"
    assert continuation_messages[-1]["tool_call_id"] == "c1"

    assert [p["role"] for p in saved_payloads] == ["assistant", "tool", "assistant"]
    assert response["choices"][0]["message"]["content"] == "Final answer from continuation"
    assert response["tldw_tool_results"][0]["tool_call_id"] == "c1"
    assert response["tldw_tool_auto_continue"] == {"attempted": True, "succeeded": True}


@pytest.mark.unit
def test_emit_chat_run_first_tool_path_metrics_omits_ineligible_reason_for_strict_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _StrictRunFirstMetrics()
    warning = []
    monkeypatch.setattr(chat_service, "logger", SimpleNamespace(warning=lambda *args, **kwargs: warning.append((args, kwargs))))

    chat_service._emit_chat_run_first_tool_path_metrics(
        metrics,
        context={
            "presentation_variant": "chat_phase2b_v1",
            "cohort": "out_of_cohort",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "streaming": False,
            "eligible": False,
            "ineligible_reason": "provider_not_in_rollout_allowlist",
        },
        tool_calls=[
            {"function": {"name": "run"}},
            {"function": {"name": "notes.search"}},
        ],
    )

    assert metrics.first_tool_calls[0]["first_tool"] == "run"
    assert metrics.fallback_calls[0]["fallback_tool"] == "notes.search"
    assert warning == []


@pytest.mark.unit
def test_emit_chat_run_first_completion_metric_omits_ineligible_reason_for_strict_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _StrictRunFirstMetrics()
    warning = []
    monkeypatch.setattr(chat_service, "logger", SimpleNamespace(warning=lambda *args, **kwargs: warning.append((args, kwargs))))

    chat_service._emit_chat_run_first_completion_metric(
        metrics,
        context={
            "presentation_variant": "chat_phase2b_v1",
            "cohort": "out_of_cohort",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "streaming": False,
            "eligible": False,
            "ineligible_reason": "provider_not_in_rollout_allowlist",
        },
        outcome="error",
    )

    assert metrics.completion_calls[0]["outcome"] == "error"
    assert warning == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_auto_continue_preserves_first_turn_run_first_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)

    async def fake_autoexec(**_kwargs):
        rec = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"ok": 1},
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

    async def fake_followup_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer from continuation",
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "notes.other", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_followup_call)

    metrics = _RunFirstMetrics()
    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=lambda: {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Calling tool",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{\"command\":\"ls\"}"},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "notes.search", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
        save_message_fn=save_message_fn,
        cleaned_args_overrides={
            "_chat_run_first_presentation_variant": "chat_phase2b_v1",
            "_chat_run_first_cohort": "default_on",
            "_chat_run_first_eligible": True,
            "_chat_effective_tool_names": ["run", "notes.search"],
        },
        metrics=metrics,
    )

    assert response["tldw_tool_auto_continue"] == {"attempted": True, "succeeded": True}
    assert metrics.first_tool_calls[0]["first_tool"] == "run"
    assert metrics.fallback_calls[0]["fallback_tool"] == "notes.search"
    assert metrics.completion_calls[0]["outcome"] == "success"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_completion_metric_records_error_when_structured_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "prepare_structured_response_request", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(chat_service, "apply_structured_response_request", lambda **_kwargs: None)
    monkeypatch.setattr(
        chat_service,
        "validate_structured_response",
        lambda **_kwargs: (_ for _ in ()).throw(StructuredGenerationParseError("parse failed")),
    )

    metrics = _RunFirstMetrics()
    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    with pytest.raises(HTTPException, match="structured_output_parse_error"):
        await _run_execute_non_stream_call(
            llm_call_func=lambda: {
                "choices": [{"message": {"role": "assistant", "content": "{\"bad\": true}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
            save_message_fn=save_message_fn,
            cleaned_args_overrides={
                "_chat_run_first_presentation_variant": "chat_phase2b_v1",
                "_chat_run_first_cohort": "default_on",
                "_chat_run_first_eligible": True,
                "response_format": {"type": "json_schema"},
                "_structured_requested_response_format": {"type": "json_schema"},
            },
            metrics=metrics,
        )

    assert metrics.completion_calls[0]["outcome"] == "error"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_provider_fallback_refreshes_run_first_metric_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    async def fake_fallback_call(**_kwargs):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Fallback response"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_fallback_call)
    monkeypatch.setattr(chat_service, "prepare_structured_response_request", lambda **_kwargs: None)
    monkeypatch.setattr(chat_service, "apply_structured_response_request", lambda **_kwargs: None)

    metrics = _RunFirstMetrics()
    provider_manager = _ProviderManagerStub("anthropic")
    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=lambda: (_ for _ in ()).throw(chat_service.ChatAPIError("primary failed")),
        save_message_fn=save_message_fn,
        cleaned_args_overrides={
            "_chat_run_first_presentation_variant": "chat_phase2b_v1",
            "_chat_run_first_cohort": "default_on",
            "_chat_run_first_eligible": True,
        },
        metrics=metrics,
        provider_manager=provider_manager,
        enable_provider_fallback=True,
        refresh_provider_params=lambda fallback_provider: (
            {
                "api_endpoint": fallback_provider,
                "api_key": "fallback-key",
                "messages_payload": [{"role": "user", "content": "hi"}],
                "model": "claude-3-7-sonnet",
                "streaming": False,
                "_chat_run_first_presentation_variant": "chat_phase2b_v1",
                "_chat_run_first_cohort": "out_of_cohort",
                "_chat_run_first_eligible": False,
                "_chat_run_first_ineligible_reason": "provider_not_in_rollout_allowlist",
            },
            "claude-3-7-sonnet",
        ),
    )

    assert response["choices"][0]["message"]["content"] == "Fallback response"
    assert len(metrics.rollout_calls) == 2
    assert metrics.rollout_calls[-1]["provider"] == "anthropic"
    assert metrics.rollout_calls[-1]["model"] == "claude-3-7-sonnet"
    assert metrics.rollout_calls[-1]["cohort"] == "out_of_cohort"
    assert metrics.completion_calls[0]["provider"] == "anthropic"
    assert metrics.completion_calls[0]["model"] == "claude-3-7-sonnet"
    assert metrics.completion_calls[0]["cohort"] == "out_of_cohort"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_auto_continue_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)

    async def fake_autoexec(**_kwargs):
        rec = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"ok": 1},
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

    async def fake_followup_call(**_kwargs):
        raise RuntimeError("continuation failed")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_followup_call)

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
    )

    assert [p["role"] for p in saved_payloads] == ["assistant", "tool"]
    assert response["choices"][0]["message"]["content"] == "Calling tool"
    assert response["tldw_tool_auto_continue"] == {"attempted": True, "succeeded": False}
