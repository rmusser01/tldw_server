from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_service import execute_non_stream_call
from tldw_Server_API.app.core.Chat.tool_auto_exec import (
    ToolExecutionBatchResult,
    ToolExecutionRecord,
)
from tldw_Server_API.app.core.Chat.tool_execution_service import request_declares_local_tool_use
from tldw_Server_API.app.core.LLM_Calls.structured_generation import (
    StructuredGenerationParseError,
)
from tldw_Server_API.tests.provider_credential_test_helpers import (
    resolved_request_fields_async,
)
from tldw_Server_API.tests.run_first_constants import PHASE2C_RUN_FIRST_COHORT

_REGISTRY_OPENAI_BASE_URL = "https://registry-openai.test/v1"


def _registry_openai_app_config() -> dict[str, dict[str, str]]:
    """Return the authoritative adapter URL snapshot for registry tests."""

    return {"openai_api": {"api_base_url": _REGISTRY_OPENAI_BASE_URL}}


class _DummyMetrics:
    def __init__(self) -> None:
        self.llm_calls: list[tuple[str, str, bool, str | None]] = []
        self.fallback_successes: list[dict[str, Any]] = []
        self.token_calls: list[dict[str, Any]] = []
        self.completion_calls: list[dict[str, Any]] = []

    def track_llm_call(
        self,
        provider: str,
        model: str,
        _latency: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        self.llm_calls.append((provider, model, success, error_type))

    def track_provider_fallback_success(self, **kwargs: Any) -> None:
        self.fallback_successes.append(kwargs)

    def track_tokens(self, **kwargs: Any) -> None:
        self.token_calls.append(kwargs)

    def track_run_first_completion_proxy(self, **kwargs: Any) -> None:
        self.completion_calls.append(kwargs)

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


def _adapter_nonstream_call(
    resolved_fields: dict[str, Any],
    *,
    messages_payload: list[dict[str, Any]] | None = None,
) -> Any:
    """Cross the production registry and synchronous adapter boundary."""

    return chat_service.perform_chat_api_call(
        api_endpoint="openai",
        messages_payload=messages_payload or [],
        model="gpt-4o-mini",
        streaming=False,
        **resolved_fields,
    )


def _install_real_openai_adapter_transport(
    monkeypatch: pytest.MonkeyPatch,
    responder,
    *,
    on_client_exit=None,
) -> None:
    """Install a deterministic transport below the real OpenAI adapter."""

    from tldw_Server_API.app.core.LLM_Calls import adapter_registry
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    monkeypatch.setattr(adapter_registry, "_registry", None)
    registry = adapter_registry.get_registry()
    assert isinstance(registry, adapter_registry.ChatProviderRegistry)
    assert registry.resolve_provider_name(" OAI ") == "openai"
    assert "openai" not in registry._base._adapter_cache

    class Response:
        def __init__(self, result: Any) -> None:
            self._result = result

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self._result

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args: Any) -> None:
            if on_client_exit is not None:
                on_client_exit()

        def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> Response:
            return Response(responder(url=url, headers=headers, payload=json))

    def client_factory(*, timeout: float) -> Client:
        assert timeout > 0
        assert adapter_registry.get_registry() is registry
        canonical_adapter = registry.get_adapter("OPENAI")
        assert isinstance(canonical_adapter, openai_adapter.OpenAIAdapter)
        assert canonical_adapter is not None
        assert registry.get_adapter("oai") is canonical_adapter
        return Client()

    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    monkeypatch.setattr(openai_adapter, "http_client_factory", client_factory)


def _install_owned_worker_drain_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Event:
    """Expose entry into cancellation draining without timing-based assertions."""

    drain_entered = asyncio.Event()
    original = bounded_daemon_module._drain_owned_task

    async def probe(task):
        drain_entered.set()
        return await original(task)

    monkeypatch.setattr(bounded_daemon_module, "_drain_owned_task", probe)
    return drain_entered
class _KeywordModeration:
    class _Policy:
        enabled = True
        output_enabled = True
        output_action = "redact"

    def __init__(self, *, keyword: str) -> None:
        self.keyword = keyword

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy()

    def evaluate_action_with_match(self, text, *_args, **_kwargs):
        if self.keyword in str(text):
            return (
                "redact",
                str(text).replace(self.keyword, "[redacted]"),
                "keyword",
                "default",
                (0, len(self.keyword)),
            )
        return ("pass", None, None, None, None)

    def check_text(self, text, *_args, **_kwargs):
        if self.keyword in str(text):
            return (True, "keyword")
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return str(text).replace(self.keyword, "[redacted]")


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


def _late_continuation_response(outcome: str, sentinel: str) -> Any:
    """Build representative late continuation results for usage accounting."""

    if outcome == "valid_text":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "late continuation",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    if outcome == "valid_raw_text":
        return "late raw continuation"
    if outcome == "valid_list_text":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "late list continuation"}
                        ],
                    }
                }
            ]
        }
    if outcome == "valid_image":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,dGVzdA=="
                                },
                            }
                        ],
                    }
                }
            ]
        }
    if outcome == "valid_tool_calls":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {
                                    "name": "notes.lookup",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
    if outcome == "valid_function_call":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "notes.lookup",
                            "arguments": "{}",
                        },
                    }
                }
            ]
        }
    if outcome == "valid_noncanonical_error_json":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "error": {
                                    "code": "fictional_story_error",
                                    "message": "ordinary assistant-authored content",
                                }
                            },
                            separators=(",", ":"),
                        ),
                    }
                }
            ]
        }
    if outcome == "valid_refusal":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": "I cannot continue that request.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    if outcome == "valid_content_filter":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": None,
                    },
                    "finish_reason": "content_filter",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 0,
                "total_tokens": 3,
            },
        }
    if outcome == "raw_done":
        return "[DONE]"
    if outcome == "sse_done":
        return "data: [DONE]\n\n"
    if outcome == "sse_success":
        return 'data: {"choices":[{"message":{"content":"framed"}}]}\n\n'
    if outcome == "empty":
        return {"choices": []}
    if outcome == "error":
        return {"error": {"message": sentinel}}
    if outcome == "error_prefix":
        return f"Error: {sentinel}"
    if outcome == "canonical_raw_code":
        return "provider_unavailable"
    serialized_error = json.dumps(
        {
            "error": {
                "code": "provider_unavailable",
                "message": sentinel,
            }
        },
        separators=(",", ":"),
    )
    if outcome == "sse_error_envelope":
        return f"data: {serialized_error}\n\n"
    if outcome == "serialized_error_envelope":
        return serialized_error
    if outcome == "malformed_tool_calls":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"function": {"arguments": "{}"}}],
                    }
                }
            ]
        }
    if outcome == "malformed_function_call":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"arguments": "{}"},
                    }
                }
            ]
        }
    if outcome == "mixed_error_and_text":
        return {
            "error": {"code": "provider_unavailable", "message": sentinel},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "must not replace the first turn",
                    }
                }
            ],
        }
    if outcome == "nested_error_prefix":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Error: assistant-authored continuation",
                    }
                }
            ]
        }
    if outcome == "nested_canonical_error":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "provider_unavailable",
                    }
                }
            ]
        }
    if outcome == "nested_structured_error_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "error": {
                            "code": "provider_unavailable",
                            "message": sentinel,
                        },
                        "content": "must not replace the first turn",
                    }
                }
            ]
        }
    if outcome == "list_part_error_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "must not replace the first turn"},
                            {
                                "type": "error",
                                "error": {
                                    "code": "provider_unavailable",
                                    "message": sentinel,
                                },
                            },
                        ],
                    }
                }
            ]
        }
    if outcome == "later_choice_error_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "must not replace the first turn",
                    }
                },
                {
                    "error": {
                        "code": "provider_unavailable",
                        "message": sentinel,
                    }
                },
            ]
        }
    raise AssertionError(f"Unsupported late continuation outcome: {outcome}")


async def _run_execute_non_stream_call(
    *,
    llm_call_func,
    save_message_fn,
    cleaned_args_overrides: dict[str, Any] | None = None,
    metrics: Any | None = None,
    provider_manager: Any | None = None,
    queue_execution_enabled: bool = False,
    enable_provider_fallback: bool = False,
    refresh_provider_params=None,
    on_success=None,
    conversation_id: str = "conv-123",
    moderation_getter=None,
) -> dict[str, Any]:
    cleaned_args = {
        "api_endpoint": "openai",
        "api_key": "test-key",
        "app_config": _registry_openai_app_config(),
        "credentials_resolved": True,
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
        final_conversation_id=conversation_id,
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id=conversation_id,
        queue_execution_enabled=queue_execution_enabled,
        enable_provider_fallback=enable_provider_fallback,
        llm_call_func=llm_call_func,
        refresh_provider_params=refresh_provider_params or (lambda *_args, **_kwargs: None),
        moderation_getter=moderation_getter or (lambda: _NoModeration()),
        on_success=on_success,
    )


class _RunFirstMetrics(_DummyMetrics):
    def __init__(self) -> None:
        super().__init__()
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
async def test_tool_autoexec_rejects_multi_choice_before_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_called = False

    def llm_call_func():
        nonlocal provider_called
        provider_called = True
        return _build_llm_response_with_tool_calls()

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)

    with pytest.raises(HTTPException) as exc_info:
        await _run_execute_non_stream_call(
            llm_call_func=llm_call_func,
            save_message_fn=save_message_fn,
            cleaned_args_overrides={
                "n": 2,
                "tool_choice": "auto",
                "tools": [{"type": "function", "function": {"name": "notes.search", "parameters": {}}}],
            },
        )

    assert provider_called is False
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "unsupported_multi_choice_tool_autoexec",
        "message": "Local tool auto-execution supports one assistant choice per request.",
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_autoexec_rejects_multi_choice_before_queue_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_called = False

    def llm_call_func():
        nonlocal provider_called
        provider_called = True
        return _build_llm_response_with_tool_calls()

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    class _FakeActiveQueue:
        is_running = True

        def __init__(self) -> None:
            self.enqueued = False

        async def enqueue(self, *_args, **_kwargs):
            self.enqueued = True
            return _build_llm_response_with_tool_calls()

    fake_queue = _FakeActiveQueue()

    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: fake_queue)

    with pytest.raises(HTTPException) as exc_info:
        await _run_execute_non_stream_call(
            llm_call_func=llm_call_func,
            save_message_fn=save_message_fn,
            queue_execution_enabled=True,
            cleaned_args_overrides={
                "n": 2,
                "tools": [{"type": "function", "function": {"name": "notes.search", "parameters": {}}}],
            },
        )

    assert fake_queue.enqueued is False
    assert provider_called is False
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "unsupported_multi_choice_tool_autoexec",
        "message": "Local tool auto-execution supports one assistant choice per request.",
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_autoexec_allows_multi_choice_when_autoexec_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)

    provider_called = False

    def llm_call_func():
        nonlocal provider_called
        provider_called = True
        return _build_llm_response_with_tool_calls()

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    response = await _run_execute_non_stream_call(
        llm_call_func=llm_call_func,
        save_message_fn=save_message_fn,
        cleaned_args_overrides={
            "n": 2,
            "tools": [{"type": "function", "function": {"name": "notes.search", "parameters": {}}}],
        },
    )

    assert provider_called is True
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "notes.search"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_autoexec_allows_multi_choice_without_request_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)

    provider_called = False

    def llm_call_func():
        nonlocal provider_called
        provider_called = True
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "plain response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    response = await _run_execute_non_stream_call(
        llm_call_func=llm_call_func,
        save_message_fn=save_message_fn,
        cleaned_args_overrides={"n": 2},
    )

    assert provider_called is True
    assert response["choices"][0]["message"]["content"] == "plain response"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_autoexec_allows_multi_choice_tool_choice_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)

    provider_called = False

    def llm_call_func():
        nonlocal provider_called
        provider_called = True
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "plain response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    response = await _run_execute_non_stream_call(
        llm_call_func=llm_call_func,
        save_message_fn=save_message_fn,
        cleaned_args_overrides={"n": 2, "tool_choice": "auto"},
    )

    assert provider_called is True
    assert response["choices"][0]["message"]["content"] == "plain response"


@pytest.mark.unit
def test_request_declares_local_tool_use_ignores_choice_without_definitions() -> None:
    assert request_declares_local_tool_use({"tool_choice": "auto"}) is False
    assert request_declares_local_tool_use({"tool_choice": "none"}) is False
    assert request_declares_local_tool_use({"tool_choice": "auto", "tools": []}) is False
    assert request_declares_local_tool_use({"tool_choice": "auto", "tools": [{"type": "function"}]}) is True
    assert request_declares_local_tool_use({"function_call": "auto"}) is False
    assert request_declares_local_tool_use({"function_call": "none"}) is False
    assert request_declares_local_tool_use({"function_call": "auto", "functions": []}) is False
    assert request_declares_local_tool_use({"function_call": "auto", "functions": [{"name": "notes_search"}]}) is True


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


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "expected_success"),
    [
        ("valid_text", True),
        ("valid_raw_text", True),
        ("valid_list_text", True),
        ("valid_image", True),
        ("valid_tool_calls", True),
        ("valid_function_call", True),
        ("valid_noncanonical_error_json", True),
        ("valid_refusal", True),
        ("valid_content_filter", True),
        ("mixed_error_and_text", False),
        ("nested_error_prefix", True),
        ("nested_canonical_error", True),
        ("nested_structured_error_and_text", False),
        ("list_part_error_and_text", False),
        ("later_choice_error_and_text", False),
        ("raw_done", False),
        ("sse_done", False),
        ("sse_success", False),
    ],
)
async def test_normal_tool_continuation_result_is_gated_before_retry_mark_and_replacement(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_success: bool,
) -> None:
    """A continuation is validated before usage retry or first-turn replacement."""

    sentinel = "normal-continuation-secret-/srv/provider"
    mark_attempts = 0
    successful_marks = 0
    saved_payloads: list[dict[str, Any]] = []
    followup_result = _late_continuation_response(outcome, sentinel)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        if any(message.get("role") == "tool" for message in payload["messages"]):
            return followup_result
        return _build_llm_response_with_tool_calls()

    _install_real_openai_adapter_transport(monkeypatch, responder)
    usage_log = AsyncMock(return_value=None)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)
    monkeypatch.setattr(chat_service, "should_force_normalize_string_responses", lambda: False)

    async def fake_autoexec(**_kwargs: Any) -> ToolExecutionBatchResult:
        record = ToolExecutionRecord(
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
            results=[record],
        )

    async def mark_used(_provider: str) -> None:
        nonlocal mark_attempts, successful_marks
        mark_attempts += 1
        if mark_attempts == 1:
            raise RuntimeError("transient accounting failure")
        successful_marks += 1

    async def save_message(
        _db: Any,
        _conversation_id: str,
        payload: dict[str, Any],
        *,
        use_transaction: bool,
    ) -> str:
        assert use_transaction is True
        saved_payloads.append(payload)
        return f"message-{len(saved_payloads)}"

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    resolved_fields = await resolved_request_fields_async(
        "openai",
        api_key="test-key",
        app_config=_registry_openai_app_config(),
        model="gpt-4o-mini",
    )
    response = await _run_execute_non_stream_call(
        llm_call_func=lambda: _adapter_nonstream_call(
            resolved_fields,
            messages_payload=[{"role": "user", "content": "hi"}],
        ),
        save_message_fn=save_message,
        cleaned_args_overrides=dict(resolved_fields),
        on_success=mark_used,
    )

    assert mark_attempts == 2
    assert successful_marks == 1
    if expected_success:
        if isinstance(followup_result, str):
            assert response == followup_result
            assert saved_payloads[-1]["content"] == followup_result
        else:
            assert response["choices"] == followup_result["choices"]
            assert response["tldw_tool_auto_continue"] == {
                "attempted": True,
                "succeeded": True,
            }
            followup_message = followup_result["choices"][0]["message"]
            if followup_message.get("content") is not None:
                assert saved_payloads[-1]["content"] == followup_message["content"]
            elif followup_message.get("tool_calls") is not None:
                assert saved_payloads[-1]["tool_calls"] == followup_message["tool_calls"]
            elif followup_message.get("refusal") is not None:
                assert response["choices"][0]["message"]["refusal"]
            elif followup_result["choices"][0].get("finish_reason") == "content_filter":
                assert response["choices"][0]["finish_reason"] == "content_filter"
            else:
                assert saved_payloads[-1]["function_call"] == followup_message["function_call"]
    else:
        assert response["choices"][0]["message"]["content"] == "Calling tool"
        assert response["tldw_tool_auto_continue"] == {
            "attempted": True,
            "succeeded": False,
        }
        assert sentinel not in repr(response)
        assert all(
            sentinel not in repr(payload)
            and "must not replace" not in str(payload.get("content", ""))
            for payload in saved_payloads
        )
    assert usage_log.await_count == (2 if expected_success else 1)
    if outcome == "valid_raw_text":
        usage = usage_log.await_args.kwargs
        assert usage["estimated"] is True
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "expected_continuation_success"),
    [
        ("valid_text", True),
        ("valid_raw_text", True),
        ("nested_error_prefix", True),
        ("nested_canonical_error", True),
        ("valid_refusal", True),
        ("valid_content_filter", True),
        ("mixed_error_and_text", False),
        ("error", False),
    ],
    ids=(
        "valid",
        "raw",
        "assistant-error-prefix",
        "assistant-provider-code",
        "refusal",
        "content-filter",
        "mixed",
        "error",
    ),
)
async def test_real_adapter_continuation_has_attempt_scoped_accounting(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_continuation_success: bool,
) -> None:
    """Initial and continuation attempts have separate health and usage signals."""

    sentinel = "continuation-accounting-secret-/srv/provider"
    continuation_result = _late_continuation_response(outcome, sentinel)
    if isinstance(continuation_result, dict):
        continuation_result["usage"] = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
    metrics = _DummyMetrics()
    provider_manager = _ProviderManagerStub("anthropic")
    usage_log = AsyncMock(return_value=None)
    mark_used = AsyncMock(return_value=None)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer continuation-accounting-key"
        if any(message.get("role") == "tool" for message in payload["messages"]):
            return continuation_result
        return _build_llm_response_with_tool_calls()

    async def fake_autoexec(**_kwargs: Any) -> ToolExecutionBatchResult:
        return ToolExecutionBatchResult(
            requested_calls=1,
            processed_calls=1,
            execution_attempts=1,
            executed_calls=1,
            truncated=False,
            results=[
                ToolExecutionRecord(
                    tool_call_id="c1",
                    tool_name="notes.search",
                    ok=True,
                    result={"ok": 1},
                    module="notes",
                    content='{"ok":true}',
                )
            ],
        )

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)
    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "continuation-accounting",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "streaming": False,
            "eligible": True,
        },
    )

    resolved_fields = await resolved_request_fields_async(
        "openai",
        api_key="continuation-accounting-key",
        app_config=_registry_openai_app_config(),
        model="gpt-4o-mini",
    )
    response = await _run_execute_non_stream_call(
        llm_call_func=lambda: _adapter_nonstream_call(
            resolved_fields,
            messages_payload=[{"role": "user", "content": "hi"}],
        ),
        save_message_fn=AsyncMock(return_value="saved-message"),
        cleaned_args_overrides=dict(resolved_fields),
        metrics=metrics,
        provider_manager=provider_manager,
        on_success=mark_used,
    )

    assert metrics.llm_calls[0] == ("openai", "gpt-4o-mini", True, None)
    assert provider_manager.successes[0][0] == "openai"
    assert mark_used.await_count == 1
    assert [call["outcome"] for call in metrics.completion_calls] == ["success"]
    if expected_continuation_success:
        if outcome == "valid_text":
            assert response["choices"][0]["message"]["content"] == "late continuation"
        elif outcome in {"nested_error_prefix", "nested_canonical_error"}:
            assert response["choices"][0]["message"]["content"]
        elif outcome == "valid_raw_text":
            assert response == "late raw continuation"
        elif outcome == "valid_refusal":
            assert response["choices"][0]["message"]["refusal"]
        else:
            assert response["choices"][0]["finish_reason"] == "content_filter"
        assert metrics.llm_calls == [
            ("openai", "gpt-4o-mini", True, None),
            ("openai", "gpt-4o-mini", True, None),
        ]
        assert [provider for provider, _latency in provider_manager.successes] == [
            "openai",
            "openai",
        ]
        assert provider_manager.failures == []
        assert len(metrics.token_calls) == 2
        assert usage_log.await_count == 2
        if outcome == "valid_raw_text":
            usage = usage_log.await_args.kwargs
            assert usage["estimated"] is True
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )
            assert metrics.token_calls[-1]["prompt_tokens"] == usage["prompt_tokens"]
            assert metrics.token_calls[-1]["completion_tokens"] == (
                usage["completion_tokens"]
            )
    else:
        assert response["choices"][0]["message"]["content"] == "Calling tool"
        assert response["tldw_tool_auto_continue"] == {
            "attempted": True,
            "succeeded": False,
        }
        assert sentinel not in repr(response)
        assert metrics.llm_calls == [
            ("openai", "gpt-4o-mini", True, None),
            ("openai", "gpt-4o-mini", False, "SanitizedProviderStreamError"),
        ]
        assert [provider for provider, _latency in provider_manager.successes] == [
            "openai"
        ]
        assert provider_manager.failures == [
            ("openai", "SanitizedProviderStreamError")
        ]
        assert len(metrics.token_calls) == 1
        assert usage_log.await_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    "valid_outcome",
    [
        "valid_text",
        "nested_error_prefix",
        "nested_canonical_error",
        "valid_refusal",
        "valid_content_filter",
    ],
    ids=(
        "text",
        "assistant-error-prefix",
        "assistant-provider-code",
        "refusal",
        "content-filter",
    ),
)
async def test_concurrent_tool_continuations_do_not_cross_results_or_marks(
    monkeypatch: pytest.MonkeyPatch,
    valid_outcome: str,
) -> None:
    """Overlapping continuation validation, persistence, and marks stay request-local."""

    sentinel = "concurrent-continuation-secret-/srv/provider"
    ready = [threading.Event(), threading.Event()]
    release = threading.Event()
    mark_ready = [asyncio.Event(), asyncio.Event()]
    release_mark = [asyncio.Event(), asyncio.Event()]
    mark_done = [asyncio.Event(), asyncio.Event()]
    mark_attempts = [0, 0]
    marked: list[list[str]] = [[], []]
    saved: list[list[dict[str, Any]]] = [[], []]
    metrics = [_DummyMetrics(), _DummyMetrics()]
    provider_managers = [
        _ProviderManagerStub("anthropic"),
        _ProviderManagerStub("anthropic"),
    ]
    usage_log = AsyncMock(return_value=None)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        index = 0 if headers["Authorization"] == "Bearer bad-continuation-key" else 1
        if not any(message.get("role") == "tool" for message in payload["messages"]):
            return _build_llm_response_with_tool_calls()
        ready[index].set()
        assert release.wait(timeout=2.0)
        if index == 0:
            invalid_result = _late_continuation_response(
                "nested_structured_error_and_text",
                sentinel,
            )
            invalid_result["usage"] = {
                "prompt_tokens": 21,
                "completion_tokens": 1,
                "total_tokens": 22,
            }
            return invalid_result
        good_result = _late_continuation_response(valid_outcome, sentinel)
        assert isinstance(good_result, dict)
        good_result.setdefault(
            "usage",
            {
                "prompt_tokens": 13,
                "completion_tokens": 2,
                "total_tokens": 15,
            },
        )
        return good_result

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "concurrent-continuation-isolation",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "streaming": False,
            "eligible": True,
        },
    )

    async def fake_autoexec(**_kwargs: Any) -> ToolExecutionBatchResult:
        record = ToolExecutionRecord(
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
            results=[record],
        )

    async def invoke(index: int) -> dict[str, Any]:
        async def mark_used(provider_name: str) -> None:
            mark_attempts[index] += 1
            if mark_attempts[index] == 1:
                raise RuntimeError("transient initial accounting failure")
            mark_ready[index].set()
            await release_mark[index].wait()
            marked[index].append(provider_name)
            mark_done[index].set()

        async def save_message(
            _db: Any,
            _conversation_id: str,
            payload: dict[str, Any],
            *,
            use_transaction: bool,
        ) -> str:
            assert use_transaction is True
            saved[index].append(payload)
            return f"message-{index}-{len(saved[index])}"

        label = "bad" if index == 0 else "good"
        resolved_fields = await resolved_request_fields_async(
            "openai",
            api_key=f"{label}-continuation-key",
            app_config=_registry_openai_app_config(),
            model="gpt-4o-mini",
        )
        return await _run_execute_non_stream_call(
            llm_call_func=lambda: _adapter_nonstream_call(
                resolved_fields,
                messages_payload=[{"role": "user", "content": "hi"}],
            ),
            save_message_fn=save_message,
            cleaned_args_overrides=dict(resolved_fields),
            metrics=metrics[index],
            provider_manager=provider_managers[index],
            on_success=mark_used,
            conversation_id=f"concurrent-continuation-{index}",
        )

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    tasks = [asyncio.create_task(invoke(index)) for index in range(2)]
    try:
        observed = await asyncio.gather(
            *(asyncio.to_thread(event.wait, 1.0) for event in ready)
        )
        assert observed == [True, True]
        release.set()
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), 1.0) for event in mark_ready)
        )
        assert marked == [[], []]
        assert all(task.done() is False for task in tasks)

        release_mark[1].set()
        await asyncio.wait_for(mark_done[1].wait(), timeout=1.0)
        assert marked == [[], ["openai"]]
        release_mark[0].set()
        await asyncio.wait_for(mark_done[0].wait(), timeout=1.0)
        bad_response, good_response = await asyncio.gather(*tasks)
    finally:
        release.set()
        for event in release_mark:
            event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert bad_response["choices"][0]["message"]["content"] == "Calling tool"
    assert bad_response["tldw_tool_auto_continue"] == {
        "attempted": True,
        "succeeded": False,
    }
    assert sentinel not in repr(bad_response)
    if valid_outcome == "valid_text":
        assert good_response["choices"][0]["message"]["content"] == "late continuation"
    elif valid_outcome in {"nested_error_prefix", "nested_canonical_error"}:
        assert good_response["choices"][0]["message"]["content"]
    elif valid_outcome == "valid_refusal":
        assert good_response["choices"][0]["message"]["refusal"]
    else:
        assert good_response["choices"][0]["finish_reason"] == "content_filter"
    assert good_response["tldw_tool_auto_continue"] == {
        "attempted": True,
        "succeeded": True,
    }
    assert mark_attempts == [2, 2]
    assert marked == [["openai"], ["openai"]]
    assert [payload["role"] for payload in saved[0]] == ["assistant", "tool"]
    assert all(
        "must not replace" not in str(payload.get("content", ""))
        for payload in saved[0]
    )
    expected_saved_roles = ["assistant", "tool"]
    if valid_outcome in {
        "valid_text",
        "nested_error_prefix",
        "nested_canonical_error",
    }:
        expected_saved_roles.append("assistant")
    assert [payload["role"] for payload in saved[1]] == expected_saved_roles
    if valid_outcome in {
        "valid_text",
        "nested_error_prefix",
        "nested_canonical_error",
    }:
        assert saved[1][-1]["content"] == {
            "valid_text": "late continuation",
            "nested_error_prefix": "Error: assistant-authored continuation",
            "nested_canonical_error": "provider_unavailable",
        }[valid_outcome]
    assert metrics[0].llm_calls == [
        ("openai", "gpt-4o-mini", True, None),
        ("openai", "gpt-4o-mini", False, "SanitizedProviderStreamError"),
    ]
    assert metrics[1].llm_calls == [
        ("openai", "gpt-4o-mini", True, None),
        ("openai", "gpt-4o-mini", True, None),
    ]
    assert len(metrics[0].token_calls) == 1
    assert len(metrics[1].token_calls) == 2
    assert all(call["provider"] == "openai" for call in metrics[0].token_calls)
    assert all(call["provider"] == "openai" for call in metrics[1].token_calls)
    assert [call["outcome"] for call in metrics[0].completion_calls] == ["success"]
    assert [call["outcome"] for call in metrics[1].completion_calls] == ["success"]
    assert all(
        calls[0]["provider"] == "openai"
        and calls[0]["model"] == "gpt-4o-mini"
        for calls in (metrics[0].completion_calls, metrics[1].completion_calls)
    )
    assert metrics[0].fallback_successes == []
    assert metrics[1].fallback_successes == []
    assert [provider for provider, _latency in provider_managers[0].successes] == [
        "openai"
    ]
    assert provider_managers[0].failures == [
        ("openai", "SanitizedProviderStreamError")
    ]
    assert [provider for provider, _latency in provider_managers[1].successes] == [
        "openai",
        "openai",
    ]
    assert provider_managers[1].failures == []
    usage_by_conversation: dict[str, int] = {}
    for usage_call in usage_log.await_args_list:
        assert usage_call.kwargs["provider"] == "openai"
        assert usage_call.kwargs["model"] == "gpt-4o-mini"
        conversation_id = usage_call.kwargs["conversation_id"]
        usage_by_conversation[conversation_id] = (
            usage_by_conversation.get(conversation_id, 0) + 1
        )
    assert usage_log.await_count == 3
    assert usage_by_conversation == {
        "concurrent-continuation-0": 1,
        "concurrent-continuation-1": 2,
    }


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_text", 1),
        ("valid_raw_text", 1),
        ("valid_list_text", 1),
        ("valid_image", 1),
        ("valid_tool_calls", 1),
        ("valid_function_call", 1),
        ("valid_noncanonical_error_json", 1),
        ("valid_refusal", 1),
        ("valid_content_filter", 1),
        ("empty", 0),
        ("error", 0),
        ("error_prefix", 0),
        ("canonical_raw_code", 0),
        ("sse_error_envelope", 0),
        ("serialized_error_envelope", 0),
        ("malformed_tool_calls", 0),
        ("malformed_function_call", 0),
        ("mixed_error_and_text", 0),
        ("nested_structured_error_and_text", 0),
        ("list_part_error_and_text", 0),
        ("later_choice_error_and_text", 0),
        ("raw_done", 0),
        ("sse_done", 0),
        ("sse_success", 0),
    ],
)
async def test_auto_continue_cancellation_drains_sync_adapter_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Cancellation drains continuation work and marks only usable late output."""

    entered = threading.Event()
    release = threading.Event()
    marked: list[str] = []
    mark_attempts: list[str] = []
    sentinel = "late-continuation-secret-/srv/provider"
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)

    async def fake_autoexec(**_kwargs):
        record = ToolExecutionRecord(
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
            results=[record],
        )

    async def blocking_followup(**_kwargs):
        def invoke_sync_adapter() -> Any:
            entered.set()
            release.wait()
            return _late_continuation_response(late_outcome, sentinel)

        return await asyncio.to_thread(invoke_sync_adapter)

    async def mark_used(provider: str) -> None:
        mark_attempts.append(provider)
        if not entered.is_set():
            raise RuntimeError("transient initial usage accounting failure")
        marked.append(provider)

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)
    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        blocking_followup,
    )

    async def save_message(*_args, **_kwargs) -> str:
        return "saved-message"

    task = asyncio.create_task(
        _run_execute_non_stream_call(
            llm_call_func=_build_llm_response_with_tool_calls,
            save_message_fn=save_message,
            on_success=mark_used,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        assert mark_attempts == ["openai"]
        assert marked == []
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert marked == ["openai"] * expected_marks
    assert mark_attempts == ["openai"] * (1 + expected_marks)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancelled_continuation_drains_real_adapter_before_classify_mark_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation preserves adapter, semantic, usage, and runtime-close ordering."""

    lifecycle: list[str] = []
    continuation_entered = threading.Event()
    release_continuation = threading.Event()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    predicate = getattr(chat_service, "_nonstream_provider_result_is_usable", None)
    assert callable(predicate), "Chat must expose one shared non-stream result predicate"

    def classify(result: Any) -> bool:
        if continuation_entered.is_set():
            lifecycle.append("semantic-classify")
        return predicate(result)

    monkeypatch.setattr(chat_service, "_nonstream_provider_result_is_usable", classify)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer cancelled-continuation-key"
        if not any(message.get("role") == "tool" for message in payload["messages"]):
            return _build_llm_response_with_tool_calls()
        continuation_entered.set()
        assert release_continuation.wait(timeout=2.0)
        return _late_continuation_response(
            "valid_text",
            "cancelled-continuation-secret-/srv/provider",
        )

    def on_client_exit() -> None:
        if continuation_entered.is_set():
            lifecycle.append("adapter-exit")

    _install_real_openai_adapter_transport(
        monkeypatch,
        responder,
        on_client_exit=on_client_exit,
    )
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    monkeypatch.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    monkeypatch.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    monkeypatch.setattr(chat_service, "should_auto_continue_tools_once", lambda: True)

    async def fake_autoexec(**_kwargs: Any) -> ToolExecutionBatchResult:
        return ToolExecutionBatchResult(
            requested_calls=1,
            processed_calls=1,
            execution_attempts=1,
            executed_calls=1,
            truncated=False,
            results=[
                ToolExecutionRecord(
                    tool_call_id="c1",
                    tool_name="notes.search",
                    ok=True,
                    result={"ok": 1},
                    module="notes",
                    content='{"ok":true}',
                )
            ],
        )

    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)

    handle = SimpleNamespace(provider="openai")

    class Runtime:
        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            if not continuation_entered.is_set():
                raise RuntimeError("transient initial credential accounting failure")
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("runtime-close")

    runtime = Runtime()

    async def mark_used(_provider: str) -> None:
        await runtime.mark_used(handle)

    resolved_fields = await resolved_request_fields_async(
        "openai",
        api_key="cancelled-continuation-key",
        app_config=_registry_openai_app_config(),
        model="gpt-4o-mini",
    )

    async def invoke() -> dict[str, Any]:
        try:
            return await _run_execute_non_stream_call(
                llm_call_func=lambda: _adapter_nonstream_call(
                    resolved_fields,
                    messages_payload=[{"role": "user", "content": "hi"}],
                ),
                save_message_fn=AsyncMock(return_value="saved-message"),
                cleaned_args_overrides=dict(resolved_fields),
                on_success=mark_used,
            )
        finally:
            await runtime.close()

    task = asyncio.create_task(invoke())
    try:
        assert await asyncio.to_thread(continuation_entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert lifecycle == []
        release_continuation.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release_continuation.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == [
        "adapter-exit",
        "semantic-classify",
        "mark",
        "runtime-close",
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_auto_continue_redacts_continuation_before_return_and_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer includes unsafe-token",
                    },
                    "finish_reason": "stop",
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
        moderation_getter=lambda: _KeywordModeration(keyword="unsafe-token"),
    )

    assert [p["role"] for p in saved_payloads] == ["assistant", "tool", "assistant"]
    assert saved_payloads[2]["content"] == "Final answer includes [redacted]"
    assert response["choices"][0]["message"]["content"] == "Final answer includes [redacted]"
    assert response["tldw_tool_auto_continue"] == {"attempted": True, "succeeded": True}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_stream_auto_continue_redacts_raw_string_continuation_before_return_and_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(chat_service, "should_force_normalize_string_responses", lambda: False)

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
        return "Final answer includes secret"

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_followup_call)

    saved_payloads: list[dict[str, Any]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=_build_llm_response_with_tool_calls,
        save_message_fn=save_message_fn,
        moderation_getter=lambda: _KeywordModeration(keyword="secret"),
    )

    assert [p["role"] for p in saved_payloads] == ["assistant", "tool", "assistant"]
    assert saved_payloads[2]["content"] == "Final answer includes [redacted]"
    assert response == "Final answer includes [redacted]"


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
    primary_error = chat_service.ChatAPIError("primary failed")
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        saved_payloads.append(payload)
        return f"m-{len(saved_payloads)}"

    response = await _run_execute_non_stream_call(
        llm_call_func=lambda: (_ for _ in ()).throw(primary_error),
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
async def test_tool_autoexec_rejects_multi_choice_fallback_args_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_log_llm_usage(**_kwargs):
        return None

    fallback_provider_called = False

    async def fake_fallback_call(**_kwargs):
        nonlocal fallback_provider_called
        fallback_provider_called = True
        return {
            "choices": [{"message": {"role": "assistant", "content": "Fallback response"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_fallback_call)

    provider_manager = _ProviderManagerStub("anthropic")
    primary_error = chat_service.ChatAPIError("primary failed")
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    with pytest.raises(HTTPException) as exc_info:
        await _run_execute_non_stream_call(
            llm_call_func=lambda: (_ for _ in ()).throw(primary_error),
            save_message_fn=save_message_fn,
            provider_manager=provider_manager,
            enable_provider_fallback=True,
            refresh_provider_params=lambda fallback_provider: (
                {
                    "api_endpoint": fallback_provider,
                    "api_key": "fallback-key",
                    "messages_payload": [{"role": "user", "content": "hi"}],
                    "model": "claude-3-7-sonnet",
                    "streaming": False,
                    "n": 2,
                    "tools": [{"type": "function", "function": {"name": "notes.search", "parameters": {}}}],
                },
                "claude-3-7-sonnet",
            ),
        )

    assert fallback_provider_called is False
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "unsupported_multi_choice_tool_autoexec",
        "message": "Local tool auto-execution supports one assistant choice per request.",
    }
    assert provider_manager.failures == [("openai", "SanitizedProviderStreamError")]


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
