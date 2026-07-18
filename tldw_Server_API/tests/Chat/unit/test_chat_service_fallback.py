import asyncio
import contextlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from loguru import logger
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat import request_queue as request_queue_module
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Chat.chat_metrics import ChatMetricsCollector
from tldw_Server_API.app.core.Chat.chat_service import (
    execute_non_stream_call,
    execute_streaming_call,
    merge_api_keys_for_provider,
    write_mandatory_moderation_audit,
)
from tldw_Server_API.app.core.Chat.request_queue import RateLimitedQueue, RequestQueue

_REGISTRY_OPENAI_BASE_URL = "https://registry-openai.test/v1"


def _registry_openai_app_config() -> dict[str, dict[str, str]]:
    """Return the authoritative adapter URL snapshot for registry tests."""

    return {"openai_api": {"api_base_url": _REGISTRY_OPENAI_BASE_URL}}


class _DummyMetrics:
    def __init__(self):
        self.llm_calls = []
        self.fallback_successes = []
        self.token_calls = []
        self.completion_calls = []
        self._collector = ChatMetricsCollector()

    def track_llm_call(self, provider, model, latency, success, error_type=None):

        self.llm_calls.append((provider, model, success, error_type))

    def track_provider_fallback_success(self, **metadata):

        self.fallback_successes.append(metadata)

    def track_tokens(self, **kwargs):

        self.token_calls.append(kwargs)

    def track_run_first_completion_proxy(self, **kwargs):

        self.completion_calls.append(kwargs)

    def track_streaming(self, conversation_id: str):
        return self._collector.track_streaming(conversation_id)


class _DummyProviderManager:
    def __init__(self):
        self.failure_records = []
        self.failure_errors = []
        self.success_records = []
        self.fallback_requests = []

    def get_available_provider(self, exclude=None):

        self.fallback_requests.append(tuple(exclude or []))
        return "openai"

    def record_failure(self, provider, error):

        self.failure_records.append((provider, type(error).__name__))
        self.failure_errors.append(error)

    def record_success(self, provider, latency):

        self.success_records.append(provider)


class _ReleaseTrackingDaemonPool(BoundedDaemonPool):
    """Record capacity release ordering while preserving the real pool behavior."""

    def __init__(self, capacity: int, lifecycle: list[str]) -> None:
        super().__init__(capacity)
        self.lifecycle = lifecycle
        self.release_count = 0

    def _release_capacity(self) -> None:
        self.lifecycle.append("capacity-release")
        self.release_count += 1
        super()._release_capacity()


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> None:
    """Wait for a thread event without consuming the default executor under test."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for thread event")
        await asyncio.sleep(0.001)


def _install_owned_worker_drain_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Event:
    """Expose entry into owned-worker cancellation draining."""

    drain_entered = asyncio.Event()
    original = bounded_daemon_module._drain_owned_task

    async def probe(task):
        drain_entered.set()
        return await original(task)

    monkeypatch.setattr(bounded_daemon_module, "_drain_owned_task", probe)
    return drain_entered


class _DummyModeration:
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


@pytest.mark.asyncio
async def test_concurrent_mandatory_audit_failure_is_bounded_and_isolated():
    sentinel = "mandatory-audit-secret-/srv/audit"
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    class UnknownAuditFailure(Exception):
        pass

    class AuditService:
        def __init__(self, index: int, *, fails: bool) -> None:
            self.index = index
            self.fails = fails

        async def log_event(self, **_kwargs) -> None:
            ready[self.index].set()
            await release.wait()
            if self.fails:
                raise UnknownAuditFailure(sentinel)

        async def flush(self, **_kwargs) -> None:
            return None

    async def invoke(index: int, *, fails: bool):
        return await write_mandatory_moderation_audit(
            audit_service=AuditService(index, fails=fails),
            audit_context=object(),
            audit_event_type=None,
            action="moderation.output",
            result="failure" if fails else "success",
            metadata={},
        )

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        failed_task = asyncio.create_task(invoke(0, fails=True))
        healthy_task = asyncio.create_task(invoke(1, fails=False))
        await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
        release.set()
        failed_result, healthy_result = await asyncio.gather(
            failed_task,
            healthy_task,
            return_exceptions=True,
        )
    finally:
        release.set()
        logger.remove(sink_id)

    assert isinstance(failed_result, MandatoryAuditWriteError)
    assert healthy_result is None
    assert sentinel not in "".join(logs)

class _ProcessorQueue:
    def is_running(self):
        return True

    async def enqueue(self, *, processor, **_kwargs):
        return asyncio.create_task(asyncio.to_thread(processor))


def _certified_pre_dispatch(error):
    error.upstream_dispatched = False
    error.output_emitted = False
    error.allow_non_stream_fallback = True
    return error


async def _noop_save_message(*_args, **_kwargs):
    return None


def _install_slow_stream_preflight(monkeypatch) -> None:
    """Advance only the aggregate factory clock during local config preflight."""

    clock = SimpleNamespace(now=0.0)

    def load_config():
        clock.now += 1.0
        return SimpleNamespace(enabled=False)

    monkeypatch.setattr(chat_service, "_provider_factory_monotonic", lambda: clock.now)
    monkeypatch.setattr(chat_service, "load_prompt_cost_guardrail_config", load_config)


def _nonstream_call_kwargs(
    *,
    llm_call_func,
    provider_manager,
    metrics,
    refresh_provider_params,
    queue_execution_enabled=False,
    enable_provider_fallback=True,
    conversation_id="conv-nonstream-boundary",
    provider="anthropic",
    model="claude-3",
):
    return {
        "current_loop": asyncio.get_running_loop(),
        "cleaned_args": {
            "api_endpoint": provider,
            "api_key": "primary-key",
            "messages_payload": [],
            "model": model,
            "streaming": False,
        },
        "selected_provider": provider,
        "provider": provider,
        "model": model,
        "request_json": "{}",
        "request": SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/chat/completions"),
            headers={},
            state=SimpleNamespace(user_id=None, api_key_id=None),
        ),
        "metrics": metrics,
        "provider_manager": provider_manager,
        "templated_llm_payload": [],
        "should_persist": False,
        "final_conversation_id": conversation_id,
        "character_card_for_context": {},
        "chat_db": None,
        "save_message_fn": _noop_save_message,
        "audit_service": None,
        "audit_context": None,
        "client_id": conversation_id,
        "queue_execution_enabled": queue_execution_enabled,
        "enable_provider_fallback": enable_provider_fallback,
        "llm_call_func": llm_call_func,
        "refresh_provider_params": refresh_provider_params,
        "moderation_getter": lambda: _DummyModeration(),
    }


def _successful_nonstream_response(content: str) -> dict[str, object]:
    """Return the smallest provider response accepted by non-stream Chat."""

    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


_REASONING_OUTPUT_FIELDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
    "thinking",
    "analysis",
)


def _reasoning_only_nonstream_response(
    field: str,
    value: Any,
) -> dict[str, object]:
    """Return an OpenAI-compatible response whose only output is reasoning."""

    return {
        "choices": [
            {
                "message": {"content": None, field: value},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _valid_reasoning_value(field: str) -> Any:
    if field == "reasoning_details":
        return {"text": "provider reasoning"}
    return "provider reasoning"


def _late_nonstream_response(outcome: str, sentinel: str) -> Any:
    """Build representative late provider results for cancellation accounting."""

    if outcome == "valid_text":
        return _successful_nonstream_response("late provider success")
    if outcome == "valid_raw_text":
        return "late raw provider success"
    if outcome == "valid_tool_calls":
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
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
                        "content": None,
                        "function_call": {
                            "name": "lookup",
                            "arguments": "{}",
                        },
                    }
                }
            ]
        }
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
                        "content": None,
                        "function_call": {"arguments": "{}"},
                    }
                }
            ]
        }
    raise AssertionError(f"Unsupported late provider outcome: {outcome}")


def _normal_nonstream_response(outcome: str, sentinel: str) -> Any:
    """Build normal adapter results for semantic success accounting tests."""

    if outcome in {
        "valid_text",
        "valid_raw_text",
        "valid_tool_calls",
        "valid_function_call",
        "empty",
        "error",
        "error_prefix",
        "canonical_raw_code",
        "sse_error_envelope",
        "serialized_error_envelope",
        "malformed_tool_calls",
        "malformed_function_call",
    }:
        return _late_nonstream_response(outcome, sentinel)
    if outcome == "valid_list_text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": "usable list text"}]
                    }
                }
            ]
        }
    if outcome == "valid_image":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,dGVzdA=="
                                },
                            }
                        ]
                    }
                }
            ]
        }
    if outcome == "valid_noncanonical_error_json":
        return json.dumps(
            {
                "error": {
                    "code": "fictional_story_error",
                    "message": "ordinary assistant-authored content",
                }
            },
            separators=(",", ":"),
        )
    if outcome == "valid_nested_noncanonical_error_json":
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "error": {
                                    "code": "fictional_story_error",
                                    "message": "ordinary assistant-authored content",
                                }
                            },
                            separators=(",", ":"),
                        )
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
                        "refusal": "I cannot help with that request.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "total_tokens": 7,
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
                "prompt_tokens": 4,
                "completion_tokens": 0,
                "total_tokens": 4,
            },
        }
    if outcome == "raw_done":
        return "[DONE]"
    if outcome == "sse_done":
        return "data: [DONE]\n\n"
    if outcome == "sse_success":
        return 'data: {"choices":[{"message":{"content":"framed"}}]}\n\n'
    if outcome == "empty_content":
        return {"choices": [{"message": {"content": ""}}]}
    if outcome == "empty_image":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "image_url", "image_url": {"url": ""}}
                        ]
                    }
                }
            ]
        }
    if outcome == "nested_error_prefix":
        return {
            "choices": [
                {"message": {"content": "Error: assistant-authored content"}}
            ]
        }
    if outcome == "nested_canonical_error":
        return {
            "choices": [
                {"message": {"content": "provider_unavailable"}}
            ]
        }
    if outcome == "mixed_error_and_text":
        return {
            "error": {"code": "provider_unavailable", "message": sentinel},
            "choices": [
                {"message": {"content": "must not override the error"}}
            ],
        }
    if outcome == "nested_structured_error_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "error": {
                            "code": "provider_unavailable",
                            "message": sentinel,
                        },
                        "content": "must not override the nested error",
                    }
                }
            ]
        }
    if outcome == "list_part_error_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "must not override the nested error"},
                            {
                                "type": "error",
                                "error": {
                                    "code": "provider_unavailable",
                                    "message": sentinel,
                                },
                            },
                        ]
                    }
                }
            ]
        }
    if outcome == "later_choice_error_and_text":
        return {
            "choices": [
                {"message": {"content": "must not override the later error"}},
                {
                    "error": {
                        "code": "provider_unavailable",
                        "message": sentinel,
                    }
                },
            ]
        }
    if outcome == "whitespace_raw_text":
        return "   "
    if outcome == "bytes_text":
        return b"provider text is not a supported non-stream result"
    if outcome == "none_result":
        return None
    if outcome == "bool_scalar":
        return True
    if outcome == "int_scalar":
        return 7
    if outcome == "float_scalar":
        return 3.5
    if outcome == "arbitrary_object":
        return object()
    if outcome == "empty_dict":
        return {}
    if outcome == "empty_list":
        return []
    if outcome == "empty_tuple":
        return ()
    if outcome == "nonempty_tuple":
        return ("provider text",)
    if outcome == "choices_not_list":
        return {"choices": {"message": {"content": "not a choice list"}}}
    if outcome == "choice_not_dict":
        return {"choices": ["not a choice object"]}
    if outcome == "message_not_dict":
        return {"choices": [{"message": "not a message object"}]}
    if outcome == "unsupported_content_object":
        return {"choices": [{"message": {"content": {"text": "unsupported"}}}]}
    if outcome == "type_error_envelope":
        return {
            "type": "error",
            "code": "provider_unavailable",
            "message": sentinel,
        }
    if outcome == "error_code_envelope":
        return {"error_code": "provider_unavailable", "message": sentinel}
    if outcome == "nested_type_error_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "type": "error",
                        "code": "provider_unavailable",
                        "content": "must not override the nested error",
                    }
                }
            ]
        }
    if outcome == "nested_error_code_and_text":
        return {
            "choices": [
                {
                    "message": {
                        "error_code": "provider_unavailable",
                        "content": "must not override the nested error",
                    }
                }
            ]
        }
    if outcome == "valid_tool_domain_error_arguments":
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-domain-error",
                                "type": "function",
                                "function": {
                                    "name": "report_result",
                                    "arguments": json.dumps(
                                        {
                                            "error": {
                                                "code": "domain_validation",
                                                "message": "ordinary tool-domain data",
                                            }
                                        },
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    raise AssertionError(f"Unsupported normal provider outcome: {outcome}")


def _adapter_nonstream_call(
    api_key: str,
    *,
    messages_payload: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
) -> Any:
    """Exercise the production sync adapter-dispatch boundary for a test key."""

    return chat_service.perform_chat_api_call(
        api_endpoint="openai",
        api_key=api_key,
        credentials_resolved=True,
        app_config=_registry_openai_app_config(),
        messages_payload=messages_payload or [],
        model=model,
        streaming=False,
    )


def _install_real_openai_adapter_transport(
    monkeypatch: pytest.MonkeyPatch,
    responder,
) -> None:
    """Install a fake transport below the real registry-backed OpenAI adapter."""

    from tldw_Server_API.app.core.LLM_Calls import adapter_registry
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    # Production resolves the process registry lazily. Reset only its cache seam so
    # each test gets an isolated real registry that monkeypatch restores afterward.
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
            return None

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


@pytest.mark.parametrize(
    ("outcome", "expected_usable"),
    [
        ("valid_text", True),
        ("valid_raw_text", True),
        ("valid_list_text", True),
        ("valid_image", True),
        ("valid_tool_calls", True),
        ("valid_function_call", True),
        ("valid_noncanonical_error_json", True),
        ("valid_nested_noncanonical_error_json", True),
        ("valid_refusal", True),
        ("valid_content_filter", True),
        ("empty", False),
        ("empty_content", False),
        ("empty_image", False),
        ("error", False),
        ("error_prefix", False),
        ("canonical_raw_code", False),
        ("sse_error_envelope", False),
        ("serialized_error_envelope", False),
        ("malformed_tool_calls", False),
        ("malformed_function_call", False),
        ("nested_error_prefix", True),
        ("nested_canonical_error", True),
        ("mixed_error_and_text", False),
        ("nested_structured_error_and_text", False),
        ("list_part_error_and_text", False),
        ("later_choice_error_and_text", False),
        ("whitespace_raw_text", False),
        ("bytes_text", False),
        ("none_result", False),
        ("bool_scalar", False),
        ("int_scalar", False),
        ("float_scalar", False),
        ("arbitrary_object", False),
        ("empty_dict", False),
        ("empty_list", False),
        ("empty_tuple", False),
        ("nonempty_tuple", False),
        ("choices_not_list", False),
        ("choice_not_dict", False),
        ("message_not_dict", False),
        ("unsupported_content_object", False),
        ("type_error_envelope", False),
        ("error_code_envelope", False),
        ("nested_type_error_and_text", False),
        ("nested_error_code_and_text", False),
        ("valid_tool_domain_error_arguments", True),
        ("raw_done", False),
        ("sse_done", False),
        ("sse_success", False),
    ],
)
def test_nonstream_provider_result_semantic_contract(
    outcome: str,
    expected_usable: bool,
) -> None:
    """The shared non-stream predicate owns every supported success/error shape."""

    predicate = getattr(
        chat_service,
        "_nonstream_provider_result_is_usable",
        None,
    )
    assert callable(predicate), "Chat must expose one shared non-stream result predicate"
    result = _normal_nonstream_response(
        outcome,
        "predicate-contract-secret-/srv/provider",
    )

    assert predicate(result) is expected_usable


@pytest.mark.parametrize("reasoning_field", _REASONING_OUTPUT_FIELDS)
@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        "",
        "   ",
        [],
        {},
        True,
        7,
        {"text": "   "},
        {"type": "reasoning.text", "text": "   "},
        {"id": "block-1", "index": 0, "signature": "opaque"},
        {"error": {"code": "provider_unavailable"}},
    ],
    ids=(
        "none",
        "empty-string",
        "whitespace",
        "empty-list",
        "empty-dict",
        "boolean",
        "number",
        "whitespace-structured",
        "metadata-with-whitespace",
        "metadata-only",
        "error-envelope",
    ),
)
def test_nonstream_reasoning_only_rejects_empty_malformed_and_error_values(
    reasoning_field: str,
    invalid_value: Any,
) -> None:
    """Reasoning field presence alone is not validated semantic output."""

    result = _reasoning_only_nonstream_response(reasoning_field, invalid_value)

    assert chat_service._nonstream_provider_result_is_usable(result) is False


@pytest.mark.parametrize("reasoning_field", _REASONING_OUTPUT_FIELDS)
def test_nonstream_reasoning_only_rejects_valid_hidden_text(
    reasoning_field: str,
) -> None:
    """Hidden reasoning is not a persistable assistant response by itself."""

    result = _reasoning_only_nonstream_response(
        reasoning_field,
        _valid_reasoning_value(reasoning_field),
    )

    assert chat_service._nonstream_provider_result_is_usable(result) is False


def test_nonstream_visible_content_accepts_long_bounded_reasoning_metadata() -> None:
    """Supplementary reasoning metadata does not invalidate visible content."""

    result = _successful_nonstream_response("visible assistant content")
    result["choices"][0]["message"]["reasoning_details"] = [
        {
            "type": "reasoning.text",
            "text": f"provider reasoning step {index}",
            "signature": f"opaque-{index}",
        }
        for index in range(96)
    ]

    assert chat_service._nonstream_provider_result_is_usable(result) is True


def test_nonstream_reasoning_details_rejects_over_depth_content_chain() -> None:
    """Reasoning traversal enforces its depth cap without recursive inspection."""

    reasoning: Any = "provider reasoning"
    for _ in range(1100):
        reasoning = {"content": [reasoning]}
    result = _reasoning_only_nonstream_response("reasoning_details", reasoning)

    assert chat_service._nonstream_provider_result_is_usable(result) is False


def test_nonstream_reasoning_details_rejects_over_node_budget() -> None:
    """A semantic leaf cannot bypass the bounded reasoning node budget."""

    reasoning = [None] * 1023 + ["provider reasoning"]
    result = _reasoning_only_nonstream_response("reasoning_details", reasoning)

    assert chat_service._nonstream_provider_result_is_usable(result) is False


def test_nonstream_reasoning_details_rejects_cycle() -> None:
    """A cyclic provider object is rejected without looping or recursion."""

    reasoning: list[Any] = []
    reasoning.append(reasoning)
    result = _reasoning_only_nonstream_response("reasoning_details", reasoning)

    assert chat_service._nonstream_provider_result_is_usable(result) is False


def test_nonstream_valid_content_cannot_mask_nested_reasoning_error() -> None:
    """A reasoning error envelope invalidates otherwise usable content."""

    result = _successful_nonstream_response("visible assistant content")
    result["choices"][0]["message"]["reasoning_details"] = {
        "blocks": [
            {
                "error": {
                    "code": "provider_unavailable",
                    "message": "must remain a failure",
                }
            }
        ]
    }

    assert chat_service._nonstream_provider_result_is_usable(result) is False


@pytest.mark.parametrize("reasoning_field", _REASONING_OUTPUT_FIELDS)
@pytest.mark.parametrize(
    "framed_or_error_text",
    [
        "provider_unavailable",
        "Error: provider credentials leaked",
        'data: {"error":{"code":"provider_unavailable","message":"leaked"}}\n\n',
        "data: [DONE]\n\n",
        'event: message\ndata: {"choices":[{"message":{"content":"framed"}}]}\n\n',
    ],
    ids=("canonical-code", "legacy-error", "sse-error", "sse-done", "sse-success"),
)
def test_nonstream_visible_content_cannot_mask_reasoning_protocol_strings(
    reasoning_field: str,
    framed_or_error_text: str,
) -> None:
    """Raw protocol/error strings in reasoning never become provider success."""

    result = _successful_nonstream_response("visible assistant content")
    result["choices"][0]["message"][reasoning_field] = framed_or_error_text

    assert chat_service._nonstream_provider_result_is_usable(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_field", _REASONING_OUTPUT_FIELDS)
async def test_reasoning_only_real_adapter_boundary_fails_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_field: str,
) -> None:
    """A real adapter's hidden-only response fails before success side effects."""

    response = _reasoning_only_nonstream_response(
        reasoning_field,
        _valid_reasoning_value(reasoning_field),
    )
    transport_calls: list[str] = []
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    mark_used = AsyncMock(return_value=None)
    log_usage = AsyncMock(return_value=None)
    save_message = AsyncMock(return_value="must-not-save")
    moderation = _DummyModeration()
    moderation.check_text = AsyncMock()  # type: ignore[method-assign]

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, object]:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["stream"] is False
        transport_calls.append(headers["Authorization"])
        return response

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", log_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=lambda: _adapter_nonstream_call("reasoning-boundary-key"),
        provider_manager=provider_manager,
        metrics=metrics,
        refresh_provider_params=lambda _provider: ({}, None),
        enable_provider_fallback=False,
        conversation_id=f"reasoning-boundary-{reasoning_field}",
        provider="openai",
        model="gpt-4o",
    )
    kwargs["on_success"] = mark_used
    kwargs["should_persist"] = True
    kwargs["save_message_fn"] = save_message
    kwargs["moderation_getter"] = lambda: moderation

    with pytest.raises(SanitizedProviderStreamError) as captured:
        await execute_non_stream_call(**kwargs)

    assert captured.value.code == "provider_unavailable"
    assert transport_calls == ["Bearer reasoning-boundary-key"]
    assert provider_manager.success_records == []
    assert metrics.llm_calls == [
        ("openai", "gpt-4o", False, "SanitizedProviderStreamError")
    ]
    mark_used.assert_not_awaited()
    log_usage.assert_not_awaited()
    save_message.assert_not_awaited()
    moderation.check_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_visible_content_with_reasoning_uses_canonical_downstream_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moderation, accounting, and persistence ignore hidden reasoning metadata."""

    visible_text = "visible assistant answer for accounting"
    hidden_text = "hidden chain of thought must not be persisted"
    opaque_signature = "opaque-provider-signature"
    response = _successful_nonstream_response(visible_text)
    response.pop("usage")
    response["choices"][0]["message"]["reasoning_details"] = [
        {
            "type": "reasoning.text",
            "text": hidden_text,
            "signature": opaque_signature,
        }
    ]
    moderation_inputs: list[str] = []
    self_monitoring_inputs: list[str] = []
    save_message = AsyncMock(return_value="assistant-message-id")
    log_usage = AsyncMock(return_value=None)
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    class TrackingModeration(_DummyModeration):
        class _Policy:
            enabled = True
            output_enabled = True

        def check_text(self, text, *_args, **_kwargs):
            moderation_inputs.append(text)
            return (False, None)

    class TrackingSelfMonitoring:
        def check_text(self, *, text, **_kwargs):
            self_monitoring_inputs.append(text)
            return SimpleNamespace(action="pass", redacted_text=None)

    _install_real_openai_adapter_transport(
        monkeypatch,
        lambda **_kwargs: response,
    )
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", log_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=lambda: _adapter_nonstream_call("reasoning-visible-key"),
        provider_manager=provider_manager,
        metrics=metrics,
        refresh_provider_params=lambda _provider: ({}, None),
        enable_provider_fallback=False,
        conversation_id="reasoning-visible-content",
        provider="openai",
        model="gpt-4o",
    )
    kwargs.update(
        should_persist=True,
        chat_db=object(),
        save_message_fn=save_message,
        moderation_getter=TrackingModeration,
        self_monitoring_service=TrackingSelfMonitoring(),
    )

    result = await execute_non_stream_call(**kwargs)

    assert result["choices"][0]["message"]["content"] == visible_text
    assert moderation_inputs == [visible_text]
    assert self_monitoring_inputs == [visible_text]
    saved_payload = save_message.await_args.args[2]
    assert saved_payload["content"] == visible_text
    assert hidden_text not in json.dumps(saved_payload)
    assert opaque_signature not in json.dumps(saved_payload)
    assert log_usage.await_args.kwargs["completion_tokens"] == len(visible_text) // 4
    assert metrics.token_calls == [
        {
            "prompt_tokens": log_usage.await_args.kwargs["prompt_tokens"],
            "completion_tokens": len(visible_text) // 4,
            "model": "gpt-4o",
            "provider": "openai",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_field", _REASONING_OUTPUT_FIELDS)
async def test_cancelled_reasoning_only_late_result_does_not_mark_provider_usage(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_field: str,
) -> None:
    """Cancellation drains hidden-only output without recording provider use."""

    entered = threading.Event()
    release = threading.Event()
    marked: list[str] = []
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    def reasoning_call() -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=2.0)
        return _reasoning_only_nonstream_response(
            reasoning_field,
            _valid_reasoning_value(reasoning_field),
        )

    async def mark_used(provider_name: str) -> None:
        marked.append(provider_name)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=reasoning_call,
        provider_manager=_DummyProviderManager(),
        metrics=_DummyMetrics(),
        refresh_provider_params=lambda _provider: ({}, None),
        enable_provider_fallback=False,
        conversation_id=f"cancelled-reasoning-{reasoning_field}",
    )
    kwargs["on_success"] = mark_used
    task = asyncio.create_task(execute_non_stream_call(**kwargs))
    try:
        await _wait_for_thread_event(entered)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert marked == []

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert marked == []


@pytest.mark.asyncio
async def test_concurrent_reasoning_valid_malformed_and_error_results_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only canonical visible output receives concurrent success accounting."""

    ready = [threading.Event(), threading.Event(), threading.Event()]
    release = threading.Event()
    marked: list[list[str]] = [[], [], []]
    metrics = [_DummyMetrics(), _DummyMetrics(), _DummyMetrics()]
    managers = [_DummyProviderManager(), _DummyProviderManager(), _DummyProviderManager()]
    log_usage = AsyncMock(return_value=None)
    valid_response = _successful_nonstream_response("visible concurrent answer")
    valid_response["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.text", "text": "supplementary provider reasoning"}
    ]
    framed_error_response = _successful_nonstream_response("must not mask an error")
    framed_error_response["choices"][0]["message"]["reasoning"] = (
        'data: {"error":{"code":"provider_unavailable","message":"secret"}}\n\n'
    )
    responses = [
        valid_response,
        _reasoning_only_nonstream_response(
            "reasoning_details",
            {"text": "hidden-only provider reasoning"},
        ),
        framed_error_response,
    ]

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, object]:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["stream"] is False
        index = {
            "Bearer valid-reasoning-key": 0,
            "Bearer malformed-reasoning-key": 1,
            "Bearer error-reasoning-key": 2,
        }[headers["Authorization"]]
        ready[index].set()
        assert release.wait(timeout=2.0)
        return responses[index]

    async def invoke(index: int) -> dict[str, Any]:
        async def mark_used(provider_name: str) -> None:
            marked[index].append(provider_name)

        api_key = (
            "valid-reasoning-key",
            "malformed-reasoning-key",
            "error-reasoning-key",
        )[index]
        kwargs = _nonstream_call_kwargs(
            llm_call_func=lambda: _adapter_nonstream_call(api_key),
            provider_manager=managers[index],
            metrics=metrics[index],
            refresh_provider_params=lambda _provider: ({}, None),
            enable_provider_fallback=False,
            conversation_id=f"concurrent-reasoning-{index}",
            provider="openai",
            model="gpt-4o",
        )
        kwargs["on_success"] = mark_used
        return await execute_non_stream_call(**kwargs)

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", log_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)

    tasks = [asyncio.create_task(invoke(index)) for index in range(3)]
    try:
        await asyncio.gather(*(_wait_for_thread_event(event) for event in ready))
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert results[0] == {
        **responses[0],
        "tldw_conversation_id": "concurrent-reasoning-0",
    }
    assert isinstance(results[1], SanitizedProviderStreamError)
    assert isinstance(results[2], SanitizedProviderStreamError)
    assert marked == [["openai"], [], []]
    assert managers[0].success_records == ["openai"]
    assert managers[1].success_records == []
    assert managers[2].success_records == []
    assert metrics[0].llm_calls == [("openai", "gpt-4o", True, None)]
    assert metrics[1].llm_calls == [
        ("openai", "gpt-4o", False, "SanitizedProviderStreamError")
    ]
    assert metrics[2].llm_calls == [
        ("openai", "gpt-4o", False, "SanitizedProviderStreamError")
    ]
    assert log_usage.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_success"),
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
async def test_primary_real_adapter_boundary_gates_exact_accounting(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_success: bool,
) -> None:
    """Primary dispatch crosses the registry and real adapter before classification."""

    sentinel = "primary-boundary-secret-/srv/provider"
    result = _normal_nonstream_response(outcome, sentinel)
    if isinstance(result, dict):
        result["usage"] = {
            "prompt_tokens": 11,
            "completion_tokens": 2,
            "total_tokens": 13,
        }
    transport_calls: list[tuple[str, dict[str, Any]]] = []
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    log_usage = AsyncMock(return_value=None)
    mark_used = AsyncMock(return_value=None)
    save_message = AsyncMock(return_value="primary-message")

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer primary-boundary-key"
        transport_calls.append((headers["Authorization"], payload))
        return result

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", log_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "real-primary-boundary",
            "provider": "openai",
            "model": "gpt-4o",
            "streaming": False,
            "eligible": True,
        },
    )

    kwargs = _nonstream_call_kwargs(
        llm_call_func=lambda: _adapter_nonstream_call("primary-boundary-key"),
        provider_manager=provider_manager,
        metrics=metrics,
        refresh_provider_params=lambda _provider: ({}, None),
        queue_execution_enabled=False,
        enable_provider_fallback=True,
        conversation_id=f"primary-real-adapter-{outcome}",
        provider="openai",
        model="gpt-4o",
    )
    kwargs.update(
        should_persist=True,
        save_message_fn=save_message,
        on_success=mark_used,
    )

    if expected_success:
        response = await execute_non_stream_call(**kwargs)
        if outcome == "valid_text":
            assert response["choices"][0]["message"]["content"] == "late provider success"
        elif outcome == "valid_raw_text":
            assert response == "late raw provider success"
        elif outcome in {"nested_error_prefix", "nested_canonical_error"}:
            assert response["choices"][0]["message"]["content"]
        elif outcome == "valid_refusal":
            assert response["choices"][0]["message"]["refusal"]
        else:
            assert response["choices"][0]["finish_reason"] == "content_filter"
        assert metrics.llm_calls == [("openai", "gpt-4o", True, None)]
        assert len(metrics.token_calls) == 1
        assert [call["outcome"] for call in metrics.completion_calls] == ["success"]
        assert metrics.fallback_successes == []
        assert provider_manager.failure_records == []
        assert provider_manager.success_records == ["openai"]
        assert provider_manager.fallback_requests == []
        log_usage.assert_awaited_once()
        if outcome == "valid_raw_text":
            usage = log_usage.await_args.kwargs
            assert usage["estimated"] is True
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )
            assert metrics.token_calls[0]["prompt_tokens"] == usage["prompt_tokens"]
            assert metrics.token_calls[0]["completion_tokens"] == (
                usage["completion_tokens"]
            )
        mark_used.assert_awaited_once_with("openai")
        if outcome in {
            "valid_text",
            "valid_raw_text",
            "nested_error_prefix",
            "nested_canonical_error",
        }:
            save_message.assert_awaited_once()
        else:
            save_message.assert_not_awaited()
    else:
        with pytest.raises(SanitizedProviderStreamError) as captured:
            await execute_non_stream_call(**kwargs)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert sentinel not in repr(captured.value)
        assert metrics.llm_calls == [
            ("openai", "gpt-4o", False, "SanitizedProviderStreamError")
        ]
        assert metrics.token_calls == []
        assert [call["outcome"] for call in metrics.completion_calls] == ["error"]
        assert metrics.fallback_successes == []
        assert provider_manager.failure_records == [
            ("openai", "SanitizedProviderStreamError")
        ]
        assert provider_manager.success_records == []
        assert provider_manager.fallback_requests == []
        log_usage.assert_not_awaited()
        mark_used.assert_not_awaited()
        save_message.assert_not_awaited()

    assert len(transport_calls) == 1
    assert transport_calls[0][1]["stream"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["direct", "queued"])
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
        ("valid_nested_noncanonical_error_json", True),
        ("valid_refusal", True),
        ("valid_content_filter", True),
        ("empty", False),
        ("empty_content", False),
        ("empty_image", False),
        ("error", False),
        ("error_prefix", False),
        ("canonical_raw_code", False),
        ("sse_error_envelope", False),
        ("serialized_error_envelope", False),
        ("malformed_tool_calls", False),
        ("malformed_function_call", False),
        ("nested_error_prefix", True),
        ("nested_canonical_error", True),
        ("mixed_error_and_text", False),
        ("nested_structured_error_and_text", False),
        ("list_part_error_and_text", False),
        ("later_choice_error_and_text", False),
        ("raw_done", False),
        ("sse_done", False),
        ("sse_success", False),
    ],
)
async def test_normal_adapter_result_is_gated_before_success_accounting(
    monkeypatch: pytest.MonkeyPatch,
    queued: bool,
    outcome: str,
    expected_success: bool,
) -> None:
    """Normal adapter results must be usable before any clean-success side effect."""

    sentinel = "normal-adapter-result-secret-/srv/provider"
    result = _normal_nonstream_response(outcome, sentinel)
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    marked: list[str] = []
    saved: list[dict[str, Any]] = []
    usage_log = AsyncMock(return_value=None)
    queue = _ProcessorQueue() if queued else None

    async def mark_used(provider_name: str) -> None:
        marked.append(provider_name)

    async def save_message(
        _db: Any,
        _conversation_id: str,
        payload: dict[str, Any],
        *,
        use_transaction: bool,
    ) -> str:
        assert use_transaction is True
        saved.append(payload)
        return "message-1"

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)
    monkeypatch.setattr(chat_service, "should_force_normalize_string_responses", lambda: False)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=lambda: result,
        provider_manager=provider_manager,
        metrics=metrics,
        refresh_provider_params=lambda _provider: ({}, None),
        queue_execution_enabled=queued,
        enable_provider_fallback=True,
        conversation_id=f"normal-result-{queued}-{outcome}",
    )
    kwargs.update(
        should_persist=True,
        save_message_fn=save_message,
        on_success=mark_used,
    )

    if expected_success:
        response = await execute_non_stream_call(**kwargs)
        assert marked == ["anthropic"]
        assert provider_manager.success_records == ["anthropic"]
        assert any(call[2] is True for call in metrics.llm_calls)
        usage_log.assert_awaited_once()
        if outcome == "valid_raw_text":
            usage = usage_log.await_args.kwargs
            assert usage["estimated"] is True
            assert usage["prompt_tokens"] >= 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )
            assert metrics.token_calls[0]["prompt_tokens"] == usage["prompt_tokens"]
            assert metrics.token_calls[0]["completion_tokens"] == (
                usage["completion_tokens"]
            )
        if outcome == "valid_refusal":
            assert response["choices"][0]["message"]["refusal"]
            assert saved == []
        elif outcome == "valid_content_filter":
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert saved == []
        else:
            assert len(saved) == 1
    else:
        with pytest.raises(SanitizedProviderStreamError) as captured:
            await execute_non_stream_call(**kwargs)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert sentinel not in repr(captured.value)
        assert marked == []
        assert provider_manager.success_records == []
        assert not any(call[2] is True for call in metrics.llm_calls)
        assert provider_manager.fallback_requests == []
        assert saved == []
        usage_log.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["direct", "queued"])
async def test_invalid_mixed_result_with_usage_records_only_bounded_failure_accounting(
    monkeypatch: pytest.MonkeyPatch,
    queued: bool,
) -> None:
    """An in-band failure cannot bill tokens or emit any clean-success signal."""

    sentinel = "mixed-usage-result-secret-/srv/provider"
    result = _normal_nonstream_response("mixed_error_and_text", sentinel)
    result["usage"] = {
        "prompt_tokens": 41,
        "completion_tokens": 1,
        "total_tokens": 42,
    }
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    log_usage = AsyncMock(return_value=None)
    mark_used = AsyncMock(return_value=None)
    save_message = AsyncMock(return_value="must-not-save")
    queue = _ProcessorQueue() if queued else None

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", log_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "semantic-result-gate",
            "provider": "anthropic",
            "model": "claude-3",
            "streaming": False,
            "eligible": True,
        },
    )

    kwargs = _nonstream_call_kwargs(
        llm_call_func=lambda: result,
        provider_manager=provider_manager,
        metrics=metrics,
        refresh_provider_params=lambda _provider: ({}, None),
        queue_execution_enabled=queued,
        enable_provider_fallback=True,
        conversation_id=f"mixed-usage-{queued}",
    )
    kwargs.update(
        should_persist=True,
        save_message_fn=save_message,
        on_success=mark_used,
    )

    with pytest.raises(SanitizedProviderStreamError) as captured:
        await execute_non_stream_call(**kwargs)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)
    assert len(metrics.llm_calls) == 1
    assert metrics.llm_calls[0][0:3] == ("anthropic", "claude-3", False)
    assert metrics.llm_calls[0][3] == "SanitizedProviderStreamError"
    assert metrics.token_calls == []
    assert [call["outcome"] for call in metrics.completion_calls] == ["error"]
    assert metrics.fallback_successes == []
    assert len(provider_manager.failure_errors) == 1
    assert isinstance(provider_manager.failure_errors[0], SanitizedProviderStreamError)
    assert sentinel not in repr(provider_manager.failure_errors[0])
    assert provider_manager.failure_records == [
        ("anthropic", "SanitizedProviderStreamError")
    ]
    assert provider_manager.success_records == []
    assert provider_manager.fallback_requests == []
    log_usage.assert_not_awaited()
    mark_used.assert_not_awaited()
    save_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["direct", "queued"])
@pytest.mark.parametrize(
    ("invalid_outcome", "valid_outcome"),
    [
        ("mixed_error_and_text", "valid_text"),
        ("mixed_error_and_text", "valid_raw_text"),
        ("mixed_error_and_text", "nested_error_prefix"),
        ("mixed_error_and_text", "nested_canonical_error"),
        ("raw_done", "valid_refusal"),
        ("sse_done", "valid_content_filter"),
        ("sse_success", "valid_refusal"),
    ],
    ids=(
        "mixed-vs-text",
        "mixed-vs-raw",
        "mixed-vs-assistant-error-prefix",
        "mixed-vs-assistant-provider-code",
        "done-vs-refusal",
        "sse-done-vs-filter",
        "sse-vs-refusal",
    ),
)
async def test_concurrent_normal_adapter_results_are_request_isolated(
    monkeypatch: pytest.MonkeyPatch,
    queued: bool,
    invalid_outcome: str,
    valid_outcome: str,
) -> None:
    """An invalid overlapping result cannot borrow another request's success state."""

    sentinel = "concurrent-normal-result-secret-/srv/provider"
    ready = [threading.Event(), threading.Event()]
    release = threading.Event()
    queue = _ProcessorQueue() if queued else None
    metrics = [_DummyMetrics(), _DummyMetrics()]
    managers = [_DummyProviderManager(), _DummyProviderManager()]
    marked: list[list[str]] = [[], []]
    saved: list[list[dict[str, Any]]] = [[], []]
    usage_log = AsyncMock(return_value=None)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["stream"] is False
        index = 0 if headers["Authorization"] == "Bearer bad-primary-key" else 1
        ready[index].set()
        assert release.wait(timeout=2.0)
        if index == 0:
            return _normal_nonstream_response(invalid_outcome, sentinel)
        return _normal_nonstream_response(valid_outcome, sentinel)

    async def invoke(index: int) -> dict[str, Any]:
        async def mark_used(provider_name: str) -> None:
            marked[index].append(provider_name)

        async def save_message(
            _db: Any,
            _conversation_id: str,
            payload: dict[str, Any],
            *,
            use_transaction: bool,
        ) -> str:
            assert use_transaction is True
            saved[index].append(payload)
            return f"message-{index}"

        kwargs = _nonstream_call_kwargs(
            llm_call_func=lambda: _adapter_nonstream_call(
                "bad-primary-key" if index == 0 else "good-primary-key"
            ),
            provider_manager=managers[index],
            metrics=metrics[index],
            refresh_provider_params=lambda _provider: ({}, None),
            queue_execution_enabled=queued,
            enable_provider_fallback=True,
            conversation_id=(
                f"concurrent-normal-{invalid_outcome}-{valid_outcome}-{queued}-{index}"
            ),
            provider="openai",
            model="gpt-4o",
        )
        kwargs.update(
            should_persist=True,
            save_message_fn=save_message,
            on_success=mark_used,
        )
        return await execute_non_stream_call(**kwargs)

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "concurrent-primary-isolation",
            "provider": "openai",
            "model": "gpt-4o",
            "streaming": False,
            "eligible": True,
        },
    )

    tasks = [asyncio.create_task(invoke(index)) for index in range(2)]
    try:
        await asyncio.gather(*(_wait_for_thread_event(event) for event in ready))
        release.set()
        invalid_result, valid_result = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert isinstance(invalid_result, SanitizedProviderStreamError)
    assert sentinel not in repr(invalid_result)
    if valid_outcome == "valid_text":
        assert isinstance(valid_result, dict)
        assert valid_result["choices"][0]["message"]["content"] == "late provider success"
    elif valid_outcome == "valid_raw_text":
        assert valid_result == "late raw provider success"
    elif valid_outcome in {"nested_error_prefix", "nested_canonical_error"}:
        assert isinstance(valid_result, dict)
        assert valid_result["choices"][0]["message"]["content"]
    elif valid_outcome == "valid_refusal":
        assert isinstance(valid_result, dict)
        assert valid_result["choices"][0]["message"]["refusal"]
    else:
        assert isinstance(valid_result, dict)
        assert valid_result["choices"][0]["finish_reason"] == "content_filter"
    assert marked == [[], ["openai"]]
    assert managers[0].failure_records == [
        ("openai", "SanitizedProviderStreamError")
    ]
    assert managers[0].success_records == []
    assert managers[1].failure_records == []
    assert managers[1].success_records == ["openai"]
    assert managers[0].fallback_requests == []
    assert managers[1].fallback_requests == []
    assert saved[0] == []
    if valid_outcome in {
        "valid_text",
        "valid_raw_text",
        "nested_error_prefix",
        "nested_canonical_error",
    }:
        assert len(saved[1]) == 1
        expected_content = {
            "valid_text": "late provider success",
            "valid_raw_text": "late raw provider success",
            "nested_error_prefix": "Error: assistant-authored content",
            "nested_canonical_error": "provider_unavailable",
        }[valid_outcome]
        assert saved[1][0]["content"] == expected_content
    else:
        assert saved[1] == []
    assert metrics[0].llm_calls == [
        ("openai", "gpt-4o", False, "SanitizedProviderStreamError")
    ]
    assert metrics[1].llm_calls == [("openai", "gpt-4o", True, None)]
    assert metrics[0].token_calls == []
    assert len(metrics[1].token_calls) == 1
    assert metrics[1].token_calls[0]["provider"] == "openai"
    assert metrics[1].token_calls[0]["model"] == "gpt-4o"
    assert [call["outcome"] for call in metrics[0].completion_calls] == ["error"]
    assert [call["outcome"] for call in metrics[1].completion_calls] == ["success"]
    assert all(
        calls[0]["provider"] == "openai" and calls[0]["model"] == "gpt-4o"
        for calls in (metrics[0].completion_calls, metrics[1].completion_calls)
    )
    assert metrics[0].fallback_successes == []
    assert metrics[1].fallback_successes == []
    usage_log.assert_awaited_once()
    assert usage_log.await_args.kwargs["provider"] == "openai"
    assert usage_log.await_args.kwargs["model"] == "gpt-4o"
    assert usage_log.await_args.kwargs["conversation_id"] == (
        f"concurrent-normal-{invalid_outcome}-{valid_outcome}-{queued}-1"
    )
    if valid_outcome == "valid_raw_text":
        usage = usage_log.await_args.kwargs
        assert metrics[1].token_calls[0]["prompt_tokens"] == usage["prompt_tokens"]
        assert metrics[1].token_calls[0]["completion_tokens"] == (
            usage["completion_tokens"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_success"),
    [
        ("valid_text", True),
        ("valid_raw_text", True),
        ("valid_list_text", True),
        ("valid_image", True),
        ("valid_tool_calls", True),
        ("valid_function_call", True),
        ("valid_nested_noncanonical_error_json", True),
        ("valid_refusal", True),
        ("valid_content_filter", True),
        ("mixed_error_and_text", False),
        ("error", False),
        ("nested_error_prefix", True),
        ("nested_canonical_error", True),
        ("nested_structured_error_and_text", False),
        ("list_part_error_and_text", False),
        ("later_choice_error_and_text", False),
        ("serialized_error_envelope", False),
        ("raw_done", False),
        ("sse_done", False),
        ("sse_success", False),
    ],
)
async def test_normal_fallback_result_is_gated_before_success_accounting(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_success: bool,
) -> None:
    """A fallback must not become healthy, billable, or persisted before validation."""

    sentinel = "normal-fallback-result-secret-/srv/provider"
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    marked: list[str] = []
    saved: list[dict[str, Any]] = []
    adapter_keys: list[str] = []
    log_usage = AsyncMock(return_value=None)
    fallback_result = _normal_nonstream_response(outcome, sentinel)
    if isinstance(fallback_result, dict):
        fallback_result["usage"] = {
            "prompt_tokens": 17,
            "completion_tokens": 2,
            "total_tokens": 19,
        }
    primary_error = _certified_pre_dispatch(
        ChatProviderError(
            provider="anthropic",
            message="primary unavailable",
            status_code=502,
        )
    )

    def fail_primary() -> None:
        raise primary_error

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["stream"] is False
        assert headers["Authorization"] == "Bearer fallback-key"
        adapter_keys.append(headers["Authorization"].removeprefix("Bearer "))
        return fallback_result

    async def mark_used(provider_name: str) -> None:
        marked.append(provider_name)

    async def save_message(
        _db: Any,
        _conversation_id: str,
        payload: dict[str, Any],
        *,
        use_transaction: bool,
    ) -> str:
        assert use_transaction is True
        saved.append(payload)
        return "fallback-message"

    def refresh(provider_name: str) -> tuple[dict[str, Any], str]:
        return (
            {
                "api_endpoint": provider_name,
                "api_key": "fallback-key",
                "credentials_resolved": True,
                "app_config": _registry_openai_app_config(),
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", log_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)
    monkeypatch.setattr(chat_service, "should_force_normalize_string_responses", lambda: False)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "real-fallback-boundary",
            "provider": _kwargs["provider"],
            "model": _kwargs["model"],
            "streaming": False,
            "eligible": True,
        },
    )

    kwargs = _nonstream_call_kwargs(
        llm_call_func=fail_primary,
        provider_manager=provider_manager,
        metrics=metrics,
        refresh_provider_params=refresh,
        queue_execution_enabled=False,
        enable_provider_fallback=True,
        conversation_id=f"normal-fallback-{outcome}",
    )
    kwargs.update(
        should_persist=True,
        save_message_fn=save_message,
        on_success=mark_used,
    )

    if expected_success:
        response = await execute_non_stream_call(**kwargs)
        assert marked == ["openai"]
        assert provider_manager.failure_records == [
            ("anthropic", "SanitizedProviderStreamError")
        ]
        assert provider_manager.success_records == ["openai"]
        assert provider_manager.fallback_requests == [("anthropic",)]
        assert [call[0:3] for call in metrics.llm_calls] == [
            ("anthropic", "claude-3", False),
            ("openai", "gpt-4o", True),
        ]
        assert [call["outcome"] for call in metrics.completion_calls] == ["success"]
        if outcome == "valid_refusal":
            assert response["choices"][0]["message"]["refusal"]
            assert saved == []
        elif outcome == "valid_content_filter":
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert saved == []
        else:
            assert len(saved) == 1
        assert len(metrics.fallback_successes) == 1
        log_usage.assert_awaited_once()
        if outcome == "valid_raw_text":
            assert len(metrics.token_calls) == 1
            assert metrics.token_calls[0]["provider"] == "openai"
            assert metrics.token_calls[0]["model"] == "gpt-4o"
            usage = log_usage.await_args.kwargs
            assert usage["estimated"] is True
            assert usage["prompt_tokens"] >= 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )
            assert metrics.token_calls[0]["prompt_tokens"] == usage["prompt_tokens"]
            assert metrics.token_calls[0]["completion_tokens"] == (
                usage["completion_tokens"]
            )
    else:
        with pytest.raises(SanitizedProviderStreamError) as captured:
            await execute_non_stream_call(**kwargs)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert sentinel not in repr(captured.value)
        assert marked == []
        assert provider_manager.failure_records == [
            ("anthropic", "SanitizedProviderStreamError"),
            ("openai", "SanitizedProviderStreamError"),
        ]
        assert provider_manager.success_records == []
        assert provider_manager.fallback_requests == [("anthropic",)]
        assert saved == []
        assert metrics.fallback_successes == []
        assert [call[0:3] for call in metrics.llm_calls] == [
            ("anthropic", "claude-3", False),
            ("openai", "gpt-4o", False),
        ]
        assert sum(call[2] is False for call in metrics.llm_calls) == 2
        assert not any(call[2] is True for call in metrics.llm_calls)
        assert [call["outcome"] for call in metrics.completion_calls] == ["error"]
        assert metrics.token_calls == []
        log_usage.assert_not_awaited()

    assert adapter_keys == ["fallback-key"]


@pytest.mark.asyncio
async def test_concurrent_fallback_results_do_not_cross_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent fallback responses retain request-local validation and accounting."""

    sentinel = "concurrent-fallback-result-secret-/srv/provider"
    ready = [threading.Event(), threading.Event()]
    release = threading.Event()
    metrics = [_DummyMetrics(), _DummyMetrics()]
    managers = [_DummyProviderManager(), _DummyProviderManager()]
    marked: list[list[str]] = [[], []]
    saved: list[list[dict[str, Any]]] = [[], []]
    usage_log = AsyncMock(return_value=None)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["stream"] is False
        index = 0 if headers["Authorization"] == "Bearer bad-fallback-key" else 1
        ready[index].set()
        assert release.wait(timeout=2.0)
        if index == 0:
            return _normal_nonstream_response("mixed_error_and_text", sentinel)
        return _normal_nonstream_response("valid_text", sentinel)

    async def invoke(index: int) -> dict[str, Any]:
        primary_error = _certified_pre_dispatch(
            ChatProviderError(
                provider="anthropic",
                message="primary unavailable",
                status_code=502,
            )
        )

        def fail_primary() -> None:
            raise primary_error

        def refresh(provider_name: str) -> tuple[dict[str, Any], str]:
            label = "bad" if index == 0 else "good"
            return (
                {
                    "api_endpoint": provider_name,
                    "api_key": f"{label}-fallback-key",
                    "credentials_resolved": True,
                    "app_config": _registry_openai_app_config(),
                    "messages_payload": [],
                    "model": "gpt-4o",
                    "streaming": False,
                },
                "gpt-4o",
            )

        async def mark_used(provider_name: str) -> None:
            marked[index].append(provider_name)

        async def save_message(
            _db: Any,
            _conversation_id: str,
            payload: dict[str, Any],
            *,
            use_transaction: bool,
        ) -> str:
            assert use_transaction is True
            saved[index].append(payload)
            return f"fallback-message-{index}"

        kwargs = _nonstream_call_kwargs(
            llm_call_func=fail_primary,
            provider_manager=managers[index],
            metrics=metrics[index],
            refresh_provider_params=refresh,
            queue_execution_enabled=False,
            enable_provider_fallback=True,
            conversation_id=f"concurrent-fallback-{index}",
        )
        kwargs.update(
            should_persist=True,
            save_message_fn=save_message,
            on_success=mark_used,
        )
        return await execute_non_stream_call(**kwargs)

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)
    monkeypatch.setattr(
        chat_service,
        "_emit_chat_run_first_rollout_metrics",
        lambda *_args, **_kwargs: {
            "presentation_variant": "run_first",
            "cohort": "concurrent-fallback-isolation",
            "provider": _kwargs["provider"],
            "model": _kwargs["model"],
            "streaming": False,
            "eligible": True,
        },
    )

    tasks = [asyncio.create_task(invoke(index)) for index in range(2)]
    try:
        await asyncio.gather(*(_wait_for_thread_event(event) for event in ready))
        release.set()
        invalid_result, valid_result = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert isinstance(invalid_result, SanitizedProviderStreamError)
    assert sentinel not in repr(invalid_result)
    assert isinstance(valid_result, dict)
    assert valid_result["choices"][0]["message"]["content"] == "late provider success"
    assert marked == [[], ["openai"]]
    assert managers[0].failure_records == [
        ("anthropic", "SanitizedProviderStreamError"),
        ("openai", "SanitizedProviderStreamError"),
    ]
    assert managers[0].success_records == []
    assert managers[0].fallback_requests == [("anthropic",)]
    assert managers[1].failure_records == [
        ("anthropic", "SanitizedProviderStreamError")
    ]
    assert managers[1].success_records == ["openai"]
    assert managers[1].fallback_requests == [("anthropic",)]
    assert metrics[0].llm_calls == [
        ("anthropic", "claude-3", False, "ChatProviderError"),
        ("openai", "gpt-4o", False, "SanitizedProviderStreamError"),
    ]
    assert metrics[1].llm_calls == [
        ("anthropic", "claude-3", False, "ChatProviderError"),
        ("openai", "gpt-4o", True, None),
    ]
    assert metrics[0].token_calls == []
    assert len(metrics[1].token_calls) == 1
    assert metrics[1].token_calls[0]["provider"] == "openai"
    assert metrics[1].token_calls[0]["model"] == "gpt-4o"
    assert [call["outcome"] for call in metrics[0].completion_calls] == ["error"]
    assert [call["outcome"] for call in metrics[1].completion_calls] == ["success"]
    assert metrics[0].completion_calls[0]["provider"] == "anthropic"
    assert metrics[0].completion_calls[0]["model"] == "claude-3"
    assert metrics[1].completion_calls[0]["provider"] == "openai"
    assert metrics[1].completion_calls[0]["model"] == "gpt-4o"
    assert metrics[0].fallback_successes == []
    assert len(metrics[1].fallback_successes) == 1
    assert metrics[1].fallback_successes[0]["requested_provider"] == "anthropic"
    assert metrics[1].fallback_successes[0]["selected_provider"] == "openai"
    assert saved[0] == []
    assert len(saved[1]) == 1
    assert saved[1][0]["content"] == "late provider success"
    usage_log.assert_awaited_once()
    assert usage_log.await_args.kwargs["provider"] == "openai"
    assert usage_log.await_args.kwargs["model"] == "gpt-4o"
    assert usage_log.await_args.kwargs["conversation_id"] == "concurrent-fallback-1"


def _streaming_call_kwargs(
    *,
    llm_call_func,
    provider_manager,
    metrics,
    conversation_id,
    save_message_fn,
    audit_service,
    on_success=None,
    on_provider_output=None,
):
    return {
        "current_loop": asyncio.get_running_loop(),
        "cleaned_args": {
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        "selected_provider": "openai",
        "provider": "openai",
        "model": "gpt-test",
        "request_json": "{}",
        "request": SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/chat/completions"),
            headers={},
            state=SimpleNamespace(user_id=None, api_key_id=None),
        ),
        "metrics": metrics,
        "provider_manager": provider_manager,
        "templated_llm_payload": [],
        "should_persist": True,
        "final_conversation_id": conversation_id,
        "character_card_for_context": None,
        "chat_db": object(),
        "save_message_fn": save_message_fn,
        "audit_service": audit_service,
        "audit_context": object(),
        "client_id": conversation_id,
        "queue_execution_enabled": False,
        "enable_provider_fallback": False,
        "llm_call_func": llm_call_func,
        "refresh_provider_params": lambda _provider: ({}, None),
        "moderation_getter": lambda: _DummyModeration(),
        "on_success": on_success,
        "on_provider_output": on_provider_output,
        "provider_factory_timeout": 1.0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=("direct", "queued"))
async def test_real_runtime_retries_failed_credential_touch_before_nonstream_return(
    monkeypatch: pytest.MonkeyPatch,
    queued: bool,
) -> None:
    """A swallowed durable-touch failure remains retryable through Chat completion."""

    touch_attempts = 0

    async def touch() -> None:
        nonlocal touch_attempts
        touch_attempts += 1
        if touch_attempts == 1:
            raise RuntimeError("bounded first-touch failure")

    async def resolver(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=provider,
            api_key="runtime-touch-key",
            app_config=_registry_openai_app_config(),
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
            _touch_cb=touch,
        )

    runtime = ProviderCredentialRuntime(
        user_id=41,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=False,
        server_config_snapshot={},
        resolver=resolver,
    )
    handle = await runtime.resolve("openai", model="gpt-4o")
    queue = _ProcessorQueue() if queued else None
    usage_log = AsyncMock(return_value=None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)

    async def mark_used(provider_name: str) -> bool:
        assert provider_name == "openai"
        return await runtime.mark_used(handle)

    try:
        response = await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=lambda: _successful_nonstream_response("touch retry"),
                provider_manager=_DummyProviderManager(),
                metrics=_DummyMetrics(),
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=queued,
                enable_provider_fallback=False,
                conversation_id=f"real-runtime-touch-{queued}",
                provider="openai",
                model="gpt-4o",
            ),
            on_success=mark_used,
        )
        assert response["choices"][0]["message"]["content"] == "touch retry"
        assert touch_attempts == 2
        assert runtime._cache["openai"].used is True
        usage_log.assert_awaited_once()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_real_runtime_touch_retries_are_request_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed durable touch cannot certify or delay another request's mark."""

    touch_attempts = [0, 0]
    touch_ready = [asyncio.Event(), asyncio.Event()]
    release_touch = [asyncio.Event(), asyncio.Event()]
    runtimes: list[ProviderCredentialRuntime] = []
    handles: list[Any] = []

    for index in range(2):
        async def touch(index: int = index) -> None:
            touch_attempts[index] += 1
            if index == 0 and touch_attempts[index] == 1:
                raise RuntimeError("bounded first-touch failure")
            touch_ready[index].set()
            await release_touch[index].wait()

        async def resolver(
            provider: str,
            *,
            index: int = index,
            **_kwargs: Any,
        ) -> ResolvedByokCredentials:
            return ResolvedByokCredentials(
                provider=provider,
                api_key=f"runtime-concurrent-key-{index}",
                app_config=_registry_openai_app_config(),
                credential_fields={},
                source="user",
                allowlisted=True,
                status=ByokResolutionStatus.RESOLVED,
                auth_source="api_key",
                _touch_cb=touch,
            )

        runtime = ProviderCredentialRuntime(
            user_id=50 + index,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            server_config_snapshot={},
            resolver=resolver,
        )
        runtimes.append(runtime)
        handles.append(await runtime.resolve("openai", model="gpt-4o"))

    usage_log = AsyncMock(return_value=None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_run_legacy_tool_autoexec", lambda _args: False)

    async def invoke(index: int) -> dict[str, Any]:
        async def mark_used(provider_name: str) -> bool:
            assert provider_name == "openai"
            return await runtimes[index].mark_used(handles[index])

        return await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=lambda: _successful_nonstream_response(
                    f"runtime touch {index}"
                ),
                provider_manager=_DummyProviderManager(),
                metrics=_DummyMetrics(),
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=False,
                enable_provider_fallback=False,
                conversation_id=f"real-runtime-concurrent-{index}",
                provider="openai",
                model="gpt-4o",
            ),
            on_success=mark_used,
        )

    tasks = [asyncio.create_task(invoke(index)) for index in range(2)]
    try:
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), timeout=1.0) for event in touch_ready)
        )
        assert touch_attempts == [2, 1]
        assert all(task.done() is False for task in tasks)

        release_touch[1].set()
        healthy_response = await asyncio.wait_for(asyncio.shield(tasks[1]), timeout=1.0)
        assert healthy_response["choices"][0]["message"]["content"] == "runtime touch 1"
        assert runtimes[1]._cache["openai"].used is True
        assert runtimes[0]._cache["openai"].used is False

        release_touch[0].set()
        retried_response = await asyncio.wait_for(asyncio.shield(tasks[0]), timeout=1.0)
        assert retried_response["choices"][0]["message"]["content"] == "runtime touch 0"
    finally:
        for event in release_touch:
            event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(runtime.close() for runtime in runtimes))

    assert touch_attempts == [2, 1]
    assert all(runtime._closed for runtime in runtimes)
    usage_by_conversation = {
        call.kwargs["conversation_id"] for call in usage_log.await_args_list
    }
    assert usage_by_conversation == {
        "real-runtime-concurrent-0",
        "real-runtime-concurrent-1",
    }


@pytest.mark.asyncio
async def test_real_runtime_retries_failed_credential_touch_before_stream_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal stream callback retries a failed first-output credential touch."""

    touch_attempts = 0

    async def touch() -> None:
        nonlocal touch_attempts
        touch_attempts += 1
        if touch_attempts == 1:
            raise RuntimeError("bounded first-touch failure")

    async def resolver(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=provider,
            api_key="runtime-stream-touch-key",
            app_config=_registry_openai_app_config(),
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
            _touch_cb=touch,
        )

    runtime = ProviderCredentialRuntime(
        user_id=42,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=False,
        server_config_snapshot={},
        resolver=resolver,
    )
    handle = await runtime.resolve("openai", model="gpt-test")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    async def mark_used(provider_name: str) -> bool:
        assert provider_name == "openai"
        return await runtime.mark_used(handle)

    async def source():
        yield 'data: {"choices":[{"delta":{"content":"touch retry"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    try:
        response = await execute_streaming_call(
            **_streaming_call_kwargs(
                llm_call_func=source,
                provider_manager=_DummyProviderManager(),
                metrics=_DummyMetrics(),
                conversation_id="real-runtime-stream-touch",
                save_message_fn=AsyncMock(return_value="saved-stream-message"),
                audit_service=SimpleNamespace(log_event=AsyncMock(return_value=None)),
                on_provider_output=mark_used,
            )
        )
        wire = "".join([
            chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            async for chunk in response.body_iterator
        ])
        assert "touch retry" in wire
        assert touch_attempts == 2
        assert runtime._cache["openai"].used is True
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_execute_non_stream_call_normalizes_raw_string(monkeypatch):
    monkeypatch.setenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES", "1")

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def llm_call_func():
        return "plain response"

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
        metrics=metrics,
        provider_manager=provider_manager,
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
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(response, dict)
    assert response["choices"][0]["message"]["content"] == "plain response"
    assert response["tldw_conversation_id"] == "conv-123"


@pytest.mark.asyncio
async def test_execute_non_stream_call_refreshes_credentials(monkeypatch):
    captured_kwargs = {}

    async def fake_perform_chat_api_call_async(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "choices": [
                {"message": {"content": "fallback success"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_perform_chat_api_call_async)
    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    primary_error = _certified_pre_dispatch(
        ChatProviderError(provider="anthropic", message="primary failed", status_code=502)
    )

    def failing_llm_call():
        raise primary_error

    async def save_message_fn(*_args, **_kwargs):
        return None

    def refresh_provider(provider_name: str):
        assert provider_name == "openai"
        return (
            {
                "api_endpoint": "openai",
                "api_key": "fresh-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "anthropic",
            "api_key": "stale-key",
            "messages_payload": [],
            "model": "claude-3",
            "streaming": False,
        },
        selected_provider="anthropic",
        provider="anthropic",
        model="claude-3",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-123",
        character_card_for_context={},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="user-123",
        queue_execution_enabled=False,
        enable_provider_fallback=True,
        llm_call_func=failing_llm_call,
        refresh_provider_params=refresh_provider,
        moderation_getter=_DummyModeration,
    )

    assert captured_kwargs["api_endpoint"] == "openai"
    assert captured_kwargs["api_key"] == "fresh-key"
    assert captured_kwargs["model"] == "gpt-4o"
    assert response["tldw_conversation_id"] == "conv-123"
    assert provider_manager.fallback_requests == [("anthropic",)]
    assert provider_manager.success_records == ["openai"]
    assert any(
        entry.get("selected_provider") == "openai" and entry.get("streaming") is False
        for entry in metrics.fallback_successes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize(
    "failure_kind",
    [
        "ambiguous_chat_error",
        "http_exception",
        "runtime_error",
        "value_error",
        "unknown_error",
    ],
)
async def test_execute_non_stream_call_untrusted_failures_never_fallback_or_leak(
    monkeypatch,
    queued,
    failure_kind,
):
    class UnknownAdapterFailure(Exception):
        pass

    sentinel = f"nonstream-{failure_kind}-secret-/srv/provider"
    primary_error = {
        "ambiguous_chat_error": ChatProviderError(
            provider="anthropic",
            message=sentinel,
            status_code=502,
        ),
        "http_exception": HTTPException(status_code=503, detail=sentinel),
        "runtime_error": RuntimeError(sentinel),
        "value_error": ValueError(sentinel),
        "unknown_error": UnknownAdapterFailure(sentinel),
    }[failure_kind]
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: _ProcessorQueue() if queued else None)

    def failing_llm_call():
        raise primary_error

    with pytest.raises(SanitizedProviderStreamError) as captured:
        await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=failing_llm_call,
                provider_manager=provider_manager,
                metrics=metrics,
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=queued,
            )
        )

    assert provider_manager.fallback_requests == []
    assert provider_manager.failure_errors
    assert all(isinstance(error, SanitizedProviderStreamError) for error in provider_manager.failure_errors)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in str(captured.value)
    assert sentinel not in "".join(str(error) for error in provider_manager.failure_errors)


@pytest.mark.asyncio
async def test_execute_non_stream_sync_provider_cancellation_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sync adapter cancellation is a provider failure, not caller cancellation."""

    sentinel = "direct-nonstream-cancel-secret-/srv/provider"
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    def cancelled_call() -> None:
        raise asyncio.CancelledError(sentinel)

    with pytest.raises(SanitizedProviderStreamError) as captured:
        await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=cancelled_call,
                provider_manager=provider_manager,
                metrics=metrics,
                refresh_provider_params=lambda _provider: ({}, None),
            )
        )

    assert captured.value.code == "provider_unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)
    assert provider_manager.fallback_requests == []
    assert provider_manager.failure_errors
    assert sentinel not in "".join(
        repr(error) for error in provider_manager.failure_errors
    )


@pytest.mark.asyncio
async def test_execute_non_stream_queued_provider_cancellation_keeps_worker_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider cancellation must not terminate the real queue's sole worker."""

    sentinel = "queued-nonstream-cancel-secret-/srv/provider"
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=1)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    def cancelled_call() -> None:
        raise asyncio.CancelledError(sentinel)

    def healthy_call() -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {"content": "healthy queued response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    try:
        with pytest.raises(SanitizedProviderStreamError) as captured:
            await asyncio.wait_for(
                execute_non_stream_call(
                    **_nonstream_call_kwargs(
                        llm_call_func=cancelled_call,
                        provider_manager=provider_manager,
                        metrics=metrics,
                        refresh_provider_params=lambda _provider: ({}, None),
                        queue_execution_enabled=True,
                        conversation_id="queued-provider-cancelled",
                    )
                ),
                timeout=1.0,
            )

        async def wait_for_request_cleanup() -> None:
            while queue._active_request_ids:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_request_cleanup(), timeout=1.0)
        assert queue.is_running()

        healthy_response = await asyncio.wait_for(
            execute_non_stream_call(
                **_nonstream_call_kwargs(
                    llm_call_func=healthy_call,
                    provider_manager=provider_manager,
                    metrics=metrics,
                    refresh_provider_params=lambda _provider: ({}, None),
                    queue_execution_enabled=True,
                    conversation_id="healthy-after-queued-provider-cancel",
                )
            ),
            timeout=1.0,
        )
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert captured.value.code == "provider_unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)
    assert healthy_response["choices"][0]["message"]["content"] == (
        "healthy queued response"
    )
    assert provider_manager.fallback_requests == []
    assert sentinel not in "".join(
        repr(error) for error in provider_manager.failure_errors
    )


@pytest.mark.asyncio
async def test_execute_non_stream_async_fallback_provider_cancellation_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An async fallback adapter's own cancellation remains a bounded failure."""

    sentinel = "async-fallback-cancel-secret-/srv/provider"
    primary_error = _certified_pre_dispatch(
        ChatProviderError(provider="anthropic", message="primary failed", status_code=502)
    )
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    class CancelledFallbackAdapter:
        async_chat_is_native = True

        async def achat(self, _request: dict[str, object]) -> None:
            raise asyncio.CancelledError(sentinel)

    registry = SimpleNamespace(
        get_adapter=lambda _provider: CancelledFallbackAdapter()
    )
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    def failing_primary_call() -> None:
        raise primary_error

    def refresh_provider(provider: str):
        return (
            {
                "api_endpoint": provider,
                "api_key": "fallback-key",
                "credentials_resolved": True,
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    with pytest.raises(SanitizedProviderStreamError) as captured:
        await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=failing_primary_call,
                provider_manager=provider_manager,
                metrics=metrics,
                refresh_provider_params=refresh_provider,
            )
        )

    assert captured.value.code == "provider_unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)
    assert provider_manager.fallback_requests == [("anthropic",)]
    assert len(provider_manager.failure_errors) == 2
    assert sentinel not in "".join(
        repr(error) for error in provider_manager.failure_errors
    )


@pytest.mark.asyncio
async def test_execute_non_stream_cancellation_drains_provider_before_usage_and_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot release credential ownership before sync work exits."""

    entered = threading.Event()
    release = threading.Event()
    marked: list[str] = []
    provider_manager = _DummyProviderManager()
    pool = BoundedDaemonPool(capacity=1)
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)

    def blocked_call() -> dict[str, object]:
        entered.set()
        release.wait()
        return {
            "choices": [
                {
                    "message": {"content": "late successful response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def mark_used(provider: str) -> None:
        marked.append(provider)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=blocked_call,
        provider_manager=provider_manager,
        metrics=_DummyMetrics(),
        refresh_provider_params=lambda _provider: ({}, None),
    )
    kwargs["on_success"] = mark_used
    task = asyncio.create_task(
        execute_non_stream_call(**kwargs)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert marked == []
        assert pool.active_count == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert provider_manager.failure_errors == []
    assert provider_manager.fallback_requests == []
    assert marked == ["anthropic"]
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_execute_non_stream_starts_without_default_executor_and_drains_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct nonqueue work starts now, never later from the default-executor queue."""

    loop = asyncio.get_running_loop()
    previous_default_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    provider_entered = threading.Event()
    provider_release = threading.Event()
    lifecycle: list[str] = []
    provider_starts = 0
    pool = _ReleaseTrackingDaemonPool(capacity=1, lifecycle=lifecycle)
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    task: asyncio.Task[dict[str, object]] | None = None

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    def blocked_call() -> dict[str, object]:
        nonlocal provider_starts
        provider_starts += 1
        lifecycle.append("provider-start")
        provider_entered.set()
        assert provider_release.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        return {
            "choices": [{"message": {"content": "late success"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def mark_used(_provider: str) -> None:
        lifecycle.append("mark-used")

    loop.set_default_executor(default_executor)
    default_blocker = loop.run_in_executor(None, block_default_executor)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
    try:
        await _wait_for_thread_event(default_entered)
        kwargs = _nonstream_call_kwargs(
            llm_call_func=blocked_call,
            provider_manager=_DummyProviderManager(),
            metrics=_DummyMetrics(),
            refresh_provider_params=lambda _provider: ({}, None),
        )
        kwargs["on_success"] = mark_used
        task = asyncio.create_task(execute_non_stream_call(**kwargs))

        await _wait_for_thread_event(provider_entered)
        assert not default_release.is_set()
        assert pool.active_count == 1

        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert lifecycle == ["provider-start"]

        provider_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert lifecycle == [
            "provider-start",
            "provider-exit",
            "capacity-release",
            "mark-used",
        ]
        assert pool.release_count == 1
        assert pool.active_count == 0
    finally:
        provider_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        replacement_executor = previous_default_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        default_executor.shutdown(wait=True, cancel_futures=True)

    await asyncio.sleep(0)
    assert provider_starts == 1


@pytest.mark.asyncio
async def test_execute_non_stream_saturation_fails_closed_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct Chat work cannot queue credential-bearing calls without a bound."""

    entered = threading.Event()
    release = threading.Event()
    extra_dispatched = threading.Event()
    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)

    def blocked_call() -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=1.0)
        return {
            "choices": [{"message": {"content": "first"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def excess_call() -> dict[str, object]:
        extra_dispatched.set()
        return {
            "choices": [{"message": {"content": "excess"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    first = asyncio.create_task(
        execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=blocked_call,
                provider_manager=_DummyProviderManager(),
                metrics=_DummyMetrics(),
                refresh_provider_params=lambda _provider: ({}, None),
            )
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await execute_non_stream_call(
                **_nonstream_call_kwargs(
                    llm_call_func=excess_call,
                    provider_manager=_DummyProviderManager(),
                    metrics=_DummyMetrics(),
                    refresh_provider_params=lambda _provider: ({}, None),
                )
            )
        assert exc_info.value.code == "provider_unavailable"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert not extra_dispatched.is_set()
        assert pool.active_count == 1
    finally:
        release.set()
        await asyncio.wait_for(first, timeout=1.0)
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("holder_mode", ["queued", "direct"])
async def test_real_queue_and_direct_chat_share_one_sync_adapter_capacity_cap(
    monkeypatch: pytest.MonkeyPatch,
    holder_mode: str,
) -> None:
    """Queued and direct Chat calls must share one process-wide adapter cap."""

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import (
        OpenAIAdapter,
    )

    holder_entered = threading.Event()
    release_holder = threading.Event()
    rejected_entered = threading.Event()
    adapter_keys: list[str] = []
    lifecycle: list[str] = []
    pool = _ReleaseTrackingDaemonPool(capacity=1, lifecycle=lifecycle)
    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=1.0)
    adapter = OpenAIAdapter()
    holder_task: asyncio.Task[dict[str, object]] | None = None

    def adapter_chat(request: dict[str, object]) -> dict[str, object]:
        api_key = str(request["api_key"])
        adapter_keys.append(api_key)
        if api_key == "holder-key":
            lifecycle.append("holder-enter")
            holder_entered.set()
            assert release_holder.wait(timeout=2.0)
            lifecycle.append("holder-exit")
        elif api_key == "rejected-key":
            rejected_entered.set()
        return _successful_nonstream_response(f"response-for-{api_key}")

    async def invoke(*, api_key: str, queued: bool, conversation_id: str):
        return await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=lambda: _adapter_nonstream_call(api_key),
                provider_manager=_DummyProviderManager(),
                metrics=_DummyMetrics(),
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=queued,
                enable_provider_fallback=False,
                conversation_id=conversation_id,
            )
        )

    monkeypatch.setattr(adapter, "chat", adapter_chat)
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        request_queue_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    await queue.start(num_workers=1)
    try:
        holder_task = asyncio.create_task(
            invoke(
                api_key="holder-key",
                queued=holder_mode == "queued",
                conversation_id=f"{holder_mode}-capacity-holder",
            )
        )
        await _wait_for_thread_event(holder_entered)
        assert pool.active_count == 1

        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await invoke(
                api_key="rejected-key",
                queued=holder_mode == "direct",
                conversation_id=f"{holder_mode}-capacity-rejected",
            )

        assert exc_info.value.code == "provider_unavailable"
        assert exc_info.value.status_code == 502
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert rejected_entered.is_set() is False
        assert adapter_keys == ["holder-key"]
        assert pool.active_count == 1

        release_holder.set()
        holder_response = await asyncio.wait_for(holder_task, timeout=1.0)
        assert holder_response["choices"][0]["message"]["content"] == (
            "response-for-holder-key"
        )
        assert pool.active_count == 0

        async def wait_for_queue_idle() -> None:
            while queue._active_request_ids:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_queue_idle(), timeout=1.0)
        recovery_response = await invoke(
            api_key="recovery-key",
            queued=holder_mode == "direct",
            conversation_id=f"{holder_mode}-capacity-recovery",
        )
        assert recovery_response["choices"][0]["message"]["content"] == (
            "response-for-recovery-key"
        )
        assert adapter_keys == ["holder-key", "recovery-key"]
        assert pool.active_count == 0
    finally:
        release_holder.set()
        if holder_task is not None and not holder_task.done():
            holder_task.cancel()
        if holder_task is not None:
            await asyncio.gather(holder_task, return_exceptions=True)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert lifecycle.count("capacity-release") == 2
    assert lifecycle.index("holder-exit") < lifecycle.index("capacity-release")


@pytest.mark.asyncio
async def test_queued_non_stream_cancellation_drains_worker_and_releases_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled queue waiter retains ownership until its provider worker exits."""
    entered = threading.Event()
    release = threading.Event()
    marked: list[str] = []
    queue = RequestQueue(max_queue_size=2, max_concurrent=1, timeout=1.0)
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    await queue.start(num_workers=1)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    def blocked_call() -> dict[str, object]:
        entered.set()
        release.wait()
        return {
            "choices": [
                {
                    "message": {"content": "queued late success"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def mark_used(provider: str) -> None:
        marked.append(provider)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=blocked_call,
        provider_manager=_DummyProviderManager(),
        metrics=_DummyMetrics(),
        refresh_provider_params=lambda _provider: ({}, None),
        queue_execution_enabled=True,
    )
    kwargs["on_success"] = mark_used
    task = asyncio.create_task(execute_non_stream_call(**kwargs))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert queue._active_request_ids
        assert marked == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        async def wait_for_capacity_release() -> None:
            while queue._active_request_ids:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_capacity_release(), timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert marked == ["anthropic"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_text", 1),
        ("valid_raw_text", 1),
        ("empty", 0),
        ("error", 0),
        ("error_prefix", 0),
        ("canonical_raw_code", 0),
        ("sse_error_envelope", 0),
        ("serialized_error_envelope", 0),
        ("malformed_tool_calls", 0),
        ("malformed_function_call", 0),
    ],
)
async def test_real_queued_chat_cancellation_classifies_late_adapter_result_before_close(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Cancellation drains queued adapter work and marks only usable output."""

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import (
        OpenAIAdapter,
    )

    sentinel = "queued-late-adapter-secret-/srv/provider"
    adapter_entered = threading.Event()
    release_adapter = threading.Event()
    lifecycle: list[str] = []
    pool = _ReleaseTrackingDaemonPool(capacity=1, lifecycle=lifecycle)
    queue = RequestQueue(max_queue_size=2, max_concurrent=1, timeout=1.0)
    adapter = OpenAIAdapter()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)

    def adapter_chat(_request: dict[str, object]) -> Any:
        lifecycle.append("adapter-enter")
        adapter_entered.set()
        assert release_adapter.wait(timeout=2.0)
        lifecycle.append("adapter-exit")
        return _late_nonstream_response(late_outcome, sentinel)

    async def mark_used(provider: str) -> None:
        lifecycle.append(f"mark-used:{provider}")

    async def invoke_with_runtime() -> dict[str, object]:
        try:
            kwargs = _nonstream_call_kwargs(
                llm_call_func=lambda: _adapter_nonstream_call("late-key"),
                provider_manager=_DummyProviderManager(),
                metrics=_DummyMetrics(),
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=True,
                enable_provider_fallback=False,
                conversation_id=f"queued-late-{late_outcome}",
            )
            kwargs["on_success"] = mark_used
            return await execute_non_stream_call(**kwargs)
        finally:
            lifecycle.append("runtime-close")

    monkeypatch.setattr(adapter, "chat", adapter_chat)
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        request_queue_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    await queue.start(num_workers=1)
    task = asyncio.create_task(invoke_with_runtime())
    try:
        await _wait_for_thread_event(adapter_entered)
        assert pool.active_count == 1
        task.cancel()
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert "runtime-close" not in lifecycle
        assert not any(event.startswith("mark-used:") for event in lifecycle)

        release_adapter.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release_adapter.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert pool.active_count == 0
    assert lifecycle.count("capacity-release") == 1
    assert sum(event.startswith("mark-used:") for event in lifecycle) == expected_marks
    assert lifecycle.index("adapter-exit") < lifecycle.index("capacity-release")
    assert lifecycle[-1] == "runtime-close"
    if expected_marks:
        assert lifecycle.index("capacity-release") < lifecycle.index("mark-used:anthropic")
        assert lifecycle.index("mark-used:anthropic") < lifecycle.index("runtime-close")
    assert sentinel not in "".join(lifecycle)


@pytest.mark.asyncio
async def test_real_queued_chat_stop_drains_adapter_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping a queue cannot orphan admitted credential-bearing adapter work."""

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import (
        OpenAIAdapter,
    )

    adapter_entered = threading.Event()
    release_adapter = threading.Event()
    lifecycle: list[str] = []
    pool = _ReleaseTrackingDaemonPool(capacity=1, lifecycle=lifecycle)
    queue = RequestQueue(max_queue_size=2, max_concurrent=1, timeout=1.0)
    adapter = OpenAIAdapter()
    stop_task: asyncio.Task[None] | None = None
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)

    def adapter_chat(_request: dict[str, object]) -> dict[str, object]:
        lifecycle.append("adapter-enter")
        adapter_entered.set()
        assert release_adapter.wait(timeout=2.0)
        lifecycle.append("adapter-exit")
        return _successful_nonstream_response("completed during queue stop")

    async def invoke_with_runtime() -> dict[str, object]:
        try:
            return await execute_non_stream_call(
                **_nonstream_call_kwargs(
                    llm_call_func=lambda: _adapter_nonstream_call("stop-key"),
                    provider_manager=_DummyProviderManager(),
                    metrics=_DummyMetrics(),
                    refresh_provider_params=lambda _provider: ({}, None),
                    queue_execution_enabled=True,
                    enable_provider_fallback=False,
                    conversation_id="queued-stop-drains-adapter",
                )
            )
        finally:
            lifecycle.append("runtime-close")

    monkeypatch.setattr(adapter, "chat", adapter_chat)
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        request_queue_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    await queue.start(num_workers=1)
    request_task = asyncio.create_task(invoke_with_runtime())
    safety_release = threading.Timer(0.25, release_adapter.set)
    try:
        await _wait_for_thread_event(adapter_entered)
        assert pool.active_count == 1
        safety_release.start()
        stop_task = asyncio.create_task(queue.stop())
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert stop_task.done() is False
        assert request_task.done() is False
        assert "runtime-close" not in lifecycle
        assert pool.active_count == 1

        release_adapter.set()
        await asyncio.wait_for(stop_task, timeout=1.0)
        await asyncio.gather(request_task, return_exceptions=True)
    finally:
        release_adapter.set()
        safety_release.cancel()
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        if queue.is_running():
            await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert pool.active_count == 0
    assert lifecycle.count("adapter-enter") == 1
    assert lifecycle.count("adapter-exit") == 1
    assert lifecycle.count("capacity-release") == 1
    assert lifecycle.index("adapter-exit") < lifecycle.index("capacity-release")
    assert lifecycle.index("capacity-release") < lifecycle.index("runtime-close")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_text", 1),
        ("valid_raw_text", 1),
        ("valid_tool_calls", 1),
        ("valid_function_call", 1),
        ("empty", 0),
        ("error", 0),
        ("error_prefix", 0),
        ("canonical_raw_code", 0),
        ("sse_error_envelope", 0),
        ("serialized_error_envelope", 0),
        ("malformed_tool_calls", 0),
        ("malformed_function_call", 0),
    ],
)
async def test_fallback_cancellation_classifies_late_result_before_marking_success(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Fallback cancellation marks only a late result with usable output."""

    entered = threading.Event()
    release = threading.Event()
    marked: list[str] = []
    sentinel = "fallback-late-result-secret-/srv/provider"
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    primary_error = _certified_pre_dispatch(
        ChatProviderError(
            provider="anthropic",
            message="primary failed",
            status_code=502,
        )
    )
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    def failing_primary() -> None:
        raise primary_error

    async def blocking_fallback(**_kwargs):
        def invoke_sync_adapter() -> Any:
            entered.set()
            release.wait()
            return _late_nonstream_response(late_outcome, sentinel)

        return await asyncio.to_thread(invoke_sync_adapter)

    async def mark_used(provider: str) -> None:
        marked.append(provider)

    def refresh(provider: str):
        return (
            {
                "api_endpoint": provider,
                "api_key": "fallback-key",
                "credentials_resolved": True,
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        blocking_fallback,
    )
    kwargs = _nonstream_call_kwargs(
        llm_call_func=failing_primary,
        provider_manager=_DummyProviderManager(),
        metrics=_DummyMetrics(),
        refresh_provider_params=refresh,
    )
    kwargs["on_success"] = mark_used
    task = asyncio.create_task(execute_non_stream_call(**kwargs))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert marked == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert marked == ["openai"] * expected_marks


@pytest.mark.asyncio
async def test_success_is_marked_before_cancelled_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed provider call is accounted before persistence can be cancelled."""

    save_entered = asyncio.Event()
    release_save = asyncio.Event()
    marked: list[str] = []
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    def successful_call() -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {"content": "provider completed"},
                    "finish_reason": "stop",
                }
            ]
        }

    async def save_message(*_args, **_kwargs) -> None:
        save_entered.set()
        await release_save.wait()

    async def mark_used(provider: str) -> None:
        marked.append(provider)

    kwargs = _nonstream_call_kwargs(
        llm_call_func=successful_call,
        provider_manager=_DummyProviderManager(),
        metrics=_DummyMetrics(),
        refresh_provider_params=lambda _provider: ({}, None),
    )
    kwargs.update(
        should_persist=True,
        chat_db=object(),
        character_card_for_context={"name": "Test"},
        save_message_fn=save_message,
        on_success=mark_used,
    )
    task = asyncio.create_task(execute_non_stream_call(**kwargs))
    try:
        await asyncio.wait_for(save_entered.wait(), timeout=1.0)
        assert marked == ["anthropic"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_save.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_perform_async_chat_genuine_caller_cancellation_stays_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real cancel request must pass through the async adapter boundary."""

    entered = asyncio.Event()
    release = asyncio.Event()
    cancellation_observed = asyncio.Event()

    class BlockingAdapter:
        async_chat_is_native = True

        async def achat(self, _request: dict[str, object]) -> None:
            entered.set()
            try:
                await release.wait()
            finally:
                cancellation_observed.set()

    registry = SimpleNamespace(get_adapter=lambda _provider: BlockingAdapter())
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    task = asyncio.create_task(
        chat_service.perform_chat_api_call_async(
            api_endpoint="openai",
            api_key="resolved-key",
            credentials_resolved=True,
            messages_payload=[],
            model="gpt-4o",
            streaming=False,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1.0)
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_perform_async_chat_requires_explicit_native_async_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmarked adapter cannot bypass bounded sync-call admission."""

    calls: list[str] = []
    pool = BoundedDaemonPool(capacity=1)

    class UnmarkedAdapter:
        def chat(self, _request: dict[str, object]) -> dict[str, object]:
            calls.append("sync")
            return {"choices": [{"message": {"content": "bounded"}}]}

        async def achat(self, _request: dict[str, object]) -> dict[str, object]:
            calls.append("async")
            return {"choices": [{"message": {"content": "unbounded"}}]}

    registry = SimpleNamespace(get_adapter=lambda _provider: UnmarkedAdapter())
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)

    result = await chat_service.perform_chat_api_call_async(
        api_endpoint="openai",
        api_key="resolved-key",
        credentials_resolved=True,
        messages_payload=[],
        model="gpt-4o",
        streaming=False,
    )

    assert calls == ["sync"]
    assert result["choices"][0]["message"]["content"] == "bounded"
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_perform_async_chat_cancellation_drains_sync_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled sync fallback cannot outlive its adapter-call scope."""

    entered = threading.Event()
    release = threading.Event()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)

    class BlockingSyncAdapter:
        async def achat(self, _request: dict[str, object]) -> None:
            raise NotImplementedError

        def chat(self, _request: dict[str, object]) -> dict[str, object]:
            entered.set()
            assert release.wait(timeout=1.0)
            return {"choices": [{"message": {"content": "completed"}}]}

    registry = SimpleNamespace(get_adapter=lambda _provider: BlockingSyncAdapter())
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    task = asyncio.create_task(
        chat_service.perform_chat_api_call_async(
            api_endpoint="openai",
            api_key="resolved-key",
            credentials_resolved=True,
            messages_payload=[],
            model="gpt-4o",
            streaming=False,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
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


@pytest.mark.asyncio
async def test_bounded_sync_call_holds_lease_until_cancelled_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primitive releases admission only after real sync work is gone."""

    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    pool = BoundedDaemonPool(capacity=1)
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)

    def blocking_call() -> str:
        entered.set()
        assert release.wait(timeout=1.0)
        exited.set()
        return "completed"

    task = asyncio.create_task(
        bounded_daemon_module.await_bounded_sync_call(
            blocking_call,
            pool=pool,
            exhaustion_message="test capacity exhausted",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        assert pool.active_count == 1
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert pool.active_count == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert exited.is_set()
        assert pool.active_count == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_perform_async_chat_starts_without_default_executor_and_drains_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter boundary dispatches directly and owns the worker through exit."""

    loop = asyncio.get_running_loop()
    previous_default_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    adapter_entered = threading.Event()
    adapter_release = threading.Event()
    lifecycle: list[str] = []
    adapter_starts = 0
    pool = _ReleaseTrackingDaemonPool(capacity=1, lifecycle=lifecycle)
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    task: asyncio.Task[dict[str, object]] | None = None

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    class BlockingAdapter:
        async_chat_is_native = False

        def chat(self, _request: dict[str, object]) -> dict[str, object]:
            nonlocal adapter_starts
            adapter_starts += 1
            lifecycle.append("adapter-start")
            adapter_entered.set()
            assert adapter_release.wait(timeout=2.0)
            lifecycle.append("adapter-exit")
            return {"choices": [{"message": {"content": "completed"}}]}

    registry = SimpleNamespace(get_adapter=lambda _provider: BlockingAdapter())
    loop.set_default_executor(default_executor)
    default_blocker = loop.run_in_executor(None, block_default_executor)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
    try:
        await _wait_for_thread_event(default_entered)
        task = asyncio.create_task(
            chat_service.perform_chat_api_call_async(
                api_endpoint="openai",
                api_key="resolved-key",
                credentials_resolved=True,
                messages_payload=[],
                model="gpt-4o",
                streaming=False,
            )
        )

        await _wait_for_thread_event(adapter_entered)
        assert not default_release.is_set()
        assert pool.active_count == 1

        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert lifecycle == ["adapter-start"]

        adapter_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert lifecycle == ["adapter-start", "adapter-exit", "capacity-release"]
        assert pool.release_count == 1
        assert pool.active_count == 0
    finally:
        adapter_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        replacement_executor = previous_default_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        default_executor.shutdown(wait=True, cancel_futures=True)

    await asyncio.sleep(0)
    assert adapter_starts == 1


@pytest.mark.asyncio
async def test_perform_async_chat_cancellation_drains_builtin_sync_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Built-in async wrappers cannot outlive their credential-bearing call."""

    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    pool = BoundedDaemonPool(capacity=1)
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)

    class BlockingBuiltInAdapter:
        async_chat_is_native = False

        def chat(self, _request: dict[str, object]) -> dict[str, object]:
            entered.set()
            assert release.wait(timeout=1.0)
            exited.set()
            return {"choices": [{"message": {"content": "completed"}}]}

        async def achat(self, request: dict[str, object]) -> dict[str, object]:
            return await asyncio.to_thread(self.chat, request)

    registry = SimpleNamespace(get_adapter=lambda _provider: BlockingBuiltInAdapter())
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(
        chat_service,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    task = asyncio.create_task(
        chat_service.perform_chat_api_call_async(
            api_endpoint="openai",
            api_key="resolved-key",
            credentials_resolved=True,
            messages_payload=[],
            model="gpt-4o",
            streaming=False,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert pool.active_count == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert exited.is_set()
        assert pool.active_count == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_perform_async_chat_saturation_fails_closed_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded adapter lane rejects excess credential-bearing work."""

    entered = threading.Event()
    release = threading.Event()
    extra_dispatched = threading.Event()
    pool = BoundedDaemonPool(capacity=1)

    class SaturatedBuiltInAdapter:
        async_chat_is_native = False

        def chat(self, request: dict[str, object]) -> dict[str, object]:
            if request["api_key"] == "first-key":
                entered.set()
                assert release.wait(timeout=1.0)
            else:
                extra_dispatched.set()
            return {"choices": [{"message": {"content": "completed"}}]}

        async def achat(self, request: dict[str, object]) -> dict[str, object]:
            return await asyncio.to_thread(self.chat, request)

    registry = SimpleNamespace(get_adapter=lambda _provider: SaturatedBuiltInAdapter())
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(
        chat_service,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    first = asyncio.create_task(
        chat_service.perform_chat_api_call_async(
            api_endpoint="openai",
            api_key="first-key",
            credentials_resolved=True,
            messages_payload=[],
            model="gpt-4o",
            streaming=False,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await chat_service.perform_chat_api_call_async(
                api_endpoint="openai",
                api_key="second-key",
                credentials_resolved=True,
                messages_payload=[],
                model="gpt-4o",
                streaming=False,
            )
        assert exc_info.value.code == "provider_unavailable"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert not extra_dispatched.is_set()
        assert pool.active_count == 1
    finally:
        release.set()
        await asyncio.wait_for(first, timeout=1.0)
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_execute_non_stream_call_preserves_sanitized_provider_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    queued: bool,
) -> None:
    """A typed adapter rate limit must remain retryable without retaining detail."""

    sentinel = f"rate-limit-{'queued' if queued else 'direct'}-secret-/srv/provider"
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    monkeypatch.setattr(
        chat_service,
        "get_request_queue",
        lambda: _ProcessorQueue() if queued else None,
    )

    def rate_limited_call() -> None:
        raise ChatRateLimitError(provider="openai", message=sentinel)

    with pytest.raises(ChatRateLimitError) as captured:
        await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=rate_limited_call,
                provider_manager=provider_manager,
                metrics=metrics,
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=queued,
            )
        )

    assert captured.value.status_code == 429
    assert str(captured.value) == "Rate limit exceeded with the chat provider."
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)
    assert provider_manager.fallback_requests == []
    assert sentinel not in "".join(str(error) for error in provider_manager.failure_errors)


def test_chat_exception_compatibility_reexports_preserve_identity():
    from tldw_Server_API.app.core import exceptions as core_exceptions
    from tldw_Server_API.app.core.Chat import Chat_Deps

    assert Chat_Deps.ChatAPIError is core_exceptions.ChatAPIError
    assert Chat_Deps.ChatAuthenticationError is core_exceptions.ChatAuthenticationError
    assert Chat_Deps.ChatConfigurationError is core_exceptions.ChatConfigurationError
    assert Chat_Deps.ChatProviderError is core_exceptions.ChatProviderError
    assert (
        Chat_Deps.SanitizedProviderStreamError
        is core_exceptions.SanitizedProviderStreamError
    )

    with pytest.raises(core_exceptions.ChatAPIError):
        raise Chat_Deps.ChatAuthenticationError(provider="openai")


@pytest.mark.asyncio
async def test_execute_non_stream_call_queue_fallback_requires_exact_certificate(monkeypatch):
    async def fallback_call(**_kwargs):
        return {
            "choices": [{"message": {"content": "queued fallback"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fallback_call)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: _ProcessorQueue())
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    primary_error = _certified_pre_dispatch(
        ChatProviderError(provider="anthropic", message="bounded primary", status_code=502)
    )
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def failing_llm_call():
        raise primary_error

    response = await execute_non_stream_call(
        **_nonstream_call_kwargs(
            llm_call_func=failing_llm_call,
            provider_manager=provider_manager,
            metrics=metrics,
            refresh_provider_params=lambda provider: (
                {
                    "api_endpoint": provider,
                    "api_key": "fallback-key",
                    "messages_payload": [],
                    "model": "gpt-4o",
                    "streaming": False,
                },
                "gpt-4o",
            ),
            queue_execution_enabled=True,
        )
    )

    assert response["choices"][0]["message"]["content"] == "queued fallback"
    assert provider_manager.fallback_requests == [("anthropic",)]
    assert all(isinstance(error, SanitizedProviderStreamError) for error in provider_manager.failure_errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["refresh", "fallback_call"])
async def test_execute_non_stream_call_bounds_fallback_http_failures(
    monkeypatch,
    failure_phase,
):
    sentinel = f"fallback-{failure_phase}-secret-/srv/provider"
    primary_error = _certified_pre_dispatch(
        ChatProviderError(provider="anthropic", message="bounded primary", status_code=502)
    )
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    async def fallback_call(**_kwargs):
        raise HTTPException(status_code=503, detail=sentinel)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fallback_call)

    def failing_llm_call():
        raise primary_error

    def refresh_provider(provider):
        if failure_phase == "refresh":
            raise HTTPException(status_code=503, detail=sentinel)
        return (
            {
                "api_endpoint": provider,
                "api_key": "fallback-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    with pytest.raises(SanitizedProviderStreamError) as captured:
        await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=failing_llm_call,
                provider_manager=provider_manager,
                metrics=metrics,
                refresh_provider_params=refresh_provider,
            )
        )

    assert sentinel not in str(captured.value)
    assert len(provider_manager.failure_errors) == 2
    assert all(isinstance(error, SanitizedProviderStreamError) for error in provider_manager.failure_errors)
    assert sentinel not in "".join(str(error) for error in provider_manager.failure_errors)


@pytest.mark.asyncio
async def test_concurrent_non_stream_replay_certificates_remain_isolated(monkeypatch):
    ready_ambiguous = threading.Event()
    ready_certified = threading.Event()
    release = threading.Event()
    fallback_calls: list[str] = []

    async def fallback_call(**kwargs):
        fallback_calls.append(kwargs["api_key"])
        return {
            "choices": [{"message": {"content": "certified fallback"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fallback_call)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    ambiguous_error = ChatProviderError("ambiguous secret", status_code=502, provider="anthropic")
    certified_error = _certified_pre_dispatch(
        ChatProviderError("bounded certified", status_code=502, provider="anthropic")
    )
    ambiguous_manager = _DummyProviderManager()
    certified_manager = _DummyProviderManager()

    def gated_failure(error, ready):
        ready.set()
        release.wait()
        raise error

    def refresh(provider):
        return (
            {
                "api_endpoint": provider,
                "api_key": "isolated-fallback-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )
    ambiguous_task = asyncio.create_task(
        execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=lambda: gated_failure(ambiguous_error, ready_ambiguous),
                provider_manager=ambiguous_manager,
                metrics=_DummyMetrics(),
                refresh_provider_params=refresh,
                conversation_id="conv-ambiguous",
            )
        )
    )
    certified_task = asyncio.create_task(
        execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=lambda: gated_failure(certified_error, ready_certified),
                provider_manager=certified_manager,
                metrics=_DummyMetrics(),
                refresh_provider_params=refresh,
                conversation_id="conv-certified",
            )
        )
    )
    assert await asyncio.to_thread(ready_ambiguous.wait, 1.0)
    assert await asyncio.to_thread(ready_certified.wait, 1.0)
    release.set()
    ambiguous_result, certified_result = await asyncio.gather(
        ambiguous_task,
        certified_task,
        return_exceptions=True,
    )

    assert isinstance(ambiguous_result, SanitizedProviderStreamError)
    assert ambiguous_result.code == "provider_unavailable"
    assert str(ambiguous_result) == "The chat service provider is currently unavailable."
    assert "ambiguous secret" not in str(ambiguous_result)
    assert ambiguous_result.__cause__ is None
    assert ambiguous_result.__context__ is None
    assert isinstance(certified_result, dict)
    assert certified_result["choices"][0]["message"]["content"] == "certified fallback"
    assert ambiguous_manager.fallback_requests == []
    assert certified_manager.fallback_requests == [("anthropic",)]
    assert fallback_calls == ["isolated-fallback-key"]
    assert all(
        isinstance(error, SanitizedProviderStreamError)
        for error in (*ambiguous_manager.failure_errors, *certified_manager.failure_errors)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "terminal_error",
        "expected_type",
        "expected_code",
        "expected_message",
        "expected_status",
    ),
    [
        (
            ChatAuthenticationError(
                "provider rejected credentials: raw-auth-sentinel",
                provider="anthropic",
            ),
            ChatAuthenticationError,
            None,
            "The selected provider credentials could not be authenticated.",
            401,
        ),
        (
            ChatAuthenticationError(
                "provider forbidden credentials: raw-forbidden-sentinel",
                provider="anthropic",
                status_code=403,
            ),
            ChatAuthenticationError,
            None,
            "The selected provider credentials could not be authenticated.",
            403,
        ),
        (
            ChatConfigurationError(
                "provider configuration invalid: raw-config-sentinel",
                provider="anthropic",
            ),
            ChatConfigurationError,
            None,
            "The selected provider configuration is invalid.",
            500,
        ),
    ],
)
async def test_execute_non_stream_call_never_falls_back_for_terminal_provider_errors(
    monkeypatch,
    terminal_error,
    expected_type,
    expected_code,
    expected_message,
    expected_status,
):
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def failing_llm_call():
        raise terminal_error

    async def save_message_fn(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    with pytest.raises(expected_type) as captured:
        await execute_non_stream_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "anthropic",
                "api_key": "runtime-key",
                "credentials_resolved": True,
                "messages_payload": [],
                "model": "claude-3",
                "streaming": False,
            },
            selected_provider="anthropic",
            provider="anthropic",
            model="claude-3",
            request_json="{}",
            request=request,
            metrics=metrics,
            provider_manager=provider_manager,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id="conv-terminal",
            character_card_for_context={},
            chat_db=None,
            save_message_fn=save_message_fn,
            audit_service=None,
            audit_context=None,
            client_id="user-terminal",
            queue_execution_enabled=False,
            enable_provider_fallback=True,
            llm_call_func=failing_llm_call,
            refresh_provider_params=lambda *_args, **_kwargs: None,
            moderation_getter=_DummyModeration,
        )

    assert captured.value is not terminal_error
    if expected_code is not None:
        assert captured.value.code == expected_code
    assert str(captured.value) == expected_message
    assert captured.value.status_code == expected_status
    assert str(terminal_error) not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert provider_manager.fallback_requests == []


@pytest.mark.asyncio
async def test_execute_non_stream_call_attaches_continuation_metadata_and_parent_id(monkeypatch):
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    metrics = _DummyMetrics()
    save_payloads: list[dict[str, object]] = []

    def llm_call_func():
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "continued"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    async def save_message_fn(*_args, **_kwargs):
        payload = _args[2] if len(_args) > 2 else _kwargs.get("payload", {})
        if isinstance(payload, dict):
            save_payloads.append(payload)
        return "msg-cont-1"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    continuation_meta = {
        "applied": True,
        "mode": "branch",
        "from_message_id": "anchor-msg-1",
    }

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "continue"}],
            "model": "gpt-4o-mini",
            "streaming": False,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "continue"}],
        should_persist=True,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _DummyModeration(),
        assistant_parent_message_id="anchor-msg-1",
        continuation_metadata=continuation_meta,
    )

    assert save_payloads
    assert save_payloads[0]["parent_message_id"] == "anchor-msg-1"
    assert response["tldw_continuation"] == continuation_meta
    assert response["tldw_message_id"] == "msg-cont-1"


@pytest.mark.asyncio
async def test_execute_streaming_call_sanitizes_http_exception(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    http_exc = HTTPException(status_code=429, detail="Rate limited")

    def failing_llm_call():

        raise http_exc

    async def save_message_fn(*_args, **_kwargs):
        return None

    # Disable queue path to exercise direct streaming behavior
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-test",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-test",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=failing_llm_call,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(resp, StreamingResponse)

    # Consume the StreamingResponse body iterator and validate error payload + DONE
    agen = resp.body_iterator
    chunks = []
    try:
        for _ in range(4):
            try:
                ln = await agen.__anext__()
            except StopAsyncIteration:
                break
            if not ln:
                continue
            chunks.append(ln)
    finally:
        try:
            await agen.aclose()
        except Exception:
            _ = None

    # Normalize to str for assertions
    chunks = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in chunks]
    joined = "".join(chunks)
    assert any("\"error\"" in c for c in chunks), f"No error frame in chunks: {chunks}"
    assert '"code": "provider_unavailable"' in joined
    assert "The chat service provider is currently unavailable." in joined
    assert "HTTPException" not in joined
    assert "Rate limited" not in joined
    assert chunks and chunks[-1].strip() == "data: [DONE]"

    # The last llm call recorded should indicate an HTTPException error type
    assert metrics.llm_calls[-1][3] in ("HTTPException", "HTTPException")


@pytest.mark.asyncio
async def test_execute_streaming_call_sync_close_never_blocks_loop_or_leaks(
    monkeypatch,
):
    sentinel = "execute-sync-close-secret-/srv/provider"
    next_started = threading.Event()
    release_next = threading.Event()
    close_started = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()
    loop_responsive = asyncio.Event()

    class BlockingCloseIterator:
        def __init__(self):
            self.yielded = False
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            if not self.yielded:
                self.yielded = True
                return "safe execute output"
            next_started.set()
            release_next.wait()
            raise StopIteration

        def close(self):
            self.close_calls += 1
            close_started.set()
            release_close.wait()
            close_finished.set()
            raise RuntimeError(sentinel)

    monkeypatch.setenv("STREAMS_UNIFIED", "0")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    iterator = BlockingCloseIterator()
    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/chat/completions"),
            headers={},
            state=SimpleNamespace(user_id=None, api_key_id=None),
        ),
        metrics=_DummyMetrics(),
        provider_manager=_DummyProviderManager(),
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-execute-blocking-close",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=_noop_save_message,
        audit_service=None,
        audit_context=None,
        client_id="client-execute-blocking-close",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: iterator,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
    )

    body = response.body_iterator
    while True:
        chunk = await body.__anext__()
        if "safe execute output" in str(chunk):
            break
    assert await asyncio.to_thread(next_started.wait, 1.0)

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    close_task = asyncio.create_task(body.aclose())
    try:
        release_next.set()
        assert await asyncio.to_thread(close_started.wait, 1.0)
        asyncio.get_running_loop().call_soon(loop_responsive.set)
        await asyncio.wait_for(loop_responsive.wait(), timeout=1.0)
        await asyncio.wait_for(close_task, timeout=1.0)
        assert not release_close.is_set()
        assert not close_finished.is_set()
    finally:
        release_next.set()
        release_close.set()
        assert await asyncio.to_thread(close_finished.wait, 1.0)
        if not close_task.done():
            await asyncio.wait_for(close_task, timeout=1.0)
        logger.remove(sink_id)

    assert iterator.close_calls == 1
    assert sentinel not in "".join(logs)


@pytest.mark.asyncio
async def test_unified_stream_disconnect_bounds_full_queue_resistant_producer(
    monkeypatch,
):
    """Disconnect cleanup must not await a cancellation-resistant producer."""

    from tldw_Server_API.app.core.Streaming import streams as streams_module

    third_put_started = asyncio.Event()
    producer_cancelled = asyncio.Event()
    release_producer = asyncio.Event()
    send_calls = 0
    original_send = streams_module.SSEStream.send_raw_sse_line

    async def resistant_send(self, line):
        nonlocal send_calls
        send_calls += 1
        if send_calls != 3:
            await original_send(self, line)
            return
        third_put_started.set()
        try:
            await original_send(self, line)
        except asyncio.CancelledError:
            producer_cancelled.set()
            await release_producer.wait()
            raise

    async def three_frames():
        for index in range(3):
            yield f'data: {{"index": {index}}}\n\n'

    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("STREAM_QUEUE_MAXSIZE", "1")
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_S", "60")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(
        chat_service,
        "create_streaming_response_with_timeout",
        lambda **_kwargs: three_frames(),
    )
    monkeypatch.setattr(
        streams_module.SSEStream,
        "send_raw_sse_line",
        resistant_send,
    )

    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/chat/completions"),
            headers={},
            state=SimpleNamespace(user_id=None, api_key_id=None),
        ),
        metrics=_DummyMetrics(),
        provider_manager=_DummyProviderManager(),
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-unified-full-disconnect",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=_noop_save_message,
        audit_service=None,
        audit_context=None,
        client_id="client-unified-full-disconnect",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: iter(()),
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
    )

    body = response.body_iterator
    disconnect_task = None
    try:
        first_frame = await asyncio.wait_for(body.__anext__(), timeout=1.0)
        assert str(first_frame).startswith("data:")
        await asyncio.wait_for(third_put_started.wait(), timeout=1.0)
        disconnect_task = asyncio.create_task(body.athrow(asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(disconnect_task), timeout=0.5)
        assert producer_cancelled.is_set()
        assert not release_producer.is_set()
    finally:
        release_producer.set()
        if disconnect_task is not None and not disconnect_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(disconnect_task, timeout=1.0)
        with contextlib.suppress(Exception):
            await body.aclose()


@pytest.mark.asyncio
async def test_eager_stream_factory_deadline_is_bounded_and_isolated(
    monkeypatch,
):
    release = threading.Event()
    blocked_started = threading.Event()
    healthy_done = asyncio.Event()
    ticked = asyncio.Event()

    def blocking_factory():
        blocked_started.set()
        release.wait()
        return iter(["late blocked output"])

    def healthy_factory():
        return iter(["healthy factory output"])

    async def ticker():
        await asyncio.sleep(0)
        ticked.set()

    def call(factory, conversation_id):
        return execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "messages_payload": [],
                "model": "gpt-test",
                "streaming": True,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=_DummyProviderManager(),
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id=conversation_id,
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id=conversation_id,
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=factory,
            refresh_provider_params=lambda _provider: ({}, None),
            moderation_getter=lambda: _DummyModeration(),
        )

    async def consume(response):
        wire = "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )
        healthy_done.set()
        return wire

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(
        chat_service,
        "PROVIDER_STREAM_FACTORY_TIMEOUT_SECONDS",
        0.02,
        raising=False,
    )
    blocked_task = asyncio.create_task(call(blocking_factory, "conv_factory_blocked"))
    assert await asyncio.to_thread(blocked_started.wait, 1.0)
    healthy_task = asyncio.create_task(call(healthy_factory, "conv_factory_healthy"))
    tick_task = asyncio.create_task(ticker())
    try:
        healthy_response = await asyncio.wait_for(healthy_task, timeout=1.0)
        healthy_wire = await asyncio.wait_for(
            consume(healthy_response),
            timeout=1.0,
        )
        await asyncio.wait_for(tick_task, timeout=1.0)
        blocked_response = await asyncio.wait_for(blocked_task, timeout=1.0)
        blocked_wire = await asyncio.wait_for(
            consume(blocked_response),
            timeout=1.0,
        )
        assert not release.is_set()
    finally:
        release.set()

    assert ticked.is_set() and healthy_done.is_set()
    assert '"code": "provider_unavailable"' in blocked_wire
    assert "late blocked output" not in blocked_wire
    assert "healthy factory output" in healthy_wire


@pytest.mark.asyncio
async def test_provider_stream_daemon_capacity_fails_closed_without_extra_work(
    monkeypatch,
):
    _install_slow_stream_preflight(monkeypatch)
    release = threading.Event()
    blocked_started = threading.Event()
    blocked_finished = threading.Event()
    extra_started = threading.Event()

    def blocked_factory():
        blocked_started.set()
        release.wait()
        blocked_finished.set()
        return None

    def extra_factory():
        extra_started.set()
        return iter(["capacity bypass output"])

    def call(factory, conversation_id):
        return execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "messages_payload": [],
                "model": "gpt-test",
                "streaming": True,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id=conversation_id,
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id=conversation_id,
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=factory,
            refresh_provider_params=lambda _provider: ({}, None),
            moderation_getter=lambda: _DummyModeration(),
            provider_factory_timeout=0.02,
        )

    async def consume(response):
        return "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    try:
        blocked_task = asyncio.create_task(call(blocked_factory, "capacity-blocked"))
        assert await asyncio.to_thread(blocked_started.wait, 1.0)
        blocked_response = await asyncio.wait_for(blocked_task, timeout=1.0)
        blocked_wire = await asyncio.wait_for(consume(blocked_response), timeout=1.0)

        rejected_response = await asyncio.wait_for(
            call(extra_factory, "capacity-rejected"),
            timeout=1.0,
        )
        rejected_wire = await asyncio.wait_for(consume(rejected_response), timeout=1.0)

        assert pool.active_count == 1
        assert not extra_started.is_set()
        assert not release.is_set()
        assert '"code": "provider_unavailable"' in blocked_wire
        assert '"code": "provider_unavailable"' in rejected_wire
        assert "capacity bypass output" not in rejected_wire
    finally:
        release.set()
        assert await asyncio.to_thread(blocked_finished.wait, 1.0)


@pytest.mark.asyncio
async def test_capacity_one_factory_releases_before_sync_bridge_starts(monkeypatch):
    target_finished = threading.Event()
    allow_release = threading.Event()
    iterator_started = threading.Event()

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target():
                target()
                target_finished.set()
                allow_release.wait()

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )

    class OneChunkIterator:
        def __init__(self):
            self.sent = False

        def __iter__(self):
            return self

        def __next__(self):
            iterator_started.set()
            if self.sent:
                raise StopIteration
            self.sent = True
            return "capacity-one bridge output"

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(
        bounded_daemon_module,
        "STREAM_DAEMON_POOL",
        GatedReleasePool(capacity=1),
    )

    response_task = asyncio.create_task(
        execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "messages_payload": [],
                "model": "gpt-test",
                "streaming": True,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id="capacity-one-handoff",
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id="capacity-one-handoff",
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=OneChunkIterator,
            refresh_provider_params=lambda _provider: ({}, None),
            moderation_getter=lambda: _DummyModeration(),
            provider_factory_timeout=1.0,
        )
    )
    try:
        assert await asyncio.to_thread(target_finished.wait, 1.0)
        await asyncio.sleep(0)
        assert not response_task.done()
        assert not iterator_started.is_set()

        allow_release.set()
        response = await asyncio.wait_for(response_task, timeout=1.0)
        wire = "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )
    finally:
        allow_release.set()
        if not response_task.done():
            response_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await response_task

    assert iterator_started.is_set()
    assert "capacity-one bridge output" in wire


@pytest.mark.asyncio
async def test_cancelled_direct_stream_factory_returns_promptly_and_keeps_lease(
    monkeypatch,
):
    factory_entered = threading.Event()
    release_factory = threading.Event()
    factory_exited = threading.Event()
    loop_responsive = asyncio.Event()
    task_finished = asyncio.Event()
    pool = BoundedDaemonPool(capacity=1)

    def blocking_factory():
        factory_entered.set()
        release_factory.wait()
        factory_exited.set()
        return iter(())

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    task = asyncio.create_task(
        execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "messages_payload": [],
                "model": "gpt-test",
                "streaming": True,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id="cancelled-blocked-direct-factory",
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id="cancelled-blocked-direct-factory",
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=blocking_factory,
            refresh_provider_params=lambda _provider: ({}, None),
            moderation_getter=lambda: _DummyModeration(),
            provider_factory_timeout=5.0,
        )
    )
    task.add_done_callback(lambda _task: task_finished.set())
    try:
        assert await asyncio.to_thread(factory_entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(task_finished.wait(), timeout=1.0)
        cancelled_promptly = task.cancelled()
        leased_while_blocked = pool.active_count

        asyncio.get_running_loop().call_soon(loop_responsive.set)
        await asyncio.wait_for(loop_responsive.wait(), timeout=0.5)
    finally:
        release_factory.set()
        assert await asyncio.to_thread(factory_exited.wait, 1.0)

    async def wait_for_pool_release():
        while pool.active_count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_pool_release(), timeout=1.0)

    healthy = await chat_service._call_stream_factory_bounded(
        lambda: iter(("healthy-after-cancel",)),
        timeout=1.0,
    )

    assert cancelled_promptly
    assert task.cancelled()
    assert leased_while_blocked == 1
    assert list(healthy) == ["healthy-after-cancel"]


@pytest.mark.asyncio
async def test_zero_factory_budget_does_not_start_sync_or_async_work():
    async_started = threading.Event()
    sync_started = threading.Event()

    with pytest.raises(asyncio.TimeoutError):
        await chat_service._call_stream_factory_bounded(
            async_started.set,
            timeout=0,
        )
    with pytest.raises(TimeoutError):
        chat_service._call_stream_factory_from_isolated_worker(
            sync_started.set,
            lambda: 0.0,
        )

    assert not async_started.is_set()
    assert not sync_started.is_set()


@pytest.mark.asyncio
async def test_factory_exception_releases_capacity_for_healthy_followup(monkeypatch):
    target_finished = threading.Event()
    allow_release = threading.Event()

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target():
                target()
                target_finished.set()
                allow_release.wait()

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )

    pool = GatedReleasePool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    healthy_started = threading.Event()

    def failing_factory():
        raise ChatProviderError(
            provider="openai",
            message="private upstream failure",
            status_code=502,
        )

    failing_task = asyncio.create_task(
        chat_service._call_stream_factory_bounded(
            failing_factory,
            timeout=1.0,
        )
    )
    try:
        assert await asyncio.to_thread(target_finished.wait, 1.0)
        await asyncio.sleep(0)
        assert not failing_task.done()
        assert pool.active_count == 1
        allow_release.set()
        with pytest.raises(ChatProviderError):
            await asyncio.wait_for(failing_task, timeout=1.0)
    finally:
        allow_release.set()
        if not failing_task.done():
            with pytest.raises(ChatProviderError):
                await asyncio.wait_for(failing_task, timeout=1.0)

    result = await chat_service._call_stream_factory_bounded(
        lambda: (healthy_started.set(), iter(("healthy",)))[1],
        timeout=1.0,
    )

    assert healthy_started.is_set()
    assert list(result) == ["healthy"]
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_direct_sync_factory_cancellation_is_sanitized_provider_failure():
    sentinel = "direct-sync-factory-cancel-secret-/srv/provider"

    def cancelled_factory():
        raise asyncio.CancelledError(sentinel)

    factory_task = asyncio.create_task(
        chat_service._call_stream_factory_bounded(
            cancelled_factory,
            timeout=1.0,
        )
    )
    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await asyncio.wait_for(asyncio.shield(factory_task), timeout=1.0)

    assert not factory_task.cancelled()
    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    healthy = await chat_service._call_stream_factory_bounded(
        lambda: iter(("healthy-direct-output",)),
        timeout=1.0,
    )
    assert list(healthy) == ["healthy-direct-output"]


@pytest.mark.asyncio
async def test_factory_handoff_timeout_closes_delivered_stream_exactly_once(
    monkeypatch,
):
    target_finished = threading.Event()
    allow_release = threading.Event()
    closed = threading.Event()
    close_calls = 0

    class ControlledClock:
        now = 0.0

        def monotonic(self):
            return self.now

    class GatedReleasePool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            def gated_target():
                target()
                target_finished.set()
                allow_release.wait()

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )

    class CloseCountingStream:
        def close(self):
            nonlocal close_calls
            close_calls += 1
            closed.set()

    clock = ControlledClock()
    pool = GatedReleasePool(capacity=1)
    real_time = chat_service.time
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    monkeypatch.setattr(
        chat_service,
        "time",
        SimpleNamespace(time=real_time.time, monotonic=clock.monotonic),
    )
    task = asyncio.create_task(
        chat_service._call_stream_factory_bounded(
            CloseCountingStream,
            timeout=1.0,
        )
    )
    try:
        assert await asyncio.to_thread(target_finished.wait, 1.0)
        clock.now = 2.0
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=1.0)
        assert close_calls == 0
        allow_release.set()
        assert await asyncio.to_thread(closed.wait, 1.0)
    finally:
        allow_release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def wait_for_pool_release():
        while pool.active_count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_pool_release(), timeout=1.0)
    assert close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_direct_factory_handoff_uses_reserved_cleanup_after_normal_reacquire(
    monkeypatch,
):
    normal_reacquired = threading.Event()
    release_normal = threading.Event()
    release_cleanup_handoff = threading.Event()
    closed = threading.Event()
    close_calls = 0

    class ReacquiringPool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            if name != "provider-stream-factory":
                return super().start(
                    target,
                    name=name,
                    released_event=released_event,
                )

            worker_released = threading.Event()
            thread = super().start(
                target,
                name=name,
                released_event=worker_released,
            )

            def reacquire_before_handoff():
                worker_released.wait()
                super(ReacquiringPool, self).start(
                    release_normal.wait,
                    name="normal-capacity-reacquired",
                )
                normal_reacquired.set()
                release_cleanup_handoff.wait()
                if released_event is not None:
                    released_event.set()

            threading.Thread(
                target=reacquire_before_handoff,
                name="direct-handoff-controller",
                daemon=True,
            ).start()
            return thread

    class CloseCountingStream:
        def close(self):
            nonlocal close_calls
            close_calls += 1
            closed.set()

    normal_pool = ReacquiringPool(capacity=1)
    cleanup_pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", normal_pool)
    monkeypatch.setattr(
        bounded_daemon_module,
        "STREAM_CLEANUP_DAEMON_POOL",
        cleanup_pool,
    )
    task = asyncio.create_task(
        chat_service._call_stream_factory_bounded(
            CloseCountingStream,
            timeout=1.0,
        )
    )
    try:
        assert await asyncio.to_thread(normal_reacquired.wait, 1.0)
        assert normal_pool.active_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_cleanup_handoff.set()
        assert await asyncio.to_thread(closed.wait, 1.0)
    finally:
        release_cleanup_handoff.set()
        release_normal.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert close_calls == 1


@pytest.mark.asyncio
async def test_direct_fallback_refresh_and_factory_share_absolute_deadline(monkeypatch):
    class ControlledClock:
        def __init__(self):
            self.now = 50.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    clock = ControlledClock()
    factory_timeouts = []
    await_timeouts = []
    provider_manager = _DummyProviderManager()

    def initial_factory():
        clock.advance(0.02)
        raise _certified_pre_dispatch(
            ChatProviderError("first failed", status_code=502, provider="anthropic")
        )

    async def refresh_provider(_provider):
        clock.advance(0.02)
        return (
            {
                "api_endpoint": "openai",
                "messages_payload": [],
                "model": "gpt-test",
                "streaming": True,
            },
            "gpt-test",
        )

    def fallback_factory(**_kwargs):
        clock.advance(0.03)
        return iter(("late fallback output",))

    async def controlled_factory(factory, *, timeout=None):
        factory_timeouts.append(timeout)
        before = clock()
        value = factory()
        if timeout is not None and clock() - before > timeout:
            raise asyncio.TimeoutError
        return value

    async def controlled_await(awaitable, timeout=None):
        await_timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "_provider_factory_monotonic", clock)
    monkeypatch.setattr(chat_service, "_call_stream_factory_bounded", controlled_factory)
    monkeypatch.setattr(chat_service, "await_stream_operation_bounded", controlled_await)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", fallback_factory)

    with pytest.raises(asyncio.TimeoutError):
        await execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "anthropic",
                "messages_payload": [],
                "model": "claude-test",
                "streaming": True,
            },
            selected_provider="anthropic",
            provider="anthropic",
            model="claude-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=provider_manager,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id="direct-deadline",
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id="direct-deadline",
            queue_execution_enabled=False,
            enable_provider_fallback=True,
            llm_call_func=initial_factory,
            refresh_provider_params=refresh_provider,
            moderation_getter=lambda: _DummyModeration(),
            provider_factory_timeout=0.06,
        )

    assert factory_timeouts == pytest.approx([0.06, 0.04, 0.02])
    assert await_timeouts == pytest.approx([0.04])


@pytest.mark.asyncio
async def test_hung_async_fallback_refresh_cannot_escape_factory_deadline(monkeypatch):
    _install_slow_stream_preflight(monkeypatch)
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    fallback_started = threading.Event()

    def initial_factory():
        raise _certified_pre_dispatch(
            ChatProviderError("first failed", status_code=502, provider="anthropic")
        )

    async def hung_refresh(_provider):
        refresh_started.set()
        await release_refresh.wait()
        return ({"messages_payload": []}, "gpt-test")

    def fallback_factory(**_kwargs):
        fallback_started.set()
        return iter(())

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", fallback_factory)
    task = asyncio.create_task(
        execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "anthropic",
                "messages_payload": [],
                "model": "claude-test",
                "streaming": True,
            },
            selected_provider="anthropic",
            provider="anthropic",
            model="claude-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=_DummyProviderManager(),
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id="hung-refresh",
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id="hung-refresh",
            queue_execution_enabled=False,
            enable_provider_fallback=True,
            llm_call_func=initial_factory,
            refresh_provider_params=hung_refresh,
            moderation_getter=lambda: _DummyModeration(),
            provider_factory_timeout=0.02,
        )
    )
    try:
        await asyncio.wait_for(refresh_started.wait(), timeout=1.0)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=1.0)
        assert not release_refresh.is_set()
        assert not fallback_started.is_set()
    finally:
        release_refresh.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_queued_stream_factory_timeout_does_not_exhaust_queue_or_block_shutdown(
    monkeypatch,
):
    _install_slow_stream_preflight(monkeypatch)
    release = threading.Event()
    blocked_started = threading.Event()
    queue = RequestQueue(max_queue_size=10, max_concurrent=1, timeout=1.0)
    await queue.start(num_workers=2)

    def blocked_factory():
        blocked_started.set()
        release.wait()
        return iter(["late queued output"])

    def healthy_factory():
        return iter(["healthy queued output"])

    def call(factory, conversation_id):
        return execute_streaming_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "messages_payload": [],
                "model": "gpt-test",
                "streaming": True,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-test",
            request_json="{}",
            request=SimpleNamespace(
                method="POST",
                url=SimpleNamespace(path="/api/v1/chat/completions"),
                headers={},
                state=SimpleNamespace(user_id=None, api_key_id=None),
            ),
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id=conversation_id,
            character_card_for_context=None,
            chat_db=None,
            save_message_fn=_noop_save_message,
            audit_service=None,
            audit_context=None,
            client_id=conversation_id,
            queue_execution_enabled=True,
            enable_provider_fallback=False,
            llm_call_func=factory,
            refresh_provider_params=lambda _provider: ({}, None),
            moderation_getter=lambda: _DummyModeration(),
            queue_request_id=f"queued-factory:{conversation_id}",
            provider_factory_timeout=0.03 if conversation_id == "blocked" else 1.0,
        )

    async def consume(response):
        return "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    assert chat_service.PROVIDER_STREAM_FACTORY_TIMEOUT_SECONDS > 1.0
    monkeypatch.setattr(chat_service, "CHAT_IDLE_TIMEOUT", 0.5, raising=False)
    stop_task = None
    try:
        blocked_response = await call(blocked_factory, "blocked")
        assert await asyncio.to_thread(blocked_started.wait, 1.0)
        healthy_response = await call(healthy_factory, "healthy")
        blocked_wire, healthy_wire = await asyncio.wait_for(
            asyncio.gather(consume(blocked_response), consume(healthy_response)),
            timeout=1.0,
        )
        assert not release.is_set()
        stop_task = asyncio.create_task(queue.stop())
        await asyncio.wait_for(asyncio.shield(stop_task), timeout=1.0)
        assert not release.is_set()
    finally:
        release.set()
        if stop_task is not None and not stop_task.done():
            await asyncio.wait_for(stop_task, timeout=1.0)
        elif not queue._stopped:
            await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert '"code": "provider_unavailable"' in blocked_wire
    assert "late queued output" not in blocked_wire
    assert "healthy queued output" in healthy_wire


@pytest.mark.asyncio
async def test_queued_factory_budget_starts_at_dispatch_after_normal_queue_wait(
    monkeypatch,
):
    class ControlledClock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = ControlledClock()
    release_blocker = threading.Event()
    blocker_started = threading.Event()
    target_started = threading.Event()
    real_time = request_queue_module.time
    queue = RequestQueue(max_queue_size=4, max_concurrent=1, timeout=10.0)
    await queue.start(num_workers=1)

    def blocker():
        blocker_started.set()
        release_blocker.wait()
        return "released"

    def target_factory():
        target_started.set()
        return iter((
            'data: {"choices":[{"delta":{"content":"after-queue-wait"}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ))

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "_provider_factory_monotonic", clock)
    monkeypatch.setattr(
        request_queue_module,
        "time",
        SimpleNamespace(time=real_time.time, monotonic=clock),
    )
    blocker_future = await queue.enqueue(
        request_id="factory-deadline-blocker",
        request_data={},
        client_id="blocker",
        processor=blocker,
    )
    assert await asyncio.to_thread(blocker_started.wait, 1.0)

    call_kwargs = _streaming_call_kwargs(
        llm_call_func=target_factory,
        provider_manager=_DummyProviderManager(),
        metrics=_DummyMetrics(),
        conversation_id="factory-deadline-after-queue-wait",
        save_message_fn=AsyncMock(return_value=None),
        audit_service=SimpleNamespace(log_event=AsyncMock(return_value=None)),
        on_success=AsyncMock(),
    )
    call_kwargs.update(
        queue_execution_enabled=True,
        queue_request_id="factory-deadline-after-queue-wait",
        provider_factory_timeout=0.05,
    )
    response = await execute_streaming_call(**call_kwargs)
    clock.now = 1.0
    release_blocker.set()
    try:
        wire = "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )
        assert await asyncio.wait_for(blocker_future, timeout=1.0) == "released"
    finally:
        release_blocker.set()
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert target_started.is_set()
    assert "after-queue-wait" in wire
    assert '"code": "provider_unavailable"' not in wire


@pytest.mark.asyncio
async def test_execute_streaming_call_queue_fallback(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    primary_error = _certified_pre_dispatch(
        ChatProviderError(provider="anthropic", message="primary failed", status_code=502)
    )

    def failing_llm_call():
        raise primary_error

    async def save_message_fn(*_args, **_kwargs):
        return None

    def refresh_provider(provider_name: str):
        assert provider_name == "openai"
        return (
            {
                "api_endpoint": "openai",
                "api_key": "fresh-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": True,
            },
            "gpt-4o",
        )

    def fake_perform_chat_api_call(**kwargs):
        assert kwargs.get("api_endpoint") == "openai"

        def _stream():
            yield 'data: {"choices": [{"delta": {"content": "fallback ok"}}]}\n\n'
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    class DummyQueue:
        def __init__(self):
            self._running = True

        def is_running(self):
            return True

        async def enqueue(self, *, processor, stream_channel, **_kwargs):
            async def _run():
                try:
                    result = await asyncio.get_running_loop().run_in_executor(None, processor)
                    if hasattr(result, "__aiter__"):
                        async for chunk in result:
                            await stream_channel.put(chunk)
                    else:
                        for chunk in result:
                            await stream_channel.put(chunk)
                finally:
                    await stream_channel.put(None)

            asyncio.create_task(_run())
            fut = asyncio.Future()
            fut.set_result({"status": "ok"})
            return fut

    monkeypatch.setattr(chat_service, "perform_chat_api_call", fake_perform_chat_api_call)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: DummyQueue())

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "anthropic",
            "api_key": "stale-key",
            "messages_payload": [],
            "model": "claude-3",
            "streaming": True,
        },
        selected_provider="anthropic",
        provider="anthropic",
        model="claude-3",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-queue",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=True,
        enable_provider_fallback=True,
        llm_call_func=failing_llm_call,
        refresh_provider_params=refresh_provider,
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(resp, StreamingResponse)

    agen = resp.body_iterator
    chunks = []
    try:
        async for ln in agen:
            if not ln:
                continue
            chunks.append(ln)
    finally:
        try:
            await agen.aclose()
        except Exception:
            _ = None

    chunks = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in chunks]
    joined = "".join(chunks)
    assert "fallback ok" in joined
    assert any("data: [DONE]" in c for c in chunks)
    assert provider_manager.success_records == ["openai"]
    assert any(
        entry.get("selected_provider") == "openai" and entry.get("queued") is True
        for entry in metrics.fallback_successes
    )


@pytest.mark.asyncio
async def test_real_queue_capacity_one_fallback_refresh_stays_on_request_loop(
    monkeypatch,
):
    request_loop = asyncio.get_running_loop()
    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    queue = RateLimitedQueue(
        max_queue_size=4,
        max_concurrent=1,
        timeout=1.0,
        global_rate_limit=10,
        per_client_rate_limit=10,
    )
    await queue.start(num_workers=1)
    refresh_entered = asyncio.Event()
    release_refresh = asyncio.Event()
    healthy_completed = asyncio.Event()
    provider_manager = _DummyProviderManager()

    def failing_factory():
        raise _certified_pre_dispatch(
            ChatProviderError(provider="anthropic", message="private primary", status_code=502)
        )

    async def refresh_provider(provider_name: str):
        assert asyncio.get_running_loop() is request_loop
        assert provider_name == "openai"
        refresh_entered.set()
        await release_refresh.wait()
        return (
            {
                "api_endpoint": "openai",
                "api_key": "fresh-bounded-key",
                "messages_payload": [],
                "model": "gpt-fallback",
                "streaming": True,
            },
            "gpt-fallback",
        )

    def fallback_factory(**kwargs):
        assert kwargs["api_key"] == "fresh-bounded-key"

        def stream():
            yield 'data: {"choices":[{"delta":{"content":"capacity one fallback"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return stream()

    async def save_message_fn(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", fallback_factory)
    response = await execute_streaming_call(
        current_loop=request_loop,
        cleaned_args={
            "api_endpoint": "anthropic",
            "messages_payload": [],
            "model": "claude-test",
            "streaming": True,
        },
        selected_provider="anthropic",
        provider="anthropic",
        model="claude-test",
        request_json="{}",
        request=SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/chat/completions"),
            headers={},
            state=SimpleNamespace(user_id=None, api_key_id=None),
        ),
        metrics=_DummyMetrics(),
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="real-queue-capacity-one",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="capacity-one",
        queue_execution_enabled=True,
        enable_provider_fallback=True,
        llm_call_func=failing_factory,
        refresh_provider_params=refresh_provider,
        moderation_getter=lambda: _DummyModeration(),
        queue_request_id="real-queue-capacity-one",
        provider_factory_timeout=1.0,
    )

    async def consume() -> str:
        return "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    consume_task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(refresh_entered.wait(), timeout=1.0)
        request_loop.call_soon(healthy_completed.set)
        await asyncio.wait_for(healthy_completed.wait(), timeout=1.0)
        assert not consume_task.done()
        release_refresh.set()
        wire = await asyncio.wait_for(consume_task, timeout=2.0)
    finally:
        release_refresh.set()
        if not consume_task.done():
            consume_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consume_task
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert "capacity one fallback" in wire
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_execute_streaming_call_finalize_runs_without_refund_cb(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    captured: dict[str, object] = {}

    def llm_call_func():
        def _stream():
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    def fake_create_streaming_response_with_timeout(*_args, finalize_callback=None, **_kwargs):
        captured["finalize_callback"] = finalize_callback

        async def _gen():
            if callable(finalize_callback):
                await finalize_callback(success=False, cancelled=False, error=True)
            yield "data: [DONE]\n\n"

        return _gen()

    async def save_message_fn(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_service, "create_streaming_response_with_timeout", fake_create_streaming_response_with_timeout)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-finalize-1",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-test",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(resp, StreamingResponse)

    agen = resp.body_iterator
    try:
        await agen.__anext__()
    except StopAsyncIteration:
        _ = None
    finally:
        with contextlib.suppress(Exception):
            # Best effort close. Some test wrappers may already exhaust/close the iterator.
            await agen.aclose()

    assert callable(captured.get("finalize_callback"))
    assert any(call[3] == "stream_error" for call in metrics.llm_calls)
    assert provider_manager.failure_records


@pytest.mark.asyncio
async def test_execute_streaming_call_refund_cb_still_conditional(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    refund_calls: list[dict[str, bool]] = []

    def llm_call_func():
        def _stream():
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    async def rg_refund_cb(*, cancelled: bool, error: bool):
        refund_calls.append({"cancelled": cancelled, "error": error})

    def fake_create_streaming_response_with_timeout(*_args, finalize_callback=None, **_kwargs):
        async def _gen():
            if callable(finalize_callback):
                await finalize_callback(success=False, cancelled=True, error=False)
            yield "data: [DONE]\n\n"

        return _gen()

    async def save_message_fn(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_service, "create_streaming_response_with_timeout", fake_create_streaming_response_with_timeout)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-finalize-2",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-test",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
        rg_refund_cb=rg_refund_cb,
    )

    agen = resp.body_iterator
    try:
        await agen.__anext__()
    except StopAsyncIteration:
        _ = None
    finally:
        with contextlib.suppress(Exception):
            await agen.aclose()

    assert refund_calls == [{"cancelled": True, "error": False}]
    assert any(call[3] == "stream_cancelled" for call in metrics.llm_calls)


@pytest.mark.asyncio
async def test_queued_refresh_awaitable_runs_on_request_loop_with_healthy_overlap():
    request_loop = asyncio.get_running_loop()
    refresh_entered = asyncio.Event()
    release_refresh = asyncio.Event()
    healthy_completed = asyncio.Event()

    async def refresh_provider(provider: str):
        assert asyncio.get_running_loop() is request_loop
        assert provider == "fallback"
        refresh_entered.set()
        await release_refresh.wait()
        return {"messages_payload": [], "api_key": "bounded"}, "fallback-model"

    refresh_task = asyncio.create_task(
        asyncio.to_thread(
            chat_service._refresh_provider_params_bounded_sync,
            refresh_provider,
            "fallback",
            lambda: 1.0,
            request_loop,
        )
    )
    try:
        await asyncio.wait_for(refresh_entered.wait(), timeout=1.0)
        request_loop.call_soon(healthy_completed.set)
        await asyncio.wait_for(healthy_completed.wait(), timeout=1.0)
        assert not refresh_task.done()
        release_refresh.set()
        refreshed_args, refreshed_model = await asyncio.wait_for(
            refresh_task,
            timeout=1.0,
        )
    finally:
        release_refresh.set()
        if not refresh_task.done():
            await asyncio.wait_for(refresh_task, timeout=1.0)

    assert refreshed_args["api_key"] == "bounded"
    assert refreshed_model == "fallback-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_outcome",
    [
        "choice-error",
        "message-error",
        "delta-error",
        "missing-finish-eof",
        "missing-finish-done",
    ],
)
async def test_concurrent_stream_semantic_gate_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
    invalid_outcome: str,
) -> None:
    """Nested errors and incomplete terminals cannot borrow stream success."""

    sentinel = "stream-semantic-secret-/srv/provider"
    invalid_text = "must-not-certify-partial-output"
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    metrics = [_DummyMetrics(), _DummyMetrics()]
    managers = [_DummyProviderManager(), _DummyProviderManager()]
    marks = [AsyncMock(), AsyncMock()]
    provider_outputs = [AsyncMock(), AsyncMock()]
    saves = [AsyncMock(return_value=None), AsyncMock(return_value=None)]
    audits = [
        SimpleNamespace(log_event=AsyncMock(return_value=None)),
        SimpleNamespace(log_event=AsyncMock(return_value=None)),
    ]
    usage_log = AsyncMock(return_value=None)

    async def _source(index: int):
        ready[index].set()
        await release.wait()
        if index == 1:
            yield (
                'data: {"choices":[{"delta":{"content":"healthy-complete"},'
                '"finish_reason":null}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
            )
            yield "data: [DONE]\n\n"
            return

        if invalid_outcome.startswith("missing-finish"):
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": invalid_text},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + "\n\n"
            )
            if invalid_outcome == "missing-finish-done":
                yield "data: [DONE]\n\n"
            return

        error = {
            "code": "provider_unavailable",
            "message": sentinel,
        }
        choice: dict[str, Any] = {
            "delta": {"content": invalid_text},
            "finish_reason": "stop",
        }
        if invalid_outcome == "choice-error":
            choice["error"] = error
        elif invalid_outcome == "message-error":
            choice["message"] = {"error": error}
        else:
            choice["delta"]["error"] = error
        yield "data: " + json.dumps({"choices": [choice]}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)

    responses = []
    for index in range(2):
        responses.append(
            await execute_streaming_call(
                **_streaming_call_kwargs(
                    llm_call_func=lambda index=index: _source(index),
                    provider_manager=managers[index],
                    metrics=metrics[index],
                    conversation_id=f"stream-semantic-{invalid_outcome}-{index}",
                    save_message_fn=saves[index],
                    audit_service=audits[index],
                    on_success=marks[index],
                    on_provider_output=provider_outputs[index],
                )
            )
        )

    async def _collect(response: StreamingResponse) -> str:
        return "".join(
            [
                chunk.decode()
                if isinstance(chunk, (bytes, bytearray))
                else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    tasks = [asyncio.create_task(_collect(response)) for response in responses]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in ready)),
            timeout=1.0,
        )
        release.set()
        invalid_wire, healthy_wire = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=1.0,
        )
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert sentinel not in invalid_wire
    assert '"code": "provider_unavailable"' in invalid_wire
    assert '"success": false' in invalid_wire
    assert '"success": true' not in invalid_wire
    if invalid_outcome.endswith("error"):
        assert invalid_text not in invalid_wire
    marks[0].assert_not_awaited()
    if invalid_outcome.startswith("missing-finish"):
        provider_outputs[0].assert_awaited_once_with("openai")
    else:
        provider_outputs[0].assert_not_awaited()
    saves[0].assert_not_awaited()
    audits[0].log_event.assert_not_awaited()
    assert managers[0].success_records == []
    assert not any(call[2] is True for call in metrics[0].llm_calls)

    assert "healthy-complete" in healthy_wire
    assert '"error"' not in healthy_wire
    assert '"success": true' in healthy_wire
    marks[1].assert_awaited_once_with("openai")
    provider_outputs[1].assert_awaited_once_with("openai")
    saves[1].assert_awaited_once()
    audits[1].log_event.assert_awaited_once()
    assert managers[1].success_records == ["openai"]
    assert any(call[2] is True for call in metrics[1].llm_calls)
    usage_log.assert_awaited_once()
    assert usage_log.await_args.kwargs["conversation_id"] == (
        f"stream-semantic-{invalid_outcome}-1"
    )


@pytest.mark.asyncio
async def test_concurrent_stream_refusal_and_content_filter_account_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contentless semantic terminals keep request-local durable accounting."""

    outcomes = ["refusal", "content-filter"]
    ready = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    metrics = [_DummyMetrics(), _DummyMetrics()]
    managers = [_DummyProviderManager(), _DummyProviderManager()]
    marks = [AsyncMock(), AsyncMock()]
    saves = [AsyncMock(return_value=None), AsyncMock(return_value=None)]
    audits = [
        SimpleNamespace(log_event=AsyncMock(return_value=None)),
        SimpleNamespace(log_event=AsyncMock(return_value=None)),
    ]
    usage_log = AsyncMock(return_value=None)

    async def _source(index: int):
        ready[index].set()
        await release[index].wait()
        if outcomes[index] == "refusal":
            yield (
                'data: {"choices":[{"delta":{"refusal":"I cannot comply."},'
                '"finish_reason":null}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":3}}\n\n'
            )
        else:
            yield (
                'data: {"choices":[{"delta":{},'
                '"finish_reason":"content_filter"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":0}}\n\n'
            )
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)

    responses = [
        await execute_streaming_call(
            **_streaming_call_kwargs(
                llm_call_func=lambda index=index: _source(index),
                provider_manager=managers[index],
                metrics=metrics[index],
                conversation_id=f"stream-semantic-{outcomes[index]}",
                save_message_fn=saves[index],
                audit_service=audits[index],
                on_success=marks[index],
            )
        )
        for index in range(2)
    ]

    async def _collect(response: StreamingResponse) -> str:
        return "".join(
            [
                chunk.decode()
                if isinstance(chunk, (bytes, bytearray))
                else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    tasks = [asyncio.create_task(_collect(response)) for response in responses]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in ready)),
            timeout=1.0,
        )
        release[1].set()
        content_filter_wire = await asyncio.wait_for(
            asyncio.shield(tasks[1]),
            timeout=1.0,
        )
        assert tasks[0].done() is False
        assert usage_log.await_count == 1
        assert usage_log.await_args.kwargs["conversation_id"] == (
            "stream-semantic-content-filter"
        )

        release[0].set()
        refusal_wire = await asyncio.wait_for(
            asyncio.shield(tasks[0]),
            timeout=1.0,
        )
    finally:
        for event in release:
            event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert "I cannot comply." in refusal_wire
    assert '"finish_reason": "content_filter"' in content_filter_wire
    assert '"success": true' in refusal_wire
    assert '"success": true' in content_filter_wire
    assert all('"code": "provider_unavailable"' not in wire for wire in (
        refusal_wire,
        content_filter_wire,
    ))
    for index in range(2):
        marks[index].assert_awaited_once_with("openai")
        saves[index].assert_not_awaited()
        audits[index].log_event.assert_awaited_once()
        assert managers[index].failure_records == []
        assert managers[index].success_records == ["openai"]
        assert any(call[2] is True for call in metrics[index].llm_calls)
    assert usage_log.await_count == 2
    assert {
        call.kwargs["conversation_id"] for call in usage_log.await_args_list
    } == {
        "stream-semantic-refusal",
        "stream-semantic-content-filter",
    }


@pytest.mark.asyncio
async def test_real_queue_done_then_adapter_failure_is_atomic_and_never_marks_success(
    monkeypatch,
):
    sentinel = "queued-adapter-secret-/srv/provider"
    adapter_failed = threading.Event()
    on_success = AsyncMock()
    save_message = AsyncMock(return_value=None)
    usage_log = AsyncMock(return_value=None)
    audit_service = SimpleNamespace(log_event=AsyncMock(return_value=None))
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    queue = RateLimitedQueue(
        max_queue_size=4,
        max_concurrent=1,
        timeout=1.0,
        global_rate_limit=10,
        per_client_rate_limit=10,
    )
    await queue.start(num_workers=1)

    def done_then_fail():
        def stream():
            yield "data: [DONE]\n\n"
            adapter_failed.set()
            raise RuntimeError(sentinel)

        return stream()

    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    response = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/chat/completions"),
            headers={},
            state=SimpleNamespace(user_id=None, api_key_id=None),
        ),
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=True,
        final_conversation_id="atomic-queued-terminal",
        character_card_for_context=None,
        chat_db=object(),
        save_message_fn=save_message,
        audit_service=audit_service,
        audit_context=object(),
        client_id="atomic-queued-terminal",
        queue_execution_enabled=True,
        enable_provider_fallback=False,
        llm_call_func=done_then_fail,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
        on_success=on_success,
        queue_request_id="atomic-queued-terminal",
        provider_factory_timeout=1.0,
    )
    try:
        assert await asyncio.to_thread(adapter_failed.wait, 1.0)
        wire = "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert '"code": "provider_unavailable"' in wire
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    on_success.assert_not_awaited()
    save_message.assert_not_awaited()
    usage_log.assert_not_awaited()
    audit_service.log_event.assert_not_awaited()
    assert provider_manager.success_records == []
    assert not any(call[2] is True for call in metrics.llm_calls)


@pytest.mark.asyncio
async def test_real_queue_combined_output_done_late_error_never_records_provider_use(
    monkeypatch,
):
    sentinel = "queued-output-secret-/srv/provider"
    adapter_waiting = threading.Event()
    release_adapter = threading.Event()
    recorded_providers: list[str] = []
    save_message = AsyncMock(return_value=None)
    usage_log = AsyncMock(return_value=None)
    audit_service = SimpleNamespace(log_event=AsyncMock(return_value=None))
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    queue = RateLimitedQueue(
        max_queue_size=4,
        max_concurrent=1,
        timeout=1.0,
        global_rate_limit=10,
        per_client_rate_limit=10,
    )
    await queue.start(num_workers=1)

    async def record_provider_use(provider: str) -> None:
        recorded_providers.append(provider)

    def combined_output_done_then_fail():
        def stream():
            yield (
                'data: {"choices":[{"delta":{"content":"queued-partial"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            adapter_waiting.set()
            release_adapter.wait()
            raise RuntimeError(sentinel)

        return stream()

    async def collect(response):
        return "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    call_kwargs = _streaming_call_kwargs(
        llm_call_func=combined_output_done_then_fail,
        provider_manager=provider_manager,
        metrics=metrics,
        conversation_id="queued-first-output-late-error",
        save_message_fn=save_message,
        audit_service=audit_service,
        on_success=record_provider_use,
    )
    call_kwargs.update(
        queue_execution_enabled=True,
        queue_request_id="queued-first-output-late-error",
    )
    response = await execute_streaming_call(**call_kwargs)
    collect_task = asyncio.create_task(collect(response))
    try:
        assert await asyncio.to_thread(adapter_waiting.wait, 1.0)
        assert recorded_providers == []
        assert not collect_task.done()
        release_adapter.set()
        wire = await asyncio.wait_for(collect_task, timeout=1.0)
    finally:
        release_adapter.set()
        if not collect_task.done():
            collect_task.cancel()
        await asyncio.gather(collect_task, return_exceptions=True)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert "queued-partial" in wire
    assert '"code": "provider_unavailable"' in wire
    assert '"success": false' in wire
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    assert recorded_providers == []
    save_message.assert_not_awaited()
    usage_log.assert_not_awaited()
    audit_service.log_event.assert_not_awaited()
    assert provider_manager.success_records == []
    assert not any(call[2] is True for call in metrics.llm_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
async def test_real_queue_producer_runs_ahead_preserves_output_then_error_once(
    monkeypatch,
    stream_kind,
):
    sentinel = f"queued-ahead-{stream_kind}-secret-/srv/provider"
    producer_failed = threading.Event()
    recorded_providers: list[str] = []
    fallback_started = threading.Event()
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    queue = RateLimitedQueue(
        max_queue_size=4,
        max_concurrent=1,
        timeout=1.0,
        global_rate_limit=10,
        per_client_rate_limit=10,
    )
    await queue.start(num_workers=1)

    async def record_provider_use(provider: str) -> None:
        recorded_providers.append(provider)

    def sync_factory():
        def stream():
            yield (
                f'data: {{"choices":[{{"delta":{{"content":"ahead-{stream_kind}"}}}}]}}\n\n'
            )
            producer_failed.set()
            raise RuntimeError(sentinel)

        return stream()

    def async_factory():
        async def stream():
            yield (
                f'data: {{"choices":[{{"delta":{{"content":"ahead-{stream_kind}"}}}}]}}\n\n'
            )
            producer_failed.set()
            raise RuntimeError(sentinel)

        return stream()

    def fallback_factory(**_kwargs):
        fallback_started.set()
        return iter(("fallback-output",))

    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", fallback_factory)
    call_kwargs = _streaming_call_kwargs(
        llm_call_func=sync_factory if stream_kind == "sync" else async_factory,
        provider_manager=provider_manager,
        metrics=metrics,
        conversation_id=f"queued-producer-ahead-{stream_kind}",
        save_message_fn=AsyncMock(return_value=None),
        audit_service=SimpleNamespace(log_event=AsyncMock(return_value=None)),
        on_success=record_provider_use,
    )
    call_kwargs.update(
        queue_execution_enabled=True,
        enable_provider_fallback=True,
        queue_request_id=f"queued-producer-ahead-{stream_kind}",
    )
    response = await execute_streaming_call(**call_kwargs)
    try:
        assert await asyncio.to_thread(producer_failed.wait, 1.0)

        async def wait_for_queue_completion() -> None:
            while f"queued-producer-ahead-{stream_kind}" in queue._active_request_ids:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_queue_completion(), timeout=1.0)
        wire = "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert wire.index(f"ahead-{stream_kind}") < wire.index('"code": "provider_unavailable"')
    assert wire.count('"code": "provider_unavailable"') == 1
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    assert recorded_providers == []
    assert provider_manager.fallback_requests == []
    assert not fallback_started.is_set()


@pytest.mark.asyncio
async def test_real_queue_post_done_junk_is_ignored_on_normal_exhaustion(
    monkeypatch,
):
    post_done_junk = "post-done-junk-must-not-persist"
    on_success = AsyncMock()
    save_message = AsyncMock(return_value=None)
    usage_log = AsyncMock(return_value=None)
    audit_service = SimpleNamespace(log_event=AsyncMock(return_value=None))
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    queue = RateLimitedQueue(
        max_queue_size=4,
        max_concurrent=1,
        timeout=1.0,
        global_rate_limit=10,
        per_client_rate_limit=10,
    )
    await queue.start(num_workers=1)

    def done_then_junk():
        def stream():
            yield "data: [DONE]\n\n"
            yield f'data: {{"choices":[{{"delta":{{"content":"{post_done_junk}"}}}}]}}\n\n'

        return stream()

    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    call_kwargs = _streaming_call_kwargs(
        llm_call_func=done_then_junk,
        provider_manager=provider_manager,
        metrics=metrics,
        conversation_id="queued-post-done-junk",
        save_message_fn=save_message,
        audit_service=audit_service,
        on_success=on_success,
    )
    call_kwargs.update(
        queue_execution_enabled=True,
        queue_request_id="queued-post-done-junk",
    )
    response = await execute_streaming_call(**call_kwargs)
    try:
        wire = "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )
    finally:
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert post_done_junk not in wire
    assert '"code": "provider_unavailable"' in wire
    assert '"success": false' in wire
    assert '"success": true' not in wire
    assert wire.count("data: [DONE]") == 1
    on_success.assert_not_awaited()
    save_message.assert_not_awaited()
    usage_log.assert_not_awaited()
    audit_service.log_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_chunks", "expected_content", "raise_after_chunks"),
    [
        (("data: [DONE]\n\n",), None, True),
        (
            (
                'data: {"choices":[{"delta":{"content":"partial-combined"}}]}\n\n'
                "data: [DONE]\n\n",
            ),
            "partial-combined",
            True,
        ),
        (
            (
                'data: {"choices":[{"delta":{"content":"partial-split"}}]}\n\n',
                "data: [DONE]\n\n",
            ),
            "partial-split",
            True,
        ),
        (
            (
                'data: {"choices":[{"delta":{"content":"partial-inband"}}]}\n\n'
                "data: [DONE]\n\n"
                'data: {"error":{"message":"direct-adapter-secret-/srv/provider"}}\n\n',
            ),
            "partial-inband",
            False,
        ),
    ],
    ids=(
        "exact-done",
        "combined-content-done",
        "split-content-done",
        "combined-done-inband-error",
    ),
)
async def test_direct_done_then_adapter_failure_records_valid_output_without_clean_success(
    monkeypatch,
    provider_chunks,
    expected_content,
    raise_after_chunks,
):
    sentinel = "direct-adapter-secret-/srv/provider"
    adapter_failed = asyncio.Event()
    on_success = AsyncMock()
    on_provider_output = AsyncMock()
    save_message = AsyncMock(return_value=None)
    usage_log = AsyncMock(return_value=None)
    audit_service = SimpleNamespace(log_event=AsyncMock(return_value=None))
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    async def done_then_fail():
        for chunk in provider_chunks:
            yield chunk
        adapter_failed.set()
        if raise_after_chunks:
            raise RuntimeError(sentinel)

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)
    response = await execute_streaming_call(
        **_streaming_call_kwargs(
            llm_call_func=done_then_fail,
            provider_manager=provider_manager,
            metrics=metrics,
            conversation_id="atomic-direct-terminal",
            save_message_fn=save_message,
            audit_service=audit_service,
            on_success=on_success,
            on_provider_output=on_provider_output,
        )
    )
    wire = "".join(
        [
            chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            async for chunk in response.body_iterator
        ]
    )

    if raise_after_chunks:
        assert adapter_failed.is_set()
    if expected_content is not None:
        assert expected_content in wire
    assert '"code": "provider_unavailable"' in wire
    assert '"success": false' in wire
    assert '"success": true' not in wire
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    on_success.assert_not_awaited()
    if expected_content is None:
        on_provider_output.assert_not_awaited()
    else:
        on_provider_output.assert_awaited_once_with("openai")
    save_message.assert_not_awaited()
    usage_log.assert_not_awaited()
    audit_service.log_event.assert_not_awaited()
    assert provider_manager.success_records == []
    assert not any(call[2] is True for call in metrics.llm_calls)


@pytest.mark.asyncio
async def test_concurrent_direct_failure_and_queued_success_record_provider_use_once(
    monkeypatch,
):
    sentinel = "direct-overlap-secret-/srv/provider"
    faulty_waiting = asyncio.Event()
    healthy_waiting = asyncio.Event()
    release = asyncio.Event()
    usage_log = AsyncMock(return_value=None)

    async def pending_done_stream(content, waiting, *, fails):
        yield f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
        if not fails:
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"
        waiting.set()
        await release.wait()
        if fails:
            raise RuntimeError(sentinel)

    faulty_on_success = AsyncMock()
    faulty_save = AsyncMock(return_value=None)
    faulty_audit = SimpleNamespace(log_event=AsyncMock(return_value=None))
    faulty_metrics = _DummyMetrics()
    faulty_manager = _DummyProviderManager()
    healthy_on_success = AsyncMock()
    healthy_save = AsyncMock(return_value=None)
    healthy_audit = SimpleNamespace(log_event=AsyncMock(return_value=None))
    healthy_metrics = _DummyMetrics()
    healthy_manager = _DummyProviderManager()
    queue = RateLimitedQueue(
        max_queue_size=4,
        max_concurrent=1,
        timeout=1.0,
        global_rate_limit=10,
        per_client_rate_limit=10,
    )
    await queue.start(num_workers=1)

    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)

    faulty_response = await execute_streaming_call(
        **_streaming_call_kwargs(
            llm_call_func=lambda: pending_done_stream(
                "faulty-partial",
                faulty_waiting,
                fails=True,
            ),
            provider_manager=faulty_manager,
            metrics=faulty_metrics,
            conversation_id="direct-overlap-faulty",
            save_message_fn=faulty_save,
            audit_service=faulty_audit,
            on_success=faulty_on_success,
        )
    )
    healthy_call_kwargs = _streaming_call_kwargs(
        llm_call_func=lambda: pending_done_stream(
            "healthy-complete",
            healthy_waiting,
            fails=False,
        ),
        provider_manager=healthy_manager,
        metrics=healthy_metrics,
        conversation_id="direct-overlap-healthy",
        save_message_fn=healthy_save,
        audit_service=healthy_audit,
        on_success=healthy_on_success,
    )
    healthy_call_kwargs.update(
        queue_execution_enabled=True,
        queue_request_id="direct-queued-overlap-healthy",
    )
    healthy_response = await execute_streaming_call(**healthy_call_kwargs)

    async def collect(response):
        return "".join(
            [
                chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                async for chunk in response.body_iterator
            ]
        )

    faulty_task = asyncio.create_task(collect(faulty_response))
    healthy_task = asyncio.create_task(collect(healthy_response))
    try:
        await asyncio.wait_for(
            asyncio.gather(faulty_waiting.wait(), healthy_waiting.wait()),
            timeout=1.0,
        )
        assert not faulty_task.done()
        assert not healthy_task.done()
        release.set()
        faulty_wire, healthy_wire = await asyncio.wait_for(
            asyncio.gather(faulty_task, healthy_task),
            timeout=1.0,
        )
    finally:
        release.set()
        for task in (faulty_task, healthy_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(faulty_task, healthy_task, return_exceptions=True)
        await asyncio.wait_for(queue.stop(), timeout=1.0)

    assert "faulty-partial" in faulty_wire
    assert '"code": "provider_unavailable"' in faulty_wire
    assert '"success": false' in faulty_wire
    assert faulty_wire.count("data: [DONE]") == 1
    assert sentinel not in faulty_wire
    faulty_on_success.assert_not_awaited()
    faulty_save.assert_not_awaited()
    faulty_audit.log_event.assert_not_awaited()
    assert faulty_manager.success_records == []
    assert not any(call[2] is True for call in faulty_metrics.llm_calls)

    assert "healthy-complete" in healthy_wire
    assert '"error"' not in healthy_wire
    assert '"success": true' in healthy_wire
    assert healthy_wire.count("data: [DONE]") == 1
    healthy_on_success.assert_awaited_once_with("openai")
    healthy_save.assert_awaited_once()
    healthy_audit.log_event.assert_awaited_once()
    usage_log.assert_awaited_once()
    assert usage_log.await_args.kwargs["conversation_id"] == "direct-overlap-healthy"
    assert healthy_manager.success_records == ["openai"]
    assert any(call[2] is True for call in healthy_metrics.llm_calls)


def test_merge_api_keys_prefers_dynamic_over_module():


    module_keys = {"openai": "module-key", "anthropic": "module-anthropic"}
    dynamic_keys = {"openai": "dynamic-key", "anthropic": ""}

    raw_openai, normalized_openai = merge_api_keys_for_provider(
        "openai",
        module_keys,
        dynamic_keys,
        {},
    )
    assert raw_openai == "dynamic-key"
    assert normalized_openai == "dynamic-key"

    raw_anthropic, normalized_anthropic = merge_api_keys_for_provider(
        "anthropic",
        module_keys,
        dynamic_keys,
        {},
    )
    assert raw_anthropic == "module-anthropic"
    assert normalized_anthropic == "module-anthropic"
