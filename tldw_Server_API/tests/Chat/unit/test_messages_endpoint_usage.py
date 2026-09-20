"""Usage-accounting regressions for the Anthropic-compatible Messages API."""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
from tldw_Server_API.app.api.v1.schemas.anthropic_messages import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.Chat.streaming_utils import MAX_TOOL_ARGUMENT_LENGTH


def _resolved_credentials(
    provider: str,
    api_key: str,
    touched: list[str],
) -> ResolvedByokCredentials:
    async def _touch() -> None:
        touched.append(api_key)

    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        _touch_cb=_touch,
    )


def _install_resolutions(
    monkeypatch: pytest.MonkeyPatch,
    resolutions: dict[int, ResolvedByokCredentials],
) -> None:
    class _Runtime:
        def __init__(self, *, user_id: int, **_kwargs: Any) -> None:
            self._resolution = resolutions[user_id]

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            resolution = self._resolution
            return SimpleNamespace(
                provider=resolution.provider,
                api_key=resolution.api_key,
                app_config=resolution.app_config,
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: Any) -> None:
            await self._resolution.touch_last_used()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id_int), [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)


class _DirectUsageRuntime:
    """Minimal runtime for exercising the accounting wrapper in isolation."""

    async def mark_used(self, credentials: ResolvedByokCredentials) -> None:
        await credentials.touch_last_used()

    async def close(self) -> None:
        return None


def _message_request(model: str, *, stream: bool = False) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=stream,
    )


def _mixed_converted_messages_response(
    case: str,
    sentinel: str,
) -> dict[str, Any]:
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    valid_choice = {
        "message": {"content": "valid assistant text"},
        "finish_reason": "stop",
    }
    if case == "later-choice":
        return {
            "choices": [
                valid_choice,
                {"error": {"message": sentinel}},
            ]
        }
    if case == "message-error-sibling":
        return {
            "choices": [
                {
                    "message": {
                        "content": "valid assistant text",
                        "error": {"message": sentinel},
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    if case == "content-error-block":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "error", "error": {"message": sentinel}},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    if case == "content-later-error-text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "text", "text": f"data: {serialized}\n\n"},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    if case == "tool-call-before-later-error":
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
                {"error": {"message": sentinel}},
            ]
        }
    raise AssertionError(f"Unknown mixed converted response case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "later-choice",
        "message-error-sibling",
        "content-error-block",
        "tool-call-before-later-error",
    ],
)
def test_converted_messages_rejects_mixed_success_and_provider_error(
    case: str,
) -> None:
    """Semantic conversion must inspect every nested and later error envelope."""
    sentinel = "messages-converted-mixed-secret-/private/provider.json"

    assert (
        messages_endpoint._convert_semantic_openai_messages_response(
            _mixed_converted_messages_response(case, sentinel),
            model="model-a",
        )
        is None
    )


def test_converted_messages_preserves_assistant_error_envelope_text() -> None:
    """A serialized error envelope inside a text block remains assistant text."""

    sentinel = "assistant-authored-example"
    response = _mixed_converted_messages_response(
        "content-later-error-text",
        sentinel,
    )

    converted = messages_endpoint._convert_semantic_openai_messages_response(
        response,
        model="model-a",
    )

    assert converted is not None
    assert sentinel in "".join(
        block["text"]
        for block in converted["content"]
        if block.get("type") == "text"
    )


def test_converted_messages_preserves_noncanonical_assistant_error_json() -> None:
    """Assistant-authored noncanonical JSON remains valid message text."""
    content = json.dumps(
        {"error": {"code": "fictional_story_error", "message": "plot device"}}
    )

    converted = messages_endpoint._convert_semantic_openai_messages_response(
        {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ]
        },
        model="model-a",
    )

    assert converted is not None
    assert converted["content"] == [{"type": "text", "text": content}]


def test_converted_messages_preserves_domain_error_in_tool_arguments() -> None:
    """A tool's domain-level error object is data, not a provider failure."""
    domain_error = {
        "error": {"code": "city_not_found", "message": "No matching city"}
    }

    converted = messages_endpoint._convert_semantic_openai_messages_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-domain-error",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": json.dumps(domain_error),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        model="model-a",
    )

    assert converted is not None
    assert converted["content"] == [
        {
            "type": "tool_use",
            "id": "call-domain-error",
            "name": "lookup_weather",
            "input": domain_error,
        }
    ]


def _converted_stream_error_chunks(case: str, sentinel: str) -> list[str]:
    """Build one converted-provider stream failure in each supported shape."""
    if case == "raw-error-prefix":
        return [f"Error: {sentinel}\n"]
    if case == "canonical-raw-code":
        return ["provider_unavailable\n"]
    if case == "sse-error-text":
        return [f"data: Error: {sentinel}\n\n"]

    error = {"message": sentinel, "code": "provider_unavailable"}
    if case == "type-error":
        payload: dict[str, Any] = {"type": "error", "message": sentinel}
    elif case == "canonical-error-code":
        payload = {"error_code": "provider_unavailable", "message": sentinel}
    elif case == "nested-message":
        payload = {"choices": [{"message": {"error": error}}]}
    elif case == "nested-delta":
        payload = {"choices": [{"delta": {"error": error}}]}
    elif case == "nested-content":
        payload = {
            "choices": [
                {
                    "delta": {
                        "content": [
                            {"type": "error", "error": error},
                        ]
                    }
                }
            ]
        }
    elif case == "later-choice":
        payload = {
            "choices": [
                {"delta": {"content": "valid text"}},
                {"error": error},
            ]
        }
    elif case == "same-delta-mixed":
        payload = {
            "choices": [
                {
                    "delta": {
                        "content": "valid text",
                        "error": error,
                    }
                }
            ]
        }
    else:
        raise AssertionError(f"Unknown converted stream error case: {case}")
    return [f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "raw-error-prefix",
        "canonical-raw-code",
        "sse-error-text",
        "type-error",
        "canonical-error-code",
        "nested-message",
        "nested-delta",
        "nested-content",
        "later-choice",
        "same-delta-mixed",
    ],
)
async def test_converted_stream_rejects_every_structural_provider_error_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """A converted stream must apply the shared structural error predicate."""
    sentinel = "converted-stream-structural-secret-/private/provider.json"
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "stream-error-key", touched)},
    )

    async def _stream_adapter(**_kwargs: Any):
        async def _stream():
            for chunk in _converted_stream_error_chunks(case, sentinel):
                yield chunk

        return _stream()

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body = "".join([item async for item in response.body_iterator])

    assert "event: error" in body
    assert "The upstream provider returned an error." in body
    assert "text_delta" not in body
    assert sentinel not in body
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("argument_style", ["tool_calls", "function_call"])
async def test_converted_stream_preserves_domain_error_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
    argument_style: str,
) -> None:
    """Domain error objects remain tool input for modern and legacy streams."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "stream-tool-key", touched)},
    )
    arguments = json.dumps(
        {"error": {"code": "city_not_found", "message": "No matching city"}},
        separators=(",", ":"),
    )
    if argument_style == "tool_calls":
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-domain-error",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": arguments,
                    },
                }
            ]
        }
        finish_reason = "tool_calls"
    else:
        delta = {
            "function_call": {
                "name": "lookup_weather",
                "arguments": arguments,
            }
        }
        finish_reason = "function_call"

    async def _stream_adapter(**_kwargs: Any):
        async def _stream():
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ]
                },
                separators=(",", ":"),
            ) + "\n\n"

        return _stream()

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body = "".join([item async for item in response.body_iterator])

    assert "event: error" not in body
    assert '"type": "tool_use"' in body
    assert '"type": "input_json_delta"' in body
    assert "city_not_found" in body
    assert touched == ["stream-tool-key"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_stream_error_marks_only_valid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural stream failure in one request cannot certify another handle."""
    touched: list[str] = []
    entered = {"valid-stream-key": asyncio.Event(), "error-stream-key": asyncio.Event()}
    release = asyncio.Event()
    sentinel = "converted-stream-concurrent-secret-/private/provider.json"
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("openai", "valid-stream-key", touched),
            2: _resolved_credentials("openai", "error-stream-key", touched),
        },
    )

    async def _stream_adapter(**kwargs: Any):
        api_key = kwargs["api_key"]

        async def _stream():
            entered[api_key].set()
            await release.wait()
            if api_key == "valid-stream-key":
                yield (
                    'data: {"choices":[{"delta":{"content":"valid"},'
                    '"finish_reason":"stop"}]}\n\n'
                )
                return
            for chunk in _converted_stream_error_chunks("nested-delta", sentinel):
                yield chunk

        return _stream()

    async def _request_and_consume(user_id: int) -> str:
        response = await messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini", stream=True),
            current_user=SimpleNamespace(id_int=user_id),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
        return "".join([item async for item in response.body_iterator])

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)
    valid_task = asyncio.create_task(_request_and_consume(1))
    error_task = asyncio.create_task(_request_and_consume(2))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, error_body = await asyncio.wait_for(
            asyncio.gather(valid_task, error_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert "text_delta" in valid_body
    assert "event: error" in error_body
    assert sentinel not in error_body
    assert touched == ["valid-stream-key"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_mixed_error_marks_only_valid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed error response cannot mark itself or another request's handle."""
    touched: list[str] = []
    entered = {"mixed-valid-key": asyncio.Event(), "mixed-error-key": asyncio.Event()}
    release = {"mixed-valid-key": asyncio.Event(), "mixed-error-key": asyncio.Event()}
    sentinel = "messages-concurrent-mixed-secret-/private/provider.json"
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("openai", "mixed-valid-key", touched),
            2: _resolved_credentials("openai", "mixed-error-key", touched),
        },
    )

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        api_key = kwargs["api_key"]
        entered[api_key].set()
        await release[api_key].wait()
        if api_key == "mixed-valid-key":
            return {
                "choices": [
                    {
                        "message": {"content": "valid output"},
                        "finish_reason": "stop",
                    }
                ]
            }
        return _mixed_converted_messages_response("later-choice", sentinel)

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    valid_task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    error_task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini"),
            current_user=SimpleNamespace(id_int=2),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["mixed-error-key"].set()
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(error_task, timeout=1.0)
        assert exc_info.value.status_code == 502
        assert sentinel not in str(exc_info.value)
        assert touched == []
        release["mixed-valid-key"].set()
        valid_response = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert valid_response.status_code == 200
    assert touched == ["mixed-valid-key"]


@pytest.mark.asyncio
async def test_messages_uses_trusted_runtime_scope_and_closes_after_nonstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Messages must use one scoped runtime and release it after dispatch."""

    runtime_args: list[dict[str, Any]] = []
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="scoped-key",
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            runtime_args.append(kwargs)

        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "gpt-4o-mini")
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "scoped-key"
        return {
            "id": "response-scoped",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (7, [11], [22], True),
        raising=False,
    )
    monkeypatch.setattr(
        messages_endpoint,
        "ProviderCredentialRuntime",
        _Runtime,
        raising=False,
    )
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)

    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini"),
        current_user=SimpleNamespace(id_int=7),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert runtime_args == [
        {
            "user_id": 7,
            "team_ids": [11],
            "org_ids": [22],
            "trusted_base_url_override": True,
            "override_snapshot_resolver": (
                messages_endpoint.capture_provider_override_call_snapshot
            ),
        }
    ]
    assert lifecycle == ["mark", "close"]


@pytest.mark.asyncio
async def test_converted_messages_legacy_function_call_marks_scoped_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid legacy function_call converts to tool_use before usage marking."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "legacy-function-key", touched)},
    )

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "legacy-function-key"
        return {
            "id": "response-legacy-function",
            "model": "legacy-model",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "function_call": {
                            "name": "lookup",
                            "arguments": '{"query":"weather"}',
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)

    response = await messages_endpoint._handle_messages(
        _message_request("openai/legacy-model"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["content"] == [
        {
            "type": "tool_use",
            "id": "tool_0",
            "name": "lookup",
            "input": {"query": "weather"},
        }
    ]
    assert payload["stop_reason"] == "tool_use"
    assert touched == ["legacy-function-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "function_call",
    [
        {"arguments": "{}"},
        {"name": "   ", "arguments": "{}"},
    ],
    ids=["missing-name", "blank-name"],
)
async def test_converted_messages_malformed_legacy_function_call_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    function_call: dict[str, Any],
) -> None:
    """Compatibility conversion cannot certify a nameless legacy tool call."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "malformed-function-key", touched)},
    )

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "function_call": function_call,
                    },
                    "finish_reason": "function_call",
                }
            ]
        }

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)

    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/legacy-model"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Upstream provider request failed."
    assert touched == []


def _converted_tool_response(
    *,
    argument_style: str,
    arguments: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one modern or legacy OpenAI tool response."""
    if argument_style == "tool_calls":
        tool_payload: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "call-schema",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": arguments,
                    },
                }
            ]
        }
        finish_reason = "tool_calls"
    elif argument_style == "function_call":
        tool_payload = {
            "function_call": {
                "name": "lookup_weather",
                "arguments": arguments,
            }
        }
        finish_reason = "function_call"
    else:
        raise AssertionError(f"Unknown argument style: {argument_style}")
    return {
        "id": "response-tool-schema",
        "model": "model-a",
        "choices": [
            {
                "message": {"content": None, **tool_payload},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 2, "completion_tokens": 3},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("argument_style", ["tool_calls", "function_call"])
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("not-json", id="malformed-json"),
        pytest.param("[]", id="json-list"),
        pytest.param("null", id="json-null"),
        pytest.param("42", id="json-number"),
        pytest.param('"value"', id="json-string"),
    ],
)
async def test_converted_nonstream_rejects_nonobject_tool_input_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    argument_style: str,
    arguments: str,
) -> None:
    """Final Anthropic tool_use input must be a JSON object before accounting."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "invalid-tool-input-key", touched)},
    )

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        return _converted_tool_response(
            argument_style=argument_style,
            arguments=arguments,
        )

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/model-a"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Upstream provider request failed."
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("argument_style", ["tool_calls", "function_call"])
@pytest.mark.parametrize(
    ("arguments", "expected_input"),
    [
        pytest.param(
            '{"query":"weather"}',
            {"query": "weather"},
            id="ordinary-object",
        ),
        pytest.param(
            '{"error":{"code":"city_not_found"}}',
            {"error": {"code": "city_not_found"}},
            id="domain-error-object",
        ),
    ],
)
async def test_converted_nonstream_accepts_object_tool_input_controls(
    monkeypatch: pytest.MonkeyPatch,
    argument_style: str,
    arguments: str,
    expected_input: dict[str, Any],
) -> None:
    """Valid ordinary and domain-error tool input objects remain compatible."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "valid-tool-input-key", touched)},
    )

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        return _converted_tool_response(
            argument_style=argument_style,
            arguments=arguments,
        )

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("openai/model-a"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert json.loads(response.body)["content"][0]["input"] == expected_input
    assert touched == ["valid-tool-input-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage_field", "invalid_value"),
    [
        pytest.param("prompt_tokens", True, id="input-boolean"),
        pytest.param("prompt_tokens", -1, id="input-negative"),
        pytest.param("prompt_tokens", "2", id="input-string"),
        pytest.param("prompt_tokens", 2.5, id="input-float"),
        pytest.param("completion_tokens", False, id="output-boolean"),
        pytest.param("completion_tokens", -1, id="output-negative"),
        pytest.param("completion_tokens", "3", id="output-string"),
        pytest.param("completion_tokens", 3.5, id="output-float"),
    ],
)
async def test_converted_nonstream_rejects_invalid_usage_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    usage_field: str,
    invalid_value: Any,
) -> None:
    """Final Anthropic usage counters must be nonboolean nonnegative integers."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "invalid-usage-key", touched)},
    )
    usage: dict[str, Any] = {"prompt_tokens": 2, "completion_tokens": 3}
    usage[usage_field] = invalid_value

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        return {
            "id": "response-invalid-usage",
            "choices": [{"message": {"content": "valid text"}}],
            "usage": usage,
        }

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/model-a"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_tool_schema_marks_only_valid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed tool input cannot mark itself or a concurrent valid handle."""
    touched: list[str] = []
    entered = {"valid-schema-key": asyncio.Event(), "invalid-schema-key": asyncio.Event()}
    release = {"valid-schema-key": asyncio.Event(), "invalid-schema-key": asyncio.Event()}
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("openai", "valid-schema-key", touched),
            2: _resolved_credentials("openai", "invalid-schema-key", touched),
        },
    )

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        api_key = kwargs["api_key"]
        entered[api_key].set()
        await release[api_key].wait()
        arguments = (
            '{"error":{"code":"city_not_found"}}'
            if api_key == "valid-schema-key"
            else "not-json"
        )
        return _converted_tool_response(
            argument_style="function_call",
            arguments=arguments,
        )

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    valid_task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("openai/model-a"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    invalid_task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("openai/model-a"),
            current_user=SimpleNamespace(id_int=2),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["invalid-schema-key"].set()
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(invalid_task, timeout=1.0)
        assert exc_info.value.status_code == 502
        assert touched == []
        release["valid-schema-key"].set()
        valid_response = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert valid_response.status_code == 200
    assert json.loads(valid_response.body)["content"][0]["input"] == {
        "error": {"code": "city_not_found"}
    }
    assert touched == ["valid-schema-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_text", 1),
        ("valid_tool_call", 1),
        ("valid_function_call", 1),
        ("empty", 0),
        ("malformed_tool_call", 0),
        ("malformed_function_call", 0),
    ],
)
async def test_converted_messages_cancellation_marks_and_closes_after_adapter_exit(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Late semantic results retry an explicit false mark before runtime close."""

    entered = asyncio.Event()
    release = asyncio.Event()
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="scoped-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> bool:
            assert selected_handle is handle
            lifecycle.append("mark")
            return lifecycle.count("mark") >= 2

        async def close(self) -> None:
            lifecycle.append("close")

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "scoped-key"
        entered.set()
        await release.wait()
        lifecycle.append("adapter-exit")
        if late_outcome == "valid_text":
            return {
                "id": "response-scoped",
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
            }
        if late_outcome == "valid_tool_call":
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
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        if late_outcome == "valid_function_call":
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {
                                "name": "lookup",
                                "arguments": "{}",
                            },
                        },
                        "finish_reason": "function_call",
                    }
                ]
            }
        if late_outcome == "empty":
            return {"choices": []}
        if late_outcome == "malformed_function_call":
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {"arguments": "{}"},
                        },
                        "finish_reason": "function_call",
                    }
                ]
            }
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

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (7, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)

    task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini"),
            current_user=SimpleNamespace(id_int=7),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert lifecycle == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    expected_lifecycle = ["adapter-exit"]
    expected_lifecycle.extend(["mark"] * (2 * expected_marks))
    expected_lifecycle.append("close")
    assert lifecycle == expected_lifecycle


@pytest.mark.asyncio
async def test_concurrent_converted_messages_mark_only_the_credentials_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent adapters cannot cross-wire successful-use callbacks."""

    touched: list[str] = []
    entered = {"key-a": asyncio.Event(), "key-b": asyncio.Event()}
    release = {"key-a": asyncio.Event(), "key-b": asyncio.Event()}
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("openai", "key-a", touched),
            2: _resolved_credentials("openai", "key-b", touched),
        },
    )

    async def _call_adapter(**kwargs: Any) -> dict[str, Any]:
        api_key = kwargs["api_key"]
        entered[api_key].set()
        await release[api_key].wait()
        return {
            "id": f"response-{api_key}",
            "choices": [{"message": {"content": api_key}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _call_adapter)
    request_data = _message_request("openai/gpt-4o-mini")
    first = asyncio.create_task(
        messages_endpoint._handle_messages(
            request_data,
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    second = asyncio.create_task(
        messages_endpoint._handle_messages(
            request_data,
            current_user=SimpleNamespace(id_int=2),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        assert touched == []
        release["key-b"].set()
        await asyncio.wait_for(second, timeout=1.0)
        assert touched == ["key-b"]
        release["key-a"].set()
        await asyncio.wait_for(first, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert touched == ["key-b", "key-a"]


@pytest.mark.asyncio
async def test_converted_invalid_response_does_not_mark_credentials_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "bad-response-key", touched)},
    )

    async def _invalid_adapter(**_kwargs: Any) -> str:
        return "not a response object"

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _invalid_adapter)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert touched == []


@pytest.mark.asyncio
async def test_stream_marks_use_on_first_valid_output_but_not_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    resolutions = {
        1: _resolved_credentials("openai", "output-key", touched),
        2: _resolved_credentials("openai", "empty-key", touched),
    }
    _install_resolutions(monkeypatch, resolutions)

    async def _stream_adapter(**kwargs: Any):
        async def _stream():
            if kwargs["api_key"] == "output-key":
                yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        return _stream()

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)
    output_response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    output_iterator = output_response.body_iterator
    first_output = await output_iterator.__anext__()
    assert "message_start" in first_output
    assert touched == []
    second_output = await output_iterator.__anext__()
    assert "content_block_start" in second_output
    assert touched == []
    third_output = await output_iterator.__anext__()
    assert "text_delta" in third_output
    assert touched == []
    remaining_output = [item async for item in output_iterator]
    assert any("message_stop" in item for item in remaining_output)
    assert touched == ["output-key"]

    empty_response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=2),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    empty_output = [item async for item in empty_response.body_iterator]
    assert len(empty_output) == 1
    assert "event: error" in empty_output[0]
    assert touched == ["output-key"]


@pytest.mark.asyncio
async def test_split_error_event_never_marks_credentials_used() -> None:
    """Transport chunk boundaries cannot turn an error event into success."""

    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "split-error-key", touched)
    chunks = [
        "event: err",
        "or\n",
        'data: {"type":"error","error":{"message":"denied"}}\n\n',
    ]

    async def _stream():
        for chunk in chunks:
            yield chunk

    output = [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ]

    assert output == chunks
    assert touched == []


@pytest.mark.asyncio
async def test_error_event_is_terminal_for_stream_accounting() -> None:
    """Later malformed output cannot recertify a stream after an error event."""

    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "terminal-error-key", touched)
    chunks = [
        'event: error\ndata: {"type":"error","error":{"message":"denied"}}\n\n',
        'event: message_start\ndata: {"type":"message_start","message":{}}\n\n',
    ]

    async def _stream():
        for chunk in chunks:
            yield chunk

    assert [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ] == chunks
    assert touched == []


@pytest.mark.asyncio
async def test_invalid_event_is_terminal_for_stream_accounting() -> None:
    """Malformed provider data cannot be followed by a usage-marking frame."""

    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "invalid-frame-key", touched)
    chunks = [
        "data: private malformed provider diagnostic\n\n",
        'event: content_block_delta\ndata: {"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"later"}}\n\n',
    ]

    async def _stream():
        for chunk in chunks:
            yield chunk

    assert [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ] == chunks
    assert touched == []


@pytest.mark.parametrize(
    "frame",
    [
        ": keepalive\n\n",
        'event: ping\ndata: {"type":"ping"}\n\n',
        "id: stream-7\nretry: 1500\n\n",
    ],
)
@pytest.mark.asyncio
async def test_control_only_event_never_marks_credentials_used(frame: str) -> None:
    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "control-key", touched)

    async def _stream():
        yield frame

    assert [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ] == [frame]
    assert touched == []


@pytest.mark.asyncio
async def test_byte_fragmented_error_event_never_marks_credentials_used() -> None:
    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "byte-error-key", touched)
    chunks = [
        b"eve",
        b"nt: er",
        b"ror\r",
        b'\ndata: {"type":"error","error":{"message":"denied"}}\r\n\r\n',
    ]

    async def _stream():
        for chunk in chunks:
            yield chunk

    output = [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ]

    assert output == chunks
    assert touched == []


@pytest.mark.asyncio
async def test_concurrent_stream_accounting_isolated_for_content_and_split_error() -> None:
    touched: list[str] = []
    entered = {"content-key": asyncio.Event(), "error-key": asyncio.Event()}
    release = asyncio.Event()

    async def _stream(key: str, chunks: list[str]):
        entered[key].set()
        await release.wait()
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    async def _consume(key: str, chunks: list[str]) -> list[str]:
        credentials = _resolved_credentials("anthropic", key, touched)
        return [
            item
            async for item in messages_endpoint._touch_on_first_stream_output(
                _stream(key, chunks),
                _DirectUsageRuntime(),
                credentials,
            )
        ]

    content_chunks = [
        "event: message_start\n"
        "data: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg-accounting",
                    "type": "message",
                    "role": "assistant",
                    "model": "model-a",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            separators=(",", ":"),
        )
        + "\n\n",
        'event: content_block_start\ndata: {"type":"content_block_start",'
        '"index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta",'
        '"index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n',
    ]
    error_chunks = [
        "event: err",
        'or\ndata: {"type":"error","error":{"message":"denied"}}\n\n',
    ]
    content_task = asyncio.create_task(_consume("content-key", content_chunks))
    error_task = asyncio.create_task(_consume("error-key", error_chunks))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        content_output, error_output = await asyncio.wait_for(
            asyncio.gather(content_task, error_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(content_task, error_task, return_exceptions=True)

    assert content_output == content_chunks
    assert error_output == error_chunks
    assert touched == ["content-key"]


def _tool_accounting_frames(case: str) -> list[str]:
    """Build valid and malformed Anthropic tool-stream frame sequences."""
    if case == "orphan-input-delta":
        return [
            "event: content_block_delta\n"
            "data: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"query":"weather"}',
                    },
                },
                separators=(",", ":"),
            )
            + "\n\n"
        ]

    identity: dict[str, Any] = {
        "type": "tool_use",
        "id": "tool-1",
        "name": "lookup_weather",
        "input": {},
    }
    if case == "missing-id":
        identity.pop("id")
    elif case == "blank-id":
        identity["id"] = "  "
    elif case == "nonstring-id":
        identity["id"] = 7
    elif case == "missing-name":
        identity.pop("name")
    elif case == "blank-name":
        identity["name"] = "  "
    elif case == "nonstring-name":
        identity["name"] = ["lookup_weather"]
    elif case != "valid-domain-error-input":
        raise AssertionError(f"Unknown tool accounting case: {case}")

    return [
        "event: content_block_start\n"
        "data: "
        + json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": identity,
            },
            separators=(",", ":"),
        )
        + "\n\n",
        "event: content_block_delta\n"
        "data: "
        + json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"error":{"code":"city_not_found"}}',
                },
            },
            separators=(",", ":"),
        )
        + "\n\n",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "missing-id",
        "blank-id",
        "nonstring-id",
        "missing-name",
        "blank-name",
        "nonstring-name",
        "orphan-input-delta",
    ],
)
async def test_tool_stream_accounting_rejects_invalid_identity_and_orphan_delta(
    case: str,
) -> None:
    """Malformed tool metadata and unowned deltas cannot certify credential use."""
    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "invalid-tool-stream-key", touched)
    frames = _tool_accounting_frames(case)

    async def _stream():
        for frame in frames:
            yield frame

    output = [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ]

    assert output == frames
    assert touched == []


@pytest.mark.asyncio
async def test_tool_stream_accounting_accepts_valid_domain_error_input_control() -> None:
    """A schema-valid tool stream may carry a domain error object as input."""
    touched: list[str] = []
    credentials = _resolved_credentials("anthropic", "valid-tool-stream-key", touched)
    frames = _tool_accounting_frames("valid-domain-error-input")

    async def _stream():
        for frame in frames:
            yield frame

    assert [
        item
        async for item in messages_endpoint._touch_on_first_stream_output(
            _stream(),
            _DirectUsageRuntime(),
            credentials,
        )
    ] == frames
    assert touched == ["valid-tool-stream-key"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_tool_stream_accounting_marks_only_valid_identity() -> None:
    """An invalid tool stream cannot mark itself or a concurrent valid handle."""
    touched: list[str] = []
    entered = {"valid-tool-key": asyncio.Event(), "invalid-tool-key": asyncio.Event()}
    release = asyncio.Event()

    async def _consume(key: str, case: str) -> list[str]:
        frames = _tool_accounting_frames(case)

        async def _stream():
            entered[key].set()
            await release.wait()
            for frame in frames:
                await asyncio.sleep(0)
                yield frame

        credentials = _resolved_credentials("anthropic", key, touched)
        return [
            item
            async for item in messages_endpoint._touch_on_first_stream_output(
                _stream(),
                _DirectUsageRuntime(),
                credentials,
            )
        ]

    valid_task = asyncio.create_task(
        _consume("valid-tool-key", "valid-domain-error-input")
    )
    invalid_task = asyncio.create_task(_consume("invalid-tool-key", "blank-name"))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_output, invalid_output = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert valid_output == _tool_accounting_frames("valid-domain-error-input")
    assert invalid_output == _tool_accounting_frames("blank-name")
    assert touched == ["valid-tool-key"]


@pytest.mark.asyncio
async def test_error_only_converted_stream_is_terminal_without_marking_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upstream error frame must not become an empty successful message."""

    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "error-key", touched)},
    )

    sentinel = "sk-converted-stream-error-/private/provider-stream.json"

    async def _stream_adapter(**_kwargs: Any):
        async def _stream():
            yield (
                'data: {"error":{"message":"'
                f'{sentinel}'
                '","type":"provider_error"}}\n\n'
            )

        return _stream()

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    body = "".join([item async for item in response.body_iterator])

    assert "event: error" in body
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert "message_start" not in body
    assert touched == []


@pytest.mark.asyncio
async def test_converted_stream_exception_is_sanitized_and_closes_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised converted-stream error must not escape or retain credentials."""

    sentinel = "sk-converted-stream-raised-/private/provider-stream.json"
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="raised-stream-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> None:
            raise AssertionError("a failed stream must not be marked used")

        async def close(self) -> None:
            lifecycle.append("runtime-close")

    async def _stream_adapter(**_kwargs: Any):
        async def _stream():
            try:
                raise RuntimeError(sentinel)
                yield ""  # pragma: no cover - keep this an async generator
            finally:
                lifecycle.append("source-close")

        return _stream()

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(
        messages_endpoint,
        "perform_chat_api_call_async",
        _stream_adapter,
    )

    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body = "".join([item async for item in response.body_iterator])

    assert "event: error" in body
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert lifecycle == ["source-close", "runtime-close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["body-aclose", "cancel-pending-read"])
async def test_converted_stream_early_termination_closes_source_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    """Disconnect paths must drain the raw source before releasing credentials."""
    lifecycle: list[str] = []
    source_waiting = asyncio.Event()
    source_closed = asyncio.Event()
    runtime_closed = asyncio.Event()
    never_release = asyncio.Event()
    handle = SimpleNamespace(
        provider="openai",
        api_key="early-close-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> None:
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("runtime-close")
            runtime_closed.set()

    async def _source():
        try:
            yield (
                'data: {"choices":[{"delta":{"content":"valid"},'
                '"finish_reason":null}]}\n\n'
            )
            source_waiting.set()
            await never_release.wait()
        finally:
            lifecycle.append("source-close")
            source_closed.set()

    raw_source = _source()

    async def _stream_adapter(**_kwargs: Any):
        return raw_source

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body_iterator = response.body_iterator

    try:
        assert "message_start" in await asyncio.wait_for(
            body_iterator.__anext__(),
            timeout=1.0,
        )
        if termination == "body-aclose":
            await asyncio.wait_for(body_iterator.aclose(), timeout=1.0)
        else:
            assert "content_block_start" in await asyncio.wait_for(
                body_iterator.__anext__(),
                timeout=1.0,
            )
            assert "text_delta" in await asyncio.wait_for(
                body_iterator.__anext__(),
                timeout=1.0,
            )
            pending = asyncio.create_task(body_iterator.__anext__())
            await asyncio.wait_for(source_waiting.wait(), timeout=1.0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        await asyncio.wait_for(source_closed.wait(), timeout=1.0)
        await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    finally:
        never_release.set()
        with contextlib.suppress(BaseException):
            await body_iterator.aclose()
        with contextlib.suppress(BaseException):
            await raw_source.aclose()

    assert lifecycle.count("source-close") == 1
    assert lifecycle.count("runtime-close") == 1
    assert lifecycle.index("source-close") < lifecycle.index("runtime-close")


@pytest.mark.asyncio
async def test_converted_sanitizer_aclose_closes_wrapped_iterator() -> None:
    """Closing the sanitizer must deterministically close its inner iterator."""

    class _TrackedIterator:
        def __init__(self) -> None:
            self.close_calls = 0
            self._emitted = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            if self._emitted:
                await asyncio.Event().wait()
            self._emitted = True
            return "one-frame"

        async def aclose(self) -> None:
            self.close_calls += 1

    source = _TrackedIterator()
    sanitized = messages_endpoint._sanitize_converted_messages_stream(source)
    assert await sanitized.__anext__() == "one-frame"

    await sanitized.aclose()

    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_converted_message_stop_detaches_blocking_source_close_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal converted stream cannot wait forever on provider cleanup."""

    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()
    runtime_closed = asyncio.Event()
    runtime_close_calls = 0
    mark_calls = 0
    handle = SimpleNamespace(
        provider="openai",
        api_key="blocking-close-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> bool:
            nonlocal mark_calls
            assert selected_handle is handle
            mark_calls += 1
            return True

        async def close(self) -> None:
            nonlocal runtime_close_calls
            runtime_close_calls += 1
            runtime_closed.set()

    class _BlockingCloseSource:
        def __init__(self) -> None:
            self.close_calls = 0
            self._frames = iter(
                (
                    'data: {"choices":[{"delta":{"content":"ok"},'
                    '"finish_reason":null}]}\n\n',
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n',
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._frames)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self) -> None:
            self.close_calls += 1
            close_entered.set()
            try:
                await release_close.wait()
            except asyncio.CancelledError:
                await release_close.wait()
            close_finished.set()

    source = _BlockingCloseSource()

    async def _stream_adapter(**_kwargs: Any):
        return source

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(
        messages_endpoint,
        "perform_chat_api_call_async",
        _stream_adapter,
    )

    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body_iterator = response.body_iterator
    terminal_items: list[str] = []
    while not any("event: message_stop" in item for item in terminal_items):
        terminal_items.append(
            await asyncio.wait_for(body_iterator.__anext__(), timeout=1.0)
        )

    completion = asyncio.create_task(body_iterator.__anext__())
    try:
        await asyncio.wait_for(close_entered.wait(), timeout=1.0)
        done, _pending = await asyncio.wait({completion}, timeout=0.3)
        assert completion in done
        with pytest.raises(StopAsyncIteration):
            completion.result()
        assert source.close_calls == 1
        assert close_finished.is_set() is False
        assert runtime_close_calls == 1
        assert runtime_closed.is_set()
    finally:
        release_close.set()
        await asyncio.gather(completion, return_exceptions=True)
        await asyncio.wait_for(close_finished.wait(), timeout=1.0)
        with contextlib.suppress(BaseException):
            await body_iterator.aclose()

    assert "event: message_stop" in "".join(terminal_items)
    assert source.close_calls == 1
    assert close_finished.is_set()
    assert mark_calls == 1
    assert runtime_close_calls == 1


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_sanitizer_close_is_request_local() -> None:
    """Closing one sanitizer cannot close another request's iterator."""

    class _TrackedIterator:
        def __init__(self, value: str) -> None:
            self.value = value
            self.close_calls = 0
            self._emitted = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            if self._emitted:
                await asyncio.Event().wait()
            self._emitted = True
            return self.value

        async def aclose(self) -> None:
            self.close_calls += 1

    sources = [_TrackedIterator("a"), _TrackedIterator("b")]
    sanitizers = [
        messages_endpoint._sanitize_converted_messages_stream(source)
        for source in sources
    ]
    assert await asyncio.gather(*(stream.__anext__() for stream in sanitizers)) == [
        "a",
        "b",
    ]

    await sanitizers[0].aclose()
    assert [source.close_calls for source in sources] == [1, 0]

    await sanitizers[1].aclose()
    assert [source.close_calls for source in sources] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_stream_aclose_is_request_local_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing one body iterator cannot release another request's runtime."""
    keys = {1: "close-stream-a", 2: "close-stream-b"}
    lifecycle: list[str] = []
    source_closed = {key: asyncio.Event() for key in keys.values()}
    runtime_closed = {key: asyncio.Event() for key in keys.values()}
    sources: dict[str, Any] = {}

    class _Runtime:
        def __init__(self, *, user_id: int, **_kwargs: Any) -> None:
            self.key = keys[user_id]
            self.handle = SimpleNamespace(
                provider="openai",
                api_key=self.key,
                app_config={},
                credentials_resolved=True,
            )

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return self.handle

        async def mark_used(self, _handle: Any) -> None:
            lifecycle.append(f"mark:{self.key}")

        async def close(self) -> None:
            lifecycle.append(f"runtime-close:{self.key}")
            runtime_closed[self.key].set()

    async def _stream_adapter(**kwargs: Any):
        key = kwargs["api_key"]

        async def _source():
            try:
                yield (
                    'data: {"choices":[{"delta":{"content":"valid"},'
                    '"finish_reason":null}]}\n\n'
                )
                await asyncio.Event().wait()
            finally:
                lifecycle.append(f"source-close:{key}")
                source_closed[key].set()

        source = _source()
        sources[key] = source
        return source

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id_int), [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _stream_adapter)

    async def _open(user_id: int):
        response = await messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini", stream=True),
            current_user=SimpleNamespace(id_int=user_id),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
        iterator = response.body_iterator
        assert "message_start" in await asyncio.wait_for(
            iterator.__anext__(),
            timeout=1.0,
        )
        return iterator

    iterator_a, iterator_b = await asyncio.gather(_open(1), _open(2))
    try:
        await asyncio.wait_for(iterator_a.aclose(), timeout=1.0)
        await asyncio.wait_for(source_closed["close-stream-a"].wait(), timeout=1.0)
        await asyncio.wait_for(runtime_closed["close-stream-a"].wait(), timeout=1.0)
        assert not source_closed["close-stream-b"].is_set()
        assert not runtime_closed["close-stream-b"].is_set()

        await asyncio.wait_for(iterator_b.aclose(), timeout=1.0)
        await asyncio.wait_for(source_closed["close-stream-b"].wait(), timeout=1.0)
        await asyncio.wait_for(runtime_closed["close-stream-b"].wait(), timeout=1.0)
    finally:
        await asyncio.gather(
            iterator_a.aclose(),
            iterator_b.aclose(),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(source.aclose() for source in sources.values()),
            return_exceptions=True,
        )

    for key in keys.values():
        assert lifecycle.index(f"source-close:{key}") < lifecycle.index(
            f"runtime-close:{key}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream-factory"])
async def test_converted_factory_exception_is_detached_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """A raised converted response failure maps to one bounded HTTP error."""

    sentinel = "sk-converted-response-raised-/private/provider-response.json"
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="raised-response-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> None:
            raise AssertionError("a failed response must not be marked used")

        async def close(self) -> None:
            lifecycle.append("runtime-close")

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)

    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini", stream=stream),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Upstream provider request failed."
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert lifecycle == ["runtime-close"]


@pytest.mark.parametrize(
    "chunks",
    [
        [
            b"event: err",
            b'or\ndata: {"type":"error","error":{"message":"{sentinel}"}}\n\n',
        ],
        [b"{sentinel}\n\n"],
        [b"data: {sentinel}\n\n"],
        [
            b'event: provider_error\ndata: {"type":"provider_error",'
            b'"message":"{sentinel}"}\n\n'
        ],
        [b'data: {"type":"error","error":{"message":"{sentinel}"}}'],
    ],
)
@pytest.mark.asyncio
async def test_native_stream_error_is_sanitized_closed_and_not_marked(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[bytes],
) -> None:
    """Native 200/SSE error events cannot expose provider text or retain resources."""

    sentinel = "sk-native-sse-error-/private/provider-stream.json"
    touched: list[str] = []
    source_closed = asyncio.Event()
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "native-error-key", touched)},
    )

    async def _source():
        try:
            for chunk in chunks:
                yield chunk.replace(b"{sentinel}", sentinel.encode())
        finally:
            source_closed.set()

    async def _prepare_stream(*_args: Any, **_kwargs: Any):
        return _source()

    monkeypatch.setattr(
        messages_endpoint,
        "_prepare_native_stream_iterator",
        _prepare_stream,
    )
    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-3-5-sonnet", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    body_items = [
        item.decode() if isinstance(item, bytes) else str(item)
        async for item in response.body_iterator
    ]
    body = "".join(body_items)

    assert "event: error" in body
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert touched == []
    assert source_closed.is_set()


@pytest.mark.asyncio
async def test_native_stream_complete_output_is_forwarded_marked_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    source_closed = asyncio.Event()
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "native-success-key", touched)},
    )
    structural_frame = (
        'event: message_start\ndata: {"type":"message_start",'
        '"message":{"id":"msg-native-success","type":"message",'
        '"role":"assistant","model":"claude-test","content":[],'
        '"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":2,"output_tokens":0}}}\n\n'
    )
    block_start_frame = (
        'event: content_block_start\ndata: {"type":"content_block_start",'
        '"index":0,"content_block":{"type":"text","text":""}}\n\n'
    )
    content_frame = (
        'event: content_block_delta\ndata: {"type":"content_block_delta",'
        '"index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
    )
    block_stop_frame = (
        'event: content_block_stop\ndata: {"type":"content_block_stop",'
        '"index":0}\n\n'
    )
    message_delta_frame = (
        'event: message_delta\ndata: {"type":"message_delta",'
        '"delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"output_tokens":1}}\n\n'
    )
    message_stop_frame = (
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    release_content = asyncio.Event()

    async def _source():
        try:
            yield structural_frame[:17].encode()
            yield structural_frame[17:].encode()
            await release_content.wait()
            yield block_start_frame.encode()
            yield content_frame.encode()
            yield block_stop_frame.encode()
            yield message_delta_frame.encode()
            yield message_stop_frame.encode()
        finally:
            source_closed.set()

    async def _prepare_stream(*_args: Any, **_kwargs: Any):
        return _source()

    monkeypatch.setattr(
        messages_endpoint,
        "_prepare_native_stream_iterator",
        _prepare_stream,
    )
    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-3-5-sonnet", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    body_iterator = response.body_iterator
    first_item = await asyncio.wait_for(body_iterator.__anext__(), timeout=1.0)
    assert first_item == structural_frame
    assert touched == []
    release_content.set()
    remaining = [item async for item in body_iterator]

    assert "".join([first_item, *remaining]) == (
        structural_frame
        + block_start_frame
        + content_frame
        + block_stop_frame
        + message_delta_frame
        + message_stop_frame
    )
    assert touched == ["native-success-key"]
    assert source_closed.is_set()


@pytest.mark.asyncio
async def test_native_stream_cancellation_closes_source_before_credential_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected Messages stream cannot outlive its credential runtime."""

    source_waiting = asyncio.Event()
    release_source = asyncio.Event()
    source_closed = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="anthropic",
        api_key="cancel-key",
        app_config={
            "anthropic_api": {"api_base_url": "https://anthropic.example/v1"}
        },
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> None:
            raise AssertionError("structural-only stream must not be marked used")

        async def close(self) -> None:
            lifecycle.append("runtime-close")
            runtime_closed.set()

    structural_frame = (
        'event: message_start\ndata: {"type":"message_start",'
        '"message":{"id":"msg-native-cancel","type":"message",'
        '"role":"assistant","model":"claude-test","content":[],'
        '"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":2,"output_tokens":0}}}\n\n'
    )

    async def _source():
        try:
            yield structural_frame.encode()
            source_waiting.set()
            await release_source.wait()
        finally:
            lifecycle.append("source-close")
            source_closed.set()

    async def _prepare_stream(*_args: Any, **_kwargs: Any):
        return _source()

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(
        messages_endpoint,
        "_prepare_native_stream_iterator",
        _prepare_stream,
    )

    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-3-5-sonnet", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body_iterator = response.body_iterator
    assert await body_iterator.__anext__() == structural_frame

    pending = asyncio.create_task(body_iterator.__anext__())
    try:
        await asyncio.wait_for(source_waiting.wait(), timeout=1.0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(source_closed.wait(), timeout=1.0)
        await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    finally:
        release_source.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await body_iterator.aclose()

    assert lifecycle == ["source-close", "runtime-close"]


@pytest.mark.asyncio
async def test_error_only_converted_response_is_rejected_without_marking_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-stream upstream error object must remain an endpoint failure."""

    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", "error-key", touched)},
    )

    async def _error_adapter(**_kwargs: Any) -> dict[str, Any]:
        return {"error": {"message": "upstream denied", "type": "provider_error"}}

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _error_adapter)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/gpt-4o-mini"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert touched == []


@pytest.mark.asyncio
async def test_bedrock_default_chain_auth_reaches_messages_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock's resolved AWS default chain does not require a bearer API key."""

    touched: list[str] = []

    async def _touch() -> None:
        touched.append("bedrock-default-chain")

    credentials = ResolvedByokCredentials(
        provider="bedrock",
        api_key=None,
        app_config={"bedrock_api": {"_runtime_auth_source": "aws_default_chain"}},
        credential_fields={},
        source="server",
        allowlisted=True,
        _touch_cb=_touch,
    )
    _install_resolutions(monkeypatch, {1: credentials})

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_provider"] == "bedrock"
        assert kwargs["api_key"] is None
        assert kwargs["credentials_resolved"] is True
        return {
            "id": "bedrock-response",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("bedrock/anthropic.claude-3-haiku"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert touched == ["bedrock-default-chain"]


def _native_message_response(case: str, sentinel: str) -> dict[str, Any]:
    valid = {
        "id": "msg-native-1",
        "type": "message",
        "role": "assistant",
        "model": "claude-native",
        "content": [{"type": "text", "text": "valid native output"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    if case == "valid-text":
        return valid
    if case == "valid-tool-use":
        return {
            **valid,
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "lookup",
                    "input": {"query": "weather"},
                }
            ],
            "stop_reason": "tool_use",
        }
    if case == "valid-tool-use-domain-error":
        return {
            **valid,
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-domain-error",
                    "name": "lookup_weather",
                    "input": {
                        "error": {
                            "code": "city_not_found",
                            "message": "No matching city",
                        }
                    },
                }
            ],
            "stop_reason": "tool_use",
        }
    if case == "valid-noncanonical-json":
        return {
            **valid,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "error": {
                                "code": "fictional_story_error",
                                "message": "plot device",
                            }
                        }
                    ),
                }
            ],
        }
    invalid_tool_inputs: dict[str, Any] = {
        "tool-input-missing": None,
        "tool-input-null": None,
        "tool-input-list": [],
        "tool-input-string": "not-an-object",
    }
    if case in invalid_tool_inputs:
        content_block = {
            "type": "tool_use",
            "id": "tool-invalid-input",
            "name": "lookup_weather",
        }
        if case != "tool-input-missing":
            content_block["input"] = invalid_tool_inputs[case]
        return {
            **valid,
            "content": [content_block],
            "stop_reason": "tool_use",
        }
    invalid_usage: dict[str, tuple[str, Any]] = {
        "usage-input-boolean": ("input_tokens", True),
        "usage-input-negative": ("input_tokens", -1),
        "usage-input-string": ("input_tokens", "2"),
        "usage-input-float": ("input_tokens", 2.5),
        "usage-output-boolean": ("output_tokens", False),
        "usage-output-negative": ("output_tokens", -1),
        "usage-output-string": ("output_tokens", "3"),
        "usage-output-float": ("output_tokens", 3.5),
    }
    if case in invalid_usage:
        field, value = invalid_usage[case]
        usage = dict(valid["usage"])
        usage[field] = value
        return {**valid, "usage": usage}
    if case == "top-level-error":
        return {"error": {"type": "api_error", "message": sentinel}}
    if case == "empty-object":
        return {}
    if case == "empty-content":
        return {**valid, "content": []}
    if case == "missing-usage":
        return {key: value for key, value in valid.items() if key != "usage"}
    if case == "mixed-top-level-error":
        return {**valid, "error": {"type": "api_error", "message": sentinel}}
    if case == "nested-error-block":
        return {
            **valid,
            "content": [
                {"type": "text", "text": "valid native output"},
                {"type": "error", "error": {"message": sentinel}},
            ],
        }
    raise AssertionError(f"Unknown native message case: {case}")


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "surface",
    [
        "native-nonstream",
        "converted-nonstream",
        "native-stream",
        "converted-stream",
        "count-tokens",
    ],
)
async def test_concurrent_messages_retries_explicit_false_credential_mark(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    """Every successful Messages surface retries false marks request-locally."""

    keys = {1: f"{surface}-retry", 2: f"{surface}-healthy"}
    provider_ready = {key: asyncio.Event() for key in keys.values()}
    release_provider = asyncio.Event()
    mark_ready = {key: asyncio.Event() for key in keys.values()}
    release_mark = {key: asyncio.Event() for key in keys.values()}
    mark_attempts: dict[str, int] = dict.fromkeys(keys.values(), 0)
    marked: list[str] = []
    close_calls: dict[str, int] = dict.fromkeys(keys.values(), 0)

    class _Runtime:
        def __init__(self, *, user_id: int, **_kwargs: Any) -> None:
            self.key = keys[user_id]
            self.handle: Any | None = None

        async def resolve(self, provider: str, *, model: str | None = None):
            del model
            self.handle = SimpleNamespace(
                provider=provider,
                api_key=self.key,
                app_config={},
                credentials_resolved=True,
            )
            return self.handle

        async def mark_used(self, selected_handle: Any) -> bool:
            assert selected_handle is self.handle
            mark_attempts[self.key] += 1
            if self.key.endswith("-retry") and mark_attempts[self.key] == 1:
                return False
            mark_ready[self.key].set()
            await release_mark[self.key].wait()
            marked.append(self.key)
            return True

        async def close(self) -> None:
            close_calls[self.key] += 1

    async def _native_post(
        _url: str,
        headers: dict[str, str],
        _payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        key = headers["x-api-key"]
        provider_ready[key].set()
        await release_provider.wait()
        if surface == "count-tokens":
            return {"input_tokens": 4}
        return _native_message_response("valid-text", "unused")

    native_frames = (
        'event: message_start\ndata: {"type":"message_start",'
        '"message":{"id":"msg-retry","type":"message",'
        '"role":"assistant","model":"claude-test","content":[],'
        '"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":2,"output_tokens":0}}}\n\n'
        'event: content_block_start\ndata: {"type":"content_block_start",'
        '"index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta",'
        '"index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
        'event: content_block_stop\ndata: {"type":"content_block_stop",'
        '"index":0}\n\n'
        'event: message_delta\ndata: {"type":"message_delta",'
        '"delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"output_tokens":1}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def _native_stream(
        _url: str,
        headers: dict[str, str],
        _payload: dict[str, Any],
        **_kwargs: Any,
    ):
        key = headers["x-api-key"]
        provider_ready[key].set()

        async def _source():
            await release_provider.wait()
            yield native_frames.encode()

        return _source()

    async def _converted_adapter(**kwargs: Any):
        key = kwargs["api_key"]
        if surface == "converted-stream":
            provider_ready[key].set()

            async def _source():
                await release_provider.wait()
                yield (
                    'data: {"choices":[{"delta":{"content":"ok"},'
                    '"finish_reason":null}]}\n\n'
                )
                yield (
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
                )

            return _source()

        provider_ready[key].set()
        await release_provider.wait()
        return {
            "id": f"response-{key}",
            "choices": [
                {"message": {"content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id_int), [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_post)
    monkeypatch.setattr(
        messages_endpoint,
        "_prepare_native_stream_iterator",
        _native_stream,
    )
    monkeypatch.setattr(
        messages_endpoint,
        "perform_chat_api_call_async",
        _converted_adapter,
    )

    async def _invoke(user_id: int) -> Any:
        user = SimpleNamespace(id_int=user_id)
        if surface == "count-tokens":
            response = await messages_endpoint._handle_count_tokens(
                AnthropicCountTokensRequest(
                    model="anthropic/claude-3-5-haiku-latest",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                current_user=user,
                request=SimpleNamespace(),
                anthropic_version=None,
                anthropic_beta=None,
            )
            return json.loads(response.body)

        native = surface.startswith("native-")
        response = await messages_endpoint._handle_messages(
            _message_request(
                (
                    "anthropic/claude-3-5-haiku-latest"
                    if native
                    else "openai/gpt-4o-mini"
                ),
                stream=surface.endswith("-stream"),
            ),
            current_user=user,
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
        if surface.endswith("-stream"):
            items = [
                item.decode() if isinstance(item, (bytes, bytearray)) else str(item)
                async for item in response.body_iterator
            ]
            return "".join(items)
        return json.loads(response.body)

    tasks = [asyncio.create_task(_invoke(user_id)) for user_id in (1, 2)]
    retry_key = keys[1]
    healthy_key = keys[2]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in provider_ready.values())),
            timeout=1.0,
        )
        release_provider.set()
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in mark_ready.values())),
            timeout=1.0,
        )
        assert mark_attempts == {retry_key: 2, healthy_key: 1}
        assert all(task.done() is False for task in tasks)

        release_mark[healthy_key].set()
        healthy_result = await asyncio.wait_for(
            asyncio.shield(tasks[1]),
            timeout=1.0,
        )
        assert tasks[0].done() is False
        assert marked == [healthy_key]

        release_mark[retry_key].set()
        retry_result = await asyncio.wait_for(
            asyncio.shield(tasks[0]),
            timeout=1.0,
        )
    finally:
        release_provider.set()
        for event in release_mark.values():
            event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if surface.endswith("-stream"):
        assert "event: message_stop" in healthy_result
        assert "event: message_stop" in retry_result
    elif surface == "count-tokens":
        assert healthy_result == {"input_tokens": 4}
        assert retry_result == {"input_tokens": 4}
    else:
        assert healthy_result
        assert retry_result
    assert mark_attempts == {retry_key: 2, healthy_key: 1}
    assert marked == [healthy_key, retry_key]
    assert close_calls == {retry_key: 1, healthy_key: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["native", "converted"])
async def test_stream_cancellation_drains_false_mark_retry_before_close(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    """Cancellation after a false mark must drain its retry before teardown."""

    first_false = asyncio.Event()
    cancellation_sent = asyncio.Event()
    lifecycle: list[str] = []
    mark_attempts = 0

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            self.handle: Any | None = None

        async def resolve(self, provider: str, *, model: str | None = None):
            del model
            self.handle = SimpleNamespace(
                provider=provider,
                api_key=f"{surface}-retry-key",
                app_config={
                    "anthropic_api": {
                        "api_base_url": "https://anthropic.example/v1"
                    }
                },
                credentials_resolved=True,
            )
            return self.handle

        async def mark_used(self, selected_handle: Any) -> bool:
            nonlocal mark_attempts
            assert selected_handle is self.handle
            mark_attempts += 1
            lifecycle.append(f"mark:{mark_attempts}")
            if mark_attempts == 1:
                first_false.set()
                return False
            return True

        async def close(self) -> None:
            lifecycle.append("runtime-close")

    native_frames = (
        'event: message_start\ndata: {"type":"message_start",'
        '"message":{"id":"msg-cancel-retry","type":"message",'
        '"role":"assistant","model":"claude-test","content":[],'
        '"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":2,"output_tokens":0}}}\n\n'
        'event: content_block_start\ndata: {"type":"content_block_start",'
        '"index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta",'
        '"index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
        'event: content_block_stop\ndata: {"type":"content_block_stop",'
        '"index":0}\n\n'
        'event: message_delta\ndata: {"type":"message_delta",'
        '"delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"output_tokens":1}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def _source():
        try:
            if surface == "native":
                yield native_frames.encode()
                return
            yield (
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":null}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
            )
        finally:
            lifecycle.append("source-close")

    async def _native_stream(*_args: Any, **_kwargs: Any):
        return _source()

    async def _converted_stream(**_kwargs: Any):
        return _source()

    monkeypatch.setattr(
        messages_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(
        messages_endpoint,
        "_prepare_native_stream_iterator",
        _native_stream,
    )
    monkeypatch.setattr(
        messages_endpoint,
        "perform_chat_api_call_async",
        _converted_stream,
    )

    response = await messages_endpoint._handle_messages(
        _message_request(
            (
                "anthropic/claude-3-5-haiku-latest"
                if surface == "native"
                else "openai/gpt-4o-mini"
            ),
            stream=True,
        ),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    async def _consume() -> None:
        async for _item in response.body_iterator:
            pass

    consumer: asyncio.Task[None] | None = None

    async def _cancel_after_false() -> None:
        await first_false.wait()
        assert consumer is not None
        consumer.cancel()
        cancellation_sent.set()

    canceller = asyncio.create_task(_cancel_after_false())
    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(cancellation_sent.wait(), timeout=1.0)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(consumer), timeout=1.0)
    finally:
        if not consumer.done():
            consumer.cancel()
        if not canceller.done():
            canceller.cancel()
        await asyncio.gather(consumer, canceller, return_exceptions=True)
        with contextlib.suppress(BaseException):
            await response.body_iterator.aclose()

    assert mark_attempts == 2
    assert lifecycle == ["mark:1", "mark:2", "source-close", "runtime-close"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "top-level-error",
        "empty-object",
        "empty-content",
        "missing-usage",
        "mixed-top-level-error",
        "nested-error-block",
        "tool-input-missing",
        "tool-input-null",
        "tool-input-list",
        "tool-input-string",
        "usage-input-boolean",
        "usage-input-negative",
        "usage-input-string",
        "usage-input-float",
        "usage-output-boolean",
        "usage-output-negative",
        "usage-output-string",
        "usage-output-float",
    ],
)
async def test_native_nonstream_messages_rejects_invalid_200_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """HTTP 200 is not success until the native message payload is semantic."""
    sentinel = "native-message-200-secret-/private/provider.json"
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "native-invalid-key", touched)},
    )

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _native_message_response(case, sentinel)

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("anthropic/claude-3-5-haiku-latest"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert sentinel not in str(exc_info.value)
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "valid-text",
        "valid-tool-use",
        "valid-tool-use-domain-error",
        "valid-noncanonical-json",
    ],
)
async def test_native_nonstream_messages_marks_only_semantic_200(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Valid native text and tool-use payloads retain compatibility."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "native-valid-key", touched)},
    )
    payload = _native_message_response(case, "unused")

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-3-5-haiku-latest"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == payload
    assert touched == ["native-valid-key"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_messages_marks_only_valid_200_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed native 200 cannot mark itself or a concurrent valid request."""
    touched: list[str] = []
    entered = {"native-valid-a": asyncio.Event(), "native-error-b": asyncio.Event()}
    release = {"native-valid-a": asyncio.Event(), "native-error-b": asyncio.Event()}
    sentinel = "native-message-concurrent-secret-/private/provider.json"
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("anthropic", "native-valid-a", touched),
            2: _resolved_credentials("anthropic", "native-error-b", touched),
        },
    )

    async def _native(
        _url: str,
        headers: dict[str, str],
        _payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        api_key = headers["x-api-key"]
        entered[api_key].set()
        await release[api_key].wait()
        if api_key == "native-valid-a":
            return _native_message_response("valid-text", sentinel)
        return _native_message_response("nested-error-block", sentinel)

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    valid_task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("anthropic/claude-3-5-haiku-latest"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    error_task = asyncio.create_task(
        messages_endpoint._handle_messages(
            _message_request("anthropic/claude-3-5-haiku-latest"),
            current_user=SimpleNamespace(id_int=2),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["native-error-b"].set()
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(error_task, timeout=1.0)
        assert exc_info.value.status_code == 502
        assert touched == []
        release["native-valid-a"].set()
        valid_response = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert valid_response.status_code == 200
    assert touched == ["native-valid-a"]
    assert sentinel not in str(exc_info.value)


@pytest.mark.asyncio
async def test_successful_native_count_tokens_marks_credentials_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "count-key", touched)},
    )

    async def _count(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {"input_tokens": 4}

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _count)
    response = await messages_endpoint._handle_count_tokens(
        AnthropicCountTokensRequest(
            model="anthropic/claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "hello"}],
        ),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert touched == ["count-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"type": "api_error", "message": "count-secret"}},
        {},
        {"input_tokens": "4"},
        {"input_tokens": True},
        {"input_tokens": -1},
        {
            "input_tokens": 4,
            "error": {"type": "api_error", "message": "count-secret"},
        },
    ],
    ids=[
        "error-envelope",
        "missing-count",
        "wrong-type",
        "boolean",
        "negative",
        "mixed",
    ],
)
async def test_native_count_tokens_rejects_invalid_200_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    """Count-token accounting requires one nonnegative integer result."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "count-invalid-key", touched)},
    )

    async def _count(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _count)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_count_tokens(
            AnthropicCountTokensRequest(
                model="anthropic/claude-3-5-haiku-latest",
                messages=[{"role": "user", "content": "hello"}],
            ),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert "count-secret" not in str(exc_info.value)
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("input_tokens", [0, 4])
async def test_native_count_tokens_marks_valid_200(
    monkeypatch: pytest.MonkeyPatch,
    input_tokens: int,
) -> None:
    """Zero and positive native token counts remain compatible."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "count-valid-key", touched)},
    )

    async def _count(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {"input_tokens": input_tokens}

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _count)
    response = await messages_endpoint._handle_count_tokens(
        AnthropicCountTokensRequest(
            model="anthropic/claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "hello"}],
        ),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"input_tokens": input_tokens}
    assert touched == ["count-valid-key"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_count_tokens_marks_only_valid_200_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid concurrent count payloads cannot cross-wire usage accounting."""
    touched: list[str] = []
    entered = {"count-valid-a": asyncio.Event(), "count-error-b": asyncio.Event()}
    release = {"count-valid-a": asyncio.Event(), "count-error-b": asyncio.Event()}
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("anthropic", "count-valid-a", touched),
            2: _resolved_credentials("anthropic", "count-error-b", touched),
        },
    )

    async def _count(
        _url: str,
        headers: dict[str, str],
        _payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        api_key = headers["x-api-key"]
        entered[api_key].set()
        await release[api_key].wait()
        if api_key == "count-valid-a":
            return {"input_tokens": 4}
        return {
            "input_tokens": 4,
            "error": {"type": "api_error", "message": "count-secret"},
        }

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _count)

    def _request() -> AnthropicCountTokensRequest:
        return AnthropicCountTokensRequest(
            model="anthropic/claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "hello"}],
        )

    valid_task = asyncio.create_task(
        messages_endpoint._handle_count_tokens(
            _request(),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    error_task = asyncio.create_task(
        messages_endpoint._handle_count_tokens(
            _request(),
            current_user=SimpleNamespace(id_int=2),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["count-error-b"].set()
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(error_task, timeout=1.0)
        assert exc_info.value.status_code == 502
        assert touched == []
        release["count-valid-a"].set()
        valid_response = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert valid_response.status_code == 200
    assert touched == ["count-valid-a"]
    assert "count-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_failed_native_message_does_not_mark_credentials_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "failed-key", touched)},
    )

    async def _fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise HTTPException(status_code=502, detail="bounded failure")

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _fail)
    with pytest.raises(HTTPException):
        await messages_endpoint._handle_messages(
            _message_request("anthropic/claude-3-5-haiku-latest"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert touched == []


class _ConvertedSSEBoundaryRuntime:
    """Request-local usage/close recorder for converted SSE boundary tests."""

    def __init__(self, key: str, marks: list[str], lifecycle: list[str]) -> None:
        self.key = key
        self.marks = marks
        self.lifecycle = lifecycle

    async def mark_used(self, _credentials: Any) -> None:
        self.marks.append(self.key)

    async def close(self) -> None:
        self.lifecycle.append(f"runtime-close:{self.key}")


def _openai_sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _consume_converted_sse_boundary(
    *,
    key: str,
    frames: list[str],
    marks: list[str],
    lifecycle: list[str],
    entered: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> str:
    async def _source():
        try:
            if entered is not None:
                entered.set()
            if release is not None:
                await release.wait()
            for frame in frames:
                yield frame
        finally:
            lifecycle.append(f"source-close:{key}")

    runtime = _ConvertedSSEBoundaryRuntime(key, marks, lifecycle)
    credentials = SimpleNamespace(provider="openai", api_key=key)
    body = "".join(
        [
            item
            async for item in messages_endpoint._touch_on_first_stream_output(
                messages_endpoint._sanitize_converted_messages_stream(
                    messages_endpoint.openai_stream_to_anthropic(
                        _source(),
                        model="model-a",
                    )
                ),
                runtime,
                credentials,
            )
        ]
    )
    source_close = f"source-close:{key}"
    runtime_close = f"runtime-close:{key}"
    assert lifecycle.count(source_close) == 1
    assert lifecycle.count(runtime_close) == 1
    assert lifecycle.index(source_close) < lifecycle.index(runtime_close)
    return body


def _converted_usage_frames(
    field: str,
    value: Any,
    *,
    sentinel: str | None = None,
) -> list[str]:
    usage: dict[str, Any] = {"prompt_tokens": 2, "completion_tokens": 3}
    usage[field] = value
    if sentinel is not None:
        usage["diagnostic"] = {"message": sentinel}
    return [
        _openai_sse_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "apparently valid"},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _openai_sse_frame(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage_field", "invalid_value"),
    [
        pytest.param("prompt_tokens", True, id="input-boolean"),
        pytest.param("prompt_tokens", -1, id="input-negative"),
        pytest.param("completion_tokens", False, id="output-boolean"),
        pytest.param("completion_tokens", -1, id="output-negative"),
    ],
)
async def test_converted_stream_rejects_invalid_usage_before_mark(
    usage_field: str,
    invalid_value: Any,
) -> None:
    """Invalid final usage cannot certify an otherwise plausible stream."""
    sentinel = f"converted-{usage_field}-usage-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=f"invalid-{usage_field}-{invalid_value!r}",
        frames=_converted_usage_frames(
            usage_field,
            invalid_value,
            sentinel=sentinel,
        ),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert marks == []


@pytest.mark.asyncio
async def test_converted_stream_accepts_nonnegative_integer_usage_control() -> None:
    """Schema-valid input/output usage remains compatible and marks once."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key="valid-stream-usage",
        frames=_converted_usage_frames("completion_tokens", 3),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert '"input_tokens": 2' in body
    assert '"output_tokens": 3' in body
    assert marks == ["valid-stream-usage"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_stream_usage_validation_is_request_local() -> None:
    """Invalid usage cannot mark itself or contaminate a valid concurrent stream."""
    sentinel = "converted-concurrent-usage-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("usage-valid", "usage-invalid")}
    release = asyncio.Event()

    valid_task = asyncio.create_task(
        _consume_converted_sse_boundary(
            key="usage-valid",
            frames=_converted_usage_frames("prompt_tokens", 2),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["usage-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_converted_sse_boundary(
            key="usage-invalid",
            frames=_converted_usage_frames(
                "completion_tokens",
                -1,
                sentinel=sentinel,
            ),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["usage-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["usage-valid"]


def _openai_tool_delta_frame(
    *,
    tool_delta: dict[str, Any],
    finish_reason: str | None,
) -> str:
    return _openai_sse_frame(
        {
            "choices": [
                {
                    "delta": {"tool_calls": [tool_delta]},
                    "finish_reason": finish_reason,
                }
            ]
        }
    )


def _converted_tool_frames(case: str, sentinel: str) -> list[str]:
    if case == "missing-identity":
        return [
            _openai_tool_delta_frame(
                tool_delta={
                    "index": 0,
                    "function": {
                        "arguments": json.dumps({"probe": sentinel}),
                    },
                },
                finish_reason="tool_calls",
            )
        ]
    if case == "missing-name":
        return [
            _openai_tool_delta_frame(
                tool_delta={
                    "index": 0,
                    "id": "call-missing-name",
                    "function": {
                        "arguments": json.dumps({"probe": sentinel}),
                    },
                },
                finish_reason="tool_calls",
            )
        ]
    if case == "arguments-before-identity":
        full_arguments = json.dumps({"probe": sentinel}, separators=(",", ":"))
        split_at = max(1, len(full_arguments) // 2)
        return [
            _openai_tool_delta_frame(
                tool_delta={
                    "index": 0,
                    "function": {"arguments": full_arguments[:split_at]},
                },
                finish_reason=None,
            ),
            _openai_tool_delta_frame(
                tool_delta={
                    "index": 0,
                    "id": "call-late-identity",
                    "function": {
                        "name": "lookup",
                        "arguments": full_arguments[split_at:],
                    },
                },
                finish_reason="tool_calls",
            ),
        ]

    arguments = {
        "malformed-json": f"not-json-{sentinel}",
        "incomplete-json": f'{{"probe":"{sentinel}"',
        "array-json": json.dumps([sentinel]),
        "null-json": "null",
        "string-json": json.dumps(sentinel),
    }.get(case)
    if arguments is None:
        raise AssertionError(f"Unknown converted tool stream case: {case}")
    return [
        _openai_tool_delta_frame(
            tool_delta={
                "index": 0,
                "id": f"call-{case}",
                "function": {"name": "lookup", "arguments": arguments},
            },
            finish_reason="tool_calls",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "missing-identity",
        "missing-name",
        "arguments-before-identity",
        "malformed-json",
        "incomplete-json",
        "array-json",
        "null-json",
        "string-json",
    ],
)
async def test_converted_tool_stream_rejects_incomplete_or_nonobject_state(
    case: str,
) -> None:
    """Only complete tool identity plus object JSON can certify a tool stream."""
    sentinel = f"converted-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=case,
        frames=_converted_tool_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert marks == []


def _valid_domain_error_tool_frames() -> list[str]:
    return [
        _openai_tool_delta_frame(
            tool_delta={
                "index": 0,
                "id": "call-domain-error",
                "function": {
                    "name": "lookup_weather",
                    "arguments": json.dumps(
                        {
                            "error": {
                                "code": "city_not_found",
                                "message": "No matching city",
                            }
                        },
                        separators=(",", ":"),
                    ),
                },
            },
            finish_reason="tool_calls",
        )
    ]


@pytest.mark.asyncio
async def test_converted_tool_stream_accepts_domain_error_object_control() -> None:
    """A completed domain-error object remains valid tool input."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key="valid-domain-tool",
        frames=_valid_domain_error_tool_frames(),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert "city_not_found" in body
    assert marks == ["valid-domain-tool"]


@pytest.mark.asyncio
async def test_cancelled_incomplete_tool_stream_closes_without_usage_mark() -> None:
    """Cancellation cannot certify partial tool JSON or release credentials early."""
    sentinel = "converted-cancel-tool-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    source_waiting = asyncio.Event()
    source_closed = asyncio.Event()
    runtime_closed = asyncio.Event()
    never_release = asyncio.Event()
    key = "cancel-incomplete-tool"
    observed: list[str] = []

    async def _source():
        try:
            yield _openai_tool_delta_frame(
                tool_delta={
                    "index": 0,
                    "id": "call-cancel-incomplete",
                    "function": {
                        "name": "lookup",
                        "arguments": f'{{"probe":"{sentinel}"',
                    },
                },
                finish_reason=None,
            )
            source_waiting.set()
            await never_release.wait()
        finally:
            lifecycle.append(f"source-close:{key}")
            source_closed.set()

    class _CancellationRuntime(_ConvertedSSEBoundaryRuntime):
        async def close(self) -> None:
            await super().close()
            runtime_closed.set()

    runtime = _CancellationRuntime(key, marks, lifecycle)
    credentials = SimpleNamespace(provider="openai", api_key=key)

    async def _consume() -> str:
        async for item in messages_endpoint._touch_on_first_stream_output(
            messages_endpoint._sanitize_converted_messages_stream(
                messages_endpoint.openai_stream_to_anthropic(
                    _source(),
                    model="model-a",
                )
            ),
            runtime,
            credentials,
        ):
            observed.append(item)
        return "".join(observed)

    task = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(source_waiting.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(source_closed.wait(), timeout=1.0)
        await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    finally:
        never_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert marks == []
    assert sentinel not in "".join(observed)
    source_close = f"source-close:{key}"
    runtime_close = f"runtime-close:{key}"
    assert lifecycle.count(source_close) == 1
    assert lifecycle.count(runtime_close) == 1
    assert lifecycle.index(source_close) < lifecycle.index(runtime_close)


def _converted_shifted_tool_index_frames(sentinel: str) -> list[str]:
    arguments = json.dumps({"probe": sentinel}, separators=(",", ":"))
    split_at = max(1, len(arguments) // 2)
    return [
        _openai_tool_delta_frame(
            tool_delta={
                "index": 0,
                "id": "call-shifted-index",
                "function": {
                    "name": "lookup",
                    "arguments": arguments[:split_at],
                },
            },
            finish_reason=None,
        ),
        _openai_tool_delta_frame(
            tool_delta={
                "index": 1,
                "id": "call-shifted-index",
                "function": {"arguments": arguments[split_at:]},
            },
            finish_reason="tool_calls",
        ),
    ]


def _converted_ordered_parallel_tool_frames() -> list[str]:
    calls = [
        {
            "index": index,
            "id": f"call-ordered-{index}",
            "function": {
                "name": f"ordered_tool_{index}",
                "arguments": json.dumps(
                    {"position": index},
                    separators=(",", ":"),
                ),
            },
        }
        for index in range(2)
    ]
    return [
        _openai_sse_frame(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": calls},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    ]


@pytest.mark.asyncio
async def test_converted_tool_stream_rejects_provider_index_drift_before_mark() -> None:
    """A tool identity cannot move between provider indexes mid-stream."""
    sentinel = "converted-shifted-index-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key="shifted-index",
        frames=_converted_shifted_tool_index_frames(sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body
    assert marks == []


@pytest.mark.asyncio
async def test_converted_tool_stream_preserves_ordered_parallel_calls_control() -> None:
    """Stable provider indexes preserve parallel tool identity and order."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key="ordered-parallel",
        frames=_converted_ordered_parallel_tool_frames(),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert body.index("call-ordered-0") < body.index("call-ordered-1")
    assert body.index("ordered_tool_0") < body.index("ordered_tool_1")
    assert body.count("event: message_stop") == 1
    assert marks == ["ordered-parallel"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_tool_index_identity_is_request_local() -> None:
    """Index drift cannot mark itself or contaminate a valid concurrent call."""
    sentinel = "converted-concurrent-tool-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("tool-valid", "tool-invalid")}
    release = asyncio.Event()

    valid_task = asyncio.create_task(
        _consume_converted_sse_boundary(
            key="tool-valid",
            frames=_converted_ordered_parallel_tool_frames(),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["tool-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_converted_sse_boundary(
            key="tool-invalid",
            frames=_converted_shifted_tool_index_frames(sentinel),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["tool-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert "call-ordered-0" in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["tool-valid"]


def _converted_open_text_frame(text: str = "partial valid text") -> str:
    return _openai_sse_frame(
        {
            "choices": [
                {
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ]
        }
    )


def _converted_invalid_terminal_frames(case: str, sentinel: str) -> list[str]:
    frames = [_converted_open_text_frame()]
    if case == "premature-eof":
        return frames
    if case == "bare-done":
        return [*frames, "data: [DONE]\n\n"]
    if case == "malformed-data-json":
        return [*frames, f'data: {{"diagnostic":"{sentinel}"\n\n']
    if case == "malformed-final-usage-json":
        return [
            *frames,
            (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                f'"usage":{{"completion_tokens":"{sentinel}"\n\n'
            ),
        ]
    if case == "malformed-final-frame":
        return [*frames, _openai_sse_frame({"diagnostic": sentinel})]
    raise AssertionError(f"Unknown converted terminal case: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "malformed-data-json",
        "premature-eof",
        "bare-done",
        "malformed-final-usage-json",
        "malformed-final-frame",
    ],
)
async def test_converted_stream_requires_explicit_valid_terminal_evidence(
    case: str,
) -> None:
    """Malformed frames and premature EOF cannot synthesize a successful terminal."""
    sentinel = f"converted-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=case,
        frames=_converted_invalid_terminal_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body
    assert marks == []


def _converted_valid_terminal_frames(terminal: str) -> list[str]:
    frames = [_converted_open_text_frame("complete valid text")]
    if terminal == "finish-reason":
        return [
            *frames,
            _openai_sse_frame(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                }
            ),
        ]
    raise AssertionError(f"Unknown converted terminal control: {terminal}")


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["finish-reason"])
async def test_converted_stream_accepts_explicit_terminal_controls(
    terminal: str,
) -> None:
    """Explicit finish_reason and [DONE] terminals remain compatible."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=terminal,
        frames=_converted_valid_terminal_frames(terminal),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert "complete valid text" in body
    assert body.count("event: message_stop") == 1
    assert marks == [terminal]


def _exact_tool_object_json(total_length: int, marker: str) -> str:
    prefix = f'{{"marker":"{marker}","padding":"'
    suffix = '"}'
    if total_length < len(prefix) + len(suffix):
        raise AssertionError("Configured tool argument bound is too small for fixture")
    value = prefix + ("x" * (total_length - len(prefix) - len(suffix))) + suffix
    assert len(value) == total_length
    assert isinstance(json.loads(value), dict)
    return value


def _tool_retained_length(
    *,
    index: int,
    tool_id: str,
    name: str,
    arguments: str,
) -> int:
    """Define the aggregate retained-character contract at the repository cap."""
    return 1 + len(str(index)) + len(tool_id) + len(name) + len(arguments)


def _exact_aggregate_bound_tool_object_json(marker: str) -> str:
    tool_id = "call-bounded-0"
    name = "bounded_tool_0"
    metadata_length = _tool_retained_length(
        index=0,
        tool_id=tool_id,
        name=name,
        arguments="",
    )
    return _exact_tool_object_json(
        MAX_TOOL_ARGUMENT_LENGTH - metadata_length,
        marker,
    )


def _buffered_tool_argument_frames(arguments_by_state: list[str]) -> list[str]:
    first_calls: list[dict[str, Any]] = []
    final_calls: list[dict[str, Any]] = []
    for index, arguments in enumerate(arguments_by_state):
        split_at = max(1, len(arguments) // 2)
        first_calls.append(
            {
                "index": index,
                "id": f"call-bounded-{index}",
                "function": {
                    "name": f"bounded_tool_{index}",
                    "arguments": arguments[:split_at],
                },
            }
        )
        final_calls.append(
            {
                "index": index,
                "function": {"arguments": arguments[split_at:]},
            }
        )
    return [
        _openai_sse_frame(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": first_calls},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _openai_sse_frame(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": final_calls},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
    ]


def _over_bound_tool_arguments(case: str, sentinel: str) -> list[str]:
    if case == "single-state":
        return [
            _exact_tool_object_json(
                MAX_TOOL_ARGUMENT_LENGTH + 1,
                sentinel,
            )
        ]
    if case == "many-states":
        per_state_length = (MAX_TOOL_ARGUMENT_LENGTH // 3) + 1
        return [
            _exact_tool_object_json(per_state_length, f"{sentinel}-{index}")
            for index in range(3)
        ]
    raise AssertionError(f"Unknown bounded tool argument case: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["single-state", "many-states"])
async def test_converted_tool_stream_rejects_retention_over_repo_bound(
    case: str,
) -> None:
    """Buffered tool JSON cannot exceed the existing repository memory bound."""
    sentinel = f"converted-{case}-bound-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=case,
        frames=_buffered_tool_argument_frames(
            _over_bound_tool_arguments(case, sentinel)
        ),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body
    assert marks == []


@pytest.mark.asyncio
async def test_converted_tool_stream_accepts_object_at_repo_bound_control() -> None:
    """A valid aggregate retained payload exactly at the cap remains compatible."""
    marker = "tool-boundary-valid-control"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key="tool-boundary-control",
        frames=_buffered_tool_argument_frames(
            [_exact_aggregate_bound_tool_object_json(marker)]
        ),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert marker in body
    assert body.count("event: message_stop") == 1
    assert marks == ["tool-boundary-control"]


def _exact_length_identity(prefix: str, total_length: int) -> str:
    if total_length < len(prefix):
        raise AssertionError("Configured tool identity bound is too small for fixture")
    value = prefix + ("x" * (total_length - len(prefix)))
    assert len(value) == total_length
    return value


def _converted_nonargument_retention_frames(case: str, sentinel: str) -> list[str]:
    calls: list[dict[str, Any]] = []
    if case == "oversized-id":
        calls.append(
            {
                "index": 0,
                "id": _exact_length_identity(
                    "call-oversized-",
                    MAX_TOOL_ARGUMENT_LENGTH + 1,
                ),
                "function": {"name": "lookup"},
            }
        )
    elif case == "oversized-name":
        calls.append(
            {
                "index": 0,
                "id": "call-oversized-name",
                "function": {
                    "name": _exact_length_identity(
                        "oversized_tool_",
                        MAX_TOOL_ARGUMENT_LENGTH + 1,
                    )
                },
            }
        )
    elif case == "many-state-shells":
        state_count = 4
        identity_length = (MAX_TOOL_ARGUMENT_LENGTH // (state_count * 2)) + 32
        for index in range(state_count):
            calls.append(
                {
                    "index": index,
                    "id": _exact_length_identity(
                        f"call-shell-{index}-",
                        identity_length,
                    ),
                    "function": {
                        "name": _exact_length_identity(
                            f"shell_tool_{index}_",
                            identity_length,
                        )
                    },
                }
            )
    else:
        raise AssertionError(f"Unknown nonargument retention case: {case}")

    frames = [
        _openai_tool_delta_frame(tool_delta=call, finish_reason=None)
        for call in calls
    ]
    return [
        *frames,
        _converted_open_text_frame(sentinel),
        _openai_sse_frame(
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["oversized-id", "oversized-name", "many-state-shells"],
)
async def test_converted_tool_stream_rejects_nonargument_retention_over_repo_bound(
    case: str,
) -> None:
    """Every retained tool identity and state shell participates in the cap."""
    sentinel = f"converted-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=case,
        frames=_converted_nonargument_retention_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body
    assert marks == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_tool_retention_bound_is_request_local() -> None:
    """Over-bound identity state cannot affect a valid boundary-sized stream."""
    sentinel = "converted-concurrent-bound-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("bound-valid", "bound-invalid")}
    release = asyncio.Event()

    valid_task = asyncio.create_task(
        _consume_converted_sse_boundary(
            key="bound-valid",
            frames=_buffered_tool_argument_frames(
                [
                    _exact_aggregate_bound_tool_object_json(
                        "concurrent-boundary-valid",
                    )
                ]
            ),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["bound-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_converted_sse_boundary(
            key="bound-invalid",
            frames=_converted_nonargument_retention_frames(
                "many-state-shells",
                sentinel,
            ),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["bound-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert "concurrent-boundary-valid" in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["bound-valid"]


@pytest.mark.asyncio
async def test_converted_adapter_eof_without_finish_reason_cannot_synthesize_terminal_or_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real OpenAI adapter's synthetic DONE cannot certify truncated output."""
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        ProviderCredentialRuntime,
    )
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.LLM_Calls.providers import (
        openai_adapter as openai_adapter_module,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import (
        OpenAIAdapter,
    )

    touched: list[str] = []
    resolved = _resolved_credentials("openai", "adapter-eof-key", touched)

    async def _resolve(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        assert provider == "openai"
        return resolved

    def _runtime(*_args: Any, **_kwargs: Any) -> ProviderCredentialRuntime:
        return ProviderCredentialRuntime(
            user_id=1,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            server_config_snapshot={},
            resolver=_resolve,
        )

    monkeypatch.setattr(
        messages_endpoint,
        "_new_messages_credential_runtime",
        _runtime,
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield _converted_open_text_frame("truncated adapter text").strip()

        def close(self) -> None:
            return None

    class _StreamContext:
        def __enter__(self):
            return _Response()

        def __exit__(self, *_args: Any) -> bool:
            return False

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def stream(self, *_args: Any, **_kwargs: Any):
            return _StreamContext()

    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    monkeypatch.setattr(
        openai_adapter_module,
        "http_client_factory",
        lambda **_kwargs: _Client(),
    )
    adapter = OpenAIAdapter()

    class _Registry:
        def get_adapter(self, provider: str) -> OpenAIAdapter:
            assert provider == "openai"
            return adapter

    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: _Registry(),
    )
    response = await messages_endpoint._handle_messages(
        _message_request("openai/gpt-4o-mini", stream=True),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )
    body = "".join([item async for item in response.body_iterator])

    assert "truncated adapter text" in body
    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert touched == []


def _converted_error_prefix_frames(*, raw: bool) -> list[str]:
    text = "Error: the requested explanation is ordinary assistant content"
    if raw:
        return [f"Error: {text}\n"]
    return [
        _converted_open_text_frame(text),
        _openai_sse_frame(
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        ),
    ]


@pytest.mark.asyncio
async def test_converted_stream_preserves_typed_error_prefix_but_rejects_raw_control() -> None:
    """Legacy raw errors stay bounded without rejecting typed assistant text."""
    marks: list[str] = []
    lifecycle: list[str] = []

    typed_body = await _consume_converted_sse_boundary(
        key="typed-error-prefix",
        frames=_converted_error_prefix_frames(raw=False),
        marks=marks,
        lifecycle=lifecycle,
    )
    raw_body = await _consume_converted_sse_boundary(
        key="raw-error-prefix",
        frames=_converted_error_prefix_frames(raw=True),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "requested explanation" in typed_body
    assert "event: error" not in typed_body
    assert raw_body.count("event: error") == 1
    assert "requested explanation" not in raw_body
    assert marks == ["typed-error-prefix"]


def _converted_tool_before_text_frames() -> list[str]:
    return [
        _openai_tool_delta_frame(
            tool_delta={
                "index": 0,
                "id": "call-tool-before-text",
                "function": {"name": "lookup", "arguments": "{}"},
            },
            finish_reason=None,
        ),
        _converted_open_text_frame("text after tool identity"),
        _openai_sse_frame(
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        ),
    ]


@pytest.mark.asyncio
async def test_converted_tool_before_text_has_sequential_anthropic_lifecycle() -> None:
    """Mixed tool/text output must never emit block index one before zero."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key="tool-before-text",
        frames=_converted_tool_before_text_frames(),
        marks=marks,
        lifecycle=lifecycle,
    )
    starts = [
        json.loads(line.removeprefix("data: "))["index"]
        for frame in body.split("\n\n")
        if "event: content_block_start" in frame
        for line in frame.splitlines()
        if line.startswith("data: ")
    ]

    assert "event: error" not in body
    assert starts == [0, 1]
    assert body.count("event: message_stop") == 1
    assert marks == ["tool-before-text"]


def _converted_terminal_contract_frames(case: str, sentinel: str) -> list[str]:
    if case == "unknown-reason":
        return [
            _converted_open_text_frame("apparently valid"),
            _openai_sse_frame(
                {
                    "choices": [
                        {"delta": {}, "finish_reason": f"error: {sentinel}"}
                    ]
                }
            ),
        ]
    if case == "tool-finish-without-tool":
        return [
            _converted_open_text_frame("apparently valid"),
            _openai_sse_frame(
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
            ),
        ]
    if case == "end-turn-with-tool":
        return [
            _openai_tool_delta_frame(
                tool_delta={
                    "index": 0,
                    "id": "call-inconsistent-stop",
                    "function": {"name": "lookup", "arguments": "{}"},
                },
                finish_reason="stop",
            )
        ]
    if case == "bare-done":
        return _converted_invalid_terminal_frames("bare-done", sentinel)
    raise AssertionError(f"Unknown converted terminal contract case: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["unknown-reason", "tool-finish-without-tool", "end-turn-with-tool"],
)
async def test_converted_stream_rejects_invalid_finish_contract_before_mark(
    case: str,
) -> None:
    """Only recognized and internally consistent finish metadata can certify use."""
    sentinel = f"converted-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_converted_sse_boundary(
        key=case,
        frames=_converted_terminal_contract_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body
    assert marks == []


@pytest.mark.asyncio
async def test_converted_content_filter_maps_to_anthropic_refusal() -> None:
    """OpenAI content filtering must retain refusal semantics."""
    marks: list[str] = []
    lifecycle: list[str] = []
    body = await _consume_converted_sse_boundary(
        key="content-filter",
        frames=[
            _converted_open_text_frame("filtered output"),
            _openai_sse_frame(
                {"choices": [{"delta": {}, "finish_reason": "content_filter"}]}
            ),
        ],
        marks=marks,
        lifecycle=lifecycle,
    )

    assert '"stop_reason": "refusal"' in body
    assert "event: error" not in body
    assert marks == ["content-filter"]


def _converted_internal_code_text_frames() -> list[str]:
    return [
        _converted_open_text_frame("provider_unavailable"),
        _openai_sse_frame(
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        ),
    ]


def _converted_max_tokens_tool_frames() -> list[str]:
    return [
        _openai_tool_delta_frame(
            tool_delta={
                "index": 0,
                "id": "call-truncated-tool",
                "function": {
                    "name": "lookup",
                    "arguments": '{"query":"unfinished',
                },
            },
            finish_reason=None,
        ),
        _openai_sse_frame(
            {"choices": [{"delta": {}, "finish_reason": "length"}]}
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "family",
    [
        "adapter-terminal",
        "error-prefix",
        "tool-order",
        "finish-contract",
        "internal-code-text",
        "max-tokens-tool",
    ],
)
async def test_concurrent_new_converted_boundary_families_are_request_local(
    family: str,
) -> None:
    """Each converted failure family remains isolated from a valid stream."""
    sentinel = f"converted-{family}-concurrent-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("family-valid", "family-invalid")}
    release = asyncio.Event()
    valid_frames = {
        "error-prefix": _converted_error_prefix_frames(raw=False),
        "tool-order": _converted_tool_before_text_frames(),
        "internal-code-text": _converted_internal_code_text_frames(),
        "max-tokens-tool": _converted_max_tokens_tool_frames(),
    }.get(family, _converted_valid_terminal_frames("finish-reason"))
    invalid_frames = {
        "adapter-terminal": _converted_terminal_contract_frames("bare-done", sentinel),
        "error-prefix": _converted_error_prefix_frames(raw=True),
        "tool-order": _converted_shifted_tool_index_frames(sentinel),
        "finish-contract": _converted_terminal_contract_frames(
            "unknown-reason",
            sentinel,
        ),
        "internal-code-text": _converted_terminal_contract_frames(
            "unknown-reason",
            sentinel,
        ),
        "max-tokens-tool": _converted_shifted_tool_index_frames(sentinel),
    }[family]
    tasks = [
        asyncio.create_task(
            _consume_converted_sse_boundary(
                key=key,
                frames=(valid_frames if key == "family-valid" else invalid_frames),
                marks=marks,
                lifecycle=lifecycle,
                entered=entered[key],
                release=release,
            )
        )
        for key in entered
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert "event: error" not in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["family-valid"]


def _converted_nonstream_finish_contract_response(
    case: str,
    sentinel: str,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": f"response-{case}",
        "model": "model-a",
        "choices": [
            {
                "message": {"content": "apparently valid"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }
    if case == "missing-reason":
        response["choices"][0].pop("finish_reason")
    elif case == "unknown-reason":
        response["choices"][0]["finish_reason"] = f"error: {sentinel}"
    elif case == "tool-finish-without-tool":
        response["choices"][0]["finish_reason"] = "tool_calls"
    elif case in {"end-turn-with-tool", "tool-control"}:
        response = _converted_tool_response(
            argument_style="tool_calls",
            arguments='{"query":"weather"}',
        )
        response["choices"][0]["finish_reason"] = (
            "stop" if case == "end-turn-with-tool" else "tool_calls"
        )
    elif case == "content-filter-control":
        response["choices"][0]["finish_reason"] = "content_filter"
    elif case == "internal-code-control":
        response["choices"][0]["message"]["content"] = "provider_unavailable"
    elif case != "text-control":
        raise AssertionError(f"Unknown non-stream finish-contract case: {case}")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "missing-reason",
        "unknown-reason",
        "tool-finish-without-tool",
        "end-turn-with-tool",
    ],
)
async def test_converted_nonstream_rejects_invalid_finish_contract_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Non-stream success requires recognized, tool-consistent finish metadata."""
    sentinel = f"nonstream-{case}-secret-/private/provider.json"
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", f"{case}-key", touched)},
    )

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        return _converted_nonstream_finish_contract_response(case, sentinel)

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("openai/model-a"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert sentinel not in str(exc_info.value)
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_stop_reason"),
    [
        ("text-control", "end_turn"),
        ("tool-control", "tool_use"),
        ("content-filter-control", "refusal"),
        ("internal-code-control", "end_turn"),
    ],
)
async def test_converted_nonstream_accepts_valid_finish_contract_controls(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_stop_reason: str,
) -> None:
    """Recognized consistent finish metadata remains a semantic success."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("openai", f"{case}-key", touched)},
    )

    async def _adapter(**_kwargs: Any) -> dict[str, Any]:
        return _converted_nonstream_finish_contract_response(case, "unused")

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    response = await messages_endpoint._handle_messages(
        _message_request("openai/model-a"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )

    assert response.status_code == 200
    assert json.loads(response.body)["stop_reason"] == expected_stop_reason
    assert touched == [f"{case}-key"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_converted_nonstream_finish_contract_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid finish contract cannot mark or contaminate a valid request."""
    sentinel = "nonstream-finish-concurrent-secret-/private/provider.json"
    touched: list[str] = []
    entered = {key: asyncio.Event() for key in ("finish-valid", "finish-invalid")}
    release = asyncio.Event()
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("openai", "finish-valid", touched),
            2: _resolved_credentials("openai", "finish-invalid", touched),
        },
    )

    async def _adapter(**kwargs: Any) -> dict[str, Any]:
        key = kwargs["api_key"]
        entered[key].set()
        await release.wait()
        return _converted_nonstream_finish_contract_response(
            "internal-code-control"
            if key == "finish-valid"
            else "unknown-reason",
            sentinel,
        )

    async def _call(user_id: int):
        return await messages_endpoint._handle_messages(
            _message_request("openai/model-a"),
            current_user=SimpleNamespace(id_int=user_id),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter)
    tasks = [asyncio.create_task(_call(user_id)) for user_id in (1, 2)]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_result, invalid_result = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert not isinstance(valid_result, BaseException)
    assert isinstance(invalid_result, HTTPException)
    assert invalid_result.status_code == 502
    assert sentinel not in str(valid_result) + str(invalid_result)
    assert touched == ["finish-valid"]


def _native_current_response(case: str, sentinel: str) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": f"msg-{case}",
        "type": "message",
        "role": "assistant",
        "model": "claude-current",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    if case == "server-tool-pause":
        response["content"] = [
            {
                "type": "server_tool_use",
                "id": "srvtoolu-paused",
                "name": "web_search",
                "input": {"query": "latest research"},
            }
        ]
        response["stop_reason"] = "pause_turn"
    elif case == "compaction-pause":
        response["content"] = [
            {
                "type": "compaction",
                "content": "Earlier conversation summarized.",
                "encrypted_content": "opaque-compaction-payload",
            }
        ]
        response["stop_reason"] = "compaction"
    elif case == "web-search-error-result":
        response["content"] = [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu-search",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "too_many_requests",
                },
            },
            {"type": "text", "text": "Search was unavailable."},
        ]
    elif case == "advisor-error-result":
        response["content"] = [
            {
                "type": "advisor_tool_result",
                "tool_use_id": "srvtoolu-advisor",
                "content": {
                    "type": "advisor_tool_result_error",
                    "error_code": "overloaded",
                },
            },
            {"type": "text", "text": "Advisor was unavailable."},
        ]
    elif case == "mcp-error-result":
        response["content"] = [
            {
                "type": "mcp_tool_result",
                "tool_use_id": "mcp_toolu-weather",
                "content": "Error: city not found",
                "is_error": True,
            },
            {"type": "text", "text": "MCP result handled."},
        ]
    elif case == "pre-output-refusal":
        response["content"] = []
        response["stop_reason"] = "refusal"
        response["stop_details"] = {
            "type": "refusal",
            "category": "cyber",
            "explanation": "This request was declined.",
        }
    elif case == "unknown-diagnostic-block":
        response["content"] = [
            {"type": "text", "text": "apparently valid"},
            {"type": "provider_debug", "diagnostic": sentinel},
        ]
    elif case == "transport-error":
        return {"error": {"type": "api_error", "message": sentinel}}
    elif case == "valid-text":
        response["content"] = [{"type": "text", "text": "valid"}]
    else:
        raise AssertionError(f"Unknown native current response case: {case}")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "server-tool-pause",
        "compaction-pause",
        "web-search-error-result",
        "advisor-error-result",
        "mcp-error-result",
        "pre-output-refusal",
    ],
)
async def test_native_nonstream_accepts_current_documented_content_flows(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Current native server-tool and beta blocks are semantic 200 responses."""
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", f"{case}-key", touched)},
    )
    payload = _native_current_response(case, "unused")

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-opus-4-8"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta="compact-2026-01-12,advisor-tool-2026-03-01",
    )

    assert response.status_code == 200
    assert json.loads(response.body) == payload
    assert touched == [f"{case}-key"]


@pytest.mark.asyncio
async def test_native_nonstream_rejects_unknown_diagnostic_block_before_leak_or_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid native text block cannot smuggle an unknown diagnostic block."""
    sentinel = "native-nonstream-diagnostic-secret-/private/provider.json"
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "diagnostic-key", touched)},
    )

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _native_current_response("unknown-diagnostic-block", sentinel)

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("anthropic/claude-opus-4-8"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert sentinel not in str(exc_info.value)
    assert touched == []


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "case",
    [
        "server-tool-pause",
        "compaction-pause",
        "web-search-error-result",
        "pre-output-refusal",
    ],
)
async def test_concurrent_native_nonstream_current_flows_are_request_local(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """A valid current native response cannot inherit a concurrent failure."""
    sentinel = f"native-{case}-concurrent-secret-/private/provider.json"
    touched: list[str] = []
    entered = {key: asyncio.Event() for key in ("current-valid", "current-invalid")}
    release = asyncio.Event()
    _install_resolutions(
        monkeypatch,
        {
            1: _resolved_credentials("anthropic", "current-valid", touched),
            2: _resolved_credentials("anthropic", "current-invalid", touched),
        },
    )

    async def _native(
        _url: str,
        headers: dict[str, str],
        _payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        key = headers["x-api-key"]
        entered[key].set()
        await release.wait()
        return _native_current_response(
            case if key == "current-valid" else "transport-error",
            sentinel,
        )

    async def _call(user_id: int):
        return await messages_endpoint._handle_messages(
            _message_request("anthropic/claude-opus-4-8"),
            current_user=SimpleNamespace(id_int=user_id),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta="compact-2026-01-12",
        )

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    valid_task = asyncio.create_task(_call(1))
    invalid_task = asyncio.create_task(_call(2))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_result, invalid_result = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task, return_exceptions=True),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert not isinstance(valid_result, BaseException)
    assert valid_result.status_code == 200
    assert isinstance(invalid_result, HTTPException)
    assert invalid_result.status_code == 502
    assert sentinel not in str(invalid_result)
    assert touched == ["current-valid"]


def _current_nonstream_cache_creation() -> dict[str, int]:
    return {
        "ephemeral_1h_input_tokens": 1,
        "ephemeral_5m_input_tokens": 2,
    }


def _current_nonstream_usage_iterations() -> list[dict[str, Any]]:
    common = {
        "input_tokens": 7,
        "output_tokens": 2,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 1,
        "cache_creation": _current_nonstream_cache_creation(),
    }
    return [
        {"type": "message", "model": "claude-opus-4-8", **common},
        {"type": "compaction", **common},
        {
            "type": "advisor_message",
            "model": "claude-advisor-current",
            **common,
        },
        {
            "type": "fallback_message",
            "model": "claude-sonnet-current",
            **common,
        },
    ]


def _native_current_usage_response(case: str) -> dict[str, Any]:
    response = _native_current_response("valid-text", "unused")
    if case == "output-tokens-details":
        response["usage"]["output_tokens_details"] = {"thinking_tokens": 2}
    elif case == "speed":
        response["usage"]["speed"] = "fast"
    elif case == "all-iteration-variants":
        response["usage"]["iterations"] = _current_nonstream_usage_iterations()
    else:
        raise AssertionError(f"Unknown native usage response case: {case}")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["output-tokens-details", "speed", "all-iteration-variants"],
)
async def test_native_nonstream_accepts_current_stable_and_beta_usage_shapes(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", f"{case}-key", touched)},
    )
    payload = _native_current_usage_response(case)

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-opus-4-8"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta="current-usage-beta",
    )

    assert response.status_code == 200
    assert json.loads(response.body) == payload
    assert touched == [f"{case}-key"]


def _native_current_refusal_response(case: str) -> dict[str, Any]:
    response = _native_current_response("pre-output-refusal", "unused")
    if case == "optional-fields-omitted":
        response["stop_details"] = {"type": "refusal"}
    elif case == "extended-beta-fields":
        response["stop_details"] = {
            "type": "refusal",
            "category": None,
            "explanation": None,
            "fallback_credit_token": "opaque-credit-token",
            "fallback_has_prefill_claim": False,
            "recommended_model": "claude-sonnet-current",
        }
    else:
        raise AssertionError(f"Unknown native refusal response case: {case}")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["optional-fields-omitted", "extended-beta-fields"],
)
async def test_native_nonstream_accepts_current_refusal_stop_details(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", f"{case}-key", touched)},
    )
    payload = _native_current_refusal_response(case)

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    response = await messages_endpoint._handle_messages(
        _message_request("anthropic/claude-opus-4-8"),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta="server-side-fallback-2026-06-01",
    )

    assert response.status_code == 200
    assert json.loads(response.body) == payload
    assert touched == [f"{case}-key"]


@pytest.mark.asyncio
async def test_native_count_tokens_accepts_context_management_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", "count-context-key", touched)},
    )
    payload = {
        "input_tokens": 4,
        "context_management": {"original_input_tokens": 6},
    }

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    response = await messages_endpoint._handle_count_tokens(
        AnthropicCountTokensRequest(
            model="anthropic/claude-opus-4-8",
            messages=[{"role": "user", "content": "hello"}],
            context_management={"edits": []},
        ),
        current_user=SimpleNamespace(id_int=1),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta="context-management-2025-06-27",
    )

    assert response.status_code == 200
    assert json.loads(response.body) == payload
    assert touched == ["count-context-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["diagnostics", "context-management"])
async def test_native_nonstream_rejects_unknown_nested_metadata_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    sentinel = f"native-{case}-secret-/private/provider.json"
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", f"{case}-key", touched)},
    )
    payload = _native_current_response("valid-text", "unused")
    if case == "diagnostics":
        payload["diagnostics"] = {
            "cache_miss_reason": None,
            "provider_debug": sentinel,
        }
    else:
        payload["context_management"] = {
            "applied_edits": [],
            "provider_debug": sentinel,
        }

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("anthropic/claude-opus-4-8"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta="cache-diagnostics-2026-05-19",
        )

    assert exc_info.value.status_code == 502
    assert sentinel not in str(exc_info.value)
    assert touched == []


def _native_inconsistent_tool_response(case: str) -> dict[str, Any]:
    response = _native_current_response("valid-text", "unused")
    if case == "tool-finish-without-tool":
        response["stop_reason"] = "tool_use"
    elif case == "end-turn-with-client-tool":
        response["content"] = [
            {
                "type": "tool_use",
                "id": "toolu-native-nonstream",
                "name": "lookup",
                "input": {"query": "weather"},
            }
        ]
        response["stop_reason"] = "end_turn"
    else:
        raise AssertionError(f"Unknown native tool consistency case: {case}")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["tool-finish-without-tool", "end-turn-with-client-tool"],
)
async def test_native_nonstream_rejects_inconsistent_client_tool_terminal_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    touched: list[str] = []
    _install_resolutions(
        monkeypatch,
        {1: _resolved_credentials("anthropic", f"{case}-key", touched)},
    )

    async def _native(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _native_inconsistent_tool_response(case)

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native)
    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._handle_messages(
            _message_request("anthropic/claude-opus-4-8"),
            current_user=SimpleNamespace(id_int=1),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )

    assert exc_info.value.status_code == 502
    assert touched == []
