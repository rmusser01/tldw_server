"""
Simplified chat endpoint tests using real database and authentication.
"""

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException, status
from loguru import logger
from starlette.requests import Request
from starlette.responses import StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
from tldw_Server_API.app.core.Chat import chat_service, streaming_utils
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.streaming_utils import StreamingResponseHandler
from tldw_Server_API.app.core.exceptions import ProviderCredentialTerminalError
from tldw_Server_API.app.core.LLM_Calls.routing.models import RouterRequest, RoutingPolicy
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.provider_credential_test_helpers import (
    issue_provider_call_credentials_async,
)

_CHAT_USER_SECRET = "sk-chat-user-secret-must-not-leak"
_REGISTRY_OPENAI_BASE_URL = "https://registry-openai.test/v1"


def _registry_openai_app_config() -> dict[str, dict[str, str]]:
    """Return the authoritative adapter URL snapshot for registry tests."""

    return {"openai_api": {"api_base_url": _REGISTRY_OPENAI_BASE_URL}}


def _install_real_openai_adapter_transport(
    monkeypatch: pytest.MonkeyPatch,
    responder,
    *,
    on_client_exit=None,
) -> None:
    """Install a deterministic transport below the real registry adapter."""

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
    """Expose cancellation drain entry without timing-based negative proof."""

    drain_entered = asyncio.Event()
    original = bounded_daemon_module._drain_owned_task

    async def probe(task):
        drain_entered.set()
        return await original(task)

    monkeypatch.setattr(bounded_daemon_module, "_drain_owned_task", probe)
    return drain_entered


class _RouterMetrics:
    def __init__(self) -> None:
        self.llm_calls: list[tuple[str, str, bool, str | None]] = []
        self.token_calls: list[dict[str, Any]] = []

    def track_llm_call(
        self,
        provider: str,
        model: str,
        _latency: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        self.llm_calls.append((provider, model, success, error_type))

    def track_tokens(self, **kwargs: Any) -> None:
        self.token_calls.append(kwargs)


class _RouterProviderManager:
    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.successes: list[str] = []

    def record_failure(self, provider: str, error: Exception) -> None:
        self.failures.append((provider, type(error).__name__))

    def record_success(self, provider: str, _latency: float) -> None:
        self.successes.append(provider)


def _real_router_inputs(scope: str) -> tuple[
    RouterRequest,
    RoutingPolicy,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Return the smallest inputs that execute the production LLM router."""

    return (
        RouterRequest(
            model="auto",
            surface="chat",
            latest_user_turn="route this request",
            scope=scope,
        ),
        RoutingPolicy(
            request_model="auto",
            server_default_provider="openai",
            boundary_mode="server_default_provider",
            strategy="llm_router",
        ),
        [
            {"provider": "openai", "model": "routed"},
            {"provider": "anthropic", "model": "claude-routed"},
        ],
        {
            "providers": [
                {
                    "name": "openai",
                    "default_model": "router-model",
                    "models_info": [{"name": "router-model"}],
                }
            ]
        },
    )


@pytest.fixture(autouse=True)
def _freeze_legacy_test_api_keys_for_runtime(monkeypatch):
    """Expose legacy module test keys through the runtime's frozen config seam."""
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS

    def load_snapshot():
        snapshot = {}
        for provider, api_key in dict(chat_endpoint.API_KEYS).items():
            section = PROVIDER_APP_CONFIG_KEYS.get(provider)
            if (
                section is not None
                and isinstance(api_key, str)
                and api_key.strip()
            ):
                snapshot[section] = {"api_key": api_key}
        return snapshot

    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        load_snapshot,
    )


def _chat_scope_request(*, active_team_id=None, active_org_id=None):
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/chat/completions",
            "headers": [],
            "query_string": b"",
        }
    )
    request.state.auth = SimpleNamespace(
        principal=SimpleNamespace(
            user_id=42,
            team_ids=[7],
            org_ids=[11],
            active_team_id=active_team_id,
            active_org_id=active_org_id,
        )
    )
    return request


def test_chat_trusted_scope_does_not_infer_singleton_team_ahead_of_active_org():
    request = _chat_scope_request(active_org_id=11)

    _user_id, team_ids, org_ids, _trusted = chat_endpoint._trusted_credential_runtime_scope(
        request,
        SimpleNamespace(id=42, id_int=42),
    )

    assert team_ids == []
    assert org_ids == [11]


@pytest.mark.parametrize("kind", ["team", "org"])
@pytest.mark.parametrize("active_id", ["malformed", 99])
def test_chat_trusted_scope_rejects_invalid_active_id(kind, active_id):
    request = _chat_scope_request(active_team_id=7, active_org_id=11)
    setattr(request.state.auth.principal, f"active_{kind}_id", active_id)

    with pytest.raises(ByokResolutionError) as exc_info:
        chat_endpoint._trusted_credential_runtime_scope(
            request,
            SimpleNamespace(id=42, id_int=42),
        )

    assert exc_info.value.code == "credential_scope_revoked"


def _real_credential_runtime_type(resolver):
    class Runtime(ProviderCredentialRuntime):
        instances: list["Runtime"] = []

        def __init__(self, **kwargs):
            super().__init__(resolver=resolver, **kwargs)
            self.__class__.instances.append(self)

    return Runtime


def _resolved_user_credential(
    provider: str,
    api_key: str,
    *,
    auth_source: str = "api_key",
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config={f"{provider}_api": {"model": "runtime-model"}},
        credential_fields={},
        source="user",
        allowlisted=True,
        status=ByokResolutionStatus.RESOLVED,
        auth_source=auth_source,
    )


def _credential_runtime_double(
    *,
    api_keys: dict[str, str | None] | None = None,
    auth_source: str | None = None,
    auth_sources: dict[str, str | None] | None = None,
    errors: dict[str, Exception] | None = None,
    refresh_errors: dict[str, BaseException] | None = None,
    refresh_api_keys: dict[str, str | None] | None = None,
):
    class RuntimeDouble:
        instances: list["RuntimeDouble"] = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.resolve_calls: list[tuple[str, bool]] = []
            self.model_resolve_calls: list[tuple[str, str | None, bool]] = []
            self.mark_calls: list[str] = []
            self.marked_used: list[str] = []
            self.close_calls = 0
            self.__class__.instances.append(self)

        async def resolve(
            self,
            provider: str,
            *,
            model: str | None = None,
            force_refresh: bool = False,
        ):
            provider = provider.strip().lower()
            self.resolve_calls.append((provider, force_refresh))
            self.model_resolve_calls.append((provider, model, force_refresh))
            if force_refresh and provider in (refresh_errors or {}):
                raise refresh_errors[provider]
            error = (errors or {}).get(provider)
            if error is not None:
                raise error
            key = (api_keys or {}).get(provider, f"{provider}-runtime-key")
            if force_refresh:
                key = (refresh_api_keys or {}).get(
                    provider,
                    f"{provider}-refreshed-runtime-key",
                )
            return await issue_provider_call_credentials_async(
                provider,
                api_key=key,
                app_config={f"{provider}_api": {"model": "runtime-model"}},
                auth_source=(auth_sources or {}).get(provider, auth_source),
                model=model,
            )

        async def mark_used(self, handle) -> None:
            self.mark_calls.append(handle.provider)
            if handle.provider not in self.marked_used:
                self.marked_used.append(handle.provider)

        async def close(self) -> None:
            self.close_calls += 1

    return RuntimeDouble


def _certified_pre_dispatch(error: BaseException) -> BaseException:
    """Mark a typed stream failure as explicitly safe to replay."""
    error.upstream_dispatched = False
    error.output_emitted = False
    error.allow_non_stream_fallback = True
    return error


def _with_replay_flags(
    error: BaseException,
    *,
    upstream_dispatched,
    allow_non_stream_fallback,
    output_emitted=False,
) -> BaseException:
    error.upstream_dispatched = upstream_dispatched
    error.output_emitted = output_emitted
    error.allow_non_stream_fallback = allow_non_stream_fallback
    return error


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_user_credential_never_uses_server_fallback_or_leaks(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    streaming,
):
    captured: dict[str, object] = {}
    resolution_calls: list[tuple[str, bool]] = []

    async def resolver(provider: str, **kwargs):
        resolution_calls.append((provider, kwargs["force_oauth_refresh"]))
        return _resolved_user_credential(provider, _CHAT_USER_SECRET)

    runtime_type = _real_credential_runtime_type(resolver)
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=streaming,
    )

    async def execute_non_stream(**kwargs):
        captured.update(kwargs["cleaned_args"])
        await kwargs["on_success"](kwargs["selected_provider"])
        return {
            "id": "chatcmpl-runtime",
            "choices": [{"message": {"role": "assistant", "content": "safe"}}],
        }

    async def execute_stream(**kwargs):
        captured.update(kwargs["cleaned_args"])

        async def body():
            await kwargs["on_provider_output"](kwargs["selected_provider"])
            yield 'data: {"choices":[{"delta":{"content":"safe"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(
                chat_endpoint,
                "get_provider_manager",
                return_value=provider_manager,
            ),
            patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
            patch.object(
                chat_endpoint,
                "resolve_provider_api_key",
                side_effect=AssertionError("server fallback used"),
            ) as server_fallback,
            patch.object(
                chat_endpoint,
                "execute_non_stream_call",
                side_effect=execute_non_stream,
            ),
            patch.object(
                chat_endpoint,
                "execute_streaming_call",
                side_effect=execute_stream,
            ),
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == status.HTTP_200_OK
    assert captured["api_key"] == _CHAT_USER_SECRET
    assert captured["credentials_resolved"] is True
    assert resolution_calls == [("openai", False)]
    assert _CHAT_USER_SECRET not in response.text
    assert _CHAT_USER_SECRET not in "".join(logs)
    server_fallback.assert_not_called()
    provider_manager.get_available_provider.assert_not_called()
    assert runtime_type.instances[0]._closed is True


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_oauth_refresh_keeps_user_credential_isolated_and_secret_free(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    streaming,
):
    refreshed_secret = f"{_CHAT_USER_SECRET}-refreshed"
    resolution_calls: list[tuple[str, bool]] = []

    async def resolver(provider: str, **kwargs):
        force_refresh = kwargs["force_oauth_refresh"]
        resolution_calls.append((provider, force_refresh))
        return _resolved_user_credential(
            provider,
            refreshed_secret if force_refresh else _CHAT_USER_SECRET,
            auth_source="oauth",
        )

    runtime_type = _real_credential_runtime_type(resolver)
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=streaming,
    )

    if streaming:

        def expired_stream():
            if False:
                yield ""
            raise _certified_pre_dispatch(
                ChatAuthenticationError(
                    "expired OAuth credential",
                    provider="openai",
                )
            )

        def refreshed_stream():
            yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        provider_results = [expired_stream(), refreshed_stream()]
    else:
        provider_results = [
            ChatAuthenticationError(
                "expired OAuth credential",
                provider="openai",
            ),
            {
                "id": "chatcmpl-refreshed",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "recovered",
                        }
                    }
                ],
            },
        ]

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(
                chat_endpoint,
                "get_provider_manager",
                return_value=provider_manager,
            ),
            patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
            patch.object(chat_endpoint, "get_request_queue", return_value=None),
            patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
            patch.object(
                chat_endpoint,
                "resolve_provider_api_key",
                side_effect=AssertionError("server fallback used"),
            ) as server_fallback,
            patch.object(
                chat_endpoint,
                "perform_chat_api_call",
                side_effect=provider_results,
            ) as provider_call,
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == status.HTTP_200_OK
    assert "recovered" in response.text
    assert resolution_calls == [("openai", False), ("openai", True)]
    assert provider_call.call_count == 2
    assert provider_call.call_args_list[0].kwargs["api_key"] == _CHAT_USER_SECRET
    assert provider_call.call_args_list[1].kwargs["api_key"] == refreshed_secret
    assert _CHAT_USER_SECRET not in response.text
    assert _CHAT_USER_SECRET not in "".join(logs)
    server_fallback.assert_not_called()
    provider_manager.get_available_provider.assert_not_called()


@pytest.mark.parametrize(
    "error_code",
    ["invalid_provider_credentials", "credential_store_unavailable"],
)
def test_chat_terminal_credential_resolution_never_falls_back_or_leaks(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    monkeypatch,
    error_code,
):
    resolution_path: list[str] = []

    if error_code == "invalid_provider_credentials":

        class FakeUserRepo:
            async def fetch_secret_for_active_user(
                self,
                user_id: int,
                provider: str,
                *,
                include_revoked: bool = False,
            ):
                assert (user_id, provider) == (1, "openai")
                assert include_revoked is True
                resolution_path.append(error_code)
                return {
                    "encrypted_blob": f"{_CHAT_USER_SECRET}:invalid-envelope",
                    "last_used_at": None,
                }

            async def fetch_secret_for_user(self, *_args, **_kwargs):
                raise AssertionError("unrestricted credential lookup used")

        async def get_user_repo():
            return FakeUserRepo()

    else:

        async def get_user_repo():
            resolution_path.append(error_code)
            raise sqlite3.OperationalError(
                f"credential store failed: {_CHAT_USER_SECRET}"
            )

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(
        byok_runtime,
        "is_provider_allowlisted",
        lambda _provider: True,
    )

    runtime_type = _real_credential_runtime_type(
        byok_runtime.resolve_byok_credentials
    )
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(
                chat_endpoint,
                "get_provider_manager",
                return_value=provider_manager,
            ),
            patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
            patch.object(
                chat_endpoint,
                "resolve_provider_api_key",
                side_effect=AssertionError("server fallback used"),
            ) as server_fallback,
            patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == error_code
    assert resolution_path == [error_code]
    assert _CHAT_USER_SECRET not in response.text
    assert _CHAT_USER_SECRET not in "".join(logs)
    server_fallback.assert_not_called()
    provider_call.assert_not_called()
    provider_manager.get_available_provider.assert_not_called()


def test_chat_completion_basic(authenticated_client, mock_chacha_db, setup_dependencies):
    """Test basic chat completion with authenticated user."""

    # Prepare request data - must include api_provider
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",  # Must specify provider
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
    )

    # Mock the LLM call and API keys
    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": "I'm doing well, thank you!"}, "finish_reason": "stop"}
            ],
        }

        # Make request
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "I'm doing well, thank you!"

        # Verify LLM was called
        mock_llm.assert_called_once()


def test_chat_completion_missing_saved_llamacpp_grammar_is_bad_request(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "llama.cpp": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="local-model",
        api_provider="llama.cpp",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        grammar_mode="library",
        grammar_id="missing-grammar",
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Invalid request."
    provider_call.assert_not_called()


@pytest.mark.skip(reason="Streaming tests hang with TestClient")
def test_chat_completion_streaming(authenticated_client, mock_chacha_db):
    """Test streaming chat completion."""

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Tell me a story")],
        stream=True,
    )

    # Mock streaming response
    def mock_stream():
        yield 'data: {"choices": [{"delta": {"content": "Once "}}]}\n\n'
        yield 'data: {"choices": [{"delta": {"content": "upon "}}]}\n\n'
        yield 'data: {"choices": [{"delta": {"content": "a "}}]}\n\n'
        yield 'data: {"choices": [{"delta": {"content": "time..."}}]}\n\n'
        yield "data: [DONE]\n\n"

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        mock_llm.return_value = mock_stream()

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_200_OK
        # For streaming, we just verify it doesn't error
        # Actual streaming validation would require async client


def test_chat_completion_with_character(authenticated_client, mock_chacha_db, setup_dependencies):
    """Test chat completion with a specific character."""

    # Add a character to the mock database
    character_id = mock_chacha_db.add_character_card(
        {
            "name": "TestBot",
            "description": "A test character",
            "personality": "Friendly and helpful",
            "system_prompt": "You are TestBot, a friendly assistant.",
        }
    )

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Who are you?")],
        character_id=str(character_id),
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "I am TestBot!"}, "finish_reason": "stop"}],
        }

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_200_OK

        # Verify the system prompt was included
        call_args = mock_llm.call_args
        assert "TestBot" in str(call_args)


def test_chat_completion_unauthorized(mock_chacha_db):
    """Test that unauthenticated requests are rejected."""
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        # Get CSRF token but don't authenticate
        response = client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")

        request_data = ChatCompletionRequest(
            model="test-model", messages=[ChatCompletionUserMessageParam(role="user", content="Hello")]
        )

        response = client.post(
            "/api/v1/chat/completions", json=request_data.model_dump(), headers={"X-CSRF-Token": csrf_token}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_chat_completion_invalid_model(authenticated_client, mock_chacha_db, setup_dependencies):
    """Test handling of invalid model requests."""

    # Use a valid provider but configure it to fail
    request_data = ChatCompletionRequest(
        model="invalid-model-xyz",
        api_provider="openai",  # Use a valid provider
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        # Simulate an error for invalid model
        mock_llm.side_effect = Exception("Invalid model: invalid-model-xyz")

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        # Should return an error status
        assert response.status_code >= 400


def test_chat_completion_explicit_model_unavailable_returns_400(
    authenticated_client, mock_chacha_db, setup_dependencies, monkeypatch
):
    """Explicit unavailable models should fail fast with a 400 response."""
    monkeypatch.setenv("CHAT_ENFORCE_STRICT_MODEL_SELECTION", "1")

    request_data = ChatCompletionRequest(
        model="missing-model-123",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.is_model_known_for_provider",
            return_value=False,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        detail = response.json().get("detail", {})
        assert detail.get("error_code") == "model_not_available"
        assert detail.get("provider") == "openai"
        assert detail.get("model") == "missing-model-123"
        mock_llm.assert_not_called()


def test_chat_completion_disables_provider_fallback_for_explicit_model(
    authenticated_client, mock_chacha_db, setup_dependencies, monkeypatch
):
    """Strict explicit model selection should disable provider fallback for the request."""
    monkeypatch.setenv("CHAT_ENFORCE_STRICT_MODEL_SELECTION", "1")

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    captured: dict[str, object] = {}

    async def fake_execute_non_stream_call(**kwargs):
        captured["selected_provider"] = kwargs.get("selected_provider")
        captured["enable_provider_fallback"] = kwargs.get("enable_provider_fallback")
        return {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {}
    provider_manager.get_available_provider = MagicMock(return_value="anthropic")

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call", side_effect=fake_execute_non_stream_call),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.get_provider_manager", return_value=provider_manager),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.is_model_known_for_provider",
            return_value=True,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_200_OK
        assert captured.get("selected_provider") == "openai"
        assert captured.get("enable_provider_fallback") is False
        provider_manager.get_available_provider.assert_not_called()


def test_chat_completion_revalidates_default_model(authenticated_client, mock_chacha_db, setup_dependencies):
    """Ensure default model selection is revalidated against provider overrides."""

    request_data = ChatCompletionRequest(
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    def _fake_validate(provider: str, model: Optional[str]):
        if model == "bad-model":
            return {"error_code": "model_not_allowed", "message": "Model not allowed"}
        return None

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.get_override_default_model", return_value="bad-model"),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.validate_provider_override", side_effect=_fake_validate),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": "Should not be returned"}, "finish_reason": "stop"}
            ],
        }

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_403_FORBIDDEN
        detail = response.json().get("detail", {})
        assert detail.get("error_code") == "model_not_allowed"


def test_chat_completion_default_model_tracks_model(authenticated_client, mock_chacha_db, setup_dependencies):
    """Ensure default model injection updates downstream model tracking."""

    request_data = ChatCompletionRequest(
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    captured = {}

    async def fake_execute_non_stream_call(**kwargs):
        captured["model"] = kwargs.get("model")
        return {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call", side_effect=fake_execute_non_stream_call),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.get_override_default_model", return_value="default-model"),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.validate_provider_override", return_value=None),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_200_OK
        assert captured.get("model") == "default-model"


def test_chat_completion_downgrades_structured_response_format_before_provider_call(
    authenticated_client, mock_chacha_db, setup_dependencies, monkeypatch
):
    """Providers that only accept json_object should not receive json_schema payloads."""

    class _JsonObjectOnlyAdapter:
        def capabilities(self):
            return {"response_format_types": ["json_object"]}

    class _Registry:
        def get_adapter(self, _provider: str):
            return _JsonObjectOnlyAdapter()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.adapter_registry.get_registry",
        lambda: _Registry(),
    )

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Return structured JSON")],
        response_format={
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
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
        }

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_200_OK
        assert mock_llm.call_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_completion_uses_one_runtime_and_marks_success_once(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    streaming,
):
    runtime_type = _credential_runtime_double()
    captured: dict[str, object] = {}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=streaming,
    )

    async def fake_execute_non_stream_call(**kwargs):
        captured.update(kwargs["cleaned_args"])
        await kwargs["on_success"](kwargs["selected_provider"])
        return {
            "id": "chatcmpl-runtime",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    async def fake_execute_streaming_call(**kwargs):
        captured.update(kwargs["cleaned_args"])

        async def body():
            await kwargs["on_provider_output"](kwargs["selected_provider"])
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(
            chat_endpoint,
            "execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
        patch.object(
            chat_endpoint,
            "execute_streaming_call",
            side_effect=fake_execute_streaming_call,
        ),
        patch.object(chat_endpoint, "API_KEYS", {"openai": "legacy-key"}),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert len(runtime_type.instances) == 1
    runtime = runtime_type.instances[0]
    assert runtime.marked_used == ["openai"]
    assert runtime.close_calls == 1
    assert captured["api_key"] == "openai-runtime-key"
    assert captured["credentials_resolved"] is True


def test_chat_completion_resolves_health_fallback_with_same_runtime(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    captured: dict[str, object] = {}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    async def fake_execute_non_stream_call(**kwargs):
        captured.update(kwargs["cleaned_args"])
        await kwargs["on_success"](kwargs["selected_provider"])
        return {
            "id": "chatcmpl-runtime-fallback",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {}
    provider_manager.get_available_provider.return_value = "anthropic"

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(
            chat_endpoint,
            "execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
        patch.object(
            chat_endpoint,
            "API_KEYS",
            {"openai": "legacy-openai", "anthropic": "legacy-anthropic"},
        ),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert len(runtime_type.instances) == 1
    runtime = runtime_type.instances[0]
    assert ("openai", False) in runtime.resolve_calls
    assert ("anthropic", False) in runtime.resolve_calls
    assert runtime.marked_used == ["anthropic"]
    assert captured["api_key"] == "anthropic-runtime-key"
    assert captured["credentials_resolved"] is True


@pytest.mark.concurrent
def test_health_fallback_keeps_model_policy_atomic_during_config_rotation(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    monkeypatch,
):
    """Fallback dispatch must authorize the model from its credential snapshot."""
    fallback_resolution_started = threading.Event()
    release_fallback_resolution = threading.Event()
    captured: dict[str, object] = {}

    class Runtime:
        instances: list["Runtime"] = []

        def __init__(self, **_kwargs):
            self.resolve_calls: list[tuple[str, str | None]] = []
            self.marked_used: list[str] = []
            self.close_calls = 0
            self._fallback_calls = 0
            self.__class__.instances.append(self)

        async def resolve(
            self,
            provider: str,
            *,
            model: str | None = None,
            force_refresh: bool = False,
        ):
            assert force_refresh is False
            normalized = provider.strip().lower()
            self.resolve_calls.append((normalized, model))
            if normalized == "anthropic":
                self._fallback_calls += 1
                if self._fallback_calls == 1:
                    fallback_resolution_started.set()
                    await asyncio.to_thread(release_fallback_resolution.wait, 2.0)
            return await issue_provider_call_credentials_async(
                normalized,
                api_key=f"{normalized}-runtime-key",
                app_config={
                    f"{normalized}_api": {
                        "model": (
                            "claude-snapshot-model"
                            if normalized == "anthropic"
                            else "gpt-4o-mini"
                        )
                    }
                },
                auth_source=None,
                model=model,
            )

        async def mark_used(self, handle) -> None:
            self.marked_used.append(handle.provider)

        async def close(self) -> None:
            self.close_calls += 1

    async def fake_execute_non_stream_call(**kwargs):
        captured.update(kwargs["cleaned_args"])
        await kwargs["on_success"](kwargs["selected_provider"])
        return {
            "id": "chatcmpl-atomic-fallback",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {}
    provider_manager.get_available_provider.return_value = "anthropic"
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    monkeypatch.setenv("DEFAULT_MODEL_ANTHROPIC", "claude-before-rotation")

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", Runtime),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(
            chat_endpoint,
            "execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
        patch.object(chat_endpoint, "get_override_default_model", return_value=None),
        patch.object(chat_endpoint, "get_llm_provider_override", return_value=None),
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                authenticated_client.post,
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
            assert fallback_resolution_started.wait(timeout=2.0)
            monkeypatch.setenv("DEFAULT_MODEL_ANTHROPIC", "claude-rotated-model")
            release_fallback_resolution.set()
            response = future.result(timeout=5.0)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert captured["model"] == "claude-snapshot-model"
    assert Runtime.instances[0].resolve_calls == [
        ("openai", "gpt-4o-mini"),
        ("anthropic", None),
        ("anthropic", "claude-snapshot-model"),
    ]
    assert Runtime.instances[0].marked_used == ["anthropic"]
    assert Runtime.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_auto_router_uses_request_credential_runtime(monkeypatch):
    runtime_type = _credential_runtime_double()
    runtime = runtime_type(
        user_id=1,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=False,
        fallback_resolver=lambda _provider: None,
    )
    captured: dict[str, object] = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"gpt-4o-mini"}'
                    }
                }
            ]
        }

    async def fake_select(**kwargs):
        router_model = SimpleNamespace(provider="openai", model="gpt-4o-mini")
        await kwargs["execute_router_call"](router_model, [{"role": "user", "content": "route"}])
        return {"provider": "openai", "model": "gpt-4o-mini"}, {}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call_async", fake_call)
    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", fake_select)

    await chat_endpoint._select_auto_chat_llm_router_choice(
        router_request=SimpleNamespace(scope=None),
        policy=SimpleNamespace(),
        candidates=[],
        provider_listing={},
        request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
        current_user=SimpleNamespace(id=1),
        request_id="request-1",
        credential_runtime=runtime,
    )

    assert runtime.resolve_calls == [("openai", False)]
    assert runtime.marked_used == ["openai"]
    assert captured["api_key"] == "openai-runtime-key"
    assert captured["credentials_resolved"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_accounting"),
    [
        ("valid_route", "route"),
        ("valid_raw_route", "route"),
        ("valid_refusal", "semantic_no_route"),
        ("valid_content_filter", "semantic_no_route"),
        ("mixed_error_and_route", "failure"),
        ("error", "failure"),
    ],
    ids=("valid", "raw", "refusal", "content-filter", "mixed", "error"),
)
async def test_real_adapter_router_boundary_has_exact_attempt_accounting(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_accounting: str,
) -> None:
    """The production router classifies real adapter output before accounting."""

    sentinel = "router-boundary-secret-/srv/provider"
    route = {
        "choices": [
            {
                "message": {
                    "content": '{"provider":"openai","model":"routed"}'
                }
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }
    if outcome == "valid_route":
        provider_result = route
    elif outcome == "valid_raw_route":
        provider_result = '{"provider":"openai","model":"routed"}'
    elif outcome == "valid_refusal":
        provider_result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": "I cannot choose a route.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": route["usage"],
        }
    elif outcome == "valid_content_filter":
        provider_result = {
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
            "usage": route["usage"],
        }
    elif outcome == "mixed_error_and_route":
        provider_result = {
            "error": {"code": "provider_unavailable", "message": sentinel},
            **route,
        }
    else:
        provider_result = {
            "error": {"code": "provider_unavailable", "message": sentinel},
            "usage": route["usage"],
        }

    metrics = _RouterMetrics()
    provider_manager = _RouterProviderManager()
    usage_log = AsyncMock(return_value=None)
    marked: list[str] = []
    handle = await issue_provider_call_credentials_async(
        "openai",
        api_key="router-boundary-key",
        app_config=_registry_openai_app_config(),
        model="router-model",
    )

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "router-model")
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            marked.append(selected_handle.provider)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer router-boundary-key"
        assert payload["model"] == "router-model"
        assert payload["max_tokens"] == 64
        return provider_result

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_MODEL", "router-model")
    monkeypatch.setattr(chat_endpoint, "get_chat_metrics", lambda: metrics)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: provider_manager)
    monkeypatch.setattr(chat_endpoint, "log_model_router_usage", usage_log)
    router_request, policy, candidates, provider_listing = _real_router_inputs(
        f"router-boundary-{outcome}"
    )

    result = await chat_endpoint._select_auto_chat_llm_router_choice(
        router_request=router_request,
        policy=policy,
        candidates=candidates,
        provider_listing=provider_listing,
        request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
        current_user=SimpleNamespace(id=1),
        request_id=f"router-boundary-{outcome}",
        credential_runtime=Runtime(),
    )

    if expected_accounting == "route":
        assert result[0] == {"provider": "openai", "model": "routed"}
        assert marked == ["openai"]
        assert metrics.llm_calls == [("openai", "router-model", True, None)]
        assert len(metrics.token_calls) == 1
        assert provider_manager.failures == []
        assert provider_manager.successes == ["openai"]
        usage_log.assert_awaited_once()
        if outcome == "valid_raw_route":
            assert metrics.token_calls[0]["prompt_tokens"] == 0
            assert metrics.token_calls[0]["completion_tokens"] == 0
            assert usage_log.await_args.kwargs["estimated"] is True
    elif expected_accounting == "semantic_no_route":
        assert result[0] is None
        assert marked == ["openai"]
        assert metrics.llm_calls == [("openai", "router-model", True, None)]
        assert len(metrics.token_calls) == 1
        assert provider_manager.failures == []
        assert provider_manager.successes == ["openai"]
        usage_log.assert_awaited_once()
    else:
        assert result[0] is None
        assert sentinel not in repr(result)
        assert marked == []
        assert metrics.llm_calls == [
            ("openai", "router-model", False, "SanitizedProviderStreamError")
        ]
        assert sum(call[2] is False for call in metrics.llm_calls) == 1
        assert not any(call[2] is True for call in metrics.llm_calls)
        assert metrics.token_calls == []
        assert provider_manager.failures == [
            ("openai", "SanitizedProviderStreamError")
        ]
        assert provider_manager.successes == []
        usage_log.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_mark_failure_does_not_reclassify_provider_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A best-effort credential mark cannot discard a valid routed response."""

    metrics = _RouterMetrics()
    provider_manager = _RouterProviderManager()
    usage_log = AsyncMock(return_value=None)
    mark_attempts: list[str] = []
    handle = await issue_provider_call_credentials_async(
        "openai",
        api_key="router-mark-failure-key",
        app_config=_registry_openai_app_config(),
        model="router-model",
    )

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "router-model")
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            mark_attempts.append(selected_handle.provider)
            raise RuntimeError("credential mark persistence unavailable")

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer router-mark-failure-key"
        assert payload["model"] == "router-model"
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_MODEL", "router-model")
    monkeypatch.setattr(chat_endpoint, "get_chat_metrics", lambda: metrics)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: provider_manager)
    monkeypatch.setattr(chat_endpoint, "log_model_router_usage", usage_log)
    router_request, policy, candidates, provider_listing = _real_router_inputs(
        "router-mark-failure"
    )

    result = await chat_endpoint._select_auto_chat_llm_router_choice(
        router_request=router_request,
        policy=policy,
        candidates=candidates,
        provider_listing=provider_listing,
        request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
        current_user=SimpleNamespace(id=1),
        request_id="router-mark-failure",
        credential_runtime=Runtime(),
    )

    assert result[0] == {"provider": "openai", "model": "routed"}
    assert mark_attempts == ["openai"]
    assert metrics.llm_calls == [("openai", "router-model", True, None)]
    assert len(metrics.token_calls) == 1
    assert provider_manager.failures == []
    assert provider_manager.successes == ["openai"]
    usage_log.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mark_mode", ["exception", "explicit-false"])
async def test_concurrent_router_mark_failure_is_request_isolated(
    monkeypatch: pytest.MonkeyPatch,
    mark_mode: str,
) -> None:
    """Failed and explicitly false marks remain retryable and request-local."""

    adapter_ready = [threading.Event(), threading.Event()]
    release_adapter = threading.Event()
    mark_ready = [asyncio.Event(), asyncio.Event()]
    release_mark = [asyncio.Event(), asyncio.Event()]
    metrics = [_RouterMetrics(), _RouterMetrics()]
    provider_managers = [_RouterProviderManager(), _RouterProviderManager()]
    active_index: ContextVar[int] = ContextVar("active_router_mark_failure_index")
    mark_attempts = [0, 0]
    marked: list[list[str]] = [[], []]
    usage_log = AsyncMock(return_value=None)
    handles = [
        await issue_provider_call_credentials_async(
            "openai",
            api_key=f"router-mark-key-{index}",
            app_config=_registry_openai_app_config(),
            model="router-model",
        )
        for index in range(2)
    ]

    class Runtime:
        def __init__(self, index: int) -> None:
            self.index = index
            self.handle = handles[index]

        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "router-model")
            return self.handle

        async def mark_used(self, selected_handle: Any) -> bool:
            assert selected_handle is self.handle
            mark_attempts[self.index] += 1
            if (
                self.index == 0
                and mark_mode == "explicit-false"
                and mark_attempts[self.index] == 1
            ):
                return False
            mark_ready[self.index].set()
            await release_mark[self.index].wait()
            if self.index == 0 and mark_mode == "exception":
                raise RuntimeError("credential mark persistence unavailable")
            marked[self.index].append(selected_handle.provider)
            return True

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["model"] == "router-model"
        index = 0 if headers["Authorization"] == "Bearer router-mark-key-0" else 1
        adapter_ready[index].set()
        assert release_adapter.wait(timeout=2.0)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 5 + index,
                "completion_tokens": 2,
                "total_tokens": 7 + index,
            },
        }

    async def invoke(index: int) -> tuple[dict[str, str] | None, dict[str, Any]]:
        token = active_index.set(index)
        router_request, policy, candidates, provider_listing = _real_router_inputs(
            f"router-mark-concurrent-{index}"
        )
        try:
            return await chat_endpoint._select_auto_chat_llm_router_choice(
                router_request=router_request,
                policy=policy,
                candidates=candidates,
                provider_listing=provider_listing,
                request=SimpleNamespace(
                    state=SimpleNamespace(user_id=index, api_key_id=None)
                ),
                current_user=SimpleNamespace(id=index),
                request_id=f"router-mark-concurrent-{index}",
                credential_runtime=Runtime(index),
            )
        finally:
            active_index.reset(token)

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_MODEL", "router-model")
    monkeypatch.setattr(
        chat_endpoint,
        "get_chat_metrics",
        lambda: metrics[active_index.get()],
    )
    monkeypatch.setattr(
        chat_endpoint,
        "get_provider_manager",
        lambda: provider_managers[active_index.get()],
    )
    monkeypatch.setattr(chat_endpoint, "log_model_router_usage", usage_log)

    tasks = [asyncio.create_task(invoke(index)) for index in range(2)]
    try:
        observed = await asyncio.gather(
            *(asyncio.to_thread(event.wait, 1.0) for event in adapter_ready)
        )
        assert observed == [True, True]
        release_adapter.set()
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), timeout=1.0) for event in mark_ready)
        )
        assert all(task.done() is False for task in tasks)

        release_mark[1].set()
        healthy_result = await asyncio.wait_for(asyncio.shield(tasks[1]), timeout=1.0)
        assert healthy_result[0] == {"provider": "openai", "model": "routed"}
        assert marked == [[], ["openai"]]

        release_mark[0].set()
        first_result = await asyncio.wait_for(
            asyncio.shield(tasks[0]),
            timeout=1.0,
        )
    finally:
        release_adapter.set()
        for event in release_mark:
            event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert first_result[0] == {"provider": "openai", "model": "routed"}
    assert mark_attempts == ([1, 1] if mark_mode == "exception" else [2, 1])
    assert marked == (
        [[], ["openai"]]
        if mark_mode == "exception"
        else [["openai"], ["openai"]]
    )
    assert all(item.llm_calls == [("openai", "router-model", True, None)] for item in metrics)
    assert all(len(item.token_calls) == 1 for item in metrics)
    assert all(item.failures == [] for item in provider_managers)
    assert all(item.successes == ["openai"] for item in provider_managers)
    assert usage_log.await_count == 2
    assert {
        call.kwargs["context"].conversation_id for call in usage_log.await_args_list
    } == {"router-mark-concurrent-0", "router-mark-concurrent-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_marks", "expected_route"),
    [
        ("valid_route", 1, True),
        ("valid_raw_route", 1, True),
        ("valid_refusal", 1, False),
        ("valid_content_filter", 1, False),
        ("malformed_route", 1, False),
        ("empty", 0, False),
        ("error", 0, False),
        ("nested_error_prefix", 0, False),
        ("nested_structured_error_and_route", 0, False),
        ("error_prefix_with_route", 0, False),
        ("mixed_error_and_route", 0, False),
    ],
)
async def test_normal_chat_router_marks_each_semantic_provider_success(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_marks: int,
    expected_route: bool,
) -> None:
    """Credential use follows provider success, independent of route extraction."""

    sentinel = "normal-router-secret-/srv/provider"
    marked: list[str] = []
    handle = await issue_provider_call_credentials_async(
        "openai",
        api_key="router-key",
        app_config={},
        model="router-model",
    )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            marked.append(selected_handle.provider)

    def provider_result() -> Any:
        valid_route = {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ]
        }
        if outcome == "valid_route":
            return valid_route
        if outcome == "valid_raw_route":
            return '{"provider":"openai","model":"routed"}'
        if outcome == "valid_refusal":
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "refusal": "I cannot choose a route.",
                        },
                        "finish_reason": "stop",
                    }
                ]
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
                ]
            }
        if outcome == "malformed_route":
            return {"choices": [{"message": {"content": "not a route"}}]}
        if outcome == "empty":
            return {"choices": []}
        if outcome == "error":
            return {"error": {"code": "provider_unavailable", "message": sentinel}}
        if outcome == "nested_error_prefix":
            return {
                "choices": [
                    {"message": {"content": f"Error: {sentinel}"}}
                ]
            }
        if outcome == "nested_structured_error_and_route":
            return {
                "choices": [
                    {
                        "message": {
                            "error": {
                                "code": "provider_unavailable",
                                "message": sentinel,
                            },
                            "content": '{"provider":"openai","model":"routed"}',
                        }
                    }
                ]
            }
        if outcome == "error_prefix_with_route":
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                'Error: {"provider":"openai","model":"routed"}'
                            )
                        }
                    }
                ]
            }
        return {
            "error": {"code": "provider_unavailable", "message": sentinel},
            **valid_route,
        }

    async def provider_call(**_kwargs: Any) -> Any:
        return provider_result()

    async def select(**kwargs: Any) -> tuple[dict[str, str] | None, dict[str, Any]]:
        router_model = SimpleNamespace(provider="openai", model="router-model")
        response = await kwargs["execute_router_call"](router_model, [])
        return chat_endpoint.extract_router_choice(response), {}

    runtime = Runtime()
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", select)

    result = await chat_endpoint._select_auto_chat_llm_router_choice(
        router_request=SimpleNamespace(scope=None),
        policy=SimpleNamespace(),
        candidates=[],
        provider_listing={},
        request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
        current_user=SimpleNamespace(id=1),
        request_id=f"normal-router-{outcome}",
        credential_runtime=runtime,
    )

    assert marked == ["openai"] * expected_marks
    if expected_route:
        assert result[0] == {"provider": "openai", "model": "routed"}
    else:
        assert result[0] is None
        assert sentinel not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "semantic_outcome",
    ["valid_route", "valid_refusal", "valid_content_filter"],
    ids=("route", "refusal", "content-filter"),
)
async def test_concurrent_chat_router_results_are_request_isolated(
    monkeypatch: pytest.MonkeyPatch,
    semantic_outcome: str,
) -> None:
    """A mixed-error route cannot borrow a concurrent semantic success."""

    sentinel = "concurrent-router-secret-/srv/provider"
    ready = [threading.Event(), threading.Event()]
    release = threading.Event()
    marked: list[list[str]] = [[], []]
    metrics = [_RouterMetrics(), _RouterMetrics()]
    provider_managers = [_RouterProviderManager(), _RouterProviderManager()]
    active_index: ContextVar[int] = ContextVar("active_router_test_index")
    usage_log = AsyncMock(return_value=None)
    handles = [
        await issue_provider_call_credentials_async(
            "openai",
            api_key=f"router-key-{index}",
            app_config=_registry_openai_app_config(),
            model="router-model",
        )
        for index in range(2)
    ]

    class Runtime:
        def __init__(self, index: int) -> None:
            self.index = index
            self.handle = handles[index]

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return self.handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is self.handle
            marked[self.index].append(selected_handle.api_key)

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert payload["model"] == "router-model"
        index = 0 if headers["Authorization"] == "Bearer router-key-0" else 1
        ready[index].set()
        assert release.wait(timeout=2.0)
        route = {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 2,
                "total_tokens": 11,
            },
        }
        if index == 0:
            return {
                "error": {"code": "provider_unavailable", "message": sentinel},
                **route,
            }
        if semantic_outcome == "valid_route":
            return route
        if semantic_outcome == "valid_refusal":
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "refusal": "I cannot choose a route.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": route["usage"],
            }
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
            "usage": route["usage"],
        }

    async def invoke(index: int) -> tuple[dict[str, str] | None, dict[str, Any]]:
        index_token = active_index.set(index)
        router_request, policy, candidates, provider_listing = _real_router_inputs(
            f"router-scope-{index}"
        )
        try:
            return await chat_endpoint._select_auto_chat_llm_router_choice(
                router_request=router_request,
                policy=policy,
                candidates=candidates,
                provider_listing=provider_listing,
                request=SimpleNamespace(
                    state=SimpleNamespace(user_id=index, api_key_id=None)
                ),
                current_user=SimpleNamespace(id=index),
                request_id=f"concurrent-router-{index}",
                credential_runtime=Runtime(index),
            )
        finally:
            active_index.reset(index_token)

    _install_real_openai_adapter_transport(monkeypatch, responder)
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_MODEL", "router-model")
    monkeypatch.setattr(
        chat_endpoint,
        "get_chat_metrics",
        lambda: metrics[active_index.get()],
    )
    monkeypatch.setattr(
        chat_endpoint,
        "get_provider_manager",
        lambda: provider_managers[active_index.get()],
    )
    monkeypatch.setattr(chat_endpoint, "log_model_router_usage", usage_log)

    tasks = [asyncio.create_task(invoke(index)) for index in range(2)]
    try:
        observed = await asyncio.gather(
            *(asyncio.to_thread(event.wait, 1.0) for event in ready)
        )
        assert observed == [True, True]
        release.set()
        invalid_result, valid_result = await asyncio.gather(*tasks)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert invalid_result[0] is None
    assert sentinel not in repr(invalid_result)
    if semantic_outcome == "valid_route":
        assert valid_result[0] == {"provider": "openai", "model": "routed"}
    else:
        assert valid_result[0] is None
    assert marked == [[], ["router-key-1"]]
    assert metrics[0].llm_calls == [
        ("openai", "router-model", False, "SanitizedProviderStreamError")
    ]
    assert metrics[1].llm_calls == [("openai", "router-model", True, None)]
    assert metrics[0].token_calls == []
    assert len(metrics[1].token_calls) == 1
    assert metrics[1].token_calls[0]["provider"] == "openai"
    assert metrics[1].token_calls[0]["model"] == "router-model"
    assert provider_managers[0].failures == [
        ("openai", "SanitizedProviderStreamError")
    ]
    assert provider_managers[0].successes == []
    assert provider_managers[1].failures == []
    assert provider_managers[1].successes == ["openai"]
    usage_log.assert_awaited_once()
    assert usage_log.await_args.kwargs["provider"] == "openai"
    assert usage_log.await_args.kwargs["model"] == "router-model"
    assert usage_log.await_args.kwargs["context"].conversation_id == "router-scope-1"


@pytest.mark.asyncio
async def test_chat_router_valid_result_drains_mark_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot abandon a valid route's in-flight credential mark."""

    mark_entered = asyncio.Event()
    release_mark = asyncio.Event()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    marked: list[str] = []
    handle = await issue_provider_call_credentials_async(
        "openai",
        api_key="router-key",
        app_config={},
        model="router-model",
    )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            mark_entered.set()
            await release_mark.wait()
            marked.append(selected_handle.provider)

    async def provider_call(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ]
        }

    async def select(**kwargs: Any) -> tuple[dict[str, str] | None, dict[str, Any]]:
        router_model = SimpleNamespace(provider="openai", model="router-model")
        response = await kwargs["execute_router_call"](router_model, [])
        return chat_endpoint.extract_router_choice(response), {}

    runtime = Runtime()
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", select)

    task = asyncio.create_task(
        chat_endpoint._select_auto_chat_llm_router_choice(
            router_request=SimpleNamespace(scope="cancelled-router-mark"),
            policy=SimpleNamespace(),
            candidates=[],
            provider_listing={},
            request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
            current_user=SimpleNamespace(id=1),
            request_id="cancelled-router-mark",
            credential_runtime=runtime,
        )
    )
    try:
        await asyncio.wait_for(mark_entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert marked == []
        release_mark.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release_mark.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert marked == ["openai"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_route", 1),
        ("valid_refusal", 1),
        ("valid_content_filter", 1),
        ("empty", 0),
        ("error", 0),
        ("malformed_route", 1),
    ],
)
async def test_auto_router_cancellation_marks_only_semantic_late_success(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """A cancelled router call marks every semantic late provider success."""

    entered = asyncio.Event()
    release = asyncio.Event()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    marked: list[str] = []
    handle = await issue_provider_call_credentials_async(
        "openai",
        api_key="router-key",
        app_config={},
        model="router-model",
    )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle) -> None:
            assert selected_handle is handle
            marked.append(selected_handle.provider)

    def late_result() -> dict[str, object]:
        if late_outcome == "valid_route":
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"provider":"openai","model":"routed"}'
                        }
                    }
                ]
            }
        if late_outcome == "valid_refusal":
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "refusal": "I cannot choose a route.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        if late_outcome == "valid_content_filter":
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
                ]
            }
        if late_outcome == "empty":
            return {"choices": []}
        if late_outcome == "error":
            return {"error": {"code": "provider_unavailable"}}
        return {"choices": [{"message": {"content": "not a route"}}]}

    async def provider_call(**_kwargs):
        entered.set()
        await release.wait()
        return late_result()

    async def select(**kwargs):
        router_model = SimpleNamespace(provider="openai", model="router-model")
        await kwargs["execute_router_call"](router_model, [])
        return None, {}

    runtime = Runtime()
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", select)

    task = asyncio.create_task(
        chat_endpoint._select_auto_chat_llm_router_choice(
            router_request=SimpleNamespace(scope=None),
            policy=SimpleNamespace(),
            candidates=[],
            provider_listing={},
            request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
            current_user=SimpleNamespace(id=1),
            request_id="cancelled-auto-router",
            credential_runtime=runtime,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
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
async def test_cancelled_router_drains_real_adapter_before_classify_mark_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Router cancellation keeps its runtime open through late-result accounting."""

    lifecycle: list[str] = []
    adapter_entered = threading.Event()
    release_adapter = threading.Event()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)
    predicate = getattr(chat_service, "_nonstream_provider_result_is_usable", None)
    assert callable(predicate), "Chat must expose one shared non-stream result predicate"

    def classify(result: Any) -> bool:
        if adapter_entered.is_set():
            lifecycle.append("semantic-classify")
        return predicate(result)

    monkeypatch.setattr(chat_service, "_nonstream_provider_result_is_usable", classify)
    if hasattr(chat_endpoint, "_nonstream_provider_result_is_usable"):
        monkeypatch.setattr(
            chat_endpoint,
            "_nonstream_provider_result_is_usable",
            classify,
        )

    handle = await issue_provider_call_credentials_async(
        "openai",
        api_key="cancelled-router-key",
        app_config=_registry_openai_app_config(),
        model="router-model",
    )

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "router-model")
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("runtime-close")

    def responder(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        assert url == f"{_REGISTRY_OPENAI_BASE_URL}/chat/completions"
        assert headers["Authorization"] == "Bearer cancelled-router-key"
        assert payload["model"] == "router-model"
        adapter_entered.set()
        assert release_adapter.wait(timeout=2.0)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ]
        }

    def on_client_exit() -> None:
        lifecycle.append("adapter-exit")

    _install_real_openai_adapter_transport(
        monkeypatch,
        responder,
        on_client_exit=on_client_exit,
    )
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_ROUTING_ROUTER_MODEL", "router-model")
    monkeypatch.setattr(chat_endpoint, "get_chat_metrics", lambda: _RouterMetrics())
    monkeypatch.setattr(
        chat_endpoint,
        "get_provider_manager",
        lambda: _RouterProviderManager(),
    )
    monkeypatch.setattr(chat_endpoint, "log_model_router_usage", AsyncMock(return_value=None))
    router_request, policy, candidates, provider_listing = _real_router_inputs(
        "cancelled-real-router"
    )
    runtime = Runtime()

    async def invoke() -> tuple[dict[str, str] | None, dict[str, Any]]:
        try:
            return await chat_endpoint._select_auto_chat_llm_router_choice(
                router_request=router_request,
                policy=policy,
                candidates=candidates,
                provider_listing=provider_listing,
                request=SimpleNamespace(
                    state=SimpleNamespace(user_id=1, api_key_id=None)
                ),
                current_user=SimpleNamespace(id=1),
                request_id="cancelled-real-router",
                credential_runtime=runtime,
            )
        finally:
            await runtime.close()

    task = asyncio.create_task(invoke())
    try:
        assert await asyncio.to_thread(adapter_entered.wait, 1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert lifecycle == []
        release_adapter.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release_adapter.set()
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
async def test_auto_router_cancellation_drains_usage_log_before_reraise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable router usage accounting finishes before cancellation escapes."""

    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    drain_entered = _install_owned_worker_drain_probe(monkeypatch)

    async def blocking_usage_log(*_args: Any, **_kwargs: Any) -> None:
        entered.set()
        await release.wait()
        finished.set()

    async def select(**kwargs: Any):
        await kwargs["log_router_usage"](
            SimpleNamespace(provider="openai", model="router-model"),
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            1.0,
        )
        return None, {}

    monkeypatch.setattr(chat_endpoint, "log_model_router_usage", blocking_usage_log)
    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", select)

    task = asyncio.create_task(
        chat_endpoint._select_auto_chat_llm_router_choice(
            router_request=SimpleNamespace(scope="chat-1"),
            policy=SimpleNamespace(),
            candidates=[],
            provider_listing={},
            request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
            current_user=SimpleNamespace(id=1),
            request_id="cancelled-auto-router-usage",
            credential_runtime=SimpleNamespace(),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert finished.is_set() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert finished.is_set() is True


@pytest.mark.asyncio
async def test_auto_router_propagates_cancellation_from_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Router selection cancellation remains control flow."""

    entered = asyncio.Event()
    release = asyncio.Event()

    async def select(**_kwargs: Any):
        entered.set()
        await release.wait()
        return None, {}

    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", select)

    task = asyncio.create_task(
        chat_endpoint._select_auto_chat_llm_router_choice(
            router_request=SimpleNamespace(scope="chat-1"),
            policy=SimpleNamespace(),
            candidates=[],
            provider_listing={},
            request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
            current_user=SimpleNamespace(id=1),
            request_id="cancelled-auto-router-selection",
            credential_runtime=SimpleNamespace(),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_auto_router_does_not_swallow_terminal_credential_error(monkeypatch):
    runtime_type = _credential_runtime_double(
        errors={
            "openai": ByokResolutionError(
                "credential_store_unavailable",
                "openai",
            )
        }
    )
    runtime = runtime_type(
        user_id=1,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=False,
        fallback_resolver=lambda _provider: None,
    )

    async def swallowing_select(**kwargs):
        router_model = SimpleNamespace(provider="openai", model="gpt-4o-mini")
        try:
            await kwargs["execute_router_call"](
                router_model,
                [{"role": "user", "content": "route"}],
            )
        except Exception as exc:  # noqa: BLE001 - selector must mimic arbitrary failures
            return None, {"error": type(exc).__name__}
        raise AssertionError("credential resolution should fail")

    monkeypatch.setattr(chat_endpoint, "select_llm_router_choice", swallowing_select)

    with pytest.raises(ByokResolutionError) as exc_info:
        await chat_endpoint._select_auto_chat_llm_router_choice(
            router_request=SimpleNamespace(scope=None),
            policy=SimpleNamespace(),
            candidates=[],
            provider_listing={},
            request=SimpleNamespace(state=SimpleNamespace(user_id=1, api_key_id=None)),
            current_user=SimpleNamespace(id=1),
            request_id="request-credential-error",
            credential_runtime=runtime,
        )

    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.parametrize(
    ("runtime_error", "expected_code"),
    [
        (
            ByokResolutionError("invalid_provider_credentials", "openai"),
            "invalid_provider_credentials",
        ),
        (
            ByokResolutionError("credential_store_unavailable", "openai"),
            "credential_store_unavailable",
        ),
        (
            ByokResolutionError("credential_scope_revoked", "openai"),
            "credential_scope_revoked",
        ),
    ],
)
def test_chat_credential_resolution_errors_are_terminal_503(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    runtime_error,
    expected_code,
):
    runtime_type = _credential_runtime_double(errors={"openai": runtime_error})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "API_KEYS", {"openai": "legacy-key"}),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == expected_code
    provider_manager.get_available_provider.assert_not_called()
    provider_call.assert_not_called()


def test_chat_missing_runtime_credentials_are_terminal_503(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(api_keys={"openai": None})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "API_KEYS", {"openai": "legacy-key"}),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "missing_provider_credentials"
    provider_manager.get_available_provider.assert_not_called()
    provider_call.assert_not_called()


def test_test_mode_invalid_runtime_key_maps_to_downstream_auth_502(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(api_keys={"openai": "invalid-runtime-key-sentinel"})
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "_shared_is_test_mode", return_value=True),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    detail = response.json()["detail"]
    assert detail["error_code"] == "provider_authentication_failed"
    assert "sentinel" not in str(detail).lower()
    provider_call.assert_not_called()


def test_health_fallback_configuration_error_preserves_structured_503(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {}
    provider_manager.get_available_provider.return_value = "anthropic"
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    original_builder = chat_endpoint.build_call_params_from_request

    def build_or_fail(**kwargs):
        if kwargs["target_api_provider"] == "anthropic":
            raise ChatConfigurationError(
                "sentinel fallback config path",
                provider="anthropic",
            )
        return original_builder(**kwargs)

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(
            chat_endpoint,
            "build_call_params_from_request",
            side_effect=build_or_fail,
        ),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = response.json()["detail"]
    assert detail["error_code"] == "provider_configuration_invalid"
    assert "sentinel" not in str(detail).lower()
    provider_call.assert_not_called()


@pytest.mark.parametrize(
    ("provider_error", "expected_status", "expected_code"),
    [
        (
            ChatAuthenticationError("sentinel upstream body", provider="openai"),
            status.HTTP_502_BAD_GATEWAY,
            "provider_authentication_failed",
        ),
        (
            ChatConfigurationError("sentinel config path", provider="openai"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "provider_configuration_invalid",
        ),
    ],
)
def test_chat_provider_auth_and_config_errors_are_sanitized_and_terminal(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    provider_error,
    expected_status,
    expected_code,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "API_KEYS", {"openai": "legacy-key"}),
        patch.object(
            chat_endpoint,
            "execute_non_stream_call",
            side_effect=provider_error,
        ),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert detail["error_code"] == expected_code
    assert "sentinel" not in str(detail).lower()
    provider_manager.get_available_provider.assert_not_called()
    assert runtime_type.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_stream_runtime_cleanup_runs_on_completion_error_and_cancel():
    runtime_type = _credential_runtime_double()

    async def consume(mode: str) -> int:
        runtime = runtime_type(
            user_id=1,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            fallback_resolver=lambda _provider: None,
        )

        async def body():
            yield b"first"
            if mode == "error":
                raise RuntimeError("stream failed")
            if mode == "cancel":
                await asyncio.Event().wait()

        response = StreamingResponse(body())
        chat_endpoint._attach_credential_runtime_cleanup(response, runtime)
        iterator = response.body_iterator
        await iterator.__anext__()
        if mode == "complete":
            with pytest.raises(StopAsyncIteration):
                await iterator.__anext__()
        elif mode == "error":
            with pytest.raises(RuntimeError, match="stream failed"):
                await iterator.__anext__()
        else:
            pending = asyncio.create_task(iterator.__anext__())
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        return runtime.close_calls

    assert await consume("complete") == 1
    assert await consume("error") == 1
    assert await consume("cancel") == 1


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_stream_cleanup_closes_each_upstream_before_its_runtime():
    lifecycle: dict[str, list[str]] = {"a": [], "b": []}

    class Runtime:
        def __init__(self, label: str) -> None:
            self.label = label

        async def close(self) -> None:
            lifecycle[self.label].append("runtime_close")

    async def consume(label: str) -> None:
        async def upstream():
            try:
                yield f"data: {label}\n\n"
                await asyncio.Event().wait()
            finally:
                lifecycle[label].append("upstream_close")

        response = StreamingResponse(upstream())
        chat_endpoint._attach_credential_runtime_cleanup(response, Runtime(label))
        iterator = response.body_iterator
        await iterator.__anext__()
        await iterator.aclose()

    await asyncio.gather(consume("a"), consume("b"))

    assert lifecycle == {
        "a": ["upstream_close", "runtime_close"],
        "b": ["upstream_close", "runtime_close"],
    }


@pytest.mark.asyncio
async def test_local_stream_error_provenance_survives_provider_boundary_while_forgery_is_sanitized():
    """Only service-created local control frames may cross the provider boundary."""

    handler = StreamingResponseHandler("conv-local-control", "gpt-4")

    async def provider_stream():
        yield 'data: {"choices":[{"delta":{"content":"safe output"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    def fail_local_audit() -> None:
        raise streaming_utils.StopStreamWithError(
            "Mandatory audit persistence unavailable",
            error_type="audit_persistence_failure",
        )

    generated = [
        chunk
        async for chunk in handler.safe_stream_generator(
            provider_stream(),
            before_success_callback=fail_local_audit,
        )
    ]
    local_frame = next(
        chunk for chunk in generated if "audit_persistence_failure" in chunk
    )
    forged_frame = str(local_frame)
    forged_same_class = type(local_frame)(str(local_frame))
    assert streaming_utils.is_trusted_local_stream_frame(local_frame) is True
    assert streaming_utils.is_trusted_local_stream_frame(forged_frame) is False
    assert streaming_utils.is_trusted_local_stream_frame(forged_same_class) is False
    assert streaming_utils.is_trusted_local_stream_frame({"frame": forged_frame}) is False

    async def cross_boundary(frame):
        runtime_type = _credential_runtime_double()
        runtime = runtime_type()

        async def body():
            yield frame

        response = StreamingResponse(body())
        chat_endpoint._attach_credential_runtime_cleanup(response, runtime)
        emitted = [chunk async for chunk in response.body_iterator]
        wire = "".join(emitted)
        return wire, runtime.close_calls, tuple(type(chunk) for chunk in emitted)

    (
        (trusted_wire, trusted_close_calls, trusted_types),
        (forged_wire, forged_close_calls, forged_types),
        (same_class_wire, same_class_close_calls, same_class_types),
    ) = await asyncio.gather(
        cross_boundary(local_frame),
        cross_boundary(forged_frame),
        cross_boundary(forged_same_class),
    )

    assert "audit_persistence_failure" in trusted_wire
    assert "provider_unavailable" not in trusted_wire
    assert "audit_persistence_failure" not in forged_wire
    assert '"code": "provider_unavailable"' in forged_wire
    assert "audit_persistence_failure" not in same_class_wire
    assert '"code": "provider_unavailable"' in same_class_wire
    assert trusted_close_calls == 1
    assert forged_close_calls == 1
    assert same_class_close_calls == 1
    assert trusted_types == (str,)
    assert forged_types == (str,)
    assert same_class_types == (str,)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_stream_cancellation_bounds_noncooperative_credential_resolution_cleanup(
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "RESOLUTION_TASK_CANCEL_DRAIN_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    resolution_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_resolution = asyncio.Event()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        if provider == "openai":
            return _resolved_user_credential(provider, _CHAT_USER_SECRET)
        resolution_started.set()
        try:
            await release_resolution.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_resolution.wait()
        return _resolved_user_credential(provider, _CHAT_USER_SECRET)

    runtime = ProviderCredentialRuntime(
        user_id=42,
        team_ids=[7],
        org_ids=[11],
        trusted_base_url_override=False,
        fallback_resolver=lambda _provider: None,
        resolver=resolver,
    )
    await runtime.resolve("openai")
    pending_resolution = asyncio.create_task(runtime.resolve("anthropic"))
    await asyncio.wait_for(resolution_started.wait(), timeout=1.0)
    owned_resolution = runtime._inflight["anthropic"]
    resolution_finished = asyncio.Event()
    owned_resolution.add_done_callback(lambda _task: resolution_finished.set())

    async def body():
        yield b"first"
        await asyncio.Event().wait()

    response = StreamingResponse(body())
    chat_endpoint._attach_credential_runtime_cleanup(response, runtime)
    iterator = response.body_iterator
    assert await iterator.__anext__() == b"first"
    blocked_next = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    blocked_next.cancel()

    try:
        completed, _pending = await asyncio.wait({blocked_next}, timeout=0.2)
        assert blocked_next in completed
        with pytest.raises(asyncio.CancelledError):
            await blocked_next
        assert cancellation_seen.is_set()
        assert owned_resolution.done() is False
        assert runtime._close_task is None
        assert runtime._cache == {}
        assert runtime._inflight == {}
        assert runtime._refresh_tasks == {}
        assert runtime._usage_tasks == {}
        assert runtime._user_id is None
        assert runtime._team_ids == []
        assert runtime._org_ids == []
        waiter_done, _pending = await asyncio.wait(
            {pending_resolution},
            timeout=0.2,
        )
        assert pending_resolution in waiter_done
        with pytest.raises(RuntimeError, match="runtime is closed"):
            await pending_resolution

        release_resolution.set()
        await asyncio.wait_for(resolution_finished.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert owned_resolution.get_coro().cr_frame is None
        assert owned_resolution._log_traceback is False
    finally:
        release_resolution.set()
        await asyncio.gather(
            owned_resolution,
            pending_resolution,
            blocked_next,
            return_exceptions=True,
        )
        await runtime.close()


@pytest.mark.parametrize(
    "failure_point",
    [
        "auto_router",
        "provider_resolution",
        "metrics_context",
        "metrics_enter",
        "metrics_exit",
    ],
)
def test_chat_closes_runtime_for_every_setup_failure_after_construction(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    failure_point,
):
    runtime_type = _credential_runtime_double()
    request_data = ChatCompletionRequest(
        model="auto" if failure_point == "auto_router" else "gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=failure_point == "metrics_exit",
    )
    failure = RuntimeError(f"injected {failure_point} failure")

    with ExitStack() as stack:
        stack.enter_context(patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type))
        stack.enter_context(patch.object(chat_endpoint, "API_KEYS", {"openai": "legacy-key"}))
        if failure_point == "auto_router":
            stack.enter_context(
                patch.object(
                    chat_endpoint,
                    "_resolve_auto_chat_routing_decision",
                    AsyncMock(side_effect=failure),
                )
            )
        elif failure_point == "provider_resolution":
            stack.enter_context(
                patch.object(
                    chat_endpoint,
                    "resolve_provider_and_model",
                    side_effect=failure,
                )
            )
        else:
            metrics = MagicMock()
            if failure_point == "metrics_context":
                metrics.track_request.side_effect = failure
            else:
                track_context = MagicMock()
                track_context.__aenter__ = AsyncMock(
                    side_effect=failure if failure_point == "metrics_enter" else None,
                )
                track_context.__aexit__ = AsyncMock(
                    side_effect=failure if failure_point == "metrics_exit" else None,
                    return_value=None,
                )
                metrics.track_request.return_value = track_context
            stack.enter_context(patch.object(chat_endpoint, "get_chat_metrics", return_value=metrics))
            if failure_point == "metrics_exit":
                async def successful_stream():
                    yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

                stack.enter_context(
                    patch.object(
                        chat_endpoint,
                        "execute_streaming_call",
                        AsyncMock(return_value=StreamingResponse(successful_stream())),
                    )
                )

        with pytest.raises(RuntimeError, match=failure_point):
            authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )

    assert len(runtime_type.instances) == 1
    assert runtime_type.instances[0].close_calls == 1


@pytest.mark.parametrize("failure_point", ["audit", "usage"])
def test_chat_closes_runtime_for_unexpected_setup_baseexception(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    failure_point,
):
    class InjectedSetupFailure(Exception):
        pass

    runtime_type = _credential_runtime_double()
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    dependency = (
        chat_endpoint.get_audit_service_for_user if failure_point == "audit" else chat_endpoint.get_usage_event_logger
    )
    service = MagicMock()
    if failure_point == "audit":
        service.log_event = AsyncMock(side_effect=InjectedSetupFailure("injected audit failure"))
    else:
        service.log_event.side_effect = InjectedSetupFailure("injected usage failure")

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.dict(app.dependency_overrides, {dependency: lambda: service}),
        pytest.raises(InjectedSetupFailure, match=failure_point),
    ):
        authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert len(runtime_type.instances) == 1
    assert runtime_type.instances[0].close_calls == 1


def test_chat_closes_runtime_when_billing_exit_fails_after_stream_response_created(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    class UnexpectedBillingExit(Exception):
        pass

    runtime_type = _credential_runtime_double()
    billing_enforcer = MagicMock()
    billing_enforcer.__aenter__ = AsyncMock(return_value=billing_enforcer)
    billing_enforcer.__aexit__ = AsyncMock(side_effect=UnexpectedBillingExit("billing exit failed"))
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    async def successful_stream():
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "enforcement_enabled", return_value=True),
        patch.object(chat_endpoint, "LimitEnforcer", return_value=billing_enforcer),
        patch.object(
            chat_endpoint,
            "execute_streaming_call",
            AsyncMock(return_value=StreamingResponse(successful_stream())),
        ),
        patch.dict(app.dependency_overrides, {chat_endpoint.get_billing_org_id: lambda: 7}),
        pytest.raises(UnexpectedBillingExit, match="billing exit failed"),
    ):
        authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert runtime_type.instances[0].close_calls == 1


def test_chat_closes_runtime_when_rg_refund_fails_after_stream_response_created(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    class UnexpectedRGRefund(Exception):
        pass

    class PendingResponseError(Exception):
        pass

    runtime_type = _credential_runtime_double()
    governor = MagicMock()
    governor.reserve = AsyncMock(return_value=(SimpleNamespace(allowed=True), "rg-handle"))
    governor.commit = AsyncMock(side_effect=UnexpectedRGRefund("rg refund failed"))
    policy_loader = MagicMock()
    policy_loader.get_policy.return_value = {}
    pending_error = PendingResponseError("response handoff pending")
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    async def successful_stream():
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch("tldw_Server_API.app.core.config.rg_enabled", return_value=True),
        patch.object(app.state, "rg_governor", governor, create=True),
        patch.object(app.state, "rg_policy_loader", policy_loader, create=True),
        patch.object(chat_endpoint.sys, "exc_info", return_value=(PendingResponseError, pending_error, None)),
        patch.object(
            chat_endpoint,
            "execute_streaming_call",
            AsyncMock(return_value=StreamingResponse(successful_stream())),
        ),
        pytest.raises(UnexpectedRGRefund, match="rg refund failed"),
    ):
        authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert governor.commit.await_count == 1
    assert runtime_type.instances[0].close_calls == 1


def test_chat_completion_openai_oauth_auth_failure_retries_once(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    runtime_type = _credential_runtime_double(auth_source="oauth")

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "perform_chat_api_call") as mock_llm,
    ):
        mock_llm.side_effect = [
            ChatAuthenticationError("expired oauth access token", provider="openai"),
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Recovered after refresh"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert mock_llm.call_count == 2
    assert runtime_type.instances[0].resolve_calls[:2] == [
        ("openai", False),
        ("openai", True),
    ]
    assert mock_llm.call_args_list[1].kwargs["credentials_resolved"] is True


def test_chat_completion_openai_oauth_propagates_original_auth_error_after_second_failure(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    runtime_type = _credential_runtime_double(auth_source="oauth")

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type, create=True),
        patch.object(chat_endpoint, "perform_chat_api_call") as mock_llm,
    ):
        mock_llm.side_effect = [
            ChatAuthenticationError("expired oauth access token", provider="openai"),
            ChatAuthenticationError("oauth refresh token revoked", provider="openai"),
        ]

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert runtime_type.instances[0].resolve_calls[:2] == [
        ("openai", False),
        ("openai", True),
    ]


def test_chat_completion_openai_oauth_refresh_failure_does_not_log_raw_error(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    sentinel = "oauth-refresh-secret-/srv/token-store"
    runtime_type = _credential_runtime_double(
        auth_source="oauth",
        refresh_errors={"openai": RuntimeError(sentinel)},
    )
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(
                chat_endpoint,
                "perform_chat_api_call",
                side_effect=ChatAuthenticationError("expired oauth access token", provider="openai"),
            ) as provider_call,
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert provider_call.call_count == 1
    assert runtime_type.instances[0].resolve_calls[:2] == [
        ("openai", False),
        ("openai", True),
    ]
    assert sentinel not in response.text
    assert sentinel not in "".join(logs)


def test_real_openai_adapter_nonstream_failure_is_bounded_across_endpoint_logs_and_audit(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter as adapter_module

    sentinel = "openai-body-secret-/srv/provider-https://upstream.invalid"
    runtime_key = "sk-runtime-secret-must-not-leak"
    factory_calls = 0

    def fail_before_dispatch(**_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise OSError(f"{sentinel} Authorization: Bearer {runtime_key}")

    runtime_type = _credential_runtime_double(api_keys={"openai": runtime_key})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    audit_service = MagicMock()
    audit_service.log_event = AsyncMock()
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
            patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False),
            patch.object(chat_endpoint, "get_request_queue", return_value=None),
            patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
            patch.object(chat_endpoint, "_shared_is_test_mode", return_value=False),
            patch.object(adapter_module, "http_client_factory", fail_before_dispatch),
            patch.dict(
                app.dependency_overrides,
                {chat_endpoint.get_audit_service_for_user: lambda: audit_service},
            ),
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    serialized_audit_calls = repr(audit_service.log_event.await_args_list)
    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "The chat service provider is currently unavailable."
    assert factory_calls == 1
    assert sentinel not in response.text
    assert runtime_key not in response.text
    assert sentinel not in "".join(logs)
    assert runtime_key not in "".join(logs)
    assert sentinel not in serialized_audit_calls
    assert runtime_key not in serialized_audit_calls


class UnknownAdapterFailure(Exception):
    """Adapter exception deliberately absent from legacy finite catch tuples."""


@pytest.mark.parametrize("execution_mode", ["direct", "queued", "fallback"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, UnknownAdapterFailure])
def test_plain_provider_exception_is_bounded_across_endpoint_logs_and_audit(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    execution_mode,
    error_type,
):
    sentinel = f"{execution_mode}-{error_type.__name__}-secret-/srv/provider-https://upstream.invalid"

    class BoundaryQueue:
        allow_in_test_mode = True

        def is_running(self):
            return True

        async def enqueue(self, *, processor, **_kwargs):
            return asyncio.create_task(asyncio.to_thread(processor))

    queue = BoundaryQueue() if execution_mode == "queued" else None
    requested_provider = "anthropic" if execution_mode == "fallback" else "openai"
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        requested_provider: SimpleNamespace(can_attempt_call=lambda: True)
    }
    provider_manager.get_available_provider.return_value = "openai"
    audit_service = MagicMock()
    audit_service.log_event = AsyncMock()
    request_data = ChatCompletionRequest(
        model="claude-3" if execution_mode == "fallback" else "gpt-4o-mini",
        api_provider=requested_provider,
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    primary_side_effect: BaseException
    if execution_mode == "fallback":
        primary_side_effect = _certified_pre_dispatch(
            ChatProviderError(
                provider="anthropic",
                message="bounded certified primary failure",
                status_code=502,
            )
        )
    else:
        primary_side_effect = error_type(sentinel)

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type))
            stack.enter_context(
                patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager)
            )
            stack.enter_context(
                patch.object(
                    chat_endpoint,
                    "ENABLE_PROVIDER_FALLBACK",
                    execution_mode == "fallback",
                )
            )
            stack.enter_context(patch.object(chat_endpoint, "get_request_queue", return_value=queue))
            stack.enter_context(patch.object(chat_service, "get_request_queue", return_value=queue))
            stack.enter_context(
                patch.object(chat_endpoint, "QUEUED_EXECUTION", execution_mode == "queued")
            )
            stack.enter_context(patch.object(chat_endpoint, "_shared_is_test_mode", return_value=False))
            stack.enter_context(
                patch.object(chat_endpoint, "_should_enforce_strict_model_selection", return_value=False)
            )
            stack.enter_context(
                patch.object(
                    chat_endpoint,
                    "perform_chat_api_call",
                    side_effect=primary_side_effect,
                )
            )
            stack.enter_context(
                patch.object(
                    chat_service,
                    "perform_chat_api_call_async",
                    AsyncMock(side_effect=error_type(sentinel)),
                )
            )
            stack.enter_context(
                patch.dict(
                    app.dependency_overrides,
                    {chat_endpoint.get_audit_service_for_user: lambda: audit_service},
                )
            )
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    serialized_audit_calls = repr(audit_service.log_event.await_args_list)
    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "The chat service provider is currently unavailable."
    assert sentinel not in response.text
    assert sentinel not in "".join(logs)
    assert sentinel not in serialized_audit_calls
    if execution_mode == "fallback":
        provider_manager.get_available_provider.assert_called_once_with(exclude=["anthropic"])
    else:
        provider_manager.get_available_provider.assert_not_called()


def test_health_fallback_to_anthropic_does_not_refresh_requested_openai_oauth(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(auth_sources={"openai": "oauth", "anthropic": None})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {}
    provider_manager.get_available_provider.return_value = "anthropic"
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        provider_call.side_effect = [
            ChatAuthenticationError("anthropic rejected key", provider="anthropic"),
            {
                "id": "must-not-retry-openai",
                "choices": [{"message": {"role": "assistant", "content": "wrong retry"}}],
            },
        ]
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert provider_call.call_count == 1
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls


def test_health_fallback_to_openai_oauth_refreshes_selected_provider_once(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(auth_sources={"anthropic": None, "openai": "oauth"})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {}
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        provider_call.side_effect = [
            ChatAuthenticationError("expired selected oauth", provider="openai"),
            {
                "id": "chatcmpl-selected-oauth",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "refreshed"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert provider_call.call_count == 2
    assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1


@pytest.mark.parametrize(
    ("lazy_error", "expected_status", "expected_code"),
    [
        (
            ChatAuthenticationError("sentinel lazy auth body", provider="openai"),
            status.HTTP_502_BAD_GATEWAY,
            "provider_authentication_failed",
        ),
        (
            ChatConfigurationError("sentinel lazy config body", provider="openai"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "provider_configuration_invalid",
        ),
    ],
)
def test_streaming_preoutput_typed_error_maps_before_response_handoff(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    lazy_error,
    expected_status,
    expected_code,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def lazy_stream():
        if False:
            yield ""
        raise lazy_error

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", return_value=lazy_stream()),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert detail["error_code"] == expected_code
    assert "sentinel" not in str(detail).lower()
    assert runtime_type.instances[0].close_calls == 1


@pytest.mark.parametrize(
    ("eager_error", "expected_status", "expected_code"),
    [
        (
            ChatAuthenticationError("sentinel eager auth body", provider="openai"),
            status.HTTP_502_BAD_GATEWAY,
            "provider_authentication_failed",
        ),
        (
            ChatConfigurationError("sentinel eager config body", provider="openai"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "provider_configuration_invalid",
        ),
    ],
)
def test_streaming_eager_typed_error_maps_before_response_handoff(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    eager_error,
    expected_status,
    expected_code,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=eager_error,
        ),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert detail["error_code"] == expected_code
    assert "sentinel" not in str(detail).lower()
    provider_manager.get_available_provider.assert_not_called()


def test_streaming_lazy_openai_oauth_refreshes_once_before_output(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def expired_stream():
        if False:
            yield ""
        raise _certified_pre_dispatch(
            ChatAuthenticationError("sentinel expired oauth", provider="openai")
        )

    def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        provider_call.side_effect = [expired_stream(), refreshed_stream()]
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert "recovered" in response.text
    assert "sentinel" not in response.text.lower()
    assert provider_call.call_count == 2
    assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1


def test_real_openai_401_refreshes_oauth_once_before_stream_output(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    import httpx

    from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    sentinel = "real-openai-401-secret-/srv/provider"
    initial_key = "openai-runtime-key"
    refreshed_key = "openai-refreshed-runtime-key"
    stream_keys: list[str] = []

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.request = httpx.Request(
                "POST",
                "https://api.openai.test/v1/chat/completions",
            )
            self.text = sentinel if status_code == 401 else ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def json(self):
            return {"error": {"message": sentinel}}

        def raise_for_status(self):
            if self.status_code == 401:
                raise httpx.HTTPStatusError(
                    sentinel,
                    request=self.request,
                    response=self,
                )

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"real recovered"}}]}',
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, _method, _url, *, headers, **_kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            stream_keys.append(token)
            return Response(401 if token == initial_key else 200)

    runtime_type = _credential_runtime_double(auth_source="oauth")
    pool = BoundedDaemonPool(capacity=1)
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    with (
        patch.dict(os.environ, {"STREAMS_UNIFIED": "1"}),
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "_shared_is_test_mode", return_value=False),
        patch.object(chat_endpoint, "_should_enforce_strict_model_selection", return_value=False),
        patch.object(openai_adapter, "http_client_factory", lambda **_kwargs: Client()),
        patch.object(bounded_daemon_module, "STREAM_DAEMON_POOL", pool),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert "real recovered" in response.text
    assert sentinel not in response.text
    assert stream_keys == [initial_key, refreshed_key]
    assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1
    assert runtime_type.instances[0].close_calls == 1
    assert pool.active_count == 0


@pytest.mark.parametrize("oauth_enabled", [True, False])
def test_streaming_keepalive_before_auth_error_does_not_certify_replay(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    oauth_enabled,
):
    runtime_type = _credential_runtime_double(auth_source="oauth" if oauth_enabled else None)
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def keepalive_then_error():
        yield ": keepalive\n\n"
        yield 'data: {"choices":[{"delta":{}}]}\n\n'
        raise _certified_pre_dispatch(
            ChatAuthenticationError("sentinel keepalive auth", provider="openai")
        )

    provider_side_effect = [keepalive_then_error()]
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=provider_side_effect,
        ) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert provider_call.call_count == 1
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert "sentinel" not in response.text.lower()


@pytest.mark.parametrize("control_line", ["id: stream-7", "retry: 1500"])
@pytest.mark.parametrize("oauth_enabled", [True, False])
def test_streaming_sse_control_field_before_auth_error_does_not_certify_replay(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    control_line,
    oauth_enabled,
):
    runtime_type = _credential_runtime_double(auth_source="oauth" if oauth_enabled else None)
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def control_then_error():
        yield f"{control_line}\n\n"
        raise ChatAuthenticationError("sentinel control auth", provider="openai")

    side_effect = [control_then_error()]
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=side_effect) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert provider_call.call_count == 1
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert "sentinel" not in response.text.lower()


@pytest.mark.parametrize("oauth_enabled", [True, False])
def test_streaming_delayed_sse_controls_do_not_certify_replay(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    oauth_enabled,
):
    runtime_type = _credential_runtime_double(auth_source="oauth" if oauth_enabled else None)
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    async def delayed_control_then_error():
        yield "id: delayed-stream\n\n"
        await asyncio.sleep(0)
        yield "retry: 2500\n\n"
        await asyncio.sleep(0)
        yield ": keepalive\n\n"
        await asyncio.sleep(0)
        raise ChatAuthenticationError("sentinel delayed control auth", provider="openai")

    side_effect = [delayed_control_then_error()]
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.dict(os.environ, {"MODERATION_STREAM_BUFFER_CHARS": "0"}),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=side_effect) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert "id: delayed-stream" not in response.text
    assert "retry: 2500" not in response.text
    assert "sentinel" not in response.text.lower()
    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert provider_call.call_count == 1
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls


def test_execution_fallback_to_openai_oauth_refreshes_nonstream_and_marks_active_handle(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(auth_sources={"anthropic": None, "openai": "oauth"})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"anthropic": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )
    recovered = {
        "id": "chatcmpl-fallback-oauth",
        "choices": [{"message": {"role": "assistant", "content": "recovered"}, "finish_reason": "stop"}],
    }

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=[
                _certified_pre_dispatch(
                    ChatProviderError(provider="anthropic", message="initial failed", status_code=502)
                ),
                recovered,
            ],
        ) as provider_call,
        patch.object(
            chat_service,
            "perform_chat_api_call_async",
            AsyncMock(side_effect=ChatAuthenticationError("sentinel fallback auth", provider="openai")),
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert "sentinel" not in response.text.lower()
    assert provider_call.call_count == 2
    assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1
    assert runtime_type.instances[0].marked_used == ["openai"]


def test_execution_fallback_oauth_refresh_failure_is_terminal_nonstream(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(
        auth_sources={"anthropic": None, "openai": "oauth"},
        refresh_errors={"openai": ByokResolutionError("invalid_provider_credentials", "openai")},
    )
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"anthropic": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=_certified_pre_dispatch(
                ChatProviderError(provider="anthropic", message="initial failed", status_code=502)
            ),
        ) as provider_call,
        patch.object(
            chat_service,
            "perform_chat_api_call_async",
            AsyncMock(side_effect=ChatAuthenticationError("sentinel fallback auth", provider="openai")),
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "invalid_provider_credentials"
    assert "sentinel" not in response.text.lower()
    assert provider_call.call_count == 1
    assert provider_manager.get_available_provider.call_count == 1


@pytest.mark.parametrize("partial_output", [False, True])
def test_execution_fallback_lazy_stream_auth_uses_active_oauth_state(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    partial_output,
):
    runtime_type = _credential_runtime_double(auth_sources={"anthropic": None, "openai": "oauth"})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"anthropic": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    usage_before_failure = []

    async def fallback_stream():
        if partial_output:
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            usage_before_failure.append(list(runtime_type.instances[0].marked_used))
        auth_error = ChatAuthenticationError("sentinel fallback stream auth", provider="openai")
        raise auth_error if partial_output else _certified_pre_dispatch(auth_error)

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=[
                _certified_pre_dispatch(
                    ChatProviderError(provider="anthropic", message="initial failed", status_code=502)
                ),
                fallback_stream(),
                refreshed_stream(),
            ],
        ) as provider_call,
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert "sentinel" not in response.text.lower()
    if partial_output:
        assert "partial" in response.text
        assert "provider_authentication_failed" in response.text
        assert "recovered" not in response.text
        assert provider_call.call_count == 2
        assert ("openai", True) not in runtime_type.instances[0].resolve_calls
        assert runtime_type.instances[0].marked_used == ["openai"]
        assert usage_before_failure == [["openai"]]
    else:
        assert "recovered" in response.text
        assert provider_call.call_count == 3
        assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1
        assert runtime_type.instances[0].marked_used == ["openai"]


def test_execution_fallback_stream_refresh_failure_is_terminal_and_sanitized(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(
        auth_sources={"anthropic": None, "openai": "oauth"},
        refresh_errors={"openai": ByokResolutionError("credential_store_unavailable", "openai")},
    )
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"anthropic": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def fallback_stream():
        if False:
            yield ""
        raise _certified_pre_dispatch(
            ChatAuthenticationError("sentinel fallback stream auth", provider="openai")
        )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=[
                _certified_pre_dispatch(
                    ChatProviderError(provider="anthropic", message="initial failed", status_code=502)
                ),
                fallback_stream(),
            ],
        ) as provider_call,
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "credential_store_unavailable"
    assert "sentinel" not in response.text.lower()
    assert provider_call.call_count == 2


def test_execution_fallback_lazy_stream_config_is_terminal_and_sanitized(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(auth_sources={"anthropic": None, "openai": "oauth"})
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"anthropic": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def config_error_stream():
        if False:
            yield ""
        raise ChatConfigurationError("sentinel fallback stream config", provider="openai")

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=[
                _certified_pre_dispatch(
                    ChatProviderError(provider="anthropic", message="initial failed", status_code=502)
                ),
                config_error_stream(),
            ],
        ) as provider_call,
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "provider_configuration_invalid"
    assert "sentinel" not in response.text.lower()
    assert provider_call.call_count == 2
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls


def test_streaming_fallback_excludes_every_attempted_provider_and_terminates(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "anthropic": SimpleNamespace(can_attempt_call=lambda: True)
    }
    provider_manager.get_available_provider.side_effect = ["openai", None]
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def failed_stream(provider_name: str):
        if False:
            yield ""
        raise _certified_pre_dispatch(
            ChatProviderError(
                provider=provider_name,
                message=f"{provider_name}-failure-secret-/srv/provider",
                status_code=502,
            )
        )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "_should_enforce_strict_model_selection", return_value=False),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=[failed_stream("anthropic"), failed_stream("openai")],
        ) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert provider_call.call_count == 2
    assert provider_manager.get_available_provider.call_args_list == [
        call(exclude=["anthropic"]),
        call(exclude=["anthropic", "openai"]),
    ]
    assert "failure-secret" not in response.text


def test_queued_lazy_stream_fallback_uses_unique_attempt_ids_while_prior_job_is_active(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    """A lazy queued failure must not collide with its still-active retry ID."""

    class DelayedCleanupQueue:
        def __init__(self) -> None:
            self._running = True
            self.active_ids: set[str] = set()
            self.request_ids: list[str] = []

        def is_running(self) -> bool:
            return True

        async def enqueue(
            self,
            *,
            request_id,
            processor,
            stream_channel,
            **_kwargs,
        ):
            if request_id in self.active_ids:
                raise ValueError(f"Duplicate request ID: {request_id}")
            self.active_ids.add(request_id)
            self.request_ids.append(request_id)
            future = asyncio.get_running_loop().create_future()

            async def pump() -> None:
                try:
                    stream = processor()
                    if hasattr(stream, "__aiter__"):
                        async for chunk in stream:
                            await stream_channel.put(chunk)
                    else:
                        for chunk in stream:
                            await stream_channel.put(chunk)
                except Exception as exc:  # noqa: BLE001 - fake adapter captures all failures
                    payload = chat_service.provider_stream_error_payload(exc)
                    await stream_channel.put(f"data: {json.dumps(payload)}\n\n")
                finally:
                    await stream_channel.put(None)
                    if not future.done():
                        future.set_result({"status": "stream_completed"})
                    try:
                        await asyncio.sleep(0.5)
                    finally:
                        self.active_ids.discard(request_id)

            asyncio.create_task(pump())
            return future

    runtime_type = _credential_runtime_double()
    queue = DelayedCleanupQueue()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "anthropic": SimpleNamespace(can_attempt_call=lambda: True)
    }
    provider_manager.get_available_provider.return_value = "openai"
    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def failed_stream():
        if False:
            yield ""
        raise _certified_pre_dispatch(
            ChatProviderError(
                provider="anthropic",
                message="queued-lazy-secret-/srv/provider",
                status_code=502,
            )
        )

    def healthy_stream():
        yield 'data: {"choices":[{"delta":{"content":"queued recovered"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "_should_enforce_strict_model_selection", return_value=False),
        patch.object(chat_endpoint, "get_request_queue", return_value=queue),
        patch.object(chat_service, "get_request_queue", return_value=queue),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", True),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=[failed_stream(), healthy_stream()],
        ) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert "queued recovered" in response.text
    assert "queued-lazy-secret" not in response.text
    assert provider_call.call_count == 2
    assert len(queue.request_ids) == len(set(queue.request_ids)) == 2
    assert queue.request_ids[0].endswith(":provider-stream:1")
    assert queue.request_ids[1].endswith(":provider-stream:2")


def test_streaming_prompt_guardrail_preserves_terminal_sse_contract_at_endpoint_boundary(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import (
        PromptCostGuardrailConfig,
    )

    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="blocked secret prompt")],
        stream=True,
    )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_service,
            "load_prompt_cost_guardrail_config",
            return_value=PromptCostGuardrailConfig(
                enabled=True,
                block_total_estimated_tokens=1,
            ),
        ),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    error_frames = _response_error_frames(response.text)
    assert len(error_frames) == 1
    assert error_frames[0]["error"] == {
        "code": "prompt_cost_guardrail_block",
        "type": "prompt_cost_guardrail_block",
        "message": "Prompt cost guardrail blocked request before provider dispatch.",
    }
    assert response.text.rstrip().endswith("data: [DONE]")
    provider_call.assert_not_called()
    assert runtime_type.instances[0].close_calls == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "expected_code",
    [
        "invalid_provider_credentials",
        "missing_provider_credentials",
        "credential_store_unavailable",
        "credential_scope_revoked",
        "provider_configuration_invalid",
        "provider_disabled",
        "model_not_allowed",
    ],
)
def test_oauth_refresh_failure_preserves_terminal_credential_taxonomy(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    streaming,
    expected_code,
):
    refresh_errors: dict[str, BaseException] = {}
    refresh_api_keys: dict[str, str | None] = {}
    if expected_code == "missing_provider_credentials":
        refresh_api_keys["openai"] = None
    elif expected_code == "provider_configuration_invalid":
        refresh_errors["openai"] = ChatConfigurationError(
            "sentinel refresh config",
            provider="openai",
        )
    elif expected_code in {"provider_disabled", "model_not_allowed"}:
        refresh_errors["openai"] = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": expected_code},
        )
    else:
        refresh_errors["openai"] = ByokResolutionError(
            expected_code,
            "openai",
        )

    runtime_type = _credential_runtime_double(
        auth_source="oauth",
        refresh_errors=refresh_errors,
        refresh_api_keys=refresh_api_keys,
    )
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "anthropic"
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=streaming,
    )

    if streaming:

        def expired_stream():
            if False:
                yield ""
            raise _certified_pre_dispatch(
                ChatAuthenticationError("sentinel initial auth", provider="openai")
            )

        provider_side_effect = [expired_stream()]
    else:
        provider_side_effect = [ChatAuthenticationError("sentinel initial auth", provider="openai")]

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=provider_side_effect,
        ) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    expected_status = (
        status.HTTP_403_FORBIDDEN
        if expected_code in {"provider_disabled", "model_not_allowed"}
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert detail["error_code"] == expected_code
    assert "sentinel" not in str(detail).lower()
    assert provider_call.call_count == 1
    provider_manager.get_available_provider.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_terminal_provider_policy_codes_preserve_taxonomy() -> None:
    """Concurrent terminal policy failures cannot collapse or exchange codes."""

    started = {
        "provider_disabled": asyncio.Event(),
        "model_not_allowed": asyncio.Event(),
    }
    release = asyncio.Event()

    async def _map_policy_failure(code: str) -> tuple[int, dict[str, str]]:
        started[code].set()
        await release.wait()
        mapped = chat_endpoint._provider_credential_http_exception(
            ProviderCredentialTerminalError(code)
        )
        return mapped.status_code, mapped.detail

    tasks = [
        asyncio.create_task(_map_policy_failure(code))
        for code in started
    ]
    await asyncio.gather(*(event.wait() for event in started.values()))
    release.set()
    results = await asyncio.gather(*tasks)

    assert results == [
        (
            status.HTTP_403_FORBIDDEN,
            {
                "error_code": "provider_disabled",
                "message": chat_endpoint._PROVIDER_CREDENTIAL_MESSAGES[
                    "provider_disabled"
                ],
            },
        ),
        (
            status.HTTP_403_FORBIDDEN,
            {
                "error_code": "model_not_allowed",
                "message": chat_endpoint._PROVIDER_CREDENTIAL_MESSAGES[
                    "model_not_allowed"
                ],
            },
        ),
    ]


def test_streaming_auth_failure_after_output_is_sanitized_without_refresh(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    usage_before_failure = []

    async def partial_stream():
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        usage_before_failure.append(list(runtime_type.instances[0].marked_used))
        raise ChatAuthenticationError("sentinel post-output body", provider="openai")

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", return_value=partial_stream()),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert "partial" in response.text
    assert "provider_authentication_failed" in response.text
    assert "sentinel" not in response.text.lower()
    data_frames = []
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line.removeprefix("data: "))
        if isinstance(payload, dict) and "error" in payload:
            data_frames.append(payload)
    assert len(data_frames) == 1
    assert response.text.index("partial") < response.text.index('"error"')
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert runtime_type.instances[0].marked_used == ["openai"]
    assert usage_before_failure == [["openai"]]


@pytest.mark.parametrize("error_type", [ChatProviderError, ChatAPIError])
def test_streaming_provider_failure_after_output_is_sanitized_without_fallback(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    error_type,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "anthropic"
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    usage_before_failure = []

    async def partial_stream():
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        usage_before_failure.append(list(runtime_type.instances[0].marked_used))
        raise error_type(
            "sentinel upstream outage body",
            status_code=502,
            provider="openai",
        )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.dict(os.environ, {"MODERATION_STREAM_BUFFER_CHARS": "0"}),
        patch.object(chat_endpoint, "perform_chat_api_call", return_value=partial_stream()) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    error_frames = []
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line.removeprefix("data: "))
        if isinstance(payload, dict) and "error" in payload:
            error_frames.append(payload)

    assert response.status_code == status.HTTP_200_OK
    assert "partial" in response.text
    assert len(error_frames) == 1
    assert error_frames[0]["error"] == {
        "code": "provider_unavailable",
        "type": "provider_unavailable",
        "message": "The chat service provider is currently unavailable.",
    }
    assert "provider_configuration_invalid" not in response.text
    assert "sentinel upstream outage body" not in response.text
    assert provider_call.call_count == 1
    provider_manager.get_available_provider.assert_not_called()
    assert runtime_type.instances[0].marked_used == ["openai"]
    assert usage_before_failure == [["openai"]]


@pytest.mark.parametrize(
    "provider_chunks",
    [
        ("data: [DONE]\n\n",),
        (),
    ],
    ids=["empty-done", "empty-eof"],
)
def test_streaming_empty_terminal_is_bounded_without_usage_mark(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    provider_chunks,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )

    def empty_stream():
        yield from provider_chunks

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", return_value=empty_stream()),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == {
        "error_code": "provider_unavailable",
        "message": chat_endpoint._PROVIDER_CREDENTIAL_MESSAGES["provider_unavailable"],
    }
    assert runtime_type.instances[0].mark_calls == []
    assert runtime_type.instances[0].marked_used == []
    assert runtime_type.instances[0].close_calls == 1


@pytest.mark.parametrize(
    ("provider_chunks", "expected_text"),
    [
        (
            (
                'data: {"choices":[{"delta":{"refusal":"I cannot comply."}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ),
            "I cannot comply.",
        ),
        (
            (
                'data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}\n\n',
                "data: [DONE]\n\n",
            ),
            "content_filter",
        ),
        (("plain provider text",), "plain provider text"),
    ],
    ids=["refusal", "content-filter", "plain-text"],
)
def test_streaming_semantic_output_is_not_rejected_as_empty(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    provider_chunks,
    expected_text,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            return_value=iter(provider_chunks),
        ),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=ChatCompletionRequest(
                model="gpt-4o-mini",
                api_provider="openai",
                messages=[
                    ChatCompletionUserMessageParam(role="user", content="Hello")
                ],
                stream=True,
            ).model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert expected_text in response.text
    assert _response_error_frames(response.text) == []
    assert runtime_type.instances[0].mark_calls == ["openai"]
    assert runtime_type.instances[0].marked_used == ["openai"]
    assert runtime_type.instances[0].close_calls == 1


def test_concurrent_empty_and_valid_streams_keep_usage_accounting_isolated(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True),
        "anthropic": SimpleNamespace(can_attempt_call=lambda: True),
    }
    adapter_barrier = threading.Barrier(2)

    def provider_call(**kwargs):
        provider = kwargs["api_endpoint"]
        adapter_barrier.wait(timeout=2.0)
        if provider == "openai":
            return iter(())
        assert provider == "anthropic"
        return iter(
            [
                'data: {"choices":[{"delta":{"content":"concurrent valid"}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )

    def post(provider: str):
        return authenticated_client.post(
            "/api/v1/chat/completions",
            json=ChatCompletionRequest(
                model="test-model",
                api_provider=provider,
                messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
                stream=True,
            ).model_dump(),
        )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=provider_call),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            empty_future = executor.submit(post, "openai")
            valid_future = executor.submit(post, "anthropic")
            empty_response = empty_future.result(timeout=5.0)
            valid_response = valid_future.result(timeout=5.0)

    assert empty_response.status_code == status.HTTP_502_BAD_GATEWAY
    assert empty_response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "concurrent valid" not in empty_response.text
    assert valid_response.status_code == status.HTTP_200_OK
    assert "concurrent valid" in valid_response.text
    assert _response_error_frames(valid_response.text) == []

    assert len(runtime_type.instances) == 2
    runtimes_by_provider = {
        runtime.resolve_calls[0][0]: runtime for runtime in runtime_type.instances
    }
    assert runtimes_by_provider["openai"].mark_calls == []
    assert runtimes_by_provider["openai"].marked_used == []
    assert runtimes_by_provider["anthropic"].mark_calls == ["anthropic"]
    assert runtimes_by_provider["anthropic"].marked_used == ["anthropic"]
    assert all(runtime.close_calls == 1 for runtime in runtime_type.instances)


def test_chat_completion_with_conversation_history(authenticated_client, mock_chacha_db, setup_dependencies):
    """Test chat with conversation history."""

    # Get the actual default character ID (it's usually 2 based on our tests)
    # First check what character exists
    default_char = mock_chacha_db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
    char_id = default_char["id"] if default_char else 2

    # Create a conversation with the correct client_id
    # The mock_chacha_db has a client_id attribute
    conv_id = mock_chacha_db.add_conversation(
        {
            "character_id": char_id,
            "title": "Test Conversation",
            "client_id": mock_chacha_db.client_id,  # Use the database's client_id
        }
    )

    # Add some history
    mock_chacha_db.add_message({"conversation_id": conv_id, "sender": "user", "content": "Previous message"})
    mock_chacha_db.add_message({"conversation_id": conv_id, "sender": "assistant", "content": "Previous response"})

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Continue our conversation")],
        conversation_id=str(conv_id),
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": "Continuing from before..."}, "finish_reason": "stop"}
            ],
        }

        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

        assert response.status_code == status.HTTP_200_OK

        # Verify history was included in the call
        call_args = mock_llm.call_args
        # Messages might be in kwargs
        if call_args.kwargs and "messages_payload" in call_args.kwargs:
            messages = call_args.kwargs["messages_payload"]
        elif call_args.kwargs and "messages" in call_args.kwargs:
            messages = call_args.kwargs["messages"]
        elif len(call_args.args) > 0:
            # Try to find messages in positional args
            messages = call_args.args[0] if isinstance(call_args.args[0], list) else []
        else:
            messages = []
        assert len(messages) > 1  # Should include history


def test_chat_completion_rate_limiting(authenticated_client, mock_chacha_db, setup_dependencies):
    """Test rate limiting functionality."""

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Response"}, "finish_reason": "stop"}],
        }

        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())
            responses.append(response.status_code)

        # All should succeed (rate limiting might not be enabled in test)
        # or we should see 429 status codes
        assert all(s in [200, 429] for s in responses)


def test_chat_completion_rg_primary_deny(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    monkeypatch,
):
    """
    When ResourceGovernor is enabled and the RG gate denies,
    the chat endpoint should surface a 429 with a policy-aware message.
    """

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello under RG")],
    )

    monkeypatch.setenv("RG_ENABLED", "1")

    gov = getattr(app.state, "rg_governor", None)
    if gov is None:
        app.state.rg_governor = SimpleNamespace()
        gov = app.state.rg_governor

    if getattr(app.state, "rg_policy_loader", None) is None:

        class _Loader:
            def get_policy(self, _policy_id):
                return {}

        app.state.rg_policy_loader = _Loader()

    async def fake_reserve(req, op_id=None):
        categories = getattr(req, "categories", {}) or {}
        if "requests" in categories and "tokens" not in categories:
            return SimpleNamespace(allowed=True, retry_after=None, details={"categories": categories}), "req-handle"
        return SimpleNamespace(allowed=False, retry_after=1, details={"categories": categories}), None

    async def fake_commit(*_args, **_kwargs):
        return None

    monkeypatch.setattr(gov, "reserve", fake_reserve, raising=True)
    monkeypatch.setattr(gov, "commit", fake_commit, raising=False)

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS",
            {"openai": "test-key"},
        ),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }

        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        body = response.json()
        assert "ResourceGovernor policy=" in str(body.get("detail"))


def test_chat_completion_rg_shadow_vs_primary_behaviour(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    monkeypatch,
):
    """
    Exercise /api/v1/chat/completions under RG allow/deny decisions to
    validate 200/429 behavior.
    """

    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello RG shadow/primary")],
    )

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("RG_ENABLED", "1")

    gov = getattr(app.state, "rg_governor", None)
    if gov is None:
        app.state.rg_governor = SimpleNamespace()
        gov = app.state.rg_governor

    if getattr(app.state, "rg_policy_loader", None) is None:

        class _Loader:
            def get_policy(self, _policy_id):
                return {}

        app.state.rg_policy_loader = _Loader()

    token_calls = {"count": 0}

    async def fake_reserve(req, op_id=None):
        categories = getattr(req, "categories", {}) or {}
        if "requests" in categories and "tokens" not in categories:
            return SimpleNamespace(allowed=True, retry_after=None, details={"categories": categories}), "req-handle"
        if "tokens" in categories:
            token_calls["count"] += 1
            if token_calls["count"] == 1:
                return SimpleNamespace(allowed=True, retry_after=None, details={"categories": categories}), "tok-handle"
            return SimpleNamespace(allowed=False, retry_after=3, details={"categories": categories}), None
        return SimpleNamespace(allowed=True, retry_after=None, details={"categories": categories}), "req-handle"

    async def fake_commit(*_args, **_kwargs):
        return None

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm,
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS",
            {"openai": "test-key"},
        ),
    ):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Shadow/primary response"},
                    "finish_reason": "stop",
                }
            ],
        }

        monkeypatch.setattr(gov, "reserve", fake_reserve, raising=True)
        monkeypatch.setattr(gov, "commit", fake_commit, raising=False)

        shadow_resp = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )
        assert shadow_resp.status_code == status.HTTP_200_OK

        primary_resp = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )
        assert primary_resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        body = primary_resp.json()
        detail = str(body.get("detail"))
        assert "ResourceGovernor policy=" in detail
        # Retry-After may be present either in headers or encoded in detail;
        # assert that at least one representation of the retry interval exists.
        header_retry = primary_resp.headers.get("Retry-After")
        assert ("retry_after=3s" in detail) or (header_retry is not None)


_PROVIDER_STREAM_SENTINEL = "sk-provider-secret /private/upstream.log https://provider.invalid/raw"


def _response_error_frames(response_text: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for line in response_text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            payload = json.loads(line.removeprefix("data: "))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "error" in payload:
            frames.append(payload)
    return frames


@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize("failure_mode", ["inband", "raised"])
@pytest.mark.parametrize("known", [True, False])
def test_streaming_provider_failure_is_bounded_at_http_boundary(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    stream_kind,
    failure_mode,
    known,
):
    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "anthropic"
    expected_code = "provider_authentication_failed" if known else "provider_unavailable"
    closed = False
    if failure_mode == "inband":
        code = "provider_authentication_failed" if known else "private_provider_error"
        frame: object = f'data: {json.dumps({"error": {"code": code, "message": _PROVIDER_STREAM_SENTINEL}})}\n\n'
        if stream_kind == "sync":
            frame = str(frame).encode()
        error = None
    else:
        frame = None
        error = (
            ChatAuthenticationError(_PROVIDER_STREAM_SENTINEL, provider="openai")
            if known
            else RuntimeError(_PROVIDER_STREAM_SENTINEL)
        )

    if stream_kind == "async":

        async def source():
            nonlocal closed
            try:
                if failure_mode == "inband":
                    yield frame
                else:
                    if False:
                        yield ""
                    raise error
            finally:
                closed = True

    else:

        def source():
            nonlocal closed
            try:
                if failure_mode == "inband":
                    yield frame
                else:
                    if False:
                        yield ""
                    raise error
            finally:
                closed = True

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
            patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
            patch.object(chat_endpoint, "get_request_queue", return_value=None),
            patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
            patch.dict(os.environ, {"MODERATION_STREAM_BUFFER_CHARS": "0"}),
            patch.object(chat_endpoint, "perform_chat_api_call", return_value=source()) as provider_call,
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == {
        "error_code": expected_code,
        "message": chat_endpoint._PROVIDER_CREDENTIAL_MESSAGES[expected_code],
    }
    assert provider_call.call_count == 1
    provider_manager.get_available_provider.assert_not_called()
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert runtime_type.instances[0].close_calls == 1
    assert closed is True
    assert _PROVIDER_STREAM_SENTINEL not in response.text
    assert _PROVIDER_STREAM_SENTINEL not in "".join(logs)


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning_content": "thinking"},
        {"reasoning": "thinking"},
        {"reasoning_details": {"text": "thinking"}},
        {"thinking": "thinking"},
        {"analysis": "thinking"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "notes.search", "arguments": "{}"},
                }
            ]
        },
        {"function_call": {"name": "notes.search", "arguments": "{}"}},
    ],
    ids=["reasoning_content", "reasoning", "reasoning_details", "thinking", "analysis", "tool", "function"],
)
def test_streaming_semantic_output_blocks_certified_replay_and_closes_once(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    delta,
):
    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "anthropic"
    closed = False

    async def source():
        nonlocal closed
        try:
            yield f'data: {json.dumps({"choices": [{"delta": delta}]})}\n\n'
            raise _certified_pre_dispatch(
                ChatProviderError(
                    _PROVIDER_STREAM_SENTINEL,
                    status_code=502,
                    provider="openai",
                )
            )
        finally:
            closed = True

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with (
            patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
            patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
            patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
            patch.object(chat_endpoint, "get_request_queue", return_value=None),
            patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
            patch.dict(os.environ, {"MODERATION_STREAM_BUFFER_CHARS": "0"}),
            patch.object(chat_endpoint, "perform_chat_api_call", return_value=source()) as provider_call,
        ):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump(),
            )
    finally:
        logger.remove(sink_id)

    frames = _response_error_frames(response.text)
    assert response.status_code == status.HTTP_200_OK
    assert len(frames) == 1
    assert frames[0]["error"] == {
        "code": "provider_unavailable",
        "type": "provider_unavailable",
        "message": "The chat service provider is currently unavailable.",
    }
    assert provider_call.call_count == 1
    provider_manager.get_available_provider.assert_not_called()
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert runtime_type.instances[0].marked_used == ["openai"]
    assert closed is True
    assert _PROVIDER_STREAM_SENTINEL not in response.text
    assert _PROVIDER_STREAM_SENTINEL not in "".join(logs)


@pytest.mark.parametrize(
    "error",
    [
        _with_replay_flags(
            ChatProviderError(_PROVIDER_STREAM_SENTINEL, status_code=502, provider="openai"),
            upstream_dispatched=0,
            allow_non_stream_fallback=True,
        ),
        _with_replay_flags(
            ChatProviderError(_PROVIDER_STREAM_SENTINEL, status_code=502, provider="openai"),
            upstream_dispatched=False,
            allow_non_stream_fallback=1,
        ),
        _with_replay_flags(
            ChatProviderError(_PROVIDER_STREAM_SENTINEL, status_code=502, provider="openai"),
            upstream_dispatched=False,
            allow_non_stream_fallback=True,
            output_emitted=True,
        ),
        _with_replay_flags(
            RuntimeError(_PROVIDER_STREAM_SENTINEL),
            upstream_dispatched=False,
            allow_non_stream_fallback=True,
        ),
    ],
    ids=["integer-dispatch", "integer-permission", "output-emitted", "untrusted-type"],
)
def test_streaming_replay_requires_literal_trusted_three_flag_certificate(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    error,
):
    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    provider_manager.get_available_provider.return_value = "anthropic"

    def source():
        if False:
            yield ""
        raise error

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", return_value=source()) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert provider_call.call_count == 1
    provider_manager.get_available_provider.assert_not_called()
    assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert _PROVIDER_STREAM_SENTINEL not in response.text


def _primed_stream_response(stream, error_state):
    wrapped = chat_endpoint._sanitize_provider_stream_call(lambda: stream, error_state)()
    handler = StreamingResponseHandler("conv", "model", heartbeat_interval=0)
    return StreamingResponse(
        handler.safe_stream_generator(wrapped),
        media_type="text/event-stream",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["elapsed", "bytes", "chunks"])
async def test_stream_prime_bounds_endless_metadata_and_closes(monkeypatch, budget):
    limits = {
        "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS": 0.5,
        "PROVIDER_STREAM_PRIME_MAX_BUFFERED_BYTES": 100_000,
        "PROVIDER_STREAM_PRIME_MAX_BUFFERED_CHUNKS": 100_000,
    }
    if budget == "elapsed":
        limits["PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS"] = 0.01
    elif budget == "bytes":
        limits["PROVIDER_STREAM_PRIME_MAX_BUFFERED_BYTES"] = 3
    else:
        limits["PROVIDER_STREAM_PRIME_MAX_BUFFERED_CHUNKS"] = 2
    for name, value in limits.items():
        monkeypatch.setattr(chat_endpoint, name, value, raising=False)

    closed = asyncio.Event()

    async def endless_metadata():
        try:
            while True:
                yield ": upstream metadata\n\n"
        finally:
            closed.set()

    state: dict[str, object] = {}
    response = _primed_stream_response(endless_metadata(), state)
    primed = await asyncio.wait_for(
        chat_endpoint._prime_provider_stream_response(response, state),
        timeout=1.0,
    )

    assert primed[1:] == ("provider_unavailable", False, False)
    assert state.get("replay_certified") is not True
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stream_prime_allows_finite_metadata_then_output(monkeypatch):
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_BUFFERED_BYTES", 10_000, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_BUFFERED_CHUNKS", 4, raising=False)
    closed = asyncio.Event()

    async def finite_stream():
        try:
            yield ": metadata one\n\n"
            yield "event: metadata\n\n"
            yield 'data: {"choices":[{"delta":{"content":"ready"}}]}\n\n'
        finally:
            closed.set()

    state: dict[str, object] = {}
    response = _primed_stream_response(finite_stream(), state)
    primed_chunks, code, has_output, complete = await asyncio.wait_for(
        chat_endpoint._prime_provider_stream_response(response, state),
        timeout=1.0,
    )

    assert primed_chunks
    assert code is None
    assert has_output is True
    assert complete is False
    await chat_endpoint._close_provider_stream_response(response)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stream_prime_budget_stops_after_first_output(monkeypatch):
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_BUFFERED_BYTES", 10_000, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_BUFFERED_CHUNKS", 1, raising=False)
    closed = asyncio.Event()

    async def long_post_prime_stream():
        try:
            yield 'data: {"choices":[{"delta":{"content":"ready"}}]}\n\n'
            await asyncio.sleep(0.02)
            for index in range(3):
                yield f'data: {{"choices":[{{"delta":{{"content":"after-{index}"}}}}]}}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"
        finally:
            closed.set()

    state: dict[str, object] = {}
    response = _primed_stream_response(long_post_prime_stream(), state)
    primed_chunks, code, has_output, complete = await asyncio.wait_for(
        chat_endpoint._prime_provider_stream_response(response, state),
        timeout=1.0,
    )
    remaining_chunks = [chunk async for chunk in response.body_iterator]
    complete_stream = "".join(str(chunk) for chunk in (*primed_chunks, *remaining_chunks))

    assert (code, has_output, complete) == (None, True, False)
    assert "after-0" in complete_stream
    assert "after-1" in complete_stream
    assert "after-2" in complete_stream
    assert "provider_unavailable" not in complete_stream
    assert state.get("prime_buffered_chunks") == 1
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stream_prime_hard_bounds_noncooperative_next_and_close(monkeypatch, caplog):
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_CLEANUP_TIMEOUT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_TASK_CANCEL_DRAIN_SECONDS", 0.005, raising=False)
    release = asyncio.Event()
    next_started = asyncio.Event()
    next_finished = asyncio.Event()
    close_started = asyncio.Event()
    close_finished = asyncio.Event()

    class NonCooperativeIterator:
        def __init__(self):
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            next_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                raise StopAsyncIteration from None
            finally:
                next_finished.set()

        async def aclose(self):
            self.close_calls += 1
            close_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            finally:
                close_finished.set()

    iterator = NonCooperativeIterator()
    response = SimpleNamespace(body_iterator=iterator)
    state: dict[str, object] = {}
    prime_task = asyncio.create_task(
        chat_endpoint._prime_provider_stream_response(response, state)
    )
    await asyncio.wait_for(next_started.wait(), timeout=1.0)
    result = await asyncio.wait_for(prime_task, timeout=1.0)

    assert result[1:] == ("provider_unavailable", False, False)
    assert not release.is_set()
    assert next_started.is_set()
    assert close_started.is_set()
    assert iterator.close_calls == 1
    assert state.get("prime_active") is False

    release.set()
    await asyncio.wait_for(next_finished.wait(), timeout=1.0)
    await asyncio.wait_for(close_finished.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert "task was destroyed" not in caplog.text.lower()
    assert "exception was never retrieved" not in caplog.text.lower()


@pytest.mark.asyncio
async def test_stream_prime_hard_bounds_noncooperative_sync_next_and_close(monkeypatch, caplog):
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_CLEANUP_TIMEOUT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_TASK_CANCEL_DRAIN_SECONDS", 0.005, raising=False)
    monkeypatch.setattr(streaming_utils, "STREAMING_SYNC_BRIDGE_ENABLED", True, raising=False)
    next_started = threading.Event()
    release_next = threading.Event()
    close_started = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()

    class NonCooperativeSyncIterator:
        def __init__(self):
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            next_started.set()
            release_next.wait()
            return ": late metadata\n\n"

        def close(self):
            self.close_calls += 1
            close_started.set()
            release_close.wait()
            close_finished.set()

    iterator = NonCooperativeSyncIterator()
    state: dict[str, object] = {}
    response = _primed_stream_response(iterator, state)
    try:
        prime_task = asyncio.create_task(
            chat_endpoint._prime_provider_stream_response(response, state)
        )
        assert await asyncio.to_thread(next_started.wait, 1.0)
        result = await asyncio.wait_for(prime_task, timeout=1.0)

        assert result[1:] == ("provider_unavailable", False, False)
        assert not release_next.is_set()
        assert next_started.is_set()
        assert state.get("prime_active") is False
        assert iterator.close_calls == 0

        release_next.set()
        assert await asyncio.to_thread(close_started.wait, 1.0)
        assert iterator.close_calls == 1
    finally:
        release_next.set()
        release_close.set()

    assert await asyncio.to_thread(close_finished.wait, 1.0)
    await asyncio.sleep(0)
    assert "task was destroyed" not in caplog.text.lower()
    assert "exception was never retrieved" not in caplog.text.lower()


@pytest.mark.asyncio
async def test_stream_prime_hard_bounds_ignore_disabled_sync_bridge_and_keep_event_loop_responsive(
    monkeypatch,
    caplog,
):
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_CLEANUP_TIMEOUT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_TASK_CANCEL_DRAIN_SECONDS", 0.005, raising=False)
    monkeypatch.setattr(streaming_utils, "STREAMING_SYNC_BRIDGE_ENABLED", False, raising=False)
    next_started = threading.Event()
    release_next = threading.Event()
    close_finished = threading.Event()
    loop_responsive = asyncio.Event()

    class BlockingSyncIterator:
        def __init__(self):
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            next_started.set()
            release_next.wait()
            raise StopIteration

        def close(self):
            self.close_calls += 1
            close_finished.set()

    iterator = BlockingSyncIterator()
    state: dict[str, object] = {}
    response = _primed_stream_response(iterator, state)
    prime_task = asyncio.create_task(
        chat_endpoint._prime_provider_stream_response(response, state)
    )
    try:
        assert await asyncio.to_thread(next_started.wait, 1.0)
        asyncio.get_running_loop().call_soon(loop_responsive.set)
        await asyncio.wait_for(loop_responsive.wait(), timeout=1.0)
        result = await asyncio.wait_for(prime_task, timeout=1.0)
        assert not release_next.is_set()
    finally:
        release_next.set()
        if not prime_task.done():
            await asyncio.wait_for(prime_task, timeout=1.0)

    assert result[1:] == ("provider_unavailable", False, False)
    assert next_started.is_set()
    assert state.get("prime_active") is False
    assert await asyncio.to_thread(close_finished.wait, 1.0)
    assert iterator.close_calls == 1
    await asyncio.sleep(0)
    assert "task was destroyed" not in caplog.text.lower()
    assert "exception was never retrieved" not in caplog.text.lower()


@pytest.mark.asyncio
async def test_concurrent_stream_errors_keep_codes_replay_and_cleanup_isolated():
    known_sentinel = "known-secret-/srv/known"
    unknown_sentinel = "unknown-secret-https://unknown.invalid"
    ready_known = asyncio.Event()
    ready_unknown = asyncio.Event()
    release = asyncio.Event()

    class GatedErrorStream:
        def __init__(self, error, ready):
            self.error = error
            self.ready = ready
            self.close_calls = 0
            self.attempted = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.attempted:
                raise StopAsyncIteration
            self.attempted = True
            self.ready.set()
            await release.wait()
            raise self.error

        async def aclose(self):
            self.close_calls += 1

    known_stream = GatedErrorStream(
        ChatAuthenticationError(known_sentinel, provider="openai"),
        ready_known,
    )
    unknown_stream = GatedErrorStream(
        UnknownAdapterFailure(unknown_sentinel),
        ready_unknown,
    )
    known_state: dict[str, object] = {}
    unknown_state: dict[str, object] = {}
    known_response = _primed_stream_response(known_stream, known_state)
    unknown_response = _primed_stream_response(unknown_stream, unknown_state)
    known_task = asyncio.create_task(
        chat_endpoint._prime_provider_stream_response(known_response, known_state)
    )
    unknown_task = asyncio.create_task(
        chat_endpoint._prime_provider_stream_response(unknown_response, unknown_state)
    )
    await asyncio.wait_for(asyncio.gather(ready_known.wait(), ready_unknown.wait()), timeout=1.0)
    release.set()
    known_result, unknown_result = await asyncio.wait_for(
        asyncio.gather(known_task, unknown_task),
        timeout=1.0,
    )
    await asyncio.gather(
        chat_endpoint._close_provider_stream_response(known_response),
        chat_endpoint._close_provider_stream_response(unknown_response),
    )

    assert known_result[1] == "provider_authentication_failed"
    assert unknown_result[1] == "provider_unavailable"
    assert known_state.get("code") == "provider_authentication_failed"
    assert unknown_state.get("code") == "provider_unavailable"
    assert known_state.get("replay_certified") is False
    assert unknown_state.get("replay_certified") is False
    known_wire = "".join(str(chunk) for chunk in known_result[0])
    unknown_wire = "".join(str(chunk) for chunk in unknown_result[0])
    assert known_sentinel not in known_wire + unknown_wire
    assert unknown_sentinel not in known_wire + unknown_wire
    assert "provider_unavailable" not in known_wire
    assert "provider_authentication_failed" not in unknown_wire
    assert known_stream.close_calls == 1
    assert unknown_stream.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plain_payload", "expected_text"),
    [
        ("hello", "hello"),
        ('"hello"', "hello"),
        ("provider_unavailable", "provider_unavailable"),
        ('"provider_unavailable"', "provider_unavailable"),
        ("Error: assistant-authored content", "assistant-authored content"),
        ('"Error: assistant-authored JSON scalar"', "assistant-authored JSON scalar"),
    ],
    ids=[
        "plain",
        "json-scalar",
        "allowlisted-code-plain",
        "allowlisted-code-json-scalar",
        "error-prefix-plain",
        "error-prefix-json-scalar",
    ],
)
async def test_plain_sse_data_is_output_after_structured_metadata(
    monkeypatch,
    plain_payload,
    expected_text,
):
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 1.0, raising=False)
    first_output_calls = 0

    async def mark_first_output():
        nonlocal first_output_calls
        first_output_calls += 1

    async def source():
        yield 'data: {"provider_metadata":{"request_id":"req-1"}}\n\n'
        yield f"data: {plain_payload}\n\n"
        raise _certified_pre_dispatch(
            ChatProviderError(
                _PROVIDER_STREAM_SENTINEL,
                status_code=502,
                provider="openai",
            )
        )

    state: dict[str, object] = {}
    wrapped = chat_endpoint._sanitize_provider_stream_call(lambda: source(), state)()
    handler = StreamingResponseHandler("conv", "model", heartbeat_interval=0)
    response = StreamingResponse(
        handler.safe_stream_generator(wrapped, on_first_output=mark_first_output),
        media_type="text/event-stream",
    )
    primed_chunks, code, has_output, complete = await chat_endpoint._prime_provider_stream_response(
        response,
        state,
    )
    remaining_chunks = [chunk async for chunk in response.body_iterator]
    wire = "".join(str(chunk) for chunk in (*primed_chunks, *remaining_chunks))

    assert (code, has_output, complete) == (None, True, False)
    assert first_output_calls == 1
    assert state.get("prime_buffered_chunks") == 2
    assert state.get("replay_certified") is False
    assert expected_text in wire
    assert [frame["error"]["code"] for frame in _response_error_frames(wire)] == [
        "provider_unavailable"
    ]
    assert _PROVIDER_STREAM_SENTINEL not in wire


@pytest.mark.asyncio
async def test_adapter_prime_concurrently_isolates_benign_metadata_and_real_error(
    monkeypatch,
):
    monkeypatch.setattr(
        chat_endpoint,
        "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS",
        1.0,
        raising=False,
    )
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    sentinel = "prime-error-secret-/srv/provider"

    async def healthy_source():
        ready[0].set()
        await release.wait()
        yield (
            'data: {"error":null,"code":"ok",'
            '"choices":[{"delta":{"content":"healthy prime output"}}]}\n\n'
        )

    async def failed_source():
        ready[1].set()
        await release.wait()
        yield (
            'data: {"error":{"code":"provider_authentication_failed",'
            f'"message":"{sentinel}"}}}}\n\n'
        )

    healthy_state: dict[str, object] = {}
    failed_state: dict[str, object] = {}
    healthy_response = _primed_stream_response(healthy_source(), healthy_state)
    failed_response = _primed_stream_response(failed_source(), failed_state)
    healthy_task = asyncio.create_task(
        chat_endpoint._prime_provider_stream_response(healthy_response, healthy_state)
    )
    failed_task = asyncio.create_task(
        chat_endpoint._prime_provider_stream_response(failed_response, failed_state)
    )
    await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
    release.set()
    healthy_result, failed_result = await asyncio.gather(healthy_task, failed_task)
    healthy_wire = "".join(str(chunk) for chunk in healthy_result[0])
    failed_wire = "".join(str(chunk) for chunk in failed_result[0])

    assert healthy_result[1:] == (None, True, False)
    assert "healthy prime output" in healthy_wire
    assert healthy_state.get("code") is None
    assert failed_result[1] == "provider_authentication_failed"
    assert failed_state.get("code") == "provider_authentication_failed"
    assert sentinel not in healthy_wire + failed_wire


@pytest.mark.asyncio
async def test_stream_prime_concurrent_budgets_are_execution_local(monkeypatch):
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_BUFFERED_BYTES", 10_000, raising=False)
    monkeypatch.setattr(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_BUFFERED_CHUNKS", 2, raising=False)
    first_a_frame = asyncio.Event()
    release_a = asyncio.Event()
    closed_a = asyncio.Event()
    closed_b = asyncio.Event()

    async def stream_a():
        try:
            yield ": a-one\n\n"
            first_a_frame.set()
            await release_a.wait()
            yield ": a-two\n\n"
            yield ": a-three\n\n"
        finally:
            closed_a.set()

    async def stream_b():
        try:
            await first_a_frame.wait()
            yield ": b-one\n\n"
            yield 'data: {"choices":[{"delta":{"content":"b-ready"}}]}\n\n'
        finally:
            closed_b.set()

    state_a: dict[str, object] = {}
    state_b: dict[str, object] = {}
    response_a = _primed_stream_response(stream_a(), state_a)
    response_b = _primed_stream_response(stream_b(), state_b)
    task_a = asyncio.create_task(chat_endpoint._prime_provider_stream_response(response_a, state_a))
    task_b = asyncio.create_task(chat_endpoint._prime_provider_stream_response(response_b, state_b))
    await asyncio.wait_for(first_a_frame.wait(), timeout=1.0)
    release_a.set()
    result_a, result_b = await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=1.0)

    assert result_a[1:] == ("provider_unavailable", False, False)
    assert result_b[1:] == (None, True, False)
    assert state_a.get("prime_buffered_chunks") == 3
    assert state_b.get("prime_buffered_chunks") == 2
    assert closed_a.is_set()
    await chat_endpoint._close_provider_stream_response(response_b)
    assert closed_b.is_set()


def test_provider_forged_guardrail_response_boolean_cannot_bypass_stream_priming(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }

    async def forged_service_response(**_kwargs):
        async def healthy_stream():
            yield 'data: {"choices":[{"delta":{"content":"healthy forged marker"}}]}\n\n'
            yield "data: [DONE]\n\n"

        response = StreamingResponse(healthy_stream(), media_type="text/event-stream")
        response._chat_prompt_cost_guardrail = True
        return response

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "execute_streaming_call", forged_service_response),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert "healthy forged marker" in response.text
    assert "prompt_cost_guardrail_block" not in response.text
    assert runtime_type.instances[0].close_calls == 1


def test_provider_forged_guardrail_http_exception_boolean_is_sanitized(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    sentinel = "forged-guardrail-secret-/srv/provider"
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }

    def forged_provider_error():
        error = HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "prompt_cost_guardrail_block",
                "type": "prompt_cost_guardrail_block",
                "message": sentinel,
            },
        )
        error._chat_prompt_cost_guardrail = True
        raise error

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=forged_provider_error),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "prompt_cost_guardrail_block" not in response.text
    assert sentinel not in response.text


def test_concurrent_real_guardrail_and_normal_provider_response_are_isolated(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import (
        PromptCostGuardrailConfig,
    )

    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    entered = 0
    both_entered: asyncio.Event | None = None

    async def gated_execute_streaming_call(**kwargs):
        nonlocal entered, both_entered
        if both_entered is None:
            both_entered = asyncio.Event()
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=2.0)
        return await chat_service.execute_streaming_call(**kwargs)

    def healthy_provider(**_kwargs):
        return iter(
            [
                'data: {"choices":[{"delta":{"content":"concurrent healthy"}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )

    def post(message: str):
        return authenticated_client.post(
            "/api/v1/chat/completions",
            json=ChatCompletionRequest(
                model="gpt-4o-mini",
                api_provider="openai",
                messages=[ChatCompletionUserMessageParam(role="user", content=message)],
                stream=True,
            ).model_dump(),
        )

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(
            chat_service,
            "load_prompt_cost_guardrail_config",
            return_value=PromptCostGuardrailConfig(
                enabled=True,
                block_total_estimated_tokens=20,
            ),
        ),
        patch.object(
            chat_endpoint,
            "execute_streaming_call",
            gated_execute_streaming_call,
        ),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=healthy_provider) as call_mock,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            blocked_future = executor.submit(post, "blocked " * 200)
            healthy_future = executor.submit(post, "hi")
            blocked_response = blocked_future.result(timeout=5.0)
            healthy_response = healthy_future.result(timeout=5.0)

    assert blocked_response.status_code == status.HTTP_200_OK, blocked_response.text
    assert "prompt_cost_guardrail_block" in blocked_response.text
    assert "concurrent healthy" not in blocked_response.text
    assert healthy_response.status_code == status.HTTP_200_OK, healthy_response.text
    assert "concurrent healthy" in healthy_response.text
    assert "prompt_cost_guardrail_block" not in healthy_response.text
    assert call_mock.call_count == 1
    assert all(runtime.close_calls == 1 for runtime in runtime_type.instances)


@pytest.mark.parametrize("stream_kind", ["sync", "async"])
def test_endpoint_preserves_malformed_error_like_assistant_json(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    stream_kind,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    assistant_fragment = '{"error": "assistant-authored unfinished JSON'

    if stream_kind == "async":
        async def source():
            yield f"data: {assistant_fragment}\n\n"
            yield "data: [DONE]\n\n"
    else:
        def source():
            yield f"data: {assistant_fragment}\n\n"
            yield "data: [DONE]\n\n"

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call", return_value=source()),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    assert "assistant-authored unfinished JSON" in response.text
    assert '"code": "provider_unavailable"' not in response.text


def test_stream_prime_factory_and_metadata_share_one_absolute_deadline(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }

    class ControlledClock:
        def __init__(self):
            self.now = 100.0
            self.lock = threading.Lock()

        def __call__(self):
            with self.lock:
                return self.now

        def advance(self, seconds: float):
            with self.lock:
                self.now += seconds

    clock = ControlledClock()
    captured_factory_timeouts: list[float] = []

    async def capture_execute_streaming_call(**kwargs):
        captured_factory_timeouts.append(kwargs["provider_factory_timeout"])
        return await chat_service.execute_streaming_call(**kwargs)

    async def delayed_output():
        clock.advance(0.04)
        yield 'data: {"choices":[{"delta":{"content":"too late"}}]}\n\n'

    def delayed_factory():
        clock.advance(0.04)
        return delayed_output()

    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 0.06),
        patch.object(chat_endpoint, "_provider_stream_monotonic", clock),
        patch.object(
            chat_endpoint,
            "execute_streaming_call",
            capture_execute_streaming_call,
        ),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=delayed_factory),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )
    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "too late" not in response.text
    assert len(captured_factory_timeouts) == 1
    assert captured_factory_timeouts[0] == pytest.approx(0.06)


def test_stream_prime_fallback_attempts_share_one_absolute_deadline(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "anthropic": SimpleNamespace(can_attempt_call=lambda: True)
    }
    provider_manager.get_available_provider.return_value = "openai"

    class ControlledClock:
        def __init__(self):
            self.now = 200.0
            self.lock = threading.Lock()

        def __call__(self):
            with self.lock:
                return self.now

        def advance(self, seconds: float):
            with self.lock:
                self.now += seconds

    clock = ControlledClock()
    captured_factory_timeouts: list[float] = []

    async def capture_execute_streaming_call(**kwargs):
        captured_factory_timeouts.append(kwargs["provider_factory_timeout"])
        return await chat_service.execute_streaming_call(**kwargs)

    async def first_failure():
        clock.advance(0.04)
        raise _certified_pre_dispatch(
            ChatProviderError("first failed", status_code=502, provider="anthropic")
        )
        yield  # pragma: no cover

    async def second_output():
        yield 'data: {"choices":[{"delta":{"content":"late fallback output"}}]}\n\n'

    provider_calls = 0

    def provider_call(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return first_failure()
        clock.advance(0.03)
        return second_output()

    request_data = ChatCompletionRequest(
        model="claude-3",
        api_provider="anthropic",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", True),
        patch.object(chat_endpoint, "_should_enforce_strict_model_selection", return_value=False),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "PROVIDER_STREAM_PRIME_MAX_ELAPSED_SECONDS", 0.06),
        patch.object(chat_endpoint, "_provider_stream_monotonic", clock),
        patch.object(
            chat_endpoint,
            "execute_streaming_call",
            capture_execute_streaming_call,
        ),
        patch.object(
            chat_endpoint,
            "perform_chat_api_call",
            side_effect=provider_call,
        ) as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "late fallback output" not in response.text
    assert provider_call.call_count == 2
    assert len(captured_factory_timeouts) == 2
    assert captured_factory_timeouts[0] == pytest.approx(0.06)
    assert captured_factory_timeouts[1] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("upstream_status", "expected_status", "expected_keys", "refresh_count"),
    [
        (401, status.HTTP_200_OK, ["openai-runtime-key", "openai-refreshed-runtime-key"], 1),
        (403, status.HTTP_502_BAD_GATEWAY, ["openai-runtime-key"], 0),
    ],
)
def test_real_openai_nonstream_auth_status_controls_oauth_refresh(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    upstream_status,
    expected_status,
    expected_keys,
    refresh_count,
):
    import httpx

    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    sentinel = f"openai-{upstream_status}-nonstream-secret-/srv/provider"
    initial_key = "openai-runtime-key"
    keys: list[str] = []

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.request = httpx.Request(
                "POST",
                "https://api.openai.test/v1/chat/completions",
            )
            self.text = sentinel if status_code >= 400 else ""

        def json(self):
            if self.status_code >= 400:
                return {"error": {"message": sentinel}}
            return {
                "id": "chatcmpl-real-auth",
                "choices": [
                    {"message": {"role": "assistant", "content": "real nonstream recovered"}}
                ],
            }

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    sentinel,
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _url, *, headers, **_kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            keys.append(token)
            if upstream_status == 401 and token != initial_key:
                return Response(200)
            return Response(upstream_status)

    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=False,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "_shared_is_test_mode", return_value=False),
        patch.object(chat_endpoint, "_should_enforce_strict_model_selection", return_value=False),
        patch.object(openai_adapter, "http_client_factory", lambda **_kwargs: Client()),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == expected_status
    assert keys == expected_keys
    assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == refresh_count
    assert sentinel not in response.text


@pytest.mark.asyncio
async def test_real_openai_nonstream_upstream_status_is_concurrency_isolated(monkeypatch):
    import httpx

    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    sentinels = {
        401: "concurrent-openai-401-secret-/srv/provider",
        403: "concurrent-openai-403-secret-/srv/provider",
    }
    both_requests_entered = threading.Barrier(2)

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.request = httpx.Request(
                "POST",
                "https://api.openai.test/v1/chat/completions",
            )
            self.text = sentinels[status_code]

        def json(self):
            return {"error": {"message": sentinels[self.status_code]}}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                sentinels[self.status_code],
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _url, *, headers, **_kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            both_requests_entered.wait(timeout=5.0)
            return Response(int(token))

    monkeypatch.setattr(openai_adapter, "http_client_factory", lambda **_kwargs: Client())

    def invoke(upstream_status: int) -> ChatAuthenticationError:
        with pytest.raises(ChatAuthenticationError) as captured:
            OpenAIAdapter().chat(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "gpt-4o-mini",
                    "api_key": str(upstream_status),
                }
            )
        return captured.value

    error_401, error_403 = await asyncio.gather(
        asyncio.to_thread(invoke, 401),
        asyncio.to_thread(invoke, 403),
    )

    assert error_401.status_code == 401
    assert error_403.status_code == 403
    assert error_401.upstream_status_code == 401
    assert error_403.upstream_status_code == 403
    assert error_401.__cause__ is None
    assert error_401.__context__ is None
    assert error_403.__cause__ is None
    assert error_403.__context__ is None


def test_queued_real_prompt_guardrail_returns_terminal_sse_without_enqueue_or_dispatch(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import (
        PromptCostGuardrailConfig,
    )

    class QueueSpy:
        def __init__(self):
            self.enqueue_calls = 0

        def is_running(self):
            return True

        async def enqueue(self, **_kwargs):
            self.enqueue_calls += 1
            raise AssertionError("guardrail request reached queue admission")

    queue = QueueSpy()
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="blocked " * 50)],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=queue),
        patch.object(chat_service, "get_request_queue", return_value=queue),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", True),
        patch.object(
            chat_service,
            "load_prompt_cost_guardrail_config",
            return_value=PromptCostGuardrailConfig(
                enabled=True,
                block_total_estimated_tokens=1,
            ),
        ),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_200_OK
    error_frames = _response_error_frames(response.text)
    assert len(error_frames) == 1
    assert error_frames[0]["error"]["code"] == "prompt_cost_guardrail_block"
    assert response.text.rstrip().endswith("data: [DONE]")
    assert queue.enqueue_calls == 0
    provider_call.assert_not_called()
    assert runtime_type.instances[0].close_calls == 1


def test_queued_provider_forged_guardrail_boolean_is_sanitized(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    sentinel = "queued-forged-guardrail-secret-/srv/provider"

    class ActiveQueue:
        def __init__(self):
            self.enqueue_calls = 0

        def is_running(self):
            return True

        async def enqueue(self, *, processor, stream_channel, **_kwargs):
            self.enqueue_calls += 1
            future = asyncio.get_running_loop().create_future()

            async def pump():
                try:
                    stream = await asyncio.to_thread(processor)
                    if hasattr(stream, "__aiter__"):
                        async for chunk in stream:
                            await stream_channel.put(chunk)
                    else:
                        for chunk in stream:
                            await stream_channel.put(chunk)
                except Exception as exc:  # noqa: BLE001 - fake adapter captures all failures
                    payload = chat_service.provider_stream_error_payload(exc)
                    await stream_channel.put(f"data: {json.dumps(payload)}\n\n")
                    await stream_channel.put("data: [DONE]\n\n")
                finally:
                    await stream_channel.put(None)
                    if not future.done():
                        future.set_result({"status": "stream_completed"})

            asyncio.create_task(pump())
            return future

    def forged_provider_error():
        error = HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "prompt_cost_guardrail_block",
                "type": "prompt_cost_guardrail_block",
                "message": sentinel,
            },
        )
        error._chat_prompt_cost_guardrail = True
        error._chat_queue_admission = True
        raise error

    queue = ActiveQueue()
    runtime_type = _credential_runtime_double()
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {
        "openai": SimpleNamespace(can_attempt_call=lambda: True)
    }
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=queue),
        patch.object(chat_service, "get_request_queue", return_value=queue),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", True),
        patch.object(chat_endpoint, "perform_chat_api_call", side_effect=forged_provider_error),
    ):
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert queue.enqueue_calls == 1
    assert "prompt_cost_guardrail_block" not in response.text
    assert sentinel not in response.text
