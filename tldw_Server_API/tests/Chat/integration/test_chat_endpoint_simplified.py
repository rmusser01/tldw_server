"""
Simplified chat endpoint tests using real database and authentication.
"""

import asyncio
import json
import os
import sqlite3
from contextlib import ExitStack
from typing import Optional
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from loguru import logger
from starlette.requests import Request
from starlette.responses import StreamingResponse

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME

_CHAT_USER_SECRET = "sk-chat-user-secret-must-not-leak"


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
            self.marked_used: list[str] = []
            self.close_calls = 0
            self.__class__.instances.append(self)

        async def resolve(self, provider: str, *, force_refresh: bool = False):
            provider = provider.strip().lower()
            self.resolve_calls.append((provider, force_refresh))
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
            return SimpleNamespace(
                provider=provider,
                api_key=key,
                app_config={f"{provider}_api": {"model": "runtime-model"}},
                auth_source=(auth_sources or {}).get(provider, auth_source),
                credentials_resolved=True,
            )

        async def mark_used(self, handle) -> None:
            if handle.provider not in self.marked_used:
                self.marked_used.append(handle.provider)

        async def close(self) -> None:
            self.close_calls += 1

    return RuntimeDouble


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
            await kwargs["on_success"](kwargs["selected_provider"])
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
            raise ChatAuthenticationError(
                "expired OAuth credential",
                provider="openai",
            )

        def refreshed_stream():
            yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
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
            async def fetch_secret_for_user(self, user_id: int, provider: str):
                assert (user_id, provider) == (1, "openai")
                resolution_path.append(error_code)
                return {
                    "encrypted_blob": f"{_CHAT_USER_SECRET}:invalid-envelope",
                    "last_used_at": None,
                }

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
            await kwargs["on_success"](kwargs["selected_provider"])
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
        return {"choices": [{"message": {"content": "openai/gpt-4o-mini"}}]}

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
        except Exception as exc:
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
        raise ChatAuthenticationError("sentinel expired oauth", provider="openai")

    def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
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


@pytest.mark.parametrize("oauth_enabled", [True, False])
def test_streaming_keepalive_before_auth_error_is_not_provider_output(
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
        raise ChatAuthenticationError("sentinel keepalive auth", provider="openai")

    def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
        yield "data: [DONE]\n\n"

    provider_side_effect = [keepalive_then_error(), refreshed_stream()] if oauth_enabled else [keepalive_then_error()]
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

    if oauth_enabled:
        assert response.status_code == status.HTTP_200_OK
        assert "recovered" in response.text
        assert provider_call.call_count == 2
        assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1
    else:
        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
        assert provider_call.call_count == 1
        assert ("openai", True) not in runtime_type.instances[0].resolve_calls
    assert "sentinel" not in response.text.lower()


@pytest.mark.parametrize("control_line", ["id: stream-7", "retry: 1500"])
@pytest.mark.parametrize("oauth_enabled", [True, False])
def test_streaming_sse_control_field_before_auth_error_is_not_provider_output(
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

    def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
        yield "data: [DONE]\n\n"

    side_effect = [control_then_error(), refreshed_stream()] if oauth_enabled else [control_then_error()]
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

    if oauth_enabled:
        assert response.status_code == status.HTTP_200_OK
        assert "recovered" in response.text
        assert provider_call.call_count == 2
        assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1
    else:
        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
        assert provider_call.call_count == 1
    assert "sentinel" not in response.text.lower()


@pytest.mark.parametrize("oauth_enabled", [True, False])
def test_streaming_delayed_sse_controls_do_not_cross_credential_handoff(
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

    async def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
        yield "data: [DONE]\n\n"

    side_effect = (
        [delayed_control_then_error(), refreshed_stream()] if oauth_enabled else [delayed_control_then_error()]
    )
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
    if oauth_enabled:
        assert response.status_code == status.HTTP_200_OK
        assert "recovered" in response.text
        assert provider_call.call_count == 2
        assert runtime_type.instances[0].resolve_calls.count(("openai", True)) == 1
    else:
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
            side_effect=[ChatProviderError(provider="anthropic", message="initial failed", status_code=502), recovered],
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
            side_effect=ChatProviderError(provider="anthropic", message="initial failed", status_code=502),
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
        yield "data: [DONE]\n\n"

    usage_before_failure = []

    async def fallback_stream():
        if partial_output:
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            usage_before_failure.append(list(runtime_type.instances[0].marked_used))
        raise ChatAuthenticationError("sentinel fallback stream auth", provider="openai")

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
                ChatProviderError(provider="anthropic", message="initial failed", status_code=502),
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
        raise ChatAuthenticationError("sentinel fallback stream auth", provider="openai")

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
                ChatProviderError(provider="anthropic", message="initial failed", status_code=502),
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
                ChatProviderError(provider="anthropic", message="initial failed", status_code=502),
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


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "expected_code",
    [
        "invalid_provider_credentials",
        "missing_provider_credentials",
        "credential_store_unavailable",
        "credential_scope_revoked",
        "provider_configuration_invalid",
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
            raise ChatAuthenticationError("sentinel initial auth", provider="openai")

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

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = response.json()["detail"]
    assert detail["error_code"] == expected_code
    assert "sentinel" not in str(detail).lower()
    assert provider_call.call_count == 1
    provider_manager.get_available_provider.assert_not_called()


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


def test_streaming_clean_empty_response_marks_runtime_used_once(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
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
        if False:
            yield ""

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

    assert response.status_code == status.HTTP_200_OK
    assert runtime_type.instances[0].marked_used == ["openai"]
    assert runtime_type.instances[0].close_calls == 1


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
