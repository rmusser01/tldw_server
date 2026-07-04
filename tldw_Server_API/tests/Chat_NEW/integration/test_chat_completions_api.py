"""
Integration tests for the /chat/completions API endpoint.

Tests the full request/response flow with real database and minimal mocking.
Only external LLM APIs are mocked to avoid actual API calls.
"""

import os

import pytest
from fastapi import status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _install_isolated_chat_db(test_client, tmp_path, monkeypatch):
    user_base = tmp_path / "user_databases" / "1"
    user_base.mkdir(parents=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    db = CharactersRAGDB(db_path=user_base / "ChaChaNotes.db", client_id="chat_macros_chat_test")
    test_client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return db


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""

# ========================================================================
# Basic Endpoint Tests
# ========================================================================

class TestChatCompletionsEndpoint:
    """Test the /v1/chat/completions endpoint."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_basic_completion_request(self, test_client, auth_headers):
        """Test basic chat completion request - REAL API CALL."""

        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers=auth_headers
        )

        # Debug output if test fails
        if response.status_code != status.HTTP_200_OK:
            print(f"\nResponse status: {response.status_code}")
            print(f"Response body: {response.json()}")
            print(f"OPENAI_API_KEY env: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:10]}..." if os.getenv('OPENAI_API_KEY') else "OPENAI_API_KEY: NOT SET")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"]
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"]

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_multi_turn_conversation(self, test_client, auth_headers):
        """Test multi-turn conversation handling - REAL API CALL."""

        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                    {"role": "user", "content": "What about 3+3?"}
                ]
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.integration
    def test_missing_auth_header(self, test_client):
        """Test request without authentication header."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        # Depending on auth configuration, might be 401 or allowed
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]

    @pytest.mark.integration
    def test_invalid_request_body(self, test_client, auth_headers):
        """Test request with invalid body."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "invalid_field": "value"
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    @pytest.mark.integration
    def test_empty_messages_list(self, test_client, auth_headers):
        """Test request with empty messages list."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": []
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    @pytest.mark.integration
    def test_time_slash_command_still_injects_and_calls_provider(self, monkeypatch, test_client, auth_headers):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
        monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        captured = {"messages": None, "kwargs": None}

        def fake_call(**kwargs):
            captured["kwargs"] = kwargs
            captured["messages"] = kwargs.get("messages_payload")
            return {"choices": [{"message": {"role": "assistant", "content": "provider ok"}}]}

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

        response = test_client.post(
            "/api/v1/chat/completions",
            json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}]},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK, response.text
        messages = captured["messages"] or []
        system_text = "\n".join(
            [_message_text(message) for message in messages if message.get("role") == "system"]
            + [str((captured["kwargs"] or {}).get("system_message") or "")]
        )
        assert "[/time]" in system_text

    @pytest.mark.integration
    def test_wrapup_macro_short_circuits_provider(self, monkeypatch, test_client, auth_headers, tmp_path):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
        db = _install_isolated_chat_db(test_client, tmp_path, monkeypatch)

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        def fail_call(**_kwargs):
            raise AssertionError("provider path should not be called for chat macros")

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)

        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": '/wrapup --question "What changed?"'}],
                    "stream": False,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK, response.text
            body = response.json()
            message = body["choices"][0]["message"]
            macro_meta = message["metadata"]["chat_macro"]
            assert message["role"] == "assistant"
            assert macro_meta["command"] == "wrapup"
            assert macro_meta["status"] == "pending"
            assert macro_meta["detail_url"].endswith(f"/runs/{macro_meta['run_id']}")

            run = ChatMacroRepository(db).get_run(macro_meta["run_id"])
            assert run is not None
            assert run.macro_command == "wrapup"
            assert run.normalized_args["question"] == ["What changed?"]
        finally:
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
            db.close_all_connections()

    @pytest.mark.integration
    def test_wrapup_macro_stream_request_returns_json_status(
        self,
        monkeypatch,
        test_client,
        auth_headers,
        tmp_path,
    ):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
        db = _install_isolated_chat_db(test_client, tmp_path, monkeypatch)

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        def fail_call(**_kwargs):
            raise AssertionError("provider path should not be called for streaming chat macros")

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)

        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "/wrapup"}],
                    "stream": True,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK, response.text
            assert response.headers["content-type"].startswith("application/json")
            body = response.json()
            assert body["object"] == "chat.completion"
            macro_meta = body["choices"][0]["message"]["metadata"]["chat_macro"]
            assert macro_meta["command"] == "wrapup"
            assert macro_meta["status"] == "pending"
        finally:
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
            db.close_all_connections()

    @pytest.mark.integration
    def test_wrapup_macro_discovery_failure_does_not_fall_through_to_provider(
        self,
        monkeypatch,
        test_client,
        auth_headers,
        tmp_path,
    ):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
        db = _install_isolated_chat_db(test_client, tmp_path, monkeypatch)

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        def fail_call(**_kwargs):
            raise AssertionError("provider path should not receive failed macro invocations")

        def fail_list(_service):
            raise RuntimeError("api_key=secret should not leak")

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)
        monkeypatch.setattr(chat_endpoint.ChatMacrosService, "list_macros", fail_list)

        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "/wrapup"}],
                    "stream": False,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK, response.text
            message = response.json()["choices"][0]["message"]
            macro_meta = message["metadata"]["chat_macro"]
            assert "Could not run /wrapup: Macro storage is unavailable." in message["content"]
            assert "secret" not in message["content"]
            assert macro_meta["status"] == "error"
            assert macro_meta["error_code"] == "storage_error"
        finally:
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
            db.close_all_connections()

    @pytest.mark.integration
    def test_wrapup_macro_rate_limit_denial_does_not_create_run(
        self,
        monkeypatch,
        test_client,
        auth_headers,
        tmp_path,
    ):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
        db = _install_isolated_chat_db(test_client, tmp_path, monkeypatch)

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        class DenyLimiter:
            config = type("Config", (), {"per_user_rpm": 1})()

            async def check_rate_limit(self, **_kwargs):
                return False, "macro test limit"

        def fail_call(**_kwargs):
            raise AssertionError("provider path should not be called when macro start is rate-limited")

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)
        monkeypatch.setattr(chat_endpoint, "get_rate_limiter", lambda: DenyLimiter())

        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "/wrapup"}],
                    "stream": False,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS, response.text
            with db.transaction() as conn:
                count = conn.execute("SELECT COUNT(*) FROM chat_macro_runs").fetchone()[0]
            assert count == 0
        finally:
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
            db.close_all_connections()

    @pytest.mark.integration
    def test_wrapup_macro_invalid_args_returns_chat_visible_error(
        self,
        monkeypatch,
        test_client,
        auth_headers,
        tmp_path,
    ):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
        db = _install_isolated_chat_db(test_client, tmp_path, monkeypatch)

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        def fail_call(**_kwargs):
            raise AssertionError("provider path should not be called for invalid chat macros")

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)

        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "/wrapup --not-a-real-arg value"}],
                    "stream": False,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK, response.text
            message = response.json()["choices"][0]["message"]
            macro_meta = message["metadata"]["chat_macro"]
            assert "Could not run /wrapup" in message["content"]
            assert macro_meta["status"] == "error"
            assert macro_meta["error_code"] == "validation_error"
            assert "run_id" not in macro_meta
        finally:
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
            db.close_all_connections()

    @pytest.mark.integration
    def test_unknown_slash_candidate_preserves_provider_flow(self, monkeypatch, test_client, auth_headers):
        monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        calls = []

        def fake_call(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"role": "assistant", "content": "provider fallback"}}]}

        monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

        response = test_client.post(
            "/api/v1/chat/completions",
            json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/not_a_macro"}]},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK, response.text
        assert calls
        assert response.json()["choices"][0]["message"]["content"] == "provider fallback"

# ========================================================================
# Provider Routing Tests
# ========================================================================

class TestProviderRouting:
    """Test routing to different LLM providers."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_openai_provider_routing(self, test_client, auth_headers):
        """Test routing to OpenAI provider - REAL API CALL."""

        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "api_provider": "openai",
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY for real integration test")
    def test_anthropic_provider_routing(self, test_client, auth_headers):
        """Test routing to Anthropic provider - REAL API CALL."""

        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "api_provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_default_provider_fallback(self, test_client, auth_headers):
        """Test fallback to default provider when not specified - REAL API CALL."""

        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data

# ========================================================================
# Database Integration Tests
# ========================================================================

class TestDatabaseIntegration:
    """Test database persistence and retrieval."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_conversation_saved_to_database(self, test_client, populated_chacha_db, auth_headers):
        """Test that conversations are saved to database with real provider."""

        # Override dependency to use our test database on the client app instance
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

        def override_get_db():

            return populated_chacha_db

        test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db

        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Save this message"}]
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

            # Check database for saved conversation
            # Get the default character first
            characters = populated_chacha_db.list_character_cards()
            default_char = next((c for c in characters if c['name'] == DEFAULT_CHARACTER_NAME), None)
            assert default_char is not None

            # Get conversations for the character
            conversations = populated_chacha_db.get_conversations_for_character(default_char['id'])
            assert len(conversations) > 0

        finally:
            # Clean up dependency override
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)

    @pytest.mark.integration
    def test_message_history_retrieval(self, test_client, populated_chacha_db, auth_headers):
        """Test retrieving conversation history."""
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

        def override_get_db():

            return populated_chacha_db

        test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db

        try:
            # Get the character from populated DB
            characters = populated_chacha_db.list_character_cards()
            assert len(characters) > 0

            # Find the character we created in the fixture (default character with client_id test_user)
            test_char = next((c for c in characters if c['name'] == DEFAULT_CHARACTER_NAME and c['client_id'] == 'test_user'), None)
            assert test_char is not None, f"Could not find test character '{DEFAULT_CHARACTER_NAME}'"

            conversations = populated_chacha_db.get_conversations_for_character(test_char["id"])
            assert len(conversations) > 0

            first_conv = conversations[0]
            messages = populated_chacha_db.get_messages_for_conversation(first_conv["id"])
            assert len(messages) > 0

        finally:
            test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the API."""

    @pytest.mark.integration
    def test_rate_limit_error_handling(self, test_client, auth_headers, monkeypatch):
        """Test rate limit with deterministic TEST_MODE chat limits (per-user RPM=2)."""
        monkeypatch.setenv("TEST_MODE", "true")
        monkeypatch.setenv("RG_ENABLED", "0")
        monkeypatch.setenv("TEST_CHAT_PER_USER_RPM", "2")
        monkeypatch.setenv("TEST_CHAT_PER_CONVERSATION_RPM", "2")
        monkeypatch.setenv("TEST_CHAT_GLOBAL_RPM", "10")
        monkeypatch.setenv("TEST_CHAT_TOKENS_PER_MINUTE", "1000")
        monkeypatch.delenv("TEST_CHAT_BURST_MULTIPLIER", raising=False)
        from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter
        initialize_rate_limiter()

        statuses = []
        for i in range(4):
            resp = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": f"Ping {i}"}]
                },
                headers=auth_headers
            )
            statuses.append(resp.status_code)
        assert 429 in statuses

    @pytest.mark.integration
    def test_auth_error_handling(self, credentialed_test_client, auth_headers):
        """Test handling of authentication errors by forcing provider to raise ChatAuthenticationError."""
        from unittest.mock import patch
        from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
        def raise_auth(*args, **kwargs):
            raise ChatAuthenticationError("Invalid API key", provider="openai")
        with patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call', new=raise_auth):
            response = credentialed_test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Test"}]
                },
                headers=auth_headers
            )
            # Provider authentication is a downstream failure, not client auth.
            assert response.status_code == status.HTTP_502_BAD_GATEWAY
            data = response.json()
            assert data["detail"]["error_code"] == "provider_authentication_failed"
            assert "Invalid API key" not in response.text

    @pytest.mark.unit
    def test_general_error_handling(self, credentialed_test_client, auth_headers):
        """Test handling of general errors by forcing provider call to raise."""
        from unittest.mock import patch
        def boom(*args, **kwargs):
            raise Exception("Unexpected error")
        with patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call', new=boom):
            response = credentialed_test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Test"}]
                },
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_502_BAD_GATEWAY
            data = response.json()
            assert data["detail"] == "The chat service provider is currently unavailable."
            assert "Unexpected error" not in response.text

# ========================================================================
# Request Queue Admission (queued execution disabled)
# ========================================================================


class _BaseQueueStub:
    """Common helpers for queue stubs used in integration tests."""

    def __init__(self):

        self.enqueue_calls = 0

    def is_running(self) -> bool:

        return True


class _InactiveQueueStub(_BaseQueueStub):
    def __init__(self):
        super().__init__()
        self._running = False

    def is_running(self) -> bool:

        return False

    async def enqueue(self, *args, **kwargs):
        raise AssertionError("enqueue should not be called when queue is inactive")


class _ActiveQueueStub(_BaseQueueStub):
    def __init__(self, *, fail_with: Exception | None = None):
        super().__init__()
        self._running = True
        self._fail_with = fail_with

    class _ResolvedAdmission:
        """Simple awaitable that completes immediately."""

        __slots__ = ()

        def __await__(self):

            return iter(())

    async def enqueue(self, *args, **kwargs):
        self.enqueue_calls += 1
        if self._fail_with is not None:
            raise self._fail_with

        return self._ResolvedAdmission()


class TestRequestQueueAdmission:
    """Validate queue gating behaviour when queued execution is disabled."""

    @pytest.mark.integration
    def test_inactive_queue_is_bypassed(
        self,
        credentialed_test_client,
        auth_headers,
        monkeypatch,
    ):
        """Requests skip enqueue when queue reports inactive state."""
        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        queue_stub = _InactiveQueueStub()
        monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: queue_stub)
        monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False, raising=False)

        response = credentialed_test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Ensure inactive queue is ignored"}],
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        assert queue_stub.enqueue_calls == 0

    @pytest.mark.integration
    def test_active_queue_admission(
        self,
        credentialed_test_client,
        auth_headers,
        monkeypatch,
    ):
        """Active queue gates admission without delaying the response."""
        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        queue_stub = _ActiveQueueStub()
        monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: queue_stub)
        monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False, raising=False)

        response = credentialed_test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Trigger queue gating"}],
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        assert queue_stub.enqueue_calls == 1

    @pytest.mark.integration
    def test_queue_admission_failure_returns_503(self, test_client, auth_headers, monkeypatch):
        """Unexpected queue failures surface as 503 to the client."""
        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        queue_stub = _ActiveQueueStub(fail_with=RuntimeError("boom"))
        monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: queue_stub)
        monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False, raising=False)

        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Failure path"}],
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

# ========================================================================
# Streaming Tests
# ========================================================================

class TestStreamingResponses:
    """Test streaming response functionality."""

    @pytest.mark.integration
    @pytest.mark.streaming
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    async def test_streaming_response(self, async_client, auth_headers):
        """Test streaming chat completion - REAL API CALL."""
        async with async_client.stream(
                "POST",
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Stream this"}],
                    "stream": True
                },
                headers=auth_headers
            ) as response:
                # Debug: print error if request failed
                if response.status_code != status.HTTP_200_OK:
                    error_text = await response.aread()
                    print(f"\nStreaming test error response: {error_text.decode() if error_text else 'No response body'}")
                assert response.status_code == status.HTTP_200_OK

                chunks = []
                async for chunk in response.aiter_text():
                    if chunk:
                        chunks.append(chunk)

                assert len(chunks) > 0
                # Should receive SSE formatted chunks
                assert any("data:" in chunk for chunk in chunks)

# ========================================================================
# Parameter Validation Tests
# ========================================================================

class TestParameterValidation:
    """Test parameter validation and constraints."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_temperature_bounds(self, test_client, auth_headers):
        """Test temperature parameter bounds - REAL API CALL."""

        # Valid temperature
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 1.5
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

        # Invalid temperature (too high)
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 2.5
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_max_tokens_validation(self, test_client, auth_headers):
        """Test max_tokens parameter validation - REAL API CALL."""

        # Valid max_tokens
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 100
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

        # Invalid max_tokens (negative)
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": -1
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
