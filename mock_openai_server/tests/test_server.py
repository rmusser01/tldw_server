"""
Comprehensive tests for the Mock OpenAI API Server.
"""

import os
import json
import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Skip entire suite unless explicitly enabled
_RUN_MOCK_OPENAI = os.getenv("RUN_MOCK_OPENAI", "").lower() in ("1", "true", "yes")
pytestmark = pytest.mark.skipif(not _RUN_MOCK_OPENAI, reason="Mock OpenAI server tests disabled; set RUN_MOCK_OPENAI=1 to enable")

# Import the app and configuration
from ..mock_openai.server import app
from ..mock_openai.config import MockConfig, load_config
from ..mock_openai.config import ResponsePattern
from ..mock_openai.responses import ResponseManager


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Get valid authentication headers."""
    return {"Authorization": "Bearer sk-test-key-12345"}


@pytest.fixture
def invalid_auth_headers():
    """Get invalid authentication headers."""
    return {"Authorization": "Bearer invalid-key"}


class TestAuthentication:
    """Test authentication functionality."""

    def test_valid_api_key(self, client, auth_headers):
        """Test with valid API key."""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200

    def test_invalid_api_key(self, client, invalid_auth_headers):
        """Test with invalid API key."""
        response = client.get("/v1/models", headers=invalid_auth_headers)
        assert response.status_code == 401

    def test_missing_api_key(self, client):
        """Test without API key."""
        response = client.get("/v1/models")
        assert response.status_code == 401


class TestChatCompletions:
    """Test chat completions endpoint."""

    def test_basic_chat_completion(self, client, auth_headers):
        """Test basic chat completion request."""
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_with_system_message(self, client, auth_headers):
        """Test chat completion with system message."""
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.5
        }

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_with_parameters(self, client, auth_headers):
        """Test chat completion with various parameters."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "n": 1
        }

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, async_client, auth_headers):
        """Test streaming chat completion."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True
        }

        async with async_client.stream(
            "POST",
            "/v1/chat/completions",
            headers=auth_headers,
            json=payload
        ) as response:
            assert response.status_code == 200

            chunks = []
            async for line in response.aiter_lines():
                if line and not line.startswith("data: [DONE]"):
                    if line.startswith("data: "):
                        chunk_data = json.loads(line[6:])
                        chunks.append(chunk_data)

            assert len(chunks) > 0
            assert chunks[0]["object"] == "chat.completion.chunk"

    def test_chat_fail_once_then_success(self, auth_headers):
        """Test a configured chat scenario failure only applies once."""
        from ..mock_openai.config import MockConfig, ServerConfig
        from ..mock_openai.server import app, get_config_instance

        cfg = MockConfig(
            server=ServerConfig(log_requests=False),
            scenario_failures={
                "chat_completions": [
                    {
                        "match": {"model": "gpt-4.1-mini"},
                        "status_code": 503,
                        "message": "UAT transient chat failure",
                        "type": "server_error",
                        "code": "uat_fail_once",
                        "times": 1,
                    }
                ]
            },
        )
        app.dependency_overrides[get_config_instance] = lambda: cfg
        try:
            client = TestClient(app)
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "hello"}],
            }
            first = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
            second = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
            assert first.status_code == 503
            assert first.json()["detail"]["error"]["message"] == "UAT transient chat failure"
            assert second.status_code == 200
        finally:
            app.dependency_overrides.clear()


class TestEmbeddings:
    """Test embeddings endpoint."""

    def test_single_embedding(self, client, auth_headers):
        """Test creating a single embedding."""
        payload = {
            "model": "text-embedding-ada-002",
            "input": "This is a test text"
        }

        response = client.post(
            "/v1/embeddings",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)

    def test_multiple_embeddings(self, client, auth_headers):
        """Test creating multiple embeddings."""
        payload = {
            "model": "text-embedding-ada-002",
            "input": ["First text", "Second text", "Third text"]
        }

        response = client.post(
            "/v1/embeddings",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) == 3
        for i, embedding_data in enumerate(data["data"]):
            assert embedding_data["index"] == i
            assert "embedding" in embedding_data

    def test_embedding_fail_once_then_success(self, auth_headers):
        """Test a configured embeddings scenario failure only applies once."""
        from ..mock_openai.config import MockConfig, ServerConfig
        from ..mock_openai.server import app, get_config_instance

        cfg = MockConfig(
            server=ServerConfig(log_requests=False),
            scenario_failures={
                "embeddings": [
                    {
                        "match": {"model": "text-embedding-3-small"},
                        "status_code": 429,
                        "message": "UAT transient embeddings failure",
                        "error_type": "rate_limit_error",
                        "code": "uat_embedding_fail_once",
                        "times": 1,
                    }
                ]
            },
        )
        app.dependency_overrides[get_config_instance] = lambda: cfg
        try:
            client = TestClient(app)
            payload = {
                "model": "text-embedding-3-small",
                "input": "hello",
            }
            first = client.post("/v1/embeddings", headers=auth_headers, json=payload)
            second = client.post("/v1/embeddings", headers=auth_headers, json=payload)
            assert first.status_code == 429
            assert first.json()["detail"]["error"]["type"] == "rate_limit_error"
            assert second.status_code == 200
        finally:
            app.dependency_overrides.clear()


class TestCompletions:
    """Test legacy completions endpoint."""

    def test_basic_completion(self, client, auth_headers):
        """Test basic completion request."""
        payload = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Once upon a time",
            "max_tokens": 50
        }

        response = client.post(
            "/v1/completions",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "text_completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]


class TestModels:
    """Test models endpoint."""

    def test_list_models(self, client, auth_headers):
        """Test listing available models."""
        response = client.get("/v1/models", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0

        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "owned_by" in model


class TestConfiguration:
    """Test configuration management."""

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "server": {
                "host": "127.0.0.1",
                "port": 9090
            },
            "streaming": {
                "enabled": False
            }
        }

        config = MockConfig.from_dict(config_dict)

        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9090
        assert config.streaming.enabled is False

    def test_pattern_matching(self):
        """Test request pattern matching."""

        pattern = ResponsePattern(
            match={"model": "gpt-4", "content_regex": ".*test.*"},
            response_file="test.json"
        )

        # Should match
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "This is a test"}]
        }
        assert pattern.matches(request_data) is True

        # Should not match - different model
        request_data["model"] = "gpt-3.5-turbo"
        assert pattern.matches(request_data) is False

        # Should not match - no "test" in content
        request_data["model"] = "gpt-4"
        request_data["messages"][0]["content"] = "Hello world"
        assert pattern.matches(request_data) is False

    def test_scenario_failures_parse_type_aliases(self):
        """Test scenario failures parse OpenAI-style type aliases."""
        config = MockConfig.from_dict(
            {
                "scenario_failures": {
                    "chat_completions": [
                        {
                            "match": {"model": "gpt-4.1-mini"},
                            "status_code": 503,
                            "message": "transient chat",
                            "type": "server_error",
                            "code": "chat_fail_once",
                            "times": 1,
                        },
                        {
                            "match": {"model": "gpt-4.1"},
                            "status_code": 429,
                            "message": "rate limited",
                            "error_type": "rate_limit_error",
                            "code": "chat_rate_limit",
                            "times": 2,
                        },
                    ]
                }
            }
        )

        failures = config.scenario_failures["chat_completions"]
        assert failures[0].error_type == "server_error"
        assert failures[1].error_type == "rate_limit_error"

    def test_response_base_dir_resolves_relative_to_config_file(self, tmp_path):
        """Test response fixtures can live next to static config directories."""
        config_dir = tmp_path / "configs"
        responses_dir = tmp_path / "responses"
        config_dir.mkdir()
        responses_dir.mkdir()
        config_path = config_dir / "hosted-success.json"
        config_path.write_text(
            json.dumps({"response_base_dir": "../responses"}),
            encoding="utf8",
        )

        config = MockConfig.from_file(config_path)

        assert config.response_base_dir == responses_dir

    def test_loaded_config_drives_request_handlers_after_server_import(
        self,
        tmp_path,
        auth_headers,
    ):
        """Test file-loaded config remains visible to request dependencies."""
        from ..mock_openai import config as config_module
        from ..mock_openai.server import app, get_config_instance

        # Prime the dependency before loading a file config, matching the import-time path.
        get_config_instance()

        config_dir = tmp_path / "configs"
        responses_dir = tmp_path / "responses"
        chat_dir = responses_dir / "chat"
        config_dir.mkdir()
        chat_dir.mkdir(parents=True)
        (chat_dir / "source-summary.json").write_text(
            json.dumps(
                {
                    "id": "chatcmpl-config-backed-uat",
                    "object": "chat.completion",
                    "created": 1770000000,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Config backed source summary response.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 6,
                        "total_tokens": 12,
                    },
                }
            ),
            encoding="utf8",
        )
        config_path = config_dir / "hosted-success.json"
        config_path.write_text(
            json.dumps(
                {
                    "response_base_dir": "../responses",
                    "responses": {
                        "chat_completions": {
                            "patterns": [
                                {
                                    "match": {"content_regex": "(?i).*source.*"},
                                    "response_file": "chat/source-summary.json",
                                    "priority": 10,
                                }
                            ],
                            "default": "chat/source-summary.json",
                        }
                    },
                }
            ),
            encoding="utf8",
        )

        try:
            load_config(config_path)
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                headers=auth_headers,
                json={
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Summarize this source.",
                        }
                    ],
                },
            )

            assert response.status_code == 200
            assert (
                response.json()["choices"][0]["message"]["content"]
                == "Config backed source summary response."
            )
        finally:
            config_module._config = None
            cache_clear = getattr(get_config_instance, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()

    def test_scenario_failure_counts_are_config_local(self):
        """Test scenario failure counters stay isolated per config instance."""
        from ..mock_openai.server import (
            maybe_raise_scenario_failure,
            reset_scenario_failure_counts,
        )

        def make_config():
            return MockConfig(
                scenario_failures={
                    "chat_completions": [
                        {
                            "match": {"model": "gpt-4.1-mini"},
                            "status_code": 503,
                            "times": 1,
                        }
                    ]
                }
            )

        request_data = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }

        first_config = make_config()
        with pytest.raises(HTTPException):
            maybe_raise_scenario_failure(
                "chat_completions",
                request_data,
                first_config,
            )
        maybe_raise_scenario_failure("chat_completions", request_data, first_config)

        second_config = make_config()
        with pytest.raises(HTTPException):
            maybe_raise_scenario_failure(
                "chat_completions",
                request_data,
                second_config,
            )

        reset_scenario_failure_counts(first_config)
        with pytest.raises(HTTPException):
            maybe_raise_scenario_failure(
                "chat_completions",
                request_data,
                first_config,
            )


class TestResponseManager:
    """Test response management."""

    def test_template_variables(self):
        """Test template variable substitution."""
        manager = ResponseManager()

        vars = manager.get_template_vars()
        assert "timestamp" in vars
        assert "request_id" in vars
        assert "chat_id" in vars

        manager.set_template_var("custom", "value")
        vars = manager.get_template_vars()
        assert vars["custom"] == "value"

    def test_default_responses(self):
        """Test default response generation."""
        manager = ResponseManager()

        # Chat response
        chat_response = manager.get_default_chat_response()
        assert chat_response["object"] == "chat.completion"
        assert "choices" in chat_response

        # Embedding response
        embedding_response = manager.get_default_embedding_response()
        assert embedding_response["object"] == "list"
        assert "data" in embedding_response

        # Completion response
        completion_response = manager.get_default_completion_response()
        assert completion_response["object"] == "text_completion"
        assert "choices" in completion_response

    def test_response_file_embedding_batches_stay_deterministic(self, tmp_path):
        """Test static embedding fixtures are reused for multi-input requests."""
        responses_dir = tmp_path / "responses"
        embedding_dir = responses_dir / "embeddings"
        embedding_dir.mkdir(parents=True)
        (embedding_dir / "default.json").write_text(
            json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.01, 0.02, 0.03, 0.04],
                        }
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                }
            ),
            encoding="utf8",
        )

        manager = ResponseManager(responses_dir=responses_dir)
        response = manager.generate_embedding_response(
            {
                "model": "text-embedding-3-small",
                "input": ["first", "second", "third"],
            },
            "embeddings/default.json",
        )

        assert len(response.data) == 3
        assert [item.index for item in response.data] == [0, 1, 2]
        assert [item.embedding for item in response.data] == [
            [0.01, 0.02, 0.03, 0.04],
            [0.01, 0.02, 0.03, 0.04],
            [0.01, 0.02, 0.03, 0.04],
        ]

    def test_cached_chat_fixture_keeps_request_model_dynamic(self, tmp_path):
        """Test cached response fixtures are not mutated by model echoing."""
        responses_dir = tmp_path / "responses"
        chat_dir = responses_dir / "chat"
        chat_dir.mkdir(parents=True)
        (chat_dir / "default.json").write_text(
            json.dumps(
                {
                    "id": "chatcmpl-dynamic-model",
                    "object": "chat.completion",
                    "created": 1770000000,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "dynamic model response",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 4,
                        "total_tokens": 8,
                    },
                }
            ),
            encoding="utf8",
        )

        manager = ResponseManager(responses_dir=responses_dir)

        hosted = manager.generate_chat_response(
            {"model": "gpt-4.1-mini", "messages": []},
            "chat/default.json",
        )
        local = manager.generate_chat_response(
            {"model": "local-uat-chat", "messages": []},
            "chat/default.json",
        )

        assert hosted.model == "gpt-4.1-mini"
        assert local.model == "local-uat-chat"


class TestErrorHandling:
    """Test error handling and simulation."""

    def test_404_not_found(self, client, auth_headers):
        """Test 404 for non-existent endpoint."""
        response = client.get("/v1/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_invalid_request_body(self, client, auth_headers):
        """Test invalid request body."""
        payload = {
            "invalid": "data"
        }

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json=payload
        )

        assert response.status_code == 422  # Validation error


class TestHealthCheck:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
