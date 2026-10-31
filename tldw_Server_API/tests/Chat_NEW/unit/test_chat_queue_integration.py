"""
Unit tests for request queue admission control in /api/v1/chat/completions.

These tests stub the request queue to validate that:
- When the queue rejects (e.g., full), the endpoint returns 429.
- When the queue admits quickly, the endpoint proceeds and returns 200 with a mocked LLM response.
"""

import asyncio
import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    DEFAULT_CHARACTER_NAME,
    get_chacha_db_for_user,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.main import app

pytest_plugins = (
    "tldw_Server_API.tests.Chat.credential_runtime_fixtures",
)


@contextmanager
def _test_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client")
    db.add_character_card({
        "name": DEFAULT_CHARACTER_NAME,
        "description": "Default",
        "personality": "Helpful",
        "scenario": "Testing",
        "system_prompt": "You are helpful",
        "first_message": "Hello",
        "creator_notes": "test"
    })
    try:
        yield db
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None


def _auth_headers(client: TestClient):
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    settings = get_settings()
    api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
    return {"X-API-KEY": api_key, "X-CSRF-Token": getattr(client, 'csrf_token', '')}


class _QueueStubReject:
    async def enqueue(self, *args, **kwargs):
        raise ValueError("Queue full: 1 requests pending")


class _QueueStubAdmit:
    async def enqueue(self, *args, **kwargs):
        fut = asyncio.Future()
        fut.set_result({"status": "ok"})
        return fut


class _QueueStubCount:
    def __init__(self):
        self.calls = []

    async def enqueue(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        fut = asyncio.Future()
        processor = kwargs.get("processor")
        if processor:
            try:
                result = processor()
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
        else:
            fut.set_result({"status": "ok"})
        return fut


@pytest.mark.unit
def test_queue_reject_returns_429(
    monkeypatch,
    execution_scoped_provider_credentials,
):
    # Patch TEST_MODE for deterministic auth/rate behavior
    monkeypatch.setenv("TEST_MODE", "true")

    with _test_db() as db, TestClient(app) as client:
        # CSRF token
        resp = client.get("/api/v1/health")
        client.csrf_token = resp.cookies.get("csrf_token", "")

        # Dependency overrides: DB and current user
        _app = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_request_queue",
            return_value=_QueueStubReject(),
        ):
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
            r = client.post("/api/v1/chat/completions", json=body, headers=_auth_headers(client))
            assert r.status_code == 429, f"Unexpected status: {r.status_code}, body={r.text}"


@pytest.mark.unit
def test_queue_admit_allows_request(
    monkeypatch,
    execution_scoped_provider_credentials,
):
    # Patch TEST_MODE for deterministic auth/rate behavior
    monkeypatch.setenv("TEST_MODE", "true")

    with _test_db() as db, TestClient(app) as client:
        # CSRF token
        resp = client.get("/api/v1/health")
        client.csrf_token = resp.cookies.get("csrf_token", "")

        _app = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        # Mock LLM response
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "This is a test response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.get_request_queue",
                return_value=_QueueStubAdmit(),
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value=mock_response,
            ),
        ):
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
            r = client.post("/api/v1/chat/completions", json=body, headers=_auth_headers(client))
            assert r.status_code == 200, f"Unexpected status: {r.status_code}, body={r.text}"
            data = r.json()
            assert isinstance(data, dict)
            assert data.get("choices")


@pytest.mark.unit
def test_queue_enabled_skips_admission(
    monkeypatch,
    execution_scoped_provider_credentials,
):
    monkeypatch.setenv("TEST_MODE", "true")
    queue_stub = _QueueStubCount()

    with _test_db() as db, TestClient(app) as client:
        resp = client.get("/api/v1/health")
        client.csrf_token = resp.cookies.get("csrf_token", "")

        _app = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Queued response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.QUEUED_EXECUTION", True),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.get_request_queue",
                return_value=queue_stub,
            ),
            patch(
                "tldw_Server_API.app.core.Chat.chat_service.get_request_queue",
                return_value=queue_stub,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value=mock_response,
            ),
        ):
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
            r = client.post("/api/v1/chat/completions", json=body, headers=_auth_headers(client))
            assert r.status_code == 200, f"Unexpected status: {r.status_code}, body={r.text}"
            assert len(queue_stub.calls) == 1
            assert queue_stub.calls[0]["kwargs"].get("processor") is not None
