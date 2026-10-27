"""mlx must be accepted by /chat/completions (TASK-13950.1)."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ALL_SUPPORTED_PROVIDER_NAMES_LIST,
    ChatCompletionRequest,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.app.main import app

_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.unit
def test_chat_request_accepts_mlx():
    req = ChatCompletionRequest(api_provider="mlx", model="m", messages=_MESSAGES)
    assert req.api_provider == "mlx"


@pytest.mark.unit
def test_every_registered_chat_adapter_is_allowlisted():
    exempt: frozenset[str] = frozenset()
    missing = set(ChatProviderRegistry.DEFAULT_ADAPTERS) - exempt - set(ALL_SUPPORTED_PROVIDER_NAMES_LIST)
    assert missing == set()


@pytest.mark.unit
def test_allowlist_names_are_registered_adapters_or_custom_openai_aliases():
    from tldw_Server_API.app.core.custom_openai_providers import (
        iter_custom_openai_provider_names,
    )

    allowed_extra = set(iter_custom_openai_provider_names())
    registered = set(ChatProviderRegistry.DEFAULT_ADAPTERS)
    registered |= {a for aliases in ChatProviderRegistry.DEFAULT_ALIASES.values() for a in aliases}
    unknown = set(ALL_SUPPORTED_PROVIDER_NAMES_LIST) - registered - allowed_extra
    assert unknown == set()


@pytest.mark.unit
def test_chat_completions_dispatches_mlx_to_sync_seam(monkeypatch):
    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with TestClient(app) as client:
        csrf = client.get("/api/v1/health").cookies.get("csrf_token", "")
        client.csrf_token = csrf
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        api_key = get_settings().SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
        headers = {"X-API-KEY": api_key, "X-CSRF-Token": csrf}
        body = {"api_provider": "mlx", "model": "m", "save_to_db": False, "messages": _MESSAGES}
        with patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
            return_value=mock_response,
        ) as mocked:
            r = client.post("/api/v1/chat/completions", json=body, headers=headers)
    assert r.status_code == 200, r.text
    assert mocked.call_args.kwargs["api_endpoint"] == "mlx"
