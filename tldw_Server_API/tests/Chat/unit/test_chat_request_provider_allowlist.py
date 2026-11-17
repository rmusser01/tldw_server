"""mlx must be accepted by /chat/completions (TASK-13950.1)."""

import os
import time
import traceback
from contextlib import suppress
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.config_llm_providers import ALL_SUPPORTED_PROVIDER_NAMES_LIST
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
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides

    read_count = 0
    load_failures = []
    list_overrides = llm_provider_overrides.AuthnzLLMProviderOverridesRepo.list_overrides

    async def observed_overrides(repo, *args, **kwargs):
        """Forward real reads and retain bounded, content-free failure evidence."""
        nonlocal read_count
        read_count += 1
        call = read_count
        started = time.monotonic()
        try:
            return await list_overrides(repo, *args, **kwargs)
        except BaseException as exc:
            if len(load_failures) < 3:
                load_failures.append(
                    {
                        "call": call,
                        "type": type(exc).__name__,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "frames": [
                            (frame.name, frame.lineno)
                            for frame in traceback.extract_tb(exc.__traceback__)[-8:]
                        ],
                    }
                )
            raise

    monkeypatch.setattr(
        llm_provider_overrides.AuthnzLLMProviderOverridesRepo, "list_overrides", observed_overrides
    )
    with TestClient(app) as client:
        csrf = client.get("/api/v1/health").cookies.get("csrf_token", "")
        client.csrf_token = csrf
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        api_key = get_settings().SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
        headers = {"X-API-KEY": api_key, "X-CSRF-Token": csrf}
        body = {"api_provider": "mlx", "model": "m", "save_to_db": False, "messages": _MESSAGES}
        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
            return_value=mock_response,
        ) as mocked:
            r = client.post("/api/v1/chat/completions", json=body, headers=headers)
            if r.status_code != 200:
                # Diagnostics must preserve the original response verdict.
                with suppress(Exception):
                    print(
                        "[mlx-override-read-failure]",
                        {"reads": read_count, "failures": load_failures, "status": r.status_code},
                    )
    assert r.status_code == 200, r.text
    assert mocked.call_args.kwargs["api_endpoint"] == "mlx"
