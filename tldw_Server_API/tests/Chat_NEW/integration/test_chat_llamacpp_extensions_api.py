"""Integration tests for llama.cpp chat-completions request extensions."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint_module


pytestmark = pytest.mark.integration


def _fake_chat_response(content: str = "Mocked response") -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def test_chat_completions_resolves_saved_grammar_into_llamacpp_payload(
    test_client,
    auth_headers,
    chacha_db,
    monkeypatch,
):
    grammar_id = chacha_db.insert_chat_grammar(
        {
            "name": "JSON output grammar",
            "grammar_text": 'root ::= "ok"',
        }
    )

    def override_get_db():
        return chacha_db

    captured: dict[str, object] = {}

    def _fake_provider_call(**kwargs):
        captured.update(kwargs)
        return _fake_chat_response(content="Grammar resolved.")

    test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    monkeypatch.setattr(chat_endpoint_module, "perform_chat_api_call", _fake_provider_call)

    try:
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "llama.cpp/local-model",
                "messages": [{"role": "user", "content": "Reply with ok"}],
                "save_to_db": False,
                "grammar_mode": "library",
                "grammar_id": grammar_id,
            },
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)

    assert response.status_code == 200
    assert captured["api_endpoint"] == "llama.cpp"
    assert captured["extra_body"] == {"grammar": 'root ::= "ok"'}


def test_chat_completions_rejects_llamacpp_fields_for_resolved_non_llamacpp_provider(
    credentialed_test_client,
    auth_headers,
):
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json={
            "api_provider": "openai",
            "model": "llama.cpp/local-model",
            "messages": [{"role": "user", "content": "Reply with ok"}],
            "save_to_db": False,
            "grammar_mode": "inline",
            "grammar_inline": 'root ::= "ok"',
        },
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request."


def test_chat_completions_refresh_params_reloads_saved_grammar_for_retry(
    test_client,
    auth_headers,
    chacha_db,
    monkeypatch,
):
    grammar_id = chacha_db.insert_chat_grammar(
        {
            "name": "Retry grammar",
            "grammar_text": 'root ::= "retry"',
        }
    )

    def override_get_db():
        return chacha_db

    lookup_count = {"calls": 0}
    original_get_chat_grammar = chacha_db.get_chat_grammar
    captured: dict[str, object] = {}

    def counting_get_chat_grammar(*args, **kwargs):
        lookup_count["calls"] += 1
        return original_get_chat_grammar(*args, **kwargs)

    async def _fake_execute_non_stream_call(**kwargs):
        refreshed_args, _ = await kwargs["refresh_provider_params"]("llama.cpp")
        captured["extra_body"] = refreshed_args.get("extra_body")
        return _fake_chat_response(content="Retry payload refreshed.")

    test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    monkeypatch.setattr(chacha_db, "get_chat_grammar", counting_get_chat_grammar)
    monkeypatch.setattr(chat_endpoint_module, "execute_non_stream_call", _fake_execute_non_stream_call)
    monkeypatch.setattr(chat_endpoint_module, "get_provider_manager", lambda: None)

    try:
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "llama.cpp/local-model",
                "messages": [{"role": "user", "content": "Reply with retry"}],
                "save_to_db": False,
                "grammar_mode": "library",
                "grammar_id": grammar_id,
            },
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)

    assert response.status_code == 200
    assert captured["extra_body"] == {"grammar": 'root ::= "retry"'}
    assert lookup_count["calls"] == 2
