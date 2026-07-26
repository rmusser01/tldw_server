import pytest
from fastapi import status
from types import SimpleNamespace


class _FakeRGLoader:
    def get_policy(self, _policy_id):
        return {}


class _FakeGovernor:
    def __init__(self):
        self.reserve_calls = []
        self.commit_calls = []

    async def reserve(self, req, op_id=None):
        self.reserve_calls.append((req, op_id))
        decision = SimpleNamespace(allowed=True, retry_after=None, details={})
        return decision, "handle-1"

    async def commit(self, handle_id, actuals=None, op_id=None):
        self.commit_calls.append((handle_id, actuals))


@pytest.mark.integration
def test_rg_refund_on_provider_error(monkeypatch, test_client, auth_headers):
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    fake_gov = _FakeGovernor()
    fake_loader = _FakeRGLoader()

    test_client.app.state.rg_governor = fake_gov
    test_client.app.state.rg_policy_loader = fake_loader

    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(chat_endpoint, "get_rate_limiter", lambda: None)

    async def fake_build_context_and_messages(
        chat_db,
        request_data,
        loop,
        metrics,
        default_save_to_db,
        final_conversation_id,
        save_message_fn,
    ):
        return (
            {"name": "Test", "system_prompt": ""},
            1,
            final_conversation_id or "conv-1",
            False,
            [{"role": "user", "content": "hi"}],
            False,
        )

    monkeypatch.setattr(chat_endpoint, "build_context_and_messages", fake_build_context_and_messages)

    def fail_call(**_kwargs):

        raise RuntimeError("boom")

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    try:
        response = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        assert response.status_code >= 500
    finally:
        test_client.app.state.rg_governor = None
        test_client.app.state.rg_policy_loader = None

    assert fake_gov.reserve_calls
    assert fake_gov.commit_calls
    handle_id, actuals = fake_gov.commit_calls[0]
    assert handle_id == "handle-1"
    assert actuals == {"tokens": 0}


@pytest.mark.integration
def test_rg_refund_on_streaming_error(monkeypatch, test_client, auth_headers):
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    fake_gov = _FakeGovernor()
    fake_loader = _FakeRGLoader()

    test_client.app.state.rg_governor = fake_gov
    test_client.app.state.rg_policy_loader = fake_loader

    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(chat_endpoint, "get_rate_limiter", lambda: None)
    monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: None)

    async def fake_build_context_and_messages(
        chat_db,
        request_data,
        loop,
        metrics,
        default_save_to_db,
        final_conversation_id,
        save_message_fn,
    ):
        return (
            {"name": "Test", "system_prompt": ""},
            1,
            final_conversation_id or "conv-1",
            False,
            [{"role": "user", "content": "hi"}],
            False,
        )

    monkeypatch.setattr(chat_endpoint, "build_context_and_messages", fake_build_context_and_messages)

    def fail_call(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    try:
        response = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        assert response.json()["detail"]["error_code"] == "provider_unavailable"
        assert "boom" not in response.text
    finally:
        test_client.app.state.rg_governor = None
        test_client.app.state.rg_policy_loader = None

    assert fake_gov.reserve_calls
    assert fake_gov.commit_calls
    handle_id, actuals = fake_gov.commit_calls[0]
    assert handle_id == "handle-1"
    assert actuals == {"tokens": 0}
