import pytest

from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
    CustomOpenAIAdapter,
    CustomOpenAIAdapter2,
)


class _FakeResp:
    status_code = 200

    def __init__(self, captured: dict):
        self._captured = captured

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": []}

    def close(self):
        self._captured["closed"] = True


def _capture_fetch(captured: dict):
    def _fetch(**kwargs):
        captured.update(kwargs)
        return _FakeResp(captured)

    return _fetch


@pytest.mark.unit
def test_custom_openai_handler_accepts_topp(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        CustomOpenAIAdapter,
        "http_fetcher",
        staticmethod(_capture_fetch(captured)),
    )

    perform_chat_api_call(
        api_provider="custom-openai-api",
        messages=[{"role": "user", "content": "ping"}],
        api_key="test-key",
        model="test-model",
        topp=0.33,
        app_config={"custom_openai_api": {"api_ip": "http://byok-slot-1:18098/v1"}},
        _endpoint_provenance="byok",
    )

    assert captured["json"]["top_p"] == 0.33
    assert captured["configured_endpoint"] is None


@pytest.mark.unit
def test_custom_openai_handler_prefers_maxp_when_both_provided(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        CustomOpenAIAdapter,
        "http_fetcher",
        staticmethod(_capture_fetch(captured)),
    )

    perform_chat_api_call(
        api_provider="custom-openai-api",
        messages=[{"role": "user", "content": "ping"}],
        api_key="test-key",
        model="test-model",
        topp=0.12,
        maxp=0.45,
        app_config={"custom_openai_api": {"api_ip": "http://byok-slot-1:18098/v1"}},
        _endpoint_provenance="byok",
    )

    assert captured["json"]["top_p"] == 0.45
    assert captured["configured_endpoint"] is None


@pytest.mark.unit
def test_custom_openai_2_handler_accepts_topp(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        CustomOpenAIAdapter2,
        "http_fetcher",
        staticmethod(_capture_fetch(captured)),
    )

    perform_chat_api_call(
        api_provider="custom-openai-api-2",
        messages=[{"role": "user", "content": "ping"}],
        api_key="key-2",
        model="model-2",
        topp=0.27,
        app_config={"custom_openai_api_2": {"api_ip": "http://byok-slot-2:18099/v1"}},
        _endpoint_provenance="byok",
    )

    assert captured["json"]["top_p"] == 0.27
    assert captured["configured_endpoint"] is None


@pytest.mark.unit
def test_custom_openai_2_merges_extra_body_and_headers():
    captured = {}
    adapter = CustomOpenAIAdapter2()
    adapter.http_fetcher = _capture_fetch(captured)

    adapter.chat(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "model": "model-2",
            "base_url": "https://custom-openai-2.test/v1",
            "extra_headers": {"X-Test": "1"},
            "extra_body": {"custom_flag": True, "model": "override"},
            "_endpoint_provenance": "request_override",
        }
    )

    assert captured["headers"]["X-Test"] == "1"
    assert captured["json"]["custom_flag"] is True
    assert captured["json"]["model"] == "model-2"
    assert captured["configured_endpoint"] is None
