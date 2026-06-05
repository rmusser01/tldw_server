import pytest
from unittest.mock import patch

from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter2


@pytest.mark.unit
def test_custom_openai_handler_accepts_topp(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return FakeResp()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    ):
        perform_chat_api_call(
            api_provider="custom-openai-api",
            messages=[{"role": "user", "content": "ping"}],
            api_key="test-key",
            model="test-model",
            topp=0.33,
        )

    assert captured["json"]["top_p"] == 0.33


@pytest.mark.unit
def test_custom_openai_handler_prefers_maxp_when_both_provided(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return FakeResp()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    ):
        perform_chat_api_call(
            api_provider="custom-openai-api",
            messages=[{"role": "user", "content": "ping"}],
            api_key="test-key",
            model="test-model",
            topp=0.12,
            maxp=0.45,
        )

    assert captured["json"]["top_p"] == 0.45


@pytest.mark.unit
def test_custom_openai_2_handler_accepts_topp(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return FakeResp()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    ):
        perform_chat_api_call(
            api_provider="custom-openai-api-2",
            messages=[{"role": "user", "content": "ping"}],
            api_key="key-2",
            model="model-2",
            topp=0.27,
        )

    assert captured["json"]["top_p"] == 0.27


@pytest.mark.unit
def test_custom_openai_2_merges_extra_body_and_headers():
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    ):
        adapter = CustomOpenAIAdapter2()
        adapter.chat(
            {
                "messages": [{"role": "user", "content": "ping"}],
                "model": "model-2",
                "base_url": "https://custom-openai-2.test/v1",
                "extra_headers": {"X-Test": "1"},
                "extra_body": {"custom_flag": True, "model": "override"},
            }
        )

    assert captured["headers"]["X-Test"] == "1"
    assert captured["json"]["custom_flag"] is True
    assert captured["json"]["model"] == "model-2"
