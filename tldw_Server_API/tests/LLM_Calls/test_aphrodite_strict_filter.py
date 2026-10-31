import pytest
from unittest.mock import patch, MagicMock

from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):

        return None

    def json(self):

        return {}

    def close(self):

        return None


@pytest.mark.unit
@pytest.mark.strict_mode
def test_aphrodite_strict_filter_drops_top_k_from_payload_non_streaming(monkeypatch):
    # Ensure helper takes the test codepath (raw httpx.Client) instead of central client
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    fake_settings = {
        "aphrodite_api": {
            "api_ip": "http://localhost:8082/v1/chat/completions",
            "streaming": False,
            "strict_openai_compat": True,
        }
    }

    captured_payload = {}

    def fake_request(_method, _url, **kwargs):
        json = kwargs.get("json")
        captured_payload.clear()
        if json:
            captured_payload.update(json)
        return DummyResponse({})

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.load_settings",
        return_value=fake_settings,
    ), patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters._hc_create_client"
    ) as mock_client_cls:

        mock_client = MagicMock()
        mock_client.request.side_effect = fake_request
        mock_client.close.return_value = None
        mock_client_cls.return_value = mock_client

        chat_api_call(
            api_endpoint="aphrodite",
            api_key=None,
            messages_payload=[{"role": "user", "content": "hello"}],
            topk=11,
            streaming=False,
        )

    assert "top_k" not in captured_payload
    assert "messages" in captured_payload
    assert "stream" in captured_payload
