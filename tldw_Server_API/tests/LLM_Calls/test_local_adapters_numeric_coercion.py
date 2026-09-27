import pytest


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def close(self):
        return None


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider_name, cfg_section, url_key",
    [
        ("vllm", "vllm_api", "api_ip"),
        ("local-llm", "local_llm", "api_ip"),
        ("ooba", "ooba_api", "api_ip"),
        ("llama.cpp", "llama_api", "api_ip"),
        ("tabbyapi", "tabby_api", "api_ip"),
        ("aphrodite", "aphrodite_api", "api_ip"),
    ],
)
def test_local_like_adapters_coerce_numeric_types(provider_name, cfg_section, url_key):
    fake_settings = {
        cfg_section: {
            url_key: "http://localhost:1234/v1",  # openai-compatible path ok
            "streaming": False,
            "top_p": "0.9",
            "top_k": "50",
            "model": "dummy",
        }
    }

    captured_payload = {}

    def fake_fetch(*, json=None, **_kwargs):
        captured_payload.clear()
        if json:
            captured_payload.update(json)
        return DummyResponse(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry

    adapter = ChatProviderRegistry().get_adapter(provider_name)
    assert adapter is not None
    adapter.http_fetcher = fake_fetch
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "model": "dummy",
            "app_config": fake_settings,
        }
    )

    assert "top_p" in captured_payload and isinstance(captured_payload["top_p"], float)
    # top_k is not part of strict OpenAI spec but most local servers accept it; check when present
    if "top_k" in captured_payload:
        assert isinstance(captured_payload["top_k"], int)


@pytest.mark.unit
def test_kobold_coerces_numeric_types():
    from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import _kobold_request
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    fake_settings = {
        "kobold_api": {
            "api_ip": "http://localhost:5000/api/v1/generate",
            "streaming": False,
            "top_p": "0.92",
            "top_k": "80",
            "max_length": "128",
        }
    }

    captured_payload = {}
    captured_timeout = None
    lifecycle = {"status_checked": False, "closed": False}

    class Dummy:
        status_code = 200

        def raise_for_status(self):
            lifecycle["status_checked"] = True

        def json(self):
            return {"results": [{"text": "ok"}]}

        def close(self):
            lifecycle["closed"] = True

    def fake_fetch(method, url, headers=None, json=None, retry=None, **_kwargs):
        nonlocal captured_timeout
        captured_payload.clear()
        if json:
            captured_payload.update(json)
        captured_timeout = _kwargs.get("timeout")
        return Dummy()

    endpoint = fake_settings["kobold_api"]["api_ip"]
    _kobold_request(
        input_data=[{"role": "user", "content": "hello"}],
        streaming=False,
        app_config=fake_settings,
        http_fetcher=fake_fetch,
        configured_endpoint_base_url=endpoint,
        configured_endpoint_scope=ConfiguredEndpointScope.from_url(endpoint),
        credentials_resolved=True,
    )

    assert isinstance(captured_payload.get("top_p"), float)
    assert captured_payload.get("top_p") == 0.92
    assert isinstance(captured_payload.get("top_k"), int)
    assert captured_payload.get("top_k") == 80
    assert captured_timeout == 180
    assert lifecycle == {"status_checked": True, "closed": True}
