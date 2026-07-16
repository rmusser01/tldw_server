from __future__ import annotations

from typing import Any, Dict


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: Dict[str, Any] | None = None):
        self.status_code = status_code
        self._json = json_obj or {"object": "chat.completion"}

    def raise_for_status(self):
        if 400 <= self.status_code:
            import httpx
            request = httpx.Request("POST", "http://example/v1/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._json

    def close(self):
        return None


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.last_post = None

    def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: int):
        self.last_post = {"url": url, "headers": headers, "json": json, "timeout": timeout}
        return _FakeResponse(200)

    def close(self):
        return None


def test_local_adapter_merges_extra_body_and_headers(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import LocalLLMAdapter

    captured: Dict[str, Any] = {}

    def _fetch(**kwargs: Any):
        captured["request"] = kwargs
        return _FakeResponse(200)

    adapter = LocalLLMAdapter()
    adapter.http_fetcher = _fetch
    monkeypatch.setenv("LOCAL_LLM_API_URL", "http://example")
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "dummy",
        "temperature": 0.1,
        "extra_body": {"temperature": 0.9, "x_extra": "y"},
        "extra_headers": {
            "Authorization": "Bearer override",
            "content-type": "text/plain",
            "X-Test": "1",
        },
        "app_config": {
            "local_llm": {
                "api_ip": "http://example",
                "api_key": "k",
                "model": "dummy",
            }
        },
    }
    _ = adapter.chat(request)
    payload = captured["request"]["json"]
    headers = captured["request"]["headers"]
    assert payload.get("temperature") == 0.1
    assert payload.get("x_extra") == "y"
    assert headers.get("Authorization") == "Bearer k"
    assert headers.get("Content-Type") == "application/json"
    assert headers.get("X-Test") == "1"
    assert "content-type" not in headers
    assert captured["request"]["configured_endpoint"].matches("http://example/v1/chat/completions")
