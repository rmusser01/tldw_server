import asyncio
import threading

import pytest


class _FakeResp:
    def __init__(self, status_code=200, json_obj=None, text="", lines=None):
        self.status_code = status_code
        self._json_obj = json_obj if json_obj is not None else {}
        self.text = text
        self._lines = list(lines or [])

    def json(self):
        return self._json_obj

    def raise_for_status(self):
        import requests

        if self.status_code and int(self.status_code) >= 400:
            err = requests.exceptions.HTTPError("HTTP error")
            err.response = self
            raise err
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, *, post_resp: _FakeResp | None = None, stream_lines=None):
        self._post_resp = post_resp
        self._stream_lines = list(stream_lines or [])
        self.last_json = None
        self.last_url = None
        self.last_headers = None
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers=None, json=None):
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        self.calls.append((url, headers, json))
        return self._post_resp or _FakeResp(status_code=200, json_obj={"ok": True})

    def stream(self, method, url, *, headers=None, json=None):
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        self.calls.append((url, headers, json))
        return _FakeResp(status_code=200, lines=self._stream_lines)


def test_dispatch_to_bedrock_adapter_non_stream(monkeypatch):
    # Patch adapter factory to avoid network
    from tldw_Server_API.app.core.LLM_Calls.providers import bedrock_adapter as mod

    fake = _FakeClient(
        post_resp=_FakeResp(status_code=200, json_obj={"choices": [{"message": {"content": "ok"}}]})
    )
    monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: fake)

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

    resp = perform_chat_api_call(
        api_provider="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        model="meta.llama3-8b-instruct",
        api_key="key",
        streaming=False,
    )
    assert isinstance(fake.last_json, dict)
    assert fake.last_json.get("stream") is False
    assert fake.last_url.endswith("/v1/chat/completions")


def test_dispatch_to_bedrock_adapter_stream(monkeypatch):
    # Patch adapter factory to provide streaming lines (no DONE marker)
    from tldw_Server_API.app.core.LLM_Calls.providers import bedrock_adapter as mod

    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        b'data: {"choices":[{"delta":{"content":" Bedrock"}}]}',
    ]
    fake = _FakeClient(stream_lines=lines)
    monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: fake)

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

    gen = perform_chat_api_call(
        api_provider="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        model="meta.llama3-8b-instruct",
        api_key="key",
        streaming=True,
    )
    chunks = list(gen)
    assert len(chunks) >= 3  # two chunks + DONE
    assert chunks[0].startswith("data: ")
    assert chunks[-1].strip().endswith("[DONE]")


@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_bedrock_api_key_is_exact_bearer_and_ignores_ambient_aws(monkeypatch, streaming):
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
    from tldw_Server_API.app.core.LLM_Calls.providers import bedrock_adapter as mod

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("BEDROCK_API_KEY", "ambient-bearer")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient-token")
    monkeypatch.setenv("BEDROCK_RUNTIME_ENDPOINT", "https://bedrock-runtime.us-west-2.amazonaws.com")
    fake = _FakeClient(
        post_resp=_FakeResp(status_code=200, json_obj={"choices": [{"message": {"content": "ok"}}]}),
        stream_lines=[b'data: {"choices":[{"delta":{"content":"ok"}}]}', b"data: [DONE]"],
    )
    monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: fake)

    response = perform_chat_api_call(
        api_provider="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        model="meta.llama3-8b-instruct",
        api_key="runtime-key",
        app_config={"bedrock_api": {"_runtime_auth_source": "api_key"}},
        credentials_resolved=True,
        streaming=streaming,
    )
    if streaming:
        chunks = list(response)
        assert sum(chunk.strip() == "data: [DONE]" for chunk in chunks) == 1

    assert fake.last_headers["Authorization"] == "Bearer runtime-key"
    assert not any(name.lower().startswith("x-amz-") for name in fake.last_headers)
    assert "ambient-bearer" not in repr(fake.calls)
    assert "ambient-token" not in repr(fake.calls)
    assert "ambient-secret" not in repr(fake.calls)


@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_bedrock_certified_default_chain_uses_sigv4(monkeypatch, streaming):
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
    from tldw_Server_API.app.core.LLM_Calls.providers import bedrock_adapter as mod

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "runtime-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "runtime-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "runtime-session")
    monkeypatch.setenv("BEDROCK_API_KEY", "must-not-use-bearer")
    monkeypatch.setenv("BEDROCK_RUNTIME_ENDPOINT", "https://bedrock-runtime.us-east-1.amazonaws.com")
    fake = _FakeClient(
        post_resp=_FakeResp(status_code=200, json_obj={"choices": [{"message": {"content": "ok"}}]}),
        stream_lines=[b'data: {"choices":[{"delta":{"content":"ok"}}]}', b"data: [DONE]"],
    )
    monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: fake)

    response = perform_chat_api_call(
        api_provider="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        model="meta.llama3-8b-instruct",
        api_key=None,
        app_config={"bedrock_api": {"_runtime_auth_source": "aws_default_chain"}},
        credentials_resolved=True,
        streaming=streaming,
    )
    if streaming:
        chunks = list(response)
        assert sum(chunk.strip() == "data: [DONE]" for chunk in chunks) == 1

    assert fake.last_headers["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=runtime-access/"
    )
    assert fake.last_headers["X-Amz-Security-Token"] == "runtime-session"
    assert "must-not-use-bearer" not in repr(fake.calls)


@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_bedrock_absent_auth_fails_before_http_despite_ambient_credentials(monkeypatch, streaming):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
    from tldw_Server_API.app.core.LLM_Calls.providers import bedrock_adapter as mod

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("BEDROCK_API_KEY", "ambient-bearer")
    fake = _FakeClient()
    monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: fake)

    with pytest.raises(ChatConfigurationError):
        response = perform_chat_api_call(
            api_provider="bedrock",
            messages=[{"role": "user", "content": "hi"}],
            model="meta.llama3-8b-instruct",
            api_key=None,
            app_config={"bedrock_api": {}},
            credentials_resolved=True,
            streaming=streaming,
        )
        if streaming:
            list(response)

    assert fake.calls == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_runtime_bedrock_calls_keep_auth_mode_base_and_payload_paired(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers import bedrock_adapter as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter import BedrockAdapter

    calls = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url, *, headers=None, json=None):
            with lock:
                calls.append(
                    {
                        "url": url,
                        "headers": dict(headers),
                        "label": json["messages"][0]["content"],
                    }
                )
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(5):
                raise TimeoutError("concurrent Bedrock calls did not release")
            return _FakeResp(status_code=200, json_obj={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: _GatedClient())
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "ambient-session")
    monkeypatch.setenv("BEDROCK_API_KEY", "ambient-bearer-must-not-cross")
    adapter = BedrockAdapter()
    tasks = [
        asyncio.create_task(
            adapter.achat(
                {
                    "messages": [{"role": "user", "content": "bearer"}],
                    "model": "meta.llama3-8b-instruct",
                    "api_key": "runtime-bearer-key",
                    "app_config": {
                        "bedrock_api": {
                            "api_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com/openai",
                            "_runtime_auth_source": "api_key",
                        }
                    },
                    "credentials_resolved": True,
                }
            )
        ),
        asyncio.create_task(
            adapter.achat(
                {
                    "messages": [{"role": "user", "content": "sigv4"}],
                    "model": "meta.llama3-8b-instruct",
                    "api_key": None,
                    "app_config": {
                        "bedrock_api": {
                            "api_base_url": "https://bedrock-runtime.us-west-2.amazonaws.com/openai",
                            "_runtime_auth_source": "aws_default_chain",
                        }
                    },
                    "credentials_resolved": True,
                }
            )
        ),
    ]

    assert await asyncio.to_thread(both_arrived.wait, 5)
    release.set()
    await asyncio.gather(*tasks)

    by_label = {call["label"]: call for call in calls}
    assert by_label["bearer"]["url"] == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"
    )
    assert by_label["bearer"]["headers"]["Authorization"] == "Bearer runtime-bearer-key"
    assert not any(
        name.lower().startswith("x-amz-")
        for name in by_label["bearer"]["headers"]
    )
    assert by_label["sigv4"]["url"] == (
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    )
    assert by_label["sigv4"]["headers"]["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=ambient-access/"
    )
    assert by_label["sigv4"]["headers"]["X-Amz-Security-Token"] == "ambient-session"
    assert "ambient-bearer-must-not-cross" not in repr(calls)
    assert "runtime-bearer-key" not in repr(by_label["sigv4"])
