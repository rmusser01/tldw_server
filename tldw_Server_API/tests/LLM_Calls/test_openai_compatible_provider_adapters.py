import pytest


class _FakeResp:
    status_code = 200

    def __init__(self, captured: dict, lines=()):
        self._captured = captured
        self._lines = lines

    def __enter__(self):
        self._captured["response_entered"] = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._captured["response_exited"] = True
        self.close()
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self._captured["response_closed"] = True


def _forbid_legacy_factory(*_args, **_kwargs):
    pytest.fail("public provider used the legacy client factory")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_name", "base_env", "base_url", "expected_suffix"),
    [
        ("NovitaAdapter", "NOVITA_BASE_URL", "https://api.novita.ai/openai", "/openai/v1/chat/completions"),
        ("PoeAdapter", "POE_BASE_URL", "https://api.poe.com/v1", "/v1/chat/completions"),
        ("TogetherAdapter", "TOGETHER_BASE_URL", "https://api.together.xyz/v1", "/v1/chat/completions"),
    ],
)
def test_public_openai_compatible_chat_uses_checked_fetch(
    monkeypatch,
    adapter_name: str,
    base_env: str,
    base_url: str,
    expected_suffix: str,
):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setenv(base_env, base_url)
    captured = {}

    def _fetch(**kwargs):
        captured.update(kwargs)
        return _FakeResp(captured)

    monkeypatch.setattr(
        adapter_module,
        "http_client_factory",
        _forbid_legacy_factory,
        raising=False,
    )

    adapter_cls = getattr(adapter_module, adapter_name)
    adapter = adapter_cls()
    adapter.http_fetcher = _fetch

    result = adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "api_key": "sk-test",
        }
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["url"].endswith(expected_suffix)
    assert captured["json"]["model"] == "test-model"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["configured_endpoint"] is None
    assert captured["allow_redirects"] is False
    assert captured["timeout"] == 120.0
    assert captured["response_closed"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_name", "base_env", "base_url"),
    [
        ("NovitaAdapter", "NOVITA_BASE_URL", "https://api.novita.ai/openai"),
        ("PoeAdapter", "POE_BASE_URL", "https://api.poe.com/v1"),
        ("TogetherAdapter", "TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
    ],
)
def test_public_openai_compatible_stream_uses_checked_streamer(
    monkeypatch,
    adapter_name: str,
    base_env: str,
    base_url: str,
):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setenv(base_env, base_url)
    captured = {}

    def _stream(**kwargs):
        captured.update(kwargs)
        return _FakeResp(
            captured,
            lines=(b'data: {"choices": []}', b"data: [DONE]"),
        )

    monkeypatch.setattr(
        adapter_module,
        "http_client_factory",
        _forbid_legacy_factory,
        raising=False,
    )
    adapter = getattr(adapter_module, adapter_name)()
    adapter.http_streamer = _stream

    chunks = list(
        adapter.stream(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "test-model",
            },
            timeout=17.0,
        )
    )

    assert chunks == ['data: {"choices": []}\n\n', "data: [DONE]\n\n"]
    assert captured["configured_endpoint"] is None
    assert captured["timeout"] == 17.0
    assert captured["response_entered"] is True
    assert captured["response_exited"] is True
    assert captured["response_closed"] is True


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_name", "base_env", "base_url"),
    [
        ("NovitaAdapter", "NOVITA_BASE_URL", "https://api.novita.ai/openai"),
        ("PoeAdapter", "POE_BASE_URL", "https://api.poe.com/v1"),
        ("TogetherAdapter", "TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
    ],
)
async def test_public_openai_compatible_async_modes_use_checked_hooks(
    monkeypatch,
    adapter_name: str,
    base_env: str,
    base_url: str,
):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setenv(base_env, base_url)
    calls = []
    lifecycle = {}

    def _fetch(**kwargs):
        calls.append(("chat", kwargs))
        return _FakeResp(lifecycle)

    def _stream(**kwargs):
        calls.append(("stream", kwargs))
        return _FakeResp(
            lifecycle,
            lines=(b'data: {"choices": []}', b"data: [DONE]"),
        )

    monkeypatch.setattr(
        adapter_module,
        "http_client_factory",
        _forbid_legacy_factory,
        raising=False,
    )
    adapter = getattr(adapter_module, adapter_name)()
    adapter.http_fetcher = _fetch
    adapter.http_streamer = _stream
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
    }

    result = await adapter.achat(request, timeout=19.0)
    chunks = [chunk async for chunk in adapter.astream(request, timeout=23.0)]

    assert result["choices"][0]["message"]["content"] == "ok"
    assert chunks == ['data: {"choices": []}\n\n', "data: [DONE]\n\n"]
    assert [kind for kind, _kwargs in calls] == ["chat", "stream"]
    assert all(kwargs["configured_endpoint"] is None for _kind, kwargs in calls)
    assert calls[0][1]["timeout"] == 19.0
    assert calls[0][1]["allow_redirects"] is False
    assert calls[1][1]["timeout"] == 23.0


@pytest.mark.unit
def test_configured_custom_uses_checked_hooks_not_public_factory(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module,
        "http_client_factory",
        lambda **_kwargs: pytest.fail("configured custom used public factory"),
        raising=False,
    )
    captured = {}
    calls = []

    def _fetch(**kwargs):
        calls.append(("chat", kwargs))
        return _FakeResp(captured)

    def _stream(**kwargs):
        calls.append(("stream", kwargs))
        return _FakeResp(
            captured,
            lines=(b'data: {"choices": []}', b"data: [DONE]"),
        )

    adapter = adapter_module.CustomOpenAIAdapter()
    adapter.http_fetcher = _fetch
    adapter.http_streamer = _stream
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "base_url": "http://request-endpoint:18097/v1",
        "_endpoint_provenance": "request_override",
    }

    assert adapter.chat(request)["choices"][0]["message"]["content"] == "ok"
    assert list(adapter.stream(request)) == [
        'data: {"choices": []}\n\n',
        "data: [DONE]\n\n",
    ]
    assert [kind for kind, _kwargs in calls] == ["chat", "stream"]
    assert all(kwargs["configured_endpoint"] is None for _kind, kwargs in calls)
    assert "allow_redirects" not in calls[0][1]
