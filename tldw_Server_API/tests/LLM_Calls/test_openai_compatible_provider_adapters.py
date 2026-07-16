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
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self._captured["response_closed"] = True


class _FakeClient:
    def __init__(self, captured: dict):
        self._captured = captured

    def __enter__(self):
        self._captured["client_entered"] = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._captured["client_exited"] = True
        return False

    def post(self, url, headers=None, json=None):
        self._captured["url"] = url
        self._captured["headers"] = headers
        self._captured["json"] = json
        return _FakeResp(self._captured)

    def stream(self, method, url, headers=None, json=None):
        self._captured["stream_method"] = method
        self._captured["stream_url"] = url
        self._captured["stream_headers"] = headers
        self._captured["stream_json"] = json
        return _FakeResp(
            self._captured,
            lines=(b'data: {"choices": []}', b"data: [DONE]"),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_name", "base_env", "base_url", "expected_suffix"),
    [
        ("NovitaAdapter", "NOVITA_BASE_URL", "https://api.novita.ai/openai", "/openai/v1/chat/completions"),
        ("PoeAdapter", "POE_BASE_URL", "https://api.poe.com/v1", "/v1/chat/completions"),
        ("TogetherAdapter", "TOGETHER_BASE_URL", "https://api.together.xyz/v1", "/v1/chat/completions"),
    ],
)
def test_openai_compatible_provider_adapter_url_resolution(
    monkeypatch,
    adapter_name: str,
    base_env: str,
    base_url: str,
    expected_suffix: str,
):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setenv(base_env, base_url)
    captured = {}
    def _factory(*args, **kwargs):
        captured["factory_timeout"] = kwargs.get("timeout")
        return _FakeClient(captured)

    monkeypatch.setattr(adapter_module, "http_client_factory", _factory)

    adapter_cls = getattr(adapter_module, adapter_name)
    adapter = adapter_cls()
    adapter.http_fetcher = lambda **_kwargs: pytest.fail("public chat used configured fetcher")
    adapter.http_streamer = lambda **_kwargs: pytest.fail("public chat used configured streamer")

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
    assert captured["factory_timeout"] == 120.0
    assert captured["client_entered"] is True
    assert captured["client_exited"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_name", "base_env", "base_url"),
    [
        ("NovitaAdapter", "NOVITA_BASE_URL", "https://api.novita.ai/openai"),
        ("PoeAdapter", "POE_BASE_URL", "https://api.poe.com/v1"),
        ("TogetherAdapter", "TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
    ],
)
def test_public_openai_compatible_stream_uses_factory_not_configured_hooks(
    monkeypatch,
    adapter_name: str,
    base_env: str,
    base_url: str,
):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setenv(base_env, base_url)
    captured = {}

    def _factory(*args, **kwargs):
        captured["factory_timeout"] = kwargs.get("timeout")
        return _FakeClient(captured)

    monkeypatch.setattr(adapter_module, "http_client_factory", _factory)
    adapter = getattr(adapter_module, adapter_name)()
    adapter.http_fetcher = lambda **_kwargs: pytest.fail("public stream used configured fetcher")
    adapter.http_streamer = lambda **_kwargs: pytest.fail("public stream used configured streamer")

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
    assert captured["factory_timeout"] == 17.0
    assert captured["client_entered"] is True
    assert captured["client_exited"] is True
    assert captured["response_entered"] is True
    assert captured["response_exited"] is True
    assert captured["response_closed"] is True


@pytest.mark.unit
def test_configured_custom_uses_checked_hooks_not_public_factory(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module,
        "http_client_factory",
        lambda **_kwargs: pytest.fail("configured custom used public factory"),
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
