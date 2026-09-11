import asyncio
import inspect
from typing import Any

import pytest

import tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter as moonshot_mod
import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl
from tldw_Server_API.app.core.LLM_Calls import chat_calls


@pytest.mark.unit
@pytest.mark.parametrize("api_name", [None, "", "   ", "none", "None"])
def test_analyze_returns_clear_error_when_api_name_missing(api_name: str | None) -> None:
    """Missing analysis providers return a clear error instead of leaking dispatch internals."""
    result = sgl.analyze(
        api_name=api_name,
        input_data="hello",
        custom_prompt_arg="summarize",
    )

    assert result == "Error: Analysis API provider is required."
    assert "NoneType" not in result
    assert "Error calling API" not in result


SECRET = "upstream-secret-body"


class _Adapter:
    def __init__(self, *, response=None, lines=None, error=None):
        self.response = response
        self.lines = lines
        self.error = error
        self.requests = []

    def chat(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        return self.response

    def stream(self, request, timeout=None):
        self.requests.append((request, timeout))
        for line in self.lines or []:
            if isinstance(line, BaseException):
                raise line
            yield line


class _Registry:
    def __init__(self, adapter):
        self.adapter = adapter

    def get_adapter(self, _provider):
        return self.adapter


def _disable_server_resolution(monkeypatch):
    monkeypatch.setattr(
        sgl,
        "ensure_app_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load config")),
    )
    monkeypatch.setattr(
        sgl,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load key")),
    )


def _provider_credentials(
    provider: str,
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
):
    """Issue an authentic capability for explicit summarization tests."""

    async def issue():
        async def resolver(
            normalized_provider: str,
            **_kwargs: Any,
        ) -> ResolvedByokCredentials:
            return ResolvedByokCredentials(
                provider=normalized_provider,
                api_key=api_key,
                app_config=app_config,
                credential_fields={},
                source="user",
                allowlisted=True,
                status=ByokResolutionStatus.RESOLVED,
                auth_source="api_key",
            )

        runtime = ProviderCredentialRuntime(
            user_id=29,
            team_ids=(),
            org_ids=(),
            trusted_base_url_override=True,
            server_config_snapshot={},
            resolver=resolver,
        )
        try:
            return await runtime.resolve(provider)
        finally:
            await runtime.close()

    return asyncio.run(issue())


class _FakeResp:
    def __init__(self, *, json_data=None, lines=None):
        self._json = json_data or {}
        self._lines = list(lines or [])
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield line

    def close(self):
        return None


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, *, json_data=None, lines=None):
        self._json_data = json_data or {}
        self._lines = list(lines or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        return None

    def post(self, *args, **kwargs):
        return _FakeResp(json_data=self._json_data)

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx(_FakeResp(lines=self._lines))


@pytest.mark.unit
def test_analyze_uses_adapter_non_stream(monkeypatch):
    fake_json = {"choices": [{"message": {"content": "summary"}}]}
    monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient(json_data=fake_json))

    result = sgl.analyze(
        api_name="openai",
        input_data="hello",
        custom_prompt_arg="summarize",
        api_key="x",
        system_message="system",
        temp=0.1,
        streaming=False,
        model_override="gpt-4o-mini",
    )

    assert result == "summary"


@pytest.mark.unit
def test_analyze_uses_adapter_stream(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient(lines=lines))

    result = sgl.analyze(
        api_name="openai",
        input_data="hello",
        custom_prompt_arg="summarize",
        api_key="x",
        system_message="system",
        temp=0.1,
        streaming=True,
        model_override="gpt-4o-mini",
    )

    assert inspect.isgenerator(result)
    assert "".join(list(result)) == "hello world"


@pytest.mark.unit
def test_explicit_nonstream_uses_exact_credentials(monkeypatch):
    adapter = _Adapter(response={"choices": [{"message": {"content": "summary"}}]})
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))
    _disable_server_resolution(monkeypatch)
    config = {"openai_api": {"model": "explicit-model", "api_timeout": 12}}

    result = sgl.analyze(
        "openai",
        "hello",
        "summarize",
        api_key="explicit-key",
        app_config=config,
        credentials_resolved=True,
        provider_credentials=_provider_credentials(
            "openai",
            api_key="explicit-key",
            app_config=config,
        ),
        raise_on_error=True,
    )

    assert result == "summary"
    request, timeout = adapter.requests[0]
    assert request["api_key"] == "explicit-key"
    assert request["app_config"] == config
    assert request["app_config"] is not config
    assert timeout == 12


@pytest.mark.unit
def test_explicit_stream_uses_exact_credentials(monkeypatch):
    adapter = _Adapter(
        lines=[
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))
    _disable_server_resolution(monkeypatch)

    result = sgl.analyze(
        "openai",
        "hello",
        None,
        api_key="stream-key",
        model_override="gpt-test",
        streaming=True,
        app_config={},
        credentials_resolved=True,
        provider_credentials=_provider_credentials(
            "openai",
            api_key="stream-key",
            app_config={},
        ),
        raise_on_error=True,
    )

    assert list(result) == ["hello"]
    assert adapter.requests[0][0]["api_key"] == "stream-key"


@pytest.mark.unit
def test_explicit_missing_key_raises_typed_sanitized_error(monkeypatch):
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(_Adapter()))
    _disable_server_resolution(monkeypatch)

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            "openai",
            "hello",
            None,
            api_key=" ",
            model_override="gpt-test",
            app_config={"openai_api": {"api_key": SECRET}},
            credentials_resolved=True,
            provider_credentials=_provider_credentials(
                "openai",
                api_key=" ",
                app_config={"openai_api": {"api_key": SECRET}},
            ),
            raise_on_error=True,
        )

    assert exc_info.value.code == "missing_credentials"
    assert exc_info.value.provider == "openai"
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
def test_adapter_error_raises_typed_error_without_upstream_body(monkeypatch):
    adapter = _Adapter(error=RuntimeError(SECRET))
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            "openai",
            "hello",
            None,
            api_key="key",
            model_override="gpt-test",
            app_config={},
            credentials_resolved=True,
            provider_credentials=_provider_credentials(
                "openai",
                api_key="key",
                app_config={},
            ),
            raise_on_error=True,
        )

    assert exc_info.value.code == "provider_failure"
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
def test_typed_chat_provider_error_is_sanitized(monkeypatch):
    adapter = _Adapter(error=ChatProviderError(message=SECRET, provider="openai"))
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            "openai",
            "hello",
            None,
            api_key="key",
            model_override="gpt-test",
            app_config={},
            credentials_resolved=True,
            provider_credentials=_provider_credentials(
                "openai",
                api_key="key",
                app_config={},
            ),
            raise_on_error=True,
        )

    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_error", "expected_code"),
    [
        (ChatAuthenticationError(message=SECRET, provider="openai"), "authentication"),
        (ChatConfigurationError(message=SECRET, provider="openai"), "configuration"),
        (ChatRateLimitError(message=SECRET, provider="openai"), "rate_limit"),
        (ChatProviderError(message=SECRET, status_code=403, provider="openai"), "authentication"),
        (ChatProviderError(message=SECRET, status_code=429, provider="openai"), "rate_limit"),
    ],
)
def test_chat_error_taxonomy_is_preserved_without_upstream_details(
    monkeypatch,
    adapter_error,
    expected_code,
):
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(_Adapter(error=adapter_error)))

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            "openai",
            "hello",
            None,
            api_key="key",
            model_override="gpt-test",
            app_config={},
            credentials_resolved=True,
            provider_credentials=_provider_credentials(
                "openai",
                api_key="key",
                app_config={},
            ),
            raise_on_error=True,
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.__cause__ is None
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
def test_dispatch_failure_is_typed_for_runtime_callers(monkeypatch):
    class BrokenRegistry:
        def get_adapter(self, _provider):
            raise ValueError(SECRET)

    monkeypatch.setattr(sgl, "get_registry", BrokenRegistry)

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            "openai",
            "hello",
            None,
            api_key="key",
            model_override="gpt-test",
            app_config={},
            credentials_resolved=True,
            provider_credentials=_provider_credentials(
                "openai",
                api_key="key",
                app_config={},
            ),
            raise_on_error=True,
        )

    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
def test_partial_stream_then_failure_raises_typed_error(monkeypatch):
    adapter = _Adapter(
        lines=[
            'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            RuntimeError(SECRET),
        ]
    )
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))
    result = sgl.analyze(
        "openai",
        "hello",
        None,
        api_key="key",
        model_override="gpt-test",
        streaming=True,
        app_config={},
        credentials_resolved=True,
        provider_credentials=_provider_credentials(
            "openai",
            api_key="key",
            app_config={},
        ),
        raise_on_error=True,
    )

    assert next(result) == "partial"
    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        next(result)
    assert exc_info.value.code == "provider_failure"
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
def test_partial_stream_authentication_failure_preserves_safe_taxonomy(monkeypatch):
    adapter = _Adapter(
        lines=[
            'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            ChatAuthenticationError(message=SECRET, provider="openai"),
        ]
    )
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))
    result = sgl.analyze(
        "openai",
        "hello",
        None,
        api_key="key",
        model_override="gpt-test",
        streaming=True,
        app_config={},
        credentials_resolved=True,
        provider_credentials=_provider_credentials(
            "openai",
            api_key="key",
            app_config={},
        ),
        raise_on_error=True,
    )

    assert next(result) == "partial"
    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        next(result)
    assert exc_info.value.code == "authentication"
    assert exc_info.value.__cause__ is None
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
def test_legacy_errors_remain_error_strings_and_chunks(monkeypatch):
    adapter = _Adapter(error=RuntimeError(SECRET), lines=[RuntimeError(SECRET)])
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))

    nonstream = sgl.analyze("openai", "hello", None, api_key="key", model_override="gpt-test")
    stream = sgl.analyze(
        "openai",
        "hello",
        None,
        api_key="key",
        model_override="gpt-test",
        streaming=True,
    )

    stream_error = list(stream)[0]
    assert nonstream.startswith("Error")
    assert stream_error.startswith("Error")
    assert SECRET not in nonstream
    assert SECRET not in stream_error


@pytest.mark.unit
def test_explicit_absent_config_cannot_trigger_summary_adapter_reload(monkeypatch):
    client = _FakeClient(json_data={"choices": [{"message": {"content": "summary"}}]})
    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda **_kwargs: client)
    monkeypatch.setattr(
        moonshot_mod,
        "load_and_log_configs",
        lambda: (_ for _ in ()).throw(AssertionError("adapter must not reload server config")),
    )

    class CopyingMoonshotAdapter(moonshot_mod.MoonshotAdapter):
        def chat(self, request, *, timeout=None):
            copied_request = {**request, "app_config": dict(request["app_config"])}
            return super().chat(copied_request, timeout=timeout)

    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(CopyingMoonshotAdapter()))

    result = sgl.analyze(
        "moonshot",
        "hello",
        None,
        api_key="explicit-key",
        model_override="moonshot-test",
        app_config=None,
        credentials_resolved=True,
        provider_credentials=_provider_credentials(
            "moonshot",
            api_key="explicit-key",
            app_config=None,
        ),
        raise_on_error=True,
    )

    assert result == "summary"


@pytest.mark.unit
def test_legacy_empty_config_still_allows_summary_adapter_reload(monkeypatch):
    client = _FakeClient(json_data={"choices": [{"message": {"content": "legacy"}}]})
    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda **_kwargs: client)
    monkeypatch.setattr(sgl, "loaded_config_data", {})
    monkeypatch.setattr(
        moonshot_mod,
        "load_and_log_configs",
        lambda: {"moonshot_api": {"model": "legacy-model"}},
    )

    result = sgl.analyze(
        "moonshot",
        "hello",
        None,
        api_key="legacy-key",
        model_override="legacy-model",
    )

    assert result == "legacy"


@pytest.mark.unit
def test_explicit_missing_summary_model_does_not_use_default_model_environment(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_MOONSHOT", "server-env-model")
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(_Adapter()))

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            "moonshot",
            "hello",
            None,
            api_key="explicit-key",
            app_config=None,
            credentials_resolved=True,
            provider_credentials=_provider_credentials(
                "moonshot",
                api_key="explicit-key",
                app_config=None,
            ),
            raise_on_error=True,
        )

    assert exc_info.value.code == "missing_model"
