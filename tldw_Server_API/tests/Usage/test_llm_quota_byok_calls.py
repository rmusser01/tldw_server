"""A call made with the user's own provider key is recorded as such and never checked against the LLM quotas (TASK-13955)."""

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

pytestmark = pytest.mark.unit

REQUEST = {"model": "m1", "messages": [{"role": "user", "content": "hello"}]}
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
REFUSAL = LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)


class _Provider(ChatProvider):
    """A provider whose four call methods answer with a usage block."""

    name = "openai"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Answer with usage."""
        return {"choices": [{"message": {"content": "hi"}}], "usage": dict(USAGE)}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """One chunk with usage."""
        yield 'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Answer with usage."""
        return {"choices": [{"message": {"content": "hi"}}], "usage": dict(USAGE)}

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """One chunk with usage."""
        yield 'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'


async def _handle(source: str, provider: str = "openai") -> ProviderCallCredentials:
    """A runtime-issued credential handle for ``provider`` whose key comes from ``source``."""

    async def _resolver(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        """Resolve every provider to a key from ``source``."""
        return ResolvedByokCredentials(
            provider=provider, api_key="sk-test", app_config={}, credential_fields={}, source=source, allowlisted=True
        )

    runtime = ProviderCredentialRuntime(
        user_id=7, team_ids=[], org_ids=[], trusted_base_url_override=False,
        fallback_resolver=lambda _provider: None, server_config_snapshot={}, resolver=_resolver,
    )
    return await runtime.resolve(provider)


@pytest.fixture
async def refused(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Quotas on, user 7 over a limit, the test's loop as the accounting loop; returns the rows written."""
    rows: list[dict[str, Any]] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7."""
        return REFUSAL if user_id == 7 else None

    async def _fake_log(**kwargs: Any) -> None:
        """Keep the row instead of writing it."""
        rows.append(kwargs)

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    return rows


async def _call_all_four(request: dict[str, Any]) -> None:
    """chat, achat, a read stream and a read astream."""
    _Provider().chat(request)
    await _Provider().achat(request)
    list(_Provider().stream(request))
    [line async for line in _Provider().astream(request)]


async def test_a_runtime_handle_says_whose_key_it_carries() -> None:
    """The handle takes its source from the resolution; a user key is the user's own only where the vendor bills it."""
    assert (await _handle("user")).credential_source == "user"
    assert (await _handle("server_default")).credential_source == "server_default"
    for provider in ("custom-openai-api", "custom-openai-api-7", "llama.cpp"):
        assert (await _handle("user", provider)).credential_source == "user_unmetered"
        assert (await _handle("server_default", provider)).credential_source == "server_default"


@pytest.mark.parametrize(
    ("provider", "section", "expected"),
    [
        ("openai", {}, "user"),
        ("openai", {"api_base_url": "https://api.openai.com/v1"}, "user"),
        ("openai", {"api_base_url": "http://operator-gpu.internal:8000/v1"}, "user_unmetered"),
        ("qwen", {"api_base_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1"}, "user"),
        ("bedrock", {"runtime_endpoint": "https://bedrock-runtime.eu-west-1.amazonaws.com"}, "user"),
        ("bedrock", {"runtime_endpoint": "https://ec2-3-4-5-6.compute-1.amazonaws.com"}, "user_unmetered"),
        ("huggingface", {"router_base_url": "http://operator-gpu.internal:8000"}, "user_unmetered"),
        ("lmstudio", {}, "user_unmetered"),
        ("openai", {"api_base_url": "https://api.openai.com.evil.example/v1"}, "user_unmetered"),
        ("openai", {"api_base_url": "https://evil-api.openai.com/v1"}, "user_unmetered"),
        ("openai", {"api_base_url": "https://api.openai.com@evil.example/v1"}, "user_unmetered"),
        ("openai", {"api_base_url": "https://API.OpenAI.com:443/v1"}, "user"),
    ],
    ids=[
        "vendor-default", "vendor-url", "operator-url", "vendor-region", "bedrock-region", "bedrock-not-bedrock",
        "router-url", "unknown-provider", "suffix-lookalike", "prefix-lookalike", "userinfo", "case-and-port",
    ],
)
def test_a_user_key_is_the_users_own_spend_only_on_the_vendors_host(provider, section, expected) -> None:
    """Any base URL but the vendor's is an endpoint the operator chose, and an unknown provider is never billed."""
    from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source

    assert billed_credential_source(provider, "user", {f"{provider}_api": section}) == expected


def test_every_provider_that_bills_per_key_names_its_vendors_host() -> None:
    """A billed provider missing from the host table would count its users' own keys."""
    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import PROVIDER_REQUIRES_KEY
    from tldw_Server_API.app.core.Usage.llm_call_context import VENDOR_HOSTS

    assert set(VENDOR_HOSTS) == {name for name, billed in PROVIDER_REQUIRES_KEY.items() if billed}


async def test_calls_on_the_users_own_key_are_not_checked_and_are_recorded_as_such(refused) -> None:
    """An over-limit user's four calls on their own key all run, and each row says ``user``."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user")}
    with llm_call_context(user_id=7):
        await _call_all_four(request)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["user"] * 4


@pytest.mark.parametrize(
    ("source", "provider"),
    [("server_default", "openai"), ("team", "openai"), ("org", "openai"), ("user", "custom-openai-api")],
)
async def test_a_call_on_any_other_key_is_still_refused(refused, source, provider) -> None:
    """The same user's call on a server, shared or operator-endpoint key is refused before the provider runs."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle(source, provider)}
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError):
        await _Provider().achat(request)
    await llm_accounting.drain_accounting()
    assert refused == []


async def test_an_admitted_call_on_a_shared_key_is_recorded_with_its_source(refused) -> None:
    """A team key's call for an admitted user is recorded with ``team``, so it still counts."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("team")}
    with llm_call_context(user_id=8):
        await _Provider().achat(request)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["team"]


async def test_the_call_context_names_the_key_of_a_call_without_a_handle(refused) -> None:
    """A caller that resolved the key itself marks the context; its call runs and is recorded as ``user``."""
    with llm_call_context(user_id=7, credential_source="user"):
        await _Provider().achat(REQUEST)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["user"]


async def test_a_handle_the_runtime_did_not_issue_is_ignored(refused) -> None:
    """A look-alike object claiming ``user`` does not skip the check."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: SimpleNamespace(credential_source="user")}
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError):
        await _Provider().achat(request)


def test_a_request_that_cannot_be_read_is_an_unknown_source() -> None:
    """Reading the source never breaks a call: an unreadable request is unknown, so the call is checked."""
    from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source

    class _Unreadable(dict):
        """A request mapping whose lookups fail."""

        def get(self, *_args: Any) -> Any:
            """Fail."""
            raise RuntimeError("unreadable")

    with llm_call_context(credential_source="user"):
        assert call_credential_source(None) == "user"
        assert call_credential_source(_Unreadable()) is None


@pytest.mark.parametrize("layer", ["sync", "async"])
async def test_the_shared_call_layer_does_not_check_a_call_on_the_users_own_key(refused, monkeypatch, layer) -> None:
    """perform_chat_api_call (what /chat/completions runs) and its async twin read the handle before they check."""
    from tldw_Server_API.app.core.Chat import chat_service

    async def _call(**kwargs: Any) -> Any:
        """The layer under test; the sync one in a worker thread, where its check can run."""
        if layer == "async":
            return await chat_service.perform_chat_api_call_async(**kwargs)
        return await asyncio.to_thread(chat_service.perform_chat_api_call, **kwargs)

    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _name: _Provider()))
    monkeypatch.setattr(chat_service, "has_model_listing", lambda _provider: False)  # no model discovery request
    args = {
        "api_provider": "openai", "messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o-mini",
        "credentials_resolved": True, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    with llm_call_context(user_id=7):
        response = await _call(**args)
        with pytest.raises(LLMQuotaExceededError):
            await _call(**{**args, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("server_default")})
    assert response["usage"] == USAGE


@pytest.mark.parametrize(
    ("provider", "section", "field"),
    [
        ("custom-openai-api", "custom_openai_api", "api_ip"),
        ("qwen", "qwen_api", "api_base_url"),
        ("moonshot", "moonshot_api", "api_base_url"),
        ("zai", "zai_api", "api_base_url"),
        ("huggingface", "huggingface_api", "api_base_url"),
    ],
)
async def test_a_junk_key_stored_for_the_operators_endpoint_is_still_checked_and_counted(
    tmp_path, monkeypatch, provider: str, section: str, field: str
) -> None:
    """A key for a custom slot, or for a billed provider the operator pointed at its own server, is not the user's spend."""
    from datetime import datetime, timedelta, timezone

    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
    from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source
    from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import _setup_byok_sqlite
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_runtime_sqlite import _upsert_user_key

    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    await _upsert_user_key(AuthnzUserProviderSecretsRepo(state["pool"]), user_id, provider, "anything")
    runtime = ProviderCredentialRuntime(
        user_id=user_id, team_ids=[], org_ids=[], trusted_base_url_override=False,
        server_config_snapshot={section: {field: "http://operator-gpu.internal:8000/v1"}},
    )
    try:
        handle = await runtime.resolve(provider)
    finally:
        await runtime.close()
    source = call_credential_source({PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle})
    assert source == "user_unmetered"

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "1")
    quota_resolver.invalidate_all()
    with llm_call_context(user_id=user_id):
        assert llm_quota_gate.user_to_check(source) == user_id
    await log_llm_usage(
        user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider=provider,
        model="operator-llama", status=200, latency_ms=1, prompt_tokens=30, completion_tokens=10,
        request_id="operator-endpoint", credential_source=source,
    )
    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await AuthnzUsageRepo(state["pool"]).sum_user_llm_tokens_since(user_id=user_id, since=since) == 40.0


async def test_a_provider_method_nested_in_an_own_key_call_is_not_checked_again(refused) -> None:
    """A self-recording caller's own-key call that delegates without the handle is not refused by the delegate."""

    class _Delegating(_Provider):
        """chat() reads its own stream, as some adapters do."""

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Delegate to stream() with a request that no longer carries the handle."""
            inner = {key: value for key, value in request.items() if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY}
            return {"chunks": list(self.stream(inner))}

    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user")}
    with llm_call_context(user_id=7, records_itself=True):
        response = await asyncio.to_thread(_Delegating().chat, request)  # off the loop, where a check can run
    assert len(response["chunks"]) == 1


def _chat_request() -> SimpleNamespace:
    """A /chat/completions request for user 7."""
    return SimpleNamespace(
        method="POST", url=SimpleNamespace(path="/api/v1/chat/completions"), headers={},
        state=SimpleNamespace(user_id=7, api_key_id=None, team_ids=None, org_ids=None),
    )


class _Metrics:
    """Accepts every chat metrics call."""

    def __getattr__(self, _name: str) -> Any:
        """Any tracker is a no-op."""
        return lambda *_args, **_kwargs: None

    @asynccontextmanager
    async def track_streaming(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        """A stream tracker that ignores everything."""
        yield SimpleNamespace(add_heartbeat=lambda: None, add_chunk=lambda: None)


def _executor_kwargs(cleaned_args: dict[str, Any], llm_call_func: Any) -> dict[str, Any]:
    """The arguments both /chat/completions executors take."""
    from tldw_Server_API.tests.Chat.unit.test_streaming_structured_events import _NoModeration

    async def _save(_db: Any, _conversation_id: Any, _payload: Any, use_transaction: bool = True) -> str:
        """Pretend to save the reply."""
        return "m-1"

    return {
        "current_loop": asyncio.get_running_loop(), "cleaned_args": cleaned_args, "selected_provider": "openai",
        "provider": "openai", "model": "gpt-4o-mini", "request_json": "{}", "request": _chat_request(),
        "metrics": _Metrics(), "provider_manager": None,
        "templated_llm_payload": [{"role": "user", "content": "hi"}], "should_persist": False,
        "final_conversation_id": "conv-byok", "character_card_for_context": {"name": "Test"},
        "chat_db": SimpleNamespace(), "save_message_fn": _save, "audit_service": None, "audit_context": None,
        "client_id": "client", "queue_execution_enabled": False, "enable_provider_fallback": False,
        "llm_call_func": llm_call_func, "refresh_provider_params": lambda *_args, **_kwargs: ({}, None),
        "moderation_getter": lambda: _NoModeration(),
    }


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_completions_records_its_reply_with_the_source_of_its_key(monkeypatch, stream) -> None:
    """/chat/completions records its own reply, streamed or not, with the source of the handle it ran on."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.tests.Chat.unit.test_streaming_structured_events import _collect_sse_chunks

    logged: list[dict[str, Any]] = []

    async def _log(**kwargs: Any) -> None:
        """Keep the row."""
        logged.append(kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", _log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [{"role": "user", "content": "hi"}], "model": "gpt-4o-mini",
        "streaming": stream, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    if stream:
        response = await chat_service.execute_streaming_call(**_executor_kwargs(cleaned_args, lambda: iter(["hi"])))
        await _collect_sse_chunks(response)
    else:
        reply = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": USAGE}
        await chat_service.execute_non_stream_call(**_executor_kwargs(cleaned_args, lambda: reply))
    assert [row["credential_source"] for row in logged] == ["user"]


async def test_a_failover_reply_is_recorded_with_the_fallback_providers_key(monkeypatch) -> None:
    """The user's own key fails over to a server key: the reply that counts is the server key's."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    logged: list[dict[str, Any]] = []
    server_handle = await _handle("server_default")
    primary_error = ChatProviderError(provider="openai", message="primary unavailable", status_code=502)
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    async def _log(**kwargs: Any) -> None:
        """Keep the row."""
        logged.append(kwargs)

    def _fail_primary() -> None:
        """The user's provider is down before dispatch."""
        raise primary_error

    async def _fallback_call(**kwargs: Any) -> dict[str, Any]:
        """The fallback provider answers on the server's key."""
        assert kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is server_handle
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": USAGE}

    def _refresh(provider: str) -> tuple[dict[str, Any], str]:
        """The fallback provider's arguments, with the server key."""
        args = {"api_endpoint": provider, "messages_payload": [], "model": "claude-x", "streaming": False}
        return {**args, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: server_handle}, "claude-x"

    monkeypatch.setattr(chat_service, "log_llm_usage", _log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _fallback_call)
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [], "model": "gpt-4o-mini", "streaming": False,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    kwargs = _executor_kwargs(cleaned_args, _fail_primary)
    kwargs.update(
        enable_provider_fallback=True,
        refresh_provider_params=_refresh,
        provider_manager=SimpleNamespace(
            get_available_provider=lambda exclude=None: "anthropic",
            record_failure=lambda *_args: None,
            record_success=lambda *_args: None,
        ),
    )
    response = await chat_service.execute_non_stream_call(**kwargs)
    assert response["choices"][0]["message"]["content"] == "hi"
    assert [(row["provider"], row["credential_source"]) for row in logged] == [("anthropic", "server_default")]


async def test_a_streamed_failover_reply_is_recorded_with_the_fallback_providers_key(monkeypatch) -> None:
    """The user's own key fails over to a server key on a stream: the row that counts is the server key's."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.tests.Chat.unit.test_streaming_structured_events import _collect_sse_chunks

    logged: list[dict[str, Any]] = []
    server_handle = await _handle("server_default")
    primary_error = ChatProviderError(provider="openai", message="primary unavailable", status_code=502)
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    async def _log(**kwargs: Any) -> None:
        """Keep the row."""
        logged.append(kwargs)

    def _fail_primary() -> None:
        """The user's provider is down before dispatch."""
        raise primary_error

    def _fallback_call(**kwargs: Any) -> Iterable[str]:
        """The fallback provider streams on the server's key."""
        assert kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is server_handle
        return iter([
            'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
            "data: [DONE]\n\n",
        ])

    def _refresh(provider: str) -> tuple[dict[str, Any], str]:
        """The fallback provider's arguments, with the server key."""
        args = {"api_endpoint": provider, "messages_payload": [], "model": "claude-x", "streaming": True}
        return {**args, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: server_handle}, "claude-x"

    monkeypatch.setattr(chat_service, "log_llm_usage", _log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", _fallback_call)
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [], "model": "gpt-4o-mini", "streaming": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    kwargs = _executor_kwargs(cleaned_args, _fail_primary)
    kwargs.update(
        enable_provider_fallback=True,
        refresh_provider_params=_refresh,
        provider_manager=SimpleNamespace(
            get_available_provider=lambda exclude=None: "anthropic",
            record_failure=lambda *_args: None,
            record_success=lambda *_args: None,
        ),
    )
    response = await chat_service.execute_streaming_call(**kwargs)
    chunks = await _collect_sse_chunks(response)
    assert any('"hi"' in chunk for chunk in chunks)
    assert [(row["provider"], row["credential_source"]) for row in logged] == [("anthropic", "server_default")]


async def test_a_refused_failover_does_not_mark_the_fallback_provider_unhealthy(refused, monkeypatch) -> None:
    """An over-limit user's own key fails over to a server key: the 402 reaches the caller, and the fallback is not blamed."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    failures: list[str] = []
    server_handle = await _handle("server_default")
    primary_error = ChatProviderError(provider="openai", message="primary unavailable", status_code=502)
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    def _fail_primary() -> None:
        """The user's provider is down before dispatch."""
        raise primary_error

    def _refresh(provider: str) -> tuple[dict[str, Any], str]:
        """The fallback provider's arguments, with the server key."""
        args = {"api_endpoint": "openai", "messages_payload": [{"role": "user", "content": "hi"}], "model": "m"}
        return {**args, "streaming": False, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: server_handle}, "m"

    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _name: _Provider()))
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [], "model": "gpt-4o-mini", "streaming": False,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    kwargs = _executor_kwargs(cleaned_args, _fail_primary)
    kwargs.update(
        enable_provider_fallback=True,
        refresh_provider_params=_refresh,
        provider_manager=SimpleNamespace(
            get_available_provider=lambda exclude=None: "openai-fallback",
            record_failure=lambda name, *_args: failures.append(name),
            record_success=lambda *_args: None,
        ),
    )
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError) as refusal:
        await chat_service.execute_non_stream_call(**kwargs)
    assert refusal.value.detail["category"] == "llm_tokens_month"
    assert failures == ["openai"]


async def test_prompt_improvement_names_whose_key_its_call_runs_on(monkeypatch) -> None:
    """The dispatcher resolves the key itself, so it puts the key's source in the call context."""
    from tldw_Server_API.app.core.LLM_Calls.routing import InMemoryRoutingDecisionStore
    from tldw_Server_API.app.core.Prompt_Management import prompt_improvement_dispatch as dispatch
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.Prompt_Management_NEW.integration.test_prompt_improvement_layered import (
        _provider_listing,
        _request_context,
        _route_user,
    )

    seen: list[str | None] = []

    async def _touch() -> None:
        """Nothing to touch."""

    async def _resolve(_provider: str, **_kwargs: Any) -> SimpleNamespace:
        """The user's own key."""
        return SimpleNamespace(source="user", api_key="sk-user", app_config=None, touch_last_used=_touch)

    async def _perform(**_kwargs: Any) -> dict[str, Any]:
        """Note the call context's source and answer with no change."""
        seen.append(current_llm_call_context().credential_source)
        content = '{"status":"no_change","improved_text":null,"findings":[]}'
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(dispatch, "resolve_byok_credentials", _resolve)
    monkeypatch.setattr(dispatch, "perform_chat_api_call_async", _perform)
    result = await dispatch.dispatch_prompt_improvement(
        request=_request_context(), current_user=_route_user(), routing_decision_store=InMemoryRoutingDecisionStore(),
        selected_model="moonshot/kimi-k2", provider_hint=None, messages=[{"role": "user", "content": "draft"}],
        request_id="byok-dispatch", configured_providers_getter=_provider_listing,
    )
    assert (result.provider, result.model) == ("moonshot", "kimi-k2")
    assert seen == ["user"]


async def test_prompt_improvement_on_an_operator_base_url_from_the_server_config_counts_the_users_key(monkeypatch) -> None:
    """The raw-key dispatcher also reads the server config snapshot, where an environment base URL lands."""
    from tldw_Server_API.app.core.LLM_Calls.routing import InMemoryRoutingDecisionStore
    from tldw_Server_API.app.core.Prompt_Management import prompt_improvement_dispatch as dispatch
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.Prompt_Management_NEW.integration.test_prompt_improvement_layered import (
        _provider_listing,
        _request_context,
        _route_user,
    )

    seen: list[str | None] = []

    async def _touch() -> None:
        """Nothing to touch."""

    async def _resolve(_provider: str, **_kwargs: Any) -> SimpleNamespace:
        """The user's own key."""
        return SimpleNamespace(source="user", api_key="sk-user", app_config=None, touch_last_used=_touch)

    async def _perform(**_kwargs: Any) -> dict[str, Any]:
        """Note the call context's source and answer with no change."""
        seen.append(current_llm_call_context().credential_source)
        content = '{"status":"no_change","improved_text":null,"findings":[]}'
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(dispatch, "resolve_byok_credentials", _resolve)
    monkeypatch.setattr(dispatch, "perform_chat_api_call_async", _perform)
    monkeypatch.setattr(
        dispatch, "load_server_config_snapshot",
        lambda: {"moonshot_api": {"api_base_url": "http://operator-gpu.internal:8000/v1"}},
    )
    result = await dispatch.dispatch_prompt_improvement(
        request=_request_context(), current_user=_route_user(), routing_decision_store=InMemoryRoutingDecisionStore(),
        selected_model="moonshot/kimi-k2", provider_hint=None, messages=[{"role": "user", "content": "draft"}],
        request_id="byok-dispatch-operator", configured_providers_getter=_provider_listing,
    )
    assert (result.provider, result.model) == ("moonshot", "kimi-k2")
    assert seen == ["user_unmetered"]
