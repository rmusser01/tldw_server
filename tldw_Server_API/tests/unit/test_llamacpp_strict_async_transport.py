"""The strict llama.cpp branch owns one cancellable checked HTTP request."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from starlette.requests import Request

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError, ChatProviderError, ChatRateLimitError
from tldw_Server_API.app.core.LLM_Calls import adapter_registry
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import LlamaCppAdapter
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


@asynccontextmanager
async def upstream(status=200, hold=False, answer=None):
    received = asyncio.Event()
    closed = asyncio.Event()
    payloads = []
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        inference = False
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            if header.startswith(b"GET "):
                # The shared gateway validates the explicit model before inference.
                data = json.dumps({"object": "list", "data": [{"id": "existing-model"}]}).encode()
                code = 200
            else:
                assert header.startswith(b"POST ")  # nosec B101 -- Local HTTP fixture admits only discovery and inference.
                inference = True
                length = next(
                    int(row.split(b":", 1)[1])
                    for row in header.split(b"\r\n")
                    if row.lower().startswith(b"content-length:")
                )
                payloads.append(json.loads(await reader.readexactly(length)))
                received.set()
                if hold:
                    await reader.read()
                    return
                data = json.dumps(
                    answer
                    if answer is not None
                    else {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
                ).encode()
                code = status
            writer.write(
                f"HTTP/1.1 {code} Test\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
                + data
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            if inference:
                closed.set()
            handlers.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", received, closed, payloads
    finally:
        server.close()
        await server.wait_closed()
        for task in tuple(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


def request_for(url):
    return {
        "model": "existing-model",
        "messages": [{"role": "user", "content": "Fictional sample"}],
        "system_message": "Classify.",
        "max_tokens": 128,
        "temperature": 0,
        "call_policy": ProviderCallPolicy(
            max_transport_attempts=1,
            allow_streaming=False,
            allow_tools=False,
            allow_stop=False,
            allow_response_format=False,
            candidate_count=1,
            maximum_timeout_seconds=2,
            required_endpoint_scope=ConfiguredEndpointScope.from_url(url),
        ),
    }


def bind_endpoint(monkeypatch, url):
    endpoint = TrustedProviderEndpoint(url, ConfiguredEndpointScope.from_url(url))
    monkeypatch.setattr(local_adapters, "resolve_trusted_provider_endpoint", lambda _: endpoint)


@pytest.mark.asyncio
async def test_strict_native_call_uses_real_one_shot_transport_without_sync_worker(monkeypatch):
    async with upstream() as (url, received, closed, payloads):
        bind_endpoint(monkeypatch, url)
        adapter = LlamaCppAdapter()
        monkeypatch.setattr(adapter, "chat", lambda *_, **__: pytest.fail("strict call entered synchronous worker"))
        answer = await adapter.achat(request_for(url), timeout=100)
        await asyncio.wait_for(closed.wait(), 2)
        assert len(payloads) == 1 and answer["choices"][0]["finish_reason"] == "stop"  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        assert payloads[0] == {  # nosec B101 -- Pytest checks classification, authority or resource retirement.
            "model": "existing-model",
            "messages": [{"role": "system", "content": "Classify."}, {"role": "user", "content": "Fictional sample"}],
            "max_tokens": 128,
            "temperature": 0,
            "n": 1,
            "stream": False,
        }


@pytest.mark.asyncio
async def test_strict_cancellation_retires_the_actual_owned_socket(monkeypatch):
    async with upstream(hold=True) as (url, received, closed, payloads):
        bind_endpoint(monkeypatch, url)
        task = asyncio.create_task(LlamaCppAdapter().achat(request_for(url)))
        await asyncio.wait_for(received.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), 2)
        assert len(payloads) == 1  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_strict_deadline_retires_socket_without_a_retry(monkeypatch):
    async with upstream(hold=True) as (url, received, closed, payloads):
        bind_endpoint(monkeypatch, url)
        request = request_for(url)
        request["call_policy"] = replace(request["call_policy"], maximum_timeout_seconds=0.15)
        with pytest.raises(ChatProviderError):
            await asyncio.wait_for(LlamaCppAdapter().achat(request), 2)
        await asyncio.wait_for(closed.wait(), 2)
        assert len(payloads) == 1  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_strict_upstream_error_does_not_retry(monkeypatch):
    async with upstream(status=429) as (url, received, closed, payloads):
        bind_endpoint(monkeypatch, url)
        with pytest.raises(ChatRateLimitError):
            await LlamaCppAdapter().achat(request_for(url))
        await asyncio.wait_for(closed.wait(), 2)
        assert len(payloads) == 1  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_scope_mismatch_is_refused_before_http(monkeypatch):
    async with upstream() as (url, received, closed, payloads):
        bind_endpoint(monkeypatch, url)
        request = request_for("http://127.0.0.1:1/v1")
        with pytest.raises(ChatConfigurationError):
            await LlamaCppAdapter().achat(request)
        assert not received.is_set()  # nosec B101 -- Pytest checks classification, authority or resource retirement.


def test_sync_strict_calls_refuse_instead_of_claiming_async_guarantees():
    with pytest.raises(ChatConfigurationError):
        LlamaCppAdapter().chat(request_for("http://127.0.0.1:1/v1"))


def test_certification_is_for_the_exact_adapter_only(monkeypatch):
    registry = adapter_registry.ChatProviderRegistry()
    monkeypatch.setattr(registry, "get_adapter", lambda _: LlamaCppAdapter())
    assert registry.get_audited_call_policy_transport("llama.cpp") is not None  # nosec B101 -- Pytest checks classification, authority or resource retirement.

    class UnreviewedSubclass(LlamaCppAdapter):
        pass

    monkeypatch.setattr(registry, "get_adapter", lambda _: UnreviewedSubclass())
    assert registry.get_audited_call_policy_transport("llama.cpp") is None  # nosec B101 -- Pytest checks classification, authority or resource retirement.


def test_neutral_payload_builder_preserves_local_gpt_named_model_token_bound():
    from tldw_Server_API.app.core.LLM_Calls.payload_utils import build_openai_chat_payload

    assert build_openai_chat_payload({"model": "gpt-5-local", "messages": [], "max_tokens": 128, "top_p": 0.7}) == {  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        "model": "gpt-5-local",
        "messages": [],
        "max_tokens": 128,
        "top_p": 0.7,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("native_unimplemented", [False, True])
async def test_classification_crosses_real_runtime_dispatch_adapter_and_http_once(tmp_path, monkeypatch, native_unimplemented):
    from uuid import uuid4

    from tldw_Server_API.app.api.v1.schemas.moderation_schemas import ModerationClassificationRequest
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionStatus, ResolvedByokCredentials
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
    from tldw_Server_API.app.core.LLM_Calls import gateway, usage_accounting
    from tldw_Server_API.app.core.Moderation import classification
    from tldw_Server_API.app.core.Moderation.review_service import ModerationReviewService

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    # Keep the real adapter instrumentation while isolating its best-effort DB writer.
    monkeypatch.setattr(usage_accounting, "submit_accounting", lambda _: None)
    async with upstream(
        answer={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "label": "flagged",
                                "reason": "An unpublished access code is present.",
                                "evidence": "ROWAN-731",
                            }
                        )
                    },
                }
            ]
        }
    ) as (url, received, closed, payloads):
        config = {"llama_api": {"api_ip": url}}

        async def resolve(provider, **_):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=None,
                app_config=config,
                credential_fields={},
                source="server",
                allowlisted=True,
                status=ByokResolutionStatus.ABSENT,
                auth_source="api_key",
            )

        runtime = ProviderCredentialRuntime(
            user_id=41,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            server_config_snapshot=config,
            resolver=resolve,
        )
        try:
            credentials = await runtime.resolve("llama.cpp", model="existing-model")
            registry = adapter_registry.ChatProviderRegistry()
            monkeypatch.setattr(gateway, "_get_llm_registry", lambda: registry)
            request = ModerationClassificationRequest(
                request_id=uuid4(),
                provider="llama.cpp",
                model="existing-model",
                criterion="Flag unpublished access codes.",
                text="Unpublished access code: ROWAN-731.",
            )
            review = ModerationReviewService(ModerationReviewStore(db_path=str(tmp_path / "review.db")))
            if native_unimplemented:
                fallbacks, native_calls = [], []

                async def unavailable(*_args, **_kwargs):
                    native_calls.append(True)
                    raise NotImplementedError

                def synchronous_fallback(*_args, **_kwargs):
                    fallbacks.append(True)
                    raise ChatConfigurationError(provider="llama.cpp", message="Unexpected synchronous fallback")

                monkeypatch.setattr(LlamaCppAdapter, "achat", unavailable)
                monkeypatch.setattr(LlamaCppAdapter, "chat", synchronous_fallback)
                with pytest.raises(classification.ClassificationError):
                    await classification.classify_sample(
                        request, actor_id="user:41", user_id=41, credentials=credentials,
                        review=review, credential_runtime=runtime,
                    )
                assert native_calls == [True] and fallbacks == [] and payloads == []  # nosec B101 -- An audited native failure never enters a sync worker or sends inference.
                assert review.list_items()["items"] == []  # nosec B101 -- A failed native call does not fabricate an advisory result.
                return
            item = await classification.classify_sample(
                request,
                actor_id="user:41",
                user_id=41,
                credentials=credentials,
                review=review,
                credential_runtime=runtime,
            )
            replay = await classification.classify_sample(
                request,
                actor_id="user:41",
                user_id=41,
                credentials=credentials,
                review=review,
                credential_runtime=runtime,
            )
            await asyncio.wait_for(closed.wait(), 2)
            assert (item["status"], item == replay, len(payloads)) == ("needs_review", True, 1)  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        finally:
            await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect_before_inference", [False, True])
async def test_raw_route_disconnect_retires_runtime_operation_monitor_and_real_socket(
    tmp_path, monkeypatch, disconnect_before_inference
):
    from types import SimpleNamespace
    from uuid import uuid4

    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.api.v1.endpoints import moderation
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionStatus, ResolvedByokCredentials
    from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
    from tldw_Server_API.app.core.LLM_Calls import usage_accounting
    from tldw_Server_API.app.core.Moderation.review_service import ModerationReviewService
    from tldw_Server_API.app.core.Slides.generation_disconnect import SlidesGenerationDisconnectMiddleware

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    monkeypatch.setattr(usage_accounting, "submit_accounting", lambda _: None)
    review = ModerationReviewService(ModerationReviewStore(db_path=str(tmp_path / "review.db")))
    actor = AuthPrincipal(  # nosec B106 -- 'access' is the token-kind enum, not a credential.
        kind="user",
        user_id=41,
        api_key_id=None,
        subject="user:41",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    async with upstream(hold=True) as (url, received, closed, payloads):
        config = {"llama_api": {"api_ip": url}}
        monkeypatch.setattr(moderation, "load_server_config_snapshot", lambda: config)

        async def resolve(provider, **_):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=None,
                app_config=config,
                credential_fields={},
                source="server",
                allowlisted=True,
                status=ByokResolutionStatus.ABSENT,
                auth_source="api_key",
            )

        runtimes = []

        class Runtime(ProviderCredentialRuntime):
            def __init__(self, **kwargs):
                kwargs.pop("override_snapshot_resolver")
                super().__init__(resolver=resolve, **kwargs)
                runtimes.append(self)

        monkeypatch.setattr(moderation, "ProviderCredentialRuntime", Runtime)
        app = FastAPI()
        app.add_middleware(SlidesGenerationDisconnectMiddleware)
        app.include_router(moderation.router, prefix="/api/v1")
        app.dependency_overrides[moderation.get_moderation_review_service] = lambda: review

        async def principal(request: Request):
            request.state.auth = AuthContext(principal=actor, ip=None, user_agent=None, request_id=None)
            return actor

        async def user():
            return SimpleNamespace(id=41, id_int=41)

        app.dependency_overrides[auth_deps.get_auth_principal] = principal
        app.dependency_overrides[auth_deps.get_request_user] = user
        body = json.dumps(
            {
                "request_id": str(uuid4()),
                "provider": "llama.cpp",
                "model": "existing-model",
                "criterion": "Flag unpublished access codes.",
                "text": "Unpublished access code: ROWAN-731.",
            }
        ).encode()
        messages = asyncio.Queue()
        messages.put_nowait({"type": "http.request", "body": body, "more_body": False})
        if disconnect_before_inference:
            messages.put_nowait({"type": "http.disconnect"})
        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/moderation/classify",
            "raw_path": b"/api/v1/moderation/classify",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "server": ("test", 80),
            "client": ("127.0.0.1", 1),
        }
        before = set(asyncio.all_tasks())
        task = asyncio.create_task(app(scope, messages.get, send))
        try:
            if not disconnect_before_inference:
                await asyncio.wait_for(received.wait(), 2)
                messages.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
            if not disconnect_before_inference:
                await asyncio.wait_for(closed.wait(), 2)
            assert (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
                [m["status"] for m in sent if m["type"] == "http.response.start"],
                len(payloads),
                review.list_items()["items"],
            ) == ([499], 0 if disconnect_before_inference else 1, [])
            assert len(runtimes) == 1 and runtimes[0]._closed  # nosec B101 -- Pytest checks classification, authority or resource retirement.
            assert not [t for t in set(asyncio.all_tasks()) - before if not t.done()]  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
