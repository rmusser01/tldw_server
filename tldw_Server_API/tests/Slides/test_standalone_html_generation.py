from __future__ import annotations

import importlib
import json
from contextvars import ContextVar
from hashlib import sha256
from types import ModuleType
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.Slides.standalone_html_config import (
    CLOSED_ADAPTER_CATALOG,
    ResolvedExecutionTarget,
    ResolvedPrompt,
    SlidesStandaloneHtmlConfig,
    StandaloneHtmlInputLimits,
    StandaloneHtmlOutputLimits,
    StandaloneHtmlProviderLimits,
)

pytestmark = pytest.mark.unit


DOCUMENT = "<!doctype html><html><head><title>Deck</title></head><body></body></html>"
_TEST_TRANSPORT: ContextVar[httpx.AsyncBaseTransport | None] = ContextVar(
    "standalone_html_generation_test_transport",
    default=None,
)


@pytest.fixture
def provider_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    try:
        module = importlib.import_module("tldw_Server_API.app.core.Slides.standalone_html_provider")
    except ModuleNotFoundError:
        missing = ModuleType("standalone_html_provider_missing")

        async def fail_missing(**_kwargs: Any) -> bytes:
            pytest.fail("standalone_html_provider is not implemented")

        missing.generate_standalone_html = fail_missing
        return missing

    def isolated_test_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        assert "transport" not in kwargs
        transport = _TEST_TRANSPORT.get()
        if transport is not None:
            kwargs["transport"] = transport
        return httpx.AsyncClient(*args, **kwargs)

    monkeypatch.setattr(module, "_AsyncClient", isolated_test_client)
    return module


def _snapshot() -> SlidesStandaloneHtmlConfig:
    adapter = CLOSED_ADAPTER_CATALOG[0]
    target = ResolvedExecutionTarget(
        provider=adapter.provider,
        model="gpt-4o-mini",
        adapter_id=adapter.adapter_id,
        endpoint_identity=adapter.endpoint_identity,
    )
    system_prompt = "system prompt"
    prompt_bytes = system_prompt.encode()
    return SlidesStandaloneHtmlConfig(
        feature_enabled=True,
        egress_enabled=True,
        enabled=True,
        disabled_reason=None,
        target=target,
        prompt=ResolvedPrompt(
            text=system_prompt,
            sha256=sha256(prompt_bytes).hexdigest(),
            contract_version="slides.standalone_html.v1",
            byte_count=len(prompt_bytes),
        ),
        allowed_targets=(target,),
        input_limits=StandaloneHtmlInputLimits(
            max_request_bytes=4_194_304,
            max_source_chars=200_000,
            max_source_tokens=50_000,
            max_audience_chars=500,
            max_source_identifier_bytes=256,
            max_note_ids=100,
            max_rag_query_chars=20_000,
            max_rag_top_k=100,
        ),
        output_limits=StandaloneHtmlOutputLimits(8_388_608, 1_048_576),
        provider_limits=StandaloneHtmlProviderLimits(10.0, 120.0, 180.0, 16_384),
        generation_config_revision="sha256:" + ("0" * 64),
        _revision_manifest="{}",
    )


@pytest.mark.asyncio
async def test_one_normal_generation_attempt_makes_exactly_one_call_and_no_fallback(
    provider_module: ModuleType,
) -> None:
    config = _snapshot()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.dumps(
            {"choices": [{"message": {"content": DOCUMENT}}]},
            separators=(",", ":"),
        ).encode()
        return httpx.Response(200, stream=httpx.ByteStream(body))

    token = _TEST_TRANSPORT.set(httpx.MockTransport(handler))
    try:
        document = await provider_module.generate_standalone_html(
            stored_target=config.target,
            system_prompt=config.prompt.text,
            user_content="untrusted source",
            provider_api_key="provider-secret",
            current_config_loader=lambda: config,
        )
    finally:
        _TEST_TRANSPORT.reset(token)

    assert document == DOCUMENT.encode()
    assert len(calls) == 1
    assert calls[0].url.host == "api.openai.com"
    assert (calls[0].url.port or 443) == 443
    assert calls[0].url.path == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_failed_generation_call_is_not_retried_or_fallen_back(
    provider_module: ModuleType,
) -> None:
    config = _snapshot()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, stream=httpx.ByteStream(b"provider unavailable"))

    token = _TEST_TRANSPORT.set(httpx.MockTransport(handler))
    try:
        with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
            await provider_module.generate_standalone_html(
                stored_target=config.target,
                system_prompt=config.prompt.text,
                user_content="untrusted source",
                provider_api_key="provider-secret",
                current_config_loader=lambda: config,
            )
    finally:
        _TEST_TRANSPORT.reset(token)

    assert exc_info.value.code == "standalone_html_provider_http_error"
    assert len(calls) == 1
    assert calls[0].url.host == "api.openai.com"
