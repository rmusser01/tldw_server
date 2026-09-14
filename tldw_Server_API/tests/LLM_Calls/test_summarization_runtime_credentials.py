from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_config import build_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
    reject_provider_call_credentials,
)
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl

pytestmark = pytest.mark.unit


async def _issue_credentials(
    provider: str,
    *,
    endpoint: str,
    api_key: str,
) -> ProviderCallCredentials:
    """Issue one authentic local/custom credential handle for a test call."""

    app_config = build_app_config_overrides(
        provider,
        {"base_url": endpoint},
    )

    async def resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=app_config,
            credential_fields={"base_url": endpoint},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
        )

    runtime = ProviderCredentialRuntime(
        user_id=41,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolver,
    )
    try:
        return await runtime.resolve(provider, model="snapshot-model")
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_local_and_custom_summaries_keep_exact_runtime_handles_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGL must not replace opaque request capabilities with loose credentials."""

    handles = {
        "local-llm": await _issue_credentials(
            "local-llm",
            endpoint="https://local-snapshot.example/v1",
            api_key="local-snapshot-key",
        ),
        "custom-openai-api-2": await _issue_credentials(
            "custom-openai-api-2",
            endpoint="https://custom-snapshot.example/v1",
            api_key="custom-snapshot-key",
        ),
    }
    gate = threading.Barrier(len(handles))
    capture_lock = threading.Lock()
    captured: list[
        tuple[str, ProviderCallCredentials, str | None, dict[str, Any]]
    ] = []

    class RecordingAdapter:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            gate.wait(timeout=5)
            with capture_lock:
                captured.append(
                    (
                        self.provider,
                        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY],
                        request["api_key"],
                        request["app_config"],
                    )
                )
            return {"choices": [{"message": {"content": self.provider}}]}

    adapters = {
        provider: RecordingAdapter(provider)
        for provider in handles
    }
    monkeypatch.setattr(
        sgl,
        "get_registry",
        lambda: type(
            "Registry",
            (),
            {
                "get_adapter": lambda _self, provider: adapters[provider],
                "is_local_provider_name": (
                    lambda _self, provider: provider == "local-llm"
                ),
            },
        )(),
    )

    def summarize(provider: str) -> str:
        return sgl.analyze(
            api_name=provider,
            input_data=f"summarize for {provider}",
            custom_prompt_arg=None,
            api_key="loose-attacker-key",
            model_override="snapshot-model",
            app_config={"attacker_api": {"api_key": "loose-attacker-key"}},
            credentials_resolved=True,
            provider_credentials=handles[provider],
            raise_on_error=True,
        )

    with ThreadPoolExecutor(max_workers=len(handles)) as executor:
        futures = {
            provider: executor.submit(summarize, provider)
            for provider in handles
        }
        results = {
            provider: future.result(timeout=10)
            for provider, future in futures.items()
        }

    assert results == {
        "local-llm": "local-llm",
        "custom-openai-api-2": "custom-openai-api-2",
    }
    assert {
        provider: handle
        for provider, handle, _api_key, _config in captured
    } == handles
    for provider, handle, api_key, durable_config in captured:
        assert handle is handles[provider]
        assert api_key == handles[provider].api_key
        assert durable_config == handles[provider].app_config
        reject_provider_call_credentials(durable_config)
        json.dumps(durable_config)


@pytest.mark.asyncio
async def test_runtime_handle_is_rejected_from_durable_summary_configuration() -> None:
    """Runtime capabilities may be attached only to the in-memory request envelope."""

    handle = await _issue_credentials(
        "custom-openai-api-2",
        endpoint="https://custom-snapshot.example/v1",
        api_key="custom-snapshot-key",
    )

    with pytest.raises(TypeError, match="cannot be serialized"):
        reject_provider_call_credentials(
            {"durable_llm_config": {"provider_credentials": handle}}
        )
    with pytest.raises(TypeError):
        json.dumps({"provider_credentials": handle})

    reject_provider_call_credentials(handle.app_config)
    json.dumps(handle.app_config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recursive_summarization", "chunked_summarization"),
    [(False, False), (True, False), (False, True)],
    ids=["direct", "recursive", "chunked"],
)
async def test_every_summary_mode_threads_exact_runtime_handle_to_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    recursive_summarization: bool,
    chunked_summarization: bool,
) -> None:
    """Nested summary calls must retain the original opaque capability."""

    handle = await _issue_credentials(
        "custom-openai-api-2",
        endpoint="https://custom-snapshot.example/v1",
        api_key="custom-snapshot-key",
    )
    captured: list[ProviderCallCredentials] = []

    def dispatch(*_args: Any, **kwargs: Any) -> str:
        captured.append(kwargs["provider_credentials"])
        return "summary"

    monkeypatch.setattr(sgl, "_dispatch_to_api", dispatch)
    monkeypatch.setattr(
        sgl,
        "improved_chunking_process",
        lambda _text, _options: [
            {"text": "first chunk"},
            {"text": "second chunk"},
        ],
    )

    result = sgl.analyze(
        api_name="custom-openai-api-2",
        input_data="source text",
        custom_prompt_arg=None,
        model_override="snapshot-model",
        recursive_summarization=recursive_summarization,
        chunked_summarization=chunked_summarization,
        provider_credentials=handle,
        raise_on_error=True,
    )

    assert result
    assert captured
    assert all(captured_handle is handle for captured_handle in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["forged", "provider-mismatch"])
async def test_summary_rejects_invalid_runtime_handle_before_adapter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """Only an authentic handle issued for the selected provider may dispatch."""

    if failure_kind == "forged":
        handle = ProviderCallCredentials(
            provider="local-llm",
            api_key="forged-key",
            app_config={"local_llm": {"api_ip": "https://forged.example/v1"}},
            auth_source="user",
            runtime_generation=0,
            runtime_identity=object(),
            credential_identity=object(),
        )
    else:
        handle = await _issue_credentials(
            "custom-openai-api-2",
            endpoint="https://custom-snapshot.example/v1",
            api_key="custom-snapshot-key",
        )

    monkeypatch.setattr(
        sgl,
        "get_registry",
        lambda: pytest.fail("invalid credentials must not reach the registry"),
    )

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(
            api_name="local-llm",
            input_data="source text",
            custom_prompt_arg=None,
            model_override="snapshot-model",
            provider_credentials=handle,
            raise_on_error=True,
        )

    assert exc_info.value.code == "configuration"
    assert exc_info.value.provider == "local-llm"
    assert "forged-key" not in repr(exc_info.value)
    assert "custom-snapshot-key" not in repr(exc_info.value)
