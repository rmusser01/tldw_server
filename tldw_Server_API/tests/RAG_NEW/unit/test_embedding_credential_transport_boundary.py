"""Credential-capability regressions for query-time OpenAI embeddings."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_config import build_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Embeddings import async_embeddings
from tldw_Server_API.app.core.Embeddings.Embeddings_Server import (
    Embeddings_Create as embeddings_create,
)
from tldw_Server_API.app.core.Embeddings.simplified_config import (
    BatchingConfig,
    EmbeddingsConfig,
    ProviderConfig,
    SecurityConfig,
)
from tldw_Server_API.app.core.LLM_Calls.openai_credentials import (
    OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG,
)
from tldw_Server_API.app.core.RAG.rag_service.hyde import (
    _resolve_runtime_embedding_call,
)


def _runtime(label: str) -> ProviderCredentialRuntime:
    fields = {
        "base_url": f"https://{label}.embedding.example/v1",
        "org_id": f"org-{label}",
        "project_id": f"project-{label}",
    }

    async def resolver(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=provider,
            api_key=f"key-{label}",
            app_config=build_app_config_overrides(provider, fields),
            credential_fields=dict(fields),
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
        )

    return ProviderCredentialRuntime(
        user_id=7,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolver,
    )


def _sync_config(tmp_path, provider: str = "openai") -> dict[str, Any]:
    model_name = (
        "text-embedding-3-small"
        if provider == "openai"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_id = f"{provider}:{model_name}"
    return {
        "embedding_config": {
            "default_model_id": model_id,
            "model_storage_base_dir": str(tmp_path),
            "models": {
                model_id: {
                    "provider": provider,
                    "model_name_or_path": model_name,
                }
            },
        }
    }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_sync_rag_embeddings_bind_capability_key_endpoint_and_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Loose fields cannot split an authentic A/B credential snapshot."""

    monkeypatch.setattr(
        embeddings_create,
        "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT",
        tmp_path.resolve(),
    )
    calls: list[tuple[str, str, str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

        def close(self) -> None:
            return None

    def fetch(**kwargs: Any) -> Response:
        headers = kwargs["headers"]
        with lock:
            calls.append(
                (
                    kwargs["url"],
                    headers["Authorization"],
                    headers["OpenAI-Organization"],
                    headers["OpenAI-Project"],
                    kwargs["json"]["input"][0],
                )
            )
            if len(calls) == 2:
                both_arrived.set()
        if not release.wait(5):
            raise TimeoutError("concurrent sync embeddings were not released")
        return Response()

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fetch)
    runtimes = [_runtime("alpha"), _runtime("beta")]
    try:
        resolved = await asyncio.gather(
            *(
                _resolve_runtime_embedding_call(
                    runtime,
                    "openai",
                    "text-embedding-3-small",
                )
                for runtime in runtimes
            )
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    embeddings_create.create_embeddings_batch,
                    [label],
                    _sync_config(tmp_path),
                    None,
                    **{
                        **call_kwargs,
                        "api_key_override": f"attacker-{label}-key",
                        "base_url_override": f"https://attacker-{label}.example/v1",
                        "credentials_resolved": True,
                    },
                )
                for label, (_handle, call_kwargs) in zip(
                    ("alpha", "beta"),
                    resolved,
                    strict=True,
                )
            ]
            try:
                assert await asyncio.to_thread(both_arrived.wait, 5)
            finally:
                release.set()
            assert all(future.result(timeout=5) == [[0.1, 0.2]] for future in futures)
    finally:
        await asyncio.gather(*(runtime.close() for runtime in runtimes))

    assert all(
        call_kwargs
        == {
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
            OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG: True,
        }
        for handle, call_kwargs in resolved
    )
    assert set(calls) == {
        (
            "https://alpha.embedding.example/v1/embeddings",
            "Bearer key-alpha",
            "org-alpha",
            "project-alpha",
            "alpha",
        ),
        (
            "https://beta.embedding.example/v1/embeddings",
            "Bearer key-beta",
            "org-beta",
            "project-beta",
            "beta",
        ),
    }
    assert "attacker-" not in repr(calls)


@pytest.mark.unit
def test_sync_openai_embedding_rejects_loose_resolved_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        embeddings_create,
        "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT",
        tmp_path.resolve(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch",
        lambda **_kwargs: pytest.fail("forged credentials must fail before transport"),
    )

    with pytest.raises(ChatConfigurationError):
        embeddings_create.create_embeddings_batch(
            ["forged"],
            _sync_config(tmp_path),
            api_key_override="attacker-key",
            base_url_override="https://attacker.example/v1",
            credentials_resolved=True,
            _require_provider_call_credentials=True,
        )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_async_rag_embeddings_bind_capability_key_endpoint_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async service preserves each opaque runtime capability to HTTP."""

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="server-key",
                api_url="https://server.embedding.example/v1",
            )
        ],
        batching=BatchingConfig(enabled=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )
    service = async_embeddings.AsyncEmbeddingService(config=config)
    pool = service.providers["openai"].pool_manager.get_pool("openai")
    calls: list[tuple[str, str, str, str, str]] = []
    lock = asyncio.Lock()
    both_arrived = asyncio.Event()
    release = asyncio.Event()

    async def request(**kwargs: Any) -> dict[str, Any]:
        headers = kwargs["headers"]
        async with lock:
            calls.append(
                (
                    kwargs["url"],
                    headers["Authorization"],
                    headers["OpenAI-Organization"],
                    headers["OpenAI-Project"],
                    kwargs["json_data"]["input"],
                )
            )
            if len(calls) == 2:
                both_arrived.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(pool, "request", request)
    runtimes = [_runtime("alpha"), _runtime("beta")]
    try:
        resolved = await asyncio.gather(
            *(
                _resolve_runtime_embedding_call(
                    runtime,
                    "openai",
                    "text-embedding-3-small",
                )
                for runtime in runtimes
            )
        )
        tasks = [
            asyncio.create_task(
                service.create_embedding(
                    label,
                    provider="openai",
                    model="text-embedding-3-small",
                    use_cache=False,
                    use_batching=False,
                    **{
                        **call_kwargs,
                        "api_key_override": f"attacker-{label}-key",
                        "base_url_override": f"https://attacker-{label}.example/v1",
                        "credentials_resolved": True,
                    },
                )
            )
            for label, (_handle, call_kwargs) in zip(
                ("alpha", "beta"),
                resolved,
                strict=True,
            )
        ]
        try:
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
        finally:
            release.set()
        assert await asyncio.gather(*tasks) == [[0.1, 0.2], [0.1, 0.2]]
    finally:
        await asyncio.gather(*(runtime.close() for runtime in runtimes))

    assert all(
        call_kwargs
        == {
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
            OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG: True,
        }
        for handle, call_kwargs in resolved
    )
    assert set(calls) == {
        (
            "https://alpha.embedding.example/v1/embeddings",
            "Bearer key-alpha",
            "org-alpha",
            "project-alpha",
            "alpha",
        ),
        (
            "https://beta.embedding.example/v1/embeddings",
            "Bearer key-beta",
            "org-beta",
            "project-beta",
            "beta",
        ),
    }
    assert "attacker-" not in repr(calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_openai_embedding_rejects_loose_resolved_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = async_embeddings.AsyncOpenAIProvider(api_key="server-key")
    pool = provider.pool_manager.get_pool("openai")

    async def fail_request(**_kwargs: Any) -> None:
        pytest.fail("forged credentials must fail before transport")

    monkeypatch.setattr(pool, "request", fail_request)

    with pytest.raises(ChatConfigurationError):
        await provider.create_embedding(
            "forged",
            api_key_override="attacker-key",
            base_url_override="https://attacker.example/v1",
            credentials_resolved=True,
            _require_provider_call_credentials=True,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_server_resolved_async_openai_override_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict RAG mode does not change the existing server-resolved path."""

    provider = async_embeddings.AsyncOpenAIProvider(api_key="static-server-key")
    pool = provider.pool_manager.get_pool("openai")
    captured: list[dict[str, Any]] = []

    async def request(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(pool, "request", request)

    result = await provider.create_embedding(
        "legacy",
        api_key_override="server-resolved-key",
        base_url_override="https://server-resolved.example/v1",
        credentials_resolved=True,
    )

    assert result == [0.1, 0.2]
    assert captured[0]["url"] == "https://server-resolved.example/v1/embeddings"
    assert captured[0]["headers"]["Authorization"] == "Bearer server-resolved-key"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_openai_inherited_batch_does_not_receive_runtime_only_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = async_embeddings.AsyncHuggingFaceProvider(api_key="server-key")
    pool = provider.pool_manager.get_pool("huggingface")

    async def request(**_kwargs: Any) -> list[list[float]]:
        return [[0.3, 0.4]]

    monkeypatch.setattr(pool, "request", request)

    result = await provider.create_embeddings_batch(
        ["first", "second"],
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    assert result[0] == pytest.approx([0.3, 0.4])
    assert result[1] == pytest.approx([0.3, 0.4])


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("use_handle", [True, False], ids=("handle", "required-flag"))
async def test_sync_non_openai_provider_rejects_openai_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    use_handle: bool,
) -> None:
    monkeypatch.setattr(
        embeddings_create,
        "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT",
        tmp_path.resolve(),
    )
    runtime = _runtime("mismatch")
    try:
        handle = await runtime.resolve("openai")
        runtime_kwargs = {
            (
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
                if use_handle
                else OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG
            ): handle if use_handle else True,
        }
        with pytest.raises(ChatConfigurationError):
            embeddings_create.create_embeddings_batch(
                ["mismatch"],
                _sync_config(tmp_path, "huggingface"),
                **runtime_kwargs,
            )
    finally:
        await runtime.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("use_handle", [True, False], ids=("handle", "required-flag"))
async def test_async_service_non_openai_provider_rejects_openai_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
    use_handle: bool,
) -> None:
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="huggingface",
                api_key="server-key",
                api_url="https://server.huggingface.example/models",
            )
        ],
        batching=BatchingConfig(enabled=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="huggingface",
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    service = async_embeddings.AsyncEmbeddingService(config=config)
    pool = service.providers["huggingface"].pool_manager.get_pool("huggingface")

    async def fail_request(**_kwargs: Any) -> None:
        pytest.fail("mismatched runtime credentials must fail before transport")

    monkeypatch.setattr(pool, "request", fail_request)
    runtime = _runtime("mismatch")
    try:
        handle = await runtime.resolve("openai")
        runtime_kwargs = {
            (
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
                if use_handle
                else OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG
            ): handle if use_handle else True,
        }
        with pytest.raises(ChatConfigurationError):
            await service.create_embedding(
                "mismatch",
                provider="huggingface",
                model="sentence-transformers/all-MiniLM-L6-v2",
                **runtime_kwargs,
            )
    finally:
        await runtime.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_transport_rejects_genuine_non_openai_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = async_embeddings.AsyncOpenAIProvider(api_key="server-key")
    pool = provider.pool_manager.get_pool("openai")

    async def fail_request(**_kwargs: Any) -> None:
        pytest.fail("provider-mismatched capability must fail before transport")

    monkeypatch.setattr(pool, "request", fail_request)
    runtime = _runtime("mismatch")
    try:
        huggingface_handle = await runtime.resolve("huggingface")
        with pytest.raises(ChatConfigurationError):
            await provider.create_embedding(
                "mismatch",
                _provider_call_credentials=huggingface_handle,
                _require_provider_call_credentials=True,
            )
    finally:
        await runtime.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_direct_non_openai_batch_rejects_runtime_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = async_embeddings.AsyncHuggingFaceProvider(api_key="server-key")
    pool = provider.pool_manager.get_pool("huggingface")

    async def fail_request(**_kwargs: Any) -> None:
        pytest.fail("runtime capability must not downgrade to static credentials")

    monkeypatch.setattr(pool, "request", fail_request)
    runtime = _runtime("mismatch")
    try:
        openai_handle = await runtime.resolve("openai")
        with pytest.raises(ChatConfigurationError):
            await provider.create_embeddings_batch(
                ["mismatch"],
                "sentence-transformers/all-MiniLM-L6-v2",
                _provider_call_credentials=openai_handle,
                _require_provider_call_credentials=True,
            )
    finally:
        await runtime.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_runtime_boundary_sanitizes_invalid_embedding_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    secret = "credential-metadata-must-not-leak"
    monkeypatch.setattr(
        embeddings_create,
        "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT",
        tmp_path.resolve(),
    )
    messages: list[str] = []
    sink = logger.add(messages.append, backtrace=True, diagnose=True)
    runtime = _runtime("sanitized")
    try:
        _handle, runtime_kwargs = await _resolve_runtime_embedding_call(
            runtime,
            "openai",
            "text-embedding-3-small",
        )
        with pytest.raises(ValueError, match="Invalid embedding_config structure") as exc_info:
            embeddings_create.create_embeddings_batch(
                ["query"],
                {"embedding_config": {"models": secret}},
                **runtime_kwargs,
            )
    finally:
        logger.remove(sink)
        await runtime.close()

    assert exc_info.value.__cause__ is None
    assert secret not in repr(exc_info.value)
    assert secret not in "".join(messages)
