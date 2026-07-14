import asyncio
from functools import partial

import pytest
from loguru import logger

from tldw_Server_API.app.core.Embeddings import async_embeddings
from tldw_Server_API.app.core.Embeddings.async_embeddings import AsyncEmbeddingService
from tldw_Server_API.app.core.Embeddings.connection_pool import get_pool_manager
from tldw_Server_API.app.core.Embeddings.simplified_config import (
    BatchingConfig,
    EmbeddingsConfig,
    ProviderConfig,
    SecurityConfig,
)
from tldw_Server_API.app.core.exceptions import NetworkError

SECRET = "upstream-secret-body"


class DummyCache:
    async def get_async(self, key):  # noqa: ANN001 - simple test stub
        return None

    async def set_async(self, key, value, ttl=None):  # noqa: ANN001 - simple test stub
        return True


class TrackingCache:
    def __init__(self) -> None:
        self.get_keys = []
        self.set_keys = []

    async def get_async(self, key):  # noqa: ANN001 - simple test stub
        self.get_keys.append(key)
        return None

    async def set_async(self, key, value, ttl=None):  # noqa: ANN001 - simple test stub
        self.set_keys.append(key)
        return True


class HitCache:
    async def get_async(self, key):  # noqa: ANN001 - simple test stub
        return [0.7, 0.8]

    async def set_async(self, key, value, ttl=None):  # noqa: ANN001 - simple test stub
        raise AssertionError("cache hits must not be rewritten")


class SharedMemoryCache:
    def __init__(self) -> None:
        self.values = {}
        self.get_keys = []
        self.set_keys = []

    async def get_async(self, key):  # noqa: ANN001 - simple test stub
        self.get_keys.append(key)
        return self.values.get(key)

    async def set_async(self, key, value, ttl=None):  # noqa: ANN001 - simple test stub
        self.set_keys.append(key)
        self.values[key] = value
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "response_payload"),
    [
        (async_embeddings.AsyncOpenAIProvider(api_key="server-key"), {"data": [{"embedding": [0.1]}]}),
        (async_embeddings.AsyncHuggingFaceProvider(api_key="server-key"), [[0.1]]),
        (
            async_embeddings.AsyncLocalAPIProvider(
                api_url="https://configured.example/embeddings",
                api_key="server-key",
            ),
            {"embeddings": [[0.1]]},
        ),
    ],
    ids=("openai", "huggingface", "local-api"),
)
async def test_runtime_endpoint_marks_provider_pool_request_sensitive(
    monkeypatch,
    provider,
    response_payload,
):
    captured = []
    pool = provider.pool_manager.get_pool(provider.provider_name)

    async def fake_request(**kwargs):
        captured.append(kwargs)
        return response_payload

    monkeypatch.setattr(pool, "request", fake_request)

    result = await provider.create_embedding(
        "hello",
        model="embedding-model",
        api_key_override="runtime-key",
        base_url_override="https://runtime.private/secret/embeddings?tenant=one",
        credentials_resolved=True,
    )

    assert result == pytest.approx([0.1])
    assert captured[0]["sensitive_observability"] is True
    assert captured[0]["bypass_circuit_breaker"] is True


@pytest.mark.asyncio
async def test_configured_provider_endpoint_does_not_mark_pool_request_sensitive(monkeypatch):
    provider = async_embeddings.AsyncOpenAIProvider(
        api_key="server-key",
        base_url="https://configured.example/v1",
    )
    captured = []
    pool = provider.pool_manager.get_pool(provider.provider_name)

    async def fake_request(**kwargs):
        captured.append(kwargs)
        return {"data": [{"embedding": [0.1]}]}

    monkeypatch.setattr(pool, "request", fake_request)

    assert await provider.create_embedding("hello") == [0.1]
    assert captured[0].get("sensitive_observability", False) is False


@pytest.mark.asyncio
async def test_openai_api_url_override_only_when_explicit_provider(monkeypatch):
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                api_url="https://example.test/v1",
            )
        ],
        batching=BatchingConfig(enabled=True),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    service = AsyncEmbeddingService(config=config)
    service.cache = DummyCache()
    service.batcher.enabled = False
    service.batcher.enabled = True

    # Ensure batching is bypassed when explicit provider + api_url override is used.
    async def _fail_submit(*_args, **_kwargs):  # noqa: ANN001 - test stub
        raise AssertionError("batching should be bypassed for explicit provider overrides")

    monkeypatch.setattr(service.batcher, "submit_request", _fail_submit)

    pool = get_pool_manager().get_pool("openai")
    urls = []

    async def _fake_request(*_args, **kwargs):  # noqa: ANN001 - test stub
        urls.append(kwargs.get("url"))
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(pool, "request", _fake_request)

    result = await service.create_embedding(
        text="hello",
        model="text-embedding-3-small",
        provider="openai",
        use_cache=False,
        use_batching=True,
    )

    assert result == [0.1, 0.2]
    assert urls[-1] == "https://example.test/v1/embeddings"


@pytest.mark.asyncio
async def test_openai_api_url_used_for_default_provider(monkeypatch):
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                api_url="https://example.test/v1",
            )
        ],
        batching=BatchingConfig(enabled=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    service = AsyncEmbeddingService(config=config)
    service.cache = DummyCache()

    pool = get_pool_manager().get_pool("openai")
    urls = []

    async def _fake_request(*_args, **kwargs):  # noqa: ANN001 - test stub
        urls.append(kwargs.get("url"))
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(pool, "request", _fake_request)

    result = await service.create_embedding(
        text="hello",
        model="text-embedding-3-small",
        provider=None,
        use_cache=False,
        use_batching=False,
    )

    assert result == [0.1, 0.2]
    assert urls[-1] == "https://example.test/v1/embeddings"


@pytest.mark.asyncio
async def test_configured_endpoint_cache_key_uses_only_stable_digest(monkeypatch):
    endpoint = "https://configured-private.example/secret/path?tenant=private"
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                api_url=endpoint,
            )
        ],
        batching=BatchingConfig(enabled=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    service = AsyncEmbeddingService(config=config)
    service.cache = TrackingCache()

    pool = get_pool_manager().get_pool("openai")

    async def _fake_request(*_args, **kwargs):  # noqa: ANN001 - test stub
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(pool, "request", _fake_request)

    text = "hello"
    result = await service.create_embedding(
        text=text,
        model="text-embedding-3-small",
        provider="openai",
        use_cache=True,
        use_batching=False,
    )

    assert result == [0.1, 0.2]

    import hashlib

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    endpoint_digest = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
    expected_key = f"openai:text-embedding-3-small:{text_hash}:{endpoint_digest}"
    assert service.cache.get_keys[-1] == expected_key
    assert service.cache.set_keys[-1] == expected_key
    assert "configured-private.example" not in expected_key
    assert "secret/path" not in expected_key
    assert "tenant=private" not in expected_key


@pytest.mark.asyncio
async def test_runtime_credentials_bypass_shared_embedding_cache(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = SharedMemoryCache()
    dispatched_credentials: list[tuple[str, str]] = []

    async def provider_success(**kwargs):
        dispatched_credentials.append(
            (kwargs["api_key_override"], kwargs["base_url_override"])
        )
        return [float(len(dispatched_credentials))]

    monkeypatch.setattr(service.providers["openai"], "create_embedding", provider_success)
    endpoint = "https://runtime.private/secret/path"

    first = await service.create_embedding(
        "same text",
        provider="openai",
        api_key_override="runtime-key-a",
        base_url_override=endpoint,
        credentials_resolved=True,
    )
    second = await service.create_embedding(
        "same text",
        provider="openai",
        api_key_override="runtime-key-b",
        base_url_override=endpoint,
        credentials_resolved=True,
    )

    assert first == [1.0]
    assert second == [2.0]
    assert dispatched_credentials == [
        ("runtime-key-a", endpoint),
        ("runtime-key-b", endpoint),
    ]
    assert service.cache.get_keys == []
    assert service.cache.set_keys == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_local_api_calls_keep_same_endpoint_keys_isolated(monkeypatch):
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local_api",
                api_key="server-local-key",
                api_url="https://configured-local.example/embeddings",
                models=["local-model"],
                fallback_model="local-model",
            )
        ],
        batching=BatchingConfig(enabled=True),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local_api",
        default_model="local-model",
    )
    service = AsyncEmbeddingService(config=config)
    service.cache = TrackingCache()
    pool = get_pool_manager().get_pool("local_api")
    endpoint = "https://shared-local.example/embeddings"
    calls: list[tuple[str, str, str]] = []
    both_arrived = asyncio.Event()
    release = asyncio.Event()

    async def fake_request(*_args, **kwargs):
        authorization = kwargs["headers"].get("Authorization", "")
        text = kwargs["json_data"]["texts"][0]
        calls.append((authorization, kwargs["url"], text))
        if len(calls) == 2:
            both_arrived.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        marker = 1.0 if authorization == "Bearer key-alpha" else 2.0
        return {"embeddings": [[marker]]}

    monkeypatch.setattr(pool, "request", fake_request)
    tasks = [
        asyncio.create_task(
            service.create_embedding(
                label,
                provider="local_api",
                model="local-model",
                api_key_override=f"key-{label}",
                base_url_override=endpoint,
                credentials_resolved=True,
            )
        )
        for label in ("alpha", "beta")
    ]
    await asyncio.wait_for(both_arrived.wait(), timeout=5)
    release.set()
    assert await asyncio.gather(*tasks) == [[1.0], [2.0]]

    assert set(calls) == {
        ("Bearer key-alpha", endpoint, "alpha"),
        ("Bearer key-beta", endpoint, "beta"),
    }
    assert service.cache.get_keys == []
    assert service.cache.set_keys == []
    assert service.providers["local_api"].api_key == "server-local-key"


@pytest.mark.asyncio
async def test_legacy_local_api_cache_identity_includes_configured_endpoint(monkeypatch):
    shared_cache = SharedMemoryCache()
    calls = []

    def make_service(endpoint, marker):
        config = EmbeddingsConfig(
            providers=[
                ProviderConfig(
                    name="local_api",
                    api_url=endpoint,
                    models=["local-model"],
                )
            ],
            batching=BatchingConfig(enabled=False),
            security=SecurityConfig(enable_rate_limiting=False),
            default_provider="local_api",
            default_model="local-model",
        )
        service = AsyncEmbeddingService(config=config)
        service.cache = shared_cache

        async def provider_success(**_kwargs):
            calls.append(endpoint)
            return [marker]

        monkeypatch.setattr(service.providers["local_api"], "create_embedding", provider_success)
        return service

    first_service = make_service("https://local-a.example/embeddings", 1.0)
    second_service = make_service("https://local-b.example/embeddings", 2.0)

    first = await first_service.create_embedding(
        "same text",
        provider="local_api",
        use_batching=False,
    )
    second = await second_service.create_embedding(
        "same text",
        provider="local_api",
        use_batching=False,
    )

    assert first == [1.0]
    assert second == [2.0]
    assert calls == [
        "https://local-a.example/embeddings",
        "https://local-b.example/embeddings",
    ]
    assert shared_cache.get_keys[0] != shared_cache.get_keys[1]
    assert shared_cache.set_keys[0] != shared_cache.set_keys[1]
    assert "local-a.example" not in repr(shared_cache.get_keys)
    assert "local-b.example" not in repr(shared_cache.get_keys)


@pytest.mark.asyncio
async def test_configured_local_api_bypasses_global_batcher_for_each_endpoint(monkeypatch):
    shared_cache = SharedMemoryCache()
    direct_calls: list[str] = []
    batch_calls: list[tuple[str, str]] = []

    async def batch_submit(*, text, provider, **_kwargs):
        batch_calls.append((text, provider))
        return [99.0]

    def make_service(configured_name: str, endpoint: str, marker: float):
        config = EmbeddingsConfig(
            providers=[
                ProviderConfig(
                    name=configured_name,
                    api_url=endpoint,
                    models=["local-model"],
                )
            ],
            batching=BatchingConfig(enabled=True),
            security=SecurityConfig(enable_rate_limiting=False),
            default_provider=configured_name,
            default_model="local-model",
        )
        service = AsyncEmbeddingService(config=config)
        service.cache = shared_cache
        service.batcher.enabled = True
        monkeypatch.setattr(service.batcher, "submit_request", batch_submit)

        async def provider_success(**_kwargs):
            direct_calls.append(endpoint)
            return [marker]

        monkeypatch.setattr(service.providers["local_api"], "create_embedding", provider_success)
        return service

    first_service = make_service(
        "local_api",
        "https://local-a.private/embeddings?tenant=one",
        1.0,
    )
    second_service = make_service(
        "local",
        "https://local-b.private/embeddings?tenant=two",
        2.0,
    )

    first = await first_service.create_embedding("same text", use_batching=True)
    second = await second_service.create_embedding("same text", use_batching=True)

    assert first == [1.0]
    assert second == [2.0]
    assert direct_calls == [
        "https://local-a.private/embeddings?tenant=one",
        "https://local-b.private/embeddings?tenant=two",
    ]
    assert batch_calls == []


def _fallback_config():
    return EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="server-key",
                fallback_provider="huggingface",
                models=["text-embedding-3-small"],
            ),
            ProviderConfig(
                name="huggingface",
                api_key="fallback-key",
                models=["fallback-model"],
                fallback_model="fallback-model",
            ),
        ],
        batching=BatchingConfig(enabled=True),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )


@pytest.mark.asyncio
async def test_explicit_concurrent_calls_keep_keys_and_urls_isolated(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()
    pool = get_pool_manager().get_pool("openai")
    seen = []

    async def fake_request(*_args, **kwargs):
        await asyncio.sleep(0)
        seen.append((kwargs["headers"]["Authorization"], kwargs["url"]))
        marker = 1.0 if kwargs["headers"]["Authorization"] == "Bearer key-a" else 2.0
        return {"data": [{"embedding": [marker]}]}

    async def fail_batch(*_args, **_kwargs):
        raise AssertionError("explicit credential calls must bypass batching")

    monkeypatch.setattr(pool, "request", fake_request)
    monkeypatch.setattr(service.batcher, "submit_request", fail_batch)

    first, second = await asyncio.gather(
        service.create_embedding(
            "one",
            provider="openai",
            use_cache=False,
            api_key_override="key-a",
            base_url_override="https://a.example/v1",
            credentials_resolved=True,
        ),
        service.create_embedding(
            "two",
            provider="openai",
            use_cache=False,
            api_key_override="key-b",
            base_url_override="https://b.example/v1",
            credentials_resolved=True,
        ),
    )

    assert first == [1.0]
    assert second == [2.0]
    assert set(seen) == {
        ("Bearer key-a", "https://a.example/v1/embeddings"),
        ("Bearer key-b", "https://b.example/v1/embeddings"),
    }
    assert service.providers["openai"].api_key == "server-key"


@pytest.mark.asyncio
async def test_provider_success_callback_runs_after_cache_miss_dispatch(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()
    callback_calls = 0

    async def provider_success(**_kwargs):
        return [0.1, 0.2]

    async def record_success():
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(service.providers["openai"], "create_embedding", provider_success)

    result = await service.create_embedding(
        "hello",
        provider="openai",
        use_cache=True,
        api_key_override="runtime-key",
        credentials_resolved=True,
        on_provider_success=record_success,
    )

    assert result == [0.1, 0.2]
    assert callback_calls == 1


@pytest.mark.asyncio
async def test_provider_success_callback_does_not_run_for_cache_hit(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = HitCache()
    callback_calls = 0

    async def fail_provider(**_kwargs):
        raise AssertionError("cache hit must not dispatch provider")

    async def record_success():
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(service.providers["openai"], "create_embedding", fail_provider)

    result = await service.create_embedding(
        "hello",
        provider="openai",
        use_cache=True,
        on_provider_success=record_success,
    )

    assert result == [0.7, 0.8]
    assert callback_calls == 0


@pytest.mark.asyncio
async def test_provider_success_callback_does_not_run_for_provider_failure(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()
    callback_calls = 0

    async def provider_failure(**_kwargs):
        raise RuntimeError("provider failed")

    async def record_success():
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(service.providers["openai"], "create_embedding", provider_failure)

    with pytest.raises(async_embeddings.EmbeddingProviderError):
        await service.create_embedding(
            "hello",
            provider="openai",
            use_cache=False,
            api_key_override="runtime-key",
            credentials_resolved=True,
            on_provider_success=record_success,
        )

    assert callback_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_vector", [[], [float("nan")], ["not-numeric"]])
async def test_runtime_success_callback_rejects_malformed_vector_before_usage_or_cache(
    monkeypatch,
    malformed_vector,
):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = TrackingCache()
    callback_calls = 0

    async def malformed_success(**_kwargs):
        return malformed_vector

    async def record_success():
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(service.providers["openai"], "create_embedding", malformed_success)

    with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
        await service.create_embedding(
            "hello",
            provider="openai",
            use_cache=True,
            api_key_override="runtime-key",
            credentials_resolved=True,
            on_provider_success=record_success,
        )

    assert exc_info.value.code == "provider_failure"
    assert callback_calls == 0
    assert service.cache.set_keys == []


@pytest.mark.asyncio
async def test_legacy_call_without_success_callback_keeps_malformed_vector_compatibility(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()

    async def malformed_success(**_kwargs):
        return []

    monkeypatch.setattr(service.providers["openai"], "create_embedding", malformed_success)

    assert await service.create_embedding(
        "hello",
        provider="openai",
        use_cache=False,
        use_batching=False,
    ) == []


@pytest.mark.asyncio
async def test_batch_provider_success_callback_runs_once_per_cache_miss(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()
    callback_calls = 0

    async def provider_success(**_kwargs):
        return [0.1, 0.2]

    async def record_success():
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(service.providers["openai"], "create_embedding", provider_success)

    result = await service.create_embeddings_batch(
        ["one", "two"],
        provider="openai",
        api_key_override="runtime-key",
        credentials_resolved=True,
        on_provider_success=record_success,
    )

    assert result == [[0.1, 0.2], [0.1, 0.2]]
    assert callback_calls == 2


@pytest.mark.asyncio
async def test_provider_success_callback_drains_runtime_usage_on_cancellation(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
        ByokResolutionStatus,
        ResolvedByokCredentials,
    )
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        ProviderCredentialRuntime,
    )

    touch_started = asyncio.Event()
    release_touch = asyncio.Event()
    touch_completed = asyncio.Event()

    async def touch() -> None:
        touch_started.set()
        await release_touch.wait()
        touch_completed.set()

    async def resolver(provider: str, **_kwargs):
        return ResolvedByokCredentials(
            provider=provider,
            api_key="runtime-key",
            app_config={},
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
            _touch_cb=touch,
        )

    runtime = ProviderCredentialRuntime(
        user_id=41,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=True,
        fallback_resolver=lambda _provider: None,
        resolver=resolver,
    )
    handle = await runtime.resolve("openai")
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()

    async def provider_success(**_kwargs):
        return [0.1, 0.2]

    monkeypatch.setattr(service.providers["openai"], "create_embedding", provider_success)
    task = asyncio.create_task(
        service.create_embedding(
            "hello",
            provider="openai",
            use_cache=False,
            api_key_override="runtime-key",
            credentials_resolved=True,
            on_provider_success=partial(runtime.mark_used, handle),
        )
    )
    await touch_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    release_touch.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert touch_completed.is_set()
    await runtime.close()


@pytest.mark.asyncio
async def test_explicit_missing_key_never_falls_back(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()

    async def fail_fallback(*_args, **_kwargs):
        raise AssertionError("credential failure must not fall back")

    monkeypatch.setattr(service, "_try_fallback_providers", fail_fallback)

    with pytest.raises(async_embeddings.EmbeddingCredentialError):
        await service.create_embedding(
            "hello",
            provider="openai",
            use_cache=False,
            api_key_override=" ",
            credentials_resolved=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_explicit_auth_failure_never_falls_back(monkeypatch, status):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()
    pool = get_pool_manager().get_pool("openai")

    async def fail_request(*_args, **_kwargs):
        raise NetworkError(f"HTTP {status}: denied")

    async def fail_fallback(*_args, **_kwargs):
        raise AssertionError("authentication failure must not fall back")

    monkeypatch.setattr(pool, "request", fail_request)
    monkeypatch.setattr(service, "_try_fallback_providers", fail_fallback)

    with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
        await service.create_embedding(
            "hello",
            provider="openai",
            use_cache=False,
            api_key_override="bad-key",
            credentials_resolved=True,
        )

    assert exc_info.value.code == "authentication"
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "status"),
    [("openai", 401), ("huggingface", 403)],
)
async def test_explicit_error_payload_is_typed_sanitized_and_never_falls_back(
    monkeypatch,
    provider,
    status,
):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()
    pool = get_pool_manager().get_pool(provider)
    log_messages = []
    sink_id = logger.add(log_messages.append, format="{message}")

    async def error_payload(*_args, **_kwargs):
        return {"error": {"message": f"{SECRET} denied", "status": status}}

    async def fail_fallback(*_args, **_kwargs):
        raise AssertionError("explicit provider payload failure must not fall back")

    monkeypatch.setattr(pool, "request", error_payload)
    monkeypatch.setattr(service, "_try_fallback_providers", fail_fallback)

    try:
        with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
            await service.create_embedding(
                "hello",
                model="text-embedding-3-small" if provider == "openai" else "fallback-model",
                provider=provider,
                use_cache=False,
                api_key_override="bad-key",
                credentials_resolved=True,
            )
    finally:
        logger.remove(sink_id)

    assert exc_info.value.code == "authentication"
    assert exc_info.value.provider == provider
    assert exc_info.value.status_code == status
    assert exc_info.value.__cause__ is None
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)
    assert SECRET not in "".join(log_messages)


@pytest.mark.asyncio
async def test_legacy_network_failure_can_use_fallback(monkeypatch):
    service = AsyncEmbeddingService(config=_fallback_config())
    service.cache = DummyCache()

    async def primary_failure(**_kwargs):
        raise RuntimeError("network unavailable")

    async def fallback_success(**_kwargs):
        return [0.4, 0.5]

    monkeypatch.setattr(service.providers["openai"], "create_embedding", primary_failure)
    monkeypatch.setattr(service.providers["huggingface"], "create_embedding", fallback_success)

    result = await service.create_embedding("hello", provider="openai", use_cache=False, use_batching=False)

    assert result == [0.4, 0.5]


@pytest.mark.asyncio
async def test_runtime_local_batch_failure_never_uses_hosted_fallback(monkeypatch):
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local",
                fallback_provider="openai",
                models=["local-model"],
            ),
            ProviderConfig(
                name="openai",
                api_key="server-key-must-not-be-used",
                models=["hosted-model"],
                fallback_model="hosted-model",
            ),
        ],
        batching=BatchingConfig(enabled=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local",
        default_model="local-model",
    )
    service = AsyncEmbeddingService(config=config)
    service.cache = DummyCache()
    hosted_calls = 0

    async def local_failure(**_kwargs):
        raise RuntimeError("local provider failed with sensitive details")

    async def hosted_success(**_kwargs):
        nonlocal hosted_calls
        hosted_calls += 1
        return [0.4, 0.5]

    monkeypatch.setattr(service.providers["local"], "create_embedding", local_failure)
    monkeypatch.setattr(service.providers["openai"], "create_embedding", hosted_success)

    with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
        await service.create_embeddings_batch(
            ["hello"],
            provider="local",
            credentials_resolved=True,
        )

    assert exc_info.value.code == "provider_failure"
    assert exc_info.value.provider == "local"
    assert exc_info.value.__cause__ is None
    assert hosted_calls == 0


@pytest.mark.asyncio
async def test_explicit_local_api_requires_per_call_url(monkeypatch):
    singleton_url = "https://singleton-secret.example/embeddings"
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local_api",
                api_key="server-key",
                api_url=singleton_url,
            )
        ],
        batching=BatchingConfig(enabled=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local_api",
        default_model="local-model",
    )
    service = AsyncEmbeddingService(config=config)
    service.cache = DummyCache()
    pool = get_pool_manager().get_pool("local_api")

    async def fail_request(*_args, **_kwargs):
        raise AssertionError("singleton endpoint must not be contacted")

    monkeypatch.setattr(pool, "request", fail_request)

    with pytest.raises(async_embeddings.EmbeddingEndpointError) as exc_info:
        await service.create_embedding(
            "hello",
            provider="local_api",
            use_cache=False,
            api_key_override=None,
            base_url_override=None,
            credentials_resolved=True,
        )

    assert singleton_url not in str(exc_info.value)
    assert singleton_url not in repr(exc_info.value)
