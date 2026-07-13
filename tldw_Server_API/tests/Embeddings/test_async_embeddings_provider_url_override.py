import asyncio

import pytest

from tldw_Server_API.app.core.Embeddings import async_embeddings
from tldw_Server_API.app.core.Embeddings.async_embeddings import AsyncEmbeddingService
from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.Embeddings.connection_pool import get_pool_manager
from tldw_Server_API.app.core.Embeddings.simplified_config import (
    BatchingConfig,
    EmbeddingsConfig,
    ProviderConfig,
    SecurityConfig,
)


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
async def test_cache_key_includes_openai_api_url_override(monkeypatch):
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
    expected_key = f"openai:text-embedding-3-small:{text_hash}:https://example.test/v1"
    assert service.cache.get_keys[-1] == expected_key
    assert service.cache.set_keys[-1] == expected_key


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

    with pytest.raises(NetworkError):
        await service.create_embedding(
            "hello",
            provider="openai",
            use_cache=False,
            api_key_override="bad-key",
            credentials_resolved=True,
        )


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
