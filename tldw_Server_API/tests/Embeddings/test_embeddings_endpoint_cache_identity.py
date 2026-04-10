import pytest


class RecordingCache:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}
        self.get_keys: list[str] = []
        self.set_keys: list[str] = []

    async def get(self, key: str) -> object | None:
        self.get_keys.append(key)
        return self.data.get(key)

    async def set(self, key: str, value: object) -> None:
        self.set_keys.append(key)
        self.data[key] = value


def test_local_api_backend_identity_strips_credentials_and_sensitive_params():
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    clean_url = "http://backend-one.example:8080/v1"
    credentialed_url = "http://alice:s3cret@backend-one.example:8080/v1?token=abc#fragment"

    clean_identity = mod._normalize_cache_backend_identity({"api_url": clean_url}, "local_api")
    credentialed_identity = mod._normalize_cache_backend_identity({"api_url": credentialed_url}, "local_api")

    assert clean_identity == "http://backend-one.example:8080/v1"
    assert credentialed_identity == clean_identity
    assert "alice" not in credentialed_identity
    assert "s3cret" not in credentialed_identity
    assert "token" not in credentialed_identity

    clean_cache_key = mod.get_cache_key("cache me", "local_api", "test-model", backend_identity=clean_identity)
    credentialed_cache_key = mod.get_cache_key("cache me", "local_api", "test-model", backend_identity=credentialed_identity)

    assert credentialed_cache_key == clean_cache_key


def test_get_cache_key_uses_pbkdf2_hmac(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    calls = []
    secret = bytes.fromhex("ab" * 32)

    def fake_pbkdf2_hmac(algorithm, password, salt, iterations, dklen=None):
        calls.append((algorithm, password, salt, iterations, dklen))
        return bytes.fromhex("ab" * 32)

    monkeypatch.setattr(mod.hashlib, "pbkdf2_hmac", fake_pbkdf2_hmac)
    monkeypatch.setattr(mod, "_embedding_cache_key_secret", lambda: secret)

    key = mod.get_cache_key("cache me", "local_api", "test-model")

    assert key == "ab" * 32
    assert calls == [
        (
            "sha256",
            b"cache me|local_api|test-model",
            secret,
            mod._EMBEDDING_CACHE_KEY_PBKDF2_ITERATIONS,
            32,
        )
    ]


def test_local_api_backend_identity_preserves_nonsensitive_query_params():
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    url_tenant_a = "http://backend-one.example:8080/v1?tenant=a"
    url_tenant_b = "http://backend-one.example:8080/v1?tenant=b"
    url_no_query = "http://backend-one.example:8080/v1"

    identity_a = mod._normalize_cache_backend_identity({"api_url": url_tenant_a}, "local_api")
    identity_b = mod._normalize_cache_backend_identity({"api_url": url_tenant_b}, "local_api")
    identity_none = mod._normalize_cache_backend_identity({"api_url": url_no_query}, "local_api")

    # Non-sensitive query params produce distinct identities
    assert identity_a != identity_b
    assert identity_a != identity_none
    assert "tenant=a" in identity_a
    assert "tenant=b" in identity_b

    # Cache keys are therefore distinct
    key_a = mod.get_cache_key("cache me", "local_api", "test-model", backend_identity=identity_a)
    key_b = mod.get_cache_key("cache me", "local_api", "test-model", backend_identity=identity_b)
    assert key_a != key_b


def test_local_api_backend_identity_strips_sensitive_but_keeps_nonsensitive():
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    url_mixed = "http://backend.example:8080/v1?tenant=prod&token=secret123&region=us-east"

    identity = mod._normalize_cache_backend_identity({"api_url": url_mixed}, "local_api")

    assert "tenant=prod" in identity
    assert "region=us-east" in identity
    assert "token" not in identity
    assert "secret123" not in identity


@pytest.mark.asyncio
async def test_local_api_url_participates_in_cache_identity(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    cache = RecordingCache()
    api_urls: list[str | None] = []

    async def fake_create_embeddings_with_circuit_breaker(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        _ = (texts, provider, model_id, metadata, dimensions)
        api_url = config["api_url"]
        api_urls.append(api_url)
        marker = 1.0 if api_url == "http://backend-one/v1" else 2.0
        return [[marker, marker]]

    monkeypatch.setattr(mod, "embedding_cache", cache)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_create_embeddings_with_circuit_breaker)

    first = await mod.create_embeddings_batch_async(
        texts=["cache me"],
        provider="local_api",
        model_id="test-model",
        api_url="http://backend-one/v1",
    )
    second = await mod.create_embeddings_batch_async(
        texts=["cache me"],
        provider="local_api",
        model_id="test-model",
        api_url="http://backend-two/v1",
    )

    assert first == [[1.0, 1.0]]
    assert second == [[2.0, 2.0]]
    assert api_urls == ["http://backend-one/v1", "http://backend-two/v1"]
    assert len(cache.get_keys) == 2
    assert len(cache.set_keys) == 2
    assert cache.get_keys[0] != cache.get_keys[1]
    assert cache.set_keys[0] != cache.set_keys[1]
    assert cache.get_keys == cache.set_keys
