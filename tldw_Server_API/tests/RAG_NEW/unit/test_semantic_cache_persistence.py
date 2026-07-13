from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache


pytestmark = pytest.mark.unit


class DummyEmbedder:
    async def embed(self, _text: str):
        return np.array([1.0, 0.0], dtype=float)


class CountingEmbedder(DummyEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        return await super().embed(text)


@pytest.mark.asyncio
async def test_semantic_cache_save_load_and_find_similar(tmp_path):
    cache_path = tmp_path / "semantic_cache.json"
    cache = SemanticCache(
        similarity_threshold=0.8,
        ttl=10,
        persist_path=str(cache_path),
        embedding_model=DummyEmbedder(),
        namespace="tenant-1",
    )

    await cache.set("alpha", {"answer": "A"}, ttl=10)
    _ = await cache.get("alpha")

    entry = list(cache._cache.values())[0]
    created_at = entry.created_at
    last_accessed = entry.last_accessed

    cache.save()

    cache_loaded = SemanticCache(
        similarity_threshold=0.8,
        ttl=10,
        persist_path=str(cache_path),
        embedding_model=DummyEmbedder(),
        namespace="tenant-1",
    )

    entry_loaded = list(cache_loaded._cache.values())[0]
    assert entry_loaded.created_at == pytest.approx(created_at)
    assert entry_loaded.last_accessed == pytest.approx(last_accessed)

    _key, cached_query, similarity = await cache_loaded.find_similar("beta")
    assert cached_query == "alpha"
    assert similarity >= 0.8


@pytest.mark.asyncio
async def test_semantic_cache_rejects_credential_handles_before_mutation(
    tmp_path: Path,
) -> None:
    secret = "sk-semantic-cache-sentinel-secret"
    cache_path = tmp_path / "semantic_cache.json"
    embedder = CountingEmbedder()
    cache = SemanticCache(
        persist_path=str(cache_path),
        embedding_model=embedder,
    )
    await cache.set("existing", {"answer": "safe"})
    cache.save()
    original_file = cache_path.read_bytes()
    original_keys = set(cache._cache)
    original_embedding_keys = set(cache._embeddings)
    original_stats = cache.get_stats()
    original_embed_calls = embedder.calls

    handle = ProviderCallCredentials(
        provider="openai",
        api_key=secret,
        app_config={"openai_api": {"organization": secret}},
        auth_source="api_key",
        runtime_generation=1,
        runtime_identity=object(),
        credential_identity=object(),
    )
    nested = {"payload": [("nested", {frozenset({handle})})]}
    cyclic: list[object] = []
    cyclic.extend((cyclic, nested))

    messages: list[str] = []
    sink_id = logger.add(messages.append)
    try:
        for query, value in (("direct", handle), ("nested", cyclic)):
            with pytest.raises(TypeError) as exc_info:
                await cache.set(query, value)
            assert str(exc_info.value) == ("ProviderCallCredentials cannot be serialized")
            assert secret not in repr(exc_info.value)
    finally:
        logger.remove(sink_id)

    assert set(cache._cache) == original_keys
    assert set(cache._embeddings) == original_embedding_keys
    assert cache.get_stats() == original_stats
    assert embedder.calls == original_embed_calls
    assert cache_path.read_bytes() == original_file
    assert secret not in "".join(messages)
    assert secret not in cache_path.read_text()

    await cache.set("later", {"answer": "still works"})
    cache.save()
    reloaded = SemanticCache(persist_path=str(cache_path))
    assert await reloaded.get("later") == {"answer": "still works"}
