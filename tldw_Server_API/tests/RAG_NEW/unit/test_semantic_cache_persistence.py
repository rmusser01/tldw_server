from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache
from tldw_Server_API.app.core.RAG.rag_service.types import Document

pytestmark = pytest.mark.unit


RETRIEVAL_CACHE_METADATA = {
    "kind": "retrieval_documents",
    "schema_version": 1,
}


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

    await cache.set(
        "alpha",
        {
            "documents": [{"id": "alpha-doc", "content": "retrieved evidence"}],
            "answer": "STALE_SENTINEL",
            "generated_answer": "STALE_SENTINEL",
            "metadata": {
                **RETRIEVAL_CACHE_METADATA,
                "generation_model": "stale-model",
            },
        },
        ttl=10,
    )
    cached = await cache.get("alpha")

    assert cached == {
        "documents": [
            {
                "id": "alpha-doc",
                "content": "retrieved evidence",
                "metadata": {},
                "score": 0.0,
            }
        ],
        "metadata": RETRIEVAL_CACHE_METADATA,
    }

    entry = list(cache._cache.values())[0]
    created_at = entry.created_at
    last_accessed = entry.last_accessed

    cache.save()

    persisted = json.loads(cache_path.read_text())
    persisted_value = next(iter(persisted["cache"].values()))["value"]
    assert persisted_value == cached
    assert "STALE_SENTINEL" not in cache_path.read_text()

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
async def test_semantic_cache_persists_strict_retrieval_only_document_metadata(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "semantic_cache.json"
    cache = SemanticCache(persist_path=str(cache_path))
    document = Document(
        id="strict-doc",
        content="strict evidence",
        score=float("nan"),
        metadata={
            "answer": "STALE_SENTINEL",
            "generation_provider": "STALE_SENTINEL",
            "retrieval_safe": {"rank": 1},
            "nested": {
                "generated_answer": "STALE_SENTINEL",
                "generation_model": "STALE_SENTINEL",
                "generation_prompt": "STALE_SENTINEL",
                "verification_report": {"answer": "STALE_SENTINEL"},
                "safe": "kept",
                "nan": float("nan"),
                "infinity": float("inf"),
            },
        },
    )

    await cache.set("strict query", {"documents": [document]})
    cache.save()

    raw = cache_path.read_text()
    assert "STALE_SENTINEL" not in raw
    assert "NaN" not in raw
    assert "Infinity" not in raw
    json.loads(raw, parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"))

    reloaded = SemanticCache(persist_path=str(cache_path))
    payload = await reloaded.get("strict query")
    assert payload["documents"] == [
        {
            "id": "strict-doc",
            "content": "strict evidence",
            "metadata": {
                "retrieval_safe": {"rank": 1},
                "nested": {"safe": "kept"},
                "source": "media_db",
            },
            "score": 0.0,
        }
    ]


@pytest.mark.asyncio
async def test_semantic_cache_loads_legacy_persisted_documents_without_answer(
    tmp_path: Path,
) -> None:
    query = "legacy persisted query"
    cache_path = tmp_path / "semantic_cache.json"
    key = SemanticCache()._generate_key(query)
    now = time.time()
    cache_path.write_text(
        json.dumps(
            {
                "cache": {
                    key: {
                        "value": {
                            "documents": [
                                {"id": "legacy-doc", "content": "legacy evidence"}
                            ],
                            "answer": "STALE_SENTINEL",
                            "generated_answer": "STALE_SENTINEL",
                            "metadata": {
                                "generation_provider": "stale-provider",
                                "verification_report": {"answer": "STALE_SENTINEL"},
                            },
                        },
                        "query": query,
                        "timestamp": now,
                        "ttl": 3600,
                        "access_count": 0,
                        "last_access": now,
                    }
                },
                "stats": {},
                "config": {},
            }
        )
    )

    cache = SemanticCache(persist_path=str(cache_path))

    assert await cache.get(query) == {
        "documents": [
            {
                "id": "legacy-doc",
                "content": "legacy evidence",
                "metadata": {},
                "score": 0.0,
            }
        ],
        "metadata": RETRIEVAL_CACHE_METADATA,
    }
    assert "STALE_SENTINEL" not in repr(cache._cache)


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
    existing_payload = {
        "documents": [{"id": "existing", "content": "safe"}],
        "metadata": RETRIEVAL_CACHE_METADATA,
    }
    await cache.set("existing", existing_payload)
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

    later_payload = {
        "documents": [{"id": "later", "content": "still works"}],
        "metadata": RETRIEVAL_CACHE_METADATA,
    }
    await cache.set("later", later_payload)
    cache.save()
    reloaded = SemanticCache(persist_path=str(cache_path))
    assert await reloaded.get("later") == {
        "documents": [
            {
                "id": "later",
                "content": "still works",
                "metadata": {},
                "score": 0.0,
            }
        ],
        "metadata": RETRIEVAL_CACHE_METADATA,
    }
