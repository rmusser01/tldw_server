import pytest

from tldw_Server_API.app.core.RAG.rag_service import semantic_cache
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache

pytestmark = pytest.mark.unit


def test_semantic_cache_stats_include_namespace():


    cache = SemanticCache(similarity_threshold=0.9, ttl=10, namespace="tenant-123")
    stats = cache.get_stats()
    assert stats.get("namespace") == "tenant-123"


def test_shared_cache_instances_are_isolated_by_namespace(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache_root"
    cache_root.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    semantic_cache._SHARED_CACHES.clear()

    cache_tenant_a = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-a",
    )
    cache_tenant_b = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-b",
    )
    cache_tenant_a_again = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-a",
    )

    assert cache_tenant_a is cache_tenant_a_again
    assert cache_tenant_a is not cache_tenant_b


def test_shared_cache_constructor_failure_does_not_retry_unscoped(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache_root"
    cache_root.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    semantic_cache._SHARED_CACHES.clear()

    class FailingScopedCache:
        created_without_scope = False

        def __init__(self, *, namespace=None, **_kwargs):
            if namespace is not None:
                raise TypeError("scoped setup failed")
            type(self).created_without_scope = True

    with pytest.raises(TypeError, match="scoped setup failed"):
        semantic_cache.get_shared_cache(
            cache_cls=FailingScopedCache,
            similarity_threshold=0.9,
            ttl=60,
            max_size=10,
            namespace="tenant-a",
        )

    assert FailingScopedCache.created_without_scope is False
    assert semantic_cache._SHARED_CACHES == {}


def test_clear_shared_caches_respects_namespace_scope(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache_root"
    cache_root.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    semantic_cache._SHARED_CACHES.clear()

    cache_tenant_a = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-a",
    )
    cache_tenant_b = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-b",
    )

    cache_tenant_a._cache["a"] = object()
    cache_tenant_b._cache["b"] = object()

    cleared = semantic_cache.clear_shared_caches(namespace="tenant-a")

    assert cleared == 1
    assert cache_tenant_a.get_stats()["size"] == 0
    assert cache_tenant_b.get_stats()["size"] == 1


@pytest.mark.asyncio
async def test_shared_cache_payloads_remain_tenant_isolated(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache_root"
    cache_root.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    semantic_cache._SHARED_CACHES.clear()

    tenant_a = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-a",
    )
    tenant_b = semantic_cache.get_shared_cache(
        cache_cls=SemanticCache,
        similarity_threshold=0.9,
        ttl=60,
        max_size=10,
        namespace="tenant-b",
    )
    await tenant_a.set(
        "shared query",
        {
            "documents": [{"id": "tenant-a-doc", "content": "tenant A"}],
            "answer": "STALE_SENTINEL",
        },
    )

    assert await tenant_b.get("shared query") is None
    assert await tenant_a.get("shared query") == {
        "documents": [
            {
                "id": "tenant-a-doc",
                "content": "tenant A",
                "metadata": {},
                "score": 0.0,
            }
        ],
        "metadata": {"kind": "retrieval_documents", "schema_version": 1},
    }
