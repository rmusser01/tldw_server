"""Keep internal persistent stores embedded despite Chroma environment settings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("environment_api", ["chromadb.api.fastapi.FastAPI", "custom.backend.API"])
@pytest.mark.parametrize("entry", ["manager", "shard", "split", "health", "pool", "store"])
async def test_internal_persistent_stores_keep_embedded_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, environment_api: str, entry: str
) -> None:
    """Exercise each real caller and Settings; stop before opening a Chroma client."""
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
    from tldw_Server_API.app.core.Embeddings.health_checks import HealthChecker
    from tldw_Server_API.app.core.Embeddings.sharding import EmbeddingShardManager
    from tldw_Server_API.app.core.RAG.rag_service.chromadb_optimizer import (
        ChromaDBOptimizationConfig,
        ChromaDBOptimizer,
        OptimizedChromaStore,
    )

    monkeypatch.setenv("CHROMA_API_IMPL", environment_api)
    monkeypatch.delenv("CHROMADB_FORCE_STUB", raising=False)
    observed = []
    collection = SimpleNamespace(count=lambda: 0)
    client = SimpleNamespace(list_collections=lambda: [], get_collection=lambda _name: collection, close=lambda: None)

    def open_client(*, path, settings):
        observed.append(settings)
        return client

    monkeypatch.setattr(chromadb, "PersistentClient", open_client)
    config = ChromaDBOptimizationConfig(enable_result_cache=False)
    if entry == "manager":
        with ChromaDBManager(user_id="embedded-settings", user_embedding_config={"USER_DB_BASE_DIR": str(tmp_path)}):
            pass
    elif entry in ("shard", "split"):
        shards = EmbeddingShardManager(base_path=str(tmp_path), num_shards=1)
        if entry == "split":
            observed.clear()
            shards._split_shard(shards.hash_ring.shards[0])
    elif entry == "health":
        checker = HealthChecker.__new__(HealthChecker)
        checker.thresholds = {"db_latency_ms": 1000}
        await checker._check_database_health()
    elif entry == "pool":
        optimizer = ChromaDBOptimizer(config)
        try:
            await optimizer.get_client(str(tmp_path))
        finally:
            optimizer.executor.shutdown()
    else:
        store = OptimizedChromaStore(str(tmp_path), "embedded-settings", optimization_config=config)
        store.optimizer.executor.shutdown()

    assert [settings.chroma_api_impl for settings in observed] == ["chromadb.api.rust.RustBindingsAPI"]


@pytest.mark.parametrize("injection", ["client", "factory"])
def test_explicit_client_injection_keeps_its_existing_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, injection: str
) -> None:
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

    monkeypatch.setenv("CHROMA_API_IMPL", "chromadb.api.fastapi.FastAPI")
    client = SimpleNamespace(close=lambda: None)
    supplied_settings = {"chroma_api_impl": "custom.backend.API"}
    received_settings = []

    def factory(path, settings):
        received_settings.append(settings)
        return client

    arguments = {"client": client} if injection == "client" else {"client_factory": factory}
    with ChromaDBManager(
        user_id="injected-settings",
        user_embedding_config={"USER_DB_BASE_DIR": str(tmp_path), "chroma_client_settings": supplied_settings},
        **arguments,
    ) as manager:
        assert manager.client is client
        if injection == "factory":
            assert received_settings == [{**supplied_settings, "persist_directory": str(manager.user_chroma_path)}]
