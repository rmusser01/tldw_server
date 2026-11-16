"""Actual indexed chunks retain identity and verified pages through public RAG."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import chromadb
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MediaDBRetriever
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorStoreConfig, VectorStoreType
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.chromadb_adapter import (
    ChromaDBAdapter,
    _distance_similarity,
)


def adapter_for(collection):
    adapter = ChromaDBAdapter(
        VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            connection_params={},
            embedding_dim=2,
            user_id="1",
        )
    )
    adapter.manager = SimpleNamespace(get_or_create_collection=lambda *_: collection)
    adapter._initialized = True
    return adapter


@given(
    st.floats(min_value=0, max_value=1e20, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e20, allow_nan=False, allow_infinity=False),
)
def test_l2_scores_remain_bounded_and_monotonic(first, second):
    near, far = sorted((first, second))
    assert 0 <= _distance_similarity(far, "l2") <= _distance_similarity(near, "l2") <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", [None, "l2", "cosine", "ip"])
async def test_real_chroma_distance_uses_collection_metric_not_adapter_default(tmp_path, metric):
    client = chromadb.PersistentClient(path=str(tmp_path / "vectors"))
    collection = client.create_collection("distance-test", metadata={"hnsw:space": metric} if metric else None)
    collection.add(ids=["near", "far"], documents=["Near", "Far"], embeddings=[[3.0, 0.0], [1.0, 1.0]])
    results = await adapter_for(collection).search("distance-test", [2.0, 0.0], k=2)
    assert all(0 <= row.score <= 1 for row in results)
    assert results[0].score >= results[1].score
    if metric in (None, "l2"):
        assert results[1].score == pytest.approx(1 / (1 + results[1].distance ** 0.5))
    if metric == "cosine":
        assert results[0].score == pytest.approx(1 - results[0].distance)


@pytest.mark.asyncio
@pytest.mark.parametrize("expand", [False, True])
async def test_public_vector_pipeline_preserves_distinct_exact_chunk_pages(
    tmp_path, media_database, monkeypatch, expand
):
    from fastapi import BackgroundTasks
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.endpoints import rag_unified as endpoint
    from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create as embeddings
    from tldw_Server_API.app.core.Monitoring import topic_monitoring_service
    from tldw_Server_API.app.core.RAG.rag_service import transport
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as pipeline
    from tldw_Server_API.app.core.RAG.rag_service.security_filters import SecurityFilters

    first, second = "The northern office conducts the review.", "The southern office receives the recommendations."
    content = first + "\n\n" + "Background material. " * 30 + "\n\n" + second
    boundary = len(content) - len(second)
    media_id, _, _ = media_database.add_media_with_keywords(
        title="Selected evidence",
        media_type="document",
        content=content,
        keywords=[],
        owner_user_id=1,
        safe_metadata=json.dumps(
            {
                "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "page_map": [
                    {"page_number": 1, "start_char": 0, "end_char": boundary, "printed_page_label": "7"},
                    {"page_number": 2, "start_char": boundary, "end_char": len(content), "printed_page_label": "8"},
                ],
            }
        ),
    )
    client = chromadb.PersistentClient(path=str(tmp_path / "vectors"))
    collection = client.create_collection("public-vector-test")
    ids = [f"media_{media_id}_chunk_0", f"media_{media_id}_chunk_1"]
    collection.add(
        ids=ids + ["foreign"],
        documents=[first, second, "Foreign source"],
        embeddings=[[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
        metadatas=[
            {"media_id": str(media_id), "kind": "chunk", "chunk_index": 0, "chunk_start": 0, "chunk_end": len(first)},
            {
                "media_id": str(media_id),
                "kind": "chunk",
                "chunk_index": 1,
                "chunk_start": boundary,
                "chunk_end": len(content),
            },
            {"media_id": "99999", "kind": "chunk", "chunk_start": 0, "chunk_end": 14},
        ],
    )
    adapter = adapter_for(collection)
    monkeypatch.setattr(
        MediaDBRetriever, "_initialize_vector_store", lambda self: setattr(self, "vector_store", adapter)
    )
    monkeypatch.setattr(
        embeddings, "get_embedding_config", lambda: {"embedding_config": {"default_model_id": None, "models": {}}}
    )
    monkeypatch.setattr(embeddings, "create_embeddings_batch", lambda *_args, **_kwargs: [[0.0, 0.0]])
    monkeypatch.setattr(DatabasePaths, "get_kanban_db_path", Mock(side_effect=FileNotFoundError("No Kanban fixture")))
    monkeypatch.setattr(transport, "log_rag_queries_for_org_context", AsyncMock())
    monkeypatch.setattr(
        topic_monitoring_service,
        "get_topic_monitoring_service",
        lambda: SimpleNamespace(schedule_evaluate_and_alert=lambda **_: None),
    )
    monkeypatch.setattr(pipeline, "SecurityFilter", lambda: SecurityFilters(enable_audit_logging=False))
    request = UnifiedRAGRequest(
        query="Which office conducts the review and which receives the recommendations?",
        sources=["media_db"],
        include_media_ids=[media_id],
        search_mode="vector",
        top_k=8,
        min_score=0.2,
        enable_generation=False,
        enable_cache=False,
        expand_query=False,
        spell_check=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        enable_parent_expansion=expand,
        parent_context_size=100,
        include_sibling_chunks=False,
        enable_security_filter=True,
        sensitivity_level="internal",
    )
    with scoped_context(user_id=1, is_admin=True):
        response = await endpoint.unified_search_endpoint(
            request_raw=Request({"type": "http", "method": "POST", "path": "/api/v1/rag/search", "headers": []}),
            request=request,
            background_tasks=BackgroundTasks(),
            current_user=User(id=1, username="test-admin", is_admin=True),
            media_db=media_database,
            chacha_db=None,
            prompts_db=None,
            collections_db=None,
        )
    docs = response.model_dump()["documents"]
    assert {doc["id"] for doc in docs} == set(ids)
    assert all(doc["metadata"]["chunk_id"] == doc["id"] for doc in docs)
    assert all(doc["content"] == content[doc["metadata"]["start_char"] : doc["metadata"]["end_char"]] for doc in docs)
    assert all(doc["metadata"]["page_spans"] for doc in docs)
    assert all(0 <= doc["score"] <= 1 for doc in docs)
    if not expand:
        assert {doc["content"] for doc in docs} == {first, second}
        assert {doc["metadata"]["page_number"] for doc in docs} == {1, 2}
    assert not response.errors
