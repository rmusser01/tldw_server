import builtins
from unittest.mock import AsyncMock, patch

import pytest

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MediaDBRetriever,
    MultiDatabaseRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_batch_pipeline,
    unified_rag_pipeline,
)


pytestmark = pytest.mark.unit


class _FakeRetriever:
    def __init__(self) -> None:
        self.config = RetrievalConfig()
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, query: str) -> list[Document]:
        self.calls.append(
            {
                "query": query,
                "max_results": self.config.max_results,
                "min_score": self.config.min_score,
                "use_fts": self.config.use_fts,
                "use_vector": self.config.use_vector,
            }
        )
        return [
            Document(
                id=f"doc:{query}",
                content="retrieved",
                metadata={},
                source=DataSource.NOTES,
                score=0.8,
            )
        ]


class _FakeMediaRetriever(MediaDBRetriever):
    def __init__(self) -> None:
        self.config = RetrievalConfig()
        self.calls: list[dict[str, object]] = []

    async def _retrieve_vector(
        self,
        query: str,
        media_type=None,
        **kwargs,
    ) -> list[Document]:
        self.calls.append(
            {
                "query": query,
                "media_type": media_type,
                "index_namespace": kwargs.get("index_namespace"),
            }
        )
        return [
            Document(
                id="media:1",
                content="retrieved",
                metadata={},
                source=DataSource.MEDIA_DB,
                score=0.9,
            )
        ]


@pytest.mark.asyncio
async def test_multi_database_retriever_derives_effective_config_from_retrieval_plan() -> None:
    retriever = MultiDatabaseRetriever({})
    fake = _FakeRetriever()
    retriever.retrievers = {DataSource.NOTES: fake}
    plan = RetrievalPlan(
        query="planned query",
        sources=("notes",),
        search_mode="vector",
        top_k=7,
        min_score=0.4,
        index_namespace=None,
        collection_names={"notes": "user_5_notes_embeddings"},
    )

    documents = await retriever.retrieve_from_plan(plan)

    assert [doc.id for doc in documents] == ["doc:planned query"]
    assert fake.calls[0]["max_results"] == 7
    assert fake.calls[0]["min_score"] == 0.4
    assert fake.calls[0]["use_fts"] is False
    assert fake.calls[0]["use_vector"] is True
    assert fake.config.max_results == 20
    assert fake.config.min_score == 0.0
    assert fake.config.use_fts is True
    assert fake.config.use_vector is True


@pytest.mark.asyncio
async def test_multi_database_retriever_plan_overrides_conflicting_legacy_namespace() -> None:
    retriever = MultiDatabaseRetriever({})
    fake = _FakeMediaRetriever()
    retriever.retrievers = {DataSource.MEDIA_DB: fake}
    plan = RetrievalPlan(
        query="planned query",
        sources=("media_db",),
        search_mode="vector",
        top_k=6,
        min_score=0.25,
        index_namespace=None,
        collection_names={"media_db": "user_5_media_embeddings"},
    )

    await retriever.retrieve_from_plan(plan, index_namespace="legacy-namespace")

    assert fake.calls[0]["query"] == "planned query"
    assert fake.calls[0]["index_namespace"] == "user_5_media_embeddings"


@pytest.mark.asyncio
async def test_unified_pipeline_retrieval_uses_plan_owned_policy() -> None:
    retrieval_plan = RetrievalPlan(
        query="planned query",
        sources=("notes",),
        search_mode="vector",
        top_k=4,
        min_score=0.33,
        index_namespace="tenant-a",
        collection_names={"notes": "user_5_notes_embeddings"},
    )

    with patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.execute_retrieval_phase",
        AsyncMock(
            return_value=RetrievedEvidence(
                documents=[
                    Document(
                        id="note_1",
                        content="retrieved",
                        metadata={},
                        source=DataSource.NOTES,
                        score=0.9,
                    )
                ],
                metadata={},
            )
        ),
    ) as mock_executor:
        result = await unified_rag_pipeline(
            query="outer query should not drive retrieval",
            sources=["media_db"],
            search_mode="fts",
            top_k=99,
            min_score=0.01,
            retrieval_plan=retrieval_plan,
            enable_cache=False,
            enable_reranking=False,
            enable_generation=False,
        )

    assert result.documents
    executor_call = mock_executor.await_args
    assert executor_call.kwargs["retrieval_plan"].query == "planned query"
    assert executor_call.kwargs["retrieval_plan"].sources == ("notes",)
    assert executor_call.kwargs["retrieval_plan"].index_namespace == "tenant-a"


@pytest.mark.asyncio
async def test_unified_pipeline_plan_overrides_conflicting_legacy_namespace() -> None:
    retrieval_plan = RetrievalPlan(
        query="planned query",
        sources=("media_db",),
        search_mode="vector",
        top_k=4,
        min_score=0.33,
        index_namespace=None,
        collection_names={"media_db": "user_5_media_embeddings"},
    )

    with patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.execute_retrieval_phase",
        AsyncMock(
            return_value=RetrievedEvidence(
                documents=[
                    Document(
                        id="media_1",
                        content="retrieved",
                        metadata={},
                        source=DataSource.MEDIA_DB,
                        score=0.9,
                    )
                ],
                metadata={},
            )
        ),
    ) as mock_executor:
        result = await unified_rag_pipeline(
            query="outer query",
            index_namespace="legacy-namespace",
            retrieval_plan=retrieval_plan,
            enable_cache=False,
            enable_reranking=False,
            enable_generation=False,
        )

    executor_call = mock_executor.await_args
    assert executor_call.kwargs["retrieval_plan"].index_namespace == "user_5_media_embeddings"
    assert result.query == "outer query"


@pytest.mark.asyncio
async def test_unified_batch_pipeline_rewrites_plan_query_per_item() -> None:
    retrieval_plan = RetrievalPlan(
        query="base batch query",
        sources=("notes",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.2,
        index_namespace="tenant-a",
        collection_names={"notes": "user_5_notes_embeddings"},
    )

    async def _fake_unified_rag_pipeline(**kwargs):
        return {"query": kwargs["query"]}

    with patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.unified_rag_pipeline",
        AsyncMock(side_effect=_fake_unified_rag_pipeline),
    ) as mock_pipeline:
        await unified_batch_pipeline(
            queries=["first query", "second query"],
            max_concurrent=1,
            retrieval_plan=retrieval_plan,
        )

    first_call, second_call = mock_pipeline.await_args_list
    assert first_call.kwargs["query"] == "first query"
    assert first_call.kwargs["retrieval_plan"].query == "first query"
    assert second_call.kwargs["query"] == "second query"
    assert second_call.kwargs["retrieval_plan"].query == "second query"


@pytest.mark.asyncio
async def test_unified_batch_pipeline_fallback_rewrites_plan_query_per_item() -> None:
    retrieval_plan = RetrievalPlan(
        query="base batch query",
        sources=("notes",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.2,
        index_namespace="tenant-a",
        collection_names={"notes": "user_5_notes_embeddings"},
    )

    async def _fake_unified_rag_pipeline(**kwargs):
        return {"query": kwargs["query"]}

    real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "batch_utils" or name.endswith(".batch_utils"):
            raise ImportError("forced batch_utils import failure")
        return real_import(name, globals, locals, fromlist, level)

    with patch(
        "builtins.__import__",
        side_effect=_failing_import,
    ), patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.unified_rag_pipeline",
        AsyncMock(side_effect=_fake_unified_rag_pipeline),
    ) as mock_pipeline:
        await unified_batch_pipeline(
            queries=["first query", "second query"],
            max_concurrent=1,
            retrieval_plan=retrieval_plan,
        )

    first_call, second_call = mock_pipeline.await_args_list
    assert first_call.kwargs["query"] == "first query"
    assert first_call.kwargs["retrieval_plan"].query == "first query"
    assert second_call.kwargs["query"] == "second query"
    assert second_call.kwargs["retrieval_plan"].query == "second query"
