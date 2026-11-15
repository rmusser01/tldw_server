"""Parent context must reach generation through the normal filtering boundary."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def expansion_pipeline(monkeypatch):
    seed = Document(
        id="selected-chunk",
        content="Transfer the program to the Department of",
        source=DataSource.MEDIA_DB,
        score=0.9,
        metadata={"media_id": "7", "chunk_index": 2, "start_char": 0, "end_char": 41},
    )
    expanded = Document(
        id=seed.id,
        content=seed.content + " Education. Contact private@example.test.",
        source=seed.source,
        score=seed.score,
        start_char=0,
        end_char=81,
        page_number=3,
        metadata={
            **seed.metadata,
            "end_char": 81,
            "page_number": 3,
            "page_spans": [{"page_number": 3, "start_char": 0, "end_char": 81}],
        },
    )
    retrieve = AsyncMock(return_value=[seed])
    monkeypatch.setattr(pipeline, "MultiDatabaseRetriever", lambda *args, **kwargs: SimpleNamespace(retrieve=retrieve))
    expand = AsyncMock(return_value=([expanded], {"context_expanded_count": 1, "siblings_added_count": 1}))
    monkeypatch.setattr(pipeline, "expand_media_context", expand, raising=False)
    return seed, expanded, expand


async def run_pipeline(**kwargs):
    return await pipeline.unified_rag_pipeline(
        query="Where should the program move?",
        sources=["media_db"],
        include_media_ids=kwargs.pop("include_media_ids", [7]),
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "flags",
    [
        {"enable_parent_expansion": True},
        {"include_sibling_chunks": True},
        {"enable_parent_expansion": True, "include_sibling_chunks": True},
    ],
)
async def test_requested_context_reaches_response_with_locations(expansion_pipeline, flags):
    seed, expanded, expand = expansion_pipeline
    request_db = SimpleNamespace()
    result = await run_pipeline(
        **flags,
        parent_context_size=350,
        sibling_window=2,
        media_db=request_db,
    )
    assert result.documents[0]["content"] == expanded.content
    assert result.documents[0]["id"] == seed.id
    assert result.documents[0]["metadata"]["page_spans"] == expanded.metadata["page_spans"]
    assert result.metadata["context_expanded_count"] == 1
    assert result.metadata["siblings_added_count"] == 1
    assert expand.await_args.kwargs["media_db"] is request_db
    assert expand.await_args.kwargs["parent_context_size"] == 350
    assert expand.await_args.kwargs["sibling_window"] == 2


async def test_disabled_context_preserves_seed_without_parent_reads(expansion_pipeline):
    seed, _, expand = expansion_pipeline
    result = await run_pipeline()
    assert result.documents[0]["content"] == seed.content
    expand.assert_not_awaited()


async def test_expanded_text_passes_injection_policy_and_security_filters(expansion_pipeline, monkeypatch):
    _, expanded, _ = expansion_pipeline
    inspected = []

    def injection_filter(documents, **kwargs):
        inspected.append(("injection", documents[0].content))
        return {"affected": 0}

    def policy_filter(documents, **kwargs):
        inspected.append(("policy", documents[0].content))
        return {"affected": 0}

    class SecurityFilter:
        def filter_by_sensitivity(self, documents, **kwargs):
            inspected.append(("security", documents[0].content))
            return []

    monkeypatch.setattr(pipeline, "downweight_injection_docs", injection_filter)
    monkeypatch.setattr(pipeline, "apply_content_policy", policy_filter)
    monkeypatch.setattr(pipeline, "SecurityFilter", SecurityFilter)
    monkeypatch.setattr(
        pipeline,
        "SensitivityLevel",
        SimpleNamespace(
            PUBLIC="public",
            INTERNAL="internal",
            CONFIDENTIAL="confidential",
            RESTRICTED="restricted",
        ),
    )
    result = await run_pipeline(
        enable_parent_expansion=True,
        include_sibling_chunks=True,
        enable_injection_filter=True,
        enable_content_policy_filter=True,
        enable_security_filter=True,
    )
    assert inspected == [(name, expanded.content) for name in ("injection", "policy", "security")]
    assert result.documents == []
    assert result.metadata["security_filter"]["excluded_count"] == 1
    assert "private@example.test" not in repr(result)


async def test_context_expansion_obeys_complete_request_deadline(expansion_pipeline, monkeypatch):
    cancelled = asyncio.Event()

    async def slow_expansion(*args, **kwargs):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    monkeypatch.setattr(pipeline, "expand_media_context", slow_expansion)
    with pytest.raises(TimeoutError):
        await run_pipeline(enable_parent_expansion=True, timeout_seconds=0.05)
    assert cancelled.is_set()


async def test_failed_parent_read_does_not_return_unfiltered_evidence(expansion_pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "expand_media_context", AsyncMock(side_effect=RuntimeError("db unavailable")))
    result = await run_pipeline(enable_parent_expansion=True)
    assert result.documents == []
    assert "pipeline_failed" in result.errors


async def test_canonical_parent_completes_clause_and_preserves_page_offsets(media_database, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    content = (
        "Earlier context. " * 12
        + "Transfer the program to the Department of Labor or the local council. "
        + "Later context. " * 12
    )
    start = content.index("Transfer")
    end = content.index(" Labor")
    page_map = [
        {"page_number": 1, "start_char": 0, "end_char": end, "printed_page_label": "10"},
        {"page_number": 2, "start_char": end, "end_char": len(content), "printed_page_label": "11"},
    ]
    media_id, _, _ = media_database.add_media_with_keywords(
        title="Program policy",
        media_type="document",
        content=content,
        keywords=[],
        safe_metadata=json.dumps(
            {
                "page_map": page_map,
                "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest(),
            }
        ),
    )
    seed = Document(
        id="matched-chunk",
        content=content[start:end],
        source=DataSource.MEDIA_DB,
        score=0.9,
        start_char=start,
        end_char=end,
        metadata={"media_id": str(media_id), "start_char": start, "end_char": end},
    )
    monkeypatch.setattr(
        pipeline,
        "MultiDatabaseRetriever",
        lambda *args, **kwargs: SimpleNamespace(
            retrieve=AsyncMock(return_value=[seed]),
        ),
    )
    with scoped_context(user_id=1, is_admin=True):
        result = await run_pipeline(
            media_db=media_database, include_media_ids=[media_id], enable_parent_expansion=True, parent_context_size=100
        )
    document = result.documents[0]
    metadata = document["metadata"]
    assert "Department of Labor or the local council." in document["content"]
    assert document["content"] == content[metadata["start_char"] : metadata["end_char"]]
    assert metadata["matched_start_char"] == start and metadata["matched_end_char"] == end
    assert metadata["page_spans"] == [
        {"page_number": 1, "start_char": start - 100, "end_char": end, "printed_page_label": "10"},
        {"page_number": 2, "start_char": end, "end_char": end + 100, "printed_page_label": "11"},
    ]
