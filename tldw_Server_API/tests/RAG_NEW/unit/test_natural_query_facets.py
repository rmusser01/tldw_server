"""Natural questions retain distinct evidence facets through the public pipeline."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MediaDBRetriever,
    RetrievalConfig,
    _derive_bounded_media_term_query,
)

QUESTION = (
    "What changes does the Department of Transport chapter propose to BXA supervision "
    "and the review of active investigations? Cite the relevant printed and PDF pages, "
    "and attribute these recommendations to the document rather than presenting them as enacted law."
)


@pytest.mark.parametrize("punctuation", ["?", ""])
def test_citation_formatting_does_not_replace_explicit_acronym_facets(punctuation):
    terms = _derive_bounded_media_term_query(QUESTION.replace("?", punctuation), max_terms=32)
    assert "bxa" in terms.split(" OR ")
    assert set(terms.split(" OR ")).isdisjoint({"cite", "relevant", "printed", "pdf", "pages", "attribute", "enacted"})


@pytest.mark.parametrize("acronym", ["UN", "EPA", "NHS"])
def test_explicit_short_organization_acronyms_are_retained(acronym):
    terms = _derive_bounded_media_term_query(f"What oversight changes does {acronym} propose?", max_terms=8)
    assert acronym.lower() in terms.split(" OR ")


def test_ordinary_short_words_and_standalone_formatting_are_not_promoted():
    terms = _derive_bounded_media_term_query("What transport changes does the lab propose for the new bus? Cite the relevant PDF pages.", max_terms=32)
    assert set(terms.split(" OR ")).isdisjoint({"lab", "new", "bus", "pdf"})


@given(st.lists(st.sampled_from(["UN", "EPA", "NHS", "IRS", "BXA"]), min_size=1, max_size=40),
       st.integers(min_value=2, max_value=32))
def test_acronym_terms_remain_unique_and_bounded(acronyms, limit):
    query = f"What oversight changes do {' '.join(acronyms)} propose? Cite the relevant printed and PDF pages."
    terms = _derive_bounded_media_term_query(query, max_terms=limit).split(" OR ")
    assert len(terms) == len(set(terms)) <= limit
    assert set(terms) <= {"oversight", "changes", *(word.lower() for word in acronyms)}


@pytest.fixture
def multifacet_source(media_database):
    review = (
        "Conduct a comprehensive review of all major active BXA investigations and activities. "
        "End activities outside its mandate and publish the findings after independent review. "
        "Qualified appointees should catalogue concerns and report to the department leadership. "
        "This proposal describes future administrative responsibilities and leaves existing safeguards in place."
    )
    offices = (
        "Place BXA under the general supervision of the Northern Directorate and the Southern Directorate. "
        "The two receiving offices should coordinate operational decisions, maintain independent records, "
        "and publish regular findings about their responsibilities. "
        "This proposed structure preserves institutional expertise while ensuring clear lines of responsibility "
        "and consistent communication between each receiving office and senior leadership."
    )
    formatting = [
        f"Appendix {index}: Relevant printed pages attribute recommendations to the document rather than "
        "presenting them as enacted law. Cite the relevant pages and provide careful attribution. "
        "The supporting material explains how contributors should format citations, footnotes, and source references."
        for index in range(9)
    ]
    passages = [*formatting, review, "Unrelated background material. " * 70, offices]
    content = "\n\n".join(passages)
    chunks = []
    offset = 0
    for text in passages:
        chunks.append({"text": text, "start_char": offset, "end_char": offset + len(text), "chunk_type": "text"})
        offset += len(text) + 2
    media_id, _, _ = media_database.add_media_with_keywords(
        title="Transport policy proposals", media_type="document", content=content, keywords=[],
        chunks=chunks, owner_user_id=1,
        safe_metadata=json.dumps({
            "page_map": [{"page_number": 1, "start_char": 0, "end_char": len(content), "printed_page_label": "21"}],
            "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }),
    )
    return media_id, content, review, offices


@pytest.mark.asyncio
async def test_persisted_search_keeps_both_facets_beyond_parent_context_budget(media_database, multifacet_source):
    media_id, _, review, offices = multifacet_source
    retriever = MediaDBRetriever(str(media_database.db_path), media_db=media_database, config=RetrievalConfig(
        max_results=8, min_score=.2, use_vector=False, fts_level="chunk",
    ))
    docs = await retriever.retrieve(QUESTION.replace("?", ""), allowed_media_ids=[media_id])
    assert {review, offices}.issubset({doc.content for doc in docs})


@pytest.mark.asyncio
async def test_public_pipeline_retains_both_facets_and_exact_scoped_pages(media_database, multifacet_source, monkeypatch):
    from fastapi import BackgroundTasks
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.endpoints import rag_unified as endpoint
    from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Monitoring import topic_monitoring_service
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as pipeline
    from tldw_Server_API.app.core.RAG.rag_service.security_filters import SecurityFilters

    media_id, content, review, offices = multifacet_source
    runtime = SimpleNamespace(_user_id=1, close=AsyncMock())
    monkeypatch.setattr(endpoint, "_build_credential_runtime", lambda *args: runtime)
    monkeypatch.setattr(endpoint, "_resolve_kanban_db_path", lambda *args: None)
    monkeypatch.setattr(endpoint, "_log_rag_queries_for_org", AsyncMock())
    monkeypatch.setattr(topic_monitoring_service, "get_topic_monitoring_service",
                        lambda: SimpleNamespace(schedule_evaluate_and_alert=lambda **kwargs: None))
    monkeypatch.setattr(pipeline, "SecurityFilter", lambda: SecurityFilters(enable_audit_logging=False))
    monkeypatch.setattr(pipeline, "create_reranker", lambda *args, **kwargs: SimpleNamespace(
        rerank=AsyncMock(side_effect=lambda query, docs: docs)))
    monkeypatch.setattr(pipeline, "spell_check_query", AsyncMock(side_effect=lambda query: query.replace("?", "")))
    # Model/index availability is the controlled environmental failure. The
    # public dispatcher, persisted FTS retrieval and all downstream filters run.
    monkeypatch.setattr(MediaDBRetriever, "_initialize_vector_store", lambda self: None)
    request = UnifiedRAGRequest(
        query=QUESTION, sources=["media_db"], include_media_ids=[media_id], search_mode="hybrid",
        fts_level="chunk", top_k=8, min_score=.2, enable_generation=False, enable_cache=True,
        expand_query=False, enable_query_decomposition=False, enable_document_grading=False,
        enable_query_rewriting_loop=False,
        spell_check=True, enable_parent_expansion=True, parent_context_size=500,
        include_sibling_chunks=True, sibling_window=1, enable_security_filter=True,
        sensitivity_level="internal", enable_content_policy_filter=True, enable_injection_filter=True,
        enable_reranking=True, reranking_strategy="flashrank", enable_pre_retrieval_clarification=False,
    )
    with scoped_context(user_id=1, is_admin=True):
        response = await endpoint.unified_search_endpoint(
            request_raw=Request({"type": "http", "method": "POST", "path": "/api/v1/rag/search", "headers": []}),
            request=request, background_tasks=BackgroundTasks(),
            current_user=User(id=1, username="test-admin", is_admin=True), media_db=media_database,
            chacha_db=None, prompts_db=None, collections_db=None,
        )
    docs = response.model_dump()["documents"]
    assert all(any(facet in doc["content"] for doc in docs) for facet in (review, offices))
    assert all(doc["content"] == content[doc["metadata"]["start_char"]:doc["metadata"]["end_char"]] for doc in docs)
    assert all(doc["metadata"]["page_spans"][0]["printed_page_label"] == "21" for doc in docs)
    assert not response.errors
