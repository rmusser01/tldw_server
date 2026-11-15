"""RAG sensitivity filtering must use trusted authority without bypassing ACLs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import ScopeContext
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up
from tldw_Server_API.app.core.RAG.rag_service.security_filters import SecurityFilters
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_user", [None, 99])
@pytest.mark.parametrize(
    ("scope_user", "is_admin", "request_user", "max_sensitivity", "deny_acl", "restricted", "expected"),
    [
        (1, True, "1", "internal", False, False, ["proposal"]),
        (1, True, "1", "public", False, False, []),
        (1, False, "1", "internal", False, False, []),
        (None, True, "1", "internal", False, False, []),
        ("absent", False, "1", "internal", False, False, []),
        (2, True, "1", "internal", False, False, []),
        (1, "true", "1", "internal", False, False, []),
        (1, True, "1", "internal", True, False, []),
        (1, True, "anonymous", "internal", False, False, []),
        (1, True, "1", "internal", False, True, []),
        (1, True, "1", "restricted", False, True, ["proposal"]),
    ],
)
async def test_security_filter_honors_only_matching_trusted_admin_authority(
    monkeypatch, runtime_user, scope_user, is_admin, request_user, max_sensitivity, deny_acl, restricted, expected
):
    """An authenticated admin can read INTERNAL prose within the requested ceiling."""
    scope = ScopeContext(scope_user, [], [], None, None, is_admin=is_admin)
    if scope_user == "absent":
        scope = None
    monkeypatch.setattr(up, "get_scope", lambda: scope)
    security = SecurityFilters(enable_audit_logging=False)
    if deny_acl:
        security.access_controller.set_document_acl("proposal", ["different-user"], [])
    monkeypatch.setattr(up, "SecurityFilter", lambda: security)
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                Document(
                    id="proposal",
                    content=(
                        "Restricted: classified evidence."
                        if restricted
                        else "The public proposal would create an education savings account."
                    ),
                    metadata={},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]
        )
    )
    monkeypatch.setattr(up, "MultiDatabaseRetriever", lambda *args, **kwargs: retriever)

    result = await up.unified_rag_pipeline(
        query="What does the proposal create?",
        user_id=request_user,
        metadata={"is_admin": True, "role": "admin"},
        credential_runtime=SimpleNamespace(_user_id=runtime_user) if runtime_user else None,
        enable_security_filter=True,
        sensitivity_level=max_sensitivity,
        detect_pii=False,
        redact_pii=False,
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
    )

    assert [doc["id"] for doc in result.documents] == (expected if runtime_user is None else [])
    assert not result.errors


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("extra", ["", " A financial account is included.", " Classified information."])
async def test_public_search_endpoint_preserves_accountability_evidence_with_internal_ceiling(
    media_database, monkeypatch, extra
):
    """The public endpoint's real filters, expansion and response retain the usable FBI seed."""
    import hashlib
    import json

    from fastapi import BackgroundTasks
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.endpoints import rag_unified as endpoint
    from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Monitoring import topic_monitoring_service

    content = (
        "Conduct an immediate review of all major active FBI investigations and activities. "
        "The FBI is accountable to the Attorney General."
        + extra
    )
    media_id, _, _ = media_database.add_media_with_keywords(
        title="Public oversight proposal", media_type="document", content=content, keywords=[], owner_user_id=1,
        safe_metadata=json.dumps({
            "page_map": [{"page_number": 1, "start_char": 0, "end_char": len(content), "printed_page_label": "5"}],
            "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }),
    )
    excerpt = "Conduct an immediate review of all major active FBI investigations and activities."
    seed = Document(
        id="core-fbi-passage", content=excerpt, source=DataSource.MEDIA_DB, score=.9,
        start_char=0, end_char=len(excerpt), metadata={"media_id": str(media_id), "chunk_index": 0},
    )
    retriever = SimpleNamespace(retrieve=AsyncMock(return_value=[seed]))

    def factory(*args, **kwargs):
        return retriever

    monkeypatch.setattr(endpoint, "MultiDatabaseRetriever", factory)
    monkeypatch.setattr(up, "MultiDatabaseRetriever", factory)
    runtime = SimpleNamespace(_user_id=1, close=AsyncMock())
    monkeypatch.setattr(endpoint, "_build_credential_runtime", lambda *args: runtime)
    monkeypatch.setattr(endpoint, "_resolve_kanban_db_path", lambda *args: None)
    monkeypatch.setattr(endpoint, "_log_rag_queries_for_org", AsyncMock())
    monkeypatch.setattr(
        topic_monitoring_service, "get_topic_monitoring_service",
        lambda: SimpleNamespace(schedule_evaluate_and_alert=lambda **kwargs: None),
    )
    monkeypatch.setattr(up, "SecurityFilter", lambda: SecurityFilters(enable_audit_logging=False))
    rerank = AsyncMock(side_effect=lambda query, docs: docs)
    monkeypatch.setattr(up, "create_reranker", lambda *args, **kwargs: SimpleNamespace(rerank=rerank))
    monkeypatch.setattr(up, "spell_check_query", AsyncMock(side_effect=lambda query: query.replace("?", "")))
    request = UnifiedRAGRequest(
        query="What does the document propose for FBI supervision and review of active investigations? Cite the relevant pages.",
        sources=["media_db"], include_media_ids=[media_id], search_mode="hybrid", fts_level="chunk", top_k=8,
        min_score=.2, enable_generation=False, enable_cache=True, spell_check=True,
        enable_parent_expansion=True, parent_context_size=500, include_sibling_chunks=True, sibling_window=1,
        enable_security_filter=True, sensitivity_level="internal", enable_content_policy_filter=True,
        enable_injection_filter=True, enable_reranking=True, reranking_strategy="flashrank",
        enable_pre_retrieval_clarification=False,
    )
    with scoped_context(user_id=1, is_admin=True):
        response = await endpoint.unified_search_endpoint(
            request_raw=Request({"type": "http", "method": "POST", "path": "/api/v1/rag/search", "headers": []}),
            request=request, background_tasks=BackgroundTasks(),
            current_user=User(id=1, username="test-admin", is_admin=True), media_db=media_database,
            chacha_db=None, prompts_db=None, collections_db=None,
        )
    payload = response.model_dump()
    expected = [] if extra else [seed.id]
    assert [doc["id"] for doc in payload["documents"]] == expected
    assert payload["metadata"]["security_filter"]["excluded_count"] == int(bool(extra))
    if not extra:
        assert "accountable to the Attorney General" in payload["documents"][0]["content"]
        assert payload["documents"][0]["metadata"]["page_spans"][0]["printed_page_label"] == "5"
        rerank.assert_awaited_once()
    assert not payload["errors"]
    runtime.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cached_retrieval_seed_is_reclassified_before_returning(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    cached = {"documents": [Document(
        id="cached-public-passage", content="The agency is accountable to the Attorney General.",
        source=DataSource.MEDIA_DB, score=.9, metadata={},
    )]}
    cache = SimpleNamespace(get=lambda *args, **kwargs: cached, find_similar=lambda *args, **kwargs: None)
    monkeypatch.setattr(up, "get_shared_cache", lambda **kwargs: cache)
    monkeypatch.setattr(up, "SecurityFilter", lambda: SecurityFilters(enable_audit_logging=False))
    with scoped_context(user_id=1, is_admin=True):
        result = await up.unified_rag_pipeline(
            query="cached public oversight proposal", user_id="1", enable_generation=False,
            enable_cache=True, adaptive_cache=False, enable_reranking=False,
            enable_security_filter=True, sensitivity_level="internal",
        )
    assert result.cache_hit is True
    assert [doc["id"] for doc in result.documents] == ["cached-public-passage"]
