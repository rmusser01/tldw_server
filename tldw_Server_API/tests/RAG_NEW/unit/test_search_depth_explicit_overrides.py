"""Search-depth defaults must preserve the public request's explicit choices."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import resolve_rag_request
from tldw_Server_API.app.core.RAG.rag_service.transport import build_standard_request_bundle
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


@pytest.fixture
def phases(monkeypatch):
    """Replace external retrieval/model work, retaining the real scoped cache."""
    document = Document(
        id="private-document",
        content="Rowan private evidence",
        score=0.9,
        source=DataSource.MEDIA_DB,
        metadata={},
    )
    retrieval = AsyncMock(return_value=SimpleNamespace(documents=[document], metadata={}))
    expansion = AsyncMock(return_value=["Rowan evidence"])
    hyde = MagicMock(return_value="hypothetical evidence")
    multi_vector = AsyncMock(return_value=[])
    web = AsyncMock(return_value=([], {}))
    reranker = MagicMock()
    reranker.return_value.rerank = AsyncMock(side_effect=lambda _query, docs: docs)
    monkeypatch.setattr(up, "MultiDatabaseRetriever", MagicMock())
    monkeypatch.setattr(up, "execute_retrieval_phase", retrieval)
    monkeypatch.setattr(up, "multi_strategy_expansion", expansion)
    monkeypatch.setattr(up, "RewriteCache", None)
    monkeypatch.setattr(up, "QueryRewriter", None)
    monkeypatch.setattr(up, "QueryAnalyzer", None)
    monkeypatch.setattr(up, "generate_hypothetical_answer", hyde)
    monkeypatch.setattr(up, "hyde_embed_text", AsyncMock(return_value=[]))
    monkeypatch.setattr(up, "apply_precomputed_spans", None)
    monkeypatch.setattr(up, "apply_multi_vector_passages", multi_vector)
    monkeypatch.setattr(up, "fallback_to_web_search", web)
    monkeypatch.setattr(up, "create_reranker", reranker)
    return SimpleNamespace(
        retrieval=retrieval,
        expansion=expansion,
        hyde=hyde,
        multi_vector=multi_vector,
        web=web,
        reranker=reranker,
    )


def request_kwargs(**overrides):
    fields = {
        "query": "Rowan evidence",
        "sources": ["media_db"],
        "search_mode": "fts",
        "min_score": 0,
        "enable_generation": False,
        "enable_cache": True,
        "adaptive_cache": False,
        "enable_reranking": False,
    }
    fields.update(overrides)
    request = UnifiedRAGRequest(**fields)
    return build_standard_request_bundle(
        request,
        current_user=SimpleNamespace(id=41),
        db_paths={},
        media_db=None,
        chacha_db=None,
        search_agent_setting_fn=lambda env, _cfg: "balanced" if env == "SEARCH_DEFAULT_MODE" else None,
    ).pipeline_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("expansion", [False, None, True])
async def test_explicit_expansion_controls_real_cache_reuse(phases, expansion):
    kwargs = request_kwargs(**({} if expansion is None else {"expand_query": expansion}))
    with scoped_context(user_id=41):
        first = await up.unified_rag_pipeline(**kwargs)
        second = await up.unified_rag_pipeline(**kwargs)
    expected_expansion = expansion is not False
    assert first.errors == second.errors == []  # nosec B101 - pytest behavior check.
    assert first.cache_hit is False  # nosec B101 - pytest behavior check.
    assert second.cache_hit is not expected_expansion  # nosec B101 - pytest behavior check.
    assert phases.expansion.await_count == (2 if expected_expansion else 0)  # nosec B101 - pytest behavior check.
    assert phases.retrieval.await_count == (2 if expected_expansion else 1)  # nosec B101 - pytest behavior check.


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["speed", "quality"])
async def test_explicit_default_values_override_public_preset_knobs(phases, mode):
    kwargs = request_kwargs(
        search_depth_mode=mode,
        top_k=10,
        enable_reranking=True,
        reranking_strategy="flashrank",
        expand_query=False,
        enable_web_fallback=False,
        enable_multi_vector_passages=False,
        web_fallback_threshold=1.0,
    )
    with scoped_context(user_id=41):
        result = await up.unified_rag_pipeline(**kwargs)
    assert result.errors == []  # nosec B101 - pytest behavior check.
    assert phases.retrieval.await_args.kwargs["retrieval_plan"].top_k == 10  # nosec B101 - pytest behavior check.
    assert phases.reranker.call_count == 1  # nosec B101 - pytest behavior check.
    strategy, config = phases.reranker.call_args.args
    assert strategy == up.RerankingStrategy.FLASHRANK  # nosec B101 - pytest behavior check.
    assert config.top_k == 10  # nosec B101 - pytest behavior check.
    assert phases.expansion.await_count == 0  # nosec B101 - pytest behavior check.
    # HyDE is internal-only, so the public request leaves its preset value omitted.
    assert phases.hyde.call_count == (1 if mode == "quality" else 0)  # nosec B101 - pytest behavior check.
    assert phases.web.await_count == 0  # nosec B101 - pytest behavior check.
    assert phases.multi_vector.await_count == 0  # nosec B101 - pytest behavior check.


@pytest.mark.asyncio
async def test_explicit_false_wins_over_legacy_expansion_alias(phases):
    kwargs = request_kwargs(expand_query=False)
    kwargs["enable_expansion"] = True
    with scoped_context(user_id=41):
        result = await up.unified_rag_pipeline(**kwargs)
    assert result.errors == []  # nosec B101 - pytest behavior check.
    assert phases.expansion.await_count == 0  # nosec B101 - pytest behavior check.


@pytest.mark.asyncio
async def test_legacy_direct_call_keeps_alias_and_preset_defaults(phases):
    result = await up.unified_rag_pipeline(
        query="Rowan evidence",
        sources=["media_db"],
        top_k=10,
        search_depth_mode="speed",
        expand_query=False,
        enable_expansion=True,
        enable_generation=False,
        enable_cache=False,
    )
    assert result.errors == []  # nosec B101 - pytest behavior check.
    assert phases.retrieval.await_args.kwargs["retrieval_plan"].top_k == 3  # nosec B101 - pytest behavior check.
    assert phases.expansion.await_count == 1  # nosec B101 - pytest behavior check.
    assert phases.reranker.call_count == 0  # nosec B101 - pytest behavior check.


@pytest.mark.asyncio
async def test_internal_canonical_hyde_false_overrides_separate_pipeline_argument(phases):
    kwargs = request_kwargs(search_depth_mode="quality", expand_query=False)
    payload = dict(kwargs["resolved_request"].payload, enable_hyde=False)
    fields = set(kwargs["resolved_request"].explicit_fields) | {"enable_hyde"}
    request = SimpleNamespace(**payload, model_fields_set=fields)
    kwargs["resolved_request"] = resolve_rag_request(request, current_user=SimpleNamespace(id=41))
    kwargs["enable_hyde"] = True
    with scoped_context(user_id=41):
        result = await up.unified_rag_pipeline(**kwargs)
    assert result.errors == []  # nosec B101 - pytest behavior check.
    assert phases.hyde.call_count == 0  # nosec B101 - pytest behavior check.


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_top_k", [10, 5])
async def test_explicit_top_k_preserves_other_canonical_plan_scalars(phases, raw_top_k):
    kwargs = request_kwargs(
        search_depth_mode="speed",
        top_k=10,
        min_score=0.7,
        expand_query=False,
    )
    kwargs["retrieval_plan"] = replace(kwargs["retrieval_plan"], top_k=3)
    kwargs.update(top_k=raw_top_k, min_score=0.0)
    with scoped_context(user_id=41):
        result = await up.unified_rag_pipeline(**kwargs)
    assert result.errors == []  # nosec B101 - pytest behavior check.
    plan = phases.retrieval.await_args.kwargs["retrieval_plan"]
    assert (plan.top_k, plan.min_score) == (10, 0.7)  # nosec B101 - pytest behavior check.
