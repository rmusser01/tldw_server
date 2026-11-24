"""A caller's ``RagRunConfig.from_kwargs(**kw)`` equals the config the kwargs shim builds (TASK-13950.24 PR M).

The 14 direct callers moved from ``unified_rag_pipeline(**kw)`` to
``run_rag_pipeline(RagRunConfig.from_kwargs(**kw))``. The shim builds its config from ``locals()``
(signature defaults filled in), so the two must agree on defaults, the 13 hidden keys and unknown keys.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.pipeline.run_config import FIELD_GROUP, RagRunConfig
from tldw_Server_API.tests.RAG_NEW.unit.test_unified_pipeline_golden import GOLDEN_CASES

pytestmark = pytest.mark.unit

_RUNTIME = object()
# One kwargs set per direct call site (the keys each passes today), plus bare and full sets.
SITE_KWARGS: dict[str, dict[str, Any]] = {
    "bare": {"query": "q"},
    "rag_unified_ablate": {"query": "q", "top_k": 5, "enable_generation": False, "generation_model": None, "max_generation_tokens": 300,
                           "credential_runtime": _RUNTIME, "enable_reranking": True, "reranking_strategy": "flashrank"},
    "rag_unified_standard": {"query": "q", "sources": ["media_db"], "user_id": "7", "workspace_id": "ws", "include_sources": False,
                             "metadata": {"a": 1}, "enable_expansion": True, "credential_runtime": _RUNTIME, "media_db": _RUNTIME,
                             "chacha_db": _RUNTIME, "prompts_db": _RUNTIME, "claims_budget_usd": 1.5},
    "slides": {"query": "q", "top_k": 8, "sources": ["media_db", "notes", "chats"], "media_db_path": "m.db", "notes_db_path": "c.db",
               "character_db_path": "c.db"},
    "data_tables": {"query": "q", "sources": ["notes"], "media_db_path": "m.db", "notes_db_path": "c.db", "character_db_path": "c.db",
                    "search_mode": "hybrid", "top_k": 3, "min_score": 0.2, "enable_generation": False},
    "abtest": {"query": "q", "search_mode": "hybrid", "hybrid_alpha": 0.7, "top_k": 4, "index_namespace": "col", "user_id": "3",
               "include_media_ids": [1, 2]},
    "eval_runner": {"query": "q", "search_mode": "fts", "top_k": 2, "enable_generation": True, "user_id": "u", "extra_unknown": 1},
    "recipe_runs": {"query": "q", "media_db_path": "m.db", "notes_db_path": None, "enable_generation": False, "top_k": 6},
    "rag_answer_quality": {"media_db_path": None, "notes_db_path": "c.db", "query": "q", "enable_citations": True},
    "mcp_rag": {"query": "q", "enable_generation": True, "include_sources": True, "workspace_id": "w", "credential_runtime": _RUNTIME,
                "enable_web_fallback": False},
    "mcp_slides": {"query": "q", "sources": ["media_db"], "media_db_path": "m.db", "notes_db_path": "c.db", "search_mode": "hybrid",
                   "enable_reranking": True},
    "workflows_search": {"query": "q", "sources": ["media_db"], "search_mode": "hybrid", "top_k": 5, "hybrid_alpha": 0.5,
                         "media_db_path": "m.db", "enable_reranking": True, "timeout_seconds": 9.0},
    "quality_eval": {"query": "q", "index_namespace": "ns", "search_mode": "hybrid", "enable_reranking": True, "enable_citations": True,
                     "enable_generation": True, "enable_post_verification": True, "adaptive_max_claims": 12},
    "everything": {**{k: object() for k in FIELD_GROUP if k != "query"}, "query": "q", "not_a_field": 1},
}


def _shim_config(monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]) -> RagRunConfig:
    seen: list[RagRunConfig] = []

    async def capture(rcfg: RagRunConfig) -> str:
        seen.append(rcfg)
        return "ok"

    monkeypatch.setattr(up, "run_rag_pipeline", capture)
    assert asyncio.run(up.unified_rag_pipeline(**kwargs)) == "ok"
    return seen[0]


def _assert_same(a: RagRunConfig, b: RagRunConfig) -> None:
    ka, kb = a.as_kwargs(), b.as_kwargs()
    assert ka.keys() == kb.keys()
    bad = {k: (ka[k], kb[k]) for k in ka if not (ka[k] is kb[k] or ka[k] == kb[k])}
    assert bad == {}


@pytest.mark.parametrize("name", list(SITE_KWARGS))
def test_site_kwargs_build_the_shim_config(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = SITE_KWARGS[name]
    _assert_same(RagRunConfig.from_kwargs(**kwargs), _shim_config(monkeypatch, kwargs))


@pytest.mark.parametrize("case", [k for k, v in GOLDEN_CASES.items() if v.entry == "pipeline"])
def test_golden_kwargs_build_the_shim_config(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = {**GOLDEN_CASES[case].kwargs, "enable_cache": False}
    _assert_same(RagRunConfig.from_kwargs(**kwargs), _shim_config(monkeypatch, kwargs))


def test_recurring_question_request_dump_builds_the_shim_config(monkeypatch: pytest.MonkeyPatch) -> None:
    request = UnifiedRAGRequest(query="q", workspace_id="ws1")
    dump = request.model_dump()
    request_only = set(dump) - set(FIELD_GROUP)
    assert len(request_only) == 36  # ignored, not rejected
    rcfg = RagRunConfig.from_kwargs(**dump)
    assert rcfg.scope.workspace_id == "ws1"
    assert not request_only & set(rcfg.as_kwargs())
    _assert_same(rcfg, _shim_config(monkeypatch, dump))
