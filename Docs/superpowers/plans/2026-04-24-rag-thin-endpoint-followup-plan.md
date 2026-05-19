# RAG Thin Endpoint Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Preserve the current `dev` RAG fixes, then finish the RAG thin-endpoint follow-up by moving standard, agentic, and streaming orchestration into core-owned modules.

**Architecture:** Treat the endpoint as an adapter that resolves requests, delegates to core executors, and serializes responses. The core pipeline owns canonical request contracts, evidence coordination, structure-database fallback, and stream event generation. Each phase starts with regression tests, includes a branch-vs-`dev` reconciliation check, and ends in a reviewable commit.

**Tech Stack:** FastAPI, Pydantic, async Python, SQLite/Postgres backend abstractions, pytest, loguru, Bandit.

---

## Context

Worktree: `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review`

Approved spec: `Docs/superpowers/specs/2026-04-24-rag-thin-endpoint-followup-design.md`

Baseline `dev` items that must be preserved before refactoring:

- `09560db87 fix: finalize wave 1 data bootstrap hardening`
- `eb251b91e fix: align websearch outbound policy handling`
- `4b376ab41 ralph loop fixes`
- `18049f109 Gate rerank debug snapshots behind explicit opt-in`

Run all commands from the worktree root. Activate the project virtual environment before Python commands:

```bash
source ../../.venv/bin/activate
```

## File Structure

Create:

- `tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py` - regression tests for the current-`dev` analytics backend refresh/bootstrap behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py` - standard RAG contract identity and endpoint-boundary tests.
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py` - import-boundary and structure-DB fallback tests for agentic core ownership.
- `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py` - core-owned streaming event executor.
- `tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py` - streaming event order and endpoint delegation tests.

Modify:

- `tldw_Server_API/app/core/RAG/rag_service/analytics_db.py` - port current-`dev` dynamic backend refresh and per-backend bootstrap tracking.
- `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py` - restore shared outbound policy block coverage from current `dev`.
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py` - restore the database-error fallback regression from current `dev`, adjusted to patch the core-owned resolver after the agentic cleanup.
- `tldw_Server_API/app/core/RAG/rag_service/request_resolution.py` - add one canonical compatibility path for legacy standard-pipeline calls.
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` - consume canonical contracts, own standard evidence coordination, and delegate streaming to core.
- `tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py` - require and use canonical resolved request and retrieval plan objects.
- `tldw_Server_API/app/core/RAG/rag_service/generation_executor.py` - require and use canonical resolved request and retrieval plan objects.
- `tldw_Server_API/app/core/RAG/rag_service/post_retrieval_coordinator.py` - remain core-only and receive canonical standard results before endpoint serialization.
- `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py` - own `AgenticConfig`, agentic toolbox behavior, and structure-DB resolver.
- `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py` - become a compatibility shell that re-exports core-owned agentic types and helper functions.
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` - keep request parsing, auth/dependency handling, response serialization, and streaming HTTP framing only.
- `tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py` - keep rerank debug snapshot opt-in behavior covered after refactors.
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py` - update thin-endpoint assertions to match core-owned evidence coordination and streaming delegation.
- `Docs/RAG/ARCHITECTURE_REVIEW.md` - record the completed branch-vs-`dev` reconciliation and follow-up boundaries.

## Phase 0: Current-`dev` Reconciliation Gate

### Task 1: Preserve Current-`dev` RAG Fixes

**Files:**

- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/analytics_db.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py`

- [x] **Step 1: Confirm the local comparison refs**

Run:

```bash
git rev-parse --short HEAD
git rev-parse --short dev
git merge-base HEAD dev
git diff --name-status dev..HEAD -- tldw_Server_API/app/core/RAG tldw_Server_API/tests/RAG_NEW
```

Expected:

```text
HEAD prints the current codex/rag-module-review commit.
dev prints the local dev commit.
merge-base prints a commit hash.
diff lists the branch RAG changes; review the four baseline files before editing.
```

- [x] **Step 2: Restore the outbound policy regression test from current `dev`**

In `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py`, restore this test body if it is absent:

```python
@pytest.mark.asyncio
async def test_scrape_url_action_surfaces_shared_policy_block(monkeypatch):
    import tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib as article_lib

    async def _fake_scrape_article(target_url: str):  # noqa: ANN001
        return {
            "extraction_successful": False,
            "url": target_url,
            "error": "Blocked by outbound policy",
            "policy_reason": "robots_unreachable",
        }

    monkeypatch.setattr(article_lib, "scrape_article", _fake_scrape_article)

    registry = create_default_registry(enable_url_scraping=True)
    out = await registry.execute(
        "scrape_url",
        {"url": "https://example.com/blocked"},
    )

    assert out.success is False
    assert out.error == "Blocked by outbound policy"
```

- [x] **Step 3: Add analytics backend refresh regression tests**

Create `tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py`:

```python
import configparser
from dataclasses import dataclass

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.RAG.rag_service import analytics_db


@dataclass(frozen=True)
class FakeBackendConfig:
    connection_string: str


class FakePostgresBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, name: str):
        self.name = name
        self.config = FakeBackendConfig(connection_string=f"postgresql://example/{name}")
        self.bootstrap_calls = 0

    def execute(self, *args, **kwargs):
        return []

    def fetch_all(self, *args, **kwargs):
        return []

    def fetch_one(self, *args, **kwargs):
        return None

    def transaction(self):
        class Transaction:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return Transaction()


def test_analytics_database_refreshes_shared_content_backend(monkeypatch, tmp_path):
    first_backend = FakePostgresBackend("first")
    second_backend = FakePostgresBackend("second")
    backend_calls = iter([first_backend, second_backend])

    monkeypatch.setattr(analytics_db, "get_content_backend", lambda config: next(backend_calls))
    monkeypatch.setattr(analytics_db, "load_comprehensive_config", lambda: configparser.ConfigParser())
    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_initialize_database", lambda self: None)
    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_ensure_bootstrap_for_backend",
        lambda self, backend: setattr(backend, "bootstrap_calls", backend.bootstrap_calls + 1),
    )

    db = analytics_db.AnalyticsDatabase(db_path=str(tmp_path / "analytics.db"))

    assert db.backend is first_backend
    assert first_backend.bootstrap_calls == 1
    assert db.backend is second_backend
    assert second_backend.bootstrap_calls == 1


def test_analytics_database_tracks_bootstrap_per_backend_target(monkeypatch, tmp_path):
    backend = FakePostgresBackend("stable")

    monkeypatch.setattr(analytics_db, "get_content_backend", lambda config: backend)
    monkeypatch.setattr(analytics_db, "load_comprehensive_config", lambda: configparser.ConfigParser())
    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_initialize_database", lambda self: None)

    db = analytics_db.AnalyticsDatabase(db_path=str(tmp_path / "analytics.db"))
    db._ensure_bootstrap_for_backend(backend)
    db._ensure_bootstrap_for_backend(backend)

    assert len(db._bootstrapped_backend_targets) == 1
```

- [x] **Step 4: Restore the structure-DB failure fallback regression test**

Create or restore `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py`:

```python
import pytest

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import DatabaseError
from tldw_Server_API.app.core.RAG.rag_service import agentic_execution


def test_open_section_falls_back_to_heuristics_on_database_error(monkeypatch):
    def raise_database_error(*args, **kwargs):
        raise DatabaseError("structure index unavailable")

    monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", raise_database_error)

    toolbox = agentic_execution.AgenticToolbox(
        documents=[
            {
                "id": "doc-1",
                "title": "Paper",
                "content": "Introduction\nOverview text\nResults\nImportant result text\nConclusion",
                "metadata": {},
            }
        ],
        query="result",
    )

    section = toolbox.open_section("doc-1", "Results")

    assert section is not None
    assert section["document_id"] == "doc-1"
    assert section["section_title"] == "Results"
    assert "Important result text" in section["content"]
```

- [x] **Step 5: Verify rerank debug snapshots remain explicitly opt-in**

In `tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py`, keep or restore these test names and assertions:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_snapshots_are_hidden_without_explicit_opt_in(self, monkeypatch):
    monkeypatch.delenv("RAG_INCLUDE_RERANK_DEBUG_DOCUMENTS", raising=False)

    result = await unified_rag_pipeline(
        query="What is the capital of France?",
        top_k=2,
        enable_cache=False,
        enable_reranking=True,
        enable_generation=False,
        debug_mode=True,
    )

    assert isinstance(result, UnifiedRAGResponse)
    assert "pre_rerank_documents" not in (result.metadata or {})
    assert "reranked_documents" not in (result.metadata or {})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_snapshots_include_truncated_content_when_explicitly_enabled(self, monkeypatch):
    monkeypatch.delenv("RAG_INCLUDE_RERANK_DEBUG_DOCUMENTS", raising=False)

    result = await unified_rag_pipeline(
        query="What is the capital of France?",
        top_k=2,
        enable_cache=False,
        enable_reranking=True,
        enable_generation=False,
        debug_mode=True,
        include_rerank_debug_documents=True,
    )

    assert isinstance(result, UnifiedRAGResponse)
    assert result.metadata["pre_rerank_documents"][0]["content"].startswith("Paris is the capital of France.")
    assert "reranked_documents" in result.metadata
```

- [x] **Step 6: Run the new and restored tests before porting**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py::test_scrape_url_action_surfaces_shared_policy_block \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py \
  -q
```

Expected:

```text
At least one assertion fails on this branch before the current-dev code is ported.
The failures point to missing analytics backend refresh/bootstrap behavior, missing outbound policy handling, missing structure-DB fallback wiring, or rerank debug snapshot leakage.
```

- [x] **Step 7: Port the analytics hardening from current `dev`**

In `tldw_Server_API/app/core/RAG/rag_service/analytics_db.py`, port the current-`dev` behavior from `09560db87`:

```python
class AnalyticsDatabase:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _get_analytics_db_path()
        self._config_parser = load_comprehensive_config()
        self._backend = self._resolve_backend()
        self._uses_shared_content_backend = self._is_shared_content_backend(self._backend)
        self._bootstrapped_backend_targets: set[str] = set()
        self._ensure_bootstrap_for_backend(self._backend)
        self._initialize_database()

    @property
    def backend(self):
        if self._uses_shared_content_backend:
            refreshed_backend = self._resolve_backend()
            self._ensure_bootstrap_for_backend(refreshed_backend)
            self._backend = refreshed_backend
        return self._backend

    def _resolve_backend(self):
        return get_content_backend(self._config_parser)

    def _is_shared_content_backend(self, backend) -> bool:
        return getattr(backend, "backend_type", None) in {
            BackendType.POSTGRESQL,
            BackendType.MYSQL,
        }

    def _describe_backend(self, backend) -> str:
        backend_type = getattr(backend, "backend_type", None)
        config = getattr(backend, "config", None)
        connection_string = getattr(config, "connection_string", "")
        return f"{backend_type}:{connection_string}"

    def _ensure_bootstrap_for_backend(self, backend) -> None:
        target = self._describe_backend(backend)
        if target in self._bootstrapped_backend_targets:
            return
        self._bootstrap_backend_schema(backend)
        self._bootstrapped_backend_targets.add(target)
```

Keep the existing table/index creation SQL from the current branch; move that SQL into `_bootstrap_backend_schema(self, backend)` if the current branch does not already expose the helper. All analytics read/write methods must use `self.backend`, not `self._backend`, so shared content backends refresh during runtime.

- [x] **Step 8: Re-run the Phase 0 focused tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py::test_scrape_url_action_surfaces_shared_policy_block \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py \
  -q
```

Expected:

```text
All selected tests pass.
```

- [x] **Step 9: Run the Phase 0 branch-vs-`dev` checkpoint**

Run:

```bash
git diff --name-status dev..HEAD -- \
  tldw_Server_API/app/core/RAG/rag_service/analytics_db.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py
```

Expected:

```text
analytics_db.py differs only because this branch also carries the RAG refactor work.
The research-agent shared-policy test is present.
The structure-DB failure fallback test is present.
The rerank debug snapshot opt-in tests are present.
```

- [x] **Step 10: Commit Phase 0**

Run:

```bash
git add \
  tldw_Server_API/app/core/RAG/rag_service/analytics_db.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py
git commit -m "fix: reconcile rag dev baseline"
```

Expected:

```text
[codex/rag-module-review <hash>] fix: reconcile rag dev baseline
```

## Phase 1: Standard Core Contract Ownership

### Task 2: Thread Canonical Standard Contracts Through Core

**Files:**

- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/request_resolution.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/generation_executor.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/post_retrieval_coordinator.py`

- [x] **Step 1: Write the contract identity tests**

Create `tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py`:

```python
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
    ResolvedRAGRequest,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan


def _resolved_request() -> ResolvedRAGRequest:
    return ResolvedRAGRequest(
        query="What changed?",
        strategy="standard",
        payload={
            "query": "What changed?",
            "strategy": "standard",
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 3,
            "min_score": 0.0,
            "enable_generation": True,
            "include_sources": True,
            "include_metadata": True,
        },
        index_namespace=None,
        rag_profile=None,
        user_id="1",
        feedback_user_id="1",
    )


def _retrieval_plan() -> RetrievalPlan:
    return RetrievalPlan(
        query="What changed?",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace=None,
        collection_names={"media_db": "user_1_media_embeddings"},
    )


@pytest.mark.asyncio
async def test_unified_pipeline_reuses_resolved_request_and_plan(monkeypatch):
    resolved = _resolved_request()
    plan = _retrieval_plan()
    seen = {}

    async def fake_retrieval_phase(**kwargs):
        seen["retrieval_resolved"] = kwargs["resolved_request"]
        seen["retrieval_plan"] = kwargs["retrieval_plan"]
        return SimpleNamespace(
            documents=[],
            sources=[],
            metadata={"retrieval": "ok"},
        )

    async def fake_generation_phase(**kwargs):
        seen["generation_resolved"] = kwargs["resolved_request"]
        seen["generation_plan"] = kwargs["retrieval_plan"]
        return {
            "answer": "generated answer",
            "sources": [],
            "metadata": {"generation": "ok"},
        }

    def fake_coordinate(result, resolved_request, *, coordinator=None):
        seen["coordinator_resolved"] = resolved_request
        return result

    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(unified_pipeline, "coordinate_standard_result_evidence", fake_coordinate)

    result = await unified_pipeline.unified_rag_pipeline(
        query=resolved.query,
        sources=list(plan.sources),
        top_k=plan.top_k,
        search_mode=plan.search_mode,
        enable_generation=True,
        resolved_request=resolved,
        retrieval_plan=plan,
    )

    assert result["answer"] == "generated answer"
    assert seen["retrieval_resolved"] is resolved
    assert seen["retrieval_plan"] is plan
    assert seen["generation_resolved"] is resolved
    assert seen["generation_plan"] is plan
    assert seen["coordinator_resolved"] is resolved


@pytest.mark.asyncio
async def test_unified_pipeline_builds_single_legacy_resolved_request(monkeypatch):
    seen = {}

    async def fake_retrieval_phase(**kwargs):
        seen["retrieval_resolved"] = kwargs["resolved_request"]
        seen["retrieval_plan"] = kwargs["retrieval_plan"]
        return SimpleNamespace(documents=[], sources=[], metadata={})

    async def fake_generation_phase(**kwargs):
        seen["generation_resolved"] = kwargs["resolved_request"]
        seen["generation_plan"] = kwargs["retrieval_plan"]
        return {"answer": "legacy answer", "sources": [], "metadata": {}}

    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(unified_pipeline, "coordinate_standard_result_evidence", lambda result, resolved_request, *, coordinator=None: result)

    result = await unified_pipeline.unified_rag_pipeline(
        query="legacy query",
        top_k=5,
        search_mode="fts",
        enable_generation=True,
    )

    assert result["answer"] == "legacy answer"
    assert seen["retrieval_resolved"] is seen["generation_resolved"]
    assert seen["retrieval_plan"] is seen["generation_plan"]
    assert seen["retrieval_resolved"].query == "legacy query"
    assert seen["retrieval_plan"].top_k == 5
```

These constructors match the current local definitions of `ResolvedRAGRequest` and `RetrievalPlan`.

- [x] **Step 2: Run the contract tests to verify failure**

Run:

```bash
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py -q
```

Expected:

```text
FAIL: the current pipeline rebuilds or omits canonical contract objects before all standard core phases see them.
```

- [x] **Step 3: Add a legacy compatibility resolver at the pipeline boundary**

In `tldw_Server_API/app/core/RAG/rag_service/request_resolution.py`, add a helper that converts legacy `unified_rag_pipeline` keyword arguments into the same `ResolvedRAGRequest` type used by endpoint requests:

```python
def resolve_legacy_standard_pipeline_request(
    *,
    query: str,
    search_mode: str,
    top_k: int,
    sources: Optional[list[str]] = None,
    min_score: float = 0.0,
    index_namespace: Optional[str] = None,
    rag_profile: Optional[str] = None,
    user_id: Optional[str] = None,
    feedback_user_id: Optional[str] = None,
    enable_generation: bool = True,
    include_sources: bool = True,
    include_metadata: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> ResolvedRAGRequest:
    payload = dict(metadata or {})
    payload.update(
        {
            "query": query,
            "strategy": "standard",
            "sources": list(sources or ["media_db"]),
            "search_mode": search_mode,
            "top_k": top_k,
            "min_score": min_score,
            "enable_generation": enable_generation,
            "include_sources": include_sources,
            "include_metadata": include_metadata,
        }
    )
    return ResolvedRAGRequest(
        query=query,
        strategy="standard",
        payload=payload,
        index_namespace=index_namespace,
        rag_profile=rag_profile,
        user_id=user_id,
        feedback_user_id=feedback_user_id or user_id,
    )
```

Do not create `SimpleNamespace` request contracts in executors.

- [x] **Step 4: Update `unified_pipeline.py` to own standard orchestration**

In `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`, add `resolved_request` next to the existing internal `retrieval_plan` parameter:

```python
    index_namespace: Optional[str] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
    resolved_request: Optional[ResolvedRAGRequest] = None,
```

Add a retrieval-only result helper in the same module:

```python
def build_retrieval_only_result(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    retrieval_result: RetrievedEvidence,
) -> RAGResult:
    metadata = dict(retrieval_result.metadata or {})
    metadata.setdefault(
        "retrieval_plan",
        {
            "query": retrieval_plan.query,
            "sources": list(retrieval_plan.sources),
            "search_mode": retrieval_plan.search_mode,
            "top_k": retrieval_plan.top_k,
            "index_namespace": retrieval_plan.index_namespace,
        },
    )
    return RAGResult(
        query=resolved_request.query,
        documents=list(retrieval_result.documents),
        metadata=metadata,
    )
```

Normalize once at the top of the standard path:

```python
if resolved_request is None:
    resolved_request = resolve_legacy_standard_pipeline_request(
        query=query,
        search_mode=search_mode,
        top_k=top_k,
        sources=sources,
        min_score=min_score,
        index_namespace=index_namespace,
        rag_profile=rag_profile,
        user_id=user_id,
        feedback_user_id=feedback_user_id,
        enable_generation=enable_generation,
        include_sources=include_sources,
        include_metadata=include_metadata,
        metadata=metadata,
    )

if retrieval_plan is None:
    retrieval_plan = build_retrieval_plan(resolved_request)

derived_evidence = await execute_retrieval_phase(
    resolved_request=resolved_request,
    retrieval_plan=retrieval_plan,
    retriever=retriever,
    retrieval_config=retrieval_config,
    allowed_media_ids=include_media_ids,
    allowed_note_ids=include_note_ids,
)

if bool((resolved_request.payload or {}).get("enable_generation", enable_generation)):
    result = await execute_generation_phase(
        resolved_request=resolved_request,
        derived_evidence=derived_evidence,
        generate_answer_fn=generate_answer_fn,
        generation_context=generation_context,
    )
else:
    result = build_retrieval_only_result(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        retrieval_result=derived_evidence,
    )

return coordinate_standard_result_evidence(result, resolved_request)
```

Preserve the public return shape.

- [x] **Step 5: Update standard executors to accept canonical objects**

In `retrieval_executor.py`, make `execute_retrieval_phase` require `resolved_request` and `retrieval_plan` keyword arguments:

```python
async def execute_retrieval_phase(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    retriever: Any,
    retrieval_config: Any | None = None,
    allowed_media_ids: list[int] | None = None,
    allowed_note_ids: list[str] | None = None,
) -> RetrievedEvidence:
    query = resolved_request.query
    top_k = retrieval_plan.top_k
    search_mode = retrieval_plan.search_mode
    sources = retrieval_plan.sources
```

In `generation_executor.py`, make `execute_generation_phase` consume the same objects:

```python
async def execute_generation_phase(
    *,
    resolved_request: ResolvedRAGRequest,
    derived_evidence: Any,
    generate_answer_fn: Callable[..., Awaitable[Any]],
    generation_context: str,
) -> RAGResult:
    generation_query = resolved_request.query
    source_documents = derived_evidence.documents
    generation_prompt = (resolved_request.payload or {}).get("generation_prompt")
```

Remove fallback construction of request-like objects inside these executors. Compatibility belongs in `resolve_legacy_standard_pipeline_request`.

- [x] **Step 6: Run the standard contract tests**

Run:

```bash
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py -q
```

Expected:

```text
All tests in test_standard_core_contract_threading.py pass.
```

- [x] **Step 7: Run existing standard RAG tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_retrieval_plan_usage.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_post_retrieval_coordinator.py \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py \
  -q
```

Expected:

```text
All selected standard pipeline tests pass.
```

- [x] **Step 8: Run the Phase 1 branch-vs-`dev` checkpoint**

Run:

```bash
git diff --stat dev..HEAD -- \
  tldw_Server_API/app/core/RAG/rag_service/request_resolution.py \
  tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py \
  tldw_Server_API/app/core/RAG/rag_service/generation_executor.py \
  tldw_Server_API/app/core/RAG/rag_service/post_retrieval_coordinator.py \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```

Expected:

```text
The diff is limited to the planned standard-core contract threading and previously approved branch work.
```

- [x] **Step 9: Commit Phase 1**

Run:

```bash
git add \
  tldw_Server_API/app/core/RAG/rag_service/request_resolution.py \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py \
  tldw_Server_API/app/core/RAG/rag_service/generation_executor.py \
  tldw_Server_API/app/core/RAG/rag_service/post_retrieval_coordinator.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py
git commit -m "refactor: thread rag standard contracts through core"
```

Expected:

```text
[codex/rag-module-review <hash>] refactor: thread rag standard contracts through core
```

### Task 3: Remove Endpoint Standard Post-Processing

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py`
- Modify: `Docs/RAG/ARCHITECTURE_REVIEW.md`

- [x] **Step 1: Write the endpoint-boundary regression test**

In `tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py`, add:

```python
import inspect

from tldw_Server_API.app.api.v1.endpoints import rag_unified


def test_standard_endpoint_does_not_import_core_evidence_coordinator():
    source = inspect.getsource(rag_unified)

    assert "coordinate_standard_result_evidence" not in source
    assert "post_retrieval_coordinator" not in source
```

This test guards the endpoint boundary by source inspection because the regression was architectural: the endpoint was doing core post-retrieval work.

- [x] **Step 2: Run the endpoint cleanup test to verify failure**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py::test_standard_endpoint_does_not_import_core_evidence_coordinator \
  -q
```

Expected:

```text
FAIL before endpoint imports and post-processing are removed.
```

- [x] **Step 3: Remove standard post-processing from the endpoint**

In `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`:

```python
# Remove imports like this from the endpoint:
from tldw_Server_API.app.core.RAG.rag_service.post_retrieval_coordinator import (
    coordinate_standard_result_evidence,
)
```

Replace any endpoint-side standard evidence coordination block with direct serialization of the core result:

```python
core_result = await unified_rag_pipeline(
    query=resolved_request.query,
    sources=list(retrieval_plan.sources),
    top_k=retrieval_plan.top_k,
    search_mode=retrieval_plan.search_mode,
    resolved_request=resolved_request,
    retrieval_plan=retrieval_plan,
    enable_generation=bool((resolved_request.payload or {}).get("enable_generation", True)),
)

return rag_result_to_response(rag_result_from_unified_search_result(core_result))
```

The endpoint must only validate, resolve, delegate, and map.

- [x] **Step 4: Update architecture review notes**

In `Docs/RAG/ARCHITECTURE_REVIEW.md`, add a short entry under the RAG review record:

```markdown
### Thin Endpoint Follow-Up

- Standard RAG evidence coordination now occurs inside `app/core/RAG/rag_service/unified_pipeline.py`.
- `app/api/v1/endpoints/rag_unified.py` remains an HTTP adapter: validation, dependency resolution, delegation, response mapping, and streaming framing.
- Current-`dev` RAG fixes were reconciled before this refactor: analytics backend bootstrap, shared outbound policy handling, structure-DB fallback, and rerank debug snapshot gating.
```

- [x] **Step 5: Run endpoint and contract tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py \
  -q
```

Expected:

```text
All selected endpoint-boundary and standard contract tests pass.
```

- [x] **Step 6: Commit Phase 1 endpoint cleanup**

Run:

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py \
  Docs/RAG/ARCHITECTURE_REVIEW.md
git commit -m "refactor: keep rag standard endpoint thin"
```

Expected:

```text
[codex/rag-module-review <hash>] refactor: keep rag standard endpoint thin
```

## Phase 2: Agentic Shell Untangling

### Task 4: Move Agentic Ownership Into Core Execution

**Files:**

- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py`

- [x] **Step 1: Write the import-boundary and fallback tests**

Create `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py`:

```python
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import DatabaseError
from tldw_Server_API.app.core.RAG.rag_service import agentic_chunker, agentic_execution


def test_agentic_execution_does_not_import_shell_module():
    source = Path(agentic_execution.__file__).read_text()

    assert "agentic_chunker" not in source


def test_agentic_chunker_reexports_core_config_and_toolbox():
    assert agentic_chunker.AgenticConfig is agentic_execution.AgenticConfig
    assert agentic_chunker.AgenticToolbox is agentic_execution.AgenticToolbox


def test_open_section_uses_core_structure_db_resolver_and_falls_back(monkeypatch):
    def raise_database_error(*args, **kwargs):
        raise DatabaseError("structure index unavailable")

    monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", raise_database_error)

    toolbox = agentic_execution.AgenticToolbox(
        documents=[
            {
                "id": "doc-1",
                "title": "Paper",
                "content": "Intro\nText\nMethods\nMethod text\nResults\nResult text",
                "metadata": {},
            }
        ],
        query="result",
    )

    result = toolbox.open_section("doc-1", "Results")

    assert result is not None
    assert result["document_id"] == "doc-1"
    assert result["section_title"] == "Results"
    assert "Result text" in result["content"]
```

- [x] **Step 2: Run the agentic ownership tests to verify failure**

Run:

```bash
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py -q
```

Expected:

```text
FAIL before AgenticConfig and the structure-DB resolver are owned by agentic_execution.py without reverse shell imports.
```

- [x] **Step 3: Move `AgenticConfig` into `agentic_execution.py`**

In `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`, define the config near the other public agentic core types:

```python
@dataclass
class AgenticConfig:
    top_k_docs: int = 3
    window_chars: int = 1200
    max_tokens_read: int = 6000
    max_tool_calls: int = 8
    extractive_only: bool = True
    quote_spans: bool = True
    enable_tools: bool = False
    use_llm_planner: bool = False
    time_budget_sec: float | None = None
    cache_ttl_sec: int = 600
    debug_trace: bool = False
    enable_query_decomposition: bool = False
    subgoal_max: int = 3
    enable_semantic_within: bool = True
    semantic_dim: int = 2048
    enable_section_index: bool = True
    prefer_structural_anchors: bool = True
    enable_table_support: bool = True
    table_trigger_keywords: tuple[str, ...] = ("table", "figure", "tabular", "dataset")
    table_min_bar_count: int = 3
    agentic_enable_vlm_late_chunking: bool = False
    agentic_vlm_backend: str | None = None
    agentic_vlm_detect_tables_only: bool = True
    agentic_vlm_max_pages: int | None = None
    agentic_vlm_late_chunk_top_k_docs: int = 2
    agentic_use_provider_embeddings_within: bool = False
    agentic_provider_embedding_model_id: str | None = None
    adaptive_budgets: bool = True
    coverage_target: float = 0.8
    min_corroborating_docs: int = 2
    max_redundancy: float = 0.9
    enable_metrics: bool = True
```

The moved type preserves the current constructor compatibility for existing API and tests.

- [x] **Step 4: Move the structure-DB resolver into `agentic_execution.py`**

In `agentic_execution.py`, own the resolver and catch database lookup errors inside core:

```python
def _get_media_db_for_structure():
    from tldw_Server_API.app.core.DB_Management.media_db_factory import get_media_db

    return get_media_db()


def _lookup_section_from_structure_index(document_id: str, section_title: str) -> Optional[dict[str, Any]]:
    try:
        media_db = _get_media_db_for_structure()
        return media_db.get_document_section(document_id=document_id, section_title=section_title)
    except DatabaseError as exc:
        logger.warning(
            "Structure index lookup failed for document_id={} section_title={}: {}",
            document_id,
            section_title,
            exc,
        )
        return None
```

Keep existing method names and database calls where they differ locally. The invariant is that `AgenticToolbox.open_section()` first attempts the core resolver and then falls back to heuristic section extraction when the resolver returns `None` or raises a database-layer exception.

- [x] **Step 5: Turn `agentic_chunker.py` into a compatibility shell**

In `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`, remove owned definitions that now live in core execution and re-export them:

```python
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import (
    AgenticConfig,
    AgenticToolbox,
    _get_media_db_for_structure,
    agentic_rag_pipeline,
)

__all__ = [
    "AgenticConfig",
    "AgenticToolbox",
    "_get_media_db_for_structure",
    "agentic_rag_pipeline",
]
```

Keep chunking-only helpers in `agentic_chunker.py` if they are not part of execution/toolbox ownership. Do not import `agentic_chunker.py` from `agentic_execution.py`.

- [x] **Step 6: Update tests to patch the core resolver**

In `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py` and `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py`, patch:

```python
monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", fake_resolver)
```

Do not patch:

```python
monkeypatch.setattr(agentic_chunker, "_get_media_db_for_structure", fake_resolver)
```

The shell re-export exists for import compatibility; core tests must target the core module.

- [x] **Step 7: Run agentic unit tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py \
  -q
```

Expected:

```text
All selected agentic ownership and fallback tests pass.
```

- [x] **Step 8: Run agentic API and streaming parity tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_stream_parity.py \
  -q
```

Expected:

```text
All selected agentic integration and stream parity tests pass, or tests are skipped only for declared local-service prerequisites.
```

- [x] **Step 9: Run the Phase 2 branch-vs-`dev` checkpoint**

Run:

```bash
git diff --stat dev..HEAD -- \
  tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py
```

Expected:

```text
The diff shows agentic execution owning config/toolbox/resolver behavior and agentic_chunker.py retaining only compatibility-shell or chunking responsibilities.
```

- [x] **Step 10: Commit Phase 2**

Run:

```bash
git add \
  tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py
git commit -m "refactor: move rag agentic ownership into core"
```

Expected:

```text
[codex/rag-module-review <hash>] refactor: move rag agentic ownership into core
```

## Phase 3: Streaming Executor Extraction

### Task 5: Move Streaming Event Generation Into Core

**Files:**

- Create: `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py`
- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_stream_parity.py`

- [x] **Step 1: Write streaming executor unit tests**

Create `tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py`:

```python
import pytest

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import stream_rag_events


def _resolved_request(strategy: str = "standard") -> ResolvedRAGRequest:
    return ResolvedRAGRequest(
        query="stream query",
        strategy=strategy,
        payload={
            "query": "stream query",
            "strategy": strategy,
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 3,
            "min_score": 0.0,
            "enable_generation": True,
        },
        index_namespace=None,
        rag_profile=None,
        user_id="1",
        feedback_user_id="1",
    )


def _retrieval_plan() -> RetrievalPlan:
    return RetrievalPlan(
        query="stream query",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace=None,
        collection_names={"media_db": "user_1_media_embeddings"},
    )


@pytest.mark.asyncio
async def test_stream_rag_events_wraps_standard_core_result_in_order():
    async def fake_standard_pipeline(**kwargs):
        return {
            "answer": "answer text",
            "sources": [{"id": "doc-1"}],
            "metadata": {"model": "test"},
        }

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=fake_standard_pipeline,
        )
    ]

    assert [event["type"] for event in events] == ["start", "result", "done"]
    assert events[0]["payload"]["mode"] == "standard"
    assert events[1]["payload"]["answer"] == "answer text"
    assert events[2]["payload"] == {}


@pytest.mark.asyncio
async def test_stream_rag_events_emits_structured_error():
    async def failing_standard_pipeline(**kwargs):
        raise RuntimeError("stream failed")

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=failing_standard_pipeline,
        )
    ]

    assert [event["type"] for event in events] == ["start", "error", "done"]
    assert events[1]["payload"]["message"] == "stream failed"
```

The event type assertions define the stable core stream boundary.

- [x] **Step 2: Write endpoint streaming delegation test**

In `tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py`, add:

```python
import inspect

from tldw_Server_API.app.api.v1.endpoints import rag_unified


def test_streaming_endpoint_delegates_event_generation_to_core():
    source = inspect.getsource(rag_unified)
    stream_source = source[source.find("async def unified_search_stream_endpoint") :]

    assert "stream_rag_events" in source
    assert "yield json.dumps" in source
    assert "unified_rag_pipeline(" not in stream_source
    assert "agentic_rag_pipeline(" not in stream_source
```

The invariant is that the streaming endpoint frames core events; it does not run standard or agentic orchestration inline.

- [x] **Step 3: Run streaming tests to verify failure**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py::test_streaming_endpoint_delegates_event_generation_to_core \
  -q
```

Expected:

```text
FAIL before streaming_executor.py exists and before the endpoint delegates stream event generation.
```

- [x] **Step 4: Add `streaming_executor.py`**

Create `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import agentic_rag_pipeline
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

RAGStreamEvent = dict[str, Any]
PipelineCallable = Callable[..., Awaitable[dict[str, Any]]]


async def stream_rag_events(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    standard_pipeline: PipelineCallable = unified_rag_pipeline,
    agentic_pipeline: PipelineCallable = agentic_rag_pipeline,
    extra_context: dict[str, Any] | None = None,
) -> AsyncIterator[RAGStreamEvent]:
    context = extra_context or {}

    yield {
        "type": "start",
        "payload": {
            "mode": resolved_request.strategy,
            "query": resolved_request.query,
        },
    }

    try:
        if resolved_request.strategy == "agentic":
            result = await agentic_pipeline(
                query=resolved_request.query,
                resolved_request=resolved_request,
                retrieval_plan=retrieval_plan,
                **context,
            )
        else:
            result = await standard_pipeline(
                query=resolved_request.query,
                top_k=retrieval_plan.top_k,
                search_mode=retrieval_plan.search_mode,
                resolved_request=resolved_request,
                retrieval_plan=retrieval_plan,
                enable_generation=bool((resolved_request.payload or {}).get("enable_generation", True)),
                **context,
            )

        yield {
            "type": "result",
            "payload": result,
        }
    except Exception as exc:
        logger.exception("RAG streaming failed")
        yield {
            "type": "error",
            "payload": {
                "message": str(exc),
                "error_type": exc.__class__.__name__,
            },
        }
    finally:
        yield {
            "type": "done",
            "payload": {},
        }
```

Keep provider, DB, auth, and cancellation objects inside `extra_context` so the endpoint can pass dependencies without the executor depending on FastAPI types.

- [x] **Step 5: Update the endpoint to frame core stream events**

In `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`, import:

```python
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import stream_rag_events
```

Replace inline streaming orchestration with:

```python
async def event_generator():
    async for event in stream_rag_events(
        resolved_request=resolved_request,
        retrieval_plan=stream_bundle.retrieval_plan,
        extra_context=stream_pipeline_kwargs,
    ):
        yield json.dumps(event) + "\n"

return StreamingResponse(
    event_generator(),
    media_type="application/x-ndjson",
)
```

The endpoint must not contain branching logic for standard vs agentic stream orchestration.

- [x] **Step 6: Run streaming executor tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py::test_streaming_endpoint_delegates_event_generation_to_core \
  -q
```

Expected:

```text
All selected streaming executor and endpoint delegation tests pass.
```

- [x] **Step 7: Run existing streaming parity tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_stream_parity.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py \
  -q
```

Expected:

```text
All selected streaming parity and agentic API tests pass, or tests are skipped only for declared local-service prerequisites.
```

- [x] **Step 8: Run the Phase 3 branch-vs-`dev` checkpoint**

Run:

```bash
git diff --stat dev..HEAD -- \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_stream_parity.py
```

Expected:

```text
The diff shows a new core streaming executor and endpoint changes limited to NDJSON framing plus dependency passing.
```

- [x] **Step 9: Commit Phase 3**

Run:

```bash
git add \
  tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_stream_parity.py
git commit -m "refactor: move rag streaming events into core"
```

Expected:

```text
[codex/rag-module-review <hash>] refactor: move rag streaming events into core
```

## Phase 4: Final Verification and Review Record

### Task 6: Verify the Full Follow-Up Slice

**Files:**

- Modify: `Docs/RAG/ARCHITECTURE_REVIEW.md`

- [x] **Step 1: Update the review record**

In `Docs/RAG/ARCHITECTURE_REVIEW.md`, append:

```markdown
### Follow-Up Verification Record

- Phase 0 reconciled current `dev` RAG fixes before refactoring.
- Phase 1 moved standard evidence coordination and canonical contract threading into core.
- Phase 2 moved agentic config/toolbox/structure-DB fallback ownership into `agentic_execution.py`.
- Phase 3 moved streaming event generation into `streaming_executor.py`.
- Endpoint responsibilities are now validation, dependency resolution, core delegation, response mapping, and HTTP stream framing.
```

- [x] **Step 2: Run focused RAG regression tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py::test_scrape_url_action_surfaces_shared_policy_block \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_db_error_fallback.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_endpoint_contract_cleanup.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_shell_ownership.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_retrieval_plan_usage.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_post_retrieval_coordinator.py \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py \
  -q
```

Expected:

```text
All selected focused tests pass.
```

- [x] **Step 3: Run broader RAG tests**

Run:

```bash
python -m pytest tldw_Server_API/tests/RAG_NEW -q
```

Expected:

```text
All RAG_NEW tests pass, or environment-dependent tests are skipped with declared skip reasons.
```

- [x] **Step 4: Run security validation on touched code**

Run:

```bash
python -m bandit -r \
  tldw_Server_API/app/core/RAG/rag_service \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  -f json \
  -o /tmp/bandit_rag_thin_endpoint_followup.json
```

Expected:

```text
Bandit completes successfully and reports no new high or medium findings in touched code.
```

- [x] **Step 5: Run final branch-vs-`dev` review**

Run:

```bash
git diff --name-status dev..HEAD -- \
  tldw_Server_API/app/core/RAG/rag_service \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/tests/RAG_NEW \
  Docs/RAG/ARCHITECTURE_REVIEW.md
```

Expected:

```text
The diff contains the planned reconciliation tests, standard core ownership, agentic ownership, streaming executor, endpoint thinning, and review-record updates.
No current-dev regression tests are deleted.
```

- [x] **Step 6: Check formatting and whitespace**

Run:

```bash
git diff --check
```

Expected:

```text
No output.
```

- [x] **Step 7: Commit final review record**

Run:

```bash
git add Docs/RAG/ARCHITECTURE_REVIEW.md
git commit -m "docs: record rag thin endpoint follow-up"
```

Expected:

```text
[codex/rag-module-review <hash>] docs: record rag thin endpoint follow-up
```

- [x] **Step 8: Capture final status**

Run:

```bash
git status --short
git log --oneline --decorate -6
```

Expected:

```text
Only pre-existing unrelated untracked files remain.
The latest commits are the Phase 0 through Phase 4 follow-up commits.
```
