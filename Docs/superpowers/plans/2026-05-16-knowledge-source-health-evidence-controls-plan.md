# Knowledge Source Health Evidence Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only pre-query source health, clearer evidence controls, answer trust summaries, and recovery copy to `/knowledge` without turning it into a source CRUD/import hub or adding durable evidence persistence.

**Architecture:** Add a focused backend source-health contract that is separate from existing per-search `metadata.source_status`. Surface that contract through the existing tldw API client and KnowledgeQA provider, then render compact health summaries in the existing context/source controls and compact trust summaries in existing answer/evidence surfaces. Reuse current `SourceCard`, `AnswerPanel`, `EvidenceRail`, and recovery components instead of introducing a new panel or persistence model.

**Tech Stack:** FastAPI, Pydantic, existing RAG retrieval/source registry, React, TypeScript, shared `@tldw/ui` KnowledgeQA components, Vitest, Playwright, pytest, Bandit.

---

## Source Spec

Implement from: `Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md`

Hard boundaries:

- `/knowledge` remains QA-only.
- Do not add inline source creation/import/edit/delete controls to `/knowledge`.
- Do not add server-backed saved evidence, shared evidence sets, or cross-device saved views.
- Do not rename or replace existing post-query `metadata.source_status` semantics.
- Evidence actions must reuse existing handoffs. `Save to note` appears only if the existing note handoff preserves source backlinks without new evidence persistence.

## File Structure

Backend:

- Create: `tldw_Server_API/app/core/RAG/rag_service/source_health.py`
  - Owns canonical source-health construction and keeps it separate from search-response `source_status`.
- Modify: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
  - Adds Pydantic response models and literal status types.
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
  - Adds `GET /api/v1/rag/source-health` with search-like auth, but no query ledger usage or retriever/database creation.
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py`
  - Pure helper/schema contract tests.
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py`
  - FastAPI endpoint shape/auth/sanitization tests.
- Regression Test: `tldw_Server_API/tests/RAG_NEW/unit/test_source_contract.py`
  - Add one assertion that source-health work does not alter canonical source id normalization.
- Regression Test: existing tests around `_build_source_status`, if available, or add a focused assertion in `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py`
  - Protects existing post-query `metadata.source_status` values: `searched`, `empty`, `unavailable`.

Frontend service/types:

- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
  - Adds `ragSourceHealth()` to the domain client.
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Adds the legacy/direct `ragSourceHealth()` method if this generated-style client still carries direct method mirrors.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`
  - Adds `KnowledgeSourceHealth`, `KnowledgeSourceHealthState`, and provider state/action fields distinct from `KnowledgeSourceStatus`.
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/sourceHealth.ts`
  - Normalizes backend health payloads, builds aggregate summaries, labels statuses, and guards unknown/partial responses.
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/sourceHealth.test.ts`
  - Covers normalization, summary, compatibility with existing post-query status, and label copy.
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.rag-source-health.test.ts`
  - Verifies client path and method.

Frontend UI:

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
  - Loads pre-query source health after server readiness, stores non-blocking load state, exposes retry action.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx`
  - Passes source-health state into `CompactToolbar`, `KnowledgeContextBar`, `KnowledgeReadyState`, and recovery components as needed.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`
  - Renders aggregate health summary and per-source status chips.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/CompactToolbar.tsx`
  - Renders compact health summary for simple/mobile contexts.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx`
  - Uses source-health copy for first-run/empty source readiness without adding inline import controls.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
  - Adds compact answer trust summary from selected sources, citations, web fallback, model/provider, source-health caveats, and existing search details.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceCard.tsx`
  - Audit existing copy/open/workspace actions; relabel `Copy text` to `Copy excerpt` only where evidence context benefits, avoiding duplicate actions.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx`
  - Adds compact action affordance or hint without adding a separate evidence persistence surface.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx`
  - Uses pre-query source health plus existing post-query diagnostics without confusing their labels.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/LowQualityRecoveryBanner.tsx`
  - Adds health-aware copy when selected sources are stale/unavailable/unknown.

Frontend tests:

- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.source-health.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx`

Extension and browser coverage:

- Test: `apps/tldw-frontend/__tests__/extension/knowledge-route-parity.test.ts`
- Test: `apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts` only if imports/lazy boundaries change.
- Optional browser smoke: `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts` with local server when viable.

Docs/backlog:

- Modify: `Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md` only if implementation discovers a design mismatch.
- Modify: Backlog task for implementation, to be created before code changes.

## Task 1: Backend Source Health Contract

**Files:**
- Create: `tldw_Server_API/app/core/RAG/rag_service/source_health.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_source_contract.py`

- [ ] **Step 1: Create the implementation Backlog task**

Use Backlog before code edits:

```bash
backlog task create "Implement /knowledge source health and evidence controls" \
  --status "In Progress" \
  --priority high \
  --labels "webui,knowledge,ux,feature" \
  --doc "Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md" \
  --doc "Docs/superpowers/plans/2026-05-16-knowledge-source-health-evidence-controls-plan.md" \
  --ac "GET /api/v1/rag/source-health returns safe pre-query health for canonical Knowledge QA sources without altering search-response metadata.source_status." \
  --ac "Knowledge QA shows source health before search and keeps search usable when health loading fails." \
  --ac "Knowledge QA answer and evidence surfaces show compact trust/evidence controls without adding durable evidence persistence." \
  --ac "Focused backend, frontend, extension parity, diff-check, and Bandit verification are recorded."
```

Expected: a new task id is created. Record it in commit/PR notes.

- [ ] **Step 2: Write failing source-health unit tests**

Create `tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py`:

```python
import pytest

from tldw_Server_API.app.core.RAG.rag_service.source_health import (
    CANONICAL_KNOWLEDGE_SOURCE_IDS,
    build_source_health_entries,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


pytestmark = pytest.mark.unit


def test_source_health_returns_all_canonical_knowledge_sources() -> None:
    response = build_source_health_entries(configured_sources=set())

    assert [entry.source_id for entry in response] == list(CANONICAL_KNOWLEDGE_SOURCE_IDS)


def test_source_health_marks_configured_sources_searchable() -> None:
    response = build_source_health_entries(
        configured_sources={DataSource.MEDIA_DB, DataSource.NOTES}
    )
    by_source = {entry.source_id: entry for entry in response}

    assert by_source["media_db"].available is True
    assert by_source["media_db"].searchable is True
    assert by_source["media_db"].index_status == "ready"
    assert by_source["media_db"].embedding_status in {"unknown", "not_applicable", "ready"}
    assert by_source["notes"].available is True
    assert by_source["prompts"].available is False
    assert by_source["prompts"].disabled_reason == "no_retriever_configured"


def test_source_health_does_not_expose_content_or_arbitrary_metadata() -> None:
    response = build_source_health_entries(
        configured_sources={DataSource.MEDIA_DB},
        unsafe_metadata={"media_db": {"title": "Secret title", "content": "secret"}},
    )
    payload = [entry.model_dump() for entry in response]

    assert "Secret title" not in repr(payload)
    assert "secret" not in repr(payload)
```

Expected: FAIL because `source_health.py` does not exist yet.

- [ ] **Step 3: Run the failing unit test**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py
```

Expected: FAIL with import/module error.

- [ ] **Step 4: Add Pydantic source-health response models**

Modify `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py` near the existing RAG response models:

```python
KnowledgeSourceIndexStatus = Literal[
    "ready",
    "indexing",
    "stale",
    "empty",
    "unavailable",
    "error",
    "unknown",
]
KnowledgeSourceEmbeddingStatus = Literal[
    "ready",
    "indexing",
    "missing",
    "unavailable",
    "not_applicable",
    "error",
    "unknown",
]


class KnowledgeSourceHealthEntry(BaseModel):
    source_id: str
    label: str
    available: bool
    searchable: bool
    item_count: Optional[int] = None
    indexed_count: Optional[int] = None
    last_updated: Optional[str] = None
    last_indexed: Optional[str] = None
    index_status: KnowledgeSourceIndexStatus = "unknown"
    embedding_status: KnowledgeSourceEmbeddingStatus = "unknown"
    disabled_reason: Optional[str] = None
    workspace_scoped: bool = False
    hidden_by_default: bool = False
    privacy_note: Optional[str] = None


class KnowledgeSourceHealthResponse(BaseModel):
    sources: list[KnowledgeSourceHealthEntry]
```

Keep this model separate from search response metadata and do not change `UnifiedRAGResponse`.

- [ ] **Step 5: Implement the source-health helper**

Create `tldw_Server_API/app/core/RAG/rag_service/source_health.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Set as AbstractSet
from typing import Any

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    KnowledgeSourceHealthEntry,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

CANONICAL_KNOWLEDGE_SOURCE_IDS: tuple[str, ...] = (
    "media_db",
    "notes",
    "chats",
    "characters",
    "kanban",
    "prompts",
    "world_books",
    "dictionaries",
)

_SOURCE_LABELS: dict[str, str] = {
    "media_db": "Documents & Media",
    "notes": "Notes",
    "chats": "Chats",
    "characters": "Characters",
    "kanban": "Task Boards",
    "prompts": "Prompts",
    "world_books": "World Books",
    "dictionaries": "Dictionaries",
}

_SOURCE_TO_DATASOURCE: dict[str, DataSource] = {
    "media_db": DataSource.MEDIA_DB,
    "notes": DataSource.NOTES,
    "chats": DataSource.CHAT_HISTORY,
    "characters": DataSource.CHARACTER_CARDS,
    "kanban": DataSource.KANBAN,
    "prompts": DataSource.PROMPTS,
    "world_books": DataSource.WORLD_BOOKS,
    "dictionaries": DataSource.DICTIONARIES,
}


def build_source_health_entries(
    *,
    configured_sources: AbstractSet[DataSource],
    unsafe_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[KnowledgeSourceHealthEntry]:
    """Build safe pre-query source readiness entries for Knowledge QA."""
    del unsafe_metadata  # Explicitly ignored to prevent accidental metadata leaks.
    entries: list[KnowledgeSourceHealthEntry] = []
    for source_id in CANONICAL_KNOWLEDGE_SOURCE_IDS:
        data_source = _SOURCE_TO_DATASOURCE[source_id]
        configured = data_source in configured_sources
        entries.append(
            KnowledgeSourceHealthEntry(
                source_id=source_id,
                label=_SOURCE_LABELS[source_id],
                available=configured,
                searchable=configured,
                index_status="ready" if configured else "unavailable",
                embedding_status="unknown" if configured else "unavailable",
                disabled_reason=None if configured else "no_retriever_configured",
            )
        )
    return entries
```

V1 intentionally leaves counts/timestamps null unless cheap, safe source-specific counts are added later.

- [ ] **Step 6: Run source-health unit tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py
```

Expected: PASS.

- [ ] **Step 7: Write failing endpoint test**

Create `tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py`:

```python
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.main import app


pytestmark = pytest.mark.integration


@pytest.fixture()
def authorized_client():
    async def _fake_get_auth_principal(_request: Request) -> AuthPrincipal:  # type: ignore[override]
        return AuthPrincipal(
            kind="service",
            user_id="0",
            api_key_id=None,
            subject="service:rag-source-health-test",
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=["media.read"],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_rag_source_health_returns_safe_canonical_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import rag_unified
    monkeypatch.setattr(
        rag_unified,
        "MultiDatabaseRetriever",
        lambda *_, **__: pytest.fail("source health must not instantiate retrievers"),
    )
    monkeypatch.setattr(rag_unified, "_resolve_existing_source_db_paths", lambda *_: {})

    with TestClient(app) as client:
        response = client.get("/api/v1/rag/source-health")

    assert response.status_code in (200, 401, 403)
    if response.status_code == 200:
        payload = response.json()
        assert "sources" in payload
        assert {entry["source_id"] for entry in payload["sources"]} >= {"media_db", "notes"}
        assert "content" not in repr(payload).lower()


def test_authorized_rag_source_health_returns_safe_canonical_shape(
    authorized_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import rag_unified
    monkeypatch.setattr(
        rag_unified,
        "MultiDatabaseRetriever",
        lambda *_, **__: pytest.fail("source health must not instantiate retrievers"),
    )
    monkeypatch.setattr(rag_unified, "_resolve_existing_source_db_paths", lambda *_: {})

    response = authorized_client.get("/api/v1/rag/source-health")

    assert response.status_code == 200
    payload = response.json()
    assert {entry["source_id"] for entry in payload["sources"]} >= {"media_db", "notes"}
    assert "content" not in repr(payload).lower()
```

Use existing auth-override patterns from `tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py`; do not weaken endpoint permissions to make the test pass.

Expected: FAIL because endpoint does not exist.

- [ ] **Step 8: Add `GET /api/v1/rag/source-health` endpoint**

Modify `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`:

```python
@router.get(
    "/source-health",
    response_model=KnowledgeSourceHealthResponse,
    summary="Knowledge source health",
    description="Read-only pre-query source readiness for Knowledge QA.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(rbac_rate_limit("rag.search")),
        Depends(RequirePermission(MEDIA_READ)),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="rag.search", count_as="call")),
    ],
)
async def source_health_endpoint(
    current_user: User = Depends(get_request_user),
) -> KnowledgeSourceHealthResponse:
    configured_sources = _build_source_health_configured_sources(
        existing_paths=_resolve_existing_source_db_paths(current_user),
    )
    return KnowledgeSourceHealthResponse(
        sources=build_source_health_entries(
            configured_sources=configured_sources
        )
    )
```

Do not instantiate `MultiDatabaseRetriever` or source-specific databases in the source-health endpoint. Do not use request-scoped source DB dependencies such as `get_media_db_for_user`, `get_chacha_db_for_user`, or `get_prompts_db_for_user`. Derive source availability from existing files or metadata that can be checked without creating directories, schema, indexes, vector stores, records, or request-scoped database handles. Do not record RAG query usage and do not call search.

- [ ] **Step 9: Add compatibility regression for post-query `metadata.source_status`**

Add or extend a focused test around `_build_source_status` in `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py`:

```python
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import _build_source_status


def test_search_source_status_semantics_remain_backward_compatible() -> None:
    class _Retriever:
        retrievers = {DataSource.MEDIA_DB: object()}

    status = _build_source_status(
        ["media_db", "prompts"],
        retriever=_Retriever(),
        documents=[],
        filtered_counts={},
    )

    assert status["media_db"]["status"] == "empty"
    assert status["prompts"]["status"] == "unavailable"
```

Expected: PASS after implementation. This protects the old search-response contract.

- [ ] **Step 10: Run backend focused tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_source_contract.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py -k "source_status or source_health or source_contract"
```

Expected: PASS. If unrelated tests in `test_unified_pipeline.py` are selected accidentally, narrow the `-k` expression to the new regression.

- [ ] **Step 11: Commit backend contract**

```bash
git add \
  tldw_Server_API/app/core/RAG/rag_service/source_health.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_source_contract.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py
git commit -m "feat: add knowledge source health contract"
```

## Task 2: Frontend Source Health Client And Normalization

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/sourceHealth.ts`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/sourceHealth.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.rag-source-health.test.ts`

- [ ] **Step 1: Write failing client path test**

Create `apps/packages/ui/src/services/__tests__/tldw-api-client.rag-source-health.test.ts` using the existing API-client test style:

```ts
import { describe, expect, it, vi } from "vitest"
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"

describe("ragSourceHealth client", () => {
  it("requests the focused source health endpoint", async () => {
    const request = vi.fn().mockResolvedValue({ sources: [] })

    await chatRagMethods.ragSourceHealth.call({ request } as any)

    expect(request).toHaveBeenCalledWith({
      path: "/api/v1/rag/source-health",
      method: "GET",
    })
  })
})
```

Expected: FAIL because `ragSourceHealth` does not exist.

- [ ] **Step 2: Write failing source health normalization tests**

Create `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/sourceHealth.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import {
  buildSourceHealthSummary,
  normalizeKnowledgeSourceHealth,
} from "../sourceHealth"

describe("Knowledge QA source health normalization", () => {
  it("normalizes partial backend payloads without colliding with search source status", () => {
    const normalized = normalizeKnowledgeSourceHealth({
      sources: [
        {
          source_id: "media_db",
          label: "Documents & Media",
          available: true,
          searchable: true,
          index_status: "ready",
          embedding_status: "not_applicable",
          disabled_reason: null,
        },
      ],
    })

    expect(normalized.bySource.media_db?.indexStatus).toBe("ready")
    expect(normalized.bySource.media_db?.embeddingStatus).toBe("not_applicable")
    expect(normalized.bySource.media_db).not.toHaveProperty("status")
  })

  it("builds a compact summary", () => {
    const normalized = normalizeKnowledgeSourceHealth({
      sources: [
        { source_id: "media_db", label: "Documents & Media", available: true, searchable: true, index_status: "ready", embedding_status: "unknown", disabled_reason: null },
        { source_id: "prompts", label: "Prompts", available: false, searchable: false, index_status: "unavailable", embedding_status: "unavailable", disabled_reason: "no_retriever_configured" },
      ],
    })

    expect(buildSourceHealthSummary(normalized)).toBe("Sources ready: 1 of 2")
  })
})
```

Expected: FAIL because `sourceHealth.ts` does not exist.

- [ ] **Step 3: Add API client method**

Modify `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`:

```ts
async ragSourceHealth(this: TldwApiClientCore): Promise<any> {
  return await this.request<any>({
    path: "/api/v1/rag/source-health",
    method: "GET",
  })
},
```

If `apps/packages/ui/src/services/tldw/TldwApiClient.ts` still has direct method mirrors, add:

```ts
async ragSourceHealth(): Promise<any> {
  return await this.request<any>({
    path: "/api/v1/rag/source-health",
    method: "GET",
  })
}
```

- [ ] **Step 4: Add source-health frontend types**

Modify `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`:

```ts
export type KnowledgeSourceIndexStatus =
  | "ready"
  | "indexing"
  | "stale"
  | "empty"
  | "unavailable"
  | "error"
  | "unknown"

export type KnowledgeSourceEmbeddingStatus =
  | "ready"
  | "indexing"
  | "missing"
  | "unavailable"
  | "not_applicable"
  | "error"
  | "unknown"

export type KnowledgeSourceHealth = {
  sourceId: RagSettings["sources"][number]
  label: string
  available: boolean
  searchable: boolean
  itemCount: number | null
  indexedCount: number | null
  lastUpdated: string | null
  lastIndexed: string | null
  indexStatus: KnowledgeSourceIndexStatus
  embeddingStatus: KnowledgeSourceEmbeddingStatus
  disabledReason: string | null
  workspaceScoped: boolean
  hiddenByDefault: boolean
  privacyNote: string | null
}

export type KnowledgeSourceHealthState = {
  bySource: Partial<Record<RagSettings["sources"][number], KnowledgeSourceHealth>>
  sources: KnowledgeSourceHealth[]
  loading: boolean
  error: string | null
  loadedAt: string | null
}
```

Add state/action fields:

```ts
sourceHealth: KnowledgeSourceHealthState
refreshSourceHealth: () => Promise<void>
```

Keep `KnowledgeSourceStatus` unchanged.

- [ ] **Step 5: Implement normalization utilities**

Create `apps/packages/ui/src/components/Option/KnowledgeQA/sourceHealth.ts`:

```ts
import { isRagSource, getRagSourceLabel } from "@/services/rag/sourceMetadata"
import type { KnowledgeSourceHealth, KnowledgeSourceHealthState } from "./types"

export const EMPTY_SOURCE_HEALTH_STATE: KnowledgeSourceHealthState = {
  bySource: {},
  sources: [],
  loading: false,
  error: null,
  loadedAt: null,
}

export function normalizeKnowledgeSourceHealth(payload: unknown): KnowledgeSourceHealthState {
  const record = payload && typeof payload === "object" ? payload as Record<string, unknown> : {}
  const rawSources = Array.isArray(record.sources) ? record.sources : []
  const sources: KnowledgeSourceHealth[] = []

  for (const raw of rawSources) {
    const entry = raw && typeof raw === "object" ? raw as Record<string, unknown> : null
    const sourceId = typeof entry?.source_id === "string" && isRagSource(entry.source_id)
      ? entry.source_id
      : null
    if (!sourceId) continue
    sources.push({
      sourceId,
      label: typeof entry?.label === "string" ? entry.label : getRagSourceLabel(sourceId),
      available: entry?.available === true,
      searchable: entry?.searchable === true,
      itemCount: typeof entry?.item_count === "number" ? entry.item_count : null,
      indexedCount: typeof entry?.indexed_count === "number" ? entry.indexed_count : null,
      lastUpdated: typeof entry?.last_updated === "string" ? entry.last_updated : null,
      lastIndexed: typeof entry?.last_indexed === "string" ? entry.last_indexed : null,
      indexStatus: normalizeIndexStatus(entry?.index_status),
      embeddingStatus: normalizeEmbeddingStatus(entry?.embedding_status),
      disabledReason: typeof entry?.disabled_reason === "string" ? entry.disabled_reason : null,
      workspaceScoped: entry?.workspace_scoped === true,
      hiddenByDefault: entry?.hidden_by_default === true,
      privacyNote: typeof entry?.privacy_note === "string" ? entry.privacy_note : null,
    })
  }

  return {
    bySource: Object.fromEntries(sources.map((source) => [source.sourceId, source])),
    sources,
    loading: false,
    error: null,
    loadedAt: new Date().toISOString(),
  }
}
```

Add small `normalizeIndexStatus`, `normalizeEmbeddingStatus`, `getSourceHealthStatusLabel`, and `buildSourceHealthSummary` helpers. Keep all copy in this helper or shared constants to avoid divergent labels.

- [ ] **Step 6: Run frontend normalization/client tests**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/KnowledgeQA/__tests__/sourceHealth.test.ts \
  src/services/__tests__/tldw-api-client.rag-source-health.test.ts
```

Expected: PASS. If service tests live outside `apps/packages/ui`, run the equivalent from repo root with `bunx vitest run`.

- [ ] **Step 7: Commit frontend client/normalization**

```bash
git add \
  apps/packages/ui/src/services/tldw/domains/chat-rag.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/types.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/sourceHealth.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/sourceHealth.test.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.rag-source-health.test.ts
git commit -m "feat: add knowledge source health client"
```

## Task 3: Provider State And Source Picker Health UI

**Files:**
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/CompactToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.source-health.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx`

- [x] **Step 1: Write failing KnowledgeContextBar health tests**

Create `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.source-health.test.tsx`:

```tsx
import { render, screen, fireEvent } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { KnowledgeContextBar } from "../context/KnowledgeContextBar"
import type { KnowledgeSourceHealthState } from "../types"

const sourceHealth: KnowledgeSourceHealthState = {
  loading: false,
  error: null,
  loadedAt: "2026-05-16T00:00:00Z",
  sources: [
    { sourceId: "media_db", label: "Documents & Media", available: true, searchable: true, itemCount: null, indexedCount: null, lastUpdated: null, lastIndexed: null, indexStatus: "ready", embeddingStatus: "not_applicable", disabledReason: null, workspaceScoped: false, hiddenByDefault: false, privacyNote: null },
    { sourceId: "prompts", label: "Prompts", available: false, searchable: false, itemCount: null, indexedCount: null, lastUpdated: null, lastIndexed: null, indexStatus: "unavailable", embeddingStatus: "unavailable", disabledReason: "no_retriever_configured", workspaceScoped: false, hiddenByDefault: false, privacyNote: null },
  ],
  bySource: {},
}

it("shows aggregate source health and per-source status", () => {
  render(<KnowledgeContextBar {...baseProps} sourceHealth={sourceHealth} onRefreshSourceHealth={vi.fn()} />)

  expect(screen.getByText("Sources ready: 1 of 2")).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: /Sources:/i }))
  expect(screen.getByText("Ready")).toBeInTheDocument()
  expect(screen.getByText("Unavailable")).toBeInTheDocument()
})
```

Define `baseProps` using existing test props from `KnowledgeContextBar.test.tsx`.

Expected: FAIL because props/UI are missing.

- [x] **Step 2: Add provider state and refresh action**

Modify `KnowledgeQAProvider.tsx`:

- Add initial state:

```ts
sourceHealth: EMPTY_SOURCE_HEALTH_STATE,
```

- Add reducer actions or local state consistent with current provider patterns:

```ts
const refreshSourceHealth = useCallback(async () => {
  dispatch({ type: "SET_SOURCE_HEALTH_LOADING" })
  try {
    const payload = await tldwClient.ragSourceHealth()
    dispatch({ type: "SET_SOURCE_HEALTH", payload: normalizeKnowledgeSourceHealth(payload) })
  } catch {
    dispatch({ type: "SET_SOURCE_HEALTH_ERROR", payload: "Source health could not be loaded. You can still search selected sources." })
  }
}, [])
```

- Trigger once after server readiness/initialization succeeds. Do not block search while it loads or fails.

- [x] **Step 3: Thread source health through layout**

Modify `KnowledgeQALayout.tsx` to pass:

```tsx
sourceHealth={sourceHealth}
onRefreshSourceHealth={refreshSourceHealth}
```

to `KnowledgeContextBar`, `CompactToolbar`, `KnowledgeReadyState`, and recovery components where needed.

- [x] **Step 4: Render compact source health in context bars**

Modify `KnowledgeContextBar.tsx`:

- Add props:

```ts
sourceHealth?: KnowledgeSourceHealthState
onRefreshSourceHealth?: () => void
```

- Render:

```tsx
<span className="text-[11px] text-text-muted">
  {buildSourceHealthSummary(sourceHealth ?? EMPTY_SOURCE_HEALTH_STATE)}
</span>
```

- In source menu rows, show one chip per selected/canonical source:

```tsx
<span className={getSourceHealthChipClass(health?.indexStatus)}>
  {getSourceHealthStatusLabel(health)}
</span>
```

Use existing color tokens; no new palette.

Modify `CompactToolbar.tsx` with a one-line summary only. Avoid cluttering mobile toolbar.

- [x] **Step 5: Update ready/empty copy**

Modify `KnowledgeReadyState.tsx` to prefer source health copy:

- If all selected sources unavailable: `Selected sources are unavailable. Open source settings or choose a different scope.`
- If health failed: `Source health could not be loaded. You can still search selected sources.`
- If no searchable items: `No searchable items yet. Open Quick Ingest or the source owner page to add content.`

Do not add inline creation controls.

- [x] **Step 6: Run source picker UI tests**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.source-health.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
```

Expected: PASS.

- [x] **Step 7: Commit source picker UI**

```bash
git add \
  apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/context/CompactToolbar.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.source-health.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
git commit -m "feat: show knowledge source health before search"
```

## Task 4: Evidence Actions And Answer Trust Summary

**Files:**
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceCard.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx`
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/trustSummary.ts`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx`
- Regression Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceList.behavior.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/EvidenceRail.motion.test.ts`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts`

- [x] **Step 1: Write failing trust summary helper test**

Create `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { buildAnswerTrustSummary } from "../trustSummary"

describe("buildAnswerTrustSummary", () => {
  it("summarizes sources, citations, web fallback, and caveats", () => {
    expect(
      buildAnswerTrustSummary({
        selectedSources: ["media_db", "notes"],
        resultCount: 12,
        citationCount: 5,
        webFallbackEnabled: true,
        webFallbackTriggered: false,
        generationProvider: null,
        generationModel: null,
        sourceHealthCaveatCount: 2,
        trustLabel: "Partial",
      })
    ).toEqual([
      "Searched Documents & Media and Notes. 12 sources returned, 5 cited.",
      "Web fallback enabled, not used.",
      "AI model: Server default.",
      "2 selected sources need attention.",
      "Trust: Partial.",
    ])
  })
})
```

Expected: FAIL because `trustSummary.ts` does not exist.

- [x] **Step 2: Implement trust summary helper**

Create `apps/packages/ui/src/components/Option/KnowledgeQA/trustSummary.ts`:

```ts
import { getRagSourceLabel } from "@/services/rag/sourceMetadata"
import type { RagSource } from "@/services/rag/unified-rag"

export function formatSourceList(sources: RagSource[]): string {
  const labels = sources.map(getRagSourceLabel)
  if (labels.length <= 1) return labels[0] ?? "selected sources"
  if (labels.length === 2) return `${labels[0]} and ${labels[1]}`
  return `${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`
}
```

Add `buildAnswerTrustSummary` with deterministic copy and no viewport-specific logic.

- [x] **Step 3: Add AnswerPanel trust strip**

Modify `AnswerPanel.tsx`:

- Use selected `settings.sources`, `results.length`, `citations.length`, `searchDetails.webFallbackEnabled`, `searchDetails.webFallbackTriggered`, `settings.generation_provider`, `settings.generation_model`, and source-health caveats.
- Render a compact strip near existing answer controls, not inside the markdown answer body.
- Keep `SearchDetailsPanel` as the detailed diagnostics surface.

Suggested markup:

```tsx
<div aria-label="Answer trust summary" className="rounded-md border border-border bg-surface2/60 px-3 py-2 text-xs text-text-muted">
  {summaryLines.map((line) => <span key={line}>{line}</span>)}
</div>
```

- [x] **Step 4: Audit SourceCard evidence actions**

Modify `SourceCard.tsx` only after reading existing action structure:

- Keep existing `Copy citation`.
- Relabel evidence-context `Copy text` to `Copy excerpt` if tests support it.
- Do not add duplicate buttons for actions already present.
- Do not add persistent `Pin evidence`.
- Add `Save to note` only if the existing API contract is verified to preserve backlinks. Otherwise add a code comment in the test or plan notes that it is deferred.

Implementation note: `Save to note` is deferred in this slice because a backlink-preserving note handoff contract was not verified.

- [x] **Step 5: Add EvidenceRail action affordance without a new panel**

Modify `EvidenceRail.tsx`:

- Add a concise hint above source list when sources exist:

```tsx
<p className="mb-2 text-xs text-text-muted">
  Use each source card to copy citations, copy excerpts, or open supported sources.
</p>
```

- Keep tabs and lazy details panel unchanged.

- [x] **Step 6: Update tests for evidence/trust UI**

Extend:

- `AnswerPanel.states.test.tsx`: assert `Answer trust summary` renders source/citation/web fallback/model copy.
- `SourceCard.behavior.test.tsx`: assert accessible names include `Copy citation` and `Copy excerpt`, and no duplicate copy controls are added.
- Add a small EvidenceRail test if current motion test is not enough; otherwise keep EvidenceRail change covered by layout test.

- [x] **Step 7: Run evidence/trust tests**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts \
  src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/SourceList.behavior.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/EvidenceRail.motion.test.ts
```

Expected: PASS.

- [x] **Step 8: Commit evidence/trust UI**

```bash
git add \
  apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/SourceCard.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/trustSummary.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceList.behavior.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/EvidenceRail.motion.test.ts
git commit -m "feat: clarify knowledge evidence trust controls"
```

## Task 5: Recovery Copy And Extension Parity

**Files:**
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/LowQualityRecoveryBanner.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx`
- Test: `apps/tldw-frontend/__tests__/extension/knowledge-route-parity.test.ts`

- [ ] **Step 1: Write failing recovery copy tests**

Extend `NoResultsRecovery.source-status.test.tsx`:

```tsx
it("distinguishes pre-query health from post-query source diagnostics", () => {
  render(
    <NoResultsRecovery
      {...defaultProps}
      sourceHealth={unavailableHealthState}
      sourceStatus={{ media_db: { status: "empty", count: 0, reason: "no_matching_entries" } }}
    />
  )

  expect(screen.getByText(/source readiness/i)).toBeInTheDocument()
  expect(screen.getByText(/search diagnostics/i)).toBeInTheDocument()
})
```

Expected: FAIL until props/UI are added.

- [ ] **Step 2: Update NoResultsRecovery**

Modify `NoResultsRecovery.tsx`:

- Keep current `sourceStatus` section but label it `Search diagnostics`.
- Add optional `sourceHealth` prop and render selected source readiness as `Source readiness`.
- Use handoff copy:
  - `Open Quick Ingest`
  - `Open source page`
  - no ambiguous `Add sources`
- Keep `Show nearest matches` only if current metadata/UI already supports it.

- [ ] **Step 3: Update LowQualityRecoveryBanner**

Modify `LowQualityRecoveryBanner.tsx`:

- Accept optional source-health caveat count or selected caveats.
- Copy:

```text
This answer has limited evidence. Try expanding sources, checking source status, or enabling web fallback.
```

- Do not imply web fallback will be enabled automatically.

- [ ] **Step 4: Run recovery tests**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run extension parity/static tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run \
  __tests__/extension/knowledge-route-parity.test.ts
```

If KnowledgeQA imports or lazy boundaries changed materially, also run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/extension/entry-shell-performance.test.ts
```

Expected: PASS or document unrelated baseline failure with exact output.

- [ ] **Step 6: Commit recovery/parity work**

```bash
git add \
  apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/panels/LowQualityRecoveryBanner.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx
git commit -m "feat: improve knowledge recovery diagnostics"
```

## Task 6: Final Verification And PR Packaging

**Files:**
- Modify: implementation Backlog task created in Task 1.
- Optional Modify: `Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md` only if implementation required design corrections.

- [ ] **Step 1: Run backend focused verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/RAG_NEW/unit/test_source_health.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_source_contract.py
```

Expected: PASS.

- [ ] **Step 2: Run frontend focused verification**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/KnowledgeQA/__tests__/sourceHealth.test.ts \
  src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.source-health.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx \
  src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run extension parity verification**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/extension/knowledge-route-parity.test.ts
```

Expected: PASS.

- [ ] **Step 4: Run browser smoke when local app is viable**

If a frontend dev server and backend are already running:

```bash
cd apps/tldw-frontend
TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://localhost:3000 \
  bunx playwright test e2e/workflows/knowledge-qa.spec.ts --grep "Knowledge QA" --reporter=line
```

If the local server is not viable, record the blocker in the Backlog task instead of fabricating browser evidence.

- [ ] **Step 5: Run diff and security checks**

Run:

```bash
git diff --check
```

Expected: no output.

Run Bandit for touched backend files:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/RAG/rag_service/source_health.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py \
  -f json -o /tmp/bandit_knowledge_source_health.json
```

Expected: exit 0 with no new findings in touched code.

- [ ] **Step 6: Update Backlog task**

Use the implementation task id from Task 1:

```bash
backlog task edit TASK-ID \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "Verification: backend pytest ..., frontend Vitest ..., extension parity ..., git diff --check ..., Bandit ..." \
  --final-summary "Implemented /knowledge source health and evidence controls. Source health is pre-query and read-only, post-query metadata.source_status remains backward compatible, evidence actions reuse existing handoffs, and /knowledge remains QA-only." \
  --status Done
```

- [ ] **Step 7: Create or update PR**

Push branch:

```bash
git push -u origin codex/knowledge-source-health-evidence-controls
```

Open PR against `dev`:

```bash
gh pr create \
  --base dev \
  --head codex/knowledge-source-health-evidence-controls \
  --title "Add Knowledge QA source health and evidence controls" \
  --body "$(cat <<'EOF'
## Summary
- Add read-only pre-query source health for Knowledge QA sources.
- Surface source readiness in /knowledge source controls.
- Add compact answer trust summary and clearer evidence/recovery controls.

## Scope
- /knowledge remains QA-only.
- No server-backed saved evidence or shared evidence sets.
- Existing post-query metadata.source_status remains backward compatible.

## Verification
- [ ] Backend focused pytest
- [ ] KnowledgeQA focused Vitest
- [ ] Extension parity test
- [ ] git diff --check
- [ ] Bandit on touched backend files

Change summary:
<human-written summary required before merge>
EOF
)"
```

Leave the human-owned `Change summary` placeholder for the requester.
