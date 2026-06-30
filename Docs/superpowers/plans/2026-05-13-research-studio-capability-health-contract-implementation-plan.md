# Research Studio Capability Health Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-owned Research Studio capability health contract, consume it in the WebUI at action boundaries, and verify authenticated local real-LLM summary generation using existing saved credentials.

**Architecture:** The backend exposes one authenticated, permission-gated Research Studio capability endpoint derived from lightweight local health collectors. The shared WebUI fetches and caches that contract with a short TTL, applies it only to the relevant chat/generation/export action boundaries, and keeps existing route/degraded-entry behavior intact. Real LLM generation is verified locally and manually, not in CI.

**Tech Stack:** FastAPI, Pydantic, existing AuthNZ dependencies, existing router groups, Next.js/WebUI shared React package, Vitest, pytest, Bandit, Playwright/CDP for local verification.

---

## File Map

Backend contract and derivation:

- Create: `tldw_Server_API/app/api/v1/schemas/research_studio_capabilities.py`
  - Pydantic response models and stable enum literals.
- Create: `tldw_Server_API/app/core/Research_Studio/__init__.py`
  - Package marker for Research Studio backend helpers.
- Create: `tldw_Server_API/app/core/Research_Studio/capabilities.py`
  - Lightweight capability derivation service. This service must not HTTP-call sibling endpoints.
- Create: `tldw_Server_API/app/api/v1/endpoints/research_studio.py`
  - Authenticated, permission-gated, rate-limited API route for `GET /api/v1/research-studio/capabilities`.
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
  - Register the new endpoint under `/api/v1`.
- Test: `tldw_Server_API/tests/Research_Studio/test_capability_derivation.py`
  - Unit tests for capability derivation and sanitization.
- Test: `tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py`
  - Endpoint/auth/schema tests.

Frontend client and gating:

- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Add `/api/v1/research-studio/capabilities` to `ClientPath`.
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Add typed `getResearchStudioCapabilities()` method.
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-capabilities.ts`
  - Shared types, default unknown/warn state, TTL logic helpers, artifact-to-capability mapping.
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts`
  - Pure helper tests.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
  - Fetch, cache, refresh, and pass capability state to panes.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx`
  - Add chat capability warning/block behavior.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
  - Add artifact/slides/audio action-boundary warning/block behavior after source selection passes.
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
  - Route-level capability fetch/default behavior.
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx`
  - Generation gating by capability mode.
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/__tests__/ChatPane.capabilities.test.tsx`
  - Chat send warning/block behavior.

Docs and verification:

- Modify: `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md`
  - Replace follow-up-only language with implemented endpoint and verification evidence.
- Modify: `Docs/superpowers/specs/2026-05-13-research-studio-capability-health-contract-design.md`
  - Update if implementation choices differ from the approved design.
- Modify: `backlog/tasks/task-304.12 - Write-Research-Studio-capability-health-implementation-plan.md`
  - Close plan task.
- Create during implementation: `backlog/tasks/task-304.13 - Implement-Research-Studio-capability-health-contract.md`
  - Track code implementation before editing source files.
- Optional manual spec: `apps/tldw-frontend/e2e/workflows/research-studio-live-generation.manual.spec.ts`
  - A skipped-by-default/manual Playwright spec can encode the local real-generation flow if it helps repeatability.

---

## Task 0: Implementation Tracking

**Files:**
- Create: `backlog/tasks/task-304.13 - Implement-Research-Studio-capability-health-contract.md`
- Modify: `backlog/tasks/task-304.12 - Write-Research-Studio-capability-health-implementation-plan.md`

- [ ] **Step 1: Create the implementation Backlog task before code edits**

Run from the PR worktree:

```bash
backlog task create "Implement Research Studio capability health contract" \
  --plain \
  --status "In Progress" \
  --priority medium \
  --labels implementation,research-studio,webui,backend,verification \
  --parent TASK-304 \
  --depends-on TASK-304.12 \
  --doc Docs/superpowers/specs/2026-05-13-research-studio-capability-health-contract-design.md \
  --doc Docs/superpowers/plans/2026-05-13-research-studio-capability-health-contract-implementation-plan.md \
  --ac "Backend exposes authenticated, permission-gated, rate-limited Research Studio capability endpoint with stable status/mode semantics." \
  --ac "Frontend consumes the capability endpoint and gates chat, text artifacts, slides, audio summary, export/download, and sync/share at action boundaries." \
  --ac "Tests cover backend derivation, endpoint auth/schema, frontend helper behavior, and UI allow/warn/block states." \
  --ac "Docs and PR notes record authenticated CDP checks and local/manual real summary generation using existing saved LLM credentials." \
  --ac "Bandit and focused frontend/backend tests pass or skips are explicitly documented."
```

Expected: a new `TASK-304.13` file in this worktree.

- [ ] **Step 2: Confirm the task was created in the PR worktree**

Run:

```bash
git status --short -- "backlog/tasks"
```

Expected: the new task file appears under the current worktree, not the main checkout.

- [ ] **Step 3: Commit boundary**

Do not commit yet unless this task is performed separately from Task 1. If committed separately:

```bash
git add "backlog/tasks/task-304.13 - Implement-Research-Studio-capability-health-contract.md"
git commit -m "Track Research Studio capability health implementation"
```

---

## Task 1: Backend Capability Models And Derivation Service

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/research_studio_capabilities.py`
- Create: `tldw_Server_API/app/core/Research_Studio/__init__.py`
- Create: `tldw_Server_API/app/core/Research_Studio/capabilities.py`
- Test: `tldw_Server_API/tests/Research_Studio/test_capability_derivation.py`

- [ ] **Step 1: Write failing backend model/service tests**

Create `tldw_Server_API/tests/Research_Studio/test_capability_derivation.py` with tests for:

```python
def test_ready_dependencies_allow_text_generation():
    response = build_research_studio_capabilities(
        aggregate_health={"status": "ok", "checks": {"chacha_notes": {"status": "healthy"}}},
        rag_health={"status": "healthy"},
        llm_health={"status": "healthy", "components": {"providers": {"initialized": True, "count": 1}}},
        slides_health={"status": "ok"},
        tts_health={"status": "healthy", "providers": {"available": 1}},
    )
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_unknown_source_health_warns_instead_of_overclaiming():
    response = build_research_studio_capabilities(
        aggregate_health={"status": "ok", "checks": {"database": {"status": "healthy"}}},
        rag_health={"status": "healthy"},
        llm_health={"status": "healthy", "components": {"providers": {"initialized": True, "count": 1}}},
        slides_health={"status": "ok"},
        tts_health={"status": "healthy", "providers": {"available": 1}},
    )
    assert response.capabilities["source_browse"].status == "unknown"
    assert response.capabilities["source_browse"].mode == "warn"


def test_llm_unavailable_blocks_chat_and_text_generation_only():
    response = build_research_studio_capabilities(
        aggregate_health={"status": "ok", "checks": {"chacha_notes": {"status": "healthy"}}},
        rag_health={"status": "healthy"},
        llm_health={"status": "unhealthy", "components": {"providers": {"initialized": False}}},
        slides_health={"status": "ok"},
        tts_health={"status": "healthy", "providers": {"available": 1}},
    )
    assert response.capabilities["chat"].mode == "block"
    assert response.capabilities["artifact_text_generation"].mode == "block"
    assert response.capabilities["export_download"].mode == "allow"


def test_slides_unavailable_blocks_only_slides():
    response = build_research_studio_capabilities(
        aggregate_health={"status": "ok", "checks": {"chacha_notes": {"status": "healthy"}}},
        rag_health={"status": "healthy"},
        llm_health={"status": "healthy", "components": {"providers": {"initialized": True, "count": 1}}},
        slides_health={"status": "unhealthy"},
        tts_health={"status": "healthy", "providers": {"available": 1}},
    )
    assert response.capabilities["slides_generation"].mode == "block"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_payload_does_not_leak_raw_exceptions_or_paths():
    response = build_research_studio_capabilities(
        aggregate_health={"status": "unhealthy", "error": "Traceback /Users/private/key.py"},
        rag_health={"status": "unhealthy", "error": "stack trace"},
        llm_health={"status": "unhealthy", "components": {"providers": {"report": {"openai": {"api_key": "secret"}}}}},
        slides_health={"status": "unhealthy", "detail": "/tmp/secret"},
        tts_health={"status": "error", "message": "provider failed"},
    )
    dumped = response.model_dump_json()
    assert "Traceback" not in dumped
    assert "/Users/" not in dumped
    assert "secret" not in dumped
    assert "api_key" not in dumped
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Studio/test_capability_derivation.py -v
```

Expected: fail because files/functions do not exist.

- [ ] **Step 3: Add Pydantic schemas**

Create `tldw_Server_API/app/api/v1/schemas/research_studio_capabilities.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ResearchStudioCapabilityId = Literal[
    "source_browse",
    "chat",
    "artifact_text_generation",
    "slides_generation",
    "audio_summary",
    "export_download",
    "sync_share",
]
ResearchStudioCapabilityStatus = Literal["ready", "degraded", "unavailable", "unknown"]
ResearchStudioCapabilityMode = Literal["allow", "warn", "block"]
ResearchStudioOverallStatus = Literal["ready", "degraded", "unavailable", "unknown"]


class ResearchStudioCapability(BaseModel):
    status: ResearchStudioCapabilityStatus
    mode: ResearchStudioCapabilityMode
    dependencies: list[str] = Field(default_factory=list)
    reason_code: str | None = None


class ResearchStudioCapabilitiesResponse(BaseModel):
    status: ResearchStudioOverallStatus
    ttl_seconds: int = 30
    capabilities: dict[str, ResearchStudioCapability]
    timestamp: datetime
```

Use `dict[str, ResearchStudioCapability]` in the wire model for OpenAPI stability, and validate the expected `ResearchStudioCapabilityId` keys in the builder/tests.

- [ ] **Step 4: Add derivation service**

Create `tldw_Server_API/app/core/Research_Studio/__init__.py` as a package marker.

Create `tldw_Server_API/app/core/Research_Studio/capabilities.py` with a pure builder that accepts already-collected health dicts:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from tldw_Server_API.app.api.v1.schemas.research_studio_capabilities import (
    ResearchStudioCapabilitiesResponse,
    ResearchStudioCapability,
)


def _status(value: Mapping[str, Any] | None) -> str:
    raw = value.get("status") if isinstance(value, Mapping) else None
    return raw.lower().strip() if isinstance(raw, str) else "unknown"


def _cap(status: str, mode: str, dependencies: list[str], reason_code: str | None = None) -> ResearchStudioCapability:
    return ResearchStudioCapability(
        status=status,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        dependencies=dependencies,
        reason_code=reason_code,
    )


def _source_browse(aggregate_health: Mapping[str, Any] | None) -> ResearchStudioCapability:
    checks = aggregate_health.get("checks") if isinstance(aggregate_health, Mapping) else {}
    chacha = checks.get("chacha_notes") if isinstance(checks, Mapping) else None
    chacha_status = _status(chacha if isinstance(chacha, Mapping) else None)
    if chacha_status in {"healthy", "ok", "ready"}:
        return _cap("ready", "allow", ["chacha_notes"])
    if chacha_status in {"unhealthy", "unavailable"}:
        return _cap("unavailable", "block", ["chacha_notes"], "source_store_unavailable")
    return _cap("unknown", "warn", ["source_store"], "source_health_unknown")
```

Then implement similar small helpers for RAG, LLM, slides, TTS, export, and sync. Keep these helpers pure and easy to unit test.

- [ ] **Step 5: Run derivation tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Studio/test_capability_derivation.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/research_studio_capabilities.py \
  tldw_Server_API/app/core/Research_Studio/__init__.py \
  tldw_Server_API/app/core/Research_Studio/capabilities.py \
  tldw_Server_API/tests/Research_Studio/test_capability_derivation.py
git commit -m "Add Research Studio capability derivation"
```

---

## Task 2: Authenticated, Permission-Gated Capability Endpoint And Router Registration

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/research_studio.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py`

- [ ] **Step 1: Write failing endpoint tests**

Create `tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py` with tests that:

- call `GET /api/v1/research-studio/capabilities`;
- assert unauthenticated requests are rejected in auth-enabled modes;
- assert authenticated callers without `media.read` are rejected in multi-user/RBAC fixtures when that fixture is available;
- assert authenticated requests return `ttl_seconds`, `status`, and all capability keys;
- assert raw exception/path/secret fields do not appear in response JSON;
- assert route is present in OpenAPI.

Use existing AuthNZ/API test fixtures in nearby endpoint tests instead of creating a new auth harness.

- [ ] **Step 2: Run endpoint tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py -v
```

Expected: fail because the endpoint is not registered.

- [ ] **Step 3: Implement local health collectors**

In `tldw_Server_API/app/core/Research_Studio/capabilities.py`, add async collection helpers that avoid HTTP calls:

```python
async def collect_research_studio_capabilities(*, user_id: int | str | None = None) -> ResearchStudioCapabilitiesResponse:
    aggregate = await _collect_aggregate_health()
    rag = await _collect_rag_health(user_id=user_id)
    llm = await _collect_llm_health()
    slides = await _collect_slides_health(user_id=user_id)
    tts = await _collect_tts_health()
    return build_research_studio_capabilities(
        aggregate_health=aggregate,
        rag_health=rag,
        llm_health=llm,
        slides_health=slides,
        tts_health=tts,
    )
```

Collector rules:

- Wrap each dependency in `try/except Exception` and return a sanitized `{ "status": "unknown", "reason_code": "<dependency>_health_unknown" }` shape.
- Prefer importing existing pure/internal helpers when available.
- If a dependency only has an endpoint function, call the function directly only when its dependencies can be supplied cheaply and safely.
- Do not warm models, make provider generation calls, or expose raw errors.

- [ ] **Step 4: Implement endpoint**

Create `tldw_Server_API/app/api/v1/endpoints/research_studio.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User
from tldw_Server_API.app.api.v1.schemas.research_studio_capabilities import ResearchStudioCapabilitiesResponse
from tldw_Server_API.app.core.Research_Studio.capabilities import collect_research_studio_capabilities
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ

router = APIRouter(prefix="/research-studio", tags=["research-studio"])


@router.get(
    "/capabilities",
    response_model=ResearchStudioCapabilitiesResponse,
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("research_studio.capabilities")),
    ],
)
async def research_studio_capabilities(
    current_user: User = Depends(get_request_user),
) -> ResearchStudioCapabilitiesResponse:
    return await collect_research_studio_capabilities(user_id=current_user.id)
```

- [ ] **Step 5: Register router**

Modify `tldw_Server_API/app/api/v1/router_groups/content.py`. Add an `ImportedRouterSpec` near the existing RAG/research specs:

```python
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.research_studio",
    log_name="research_studio",
    prefix=f"{API_V1_PREFIX}",
    tags=("research-studio",),
    route_key="research-studio",
),
```

- [ ] **Step 6: Run endpoint tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py -v
```

Expected: pass.

- [ ] **Step 7: Run OpenAPI-focused check**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_openapi_phase4_contract.py -q
```

Expected: pass, or no regression attributable to the new route.

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/research_studio.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/core/Research_Studio/capabilities.py \
  tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py
git commit -m "Expose Research Studio capability health"
```

---

## Task 3: Shared Frontend Capability Client

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-capabilities.ts`
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts`

- [ ] **Step 1: Write failing helper tests**

Create `research-studio-capabilities.test.ts` with tests for:

- unknown fallback maps all capabilities to `mode: "warn"`;
- stale payload detection honors `ttl_seconds`;
- `summary`, `report`, `flashcards`, `quiz`, `timeline`, `compare_sources`, `mindmap`, and `data_table` map to `artifact_text_generation`;
- `slides` maps to `slides_generation`;
- `audio_overview` maps to `audio_summary`;
- `block` affects only the selected capability.

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts
```

Expected: fail because helper does not exist.

- [ ] **Step 3: Add path to API guard**

Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`:

```ts
  | "/api/v1/research-studio/capabilities"
```

- [ ] **Step 4: Add API client method**

In `apps/packages/ui/src/services/tldw/TldwApiClient.ts`, add types or import them from the helper, then add:

```ts
async getResearchStudioCapabilities(): Promise<ResearchStudioCapabilitiesResponse> {
  return await this.request<ResearchStudioCapabilitiesResponse>({
    path: "/api/v1/research-studio/capabilities",
    method: "GET"
  })
}
```

- [ ] **Step 5: Add helper**

Create `apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-capabilities.ts`:

```ts
export type ResearchStudioCapabilityId =
  | "source_browse"
  | "chat"
  | "artifact_text_generation"
  | "slides_generation"
  | "audio_summary"
  | "export_download"
  | "sync_share"

export type ResearchStudioCapabilityMode = "allow" | "warn" | "block"
export type ResearchStudioCapabilityStatus = "ready" | "degraded" | "unavailable" | "unknown"

export type ResearchStudioCapability = {
  status: ResearchStudioCapabilityStatus
  mode: ResearchStudioCapabilityMode
  dependencies: string[]
  reason_code?: string | null
}

export type ResearchStudioCapabilitiesResponse = {
  status: ResearchStudioCapabilityStatus
  ttl_seconds?: number
  capabilities: Record<ResearchStudioCapabilityId, ResearchStudioCapability>
  timestamp?: string
}
```

Also implement:

- `buildUnknownResearchStudioCapabilities()`;
- `getArtifactCapabilityId(type: ArtifactType)`;
- `isResearchStudioCapabilitiesStale(payload, fetchedAtMs, nowMs)`;
- `getCapability(payload, id)`;
- `getCapabilityCopy(capability, actionLabel)`.

- [ ] **Step 6: Run helper tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts
```

Expected: pass.

- [ ] **Step 7: Run OpenAPI guard verification if available**

Run:

```bash
cd apps/packages/ui
bun run verify:openapi
```

Expected: pass. If it fails because the backend OpenAPI cannot be imported in the local environment, record the exact failure in the Backlog task and run the backend OpenAPI pytest from Task 2.

- [ ] **Step 8: Commit**

```bash
git add \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-capabilities.ts \
  apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts
git commit -m "Add Research Studio capability client"
```

---

## Task 4: Frontend Route Fetch And Action Boundary Gating

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/__tests__/ChatPane.capabilities.test.tsx`

- [x] **Step 1: Write failing WorkspacePlayground fetch tests**

Extend `WorkspacePlayground.stage2.responsive.test.tsx` or add a focused test that mocks `tldwClient.getResearchStudioCapabilities` and asserts:

- endpoint is fetched after route render;
- malformed/rejected fetch falls back to unknown/warn;
- cached stale payload refresh can be triggered before generation.

- [x] **Step 2: Write failing ChatPane capability tests**

Add `ChatPane.capabilities.test.tsx` with tests:

- `mode: "allow"` keeps composer and Send behavior unchanged;
- `mode: "warn"` shows degraded copy and keeps Send enabled;
- `mode: "block"` disables Send and shows reason copy;
- no selected sources still shows existing no-source/source context copy first.

- [x] **Step 3: Write failing StudioPane gating tests**

Extend `StudioPane.stage3.test.tsx`:

- no selected sources shows source-readiness guidance before capability warnings;
- `artifact_text_generation: block` disables Summary but does not disable browser-local existing artifact viewing;
- `slides_generation: block` disables only Slides;
- `audio_summary: block` disables only Audio Summary;
- `warn` shows inline copy and still allows the clicked generation handler.

- [x] **Step 4: Run tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx \
  src/components/Option/WorkspacePlayground/ChatPane/__tests__/ChatPane.capabilities.test.tsx
```

Expected: fail because props/helper wiring does not exist.

- [x] **Step 5: Fetch capabilities in WorkspacePlayground**

In `WorkspacePlayground/index.tsx`:

- add state `{ payload, fetchedAtMs, loading, error }`;
- call `tldwClient.getResearchStudioCapabilities()` after hydration;
- fallback to `buildUnknownResearchStudioCapabilities()` on error;
- refresh on route focus and expose a `refreshResearchStudioCapabilities` callback;
- before expensive actions, refresh if stale.

Do not block route rendering while this fetch is pending.

- [x] **Step 6: Pass capability state to ChatPane and StudioPane**

Add props:

```ts
researchStudioCapabilities?: ResearchStudioCapabilitiesResponse
onRefreshResearchStudioCapabilities?: () => Promise<ResearchStudioCapabilitiesResponse>
```

Pass only what each pane needs if smaller props are clearer.

- [x] **Step 7: Implement ChatPane boundary behavior**

In `ChatPane/index.tsx`:

- derive `chatCapability = getCapability(payload, "chat")`;
- keep existing `isChatUnavailable` behavior;
- set blocked when `chatCapability.mode === "block"`;
- show inline degraded copy when `mode === "warn"`;
- avoid sending if blocked;
- leave source-selection/no-source messaging precedence intact.

- [x] **Step 8: Implement StudioPane boundary behavior**

In `StudioPane/index.tsx`:

- use `getArtifactCapabilityId(type)` inside `renderOutputButton`;
- if `!hasSelectedSources`, return existing `studio-source-readiness` block before capability UI;
- if capability mode is `block`, disable only that output type and update tooltip text;
- if capability mode is `warn`, show inline warning copy near the output group and allow generation;
- ensure regeneration menus use the same capability mapping.

- [x] **Step 9: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx \
  src/components/Option/WorkspacePlayground/ChatPane/__tests__/ChatPane.capabilities.test.tsx
```

Expected: pass.

- [x] **Step 10: Commit**

```bash
git add \
  apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx \
  apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx \
  apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx \
  apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx \
  apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/__tests__/ChatPane.capabilities.test.tsx
git commit -m "Gate Research Studio actions by capability"
```

---

## Task 5: Docs, Runbook, And Manual Verification Harness

**Files:**
- Modify: `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md`
- Modify: `Docs/superpowers/specs/2026-05-13-research-studio-capability-health-contract-design.md`
- Optional create: `apps/tldw-frontend/e2e/workflows/research-studio-live-generation.manual.spec.ts`
- Modify: `backlog/tasks/task-304.13 - Implement-Research-Studio-capability-health-contract.md`

- [x] **Step 1: Update runbook from planned to implemented contract**

In `Research_Studio_Trust_Status_Telemetry_Runbook.md`:

- replace follow-up language with the implemented endpoint path;
- document `status`, `mode`, `ttl_seconds`, dependencies, and reason-code rules;
- document that aggregate `/api/v1/health` still controls broad app entry only;
- document manual real-generation evidence requirements.

- [x] **Step 2: Add a manual Playwright spec or scripted checklist**

If adding a manual spec, create `apps/tldw-frontend/e2e/workflows/research-studio-live-generation.manual.spec.ts` and guard it:

```ts
test.skip(
  process.env.TLDW_RESEARCH_STUDIO_LIVE_GENERATION !== "1",
  "Manual local-only real LLM verification; not a CI gate."
)
```

The spec should:

- require `TLDW_WEB_URL`, `TLDW_SERVER_URL`, and a valid local auth context;
- never print full API keys;
- open `/research-studio`;
- verify `/api/v1/research-studio/capabilities`;
- select/create a small source;
- generate a Summary artifact;
- assert non-empty generated text and record length, not full content.

If a manual spec is too brittle for the current UI setup, write an explicit checklist in the runbook and execute it with CDP commands during verification.

- [x] **Step 3: Add credential discovery checklist without exposing secrets**

Document commands such as:

```bash
source .venv/bin/activate
python - <<'PY'
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
s = get_settings()
key = getattr(s, "SINGLE_USER_API_KEY", "") or ""
print({"auth_mode": getattr(s, "AUTH_MODE", None), "single_user_key_present": bool(key), "single_user_key_len": len(key)})
PY
```

Do not print the key itself. Use existing saved provider credentials from project config/environment; do not add a new provider secret.

- [x] **Step 4: Commit docs/harness**

```bash
git add \
  Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md \
  Docs/superpowers/specs/2026-05-13-research-studio-capability-health-contract-design.md \
  apps/tldw-frontend/e2e/workflows/research-studio-live-generation.manual.spec.ts
git commit -m "Document Research Studio capability verification"
```

Omit the manual spec path from `git add` if no spec was created.

---

## Task 6: Local Authenticated CDP And Real LLM Generation Verification

**Files:**
- Modify: `backlog/tasks/task-304.13 - Implement-Research-Studio-capability-health-contract.md`
- Modify: PR body for #1616

- [x] **Step 1: Verify local backend credential availability without exposing secrets**

Run:

```bash
source .venv/bin/activate
python - <<'PY'
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
s = get_settings()
key = getattr(s, "SINGLE_USER_API_KEY", "") or ""
print({"auth_mode": getattr(s, "AUTH_MODE", None), "single_user_key_present": bool(key), "single_user_key_len": len(key)})
PY
```

Expected: key present for single-user mode, or valid JWT flow identified for multi-user mode.

- [x] **Step 2: Verify saved LLM provider availability without exposing secrets**

Run a safe config/provider listing command. Prefer existing provider endpoints or config helpers that redact secrets. Example:

```bash
curl -sf -H "X-API-Key: ${TLDW_E2E_API_KEY}" \
  "http://127.0.0.1:8000/api/v1/llm/providers" \
  | jq '{provider_count: (.providers // [] | length), providers: (.providers // [] | map(.name // .provider // .id))}'
```

Do not paste API keys or raw provider secret fields into task notes.

- [x] **Step 3: Start backend and WebUI**

Use existing project startup commands and existing saved credentials. Example:

```bash
source .venv/bin/activate
python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000
```

In another session:

```bash
cd apps/tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev --hostname localhost --port 3002
```

- [x] **Step 4: Run authenticated CDP route and endpoint smoke**

Use Playwright/CDP, not Computer Use. Verify:

- `/research-studio` renders;
- `/workspace-playground` aliases to `/research-studio`;
- `/workspace-studio?tab=studio` aliases to `/research-studio?tab=studio`;
- mobile `/research-studio?tab=studio` opens Studio;
- page context can call `/api/v1/health`;
- page context can call `/api/v1/research-studio/capabilities`.

Record screenshot paths under `/private/tmp/research-studio-capabilities-*.png` if screenshots are captured.

- [x] **Step 5: Generate a real Summary artifact**

Using existing saved LLM credentials:

- select or create a small deterministic local source;
- select the source in Research Studio;
- generate `summary`;
- wait for completion;
- assert generated output is non-empty.

Record only:

- provider name;
- model name if safe/visible;
- source type/title if non-sensitive;
- artifact type;
- output character count or short non-sensitive excerpt length;
- screenshot path if captured;
- caveats.

If saved local credentials are missing or invalid, stop and record this as a PR verification blocker rather than falling back to a mock.

Evidence recorded during implementation:
- Local saved single-user API key present; provider-list check found configured
  `openai` and `gpt-4o-mini`.
- Authenticated `GET /api/v1/health` returned `200` with `status: ok`.
- Authenticated `GET /api/v1/research-studio/capabilities` returned `200` in
  25 ms with all seven capability IDs after switching TTS capability collection
  to config-only readiness.
- Opt-in Playwright/CDP run:
  `TLDW_RESEARCH_STUDIO_LIVE_GENERATION=1 ... bunx playwright test e2e/workflows/research-studio-live-generation.manual.spec.ts --project=chromium --reporter=line --workers=1`
  passed 2 tests. Real Summary generation used provider `openai`, model
  `gpt-4o-mini`, source type `document`, artifact type `summary`, output
  character count `357`, screenshot
  `/private/tmp/research-studio-live-summary-research-studio-summary-1778643186763-r8aqmx.png`.

- [x] **Step 6: Update Backlog and PR body**

Update `TASK-304.13` with:

- automated test commands and results;
- Bandit command and result;
- authenticated CDP result;
- real LLM generation result;
- local caveats.

Update PR #1616 body with the same high-level verification evidence.

---

## Task 7: Final Verification, Bandit, Squash, And PR Update

**Files:**
- Modify: `backlog/tasks/task-304.13 - Implement-Research-Studio-capability-health-contract.md`
- Modify: PR #1616 body

- [x] **Step 1: Run backend focused tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Research_Studio/test_capability_derivation.py \
  tldw_Server_API/tests/Research_Studio/test_capability_endpoint.py \
  -v
```

Expected: pass.

- [x] **Step 2: Run frontend focused tests**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/research-studio-capabilities.test.ts \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx \
  src/components/Option/WorkspacePlayground/ChatPane/__tests__/ChatPane.capabilities.test.tsx
```

Expected: pass.

- [x] **Step 3: Run existing Research Studio route tests**

```bash
cd apps/tldw-frontend
bun run test:run __tests__/extension/route-registry.workspace-playground.test.ts components/networking/__tests__/ServerReadinessGate.test.tsx
```

Expected: pass.

- [x] **Step 4: Run Bandit on touched backend code**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/research_studio.py \
  tldw_Server_API/app/api/v1/schemas/research_studio_capabilities.py \
  tldw_Server_API/app/core/Research_Studio \
  -f json -o /tmp/bandit_research_studio_capabilities.json
```

Expected: no new high/medium findings in touched code. If Bandit flags a finding, fix or document why it is a false positive with evidence.

- [x] **Step 5: Run diff hygiene**

```bash
git diff --check
```

Expected: no output.

- [x] **Step 6: Close Backlog task**

Use `backlog task edit TASK-304.13` to check AC/DoD, append verification notes, record local manual generation evidence, and set status Done.

- [ ] **Step 7: Squash branch to one PR commit**

The PR previously required a single squashed commit. Before pushing:

```bash
git fetch origin dev
git log --oneline origin/dev..HEAD
git reset --soft origin/dev
git commit -m "Remediate Research Studio UX issues" \
  -m "Make /research-studio canonical, add Research Studio capability health, gate actions at capability boundaries, and verify local real summary generation with existing saved LLM credentials."
```

Then rerun focused tests from Steps 1-5 on the squashed commit.

- [ ] **Step 8: Force-update PR branch**

```bash
git push --force-with-lease origin codex/research-studio-degraded-health
```

- [ ] **Step 9: Verify PR commit count and checks**

```bash
gh pr view 1616 --repo rmusser01/tldw_server --json commits,baseRefName,headRefName,mergeStateStatus,url
gh pr checks 1616 --repo rmusser01/tldw_server
```

Expected: PR has one commit against `dev`; checks may be pending immediately after push.

---

## Review Notes

- This plan intentionally keeps real LLM generation local/manual. CI should not require provider credentials.
- If implementation discovers no reliable backend signal for `source_browse`, return `unknown/warn` and document the limitation. Do not invent readiness.
- If implementation discovers Research Studio chat does not always require RAG, update the capability dependency model and tests before coding the UI.
- Do not record secrets, full generated text, private file paths, or raw provider error bodies in Backlog or PR notes.
