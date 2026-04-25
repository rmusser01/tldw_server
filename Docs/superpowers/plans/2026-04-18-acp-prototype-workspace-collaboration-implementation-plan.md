# ACP Prototype Workspace Collaboration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first integrated `Prototype Workspace` slice: owner-created prototype workspaces with canonical snapshots, isolated collaborator sessions, private-link external collaboration, explicit promotion, and brokered previews on top of ACP/Sandbox.

**Architecture:** Add a new shared-metadata prototype domain in the AuthNZ/ops persistence layer, keep runtime state and filesystem payloads in ACP/Sandbox-backed infrastructure, and route long-running branch/session/publish work through Jobs. Expose a small, explicit API surface for prototype workspaces, sessions, previews, and promotions, then ship a minimal owner/collaborator WebUI that reuses existing ACP and sharing primitives without overloading research workspace semantics.

**Tech Stack:** FastAPI, Pydantic, AuthNZ `DatabasePool` repos + migrations, Jobs `WorkerSDK`, ACP/Sandbox services, React, TanStack Query, Zustand, Bun/Vitest, pytest/httpx, Bandit

---

## Scope Check

The approved spec spans persistence, sharing, runtime orchestration, and UI, but this is still one vertical slice rather than multiple unrelated projects. The implementation order below keeps the slice deployable in sequence:

1. metadata and identity foundation
2. sharing and external actor exchange
3. runtime jobs, snapshots, preview brokering, and promotion validation
4. API surface
5. minimal owner/collaborator UI

Do not split this into separate plans unless implementation stalls on infrastructure that clearly needs its own design cycle.

## Stage Overview

## Stage 1: Establish The Implementation Baseline
**Goal**: Work from an isolated branch/worktree and pin the current sharing, sandbox, ACP, and frontend contracts before adding prototype-specific code.
**Success Criteria**: A dedicated `codex/` branch exists, backend and frontend dependencies are ready, and baseline targeted tests pass before new red tests are added.
**Tests**:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_sharing_integration.py -q`
- `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/useSharing.auth.test.tsx src/store/__tests__/acp-sessions.test.ts`
**Status**: Not Started

## Stage 2: Build Shared Prototype Metadata And External Collaboration Identity
**Goal**: Add prototype tables, repo access, and the `Prototype Shared Actor` collaboration identity flow before touching ACP/Sandbox orchestration.
**Success Criteria**: Shared AuthNZ migrations create the prototype tables, repo tests pass, and private-link exchange can produce an audited external prototype actor without forcing a full user account.
**Tests**:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py -q`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py -q`
**Status**: Not Started

## Stage 3: Add Jobs-Backed Runtime, Snapshot, Preview, And Publish Validation
**Goal**: Implement idempotent branch-session creation, snapshot save, brokered preview handles, and publish validation against a fresh canonical runtime.
**Success Criteria**: Jobs state transitions are deterministic, preview handles are brokered rather than raw runtime URLs, and failed publish validation never advances the canonical pointer.
**Tests**:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py -q`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py -q`
**Status**: Not Started

## Stage 4: Expose The Prototype API Surface And Minimal WebUI
**Goal**: Ship the backend endpoints and a minimal owner/collaborator frontend flow that can create, share, enter, operate, and request promotion for a prototype workspace.
**Success Criteria**: Backend endpoint integration tests pass, frontend query/store/component tests pass, and a collaborator can enter through either internal auth or private-link exchange and reach a branch-session view.
**Tests**:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q`
- `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx`
**Status**: Not Started

## Stage 5: Harden, Document, And Verify The Slice End-To-End
**Goal**: Close the loop with audit/security checks, docs, and targeted end-to-end verification, including Bandit on the touched Python scope.
**Success Criteria**: Targeted backend/frontend suites pass, Bandit reports no new findings in touched Python paths, and implementation docs explain the prototype domain and its guardrails.
**Tests**:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`
- `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx`
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/core/Prototype_Workspaces tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py -f json -o /tmp/bandit_prototype_workspaces.json`
**Status**: Not Started

## File Map

- `tldw_Server_API/app/core/AuthNZ/migrations.py`
  Responsibility: add the next AuthNZ migration to create prototype workspace metadata tables, indexes, and the external shared-actor table in shared persistence.
- `tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py`
  Responsibility: single shared repo for `prototype_workspaces`, `prototype_snapshots`, `prototype_sessions`, `prototype_shared_actors`, and `prototype_promotion_requests`.
- `tldw_Server_API/app/core/AuthNZ/repos/__init__.py`
  Responsibility: export the new repo consistently with existing AuthNZ repo patterns.
- `tldw_Server_API/app/core/Prototype_Workspaces/models.py`
  Responsibility: internal typed models and enums for runtime profile, promotion status, actor kind, and preview state.
- `tldw_Server_API/app/core/Prototype_Workspaces/access.py`
  Responsibility: resolve internal user vs `Prototype Shared Actor` access context, enforce “exactly one actor identity” rules, and gate promoter-only operations.
- `tldw_Server_API/app/core/Prototype_Workspaces/service.py`
  Responsibility: orchestration service for create/list/get/update workspace, create branch session, save snapshot, submit promotion request, and mark stale.
- `tldw_Server_API/app/core/Prototype_Workspaces/preview_broker.py`
  Responsibility: issue `preview_handle` records, validate access, mint short-lived signed preview grants, and revoke brokered preview access.
- `tldw_Server_API/app/core/Prototype_Workspaces/jobs.py`
  Responsibility: enqueue prototype job requests with stable idempotency keys and explicit payload contracts.
- `tldw_Server_API/app/core/Prototype_Workspaces/jobs_worker.py`
  Responsibility: WorkerSDK consumer for prototype session bootstrap, preview boot/restart, snapshot save, and publish validation/promotion jobs.
- `tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`
  Responsibility: request/response models for prototype workspace CRUD, session state, preview state, promotions, and collaborator entry responses.
- `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`
  Responsibility: REST endpoint family for `/api/v1/prototype-workspaces/*`, `/prototype-sessions/*`, `/prototype-previews/*`, and `/prototype-promotions/*`.
- `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
  Responsibility: extend public/private sharing contracts with prototype-specific token exchange payloads and responses.
- `tldw_Server_API/app/api/v1/endpoints/sharing.py`
  Responsibility: add private-link exchange for prototype collaboration sessions without breaking existing workspace/chat sharing behavior.
- `tldw_Server_API/app/core/Sharing/share_token_service.py`
  Responsibility: support prototype resource types and password-verified private-link exchange semantics.
- `tldw_Server_API/app/main.py`
  Responsibility: register the new prototype router in the same optional-router pattern already used for sharing and sandbox endpoints.
- `tldw_Server_API/tests/PrototypeWorkspaces/conftest.py`
  Responsibility: shared fake-pool/test-app fixtures for prototype repo and endpoint tests.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py`
  Responsibility: migration/repo coverage for metadata persistence, actor identity exclusivity, and cross-table invariants.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py`
  Responsibility: private-link to `Prototype Shared Actor` exchange tests and revocation/expiry enforcement.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py`
  Responsibility: job payload/idempotency/cleanup coverage for branch session bootstrap, preview boot, and snapshot save.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py`
  Responsibility: brokered preview handle issuance, renewal, access, and revocation tests.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py`
  Responsibility: publish-validation, stale protection, and last-known-good pointer tests.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`
  Responsibility: API integration coverage for owner flow, collaborator flow, and promotion flow.
- `apps/packages/ui/src/types/prototype-workspace.ts`
  Responsibility: frontend types for prototype workspace records, sessions, previews, promotion requests, and external collaborator entry results.
- `apps/packages/ui/src/services/tldw/domains/prototype-workspaces.ts`
  Responsibility: `TldwApiClient` domain methods for prototype APIs and preview/sign-in exchange.
- `apps/packages/ui/src/services/tldw/domains/index.ts`
  Responsibility: export the new domain methods.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  Responsibility: mix the new prototype domain methods into the shared API client.
- `apps/packages/ui/src/hooks/usePrototypeWorkspaces.ts`
  Responsibility: TanStack Query hooks for prototype workspace CRUD, session creation, promotion requests, and private-link exchange.
- `apps/packages/ui/src/store/prototype-workspace.ts`
  Responsibility: minimal client-side store for selected workspace/session/promotion candidate state and collaborator entry context.
- `apps/packages/ui/src/hooks/useSharing.ts`
  Responsibility: add prototype-share public exchange and token-verification hooks where reuse is cleaner than creating a parallel client.
- `apps/packages/ui/src/types/sharing.ts`
  Responsibility: extend share resource and public-link types for prototype collaboration.
- `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspacePage.tsx`
  Responsibility: top-level owner/collaborator screen selection and data loading.
- `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspaceOwnerView.tsx`
  Responsibility: canonical preview, branch inventory, candidate list, and promoter/share controls.
- `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspaceSessionView.tsx`
  Responsibility: collaborator branch preview plus entry points into ACP-backed prompt/files/terminal controls.
- `apps/packages/ui/src/components/Option/PrototypeWorkspace/index.tsx`
  Responsibility: public export surface for the feature.
- `apps/packages/ui/src/routes/option-prototype-workspaces.tsx`
  Responsibility: route-level shell for the prototype workspace feature.
- `apps/packages/ui/src/hooks/__tests__/usePrototypeWorkspaces.test.tsx`
  Responsibility: hook contract tests for query keys, request shapes, and mutation invalidation.
- `apps/packages/ui/src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx`
  Responsibility: owner/collaborator rendering and action-state tests.
- `apps/packages/ui/src/routes/__tests__/option-prototype-workspaces.route.test.tsx`
  Responsibility: route wiring and guard behavior tests.
- `Docs/API-related/Prototype_Workspaces_API.md`
  Responsibility: API and runtime-policy doc for the new slice, including external collaborator guardrails.
- `Docs/superpowers/specs/2026-04-18-acp-prototype-workspace-collaboration-design.md`
  Responsibility: design reference only; update only if implementation reveals a material design mismatch that needs to be documented.
- `Docs/superpowers/plans/2026-04-18-acp-prototype-workspace-collaboration-implementation-plan.md`
  Responsibility: execution checklist and status tracking; update only with real progress.

### Task 1: Prepare The Worktree And Baseline

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-18-acp-prototype-workspace-collaboration-implementation-plan.md`

- [ ] **Step 1: Create or switch to an isolated worktree**

```bash
git worktree add ../tldw_server2-prototype-workspaces -b codex/prototype-workspace-collaboration
```

Expected: a new worktree exists on branch `codex/prototype-workspace-collaboration`.

- [ ] **Step 2: Install backend and frontend dependencies in the new worktree**

Run: `source .venv/bin/activate && python -V`

Run: `cd apps && bun install --frozen-lockfile`

Expected: Python commands resolve inside the repo virtualenv and frontend packages are ready without lockfile drift.

- [ ] **Step 3: Run the baseline backend sharing tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py tldw_Server_API/tests/Sharing/test_sharing_integration.py -q`

Expected: PASS. This confirms the existing sharing path is stable before prototype-specific extension work starts.

- [ ] **Step 4: Run the baseline frontend sharing/ACP store tests**

Run: `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/useSharing.auth.test.tsx src/store/__tests__/acp-sessions.test.ts`

Expected: PASS. This confirms the existing client sharing and ACP session store contracts are stable before adding prototype flows.

- [ ] **Step 5: Commit the clean-room baseline checkpoint**

```bash
git add Docs/superpowers/plans/2026-04-18-acp-prototype-workspace-collaboration-implementation-plan.md
git commit -m "docs: add prototype workspace implementation plan"
```

### Task 2: Add Shared Prototype Metadata Tables And Repo Contracts

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/conftest.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/__init__.py`

- [ ] **Step 1: Write the failing repo and migration tests**

```python
async def test_create_workspace_snapshot_and_session_enforce_single_actor_identity(repo):
    workspace = await repo.create_workspace(owner_user_id=1, title="demo", creation_source="prompt")
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=11,
        display_name="Stakeholder A",
        runtime_policy_profile="locked_collab",
    )

    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id="snap_1",
        actor_shared_actor_id=actor["id"],
        actor_type="external_collaborator",
    )

    assert session["actor_shared_actor_id"] == actor["id"]
    assert session["actor_user_id"] is None
```

Also add a migration test asserting the next AuthNZ migration creates:
- `prototype_workspaces`
- `prototype_snapshots`
- `prototype_sessions`
- `prototype_shared_actors`
- `prototype_promotion_requests`

- [ ] **Step 2: Run the new repo tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py -q`

Expected: FAIL because the migration and repo do not exist yet.

- [ ] **Step 3: Implement the minimal migration and repo**

```python
async def create_session(
    self,
    *,
    prototype_workspace_id: str,
    base_snapshot_id: str,
    actor_type: str,
    actor_user_id: int | None = None,
    actor_shared_actor_id: str | None = None,
) -> dict[str, Any]:
    if (actor_user_id is None) == (actor_shared_actor_id is None):
        raise ValueError("exactly one actor identity must be set")
    ...
```

Implementation requirements:
- use the next AuthNZ migration number after `083`
- keep repo JSON loading/boolean normalization patterns consistent with `shared_workspace_repo.py`
- add indexes for `(prototype_workspace_id, updated_at)` and `(share_link_id, revoked_at)`

- [ ] **Step 4: Re-run the repo tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the metadata foundation**

```bash
git add tldw_Server_API/app/core/AuthNZ/migrations.py \
        tldw_Server_API/app/core/AuthNZ/repos/__init__.py \
        tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py \
        tldw_Server_API/tests/PrototypeWorkspaces/conftest.py \
        tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py
git commit -m "feat: add prototype workspace shared metadata store"
```

### Task 3: Extend Sharing For Prototype Private-Link Exchange

**Files:**
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/access.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/app/core/Sharing/share_token_service.py`

- [ ] **Step 1: Add failing tests for private-link to shared-actor exchange**

```python
def test_public_prototype_exchange_creates_shared_actor(client, prototype_share_token):
    resp = client.post(
        f"/api/v1/sharing/public/{prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM", "password": "demo-pass"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["actor_type"] == "external_collaborator"
    assert body["shared_actor_id"].startswith("psa_")
    assert body["session_token"]
```

Also cover:
- revoked link returns 404
- bad password returns 403
- non-prototype token returns 422
- repeated exchange on the same browser session resumes the same shared actor when allowed by policy

- [ ] **Step 2: Run the exchange tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py -q`

Expected: FAIL because the endpoint and response schema do not exist yet.

- [ ] **Step 3: Implement the prototype exchange endpoint and access context helper**

```python
class PrototypeLinkExchangeResponse(BaseModel):
    shared_actor_id: str
    actor_type: Literal["external_collaborator"] = "external_collaborator"
    session_token: str
    runtime_policy_profile: str
```

Implementation requirements:
- do not mutate the global `AuthPrincipal` model for this flow
- keep external collaboration identity inside `Prototype_Workspaces/access.py`
- require password verification when the token is protected
- create or resume a `Prototype Shared Actor` and return a short-lived collaboration token

- [ ] **Step 4: Re-run sharing and exchange tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q`

Expected: PASS. Existing sharing tests should remain green.

- [ ] **Step 5: Commit the sharing extension checkpoint**

```bash
git add tldw_Server_API/app/core/Prototype_Workspaces/access.py \
        tldw_Server_API/app/api/v1/schemas/sharing_schemas.py \
        tldw_Server_API/app/api/v1/endpoints/sharing.py \
        tldw_Server_API/app/core/Sharing/share_token_service.py \
        tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py
git commit -m "feat: add prototype private-link collaboration exchange"
```

### Task 4: Implement Jobs-Backed Branch Sessions, Preview Broker, And Publish Validation

**Files:**
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/models.py`
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/service.py`
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/preview_broker.py`
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/jobs.py`
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/jobs_worker.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py`

- [ ] **Step 1: Write failing runtime and publish-validation tests**

```python
async def test_promote_candidate_requires_publish_validation(repo, prototype_service):
    candidate = await repo.create_snapshot(
        prototype_workspace_id="pw_1",
        parent_snapshot_id="snap_base",
        created_from_session_id="sess_1",
        author_user_id=1,
        storage_ref="snapshots/pw_1/snap_candidate.tar",
    )

    result = await prototype_service.promote_candidate(
        prototype_workspace_id="pw_1",
        candidate_snapshot_id=candidate["id"],
        reviewer_user_id=1,
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "publish_validation_failed"
```

Also add tests for:
- idempotent `create branch session`
- one active broker target per preview scope
- revocation invalidates future preview grants
- stale candidate never advances canonical state
- failed validation preserves `last_known_good_snapshot_id`

- [ ] **Step 2: Run the new runtime tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q`

Expected: FAIL because the runtime services and worker do not exist yet.

- [ ] **Step 3: Implement the service, preview broker, and worker skeleton**

```python
PROTOTYPE_JOB_TYPES = {
    "branch_session_bootstrap",
    "preview_boot",
    "snapshot_save",
    "publish_validate_and_promote",
}

def build_promote_idempotency_key(
    prototype_workspace_id: str,
    candidate_snapshot_id: str,
    canonical_snapshot_id: str,
) -> str:
    return f"prototype:promote:{prototype_workspace_id}:{candidate_snapshot_id}:{canonical_snapshot_id}"
```

Implementation requirements:
- keep preview access brokered as `preview_handle` records, not raw runtime URLs
- use WorkerSDK patterns already used in `app/core/File_Artifacts/jobs_worker.py`
- publish validation must restore a snapshot into a fresh runtime profile before flipping the canonical pointer
- cleanup retries must terminate orphan preview/runtime resources before creating replacements

- [ ] **Step 4: Re-run the runtime tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the runtime orchestration layer**

```bash
git add tldw_Server_API/app/core/Prototype_Workspaces/models.py \
        tldw_Server_API/app/core/Prototype_Workspaces/service.py \
        tldw_Server_API/app/core/Prototype_Workspaces/preview_broker.py \
        tldw_Server_API/app/core/Prototype_Workspaces/jobs.py \
        tldw_Server_API/app/core/Prototype_Workspaces/jobs_worker.py \
        tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py \
        tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py \
        tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py
git commit -m "feat: add prototype runtime jobs and preview broker"
```

### Task 5: Expose The Prototype Backend API

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`
- Modify: `tldw_Server_API/app/main.py`

- [ ] **Step 1: Write failing endpoint integration tests for the owner and collaborator flows**

```python
def test_owner_can_create_workspace_and_request_branch_session(client):
    created = client.post(
        "/api/v1/prototype-workspaces",
        json={"title": "Sales dashboard", "creation_source": "prompt", "prompt": "Build a B2B dashboard"},
    )
    assert created.status_code == 201

    workspace_id = created.json()["id"]
    session = client.post(f"/api/v1/prototype-workspaces/{workspace_id}/sessions", json={})
    assert session.status_code == 202
    assert session.json()["job_type"] == "branch_session_bootstrap"
```

Also cover:
- collaborator session creation using prototype session token
- promotion request submission
- owner promotion review
- preview grant renewal
- stale promotion response shape

- [ ] **Step 2: Run the endpoint tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q`

Expected: FAIL because the router and schemas do not exist yet.

- [ ] **Step 3: Implement the router and schemas with minimal happy-path behavior first**

```python
@router.post("/prototype-workspaces", response_model=PrototypeWorkspaceResponse, status_code=201)
async def create_prototype_workspace(
    body: PrototypeWorkspaceCreateRequest,
    user: User = Depends(get_request_user),
):
    return await _get_service().create_workspace(owner_user_id=user.id, body=body)
```

Implementation requirements:
- keep endpoint families consistent with the approved spec
- use explicit Pydantic models for job responses, preview responses, and promotion responses
- include the new router in `main.py` using the same conditional include style already used for sharing/sandbox routers

- [ ] **Step 4: Re-run the endpoint integration tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the backend API slice**

```bash
git add tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py \
        tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py \
        tldw_Server_API/app/main.py \
        tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py
git commit -m "feat: add prototype workspace api endpoints"
```

### Task 6: Ship The Minimal Owner And Collaborator WebUI

**Files:**
- Create: `apps/packages/ui/src/types/prototype-workspace.ts`
- Create: `apps/packages/ui/src/services/tldw/domains/prototype-workspaces.ts`
- Create: `apps/packages/ui/src/hooks/usePrototypeWorkspaces.ts`
- Create: `apps/packages/ui/src/store/prototype-workspace.ts`
- Create: `apps/packages/ui/src/components/Option/PrototypeWorkspace/index.tsx`
- Create: `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspacePage.tsx`
- Create: `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspaceOwnerView.tsx`
- Create: `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspaceSessionView.tsx`
- Create: `apps/packages/ui/src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx`
- Create: `apps/packages/ui/src/hooks/__tests__/usePrototypeWorkspaces.test.tsx`
- Create: `apps/packages/ui/src/routes/option-prototype-workspaces.tsx`
- Create: `apps/packages/ui/src/routes/__tests__/option-prototype-workspaces.route.test.tsx`
- Modify: `apps/packages/ui/src/services/tldw/domains/index.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/hooks/useSharing.ts`
- Modify: `apps/packages/ui/src/types/sharing.ts`

- [ ] **Step 1: Write failing client and component tests**

```tsx
it("renders owner controls when the current viewer owns the prototype", async () => {
  mockUsePrototypeWorkspace.mockReturnValue({
    data: { id: "pw_1", title: "Sales dashboard", viewer_role: "owner" },
    isLoading: false,
  })

  render(<PrototypeWorkspacePage workspaceId="pw_1" />)

  expect(screen.getByText("Promotion controls")).toBeInTheDocument()
  expect(screen.getByText("Sharing controls")).toBeInTheDocument()
})
```

Also cover:
- collaborator view renders branch-session panel instead of owner inventory
- private-link exchange hook calls the new sharing endpoint
- route component mounts the feature inside `OptionLayout`/`PageShell`

- [ ] **Step 2: Run the new frontend tests to verify they fail**

Run: `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx`

Expected: FAIL because the new hooks, types, and route do not exist yet.

- [ ] **Step 3: Implement the minimal domain methods, hooks, store, and screens**

```ts
export const prototypeWorkspaceMethods = {
  async createPrototypeWorkspace(this: TldwApiClientCore, payload: CreatePrototypeWorkspaceRequest) {
    return await bgRequest({
      path: "/api/v1/prototype-workspaces",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    })
  },
}
```

Implementation requirements:
- keep the API client integration consistent with `workspace-api.ts`
- reuse existing ACP UI primitives instead of inventing a parallel terminal/chat system
- keep the first UI intentionally narrow: canonical preview, branch session entry, candidate list, promotion request action, and share controls

- [ ] **Step 4: Re-run the frontend tests**

Run: `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx`

Expected: PASS.

- [ ] **Step 5: Commit the frontend MVP**

```bash
git add apps/packages/ui/src/types/prototype-workspace.ts \
        apps/packages/ui/src/services/tldw/domains/prototype-workspaces.ts \
        apps/packages/ui/src/services/tldw/domains/index.ts \
        apps/packages/ui/src/services/tldw/TldwApiClient.ts \
        apps/packages/ui/src/hooks/usePrototypeWorkspaces.ts \
        apps/packages/ui/src/store/prototype-workspace.ts \
        apps/packages/ui/src/hooks/useSharing.ts \
        apps/packages/ui/src/types/sharing.ts \
        apps/packages/ui/src/components/Option/PrototypeWorkspace \
        apps/packages/ui/src/routes/option-prototype-workspaces.tsx \
        apps/packages/ui/src/routes/__tests__/option-prototype-workspaces.route.test.tsx \
        apps/packages/ui/src/hooks/__tests__/usePrototypeWorkspaces.test.tsx
git commit -m "feat: add prototype workspace web ui"
```

### Task 7: Verify, Document, And Close The Slice

**Files:**
- Create: `Docs/API-related/Prototype_Workspaces_API.md`
- Modify: `Docs/superpowers/plans/2026-04-18-acp-prototype-workspace-collaboration-implementation-plan.md`

- [ ] **Step 1: Write the operator/developer doc**

Document:
- prototype workspace lifecycle
- runtime profiles (`template_demo`, `repo_bootstrap`, `locked_collab`)
- external collaborator `Prototype Shared Actor` model
- preview broker behavior
- promotion and publish-validation guarantees

- [ ] **Step 2: Run the targeted backend test suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q`

Expected: PASS.

- [ ] **Step 3: Run the targeted frontend test suite**

Run: `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx src/hooks/__tests__/useSharing.auth.test.tsx`

Expected: PASS.

- [ ] **Step 4: Run Bandit on the touched Python scope**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/core/Prototype_Workspaces tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py -f json -o /tmp/bandit_prototype_workspaces.json`

Expected: JSON output written to `/tmp/bandit_prototype_workspaces.json` with no new findings in touched code.

- [ ] **Step 5: Commit docs and verification notes**

```bash
git add Docs/API-related/Prototype_Workspaces_API.md \
        Docs/superpowers/plans/2026-04-18-acp-prototype-workspace-collaboration-implementation-plan.md
git commit -m "docs: add prototype workspace api and verification notes"
```

## Verification Checklist

- [ ] Shared AuthNZ migration creates and indexes all prototype tables.
- [ ] External private-link exchange yields a `Prototype Shared Actor`, not a full AuthNZ user.
- [ ] Branch session bootstrap and promotion jobs are idempotent and cleanup-aware.
- [ ] Preview access is brokered by `preview_handle` rather than leaked raw runtime URLs.
- [ ] Publish validation must succeed before canonical pointer updates.
- [ ] Frontend owner and collaborator views both work against the same backend contracts.
- [ ] Bandit is green on touched Python scope.

## Open Decisions To Confirm During Execution

- [ ] Whether external shared actors should resume across browser sessions by cookie/session-binding only or by explicit invite/display-name re-entry.
- [ ] Whether the first UI cut should live only under an option route or also surface an entry point from existing workspace navigation.
- [ ] Whether prototype preview renewal should sit in `useSharing.ts` or move immediately into `usePrototypeWorkspaces.ts` after the first passing vertical slice.
