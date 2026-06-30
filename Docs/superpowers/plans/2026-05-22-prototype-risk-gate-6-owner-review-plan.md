# Prototype Risk Gate 6 Owner Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Harden the prototype workspace owner review and promotion UX for GitHub issue #1458.

**Architecture:** Add the smallest backend contract surface the owner UI needs: promotion request summaries embedded in owner workspace detail, plus a typed review mutation for the existing review endpoint. The frontend then renders promotion request state separately from runtime state and disables approve/reject actions when backend semantics would reject, the branch session is not actionable, or validation/review is already terminal.

**Tech Stack:** FastAPI/Pydantic backend, AuthNZ prototype repository, React, TanStack Query, Vitest/Testing Library, Bun.

---

### Task 1: Expose Promotion Requests In Owner Detail

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`
- Test: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py`
- Test: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`

- [x] **Step 1: Write failing repository test**

Add coverage that `list_promotion_requests_for_workspace(workspace_id)` returns only that workspace's requests ordered by newest update first.

- [x] **Step 2: Run repository test red**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py -q -k promotion_requests_for_workspace`

Expected: FAIL because the repository method does not exist.

- [x] **Step 3: Add minimal repository method**

Implement the method using the existing AuthNZ repo query style and `_normalize_promotion_request_row`.

- [x] **Step 4: Add endpoint/schema failing test**

Add coverage that `GET /api/v1/prototype-workspaces/{id}` includes `promotion_requests` with pending/stale records and candidate/session ids.

- [x] **Step 5: Run endpoint test red**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q -k promotion_requests`

Expected: FAIL because the response schema/detail builder does not expose promotion requests.

- [x] **Step 6: Add response schema and detail builder field**

Add `PrototypePromotionRequestSummaryResponse` and `promotion_requests` to `PrototypeWorkspaceDetailResponse`; call the repo method in `_build_workspace_detail_response`.

- [x] **Step 7: Run backend focused tests green**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q -k "promotion_requests or owner_can_review_promotion_request or stale_promotion_response_shape"`

Expected: PASS.

### Task 2: Add Frontend Review Client And Hook

**Files:**
- Modify: `apps/packages/ui/src/types/prototype-workspace.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/prototype-workspaces.ts`
- Modify: `apps/packages/ui/src/hooks/usePrototypeWorkspaces.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/usePrototypeWorkspaces.test.tsx`

- [x] **Step 1: Write failing hook/client tests**

Add tests for:
- `useReviewPrototypePromotionRequest` posts approve/reject payloads to `/prototype-promotions/{id}/review`.
- structured review errors from the contract fixture are preserved.

- [x] **Step 2: Run frontend hook tests red**

Run: `bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because review types/client/hook do not exist.

- [x] **Step 3: Add types, client function, and mutation hook**

Add `PrototypePromotionReviewInput`, `PrototypePromotionReviewResult`, `reviewPrototypePromotionRequestRequest`, and `useReviewPrototypePromotionRequest`. Invalidate workspace detail and promotion query keys on success.

- [x] **Step 4: Run hook tests green**

Run: `bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: PASS.

### Task 3: Harden Owner Review UX

**Files:**
- Modify: `apps/packages/ui/src/components/Option/PrototypeWorkspace/PrototypeWorkspaceOwnerView.tsx`
- Test: `apps/packages/ui/src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx`

- [x] **Step 1: Write failing owner-view tests**

Cover:
- pending promotion request renders candidate/session/reviewer context and enables approve/reject.
- stale/conflict/failed/promoted/rejected states render distinct user-facing state copy.
- approve is disabled when the request is terminal, stale, conflict, failed, lacks a candidate snapshot, has a revoked/expired branch session, or review is in flight.
- branch inventory distinguishes runtime status from preview status and marks revoked/expired sessions as not actionable.

- [x] **Step 2: Run owner-view tests red**

Run: `bunx vitest run src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because the component does not render promotion requests or review actions yet.

- [x] **Step 3: Implement owner review surface**

Render a promotion review section that derives action availability from request status and snapshot/session presence. Wire approve/reject to the review mutation and show validation results without mixing them with runtime branch state.

- [x] **Step 4: Run owner-view tests green**

Run: `bunx vitest run src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: PASS.

### Task 4: Verify And Close Out

**Files:**
- Modify: `backlog/tasks/task-479 - Risk-Gate-6-prototype-owner-review-and-promotion-UX-hardening.md`

- [x] **Step 1: Run focused frontend suite**

Run: `bunx vitest run src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceSessionView.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx --maxWorkers=1 --no-file-parallelism`

- [x] **Step 2: Run focused backend suite**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`

- [x] **Step 3: Run Bandit for touched backend paths**

Run: `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py -f json -o /tmp/bandit_prototype_risk_gate_6.json`

- [x] **Step 4: Run whitespace check**

Run: `git diff --check`

- [x] **Step 5: Update Backlog task**

Record verification results, known skips, and final summary in `TASK-479`.

- [x] **Step 6: Commit**

Commit the implementation with a message that names Risk Gate 6 and issue #1458.
