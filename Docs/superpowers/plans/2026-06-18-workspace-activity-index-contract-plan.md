# Workspace Activity Index Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #1994 by adding a server-owned Workspace contained-resource index and recent activity contract without duplicating each owner tool UI.

**Architecture:** Add an append-only Workspace activity event table in ChaChaNotes and a read-only Workspace index service that composes existing Workspace registry, membership summaries, resolved membership previews, runtime bindings, and recovery warnings. Expose the contract through `GET /api/v1/workspaces/{workspace_id}/index`, then add a minimal frontend TypeScript contract/normalizer so UI surfaces can render it later without inventing wire semantics.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes SQLite/PostgreSQL schema helpers, existing Workspace membership/runtime services, Vitest for TypeScript normalizers, pytest for backend API/DB coverage.

---

## Stage 1: Backend Red Tests
**Goal:** Prove the missing DB and API behavior before production changes.
**Success Criteria:** Tests fail because activity primitives, schemas, and index endpoint do not exist.
**Tests:** `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py -q`
**Status:** Complete

### Task 1: DB Activity Contract Tests
**Files:**
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py`
- Modify: none

- [ ] Add tests that create a workspace, record membership/runtime-style activity events, and assert `list_workspace_activity_events()` returns stable timestamped rows in newest-first order.
- [ ] Add a redaction regression asserting event metadata does not expose obvious secret-shaped keys or absolute path values.
- [ ] Run the focused test file and confirm failure on missing `record_workspace_activity_event`.

### Task 2: API Index Contract Red Test
**Files:**
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py`
- Reuse fixtures/patterns from: `tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py`

- [ ] Build a small FastAPI test app with `workspaces_endpoint.router`.
- [ ] Seed memberships for at least `chat` and `prompt`, plus one unresolved/missing summary case.
- [ ] Assert `GET /api/v1/workspaces/workspace-1/index` returns:
  - `schema_version: 1`
  - Workspace identity/profile/archive state.
  - Workspace-level membership totals grouped by resource type and role.
  - Resource groups with bounded resolved item previews and owner `href` values.
  - Runtime binding summary and warnings.
  - Recent activity rows with `event_type`, `category`, timestamp, and safe metadata.
- [ ] Run the focused API test and confirm failure on missing route/schema.

---

## Stage 2: Activity Persistence
**Goal:** Add durable, bounded, secret-safe activity records for Workspace events.
**Success Criteria:** DB tests pass for event insert/list ordering and metadata safety.
**Tests:** `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py -q`
**Status:** Complete

### Task 3: Add ChaChaNotes Activity Table
**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

- [ ] Add `workspace_activity_events` to SQLite and PostgreSQL schema creation near other Workspace tables.
- [ ] Columns: `workspace_id`, `event_id`, `event_type`, `category`, `actor_user_id`, `resource_type`, `resource_id`, `summary`, `metadata_json`, `created_at`, `client_id`, `version`.
- [ ] Add indexes for `(workspace_id, created_at)` and `(workspace_id, category, created_at)`.
- [ ] Keep all DDL as fixed strings; do not interpolate identifiers.

### Task 4: Add Activity DB APIs
**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

- [ ] Add `record_workspace_activity_event(workspace_id, data, user_id=None)` with bounded JSON metadata normalization and secret/path redaction.
- [ ] Add `list_workspace_activity_events(workspace_id, limit=50, category=None)` with deterministic newest-first ordering.
- [ ] Normalize rows into response-safe dictionaries.
- [ ] Re-run the focused DB tests and fix only the minimal code needed.

---

## Stage 3: Read-Only Index Service And Schemas
**Goal:** Compose the index contract from existing Workspace services without owner-domain duplication.
**Success Criteria:** Service unit/API tests pass for grouped resources, totals, warnings, and recent activity.
**Tests:** `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py -q`
**Status:** Complete

### Task 5: Add Pydantic Response Models
**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`

- [ ] Add `WorkspaceIndexResponse`, `WorkspaceIndexResourceGroup`, `WorkspaceIndexRuntimeSummary`, `WorkspaceIndexWarning`, and `WorkspaceActivityEventResponse`.
- [ ] Reuse existing membership summary/response models where possible.
- [ ] Use explicit literals for warning severity/category where the set is known; keep reason codes as strings for forward compatibility.

### Task 6: Add Workspace Index Service
**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/activity_index.py`

- [ ] Implement `WorkspaceActivityIndexService.build_index(...)`.
- [ ] Use `WorkspaceMembershipService.workspace_membership_summary()` for global totals.
- [ ] For each active resource type with a nonzero count, call `list_workspace_memberships(resource_type=..., resolve=True, limit=group_limit)`.
- [ ] Use `db.list_workspace_runtime_bindings(..., limit=50)` for runtime grouping and warning inputs.
- [ ] Use `db.list_workspace_activity_events(..., limit=activity_limit)` for recent activity.
- [ ] Add warnings for archived/deleted workspace state, unresolved membership summaries, missing/degraded runtime bindings, and partial dependency failures.
- [ ] Return owner-domain `href` from membership summaries; never synthesize edit UI behavior inside the index.

---

## Stage 4: Endpoint And Event Hooks
**Goal:** Expose the index and start recording events from existing write paths.
**Success Criteria:** API tests pass and existing membership/runtime tests remain green.
**Tests:**
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py -q`
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py -q`
**Status:** Complete

### Task 7: Add Read Endpoint
**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`

- [ ] Add `GET /{workspace_id}/index` with `group_limit` and `activity_limit` query params.
- [ ] Inject the same optional Media/Prompts/Workflows/Watchlists DB dependencies used by memberships.
- [ ] Map DB/service errors through existing workspace HTTP error helpers.
- [ ] Keep the endpoint read-only and behind `WORKSPACES_READ_RATE_LIMIT`.

### Task 8: Record Activity Events
**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/membership_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

- [ ] Record `membership.linked`, `membership.restored`, and `membership.unlinked` events from membership service paths.
- [ ] Record `runtime_binding.upserted` and `runtime_binding.archived` events from runtime binding endpoint helper paths.
- [ ] Keep event recording best-effort only where the primary write already succeeded; log warning on event write failure without rolling back the primary write unless both occur inside the same DB transaction.
- [ ] Ensure metadata contains IDs, roles, transfer policy, and binding status/kind only; no secrets, raw paths, prompt bodies, model outputs, or file contents.

---

## Stage 5: Frontend Contract And Documentation
**Goal:** Provide the minimal client-side contract and document that this is an inspection/navigation surface.
**Success Criteria:** TypeScript normalizer tests pass and docs link the endpoint to #1994 acceptance criteria.
**Tests:** `./node_modules/.bin/vitest run src/services/workspace-index/__tests__/normalizers.test.ts --maxWorkers=1`
**Status:** Complete

### Task 9: Add Frontend Contract Normalizers
**Files:**
- Create: `apps/packages/ui/src/services/workspace-index/contracts.ts`
- Create: `apps/packages/ui/src/services/workspace-index/normalizers.ts`
- Create: `apps/packages/ui/src/services/workspace-index/index.ts`
- Create: `apps/packages/ui/src/services/workspace-index/__tests__/normalizers.test.ts`

- [ ] Define TypeScript contracts matching `WorkspaceIndexResponse`.
- [ ] Normalize warnings, resource groups, runtime summary, and activity rows into display-safe values.
- [ ] Preserve `href` links from the server without inventing owner UI routes.
- [ ] Add tests for unknown warning reason codes and empty index payloads.

### Task 10: Update Docs And Backlog
**Files:**
- Modify: `Docs/Design/Workspace_Container_Contract_2026_06.md`
- Modify: `backlog/tasks/task-2387 - Implement-Workspace-activity-and-contained-resource-index-contract-for-issue-1994.md`

- [ ] Document the new endpoint as an inspection/navigation contract, not a Workspace dashboard.
- [ ] List event categories and warnings with security constraints.
- [ ] Update Backlog with touched files and verification results.

---

## Stage 6: Final Verification
**Goal:** Prove the slice is merge-ready.
**Success Criteria:** Focused backend/frontend tests pass, whitespace check passes, Bandit reports no new findings for touched backend files.
**Tests:**
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py -q`
- `cd apps/packages/ui && ./node_modules/.bin/vitest run src/services/workspace-index/__tests__/normalizers.test.ts --maxWorkers=1`
- `git diff --check`
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Workspaces/activity_index.py tldw_Server_API/app/api/v1/endpoints/workspaces.py -f json -o /tmp/bandit_workspace_activity_index.json`
**Status:** Complete

### Task 11: Finalize
**Files:**
- Modify: Backlog task final summary.
- Create commit and PR after verification.

- [x] Rebase onto actual `origin/dev` once Git DNS is healthy.
- [x] Run final verification commands and record results.
- [x] Commit with a message referencing `TASK-2387` and GitHub #1994.
- [x] Push `codex/workspace-activity-index-contract` and open a PR against `dev`.
