# Paperless Source Review Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist and expose per-workspace source review state so Research Workspace can separate document triage (`unset`, `needs_review`, `reviewed`) from ingestion/indexing readiness.

**Architecture:** Add review lifecycle columns and transition helpers to `workspace_sources`, then expose those fields through the existing workspace source/status/context contracts without changing processing lifecycle semantics. Add single-source and batch review updates, WebUI source filters/actions, and opt-in creation defaults for workspace attachment paths that create `workspace_sources`.

**Tech Stack:** SQLite-backed `CharactersRAGDB`, FastAPI, Pydantic, existing workspace endpoints, React, Zustand workspace store, Vitest, Pytest, Bandit.

---

## Product Contract

- Review state belongs to each `workspace_sources` association, not to global media rows.
- Valid review states are `unset`, `needs_review`, and `reviewed`.
- Review state must never be overloaded into processing lifecycle fields such as `queued`, `indexing`, `queryable`, or `failed`.
- Existing workspace sources migrate/default to `unset` and appear in an explicit unreviewed bucket, not as reviewed.
- `review_state_updated_at` changes whenever a review-state write is accepted.
- `reviewed_at` and `reviewed_by_user_id` are set only when state is `reviewed`; both are cleared when state becomes `unset` or `needs_review`.
- Source creation paths may create `unset` or `needs_review` sources only. `reviewed` must be reached through an authenticated review transition so the actor/timestamp contract is not bypassed.
- Quick Ingest/browser-extension workspace attachments may default a new workspace source to `needs_review` only when the workspace attach path explicitly opts in.
- Later tasks own saved source views, duplicate recovery, document-storage policy, and a Paperless-style document detail panel. This task only ships the persisted lifecycle and basic WebUI controls needed to use it.

## File Map

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: workspace source table definitions, migration/backfill, normalization, single/batch transition helpers.
- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`: `WorkspaceSourceReviewState`, source response fields, batch request schema, status/context response fields.
- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`: source response helpers, create/update source handling, new batch review-state endpoint, context payload propagation.
- `tldw_Server_API/app/core/Workspaces/status_projection.py`: source-status payload propagation for review fields while preserving processing lifecycle state.
- `tldw_Server_API/app/core/WebClipper/service.py`: opt-in default review state for promoted workspace sources created from browser clips.
- `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py`: DB persistence, migration/defaults, transition, batch behavior.
- `tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py`: API source create/update/batch contract and permission/error behavior.
- `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`: source status response includes review fields separately from processing lifecycle.
- `tldw_Server_API/tests/Workspaces/test_workspace_context_api.py`: workspace context source rows include review fields.
- `tldw_Server_API/app/core/WebClipper/schemas.py`: WebClipper workspace payload opt-in field for review default.
- `tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`: opt-in workspace source default for WebClipper source promotion.
- `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`: WebClipper API accepts and applies the workspace review opt-in.
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`: TS API response/request types plus batch review-state client method.
- `apps/packages/ui/src/store/workspace-api.ts`: server-to-local source mapping for review fields.
- `apps/packages/ui/src/types/workspace.ts`: local `WorkspaceSourceReviewState` and timestamp/actor fields.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts`: local-to-server source create payload mapping.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-list-view.ts`: dedicated `reviewStateFilters` and summary logic.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceAdvancedControls.tsx`: review-state filter controls.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`: row/detail badges, single-source actions, bulk actions.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx`: opt-in default for sources explicitly added for review.
- `apps/packages/ui/src/entries/background.ts`: inspected context-menu ingest boundary; current media-only Quick Ingest path is not a workspace source attachment path for this task.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts`: review-state filtering.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx`: row/detail/bulk review UI.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`: source create payload includes review state when present locally.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`: add-source review default UI behavior.
- `apps/packages/ui/src/services/web-clipper/types.ts`: frontend WebClipper workspace payload opt-in type.
- `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`: sidepanel WebClipper review opt-in control and save payload.
- `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`: sidepanel WebClipper opt-in save payload behavior.
- `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts` or `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`: API client route coverage.

## Stage 1: Backend Persistence and Transition Semantics

**Goal:** Store review state on workspace source rows and make transitions deterministic.

**Success Criteria:** New and existing workspace sources expose `review_state`, `review_state_updated_at`, `reviewed_at`, and `reviewed_by_user_id`; defaults are `unset`; review transitions set/clear reviewed-only fields.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py`

- [ ] **Step 1: Write failing DB tests**
  - Add a source without review fields and assert `review_state == "unset"`, `review_state_updated_at` is populated, and reviewed-only fields are `None`.
  - Add a source with `review_state: "needs_review"` and assert it persists.
  - Update a source to `reviewed` and assert `reviewed_at`, `reviewed_by_user_id`, `review_state_updated_at`, and `version` update.
  - Update the same source back to `needs_review` and assert `reviewed_at` and `reviewed_by_user_id` are cleared.
  - Batch update two source IDs and assert both rows transition while unrelated rows are unchanged.
  - Create a database from a minimal pre-review `workspace_sources` schema and assert initialization/backfill produces `unset` review state for existing rows.

- [ ] **Step 2: Run DB tests and verify failures**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py -q`
  - Expected: FAIL because review columns and helpers do not exist.

- [ ] **Step 3: Add schema columns and migration/backfill**
  - Add columns to every `workspace_sources` table definition:
    - `review_state TEXT NOT NULL DEFAULT 'unset'`
    - `review_state_updated_at TEXT`
    - `reviewed_at TEXT`
    - `reviewed_by_user_id TEXT`
  - Add migration/backfill logic that safely handles existing DBs:
    - Missing `review_state` becomes `unset`.
    - Missing `review_state_updated_at` becomes `COALESCE(added_at, current_utc_timestamp)`.
    - `reviewed_at` and `reviewed_by_user_id` stay `NULL` unless a valid existing `reviewed` row is ever introduced.
  - Enforce allowed values in DB normalization code even if older SQLite migration paths cannot add a `CHECK` constraint.

- [ ] **Step 4: Add transition helpers**
  - Add a private normalizer with this behavior:
    - `None` or blank input normalizes to `unset` for create.
    - Create accepts only `unset` and `needs_review`.
    - Update/transition accepts `unset`, `needs_review`, and `reviewed`.
    - Any other value raises `InputError`.
  - Add a private transition builder with this behavior:
    - All accepted review-state writes set `review_state_updated_at` to `now`.
    - `reviewed` sets `reviewed_at = now` and `reviewed_by_user_id = normalized actor user id`.
    - `unset` and `needs_review` set `reviewed_at = NULL` and `reviewed_by_user_id = NULL`.
  - Extend `add_workspace_source()` so new rows can receive `review_state` and always get `review_state_updated_at`.
  - Extend `update_workspace_source()` with an optional `actor_user_id` keyword and review-state transition support while preserving optimistic locking.
  - Add `update_workspace_source_review_states(workspace_id, source_ids, review_state, actor_user_id)` for batch operations. Normalize duplicate IDs, fail with `ConflictError` when any requested source is outside/missing from the workspace, and bump versions for changed rows.

- [ ] **Step 5: Run DB tests and verify pass**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py -q`
  - Expected: PASS.

## Stage 2: Backend API Contract

**Goal:** Expose review fields through existing source read models and add review-state write endpoints without mixing them with processing status.

**Success Criteria:** `/sources`, `/sources/status`, and `/context` include review fields; single-source update accepts `review_state`; batch review update exists; processing `state` still only describes ingestion/readiness.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Modify: `tldw_Server_API/app/core/Workspaces/status_projection.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_context_api.py`

- [ ] **Step 1: Write failing API tests**
  - `POST /api/v1/workspaces/{workspace_id}/sources` without review state returns `review_state: "unset"` and review timestamps.
  - `POST /sources` with `review_state: "needs_review"` returns `needs_review`.
  - `PUT /sources/{source_id}` with `review_state: "reviewed"` records the authenticated user as `reviewed_by_user_id`.
  - `PUT /sources/{source_id}` with `review_state: "needs_review"` clears reviewed-only fields.
  - Invalid review state returns a validation/input error.
  - `PUT /sources/review-state` updates selected IDs and returns updated source rows.
  - Batch update with a missing source ID fails without partially updating rows.
  - `/sources/status` includes review fields and still returns lifecycle `state: "queryable"` or another processing state.
  - `/context` source items include review fields.

- [ ] **Step 2: Run focused API tests and verify failures**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspace_context_api.py -q`
  - Expected: FAIL because schemas/endpoint payloads do not include review fields.

- [ ] **Step 3: Add schemas**
  - Add `WorkspaceSourceReviewState = Literal["unset", "needs_review", "reviewed"]`.
  - Add `WorkspaceSourceCreateReviewState = Literal["unset", "needs_review"]`.
  - Add optional `review_state: WorkspaceSourceCreateReviewState = "unset"` to `WorkspaceSourceCreateRequest`.
  - Add optional `review_state: WorkspaceSourceReviewState | None = None` to `WorkspaceSourceUpdateRequest`.
  - Add to `WorkspaceSourceResponse`, `WorkspaceSourceStatusResponse`, and context source schemas:
    - `review_state: WorkspaceSourceReviewState = "unset"`
    - `review_state_updated_at: str | None = None`
    - `reviewed_at: str | None = None`
    - `reviewed_by_user_id: str | None = None`
  - Add `WorkspaceSourceReviewStateBatchRequest`:
    - `source_ids: list[str] = Field(..., min_length=1, max_length=500)`
    - `review_state: WorkspaceSourceReviewState`

- [ ] **Step 4: Wire endpoint payloads**
  - Update `_src_to_response()` to include the four review fields.
  - Pass `actor_user_id=str(getattr(current_user, "id", ""))` into source updates that include `review_state`.
  - Add `PUT /api/v1/workspaces/{workspace_id}/sources/review-state` before the dynamic `/{source_id}` route. Use the same workspace write dependency and `_require_workspace()` pattern as source selection/reorder.
  - Return `list[WorkspaceSourceResponse]` from the batch endpoint so the WebUI can update local rows without a full reload.
  - Update `_context_source_payload()` to carry review fields through `_src_to_response()`.

- [ ] **Step 5: Wire status projection without lifecycle drift**
  - Add the four review fields to `_base_status()` output in `status_projection.py`.
  - Do not add review states to `WorkspaceSourceLifecycleState`.
  - Do not alter source summary counts (`queryable`, `processing`, `failed`, etc.) in this task.

- [ ] **Step 6: Run focused API tests and verify pass**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspace_context_api.py -q`
  - Expected: PASS.

## Stage 3: WebUI Types, Filtering, and Review Actions

**Goal:** Make review state visible and actionable in Research Workspace with filters separate from readiness/status filters.

**Success Criteria:** Source rows/details show review state; users can mark one or multiple sources reviewed/needs-review; source view state has dedicated `reviewStateFilters`; Needs Review and Unreviewed presets can be represented without using processing status.

**Files:**
- Modify: `apps/packages/ui/src/types/workspace.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/store/workspace-api.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-list-view.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceAdvancedControls.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`

- [ ] **Step 1: Write failing TypeScript/unit tests**
  - `filterSources()` can filter `reviewStateFilters: ["needs_review"]`.
  - `filterSources()` can filter `reviewStateFilters: ["unset"]`.
  - `buildSourceFilterSummary()` reports review filters separately from status filters.
  - API client exposes `updateWorkspaceSourceReviewState()` calling `/api/v1/workspaces/{workspace_id}/sources/review-state`.
  - Source row test renders a needs-review/reviewed/unreviewed indicator from source data.
  - Bulk action test selects multiple sources, clicks mark reviewed, and applies returned rows.

- [ ] **Step 2: Run focused frontend tests and verify failures**
  - Run: `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts --maxWorkers=1 --no-file-parallelism`
  - Expected: FAIL because review-state types, filters, client method, and UI actions do not exist.

- [ ] **Step 3: Add TS contract and mapping**
  - Add `WorkspaceSourceReviewState = "unset" | "needs_review" | "reviewed"` to `apps/packages/ui/src/types/workspace.ts`.
  - Add `reviewState`, `reviewStateUpdatedAt`, `reviewedAt`, and `reviewedByUserId` to `WorkspaceSource`.
  - Add snake_case review fields to `WorkspaceSourceApiResponse`, `WorkspaceSourceStatusApiResponse`, and `WorkspaceContextSource`.
  - Map server fields in `apps/packages/ui/src/store/workspace-api.ts`.
  - Add batch request/response client method in `workspace-api.ts`.

- [ ] **Step 4: Add filters and presets**
  - Add `reviewStateFilters: WorkspaceSourceReviewState[]` to `SourceListViewState` and `DEFAULT_SOURCE_LIST_VIEW_STATE`.
  - Update `hasActiveSourceFilters()`, `filterSources()`, and `buildSourceFilterSummary()`.
  - Add review-state filter controls to `SourceAdvancedControls.tsx`.
  - Add helper presets in source pane code for:
    - Needs Review: `reviewStateFilters: ["needs_review"]`
    - Unreviewed: `reviewStateFilters: ["unset"]`
  - Keep `statusFilters` reserved for `processing`, `ready`, and `error`.

- [ ] **Step 5: Add row/detail/bulk actions**
  - Show a compact review badge near the existing source status/readiness affordances.
  - Add single-source actions for `Mark reviewed` and `Needs review`; use the new batch endpoint with a one-item `source_ids` list unless local state is also extended to carry the server `version` needed by `PUT /sources/{source_id}`.
  - Add bulk actions for selected sources using the new batch endpoint.
  - After successful writes, merge returned source rows into local state and preserve current selection.
  - Disable or show precise error state when the API client method is unavailable.

- [ ] **Step 6: Run focused frontend tests and verify pass**
  - Run: `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts --maxWorkers=1 --no-file-parallelism`
  - Expected: PASS.

## Stage 4: Opt-In Needs-Review Defaults for Workspace Attachments

**Goal:** Let workspace attachment paths intentionally create new sources as `needs_review` without changing default behavior.

**Success Criteria:** New workspace sources remain `unset` unless the attach path explicitly opts in; existing workspace attachment paths in this slice (Research Workspace Add Source/server reconciliation and sidepanel WebClipper) can request `needs_review`; the current media-only Quick Ingest context-menu path is recorded as a gap for the later unified ingest/storage-policy task.

**Files:**
- Modify: `tldw_Server_API/app/core/WebClipper/schemas.py`
- Modify: `tldw_Server_API/app/core/WebClipper/service.py`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts`
- Modify: `apps/packages/ui/src/services/web-clipper/types.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Inspect only: `apps/packages/ui/src/entries/background.ts`
- Test: `tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
- Test: `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`
- Test: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

- [ ] **Step 1: Identify active workspace attachment payloads**
  - Confirm whether Add Source, Quick Ingest, and `apps/packages/ui/src/entries/background.ts` context-menu ingest currently create `workspace_sources` directly or only ingest media.
  - Expected current split: WebClipper promotion and workspace reconciliation create `workspace_sources`; background context-menu Quick Ingest is media-only and must be recorded as not applicable for this task.
  - Do not invent a workspace attach flow for media-only Quick Ingest in this task. Record that named gap in Backlog for the later unified ingest/storage-policy task.

- [ ] **Step 2: Write failing opt-in tests for existing attach paths**
  - In `test_web_clipper_service.py`, assert `WebClipperSaveRequest.WorkspacePayload(workspace_id="ws-1", default_review_state="needs_review")` is accepted and promoted workspace sources get `review_state: "needs_review"`.
  - In `test_web_clipper_service.py`, assert `default_review_state="reviewed"` or any value outside `needs_review` is rejected; browser capture may ask for triage, not claim review completion.
  - In `test_web_clipper_api.py`, POST `/api/v1/web-clipper/save` with `workspace.default_review_state: "needs_review"` and assert the created workspace source row is `needs_review`.
  - For WebClipper service promotion, create one workspace source with no opt-in and assert `unset`.
  - Create one workspace source with opt-in review default and assert `needs_review`.
  - In `workspace-server-reconcile.test.ts`, assert the outgoing create request includes `review_state: "needs_review"` only when the local source has `reviewState: "needs_review"`.
  - In `AddSourceModal.stage2.intake.test.tsx`, assert the review-default control marks newly added local sources as `needs_review` only when selected.
  - In `WebClipperPanel.save-flow.test.tsx`, assert the sidepanel WebClipper payload includes `workspace: { workspace_id, default_review_state: "needs_review" }` only when the user enables the review opt-in for a Workspace/Both save.

- [ ] **Step 3: Implement minimal opt-in propagation**
  - Add `default_review_state: Literal["needs_review"] | None = None` to `WebClipperSaveRequest.WorkspacePayload` in `tldw_Server_API/app/core/WebClipper/schemas.py`.
  - Add `default_review_state?: "needs_review" | null` to `WebClipperWorkspacePayload` in `apps/packages/ui/src/services/web-clipper/types.ts`.
  - Add a sidepanel WebClipper control that is only relevant when `destination_mode` is `workspace` or `both`, and have `WebClipperPanel.tsx` include `default_review_state: "needs_review"` only when that control is enabled.
  - In `WebClipperService._ensure_workspace_source()`, map `request.workspace.default_review_state == "needs_review"` to `source_data["review_state"] = "needs_review"`.
  - Pass `review_state: "needs_review"` into `db.add_workspace_source()` only when the existing request/preset explicitly asks for review.
  - Do not make extension or Quick Ingest global defaults change all ingested media.
  - Do not allow WebClipper to submit `reviewed`; review completion must still come from authenticated workspace review actions.
  - Do not add saved per-workspace default settings in this task; that belongs with later source view/default workflow work unless the setting already exists.

- [ ] **Step 4: Run opt-in tests and verify pass**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py -q`
  - Run: `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx --maxWorkers=1 --no-file-parallelism`
  - Expected: PASS.

## Stage 5: Verification, Review, and Backlog Finalization

**Goal:** Prove the persisted review lifecycle works across backend, API contracts, and WebUI behavior without introducing security regressions.

**Success Criteria:** Focused tests pass, Bandit runs on touched backend paths, manual/browser smoke confirms reload persistence and filters, and Backlog records verification.

**Files:**
- Modify: `backlog/tasks/task-12093.1 - Implement-persisted-source-review-lifecycle.md`

- [ ] **Step 1: Run focused backend tests**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspace_context_api.py tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py -q`
  - Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**
  - Run: `bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts --maxWorkers=1 --no-file-parallelism`
  - Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend scope**
  - Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/Workspaces/status_projection.py tldw_Server_API/app/core/WebClipper/schemas.py tldw_Server_API/app/core/WebClipper/service.py -f json -o /tmp/bandit_task_12093_1.json`
  - Expected: PASS or only pre-existing findings outside changed lines; fix new findings before completion.

- [ ] **Step 4: Manual WebUI smoke**
  - Start backend and WebUI with the repo-standard commands.
  - Open a Research Workspace with at least two sources.
  - Mark one source reviewed, reload the workspace, and confirm it remains reviewed.
  - Mark it back to needs-review and confirm reviewed actor/timestamp are not displayed.
  - Apply Needs Review and Unreviewed filters and confirm they do not affect processing status filters.
  - Confirm source readiness/queryability badges still reflect ingestion/indexing state.

- [ ] **Step 5: Update Backlog and commit**
  - Record touched files, test commands/results, Bandit output path, manual smoke notes, and any deferred attach-path gaps in `TASK-12093.1`.
  - Commit code and Backlog updates together with a message that references `TASK-12093.1`.
