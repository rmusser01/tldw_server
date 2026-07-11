# Paperless-Inspired Source Saved Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable source-list presets and versioned, server-backed personal saved views scoped to one user and workspace.

**Architecture:** Extend the existing `SourceListViewState` only for lifecycle filtering, then serialize a narrower V1 filter/sort contract. Persist named views in `CharactersRAGDB` with owner/workspace predicates, optimistic locking, portable normalized-name uniqueness, and PostgreSQL RLS; expose typed workspace CRUD routes; orchestrate the UI through a focused hook and accessible control component.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL, pytest, React 18, TypeScript, Ant Design, Vitest, Testing Library.

**Spec:** `Docs/superpowers/specs/2026-07-10-paperless-source-saved-views-design.md`
**Backlog:** `TASK-12093.2`

---

## Stage 1: Canonical Source View State
**Goal:** Represent every required preset and create a strict V1 client serialization boundary.
**Success Criteria:** Presets, lifecycle filtering, validated full-state serialization/apply, invalid response handling, Modified detection, and visible lifecycle predicates are deterministic and accessible.
**Tests:** `source-list-view.test.ts`, new `source-saved-views.test.ts`, existing SourcesPane filter tests.
**Status:** Complete

## Stage 2: Persistence And Tenant Isolation
**Goal:** Add portable saved-view storage, CRUD, optimistic locking, limits, and PostgreSQL RLS.
**Success Criteria:** SQLite/PostgreSQL schemas and owner/workspace-scoped DB methods satisfy all conflict and invalid-row contracts.
**Tests:** New DB tests, PostgreSQL integration tests, existing RLS policy contract tests.
**Status:** Complete

## Stage 3: Typed Workspace API
**Goal:** Expose validated CRUD routes with recoverable invalid rows and machine-readable conflicts.
**Success Criteria:** API isolation, validation, error codes, reset, and rate-limit contracts pass.
**Tests:** New workspace saved-view API tests and rate-limit contract tests.
**Status:** Complete

## Stage 4: Client Orchestration And Accessible UI
**Goal:** Load, save, apply, replace, reset, and delete saved views without blocking ordinary filters or built-ins.
**Success Criteria:** Reload restore, duplicate confirmation, invalid recovery, nonblocking failures, keyboard operation, and focus behavior pass.
**Tests:** API client, hook, and component/integration Vitest suites.
**Status:** Not Started

## Stage 5: Integration And Release Gates
**Goal:** Register tests in CI, run focused/full verification, security checks, and finalize Backlog evidence.
**Success Criteria:** Relevant tests, shard guard, compile/diff checks, Bandit, and final task review pass.
**Tests:** Aggregated commands listed in Task 6.
**Status:** Not Started

---

### Task 1: Add Lifecycle Filtering And The V1 Client State Contract

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-list-view.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/use-source-list-view-state.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceAdvancedControls.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-saved-views.ts`
- Create: `apps/packages/ui/src/types/workspace-source-saved-view.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-saved-views.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx`

- [x] **Step 0: Capture the pre-implementation static-analysis baselines**

Run `bun run typecheck` from `apps/tldw-frontend` and record the exit code plus diagnostics in `TASK-12093.2` implementation notes. Later typecheck verification must introduce no new diagnostics.

From the repository root, capture Ruff's existing-file scope as normalized JSON before any backend implementation. The clean starting point currently has 27 diagnostics in this scope; the recaptured normalized file must contain 27 entries. Omitting locations lets the final comparison tolerate line movement while still detecting additions by file, rule, and message multiplicity.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check --output-format json --output-file /tmp/task_12093_2_ruff_before_raw.json tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py
jq -S 'map({filename, code, message}) | sort_by(.filename, .code, .message)' /tmp/task_12093_2_ruff_before_raw.json > /tmp/task_12093_2_ruff_before.json
jq 'length' /tmp/task_12093_2_ruff_before.json
```

The Ruff command is expected to exit 1 because this is known whole-file debt; continue only when `jq 'length'` prints `27`, then record that count in the task notes. Do not baseline any new test module: those files must pass Ruff outright.

- [x] **Step 1: Write failing lifecycle-filter tests**

Add tests proving `lifecycleStateFilters: ["partially_queryable"]` matches only sources whose `statusDetails.lifecycleState` is `partially_queryable`, participates in `hasActiveSourceFilters`, appears in `buildSourceFilterSummary`, remains visibly represented with an accessible keyboard-removable chip/summary whether Advanced is collapsed or expanded, and is cleared by a full reset.

- [x] **Step 2: Run the lifecycle tests and verify RED**

Run from `apps/tldw-frontend`:

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because `SourceListViewState` and filtering do not yet support lifecycle states.

- [x] **Step 3: Implement the minimal lifecycle filter**

Add `lifecycleStateFilters: WorkspaceSourceLifecycleState[]` to the state/default, filter against `source.statusDetails?.lifecycleState`, include it in active-filter detection, and add stable labels. In `SourceAdvancedControls`, keep an active lifecycle predicate visibly represented in both disclosure states with a focusable clear action. Add `min={0}` to every numeric range input. Keep processing `statusFilters` and review filters separate.

- [x] **Step 4: Write failing V1 contract and preset tests**

Test:

```typescript
expect(SOURCE_VIEW_PRESETS.partiallyIndexed.state.lifecycleStateFilters)
  .toEqual(["partially_queryable"])
expect(SOURCE_VIEW_PRESETS.largeFiles.state.fileSizeMin)
  .toBe(50 * 1024 * 1024)
const serialized = serializeSourceListViewState({ ...state, expanded: true })
expect(serialized.ok).toBe(true)
if (serialized.ok) expect(serialized.state).not.toHaveProperty("expanded")
expect(deserializeSourceViewState(validV1)).toEqual(expectedState)
expect(deserializeSourceViewState(malformed)).toBeNull()
```

Cover all seven menu entries, full default filling, exact enum arrays, nonnegative finite numeric ranges, minimum-not-greater-than-maximum invariants, real `YYYY-MM-DD` calendar dates with `date_from <= date_to`, canonical equality/signature generation, and preserving `expanded` only when applying a view. Invalid local state must return field-specific validation issues and must not produce a request payload.

The V1 wire fields are all optional on input with these server/client defaults, and all are present in canonical responses/signatures:

| Field | Type | Default |
| --- | --- | --- |
| `type_filters` | `pdf | video | audio | website | document | text` array | `[]` |
| `status_filters` | `processing | ready | error` array | `[]` |
| `review_state_filters` | `unset | needs_review | reviewed` array | `[]` |
| `lifecycle_state_filters` | `queued | ingesting | extracting | chunking | indexing | queryable | partially_queryable | failed | retrying | missing_media | blocked_by_permissions | unknown` array | `[]` |
| `date_field` | `added_at | source_created_at` | `added_at` |
| `date_from`, `date_to` | `YYYY-MM-DD` string or null | `null` |
| `require_url`, `require_file_size`, `require_duration`, `require_page_count` | boolean | `false` |
| `file_size_min`, `file_size_max`, `duration_min`, `duration_max`, `page_count_min`, `page_count_max` | finite nonnegative number or null | `null` |
| `sort` | `manual | name_asc | name_desc | added_desc | added_asc | source_created_desc | source_created_asc | file_size_desc | file_size_asc | duration_desc | duration_asc | page_count_desc | page_count_asc` | `manual` |

Reject unknown enum values/fields. Reject booleans as numeric values. Deduplicate arrays and emit them in the declaration order shown above so equality signatures do not depend on click order.

- [x] **Step 5: Run the V1 contract tests and verify RED**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-saved-views.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the module does not exist.

- [x] **Step 6: Implement the pure saved-view module**

Define the V1 wire type, saved-view sort union, and invalid-reason union once in neutral `@/types/workspace-source-saved-view`; both the pure component helper and `workspace-api.ts` import them. `source-list-view.ts` aliases or imports the neutral sort union so the neutral module never depends on a component module. Create explicit constants and helpers, including:

```typescript
export const SOURCE_SAVED_VIEW_SCHEMA_VERSION = 1
export const LARGE_SOURCE_FILE_BYTES = 50 * 1024 * 1024

export interface WorkspaceSourceSavedViewStateV1 {
  type_filters: WorkspaceSourceType[]
  status_filters: WorkspaceSourceStatus[]
  review_state_filters: WorkspaceSourceReviewState[]
  lifecycle_state_filters: WorkspaceSourceLifecycleState[]
  date_field: "added_at" | "source_created_at"
  date_from: string | null
  date_to: string | null
  require_url: boolean
  require_file_size: boolean
  require_duration: boolean
  require_page_count: boolean
  file_size_min: number | null
  file_size_max: number | null
  duration_min: number | null
  duration_max: number | null
  page_count_min: number | null
  page_count_max: number | null
  sort: WorkspaceSourceSavedViewSort
}

export type SourceViewStateValidationResult =
  | { ok: true; state: WorkspaceSourceSavedViewStateV1 }
  | { ok: false; issues: Array<{ field: string; message: string }> }

export const serializeSourceListViewState = (
  state: SourceListViewState
): SourceViewStateValidationResult => validateThenCanonicalize(state)

export const applySavedSourceViewState = (
  current: SourceListViewState,
  saved: WorkspaceSourceSavedViewStateV1
): SourceListViewState => ({
  ...deserializeAndDefault(saved),
  expanded: current.expanded
})
```

Keep presets immutable and fixed-order. Retain Unreviewed as the preset shipped by Child Task 1. Signature generation consumes only an `ok` canonical state; invalid local state yields no signature, is considered Modified, and produces inline save validation instead of throwing during render.

- [x] **Step 7: Add full-state apply to the page hook**

Add `applySourceListViewState(next)` alongside patch/reset. It must replace every persisted field and preserve only `expanded` when asked by the saved-view helper.

- [x] **Step 8: Run Stage 1 tests and verify GREEN**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-saved-views.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [x] **Step 9: Commit Stage 1**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-list-view.ts apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-saved-views.ts apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceAdvancedControls.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/use-source-list-view-state.ts apps/packages/ui/src/types/workspace-source-saved-view.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-saved-views.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx
git commit -m "Add canonical source saved view state (TASK-12093.2)"
```

### Task 2: Add Saved-View Storage, Conflicts, And PostgreSQL RLS

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_postgres.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`

- [x] **Step 1: Write failing SQLite CRUD/isolation tests**

Cover:

- create/list/get/update/delete;
- deterministic ordering;
- NFKC + case-fold `name_key` uniqueness;
- owner and active-workspace isolation on every method, including `workspaces.client_id = owner_user_id`;
- 100-view and 16 KiB limits;
- expected-version conflicts;
- soft-deleted workspaces immediately becoming inaccessible while retaining rows, plus physical workspace deletion cascading rows;
- create/update/delete serialization against workspace soft deletion;
- raw corrupt, unsupported-version, and valid-but-invalid-state rows remaining retrievable for API recovery.
- opening a real V52 SQLite database migrates to V53 and creates the additive table without rebuilding unrelated workspace tables.

Use two owner IDs against one test database to prove isolation independently of per-user file paths. User B must be unable to probe, list, create, update, or delete views for user A's workspace even when B knows the workspace ID.

- [x] **Step 2: Run DB tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py -q
```

Expected: FAIL because the table and methods do not exist.

- [x] **Step 3: Add portable schema and conflict types**

Add constants for limits and stable codes plus a focused conflict subclass carrying `code` and safe metadata. For this RED/GREEN cycle, implement only the SQLite half: bump `_CURRENT_SCHEMA_VERSION` from 52 to 53, add `_migrate_from_v52_to_v53` to `_sqlite_linear_migration_steps`, and have it call a dedicated idempotent `_ensure_workspace_source_saved_view_schema_sqlite` helper. The normal SQLite post-migration ensure path calls it too. Add the table to fresh SQLite initialization with a named unique constraint:

```sql
PRIMARY KEY (workspace_id, id),
CONSTRAINT uq_workspace_source_saved_views_owner_name
  UNIQUE (owner_user_id, workspace_id, name_key),
FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
```

Use UUID text IDs, ISO UTC text, and integer versions consistently across backends.

- [x] **Step 4: Implement scoped DB methods**

Implement normalization and CRUD methods that always accept `owner_user_id` and `workspace_id`. Every operation first verifies an active workspace whose `client_id` matches `owner_user_id`; every mutation performs that check while locking/serializing the workspace row in the same transaction. Enforce the state limit as `len(state_json.encode("utf-8")) <= 16 * 1024`. Use bound parameters and preflight duplicates for useful metadata. Name the unique constraint and translate only that constraint/SQLSTATE: after the failed transaction has rolled back, look up the conflicting owned row in a fresh transaction to populate `view_id` and `version`. Never log `state_json`.

For this step, SQLite uses the existing immediate write transaction so workspace validation, count, duplicate preflight, and every mutation occur in the same transaction. Soft deletion retains saved-view rows but makes them inaccessible; only a physical workspace delete exercises the foreign-key cascade. Add the PostgreSQL row-lock branch only after its failing tests in Step 7.

- [x] **Step 5: Run SQLite tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [x] **Step 6: Write failing PostgreSQL schema/RLS tests**

Add integration coverage for the table, named unique key, optimistic update, ordering, active workspace-owner predicates, concurrent count-limit serialization, concurrent create duplicates, two-row rename-to-same-name races, and create/update/delete races with workspace soft deletion. Add a two-principal PostgreSQL RLS test following the existing source-review RLS pattern: set `app.current_user_id` for principal A and prove principal B cannot select, create, update, delete, or directly insert a view for A's workspace. Assert fresh creation and V52 migration install the policy immediately, before any separate startup ensure, by checking `relrowsecurity`, `relforcerowsecurity`, `qual`, and `with_check`. Also execute `ensure_chacha_rls()` against PostgreSQL while the saved-view table is absent and require a successful no-op, proving the `to_regclass(...)` guard is executable rather than only textually present. Extend the RLS contract test to require an owner-and-active-workspace predicate equivalent to:

```sql
ALTER TABLE IF EXISTS workspace_source_saved_views ENABLE ROW LEVEL SECURITY
ALTER TABLE IF EXISTS workspace_source_saved_views FORCE ROW LEVEL SECURITY
USING (
  owner_user_id = current_setting('app.current_user_id', true)
  AND EXISTS (
    SELECT 1 FROM workspaces w
    WHERE w.id = workspace_source_saved_views.workspace_id
      AND w.client_id = current_setting('app.current_user_id', true)
      AND w.deleted = false
  )
)
WITH CHECK (
  owner_user_id = current_setting('app.current_user_id', true)
  AND EXISTS (same active-workspace subquery)
)
```

- [x] **Step 7: Run PostgreSQL/RLS tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_postgres.py -q
```

Expected: RLS assertion FAIL; PostgreSQL test may SKIP locally only when the standard fixture is unavailable.

- [x] **Step 8: Implement PostgreSQL migration, locking, and RLS**

Add `_ensure_workspace_source_saved_view_schema_postgres`, the `current_version < 53` migration branch, and the normal post-migration ensure call. Every mutation first locks `SELECT id FROM workspaces WHERE id = ? AND client_id = ? AND deleted = false FOR UPDATE`; absence maps to the focused not-found path. This serializes creates, renames, deletes, and workspace deletion for the same workspace. Apply idempotent enable/force/drop/create RLS statements with both owner-and-active-workspace `USING` and `WITH CHECK` inside this schema helper's transaction immediately after table creation. Keep `build_chacha_rls_sql()` as the general startup path, but guard the saved-view policy block with `to_regclass(...)` so a missing table is a successful no-op. Do not create a separate RLS initializer.

- [x] **Step 9: Re-run PostgreSQL/RLS tests and verify GREEN/SKIP**

Expected: contract PASS and integration PASS when PostgreSQL is available, otherwise the fixture-controlled SKIP.

- [x] **Step 10: Commit Stage 2**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_postgres.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git commit -m "Persist workspace source saved views (TASK-12093.2)"
```

### Task 3: Add Typed Workspace Saved-View Routes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_source_saved_views_api.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py`
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

- [x] **Step 1: Write failing schema and API tests**

Cover strict state/create/patch validation (`extra="forbid"`), create/list/patch/delete, active workspace ownership, owner/workspace isolation, workspace 404, duplicate-name metadata including concurrent-race recovery, count-limit code, version conflict metadata, deterministic list order, invalid JSON, invalid V1 state, unsupported schema version, reset via PATCH, and route rate-limit categories. Exact `422` cases include boolean/zero/negative versions, boolean numeric fields, explicit null PATCH operations, invalid/inverted dates and ranges, and unknown top-level fields.

Assert conflict shape exactly:

```python
assert response.status_code == 409
assert response.json()["detail"] == {
    "code": "source_view_name_exists",
    "view_id": existing_id,
    "version": 1,
}
```

Pin the wire contract and PATCH invariants in tests:

- `GET /source-views` -> `200 {"items": WorkspaceSourceSavedViewResponse[]}`.
- `POST /source-views` body -> `{"name": str, "schema_version": 1, "state": WorkspaceSourceSavedViewStateV1}`; success -> `201` plus one response object.
- `PATCH /source-views/{view_id}` body -> `{"version": int, "name"?: str, "schema_version"?: 1, "state"?: WorkspaceSourceSavedViewStateV1}`; success -> `200` plus one response object. At least one of `name` or `state` is required. If `state` is present, `schema_version` is required and must be `1`. `schema_version` is forbidden when `state` is absent. A reset always sends both state fields atomically.
- `DELETE /source-views/{view_id}` -> `204` with no body. Delete is intentionally unconditional after owner/workspace checks; a repeated delete returns `404` rather than a version conflict.

Every response object is exactly `id`, `workspace_id`, `name`, `schema_version`, `state`, `valid`, `invalid_reason`, `version`, `created_at`, and `updated_at`. A valid row has canonical/full non-null state and null `invalid_reason`; an invalid row has null state and one stable non-null reason. Unsupported schema version is determined before JSON parsing and therefore takes precedence over malformed JSON.

All `409` details are exact:

```json
{"code":"source_view_name_exists","view_id":"...","version":2}
{"code":"source_view_limit_reached","limit":100}
{"code":"source_view_version_conflict","view_id":"...","current_version":3}
```

- [x] **Step 2: Run API tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_saved_views_api.py tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py -q
```

Expected: FAIL because schemas and routes do not exist.

- [x] **Step 3: Add strict Pydantic contracts**

Define V1 state, create, patch, response, and list models matching the exact wire contract above. Set `extra="forbid"` on state, create, and patch. V1 input fields use the default matrix from Task 1 and emit a complete canonical model. Canonicalize enum arrays server-side by deduplicating and declaration-order sorting. Require a strict positive integer optimistic `version`; schema/numeric validators reject booleans while allowing ordinary JSON integer/float range values, and require finite nonnegative values with ordered ranges. Create accepts only a strictly validated schema version 1. Patch requires `version` plus at least one of `name` or `state`; a model validator rejects version-only requests, explicit null operations, `schema_version` without state, and anything except schema version 1 whenever state is present. Response-model validation enforces the valid/state/reason invariants; use stable invalid reasons `invalid_json`, `invalid_state`, and `unsupported_schema_version`.

- [x] **Step 4: Add safe row-to-response conversion**

Check stored schema version before parsing JSON. Parse known-version JSON without exposing it in logs, validate it through the Pydantic state model, and persist/measure canonical JSON using one deterministic UTF-8 serializer (sorted keys and compact separators); the limit is bytes, not characters. Convert malformed rows into recoverable response objects instead of failing the list. Reset is not a special implicit operation: the client PATCHes the canonical V1 default state together with `schema_version = 1`, and the DB updates both columns and increments `version` atomically.

- [x] **Step 5: Add CRUD endpoints and explicit conflict mapping**

Add GET/POST/PATCH/DELETE under `/{workspace_id}/source-views`. Use a focused saved-view workspace guard that requires `db.get_workspace(workspace_id)` to be active and its `client_id` to equal `str(current_user.id)`; do not broaden or silently change the generic `_require_workspace` contract in this task. Pass the same owner ID to every DB operation and special-case the focused saved-view conflict/not-found exceptions into structured responses. Use read/write/delete rate-limit dependencies consistently.

- [x] **Step 6: Re-run API tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [x] **Step 7: Generate and verify the API fingerprint**

Run from `apps/tldw-frontend`:

```bash
PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python bun run generate:api-types
```

Review and stage only the committed `lib/api/openapi.fingerprint.json`; generated OpenAPI/TypeScript artifacts remain gitignored. Then run from the repository root:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected: fingerprint generation succeeds and the drift check exits 0.

- [x] **Step 8: Commit Stage 3**

```bash
git add tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/tests/Workspaces/test_workspace_source_saved_views_api.py tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "Expose workspace source saved view API (TASK-12093.2)"
```

### Task 4: Add API Client And Saved-View Orchestration Hook

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/use-source-saved-views.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx`

- [ ] **Step 1: Write failing API-client request tests**

Test encoded workspace/view paths, the `{"items": [...]}` list envelope, exact POST/PATCH bodies and statuses, and bodyless DELETE for source views. Ensure slash-delimited or blank view IDs fail before requests.

- [ ] **Step 2: Run API-client tests and verify RED**

```bash
bun run test:run -- ../packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because methods/types are absent.

- [ ] **Step 3: Add API response/request types and methods**

Import the exact V1 snake_case state and invalid-reason types from `@/types/workspace-source-saved-view`; do not redeclare them in the service layer. Model structured conflict details and valid/invalid response envelopes. Add list/create/update/delete methods to `workspaceApiMethods` using `workspacePath` and `encodeWorkspacePathSegment`.

- [ ] **Step 4: Write failing orchestration-hook tests**

Test null workspace availability (no request), non-null to null synchronously clearing rows/active/conflict/limit/error/mutation state, stale completions after nulling, workspace-scoped loading, retry, apply valid view, disable invalid apply, create success, local serialization validation blocking requests with field issues, `source_view_name_exists` confirmation data, `source_view_limit_reached` non-retryable limit state, replacement, version-conflict refresh, reset to V1 defaults, delete active view, workspace switch, deferred list and mutation responses after a switch, A to B to A stale-response rejection, and built-ins remaining usable when list fails.

Mock the real `bgRequest` error shape, not a simplified code-only error:

```typescript
const conflict = {
  status: 409,
  details: {
    detail: { code: "source_view_name_exists", view_id: "view-1", version: 2 }
  }
}
```

The hook's conflict parser reads `error.details?.detail`, validates the code-specific fields, and treats malformed details as an ordinary retryable request error. A valid `source_view_limit_reached` detail is a non-retryable state containing the server limit and deletion guidance.

- [ ] **Step 5: Run hook tests and verify RED**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the hook is absent.

- [ ] **Step 6: Implement minimal hook state machine**

The hook accepts raw `workspaceId: string | null`, current view state, and `onApplyState`. It exposes `available = workspaceId !== null` and performs no request while unavailable. Every identity change first increments a monotonic generation and synchronously clears rows, active snapshot, conflicts, limit state, errors, announcements, and mutation state. Every async operation captures the generation and may commit only when it still matches; comparing workspace IDs alone is insufficient because A to B to A can reuse an ID. Abort requests where supported as an additional optimization. The hook otherwise owns only server view loading/mutations, active-view snapshot/signature, Modified detection, local serialization issues, duplicate/replace state, saved-view-limit state, success announcements, exact nested `error.details.detail` conflict extraction, and retryable errors. It is a single page-level controller: do not instantiate it inside `SourcesPane` and do not move source filters into a second store.

- [ ] **Step 7: Re-run client/hook tests and verify GREEN**

Run both commands above. Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add apps/packages/ui/src/services/tldw/domains/workspace-api.ts apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/use-source-saved-views.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx
git commit -m "Add source saved view client orchestration (TASK-12093.2)"
```

### Task 5: Build Accessible Source View Controls And Integrate Them

**Files:**
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceViewControls.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourceViewControls.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx`

- [ ] **Step 1: Write failing presentational accessibility tests**

Cover grouped built-in/saved items, fixed ordering, Save accessible name, keyboard menu navigation, Enter/Space activation, Escape close, modal focus trap/return, Modified label, retry error, invalid warning, disabled invalid Apply, Reset/Delete, field-specific local-state save validation, replacement confirmation, and non-retryable saved-view-limit guidance. When unavailable, Save/manage actions are disabled, expose an accessible "Select a workspace" explanation, and do not open a dialog; built-ins remain enabled. Opening captures the controller generation; a generation change closes the overlay and discards its draft, and submit refuses a stale generation. Focus returns to the invoker only while it remains connected/focusable, otherwise to the visible saved-view trigger or Sources pane landmark. Request errors use `role="alert"`/an appropriate live region, mutation controls expose an accessible busy/disabled state, and create/replace/reset/delete success uses one polite status region.

- [ ] **Step 2: Run control tests and verify RED**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourceViewControls.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement the focused control component**

Use existing Ant Design menu/modal primitives and Lucide icons. Keep the row compact and unframed near search/advanced controls. Split repeated trigger/menu controls from a single `SourceViewOverlayHost` exported by the same module; only the page-level host renders modal portals. It stores the invoking element and captured controller generation, resets all overlay-local state on generation change, validates generation again before submit, and uses the documented focus fallback. Every icon-only control needs both a tooltip and `aria-label`; do not add nested cards or explanatory feature copy.

- [ ] **Step 4: Write failing integration tests**

Test that Research Workspace creates exactly one saved-view controller beside `useSourceListViewState`, passes the same controller to simultaneous desktop/drawer SourcesPane mounts, and renders exactly one page-level overlay host. Invoking Save or confirmation from either pane yields one dialog/alert portal and closing restores focus to that pane's actual invoking control. Open Save and Replace in workspace A, switch to B and separately to null, and verify the dialog closes synchronously, drafts/confirmation state are discarded, and stale submission cannot call either controller generation. Unmount the invoking pane before close and verify focus falls back to the visible trigger or Sources landmark. Verify a null active workspace causes no request, old rows disappear synchronously, Save neither opens a dialog nor issues a request, and the ID is never converted to the synthetic pane fallback `"local"`. Test applying presets before filtering, saving current state, preserving `expanded`, and leaving search/folder/selection state untouched. After remount/reload, verify the server-backed view is listed and the user can reselect it to restore filters/sort. Do not persist or automatically restore an active/default view. Test workspace changes reload the correct list, clear active selection, and ignore deferred responses after non-null to null and A to B to A generation changes.

- [ ] **Step 5: Run integration tests and verify RED**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL until props/hook wiring exists.

- [ ] **Step 6: Wire controls into Research Workspace**

Expose `applySourceListViewState` from the page hook. Instantiate `useSourceSavedViews` exactly once in `ResearchWorkspace`, beside the source-list state hook, with the raw nullable active-store workspace ID. Pass one controller model and full-state apply callback to every SourcesPane instance; panes render portal-free `SourceViewControls` above `SourceAdvancedControls` but never create their own controller or issue requests for the synthetic `"local"` fallback. Render one `SourceViewOverlayHost` at page level and route dialog opens through it with the invoking element and current controller generation; generation changes synchronously reset the host. Keep server-view errors local and nonblocking.

- [ ] **Step 7: Re-run Stage 4 UI tests and verify GREEN**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourceViewControls.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceViewControls.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourceViewControls.test.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx
git commit -m "Add source saved view controls (TASK-12093.2)"
```

### Task 6: Register CI Coverage, Verify, And Finalize Tracking

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `backlog/tasks/task-12093.2 - Implement-saved-source-filter-presets-and-views.md`
- Modify during execution: `Docs/superpowers/plans/2026-07-10-paperless-source-saved-views-implementation-plan.md`

- [ ] **Step 1: Prove the shard guard sees the new backend tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml
```

Expected before workflow edit: FAIL listing the new saved-view test modules.

- [ ] **Step 2: Add every new backend test module to all repeated matching CI shard matrices**

Place DB/PostgreSQL tests with ChaCha/workspace content tests and API tests with workspace endpoint tests. Do not use the ignore file.

- [ ] **Step 3: Re-run shard coverage and verify GREEN**

Expected: `new_uncovered=0` and exit 0.

- [ ] **Step 4: Run focused backend verification**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_postgres.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/Workspaces/test_workspace_source_saved_views_api.py tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py -q
```

Expected: PASS, with only standard PostgreSQL fixture skips when unavailable.

- [ ] **Step 5: Run focused frontend verification**

```bash
bun run test:run -- ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-saved-views.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourceViewControls.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx ../packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 6: Run compile, diff, and security gates**

Re-run the OpenAPI drift check from the repository root:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected: the drift check exits 0.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check --output-format json --output-file /tmp/task_12093_2_ruff_after_raw.json tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py
jq -S 'map({filename, code, message}) | sort_by(.filename, .code, .message)' /tmp/task_12093_2_ruff_after_raw.json > /tmp/task_12093_2_ruff_after.json
diff -u /tmp/task_12093_2_ruff_before.json /tmp/task_12093_2_ruff_after.json
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_postgres.py tldw_Server_API/tests/Workspaces/test_workspace_source_saved_views_api.py
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py -f json -o /tmp/bandit_task_12093_2.json
```

The existing-file Ruff command is expected to exit 1 while the 27 baseline findings remain. Expected: the normalized Ruff diff is empty, all three new test modules pass Ruff outright, compile and diff checks exit 0, and Bandit JSON contains zero results. Record the unchanged baseline debt rather than modifying unrelated code. Any dynamic SQL suppression must include an explicit rationale.

Run frontend static gates from `apps/tldw-frontend`:

```bash
bunx eslint ../packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-list-view.ts ../packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-saved-views.ts ../packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/use-source-saved-views.ts ../packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceAdvancedControls.tsx ../packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceViewControls.tsx ../packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx ../packages/ui/src/components/Option/ResearchWorkspace/use-source-list-view-state.ts ../packages/ui/src/components/Option/ResearchWorkspace/index.tsx ../packages/ui/src/types/workspace-source-saved-view.ts ../packages/ui/src/services/tldw/domains/workspace-api.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-list-view.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-saved-views.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourceViewControls.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage4.filters-and-sort.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx ../packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
bun run typecheck
```

Expected: touched-file ESLint passes. Typecheck introduces no diagnostics beyond the pre-implementation baseline captured before Task 1; if the repository baseline is nonzero, compare and record the exact unchanged failures.

- [ ] **Step 7: Update plan stages and Backlog task through official Backlog.md tooling**

Check acceptance criteria and Definition of Done only after evidence exists. Record modified files, test counts, PostgreSQL skips, Bandit output, residual risks, and the final summary.

- [ ] **Step 8: Request code review and address findings**

Use `superpowers:requesting-code-review`. Re-run affected tests after fixes and resolve every actionable finding before completion.

- [ ] **Step 9: Apply the repository plan-file ownership rule**

Workers and later sessions must not delete a plan they did not create. After every stage is Complete and evidence is recorded, only the orchestrating agent that created this file may delete it with `apply_patch` and remove its active Backlog documentation link. Any other executor leaves it as linked evidence and reports that ownership prevented cleanup.

- [ ] **Step 10: Commit final CI/tracking updates**

```bash
git add -A .github/workflows/ci.yml "backlog/tasks/task-12093.2 - Implement-saved-source-filter-presets-and-views.md" Docs/superpowers/plans/2026-07-10-paperless-source-saved-views-implementation-plan.md
git commit -m "Finalize source saved views task (TASK-12093.2)"
```
