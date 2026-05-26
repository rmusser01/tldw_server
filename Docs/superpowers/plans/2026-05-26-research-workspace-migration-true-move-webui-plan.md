# Research Workspace Migration True Move WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `/research-workspace` client migration driver that safely moves legacy local Research Workspace data into the server migration protocol without deleting local content unless server eligibility and local inventory gates both pass.

**Architecture:** Keep migration orchestration out of the main page component. Add typed migration protocol API methods, a pure manifest/chunk builder over the existing legacy inventory, and a driver state machine that can be invoked from `/research-workspace`. The first implementation must be deletion-safe against the current backend, which finalizes sessions but does not yet emit `client_delete_eligible=true`.

**Tech Stack:** TypeScript, React, Zustand workspace store, existing `bgRequest` API client domain, Vitest, Playwright/CDP for live validation.

---

## Design Constraints

- `/research` and `/research-workspace` stay separate.
- Do not add `/workspace-playground` aliases, redirects, routes, or user-facing current labels.
- Ignore the old `workspace_migrated` flag as authoritative migration proof.
- Unknown workspace-prefixed localStorage keys or unknown `tldw-workspace-storage` IndexedDB stores block local deletion.
- Local content deletion requires both:
  - local inventory eligibility from `evaluateResearchWorkspaceLegacyDeletionEligibility`, and
  - server `client_delete_eligible === true` from the migration session response.
- With the current backend, the expected safe result after finalize is `finalized` plus `client_delete_eligible=false`; the UI must show recoverable blocked state, not delete content.
- Do not reintroduce persistent top-level trust/banner clutter. Migration status belongs in contextual first-run/settings/recovery UI.

## Files

- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`
- Replace/refactor: `apps/packages/ui/src/store/workspace-migration.ts`
- Modify: `apps/packages/ui/src/store/__tests__/workspace-migration.test.ts`
- Modify: `apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`
- Modify: `apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`
- Modify: `backlog/tasks/task-471 - Wire-Research-Workspace-migration-true-move-WebUI-flow.md`

## Task 1: Add Typed Migration Protocol Client Methods

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Test: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`

- [x] **Step 1: Write failing API client tests**

Add tests proving:

```ts
await workspaceApiMethods.createWorkspaceMigration({
  id: "mig-1",
  idempotency_key: "mig-1:hash",
  target_workspace_id: "ws-1",
  target_workspace_name: "Workspace",
  source_product: "research-workspace-webui",
  manifest_hash: "a".repeat(64),
  declared_chunks: [],
  manifest: {},
  diagnostics: {}
})

await workspaceApiMethods.putWorkspaceMigrationChunk("mig-1", "chunk-1", {
  sha256: "b".repeat(64),
  byte_count: 12,
  chunk_kind: "workspace_bundle",
  metadata: {}
})

await workspaceApiMethods.finalizeWorkspaceMigration("mig-1", {
  manifest_hash: "a".repeat(64)
})

await workspaceApiMethods.getWorkspaceMigration("mig-1")
await workspaceApiMethods.ackWorkspaceMigrationClientDelete("mig-1", {
  acknowledged_manifest_hash: "a".repeat(64)
})
```

Expected request paths:

- `POST /api/v1/workspaces/migrations`
- `PUT /api/v1/workspaces/migrations/mig-1/chunks/chunk-1`
- `POST /api/v1/workspaces/migrations/mig-1/finalize`
- `GET /api/v1/workspaces/migrations/mig-1`
- `POST /api/v1/workspaces/migrations/mig-1/client-delete-ack`

- [x] **Step 2: Run the failing API client tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because migration methods are missing.

- [x] **Step 3: Add request/response types and methods**

Add interfaces mirroring backend schemas:

- `WorkspaceMigrationChunkDeclaration`
- `WorkspaceMigrationCreateRequest`
- `WorkspaceMigrationChunkUploadRequest`
- `WorkspaceMigrationFinalizeRequest`
- `WorkspaceMigrationClientDeleteAckRequest`
- `WorkspaceMigrationChunkReceiptResponse`
- `WorkspaceMigrationResponse`

Add methods to `workspaceApiMethods` using `bgRequest` and `encodeURIComponent` for path parameters.

- [x] **Step 4: Run API client tests again**

Expected: PASS.

## Task 2: Replace Old Flag-Based Migration With a Safe Manifest Builder

**Files:**
- Modify: `apps/packages/ui/src/store/workspace-migration.ts`
- Modify: `apps/packages/ui/src/store/__tests__/workspace-migration.test.ts`
- Modify: `apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`
- Modify: `apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`

- [x] **Step 1: Write failing tests for unsafe legacy behavior**

Assert that the old `workspace_migrated` flag does not skip true migration and that no function writes `workspace_migrated=true` as proof of migration.

- [x] **Step 2: Write failing tests for manifest building**

Use injected discovery/read dependencies so tests do not depend on real browser storage:

```ts
const plan = await buildResearchWorkspaceMigrationPlan({
  targetWorkspaceId: "ws-1",
  targetWorkspaceName: "Workspace One",
  discoveredLocalStorageKeys: [
    "tldw-workspace",
    "tldw-workspace:workspace:ws-1:snapshot",
    "tldw-workspace:workspace:ws-1:chat"
  ],
  discoveredIndexedDbStores: [],
  readLocalStorageValue: async (key) => key === "tldw-workspace" ? "{\"workspaces\":[]}" : "{}"
})
```

Expected plan fields:

- stable `migrationId`
- 64-char `manifestHash`
- declared chunks with byte count and SHA-256
- manifest-covered surface IDs
- local deletion eligibility computed from the inventory
- no local deletion side effects

- [x] **Step 3: Implement pure helpers**

Keep helpers deterministic and side-effect-free except injected reads:

- `discoverResearchWorkspaceLegacyStorage`
- `buildResearchWorkspaceMigrationPlan`
- `sha256Text`
- `byteLengthText`
- `buildResearchWorkspaceMigrationTombstoneKey`
- `buildResearchWorkspaceMigrationTombstone`

Use Web Crypto in browser and test-compatible fallback where existing project patterns allow it.

- [x] **Step 4: Preserve inventory safety**

Extend inventory tests if needed so content-bearing surfaces are deletion-blocking unless covered, UI-only surfaces are retained, and unknown surfaces block.

- [x] **Step 5: Run focused store tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/store/__tests__/workspace-migration.test.ts src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

## Task 3: Add Migration Driver State Machine

**Files:**
- Modify: `apps/packages/ui/src/store/workspace-migration.ts`
- Modify: `apps/packages/ui/src/store/__tests__/workspace-migration.test.ts`

- [x] **Step 1: Write failing orchestration tests**

Cover:

- creates migration session with idempotency key and manifest
- records each chunk receipt
- finalizes after chunks
- fetches recovery state
- does not delete local content when `client_delete_eligible=false`
- writes tombstone and sends delete ack only when server eligibility and local inventory eligibility are both true
- returns recoverable conflict/error states without throwing away local content

- [x] **Step 2: Implement `runResearchWorkspaceMigration`**

Input dependencies:

- migration API methods
- storage discovery/reader
- optional local delete/tombstone writer
- clock/build metadata

Return a structured result:

- `status`: `not_needed | blocked | finalized_not_delete_eligible | deleted | failed`
- `migrationId`
- `serverMigration`
- `blockingSurfaces`
- `unknownSurfaces`
- `deletedSurfaceIds`
- `message`

- [x] **Step 3: Ensure destructive operations are dependency-injected**

The default export must not delete anything unless the caller passes delete dependencies and the server response is eligible. Tests should prove deletes are not called on backend-ineligible finalize.

- [x] **Step 4: Run focused migration tests**

Run the same workspace migration test command as Task 2.

Expected: PASS.

## Task 4: Wire Driver Into `/research-workspace` Contextually

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`

- [x] **Step 1: Write failing UI tests**

Cover:

- detects legacy content on first `/research-workspace` load
- starts migration once per active workspace/idempotency key
- shows compact contextual migration status when migration is blocked or finalized but not deletion-eligible
- does not render a persistent top-level trust bar
- does not mention `/workspace-playground`

- [x] **Step 2: Add a small migration status surface**

Use existing Research Workspace visual patterns. Prefer a compact inline status within first-run/recovery/settings context instead of a page-wide banner.

Required copy:

- `Legacy workspace data found`
- `Server receipt saved`
- `Local data retained until server deletion eligibility is available`
- `Review recovery details`

- [x] **Step 3: Keep normal workspace loading usable**

Migration errors must not block opening the server-backed workspace. They should leave a recoverable status and keep local content untouched.

- [x] **Step 4: Run focused Research Workspace tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx src/store/__tests__/workspace-migration.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

## Task 5: Live Backend + WebUI + CDP Validation

**Files:**
- Update: `backlog/tasks/task-471 - Wire-Research-Workspace-migration-true-move-WebUI-flow.md`

- [ ] **Step 1: Start live backend and WebUI**

Use the configured backend/WebUI process pattern for this worktree. Use CDP/Playwright, not Computer Control.

- [ ] **Step 2: Seed legacy local storage in CDP**

Seed a minimal legacy workspace payload plus a split snapshot/chat key. Include an unknown workspace-prefixed key in one run to verify deletion blocking.

- [ ] **Step 3: Visit `/research-workspace`**

Expected:

- current route loads
- old `/workspace-playground` remains normal 404/no redirect
- migration session is created
- chunk receipts are accepted
- finalize returns recovery manifest
- local content is retained while backend reports `client_delete_eligible=false`
- UI shows contextual retained-local-data state

- [ ] **Step 4: Record known backend gap**

If backend still never emits deletion eligibility, record that true local deletion is blocked by backend protocol support and create/update the next backend task instead of faking success.

## Verification Before Completion

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/store/__tests__/workspace-migration.test.ts src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx --maxWorkers=1 --no-file-parallelism
```

Run:

```bash
git diff --check
```

Run live backend + WebUI + CDP validation before marking TASK-471 done. Bandit is not required unless backend Python changes are made.
