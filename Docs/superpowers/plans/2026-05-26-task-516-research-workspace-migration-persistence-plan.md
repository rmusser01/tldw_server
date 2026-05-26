# TASK-516 Research Workspace Migration Persistence Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make successful Research Workspace true-move migration deletion durable so covered legacy local content is not re-persisted after the client sends `client-delete-ack`.

**Architecture:** Keep deletion gating in the migration driver, but make the workspace persistence layer aware of migration tombstones before it writes split localStorage payloads. A verified tombstone should suppress future `tldw-workspace` and `tldw-workspace:workspace:*` persistence for that workspace while retaining blocked/recovery behavior when no tombstone exists.

**Tech Stack:** React/WebUI Research Workspace, Zustand workspace persistence, Vitest, live Playwright/CDP validation against FastAPI.

---

### Task 1: Reproduce durable-delete failure

**Files:**
- Test: `apps/packages/ui/src/store/__tests__/workspace.test.ts`
- Read: `apps/packages/ui/src/store/workspace.ts`
- Read: `apps/packages/ui/src/store/workspace-migration.ts`

- [x] **Step 1: Write the failing test**

Add a regression that seeds a valid legacy workspace with snapshot and chat session, writes the migration tombstone for that workspace, then triggers workspace persistence. Assert that `tldw-workspace`, the split snapshot key, and the split chat key are not written.

- [x] **Step 2: Run the focused test to verify it fails**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/store/__tests__/workspace.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: the new test fails because the persistence layer still writes covered local content keys after a tombstone exists.

### Task 2: Suppress legacy persistence after verified tombstone

**Files:**
- Modify: `apps/packages/ui/src/store/workspace.ts`
- Possibly modify: `apps/packages/ui/src/store/workspace-migration.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace.test.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace-migration.test.ts`

- [x] **Step 1: Implement the minimal tombstone-aware guard**

Add a small helper near the workspace storage helpers that detects a verified Research Workspace migration tombstone for a workspace id. In `writeSplitWorkspacePersistence`, skip writing the monolithic index, split snapshot, and split chat entries for tombstoned workspace ids, and remove any existing covered keys.

- [x] **Step 2: Preserve blocked and ineligible paths**

Do not suppress persistence when there is no tombstone. Existing recovery paths must keep their local data until both local inventory and server eligibility allow deletion.

- [x] **Step 3: Run focused tests**

Run:

```bash
bunx vitest run src/store/__tests__/workspace.test.ts src/store/__tests__/workspace-migration.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: all focused workspace persistence and migration tests pass.

### Task 3: Live validation and closeout

**Files:**
- Modify: `backlog/tasks/task-516 - Prevent-Research-Workspace-migration-from-re-persisting-deleted-local-content.md`

- [x] **Step 1: Run live CDP eligible true-move validation**

Use current-checkout backend and WebUI. Seed a valid legacy workspace with snapshot and chat. Expected: migration declares all chunks, finalizes with `client_delete_eligible=true`, posts `client-delete-ack`, writes a tombstone, and leaves no covered content localStorage keys after page activity.

Result: live Playwright validation against `http://127.0.0.1:3001/research-workspace` and backend `http://127.0.0.1:18001` posted create/chunk/finalize/get/client-delete-ack, wrote a `contentRetained:false` tombstone, showed `Legacy workspace data moved`, and left no `tldw-workspace` or matching `tldw-workspace:workspace:*` keys after an additional idle wait. The running UI normalized the seeded split chat key away before migration, so live receipts covered the main index plus snapshot. The regression test covers the chat key deletion/re-persist path directly.

- [x] **Step 2: Run live CDP blocked-retention validation**

Seed valid legacy content plus an unknown workspace-prefixed key. Expected: backend may save/finalize receipts, but local deletion is blocked, content keys remain, and no `client-delete-ack` is sent.

Result: live Playwright validation posted create/chunk/finalize/get, showed `Local data retained`, retained `tldw-workspace` plus the workspace snapshot key after an additional idle wait, retained the unknown workspace-prefixed key, wrote no tombstone, and sent no `client-delete-ack`.

- [x] **Step 3: Record verification**

Update TASK-516 with the test commands, live endpoint evidence, Bandit applicability, and any residual known risks.
