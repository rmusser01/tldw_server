# Research Workspace Legacy Storage Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-destructive inventory and schema-mapping gate for Research Workspace legacy local storage before any true migration or deletion work.

**Architecture:** Create a focused frontend inventory module that classifies known localStorage keys and IndexedDB stores as content, metadata, UI-only, derived, obsolete, or unsupported. Add a deletion-eligibility evaluator that fails closed when content-bearing or unknown workspace-prefixed storage is not covered by a migration manifest. Document the mapping so future migration API work cannot delete unmapped local payloads.

**Tech Stack:** TypeScript, Vitest, existing workspace persistence constants and Research Workspace storage patterns.

---

### Task 1: Inventory Contract And Tests

**Files:**
- Create: `apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`
- Create: `apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`

- [x] **Step 1: Write failing tests for known storage classification**
  - Verify `tldw-workspace`, split snapshot keys, split chat keys, IndexedDB chat store, and IndexedDB artifact store classify as content-bearing migration surfaces.
  - Verify Research Workspace UI-only keys classify as retained local preferences.
  - Verify old one-time flags and legacy telemetry are non-content and non-authoritative.

- [x] **Step 2: Run inventory tests and confirm RED**
  - Run: `cd apps/packages/ui && bunx vitest run src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
  - Expected: FAIL because the inventory module does not exist yet.

- [x] **Step 3: Implement inventory module**
  - Export `RESEARCH_WORKSPACE_LEGACY_STORAGE_INVENTORY`.
  - Export `classifyResearchWorkspaceLegacyStorageSurface`.
  - Include exact localStorage keys, split-key patterns, IndexedDB database/store identifiers, and deletion policies.
  - Do not delete or mutate storage.

- [x] **Step 4: Run inventory tests and confirm GREEN**
  - Run: `cd apps/packages/ui && bunx vitest run src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
  - Expected: PASS.

### Task 2: Deletion Eligibility Gate

**Files:**
- Modify: `apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`
- Modify: `apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`

- [x] **Step 1: Write failing tests for fail-closed deletion eligibility**
  - Verify unknown `tldw-workspace`-prefixed localStorage keys block deletion.
  - Verify content-bearing localStorage or IndexedDB surfaces block deletion until covered by a manifest.
  - Verify UI-only surfaces do not block content deletion but are reported as retained.

- [x] **Step 2: Run tests and confirm RED**
  - Run: `cd apps/packages/ui && bunx vitest run src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
  - Expected: FAIL because the evaluator does not exist yet.

- [x] **Step 3: Implement deletion eligibility evaluator**
  - Export `evaluateResearchWorkspaceLegacyDeletionEligibility`.
  - Accept discovered localStorage keys, discovered IndexedDB stores, and manifest-covered surface IDs.
  - Return `eligible`, `blockingSurfaces`, `coveredContentSurfaces`, `retainedLocalSurfaces`, and `unknownSurfaces`.
  - Fail closed for content, unsupported, and unknown workspace-prefixed surfaces.

- [x] **Step 4: Run tests and confirm GREEN**
  - Run: `cd apps/packages/ui && bunx vitest run src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
  - Expected: PASS.

### Task 3: Documentation And Verification

**Files:**
- Create: `Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md`
- Modify: `backlog/tasks/task-468 - Add-Research-Workspace-legacy-storage-inventory-and-migration-safety-gate.md`

- [x] **Step 1: Write inventory documentation**
  - Document every known storage surface, classification, server destination, deletion eligibility rule, and unknown-surface behavior.
  - Explicitly state no local content deletion is implemented in this slice.

- [x] **Step 2: Run focused tests**
  - Run: `cd apps/packages/ui && bunx vitest run src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
  - Expected: PASS.

- [x] **Step 3: Run hygiene checks**
  - Run: `git diff --check -- Docs/superpowers/plans/2026-05-23-research-workspace-legacy-storage-inventory-plan.md Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md "backlog/tasks/task-468 - Add-Research-Workspace-legacy-storage-inventory-and-migration-safety-gate.md"`
  - Run `git diff --no-index --check /dev/null` for new untracked TypeScript files if needed.
  - Expected: no whitespace diagnostics.

- [x] **Step 4: Record Backlog closeout**
  - Update `TASK-468` with modified files, verification commands, Bandit skip reason, and final summary.
