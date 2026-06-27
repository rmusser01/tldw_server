# Research Workspace Import Real-File Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify the Research Workspace import flow with a real supported bundle file.

**Architecture:** This is primarily a UAT certification slice. The import/export implementation lives in the Research Workspace header and workspace store, while the acceptance evidence belongs in the live UAT matrix and Backlog task notes. Product code changes are only allowed after a failing focused test reproduces a real defect found during certification.

**Tech Stack:** React, Vitest, Research Workspace Zustand store, in-app browser CDP/Playwright surface, Backlog.md.

---

## Stage 1: Import/Export Surface Inventory

**Goal:** Confirm the exact UI, store, and tests that define accepted bundle formats and import feedback.

**Success Criteria:** The import path, file input selector, accepted file extensions, store import function, and existing rejected-file coverage are identified before browser testing.

**Tests:** Read-only inspection only.

**Status:** Complete

**Files:**
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
- Inspect: `apps/packages/ui/src/store/workspace-bundle.ts`
- Inspect: `apps/packages/ui/src/store/workspace-slices/workspace-list-slice.ts`
- Inspect: `apps/packages/ui/src/store/__tests__/workspace.test.ts`

- [x] **Step 1: Read the bundle parser and UI import code.**

  Run:
  ```bash
  sed -n '1,260p' apps/packages/ui/src/store/workspace-bundle.ts
  sed -n '1,260p' apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx
  ```

  Expected: Identify supported import file types and the UI element that receives the file.

- [x] **Step 2: Read existing focused tests.**

  Run:
  ```bash
  sed -n '1000,1585p' apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
  sed -n '2480,2685p' apps/packages/ui/src/store/__tests__/workspace.test.ts
  ```

  Expected: Identify existing tests for successful import, rejected/unsafe file behavior, progress/success feedback, and expected workspace/source state.

## Stage 2: Browser Real-File Import Attempt

**Goal:** Exercise the live Research Workspace import control using an actual supported bundle file.

**Success Criteria:** A live browser pass attaches a real supported bundle file and records the observed import feedback and workspace state.

**Tests:** Browser/CDP observation plus console/network capture.

**Status:** Complete

**Files:**
- Use fixture artifact: `/tmp/research-workspace-import-TASK-12020-25.workspace.json`
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`

- [x] **Step 1: Create a supported workspace bundle fixture outside the repo.**

  Run a small local script or direct store helper to produce `/tmp/research-workspace-import-TASK-12020-25.workspace.json` with one imported workspace and at least one source item. The fixture must be disposable and must not be committed.

- [x] **Step 2: Connect to the in-app browser and open the Research Workspace page.**

  Expected: The page loads from the local WebUI with a clean browser/session state or the setup limitation is recorded.

- [x] **Step 3: Attach the fixture to the import input.**

  Expected: Import completes, success/progress feedback is visible, the imported workspace/source or artifact appears, and the current workspace is not corrupted.

  Result: Initially blocked by the in-app browser surface: the input locator had
  no `setInputFiles`, upload, dispatch, or mutable evaluate method, and the
  first standalone Playwright runner failed before page execution with macOS
  Chromium `bootstrap_check_in ... Permission denied (1100)`. After local
  network permission was restored, a shell-launched standalone Chromium reached
  the clean current WebUI on `127.0.0.1:8083`, attached a disposable
  `.workspace.json` bundle through the visible Import Workspace input, and
  rendered the imported `Failed output` artifact. Evidence:
  `/private/tmp/task12020_31_failed_artifact_imported.png`.

- [x] **Step 4: Capture reliability observations.**

  Expected: Console errors, failed network requests, slow interactions, and confusing feedback are recorded.

## Stage 3: Defect Handling

**Goal:** Preserve TDD discipline if certification reveals a product bug.

**Success Criteria:** Any product defect has a failing focused test before code changes, or no product code is changed when the outcome is pass/block evidence only.

**Tests:** Run only the focused failing/passing test set for touched product behavior.

**Status:** Complete

**Files:**
- Possible test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
- Possible test: `apps/packages/ui/src/store/__tests__/workspace.test.ts`
- Possible implementation: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Possible implementation: `apps/packages/ui/src/store/workspace-slices/workspace-list-slice.ts`

- [x] **Step 1: Write a failing test for any confirmed product defect.**

  Run:
  ```bash
  cd apps/packages/ui && npm run test -- src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

  Expected: The new test fails for the observed product behavior, not because of a test setup error.

  Result: No product defect was confirmed. The later shell-launched browser pass
  attached and imported a real file successfully. No production code was changed
  for TASK-12020.25.

- [x] **Step 2: Implement the smallest code change needed to pass the focused test.**

  Expected: The UI/store behavior matches the observed acceptance criterion without broad refactoring.

  Result: Not applicable; no product code change was made.

- [x] **Step 3: Re-run the focused test.**

  Run:
  ```bash
  cd apps/packages/ui && npm run test -- src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism
  ```

  Expected: The focused test passes.

  Result: Covered in Stage 4 focused verification.

## Stage 4: Documentation, Verification, and Backlog Closeout

**Goal:** Update the UAT evidence and task status honestly based on the browser result.

**Success Criteria:** The UAT matrix and Backlog notes state whether import is certified, product-failed, or environment-blocked; verification commands are recorded; Bandit is skipped only if no Python code was touched.

**Tests:** Focused UI/store tests, `git diff --check`, and applicable Bandit or explicit non-Python skip.

**Status:** Complete

**Files:**
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Update via MCP: `TASK-12020.25`
- Update via MCP when relevant: `TASK-12020.11`

- [x] **Step 1: Update the live UAT matrix.**

  Expected: Import/export coverage names exact fixture, browser result, feedback observed, and any rejected-file evidence.

- [x] **Step 2: Run focused verification.**

  Run:
  ```bash
  cd apps/packages/ui && npm run test -- src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx src/store/__tests__/workspace.test.ts --maxWorkers=1 --no-file-parallelism
  git diff --check
  ```

  Expected: Tests pass and whitespace check reports no errors, unless no code changed and existing focused coverage is cited as evidence.

  Result: `npm run test -- src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx src/store/__tests__/workspace.test.ts --maxWorkers=1 --no-file-parallelism` passed with 110 tests. `git diff --check` passed.

- [x] **Step 3: Record Bandit status.**

  Expected: Run Bandit on touched Python code, or record that this slice touched only frontend/docs and Bandit is not applicable.

  Result: No Python files were touched for TASK-12020.25; Bandit is not applicable.

- [x] **Step 4: Finalize Backlog records.**

  Expected: `TASK-12020.25` includes evidence, verification, known blockers/skips, checked acceptance criteria, checked Definition of Done, and final summary. `TASK-12020.11` is updated when the import finding changes the parent UAT evidence.

  Result: `TASK-12020.25` is Done with live attached-file import evidence.
  `TASK-12020.11` and the UAT matrix include the updated import pass note.
  `TASK-12020.30` captured and closed the unrelated header dropdown assertion
  adjustment found during verification.
