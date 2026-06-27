# Research Workspace Destructive Recovery Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify remaining Research Workspace destructive and recovery workflows that were not already covered by source bulk remove/undo or share-link revoke.

**Architecture:** This is a UAT certification slice. Product code changes are only allowed after a focused failing test reproduces a confirmed product defect. Most work records browser evidence in the live UAT matrix and Backlog.

**Tech Stack:** React, Ant Design modal/dropdown flows, Zustand Research Workspace store, in-app browser CDP/Playwright surface, Vitest, Backlog.md.

---

## Stage 1: Destructive Action Inventory

**Goal:** Identify every visible destructive or potentially destructive Research Workspace action not already certified by TASK-12020.22 or TASK-12020.23.

**Success Criteria:** The inventory names action family, UI entry point, confirmation behavior, recovery affordance, existing automated coverage, and whether live certification is feasible in the current browser.

**Tests:** Read-only source/test inspection.

**Status:** Complete

**Files:**
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/QuickNotesSection.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactExport.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/TransferSourcesModal.tsx`
- Inspect: focused tests under `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/`

- [x] **Step 1: Read destructive handlers and confirmation copy.**

  Expected: Workspace archive/delete/restore, collection delete, banner reset, chat clear/message delete, note clear, artifact delete, and source transfer undo behavior are mapped.

- [x] **Step 2: Read focused tests for those handlers.**

  Expected: Existing automated coverage is listed and gaps are identified before live browser testing.

## Stage 2: Live Browser Certification

**Goal:** Exercise feasible destructive/recovery workflows in the current in-app browser without risking user data outside disposable Research Workspace state.

**Success Criteria:** At least one success, one cancel, and one recovery path are captured for each feasible destructive action family, or the blocker is documented explicitly.

**Tests:** Browser observations, screenshots, console/network logs.

**Status:** Complete with follow-up defects

**Files:**
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`

- [x] **Step 1: Prepare disposable browser state.**

  Expected: Create or switch to disposable workspaces/sources/artifacts/notes where possible before destructive actions.

- [ ] **Step 2: Certify workspace-level actions.**

  Expected: Archive cancel/success/undo, delete confirmation gating/cancel/success/undo, and restore archived workspace behavior are recorded when visible.

  Progress: Archive cancel was live-CDP-confirmed with screenshot
  `/private/tmp/task12020_27_archive_cancel_dialog.png`. Archive success on
  disposable duplicated workspaces switched back to `New Research`, showed
  `Workspace archived.`, and rendered no visible `Undo`; evidence screenshot
  `/private/tmp/task12020_27_archive_success_missing_undo.png`. The missing
  recovery control was split to TASK-12020.31. TASK-12020.31 now has rendered
  UI regressions and source fixes for the ignored message action field. A clean
  temporary webpack server on `127.0.0.1:8083` loaded the current bundle, and the
  in-app browser confirmed archive success rendered a visible `Undo` button and
  restored the archived duplicate after activation. Evidence:
  `/private/tmp/task12020_31_archive_undo_visible.png` and
  `/private/tmp/task12020_31_archive_undo_restored.png`.

- [x] **Step 3: Certify chat/note actions.**

  Expected: Clear chat and clear note confirmation/cancel/success/undo behavior are recorded when enabled.

  Result: A clean temporary webpack server on `127.0.0.1:8083` and
  shell-launched standalone Chromium validated the remaining chat and note
  recovery paths. Chat clear rendered `Chat cleared.` with visible `Undo` and
  restored the seeded assistant answer after `Undo`; evidence:
  `/private/tmp/task12020_27_chat_clear_undo_visible.png` and
  `/private/tmp/task12020_27_chat_clear_restored.png`. Message delete used the
  visible message overflow menu, confirmed `Delete this message?`, rendered
  `Message deleted.` with visible `Undo`, and restored the assistant message;
  evidence: `/private/tmp/task12020_27_message_delete_undo_visible.png` and
  `/private/tmp/task12020_27_message_delete_restored.png`. Quick Notes clear
  rendered `Note cleared.` with visible `Undo`, emptied the note store, and
  restored the title/content after `Undo`; evidence:
  `/private/tmp/task12020_27_note_preloaded_visible.png`,
  `/private/tmp/task12020_27_note_clear_undo_visible.png`, and
  `/private/tmp/task12020_27_note_clear_restored.png`.

  Defect split: importing a valid `.workspace.json` bundle with chat content
  restores sources but wipes the imported chat session on first render. Evidence:
  `/private/tmp/task12020_27_probe_after_import.png`. Follow-up:
  TASK-12020.32.

- [ ] **Step 4: Certify artifact actions.**

  Expected: Delete artifact confirmation/cancel/success/undo behavior is recorded when artifacts are present or generated/seeded.

  Progress: Artifact delete cancel was live-CDP-confirmed with screenshot
  `/private/tmp/task12020_27_artifact_delete_cancel_dialog.png`. Artifact delete
  success removed the failed output and showed `Output deleted.`, but rendered no
  visible `Undo`; evidence screenshot
  `/private/tmp/task12020_27_artifact_delete_missing_undo.png`. The shared
  recovery-control defect was split to TASK-12020.31. TASK-12020.31 now covers
  the failed-artifact delete toast with a rendered-content regression requiring
  an accessible `Undo` button. After local network permission was restored, a
  shell-launched standalone Chromium imported an attached disposable
  `.workspace.json` bundle containing one failed artifact, deleted the failed
  output, verified `Output deleted.` with a visible `Undo`, clicked `Undo`, and
  verified `Output restored` with the `Failed output` card restored. Evidence:
  `/private/tmp/task12020_31_failed_artifact_imported.png`,
  `/private/tmp/task12020_31_failed_artifact_deleted_undo_visible.png`, and
  `/private/tmp/task12020_31_failed_artifact_restored.png`.

- [x] **Step 5: Certify source organization actions not already covered.**

  Expected: Collection/folder/source-transfer destructive or recovery paths are recorded, excluding source bulk remove/undo already covered by TASK-12020.23.

  Result: Source organization recovery was live-confirmed on the clean current
  bundle. Per-source remove clicked `remove-source-task27-source-a`, rendered
  `Task 27 disposable source A removed. You can undo this for a few seconds.`,
  removed the source from the store, then `Undo` restored it with `Source
  restored`; evidence:
  `/private/tmp/task12020_27_single_source_remove_undo_visible.png` and
  `/private/tmp/task12020_27_single_source_remove_restored.png`. Source
  transfer moved the selected source into an existing destination workspace,
  rendered `Sources transferred. You can undo for a few seconds.`, removed it
  from origin and added it to the destination snapshot, then `Undo` restored the
  origin and removed it from the destination snapshot; evidence:
  `/private/tmp/task12020_27_source_transfer_undo_visible.png` and
  `/private/tmp/task12020_27_source_transfer_restored.png`.

  Defect split: selected-source batch `Remove (1)` remained enabled but inert in
  a clean seeded workspace: clicking it produced no state change, no dialog
  completion, and no toast. Source inspection found the batch handler schedules
  an undo action without applying removal immediately. Evidence:
  `/private/tmp/task12020_27_source_probe_before.png` and
  `/private/tmp/task12020_27_source_probe_after.png`. Follow-up:
  TASK-12020.33.

## Stage 3: Defect Handling

**Goal:** Split or fix any product defect that blocks safe task completion.

**Success Criteria:** Product defects are either fixed with a failing test first or split into a Backlog child task with exact evidence.

**Tests:** Focused Vitest tests for any touched behavior.

**Status:** Complete

**Files:**
- Possible tests: relevant `ResearchWorkspace/__tests__/*.test.tsx`
- Possible implementation: corresponding Research Workspace component or store slice

- [x] **Step 1: Write a failing test for any product-side defect selected for immediate fix.**

  Expected: The test fails for the observed unsafe/destructive behavior.

  Result: No immediate product fix was selected for TASK-12020.27. The blocking
  visible-Undo defect was split to TASK-12020.31 for TDD remediation. The later
  live UAT defects were split to TASK-12020.32 and TASK-12020.33.

- [x] **Step 2: Implement the smallest safe fix or split the defect.**

  Expected: No unrelated refactors. Split tasks include severity, steps, expected, actual, evidence, and recommendation.

- [x] **Step 3: Re-run focused tests.**

  Expected: Focused tests pass or the product defect remains explicitly tracked.

  Result: Product defects remain explicitly tracked in TASK-12020.31,
  TASK-12020.32, and TASK-12020.33. No production code was changed under
  TASK-12020.27.

## Stage 4: Documentation, Verification, and Backlog Closeout

**Goal:** Record the certification result without overclaiming coverage.

**Success Criteria:** UAT matrix, TASK-12020.11, and TASK-12020.27 describe covered paths, blocked paths, verification, screenshots, and remaining risks.

**Tests:** Focused UI/store tests, `git diff --check`, and Bandit only for touched Python.

**Status:** Complete

**Files:**
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Update via MCP: `TASK-12020.11`
- Update via MCP: `TASK-12020.27`

- [x] **Step 1: Update the live UAT matrix.**

  Expected: Destructive/recovery row evidence is grouped by action family and names screenshots/logs.

- [x] **Step 2: Run focused verification.**

  Expected: Relevant focused tests pass, and `git diff --check` passes.

  Result: Scoped whitespace verification passed:
  `git diff --check -- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md Docs/Plans/IMPLEMENTATION_PLAN_research_workspace_destructive_recovery_TASK_12020_27.md`.

- [x] **Step 3: Record Bandit status.**

  Expected: Bandit is run for touched Python code or explicitly skipped for frontend/docs-only scope.

  Result: Bandit skipped for this continuation because only documentation and
  Backlog records were changed; no Python code was touched for TASK-12020.27.

- [x] **Step 4: Finalize Backlog records.**

  Expected: TASK-12020.27 includes inventory, evidence, verification, known blockers, checked criteria, and final summary. TASK-12020.11 is updated when the final matrix state changes.
