# Research Workspace Import Chat and Batch Remove Implementation Plan

> For agentic workers: use the current task workflow, TDD, and verification-before-completion before marking this plan complete.

**Goal:** Close TASK-12020.32 and TASK-12020.33 by preserving imported chat sessions on first render and ensuring selected-source batch removal applies immediately with Undo recovery.

**Architecture:** Keep the fixes inside the existing Research Workspace chat pane and sources pane contracts. Add focused regression coverage before production changes so the live UAT defects cannot recur without failing tests.

**Tech Stack:** React, Zustand, Vitest, Testing Library, Ant Design.

---

## Stage 1: Chat Import Preservation

**Goal:** Prevent ChatPane from overwriting an imported workspace chat session with an empty local message-option state during first render.

**Success Criteria:** A workspace with a saved chat session loads that session on mount and does not first persist an empty session for the same workspace key.

**Tests:** `ChatPane.stage1.test.tsx` includes a red/green regression for imported chat session mount.

**Status:** Complete

- [x] Add a failing ChatPane test that seeds `workspaceSessions` for the active workspace, renders `ChatPane` with empty message-option state, and asserts the imported messages are loaded without an earlier empty `saveWorkspaceChatSession` call.
- [x] Update ChatPane session synchronization so the first persistence for a workspace occurs only after the component has reconciled the active workspace session.
- [x] Re-run the focused ChatPane test.

**Result:** RED reproduced the empty autosave overwrite; GREEN passed after guarding autosave until `workspaceSessionRef.current` matches the active workspace session key.

## Stage 2: Selected-Source Batch Remove

**Goal:** Ensure `Remove (n)` immediately removes the effective selected sources and keeps Undo able to restore sources plus folder memberships.

**Success Criteria:** Clicking through the batch remove confirmation calls `removeSources` with the effective selected IDs, shows Undo feedback, and the scheduled undo restores sources to their prior positions and folders.

**Tests:** `SourcesPane.stage3.folders.test.tsx` includes a red/green click-through regression for selected-source batch remove.

**Status:** Complete

- [x] Add a failing SourcesPane test that clicks `Remove (n)`, confirms the Ant Design popconfirm, and asserts `removeSources(["s1", "s2"])` is called for an effective selection.
- [x] If the current implementation already passes, keep the regression and update TASK-12020.33 notes to classify the live defect as stale-bundle/environment evidence; otherwise apply the minimal handler fix.
- [x] Add or extend undo assertions so restore preserves original source order, direct selection, and folder memberships.
- [x] Re-run the focused SourcesPane test.

**Result:** The new click-through regression passed immediately against current source, confirming `scheduleWorkspaceUndoAction` already applies `removeSources`. The regression now protects the previously observed live path.

## Stage 3: Documentation, Verification, and Task Closure

**Goal:** Record the fixes and verification evidence without overstating final UAT certification.

**Success Criteria:** Plan, Backlog tasks, and UAT matrix identify the fixed behaviors and any remaining live-browser recheck requirement.

**Tests:** Focused Vitest files pass; scoped `git diff --check` passes; Bandit skipped/documented for frontend-only changes.

**Status:** Complete

- [x] Update this plan statuses as stages complete.
- [x] Update the live UAT matrix row RW-UAT-030 and task records with evidence.
- [x] Run focused tests and scoped diff checks.
- [x] Mark TASK-12020.32 and TASK-12020.33 complete only if their acceptance criteria are satisfied.

**Verification:** Focused Vitest passed for `ChatPane.stage1.test.tsx`, `SourcesPane.stage3.folders.test.tsx`, and the existing workspace import/export store regression. Live browser recheck remains environment-blocked because Chromium headless fails with macOS `MachPortRendezvousServer` permission denial, direct Chrome-for-Testing aborts in crashpad startup, and the stale 8081 Next process cannot be killed from this shell.
