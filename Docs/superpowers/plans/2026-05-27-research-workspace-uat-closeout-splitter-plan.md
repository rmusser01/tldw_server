# Research Workspace UAT Closeout Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the Research Workspace live UAT matrix after TASK-478.25 and split remaining fixture-backed gaps into explicit follow-up tasks.

**Architecture:** This is a tracking and documentation slice. It does not change product behavior; it updates Backlog tasks and the UAT matrix so future implementation work can proceed independently.

**Tech Stack:** Backlog.md MCP, Markdown task files, Research Workspace UAT matrix.

---

### Task 1: Confirm Current Evidence State

**Files:**
- Read: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Read: `backlog/tasks/task-478 - Research-Workspace-UAT-remediation-workstream.md`
- Read: `backlog/tasks/task-478.26 - Close-Research-Workspace-UAT-matrix-and-split-fixture-backed-gaps.md`

- [x] Verify all active TASK-478 child repair tasks through TASK-478.25 are done.
- [x] Verify `TASK-478.24` exists in completed storage, so matrix references remain valid.
- [x] Identify remaining Partial/Watch risks that need fixture-backed follow-up instead of more shell UI changes.

### Task 2: Create Follow-Up Tasks

**Files:**
- Create: `backlog/tasks/task-478.27 - Validate-MCP-workspace-set-policy-and-tool-execution-for-Research-Workspace.md`
- Create: `backlog/tasks/task-478.28 - Validate-ACP-workspace-scoped-run-history-and-diagnostics-for-Research-Workspace.md`
- Create: `backlog/tasks/task-478.29 - Validate-Sandbox-enabled-runtime-workspace-run-diagnostics-for-Research-Workspace.md`
- Create: `backlog/tasks/task-478.30 - Validate-long-running-Research-Workspace-vector-indexing-completion-with-real-embeddings.md`
- Create: `backlog/tasks/task-478.31 - Resolve-frontend-TypeScript-baseline-blockers-for-Research-Workspace-UAT-gate.md`

- [x] Create one task per independent remaining gap.
- [x] Keep ownership clear: MCP owns tools/policy, ACP owns agent runs, Sandbox owns runs/diagnostics, source status owns vector completion, verification owns TypeScript baseline.
- [x] Add acceptance criteria that require live backend/WebUI/CDP evidence where behavior claims are made.

### Task 3: Update Matrix And Parent Notes

**Files:**
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify: `backlog/tasks/task-478 - Research-Workspace-UAT-remediation-workstream.md`

- [x] Link each remaining Partial/Watch row to its follow-up task.
- [x] Remove stale wording that says migration import/export recovery is still unvalidated.
- [x] Keep TASK-478 parent In Progress because fixture-backed validation follow-ups remain open.

### Task 4: Finalize Closeout Task

**Files:**
- Modify: `backlog/tasks/task-478.26 - Close-Research-Workspace-UAT-matrix-and-split-fixture-backed-gaps.md`

- [x] Record created follow-up tasks.
- [x] Document that this slice is docs/task-only, with no Bandit requirement.
- [x] Run repository hygiene checks and commit the tracking change.
