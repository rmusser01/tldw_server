---
id: TASK-478.26
title: Close Research Workspace UAT matrix and split fixture-backed gaps
status: Done
labels:
- research-workspace
- uat
- workspace-model
- tracking
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 26
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the current Research Workspace UAT remediation stream by reconciling the live UAT matrix and parent TASK-478 notes after TASK-478.25, then split remaining Partial rows into explicit fixture-backed follow-up tasks. Scope is tracking, documentation, and task hygiene unless verification discovers a small reference/metadata bug. Do not implement MCP/ACP/Sandbox/vector runtime behavior in this closeout slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Research Workspace UAT matrix and TASK-478 parent notes are reconciled after TASK-478.25.
- [x] #2 Remaining Partial/Watch rows are split into explicit follow-up Backlog tasks with clear owners and live-evidence requirements.
- [x] #3 TASK-478.24 reference is verified or corrected so the matrix does not point at a missing task.
- [x] #4 Parent TASK-478 status is intentionally left open or closed with a documented rationale.
- [x] #5 Verification and non-applicability of backend/security checks are recorded for this docs/task-only slice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Plan: `Docs/superpowers/plans/2026-05-27-research-workspace-uat-closeout-splitter-plan.md`
- Verified `TASK-478.24` exists under `backlog/completed`, so the RW-UAT-023 matrix reference is valid.
- Created follow-up tasks:
  - `TASK-478.27` for MCP workspace-set policy/tool execution.
  - `TASK-478.28` for ACP workspace-scoped run history and diagnostics.
  - `TASK-478.29` for Sandbox enabled-runtime workspace run diagnostics.
  - `TASK-478.30` for long-running vector indexing completion with real embeddings.
  - `TASK-478.31` for frontend TypeScript baseline verification blockers.
- Updated `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` so Partial/Watch follow-ups point at the new tasks and migration recovery no longer appears as unvalidated.
- Updated parent `TASK-478` notes and intentionally left it In Progress because fixture-backed execution follow-ups remain open.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.26 reconciled the Research Workspace UAT matrix after TASK-478.25 and split the remaining risks into explicit follow-up tasks instead of overclaiming Pass. The MCP, ACP, Sandbox, vector-indexing, and TypeScript verification gaps now each have their own Backlog task with live-evidence-focused acceptance criteria. Parent TASK-478 remains In Progress because these fixture-backed follow-ups are still open.

Verification:
- `git diff --check` passed.
- `task_list(parentTaskId=TASK-478)` shows TASK-478.26 through TASK-478.31 with the intended statuses.
- Stale migration import/export Partial wording was removed from the UAT matrix and parent task notes.
- No product code or backend Python changed, so focused app tests and Bandit are not applicable for this docs/task-only slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
