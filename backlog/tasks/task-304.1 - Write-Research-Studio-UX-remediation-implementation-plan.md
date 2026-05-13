---
id: TASK-304.1
title: Write Research Studio UX remediation implementation plan
status: Done
assignee:
  - Codex
created_date: '2026-05-12 15:56'
updated_date: '2026-05-12 16:10'
labels:
  - plan
  - research-studio
  - ux
  - webui
  - extension
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for the approved Research Studio UX remediation design. The plan must turn `Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md` into staged, executable work orders that preserve internal workspace-playground compatibility while moving the user-facing product route/name to Research Studio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written under `Docs/superpowers/plans/` with the required writing-plans header and task-by-task checkbox structure.
- [x] #2 Plan maps exact files to create, modify, and test for each stage, including WebUI, extension, shared UI, docs, and CDP verification scope.
- [x] #3 Plan keeps immediate implementation slices independently reviewable and orders degraded-health pass-through before browser-visible route alias verification.
- [x] #4 Plan explicitly preserves internal storage/export/telemetry identifiers and avoids generated docs output unless a docs rebuild is intentionally in scope.
- [x] #5 Plan includes concrete verification commands and expected outcomes for focused Vitest, route tests, CDP/Playwright checks, diff hygiene, and Bandit skip rationale for frontend-only/doc-only tasks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. COMPLETED - inspected approved design spec and current constraints.
2. COMPLETED - wrote implementation plan under `Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md`.
3. COMPLETED - decomposed work into staged tasks: tracking/baseline, degraded health, canonical aliases, mobile tab route state, naming/handoffs, work-product IA, no-source disclosure, returning-user efficiency, capability-aware health, and release verification.
4. COMPLETED - included exact file targets, TDD steps, verification commands, CDP checks, commit boundaries, non-goals, and compatibility boundaries.
5. COMPLETED - ran local plan self-review and `git diff --check` on the plan file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote implementation plan at `Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md`. Local self-review caught and fixed incorrect relative `cd` paths in verification commands. `git diff --check` passes for the plan file.

Definition of Done notes: verification was `git diff --check -- Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md`; documentation artifact is the implementation plan itself; Bandit not run because this task changed planning/task documentation only and no backend Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Research Studio UX remediation implementation plan with staged, implementation-ready tasks, explicit file targets, TDD checkpoints, verification commands, CDP smoke checks, and compatibility boundaries. Verified diff hygiene for the plan file.
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
