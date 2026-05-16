---
id: TASK-394.1
title: Map Quick Ingest active path and legacy tests
status: Done
assignee: []
created_date: '2026-05-16 00:42'
updated_date: '2026-05-16 01:47'
labels:
  - quick-ingest
  - ux
  - audit
  - task-1
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 1: recover and document the active quick-ingest workflow, entry points, services, tests, and legacy/stale surfaces before code changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active path map exists with entry points, modal flow, API services, persistence, extension hooks, and tests
- [x] #2 Legacy/stale tests and surfaces are classified without product-code changes
- [x] #3 Task 1 verification evidence is recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 completed on branch codex/quick-ingest-ux-remediation. Artifact created and refined at Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md. Review checkpoints passed: spec compliance approved after citation fixes; artifact quality approved with no blocking issues. Verification: rg heading checks passed, git diff --check passed, worktree clean after commits. Bandit not run because this slice changed docs/task-tracking only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Quick Ingest active path map and classified active launch paths, legacy QuickIngestModal reachability, stale selector risk, extension sidepanel launch behavior, and e2e helper masking risks. Task 1 commits changed only the active-path artifact; no product code or tests were modified. Verification included rg heading/path checks, git diff --check, clean git status after commits, and two-stage subagent review approval.
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
