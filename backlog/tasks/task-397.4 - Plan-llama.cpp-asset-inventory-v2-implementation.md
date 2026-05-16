---
id: TASK-397.4
title: Plan llama.cpp asset inventory v2 implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 06:10'
labels:
  - llamacpp
  - planning
  - backend
  - webui
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-managed-runtime-stage1-implementation-plan.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a follow-up implementation plan for the next llama.cpp managed runtime slice after Stage 1: local asset inventory v2, folder/file import/register workflows, stale-path state, and mmproj discovery/pairing boundaries while keeping remote downloads deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan created at Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md.
- [x] #2 Plan scopes Asset Inventory V2 to local file/folder registration, stale-path state, and mmproj candidate pairing while explicitly deferring remote downloads.
- [x] #3 Plan preserves legacy /api/v1/llamacpp/inventory and /api/v1/llamacpp/models/register-path compatibility.
- [x] #4 Plan includes TDD steps, exact touched files, verification commands, and known non-code security validation skip.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started from merged Stage 1 plan and the managed runtime roadmap. This task is plan-only; implementation will be a follow-up code task.

Verification: git diff --check passed with no output. ASCII scan over the plan and task files found no non-ASCII characters. Bandit skipped because this task only changes planning/task markdown and no Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Asset Inventory V2 implementation plan at Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md. The plan scopes the next llama.cpp slice to local asset discovery, file registration, folder import registration, stale-path warnings, mmproj candidate pairing, legacy inventory compatibility, and a minimal Admin assets panel. It explicitly defers remote downloads/catalogs, model-family routing, full Admin Console V2, and automatic profile mutation.
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
