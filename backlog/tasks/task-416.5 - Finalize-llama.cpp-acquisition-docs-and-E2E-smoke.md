---
id: TASK-416.5
title: Finalize llama.cpp acquisition docs and E2E smoke
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 14:42'
labels:
  - llamacpp
  - webui
  - docs
  - e2e
  - local-llm
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
parent_task_id: TASK-416
priority: medium
modified_files:
  - Docs/API-related/llamacpp_integration_modes.md
  - Docs/Published/API-related/llamacpp_integration_modes.md
  - Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
  - apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts
  - backlog/tasks/task-416.5 - Finalize-llama.cpp-acquisition-docs-and-E2E-smoke.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs explain local register path vs import folder behavior and the non-mutating import preview.
- [x] #2 Docs explain Jobs-backed remote downloads, validation-to-inventory flow, and the explicit no auto-profile/start/wiring boundary.
- [x] #3 Admin E2E smoke covers import preview, confirmed folder import, queued download, completed download, and asset refresh without real remote downloads.
- [x] #4 Focused backend/frontend tests and diff checks are run or any skips are documented with reasons.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task 5 from the llama.cpp model acquisition/import workflow plan: document the acquisition/import lifecycle, add or extend admin E2E smoke coverage using mocked APIs, run focused backend/frontend verification where available, run Bandit for touched Python scope if Python files are touched, and close out the plan/task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Task 5 closeout from the llama.cpp acquisition/import workflow plan. Updated public and published integration-mode docs with local register vs folder import behavior, non-mutating previews, Jobs-backed remote downloads, validation-to-inventory flow, no automatic profile/start/chat wiring, private-network URL policy, and destination allowlist rules. Extended the tier-4 admin Playwright smoke with mocked import preview, confirmed folder import, queued download, completed download, and asset refresh coverage.

Verification: backend focused pytest passed (102 passed, 5 warnings); frontend focused Vitest passed (33 passed across 3 files); Playwright admin llama.cpp smoke passed after elevated localhost dev-server run (6 passed); git diff --check passed. Bandit skipped because this closeout changed docs and TypeScript E2E only, with no touched Python scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Finalized llama.cpp acquisition/import closeout documentation and mocked admin E2E smoke coverage. The docs now explain the explicit acquisition boundaries and safety policies, while the Playwright smoke proves import and download flows stay mocked, Jobs-backed, and separate from profile creation/runtime launch/chat wiring.
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
