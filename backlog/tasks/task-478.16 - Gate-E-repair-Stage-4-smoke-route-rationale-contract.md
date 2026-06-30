---
id: TASK-478.16
title: 'Gate E: repair Stage 4 smoke route rationale contract'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-26 06:08'
labels: []
milestone: Research Workspace UAT Remediation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2055'
  - >-
    https://github.com/rmusser01/tldw_server/actions/runs/26434667973/job/77814693650
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the UX Smoke Gate failure on PR #2055 where Stage 4 high-risk route governance crashes with `TypeError: Cannot read properties of undefined (reading 'trim')` because routes in `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts` omit required `rationale` values. Scope: provide explicit rationale metadata for active high-risk routes and verify the Stage 4 smoke gate locally where feasible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing Stage 4 route rationale values are reported as governance problems instead of crashing with TypeError.
- [x] #2 Every active Stage 4 high-risk route has an explicit rationale value.
- [x] #3 High-risk /admin/mlx and /settings/image-generation routes have shared route metadata.
- [x] #4 Focused Stage 4 metadata governance test passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Guarded route.rationale in the Stage 4 governance helper so missing values produce actionable governance messages.
- Added explicit rationale values for all active Stage 4 high-risk routes.
- Added route metadata for /admin/mlx and /settings/image-generation, which were present in the high-risk smoke list but missing from shared route metadata.
- Focused verification passed: bunx playwright test e2e/smoke/stage4-axe-high-risk-routes.spec.ts --reporter=line --grep "metadata-aligned".
- Full local Stage 4 using next start could not run without a local production build; a dev-server full run moved past the metadata crash but hit local dev-server restart/network artifacts, so CI remains the authoritative full-gate check.
- Bandit not applicable: only TypeScript route metadata/smoke-test files and Backlog task metadata were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired the Stage 4 high-risk route governance contract for PR #2055: missing rationales no longer crash the helper, high-risk routes now carry rationale copy, and the two missing shared route metadata entries are defined.
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
