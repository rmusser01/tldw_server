---
id: TASK-12958
title: Build authoritative service prompt inventory and rollout matrix
status: Done
assignee: []
created_date: '2026-07-14 00:15'
updated_date: '2026-07-14 00:26'
labels:
  - prompts
  - inventory
  - planning
  - backend
  - webui
  - browser-extension
dependencies:
  - TASK-12956
documentation:
  - Docs/Design/service-prompt-inventory.md
  - Helper_Scripts/validate_service_prompt_inventory.mjs
  - Helper_Scripts/tests/validate_service_prompt_inventory.test.mjs
  - Docs/superpowers/plans/2026-07-12-service-prompts-01-inventory.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inventory contains the approved eligibility rubric, exact 16-column matrix, and missing-row release blocker.
- [x] #2 All 232 candidates are classified as 73 eligible, 75 deferred, and 84 excluded, with stable IDs only for eligible definitions.
- [x] #3 All 73 eligible contracts have exact call-site, ownership, assembly, precedence, and rollout-domain evidence.
- [x] #4 The durable validator and negative regression tests pass, and the matrix has explicit human approval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-12-service-prompts-01-inventory.md and the five rollout-domain plans; runtime implementation remains in child tasks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recreated with a collision-free current-dev task ID. Final review found no Critical, Important, or Minor issues. Full pytest and frontend shards were intentionally skipped for this planning-only work at requester direction; Bandit is not applicable because no Python changed.

Fresh current-dev verification: Node regression tests 5/5; validator and test syntax pass; inventory validator reports 232 rows, 73 eligible IDs/contracts, exact 73/75/84 decisions, exact six protected Jobs IDs, 636 source spans, 880 line components, exact 16/32/2/21/2 rollout coverage, zero unresolved references, and no errors; all imported task references use the current collision-free IDs; git diff --check passes.

Draft PR: https://github.com/rmusser01/tldw_server/pull/2726. The PR remains draft until the requester supplies the required human-written Change summary.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the human-approved 232-row Service Prompt inventory, 73 exact eligible contracts, five rollout-domain plans, a durable validator, and five negative validator regression tests.
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
