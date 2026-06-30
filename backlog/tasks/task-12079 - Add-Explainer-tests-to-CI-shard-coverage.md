---
id: TASK-12079
title: Add Explainer tests to CI shard coverage
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-30 20:11'
labels:
  - ci
  - tests
  - explainer
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assign the existing Explainer pytest files reported by the shard coverage guard to a CI shard so the pull request can pass coverage validation after rebasing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explainer pytest files reported by the guard are covered by the CI shard map.
- [x] #2 The local shard coverage guard passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the CI shard matrix and choose the nearest backend shard for Explainer tests. - Complete
2. Add the missing Explainer test paths to the shard map. - Complete
3. Run the shard coverage guard locally and record the result. - Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Assigned tldw_Server_API/tests/Explainer to the existing product-flashcards shard block in each repeated full-suite matrix block. The local shard coverage guard now reports new_uncovered=0. Bandit is not applicable because this task only changes CI YAML and task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added `tldw_Server_API/tests/Explainer` to the repeated `product-flashcards` shard blocks in `.github/workflows/ci.yml`, keeping the shard count stable while assigning the existing Explainer tests to the full-suite matrices. Verified with `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`, which now reports `new_uncovered=0`.
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
