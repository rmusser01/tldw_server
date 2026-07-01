---
id: TASK-12084
title: Fix Explainer tests shard coverage assignment
status: Done
assignee: []
created_date: '2026-07-01 02:40'
updated_date: '2026-07-01 02:41'
labels:
  - ci
  - explainer
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2554 CI Shard coverage guard reports tldw_Server_API/tests/Explainer test files are not assigned to any full-suite shard after rebasing on latest dev. Add the Explainer test directory to the CI shard matrix rather than ignoring the tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explainer test files are covered by at least one ci.yml shard path.
- [x] #2 Helper_Scripts/ci/check_shard_coverage.py passes locally.
- [x] #3 The PR's Shard coverage guard can pass after push.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a dedicated explainer-core shard to each duplicated full-suite matrix in .github/workflows/ci.yml so tldw_Server_API/tests/Explainer is collected instead of silently skipped. Verified with /usr/bin/python3 Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml: PASS, new_uncovered=0. Bandit skipped because only CI YAML and Backlog task metadata changed.
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
