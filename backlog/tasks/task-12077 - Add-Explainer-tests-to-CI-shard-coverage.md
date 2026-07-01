---
id: TASK-12077
title: Add Explainer tests to CI shard coverage
status: Done
labels:
- ci
- tests
priority: High
modified_files:
- .github/workflows/ci.yml
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
- backlog/tasks/task-12077 - Add-Explainer-tests-to-CI-shard-coverage.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assign the existing tldw_Server_API/tests/Explainer test files to full-suite CI shard coverage so the PR shard coverage guard stops failing on newly unsharded tests already present on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added tldw_Server_API/tests/Explainer to each full-suite chatbooks-streaming shard matrix and updated the CI workflow-contract test so the existing Explainer tests are included in shard coverage. Verification: local shard guard first reproduced the CI failure with four unsharded Explainer files; after the change, Helper_Scripts/ci/check_shard_coverage.py reported new_uncovered=0. The workflow-contract pytest file passed 38/38. Bandit on the touched pytest contract file reported only existing low-severity B101 assert_used findings; rerunning Bandit with B101 excluded passed with no other findings. PR: https://github.com/rmusser01/tldw_server/pull/2561.
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
