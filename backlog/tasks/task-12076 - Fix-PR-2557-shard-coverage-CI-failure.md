---
id: TASK-12076
title: Fix PR 2557 shard coverage CI failure
status: Done
assignee: []
created_date: '2026-06-30 14:47'
updated_date: '2026-06-30 14:49'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2557'
  - >-
    https://github.com/rmusser01/tldw_server/actions/runs/28427945019/job/84281586224
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assign the new Explainer pytest files to the PR #2557 CI full-suite shard matrix so the shard coverage guard passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shard coverage guard passes locally.
- [x] #2 Only the relevant CI shard mapping is changed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the Shard coverage guard failed in PR #2557 because tldw_Server_API/tests/Explainer/{test_explainer_chatbook_export.py,test_explainer_endpoints.py,test_explainer_jobs.py,test_explainer_repository.py} were not covered by any full-suite shard path. Added a product-explainer shard to each repeated full-suite matrix copy in .github/workflows/ci.yml so the Explainer test directory is explicitly collected instead of ignored or baselined. Bandit skip: touched files are workflow YAML and Backlog task markdown only; no production Python changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added product-explainer shard entries for tldw_Server_API/tests/Explainer in all five CI full-suite matrix copies. Validation: Helper_Scripts/ci/check_shard_coverage.py passed with new_uncovered=0; tldw_Server_API/tests/CI/test_required_workflow_contracts.py passed (38 passed); tldw_Server_API/tests/Explainer passed (46 passed); pre-commit on touched files passed; git diff --check passed. actionlint was unavailable locally.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Validation recorded in the task.
<!-- DOD:END -->
