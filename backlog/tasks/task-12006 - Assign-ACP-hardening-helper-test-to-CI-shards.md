---
id: TASK-12006
title: Track ACP hardening helper test in CI coverage baseline
status: Done
created_date: 2026-06-24 03:49
labels:
- ci
- tests
priority: medium
updated_date: 2026-06-24 03:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the PR shard coverage guard failure after rebasing by adding `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_helpers.py` to the shard coverage baseline used for the existing ACP backlog.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shard coverage guard passes locally for ci.yml.
- [x] #2 The ACP hardening helper test is covered by the shard coverage baseline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the ACP hardening helper test path to `Helper_Scripts/ci/shard_coverage_baseline.txt` beside the existing ACP backlog entries.
2. Run the shard coverage guard locally.
3. Record verification and final summary on this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Touched files:
- `Helper_Scripts/ci/shard_coverage_baseline.txt`

Verification:
- `source .venv/bin/activate && python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` -> OK, `new_uncovered=0`.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_helpers.py -q` -> 8 passed.

Known skips/blockers: Bandit not applicable because this task only changes CI shard coverage metadata.
Direct `.github/workflows/ci.yml` assignment was not pushed because GitHub rejected workflow-file updates from the current OAuth token without `workflow` scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_helpers.py` to the shard coverage baseline beside the existing ACP backlog entries, resolving the shard coverage guard failure after rebasing PR #2459 onto the latest `dev`.
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
