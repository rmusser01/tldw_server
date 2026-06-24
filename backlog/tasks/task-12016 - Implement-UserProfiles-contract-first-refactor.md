---
id: TASK-12016
title: Implement UserProfiles contract-first refactor
status: In Progress
created_date: 2026-06-24 21:44
labels:
- userprofiles
- refactor
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-06-24-userprofiles-contract-first-refactor-design.md
- Docs/superpowers/plans/2026-06-24-userprofiles-contract-first-refactor-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/UserProfiles/planner.py
- tldw_Server_API/tests/UserProfile/test_profile_update_planner.py
- backlog/tasks/task-12016.4 - Add-UserProfiles-update-planner-seam.md
updated_date: 2026-06-24 23:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved UserProfiles contract-first refactor implementation plan with subagent-driven task execution, preserving current v1 behavior first and gating the clean v2 surface until compatibility routes are stable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Plan tasks are implemented in order with test-first red/green evidence for behavior changes.
- [ ] #2 Existing v1 profile routes preserve their current response contracts unless an explicitly tested compatibility adapter changes internals only.
- [ ] #3 Planner, command, query, bulk, v2, and verification milestones meet the plan readiness gates.
- [ ] #4 SQLite and targeted Postgres profile tests pass where required by the plan.
- [ ] #5 Bandit passes on touched UserProfiles/API/service scopes before final completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Commit daf9cbd0b added legacy v1 profile contract characterization tests only. Spec-compliance reviewer approved with no gaps; code-quality reviewer approved with no findings. Controller reran targeted test command on 2026-06-24: 4 passed, 260 warnings.
Task 2 complete after review loop. Commits 144d82312 and 1a7157899 added typed contracts, hardened immutable payload/rejected handling, and strengthened tests. Spec re-review approved with no gaps; code-quality re-review approved with no findings. Controller reran `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/UserProfile/test_profile_contracts.py -q`: 3 passed, 19 warnings. Controller ran Bandit on contracts.py: 0 findings.
Task 3 complete after review loop. Commits f18e0166f and 04a21329a added domain error taxonomy, preserved v1 compatibility buckets, stabilized mixed domain error precedence, and pinned direct-vs-legacy not-found details. Spec re-review approved with no gaps; code-quality re-review approved with no blocking findings. Controller reran mapping + update + legacy characterization suite: 28 passed, 897 warnings. Controller ran Bandit on touched production error-mapping files: exit code 0.
Task 4 complete. Commit fecff0896 added ProfileUpdatePlanner and planner tests, including forced dry-run delegation. Spec review approved with no gaps; code-quality review approved with no blocking findings. Controller reran planner + profile update tests: 20 passed, 742 warnings. Controller ran Bandit on planner.py: exit code 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed.
- [ ] #2 Each delegated implementation milestone receives spec-compliance and code-quality review before being marked complete.
- [ ] #3 Final verification evidence is recorded in the task.
- [ ] #4 User receives summary of changed files, commits, tests, and remaining risks.
<!-- DOD:END -->
