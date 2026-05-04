---
id: TASK-20
title: 'Address PR #1243 review comments'
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-03 22:50'
updated_date: '2026-05-03 23:06'
labels:
  - worker-registry
  - pr-review
  - issue-1114
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1243'
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1243 inline review comments are verified against current branch code before changes.
- [x] #2 Startup service group handling cannot start registry-owned workers without lifecycle worker inventory or retained shutdown ownership.
- [x] #3 Focused tests cover the review-fix behavior and pass locally.
- [x] #4 Touched-scope Bandit and git diff --check are run and results recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify each PR review comment against current codex/worker-lifecycle-cleanup-1114 code. 2. Add failing focused tests for any real behavior or test-coverage gaps. 3. Implement minimal code/test/doc changes to satisfy verified comments. 4. Run focused pytest, Bandit on touched Python source, and git diff --check. 5. Commit and push the PR branch, then record verification in this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified PR comments: startup_service_groups still allowed worker_inventory=None while dropping registry-owned worker handles; telemetry guarded-exception test lacked a hook-invocation sentinel; post-worker shutdown helper keys currently match both strict production signatures; Qodo TODO comment targets a phase2-followup plan file that is absent from current HEAD. Red/green: added test_start_service_groups_requires_worker_inventory_before_starting_registry_owned_workers and confirmed it failed before the guard, then passed after start_service_groups now fails fast when worker_inventory is None. Added telemetry sentinel assertion and a post-worker helper/signature contract test. Verification: pytest startup_service_groups + shutdown_telemetry_services + shutdown_post_worker_services -q -> 16 passed, 5 warnings; pytest startup_service_tail -q -> 1 passed, 5 warnings; Bandit on startup_service_groups.py -> 0 findings; git diff --check -> clean. Follow-up CodeRabbit comments after push addressed: wrapped `_base_shutdown_kwargs` in Backlog final summary markdown and converted the post-worker failure stub to async. Verification: pytest test_shutdown_post_worker_services.py -q -> 10 passed, 5 warnings; git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1243 review follow-ups by enforcing lifecycle WorkerRegistry ownership in start_service_groups before registry-owned worker helpers can start, adding regression coverage for the missing worker_inventory path, proving the telemetry guarded-exception hook is invoked, and adding a contract test that `_base_shutdown_kwargs` matches the strict post-worker shutdown signatures. Verified the Qodo TODO comment targets a phase2-followup plan file that is not present in current HEAD, so no doc edit was needed for that stale comment.
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
