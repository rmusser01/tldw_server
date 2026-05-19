---
id: TASK-21
title: Address startup worker group legacy-handle review comment
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-03 23:10'
updated_date: '2026-05-03 23:12'
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
- [x] #1 start_worker_groups does not start cleanup or compactor/websub workers when worker_inventory is absent.
- [x] #2 Focused regression tests prove the missing-inventory path fails before any startup helper can create legacy tasks.
- [x] #3 Focused pytest coverage for startup worker groups passes after the fix.
- [x] #4 Touched-scope Bandit and git diff --check are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify current start_worker_groups behavior and choose the fail-fast contract because bootstrap always constructs WorkerRegistry. 2. Add a failing regression test for worker_inventory=None that proves no startup helpers are invoked. 3. Implement the minimal RuntimeError guard in start_worker_groups and keep downstream helper signatures unchanged for now. 4. Run focused pytest for startup_worker_groups and related startup contract coverage, then Bandit on the touched service file and git diff --check. 5. Update the Backlog task with verification details.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the review concern against current code: start_worker_groups dropped the return values from start_cleanup_workers and start_compactor_websub_workers while both downstream helpers still expose worker_inventory=None branches that create legacy tasks/stop handles. Chose the fail-fast contract because initialize_startup_worker_bootstrap always constructs WorkerRegistry and the existing startup worker group tests only cover inventory-backed startup. Red/green: added test_start_worker_groups_requires_worker_inventory_before_starting_legacy_workers and confirmed it failed before the guard because start_worker_groups accepted worker_inventory=None and proceeded into helper startup. Implemented a top-level RuntimeError guard in start_worker_groups before any helper invocation. Verification: pytest tldw_Server_API/tests/Services/test_startup_worker_groups.py -q -> 2 passed, 5 warnings; pytest tldw_Server_API/tests/Services/test_startup_worker_bootstrap.py -q -> 2 passed, 5 warnings; Bandit on startup_worker_groups.py -> 0 findings; git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the worker-group legacy-handle review issue by making worker_inventory mandatory at the start_worker_groups orchestration boundary. This prevents the compatibility branches in cleanup and compactor/websub startup helpers from creating orphaned background tasks after the refactor dropped their returned handle objects. Added a regression test that proves the missing-inventory path fails before any startup helper runs, and kept the change scoped to the orchestration boundary rather than reviving shutdown plumbing for a mode bootstrap no longer uses.
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
