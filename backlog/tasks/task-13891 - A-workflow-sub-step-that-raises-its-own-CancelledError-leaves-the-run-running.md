---
id: TASK-13891
title: A workflow sub-step that raises its own CancelledError leaves the run running
status: Done
labels:
- workflows
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the PR #3290 review (TASK-13863), pre-existing on dev. A sub-step that raises asyncio.CancelledError itself, while nothing is cancelling it, in a map step or in a parallel step with fail_fast, makes the step raise CancelledError (flow.py's _run_cancelling_siblings re-raises the child's error, as dev's gather did). The engine's step loop (engine.py, the try around asyncio.wait_for in the step attempt) catches only TimeoutError and Exception, so the CancelledError leaves start_run, and the run stays 'running' with events ['run_started', 'step_started'] until the orphan reaper finds it (probe3290_engine.py, map_item_own_cancelled_error and parallel_failfast_own_cancelled_error, on dev and on the PR). parallel without fail_fast records the sub-step's own CancelledError as parallel_step_error since TASK-13863. Options: the engine treats a CancelledError that escapes a step while asyncio.current_task().cancelling() == 0 as a step failure; or the helper converts a child's own CancelledError into an AdapterError when its own task is not being cancelled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A map sub-step that raises its own CancelledError fails the step and the run ends in a terminal status, with a test
- [x] #2 A parallel sub-step with fail_fast that raises its own CancelledError does the same, with a test
- [x] #3 A real cancellation of the step (its timeout, or the run task cancelled) still propagates as before, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED on dev 617094d014 (the plan measured e06eeec68c; no file this plan touches changed between them): test_engine_step_own_cancelled_error.py gave 6 failed, 8 passed. Each failure was CancelledError raised out of start_run or continue_run, for map, parallel with fail_fast, and a plain step (a plain step was not in the task text, which is why the fix is in the engine rather than in _run_cancelling_siblings). The passes are the AC 3 guards, each for start_run and continue_run: the step timeout, a cancelled run task, a cancel through the run-cancel API, and an unreadable cancel flag.
Fix: both engine step loops catch asyncio.CancelledError. They re-raise when the run's task has cancel requests pending (current_task().cancelling() > 0) or the run was cancelled through the API (db.is_cancel_requested(run_id); engine.cancel() cancels no task). The flag read is suppressed for _CANCEL_FLAG_READ_ERRORS (the engine's noncritical tuple plus sqlite3.Error and the backend DatabaseError) and defaults to cancelled, so a failed read propagates the CancelledError as on dev. Otherwise the step failed with AdapterError('step_raised_cancelled_error'), which is retriable like any adapter error. GREEN: 14 passed; with the fan-out and map integration files, 27 passed. Six mutants (each clause's task half, flag half and read default): each is killed, by exactly the tests the plan names (1, 1, 2, 2, 1 and 1 failed).
Plan reviews: an earlier draft re-raised on cancelling() only, so an API cancel followed by the step's own CancelledError ended failed/invariant_violation instead of dev's cancelled; the flag half and its test fix that. The same overwrite for any other exception after an API cancel is pre-existing and filed as TASK-13922. Docs: none describe the engine's handling of a step's CancelledError; no change. Bandit -ll on engine.py: 0 results. Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A step that raises its own CancelledError (a map item, a parallel sub-step with fail_fast, or a plain step) now fails the step and the run, with error step_raised_cancelled_error, in both start_run and continue_run. On dev the run stayed running for the orphan reaper. A real cancellation still propagates as before: the run task cancelled (for example at shutdown) leaves the run for the reaper, a cancel through the run-cancel API keeps the run cancelled, an unreadable cancel flag propagates the CancelledError, and the step timeout is still a step_timeout. Follow-up: TASK-13922 (an API cancel followed by any other step error ends invariant_violation).
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
