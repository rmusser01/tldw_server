---
id: TASK-13891
title: A workflow sub-step that raises its own CancelledError leaves the run running
status: To Do
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
- [ ] #1 A map sub-step that raises its own CancelledError fails the step and the run ends in a terminal status, with a test
- [ ] #2 A parallel sub-step with fail_fast that raises its own CancelledError does the same, with a test
- [ ] #3 A real cancellation of the step (its timeout, or the run task cancelled) still propagates as before, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
