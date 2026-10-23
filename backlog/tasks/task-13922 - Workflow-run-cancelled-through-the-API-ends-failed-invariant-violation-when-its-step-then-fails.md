---
id: TASK-13922
title: Workflow run cancelled through the API ends failed/invariant_violation when
  its step then fails
status: To Do
labels:
- workflows
- backend
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WorkflowEngine.cancel() (engine.py, behind POST /api/v1/workflows/runs/{id}/control with action cancel) sets the run to cancelled and cancel_requested, and kills recorded subprocess groups, but cancels no asyncio task. If the step in flight then raises (for example a RuntimeError from its killed subprocess), the step loop's failure path calls _update_run_status_guarded(status='failed') from cancelled. That transition is not allowed, and its default on_reject='fail_run' overwrites the run with status failed and error invariant_violation. Events: run_cancelled, step_failed, transition_rejected, run_failed. Measured on origin/dev e06eeec68c and with TASK-13891's fix (reviewer probe rv_fu2wf_probe_cancel_api.py). TASK-13891 keeps 'cancelled' for a step's own CancelledError after an API cancel, as dev did, and leaves this case as it was.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run cancelled through the API stays cancelled, with no run_failed event, when its in-flight step then raises any exception, in start_run and continue_run, with a test
- [ ] #2 A run that was not cancelled still fails as before when a step raises
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
