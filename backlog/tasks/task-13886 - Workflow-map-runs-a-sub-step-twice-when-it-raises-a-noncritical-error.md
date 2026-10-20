---
id: TASK-13886
title: Workflow map runs a sub-step twice when it raises a noncritical error
status: To Do
labels:
- workflows
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
run_map_adapter's _run_one (Workflows/adapters/control/flow.py) wraps the tracing span and the sub-step call in one try whose except _FLOW_NONCRITICAL_EXCEPTIONS branch, meant for a tracing failure, calls the sub-step again. A sub-step that raises AdapterError, RuntimeError, ValueError, OSError, TimeoutError or another member of the tuple therefore runs twice before the map step fails (probe while planning TASK-13863: adapter calls ['a', 'a']). A sub-step with a side effect, such as webhook or notify, repeats it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A map sub-step that raises runs once, and the map step fails with its error, with a test
- [ ] #2 A tracing failure still runs the sub-step once, with a test
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
