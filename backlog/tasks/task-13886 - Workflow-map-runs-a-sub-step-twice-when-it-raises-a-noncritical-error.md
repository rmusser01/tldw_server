---
id: TASK-13886
title: Workflow map runs a sub-step twice when it raises a noncritical error
status: Done
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
- [x] #1 A map sub-step that raises runs once, and the map step fails with its error, with a test
- [x] #2 A tracing failure still runs the sub-step once, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED on dev 617094d014 (the plan measured e06eeec68c; none of the touched files changed between them): test_a_map_item_that_raises_runs_once_and_fails_the_step failed with assert [0, 0] == [0]; the span guard passed (dev already falls back once). 2 tests, 1 failed, 1 passed.
Fix: _run_one resolves the adapter first and sets started inside the workflows.map.item span; the noncritical except re-raises once the sub-step has started, and the untraced fallback runs only when the span failed to open. GREEN: both tests pass, with the map tests in test_control_fanout_cancellation.py, test_workflows_map_substeps_integration.py and test_control_adapters.py (15 passed).
Behaviour change: a span that fails on exit after the item returned now fails the item with the tracing error instead of running it again (plan Review Focus 3). Docs: Docs/Design/Workflows.md's map bullet stays true; no change. Bandit -ll on flow.py: 0 results. Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A map sub-step that raises now runs once, and the map step fails with its error; on dev the tracing fallback ran it a second time, repeating side effects such as a webhook. A span that fails to open still runs the sub-step once, untraced. Pinned by two tests in test_control_fanout_cancellation.py.
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
