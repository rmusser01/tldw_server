---
id: TASK-13919
title: Workflow workflow_call cannot run a sub-workflow
status: To Do
labels:
- workflows
- backend
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
run_workflow_call_adapter (Workflows/adapters/control/orchestration.py) never creates a child run. It imports get_workflows_db from tldw_Server_API.app.core.Workflows.workflows_db (:48), a module that has never existed, so every call returns {'error': 'workflow_call_error'} (probe while planning TASK-13887, origin/dev e06eeec68c). Behind that: it calls create_run(status='pending') (:67), which has no status parameter; it creates the child without a definition_snapshot, so the child would have no steps; and it passes mode='sync' as a string (:74). Its unit tests inject a fake module through sys.modules. The WebUI step registry offers the step (apps/packages/ui/src/components/WorkflowEditor/step-registry.ts). The child run also needs a tenant and user check on get_workflow, and a guard against a workflow calling itself. This task's plan decides whether to repair the step or remove it from the registry. TASK-13887 AC 2 (a cancelled or timed-out child run ends terminal) moved here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A workflow_call step with wait true runs the named workflow's definition as a child run and returns its status and outputs, with a test that injects no fake module
- [ ] #2 A cancelled or timed-out workflow_call child run ends in a terminal status instead of staying running, with a test (from TASK-13887 AC 2)
- [ ] #3 workflow_call runs only workflows its run's tenant and user may run, and refuses a call cycle, with tests
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
