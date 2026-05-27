---
id: TASK-478.29
title: Validate Sandbox enabled-runtime workspace run diagnostics for Research Workspace
status: To Do
labels:
- research-workspace
- sandbox
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 29
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- TASK-478.23
- TASK-478.24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-023 Partial gap in an environment where the sandbox route/runtime is enabled. Validate that a real workspace-linked sandbox run can be created for or from the active Research Workspace ID and appears in the workspace-scoped diagnostics list. Preserve sandbox ownership of run execution and diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Validation environment has sandbox route/runtime enabled, or the task records the exact blocker without changing RW-UAT-023 to Pass.
- [ ] #2 Live fixture creates a real sandbox run with the active Research Workspace ID in workspace context.
- [ ] #3 Workspace-scoped sandbox diagnostics returns the created run and exposes truthful admission/runtime state.
- [ ] #4 Research Workspace opens sandbox-owned diagnostics without owning sandbox execution state.
- [ ] #5 RW-UAT-023 is updated only as far as live backend + WebUI + CDP evidence supports.
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
