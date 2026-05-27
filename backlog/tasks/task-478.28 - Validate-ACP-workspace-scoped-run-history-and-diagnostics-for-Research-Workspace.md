---
id: TASK-478.28
title: Validate ACP workspace-scoped run history and diagnostics for Research Workspace
status: To Do
labels:
- research-workspace
- acp
- agents
- uat
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 28
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- TASK-478.22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-022 Partial gap with a fixture-backed live validation that creates or finds a real ACP run scoped to the active Research Workspace canonical ID, verifies run-history filtering and diagnostics, and updates the matrix only as far as live evidence supports. Preserve ACP as the owner of agent execution state and do not duplicate ACP runs inside Research Workspace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Live fixture creates or discovers a real ACP run scoped to the active Research Workspace canonical workspace ID.
- [ ] #2 ACP run history filters by `canonical_workspace_id` and `canonical_workspace_source=research_workspace`.
- [ ] #3 Diagnostics for the workspace-scoped run open from ACP-owned UI/API state and distinguish unavailable/no-run states from real runs.
- [ ] #4 Research Workspace does not duplicate ACP run ownership or storage.
- [ ] #5 RW-UAT-022 is updated only as far as live backend + WebUI + CDP evidence supports.
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
