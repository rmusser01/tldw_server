---
id: TASK-478.22
title: 'Gate F: validate ACP canonical bridge for Research Workspace'
status: In Progress
labels:
- research-workspace
- acp
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 22
parent_task_id: TASK-478
references:
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- TASK-478.7
- TASK-478.21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the RW-UAT-022 gap by validating the live Research Workspace to ACP canonical bridge. The contract defines ACP as part of the canonical workspace model with `canonical_workspace_source: research_workspace`, but the current matrix only has contract/test evidence and no live WebUI walkthrough keyed by the active Research Workspace ID. Scope this slice to proving the active canonical Research Workspace ID can be carried into ACP-owned UI/API state, that ACP distinguishes no-agent/no-run/unavailable states from existing run history, and that the UAT matrix is updated only as far as live backend + WebUI + CDP/Playwright evidence supports. Do not duplicate ACP run ownership inside Research Workspace and do not reintroduce `/workspace-playground` route aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Active Research Workspace can navigate or hand off its canonical workspace ID to the ACP-owned surface without duplicating ACP run state in Research Workspace.
- [ ] #2 ACP UI/API distinguishes no agent configured, no run history, unavailable ACP service, and existing workspace-scoped run history with actionable copy.
- [ ] #3 Focused tests cover Research Workspace ID propagation into ACP route/API state and at least one explicit empty/unavailable state.
- [ ] #4 Live backend + WebUI validation via CDP/Playwright records evidence in `RW-UAT-022`, moving the row only as far as the evidence supports.
- [ ] #5 No Research Workspace route, metadata, API, or tests reintroduce `/workspace-playground` aliases, redirects, or active `workspace_playground` labels.
- [ ] #6 Bandit is run for touched backend Python paths or explicitly skipped for frontend/docs-only scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approach: keep ACP as the owner of agent/run state. Research Workspace should carry the canonical workspace context into the ACP surface through existing or minimal route/query/API handoff, then ACP should derive and display run history or actionable empty/unavailable states from its own APIs.

Initial steps:
1. Inspect current ACP canonical bridge endpoints, ACP WebUI route/state handling, and existing tests before choosing the smallest handoff.
2. Add focused tests for Research Workspace ID propagation into ACP route/API state and ACP empty/unavailable state copy.
3. Validate against a live backend and WebUI using CDP/Playwright.
4. Update RW-UAT-022 honestly: keep Partial unless live run creation/history filtering is fully proven.
<!-- SECTION:PLAN:END -->

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
