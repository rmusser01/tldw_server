---
id: TASK-478.23
title: 'Gate F: validate Sandbox handoff for Research Workspace'
status: Done
labels:
- research-workspace
- sandbox
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 23
parent_task_id: TASK-478
references:
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- TASK-478.7
- TASK-478.22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the RW-UAT-023 gap by validating how Research Workspace carries its active canonical workspace context into sandbox-owned diagnostics and admission surfaces. The canonical workspace model includes sandbox execution context as part of Workspaces, but the live UAT matrix currently has no Research Workspace to sandbox admission/diagnostics evidence. Scope this slice to designing and implementing the smallest first-class handoff that lets the user see sandbox readiness/admission state for the active Research Workspace without duplicating sandbox run ownership inside Research Workspace, then validate with a live backend + WebUI + CDP/Playwright. Do not add `/workspace-playground` aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active Research Workspace can navigate or hand off its canonical workspace ID/source label to a sandbox-owned or sandbox-backed surface without duplicating sandbox run state in Research Workspace.
- [x] #2 Sandbox UI/API state distinguishes no sandbox runtime configured, runtime unavailable, admission denied, no sandbox runs, and existing workspace-scoped diagnostics where supported.
- [x] #3 Focused tests cover Research Workspace ID propagation into sandbox route/API state and at least one explicit unavailable/admission-denied/empty state.
- [x] #4 Live backend + WebUI validation via CDP/Playwright records evidence in `RW-UAT-023`, moving the row only as far as the evidence supports.
- [x] #5 No Research Workspace route, metadata, API, or tests reintroduce `/workspace-playground` aliases, redirects, or active `workspace_playground` labels.
- [x] #6 Bandit is run for touched backend Python paths or explicitly skipped for frontend/docs-only scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Initial approach: inspect existing sandbox discovery/status/admission APIs and WebUI sandbox/admin surfaces, then choose the narrowest handoff that makes Research Workspace's canonical workspace context visible to sandbox-owned diagnostics. Prefer additive query/context parameters and existing sandbox discovery/status contracts over inventing a Research Workspace-specific sandbox status model. Validate with focused tests and live Playwright; keep `RW-UAT-023` Partial unless a real workspace-scoped sandbox run/diagnostic path is proven live.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the sandbox handoff as a sandbox-owned, user-safe diagnostics path. Backend changes add workspace-aware admin run filters, a stable `GET /api/v1/sandbox/workspaces/{workspace_id}/diagnostics` route registered outside the sandbox admin route gate, and a shared sandbox service singleton so admin and diagnostics readers see the same memory-backed run state. WebUI changes add a typed client method, diagnostics panel, remediation action, and workspace-settings entry without adding a trust bar or route alias. Design review found and fixed the live route-policy issue where the diagnostics route returned 404 when the broader sandbox admin router was disabled. Live evidence: direct curl returned 200 for the diagnostics endpoint on the worktree backend; Playwright opened `/research-workspace`, opened Workspace settings > Sandbox diagnostics, observed `source_label=research_workspace`, `limit=10`, the active workspace ID in the request path, and response status `<400`. Verification: focused backend tests passed (`4 passed, 2 warnings`), route-group contract passed, focused frontend tests passed (`48 passed`), Bandit on touched backend Python returned 0 errors/0 findings, and `git diff --check` passed. The route-label guard found only historical docs, defensive negative tests, and existing legacy local-storage import references; no new active alias/redirect/label was added. Known limit: `RW-UAT-023` remains Partial because no real workspace-scoped sandbox run was created or observed in this slice. Known unrelated test issue: the broader sandbox admin pagination file still has a background-worker teardown hang outside the new workspace filter test.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace now has a live sandbox-owned diagnostics handoff for the active canonical workspace, with user-safe runtime/admission state and recent workspace-scoped run summaries. The route remains available even when sandbox admin/ops routes are disabled, while admin sandbox routes stay gated. The UAT matrix was updated to Partial with live backend + WebUI evidence; completion still requires proving an actual workspace-linked sandbox execution appears in diagnostics.
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
