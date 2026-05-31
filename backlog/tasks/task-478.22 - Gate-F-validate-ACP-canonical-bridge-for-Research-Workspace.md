---
id: TASK-478.22
title: 'Gate F: validate ACP canonical bridge for Research Workspace'
status: Done
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
modified_files:
- Docs/superpowers/specs/2026-05-26-research-workspace-acp-canonical-bridge-design.md
- Docs/superpowers/plans/2026-05-26-research-workspace-acp-canonical-bridge-plan.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
- apps/packages/ui/src/components/Option/AgentTasks/index.tsx
- apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx
- apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts
- tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py
- tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the RW-UAT-022 gap by validating the live Research Workspace to ACP canonical bridge. The contract defines ACP as part of the canonical workspace model with `canonical_workspace_source: research_workspace`, but the current matrix only has contract/test evidence and no live WebUI walkthrough keyed by the active Research Workspace ID. Scope this slice to proving the active canonical Research Workspace ID can be carried into ACP-owned UI/API state, that ACP distinguishes no-agent/no-run/unavailable states from existing run history, and that the UAT matrix is updated only as far as live backend + WebUI + CDP/Playwright evidence supports. Do not duplicate ACP run ownership inside Research Workspace and do not reintroduce `/workspace-playground` route aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active Research Workspace can navigate or hand off its canonical workspace ID to the ACP-owned surface without duplicating ACP run state in Research Workspace.
- [x] #2 ACP UI/API distinguishes no agent configured, no run history, unavailable ACP service, and existing workspace-scoped run history with actionable copy.
- [x] #3 Focused tests cover Research Workspace ID propagation into ACP route/API state and at least one explicit empty/unavailable state.
- [x] #4 Live backend + WebUI validation via CDP/Playwright records evidence in `RW-UAT-022`, moving the row only as far as the evidence supports.
- [x] #5 No Research Workspace route, metadata, API, or tests reintroduce `/workspace-playground` aliases, redirects, or active `workspace_playground` labels.
- [x] #6 Bandit is run for touched backend Python paths or explicitly skipped for frontend/docs-only scope.
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
Spec and plan created:
- Docs/superpowers/specs/2026-05-26-research-workspace-acp-canonical-bridge-design.md
- Docs/superpowers/plans/2026-05-26-research-workspace-acp-canonical-bridge-plan.md

Design reviewed locally because the available sub-agent tool requires explicit user delegation authorization; no implementation code has been changed before TDD coverage.
Implemented ACP canonical project filtering and WebUI handoff validation.

Changes:
- Added `canonical_workspace_id` and `canonical_workspace_source` query filters to `GET /api/v1/agent-orchestration/projects` while keeping ACP as the owner of project/task/run state.
- Matched canonical workspace links from bound ACP workspace metadata and transitional project metadata; normalized legacy stored `workspace_playground` source only for compatibility matching.
- Updated Research Workspace ACP run history and Agent Tasks workspace-scoped routes to request server-side canonical filters and retain client-side guards.
- Added focused backend, Vitest, and live Playwright coverage.
- Updated `RW-UAT-022` as Partial: live evidence proves canonical filtered handoff and explicit ACP unavailable/error terminal state, but not a real completed ACP run diagnostics flow.

Verification:
- RED backend: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -k "list_projects and canonical" -q` failed before query params existed.
- GREEN backend: same focused pytest passed, `2 passed, 37 deselected, 5 warnings`.
- GREEN UI: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx --maxWorkers=1` passed, `48 passed`.
- LIVE: real backend on `127.0.0.1:18002` plus WebUI on `localhost:18080`; `npx playwright test e2e/workflows/research-workspace.real-backend.spec.ts --project=chromium --grep "passes active workspace ID into ACP run history requests" --reporter=line` passed, `1 passed`.
- Route-label guard found only historical docs, negative regression tests, old storage-key compatibility, Slides fixture legacy metadata, and intentional internal legacy-source normalization; no new active route alias/redirect was added.
- Bandit: `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py -f json -o /tmp/bandit_task47822.json` reported zero results.
- `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.22 validates the Research Workspace to ACP canonical bridge without moving ACP execution ownership into Research Workspace. The API now accepts canonical workspace filters on ACP project listing; Research Workspace ACP run history and Agent Tasks use those filters; focused backend/UI tests and a live backend+WebUI Playwright run prove the active Research Workspace ID is carried into the ACP-owned surface and yields a truthful terminal state. `RW-UAT-022` remains Partial because live diagnostics for a real workspace-scoped ACP run still need a fixture-backed validation slice.
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
