---
id: TASK-478.24
title: 'Gate F: prove workspace-linked sandbox execution appears in Research Workspace
  diagnostics'
status: Done
labels:
- research-workspace
- sandbox
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 24
parent_task_id: TASK-478
references:
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- TASK-478.23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-023 Partial gap by proving an actual workspace-linked sandbox execution can be created or observed for the active Research Workspace and appears in the sandbox-owned diagnostics envelope/panel. Scope this slice to the smallest executable proof path: keep Research Workspace from owning sandbox run state, pass canonical workspace context into sandbox execution where appropriate, validate the run appears in diagnostics with live backend + WebUI + CDP/Playwright, and update RW-UAT-023 only as far as evidence supports. Do not add `/workspace-playground` aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused backend test creates a sandbox run through `POST /api/v1/sandbox/runs` with `workspace_id` and verifies `GET /api/v1/sandbox/workspaces/{workspace_id}/diagnostics` returns that same run for the authenticated user/workspace.
- [x] #2 Research Workspace live E2E either proves the active workspace ID can be used to create/read a workspace-linked sandbox run or records the exact route-policy/runtime blocker without overclaiming.
- [x] #3 `RW-UAT-023` is updated only as far as evidence supports.
- [x] #4 No `/workspace-playground` aliases, redirects, active labels, or route metadata are added.
- [x] #5 Verification records focused tests and Bandit/touched-scope security result or an explicit non-code skip reason.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added backend contract coverage that mounts the sandbox run API and sandbox
  workspace diagnostics router together, creates a workspace-linked run through
  `POST /api/v1/sandbox/runs`, and verifies the diagnostics envelope returns the
  created run with canonical workspace metadata.
- Live validation on a fresh backend/WebUI pair (`127.0.0.1:18017` and
  `127.0.0.1:8081`) proved the current route-policy blocker: `POST
  /api/v1/sandbox/runs` returns 404 because the sandbox router is disabled by
  policy.
- Fixed the misleading diagnostics state for this configuration. When runtime
  discovery is available but the sandbox route is disabled, diagnostics now
  reports `admission.state=blocked` and `reason_code=sandbox_route_disabled`.
- Updated the Research Workspace sandbox diagnostics panel regression coverage
  and live E2E expectations so the route-policy message is treated as an
  explicit terminal state, not an error banner or hidden failure.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed TASK-478.24 without overclaiming `RW-UAT-023`. The backend contract now
proves that workspace-linked sandbox runs created through the sandbox API appear
in workspace diagnostics when the route is available. The current live app still
cannot create a real workspace-linked sandbox run because the sandbox API route is
policy-disabled; the UAT matrix remains `Partial` and records that blocker. The
diagnostics endpoint now exposes the route-policy block directly so Research
Workspace does not imply sandbox actions may run.
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

## Verification

- `python -m pytest tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py -q` -> 5 passed.
- `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceSandboxDiagnosticsPanel.test.tsx` -> 1 file passed, 4 tests passed.
- `bunx playwright test e2e/workflows/research-workspace.real-backend.spec.ts --grep "passes active workspace ID into sandbox diagnostics requests" --reporter=line --workers=1` against live backend/WebUI `18017`/`8081` -> 1 passed.
- `bunx playwright test e2e/workflows/research-workspace.real-backend.spec.ts --grep "shows workspace-linked sandbox run" --reporter=line --workers=1` against live backend/WebUI `18017`/`8081` -> 1 skipped because `POST /api/v1/sandbox/runs` is unavailable under current route policy.
- Live curl `POST /api/v1/sandbox/runs` against `127.0.0.1:18017` -> 404 Not Found.
- Live curl `GET /api/v1/sandbox/workspaces/manual-rw-proof-18017/diagnostics?source_label=research_workspace&limit=10` -> 200 OK with `admission.state=blocked` and `reason_code=sandbox_route_disabled`.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py -f json -o /tmp/bandit_task_478_24.json` -> 0 findings.
- `git diff --check` -> passed.
