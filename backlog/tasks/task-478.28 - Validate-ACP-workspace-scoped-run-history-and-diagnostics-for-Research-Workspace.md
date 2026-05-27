---
id: TASK-478.28
title: Validate ACP workspace-scoped run history and diagnostics for Research Workspace
status: Done
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
- [x] #1 Live fixture creates or discovers a real ACP run scoped to the active Research Workspace canonical workspace ID.
- [x] #2 ACP run history filters by `canonical_workspace_id` and `canonical_workspace_source=research_workspace`.
- [x] #3 Diagnostics for the workspace-scoped run open from ACP-owned UI/API state and distinguish unavailable/no-run states from real runs.
- [x] #4 Research Workspace does not duplicate ACP run ownership or storage.
- [x] #5 RW-UAT-022 is updated only as far as live backend + WebUI + CDP evidence supports.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a fixture-backed Playwright/CDP test in `apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-tasks.spec.ts` that creates a canonical Research Workspace, bridges it to ACP with `canonical_workspace_source=research_workspace`, creates an ACP-owned project/task/run, verifies canonical project filtering, opens Research Workspace ACP run history, and follows the ACP diagnostics handoff.
- Registered `agent_orchestration` in the core router group so the full app exposes `/api/v1/agent-orchestration/projects` outside isolated endpoint tests.
- Updated ACP runner client session creation to include `mcpServers: []` by default and retry native ACP `session/new` without `agentType` when a runner rejects that field, matching live native ACP behavior.
- Verification:
  - `TLDW_SERVER_URL=http://127.0.0.1:18001 TLDW_E2E_SERVER_URL=http://127.0.0.1:18001 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_WEB_URL=http://127.0.0.1:18080 TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18080' npx playwright test e2e/workflows/tier-3-automation/agent-tasks.spec.ts --project=tier-3 --grep "binds a Research Workspace to a real ACP run history and diagnostics path" --reporter=line` passed, `1 passed (8.4s)`.
  - `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py::test_iter_core_router_specs_populates_expected_specs -q` passed.
  - `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py::test_standard_runner_create_session_sends_session_env tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py::test_standard_runner_create_session_sends_explicit_empty_mcp_servers tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py::test_standard_runner_create_session_retries_without_agent_type_for_native_acp -q` passed.
  - `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/core.py tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py -f json -o /tmp/bandit_task_478_28.json` passed with zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.28 validates the Research Workspace to ACP run-history handoff with live evidence. The focused Playwright/CDP test creates a canonical Research Workspace, bridges it into ACP, creates an ACP-owned project/task/run, confirms server-side canonical project filtering, opens the Research Workspace ACP run history modal, sees the ACP-owned session/run state, and navigates to `/acp-playground?session=<session>&view=diagnostics`. RW-UAT-022 is now Pass. Research Workspace still only links and filters by canonical workspace ID/source; ACP remains the owner of execution workspaces, projects, tasks, runs, sessions, diagnostics, artifacts, audit, and reviewer state.
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
