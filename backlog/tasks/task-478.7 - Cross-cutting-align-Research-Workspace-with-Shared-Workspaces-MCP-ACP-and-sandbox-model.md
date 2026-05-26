---
id: TASK-478.7
title: 'Cross-cutting: align Research Workspace with Shared Workspaces, MCP, ACP,
  and sandbox model'
status: Done
labels:
- research-workspace
- shared-workspaces
- mcp
- acp
- sandbox
- architecture
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Product-model issue: Research Workspace must not become a separate one-off workspace concept. The canonical workspace model includes the existing/larger Shared Workspaces epic, the MCP hub for agent/tool access, ACP flows, and sandbox execution context.

User goal: manage research sources, chats, notes, tools, agents, and sandboxed work as parts of one coherent workspace model rather than separate product islands.

Scope:
- Map Research Workspace entities to the canonical Shared Workspaces model: workspace identity, sources, notes, chats, tags/collections, permissions/sharing, tools, agents, ACP sessions, MCP resources, and sandbox state.
- Define which APIs own workspace-scoped resources and which UI surfaces are aliases/views over the same model.
- Ensure ingestion/indexing status and selected-source/query context can be exposed to MCP/ACP/tool workflows where appropriate.
- Identify migration or compatibility risks from the old `workspace-playground` naming without adding redirects or route aliases.
- Document the model contract before broad implementation so parallel tasks do not encode conflicting assumptions.

Acceptance criteria:
- A concise workspace-model contract exists and is referenced by implementation tasks that touch workspace identity, source status, tools/agents, ACP, MCP, or sandbox behavior.
- Research Workspace APIs/UI do not duplicate or fork Shared Workspaces semantics unnecessarily.
- Future extension and agent handoffs can target the canonical workspace/resource identifiers.
- Any intentionally deferred MCP/ACP/sandbox functionality is tracked explicitly rather than implied.

Depends on: none; should be decided early alongside Gate B status work.
Blocks: extension handoff validation and any agent/tool workflow acceptance.
Parallelization: can run in parallel with Gate A model fixes and Gate B ingestion-status implementation, but should be resolved before final API/UI contracts are frozen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md` to define Research Workspace as a UI/workflow shell over the canonical Workspaces API, Shared Workspaces, MCP hub, ACP, and sandbox ownership boundaries.
- Replaced active canonical source usage with `research_workspace`; legacy `workspace_playground` is only normalized for stored metadata compatibility.
- Removed WebUI and extension route aliases/redirects for `/workspace-playground`, `/workspace-studio`, and `/research-studio`; `/research-workspace` is the canonical route.
- Renamed active backend capability endpoint/schema/core helpers from Research Studio to Research Workspace. OpenAPI now exposes `ResearchWorkspaceCapabilitiesResponse` / `ResearchWorkspaceCapability` components and no `ResearchStudio*` capability schemas.
- Renamed active frontend capability, route-state, feature-rollout, and API-client identifiers/storage keys to Research Workspace terms.
- Updated Agent Tasks, SharedWithMe, Quick Chat, smoke/e2e references, route metadata tests, extension route tests, and design guard tests to use canonical Research Workspace naming.

Verification:
- Live backend: `/api/v1/research-workspace/capabilities` returned 200, `/api/v1/research-studio/capabilities` returned 404, and OpenAPI contains the Research Workspace path plus ResearchWorkspace schemas with no ResearchStudio schemas.
- `python -m pytest tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py -q` passed: 15.
- `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -k canonical -q` passed: 9 selected.
- `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_workspace_db.py -k canonical -q` passed: 7 selected.
- `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion.py tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py -q` passed: 19.
- Focused `bunx vitest run` suites for Research Workspace capability/route-state/responsive, header, ChatPane, StudioPane, AgentTasks, QuickChat, routes, feature rollout, SharedWithMe, and extension route/navigation/performance tests passed.
- Bandit on touched Python scope passed with 0 findings in `/tmp/bandit_task4787.json`.

Known skips/blockers:
- Full extension live handoff remains deferred to TASK-478.12 pending extension build availability.
- Remaining old labels are intentional negative tests, legacy stored-metadata normalization, or legacy bundle import constants; no active route aliases/redirects remain.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace now uses the canonical Workspace model across active API/UI/test/docs surfaces, with MCP/ACP/Sandbox ownership documented and deferred implementation tracked. Old `workspace-playground` and `/research-studio` routes are removed rather than redirected; `research_workspace` is the active canonical source label. Live backend and focused frontend/backend/security verification passed.
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
