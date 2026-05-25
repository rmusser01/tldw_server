---
id: TASK-478.7
title: 'Cross-cutting: align Research Workspace with Shared Workspaces, MCP, ACP,
  and sandbox model'
status: To Do
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
