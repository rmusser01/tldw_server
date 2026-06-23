---
id: TASK-2343
title: Plan MCP prompt catalog implementation
status: Done
labels:
- mcp
- prompts
- plan
references:
- TASK-2342
- TASK-2341
documentation:
- Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md
modified_files:
- Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md
- Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md
- backlog/tasks/task-2343 - Plan-MCP-prompt-catalog-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the detailed implementation plan for MCP protocol-level prompt catalog support. The plan must implement the approved TASK-2342 design: user-library prompts plus explicitly allowlisted config prompts through prompts/list and prompts/get, excluding Prompt Studio, with listChanged:false, context-aware prompt hooks, keyset pagination, stable namespace names, and compatibility for existing prompt tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create a detailed task-by-task implementation plan under Docs/superpowers/plans/.
- [x] #2 Plan covers tests first, protocol changes, catalog source helpers, formatter behavior, config/module activation, docs, and verification commands.
- [x] #3 Plan calls out Bandit, targeted pytest commands, and compatibility expectations.
- [x] #4 Plan preserves the approved exclusion of Prompt Studio and links the broader registry service to TASK-2341.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created and completed `Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md`. The plan was kept with the implementation package because TASK-2344 references it and it records the TDD, verification, docs, and review checklist used for the MCP prompt catalog implementation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all plan/design review findings before implementation. Updated the spec and implementation plan for the actual MCPProtocol test scaffold, real PromptsDatabase.add_prompt tuple API, a coherent cursor state machine with library_done, exact-boundary pagination into config prompts, partial cursor rejection, namespaced prompt permission checks without modules.read fallback, COLLATE NOCASE keyset ordering, default test-tree HTTP coverage, branch-base diff verification, explicit Prompt Studio exclusion coverage, list-time DB fallback behavior, and disabled config direct-get behavior. Verification was documentation/plan review plus targeted rg scans; no executable code changed, so pytest and Bandit are deferred to the implementation plan.
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
