---
id: TASK-2342
title: Design MCP prompt catalog support for user and allowlisted config prompts
status: In Progress
labels:
- mcp
- prompts
- design
references:
- TASK-2341
documentation:
- Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md
modified_files:
- Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the v1 MCP prompt support slice that exposes all readable non-deleted user prompt-library records and explicitly allowlisted config-file prompts through MCP protocol-level prompts/list and prompts/get, excluding Prompt Studio. The approved approach uses a Prompt Catalog Adapter Layer behind PromptsModule rather than a broad shared prompt registry service.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capture the approved architecture, protocol/data shape, permissions/config/error handling, and testing/rollout design in a spec document.
- [ ] #2 Specify that user prompt-library entries use stable library:<uuid> MCP names and config entries use config:<module>.<key-or-group> names.
- [ ] #3 Specify listChanged:false, cursor pagination, context-aware prompt hooks, and no live list_changed notifications in v1.
- [ ] #4 Document that Prompt Studio prompts are excluded and the broader shared prompt registry service is tracked separately by TASK-2341.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec written and self-reviewed. Scope covers Approach B: Prompt Catalog Adapter Layer behind PromptsModule. Approach C broad registry follow-up is tracked separately by TASK-2341.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Spec drafted for user review at Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md. Verification performed: placeholder/ambiguity scan found no remaining TBD/TODO/FIXME/loose review terms after cleanup. Bandit not applicable because this change is documentation/task tracking only.
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
