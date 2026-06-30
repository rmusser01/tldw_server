---
id: TASK-2341
title: Design shared prompt registry service beyond MCP prompts
status: To Do
labels:
- mcp
- prompts
- design
- future-work
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Future follow-up for the broader Approach C considered during MCP prompt support brainstorming. Define a shared prompt registry service that can eventually unify MCP prompt exposure, WebUI prompt selection, prompt-loader config templates, user prompt libraries, and other prompt-like sources while excluding Prompt Studio unless explicitly re-scoped. This is intentionally out of scope for the initial MCP prompt support slice, which will use a narrower Prompt Catalog Adapter Layer behind PromptsModule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inventory prompt-like sources across user prompt libraries, config prompt files, prompt-loader consumers, character/chat prompt surfaces, and internal templates.
- [ ] #2 Define source ownership, publishing policy, RBAC boundaries, metadata schema, stable identifiers, and compatibility rules for legacy and structured prompts.
- [ ] #3 Specify migration path from MCP-local prompt catalog adapters to a shared registry without breaking existing MCP prompts/list and prompts/get behavior.
- [ ] #4 Explicitly decide whether Prompt Studio remains excluded or becomes an optional registry source in a later phase.
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
