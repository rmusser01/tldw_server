---
id: TASK-447
title: Expose Persona-local MCP tool discovery in policy editor
status: Done
labels:
- persona
- frontend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- TASK-446
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire existing MCP executable-tool discovery into the Persona Garden Policies editor so Persona-local MCP allow rules can be selected from already-authorized tools, while still preserving manual/error fallback behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Policy editor uses the existing MCP tool picker for mcp_tool rules instead of plain text only.
- [x] #2 Policy editor still supports manual/unavailable MCP fallback through the existing picker behavior and does not introduce new backend permissions.
- [x] #3 Policy editor shows selected Persona capability/default-tool context from the catalog when available.
- [x] #4 Focused frontend tests cover picker-driven rule-name updates and catalog default-tool/capability context.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Integrated `McpToolPicker` into `mcp_tool` policy rows while keeping `skill` rules on the existing text input path.
- Added read-only catalog context for selected Persona default tools and capabilities, passed from the selected catalog Persona into the Policies panel.
- Reused the existing MCP picker/manual fallback behavior and existing policy endpoints; no backend permission or discovery surface changes were introduced.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented Persona-local MCP discovery/default context in the Persona Garden Policies editor.
- RED verification: focused `ScopePolicyEditors` tests failed before implementation because picker and catalog context were missing.
- GREEN verification: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx` passed 6 tests.
- Adjacent verification: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx src/components/PersonaGarden/__tests__/PersonaGardenPanels.i18n.test.tsx src/components/PersonaGarden/__tests__/McpToolPicker.test.tsx` passed 14 tests.
- Route regression verification: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx` passed 74 tests.
- Static checks: `git diff --check` passed before final staging. `bunx tsc --noEmit --pretty false` still exits 2 on unrelated repo-wide baseline errors; visible output did not include changed Persona files.
- Bandit not applicable: frontend-only TypeScript/React changes.

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
