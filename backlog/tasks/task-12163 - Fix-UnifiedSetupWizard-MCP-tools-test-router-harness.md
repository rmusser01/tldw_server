---
id: TASK-12163
title: Fix UnifiedSetupWizard MCP tools test router harness
status: Done
labels:
- tests
- mcp
- first-run
- webui
modified_files:
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
- backlog/tasks/task-12163 - Fix-UnifiedSetupWizard-MCP-tools-test-router-harness.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the validation failure found during the first-run MCP setup follow-up pass. The UnifiedSetupWizard tests render the MCP tools step without a React Router context, while McpToolsStep legitimately uses Link for MCP Hub handoff actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused UnifiedSetupWizard MCP tools regressions pass with the same React Router context the app provides.
- [x] #2 The broader first-run MCP onboarding and MCP Hub focused frontend validation suite passes.
- [x] #3 Only test/backlog metadata changes are made unless product code is required by evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: the focused first-run MCP frontend validation suite failed with 2 UnifiedSetupWizard tests because McpToolsStep rendered React Router Link without a router context after navigating back from first chat. GREEN: wrapping those two wizard renders in MemoryRouter made the focused UnifiedSetupWizard file pass with 31 tests, then the broader first-run MCP onboarding/MCP Hub suite passed with 88 tests. Backend first-run MCP validation also passed with 72 selected tests before the frontend harness fix. Bandit not applicable because this slice touched only frontend test code and Backlog metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the validation harness issue found during the first-run MCP follow-up pass. The affected UnifiedSetupWizard tests now provide the same React Router context that the app provides when they render the MCP tools step after returning from first chat. Verification: backend focused first-run MCP pytest passed with 72 selected tests; UnifiedSetupWizard Vitest passed with 31 tests; broader first-run MCP onboarding and MCP Hub focused Vitest passed with 88 tests. Bandit skipped because no Python/runtime code changed.
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
