---
id: TASK-2269
title: Implement MCP gateway tool-use reporting wrapper and config
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 03:03'
labels:
  - mcp
  - gateway
  - observability
  - evals
  - implementation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from the MCP tool-use eval/reporting plan: gateway runtime wrapper, profile bridge side-channel, bootstrap config, and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway runtime wrapper records direct calls with profile/model dimensions and delegates all non-call methods.
- [x] #2 Gateway runtime wrapper records policy denials and sanitized generic failures without changing call behavior.
- [x] #3 Profile bridge tool calls attach safe side-channel metadata so reporting captures requested bridge tool id and effective backend tool name without raw arguments.
- [x] #4 Gateway bootstrap config includes disabled-by-default tool-use reporting settings, memory/sqlite store options, and wraps runtime only when enabled.
- [x] #5 Focused gateway reporting and gateway package tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 5 gateway runtime wrapper and config for MCP tool-use reporting. The gateway can now record metadata-only tool-use events for direct calls, denials, backend failures, and profile bridge calls while avoiding raw arguments. Reporting remains disabled by default and wraps bootstrapped runtimes only when configured.
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
