---
id: TASK-2268
title: Implement MCP tool-use reporting protocol capture
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 03:03'
labels:
  - mcp
  - observability
  - evals
  - implementation
dependencies: []
references:
  - Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md
  - >-
    Docs/superpowers/plans/2026-06-06-mcp-tool-use-eval-reporting-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Task 4 from the MCP tool-use evaluation reporting plan: protocol-side metadata-only event recording for tools/call success, early failures, prepare failures, execution failures, and idempotency replay.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Protocol tool-use capture records success, denial, invalid params, unavailable, execution failure, and idempotency replay metadata without affecting tool behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
TDD slice for Task 4: add failing protocol capture tests for success, prepare denial, early invalid tool names, recorder failure isolation, skip marker behavior, execution failure capture, and idempotency replay; implement safe recorder dispatch and protocol event builders; instrument process_request early tools/call failures, _handle_tools_call prepare failures, and execute_prepared_tool_call success/error/idempotency paths; verify focused/adjacent tests, Bandit, and whitespace.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented protocol-side MCP tool-use reporting capture. Added safe recorder dispatch, metadata-only event construction, process_request early failure recording, _handle_tools_call prepare failure recording, execute_prepared_tool_call success/error/idempotency recording, skip-marker handling for already observed calls, and focused protocol tests. Verification: focused pytest passed (35 passed, 4 warnings), Bandit exited 0 with JSON report at /tmp/bandit_mcp_tool_use_reporting_task4.json, and git diff --check passed.
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
