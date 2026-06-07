---
id: TASK-2268
title: Implement MCP tool-use reporting protocol capture
status: Done
labels:
- mcp
- observability
- evals
- implementation
references:
- Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md
- Docs/superpowers/plans/2026-06-06-mcp-tool-use-eval-reporting-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-06-mcp-tool-use-eval-reporting-implementation-plan.md
- mcp_unified/tool_use_reporting/__init__.py
- mcp_unified/tool_use_reporting/recorder.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py
- tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Task 4 from the MCP tool-use evaluation reporting plan: protocol-side metadata-only event recording for tools/call success, early failures, prepare failures, execution failures, and idempotency replay.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
TDD slice for Task 4: add failing protocol capture tests for success, prepare denial, early invalid tool names, recorder failure isolation, skip marker behavior, execution failure capture, and idempotency replay; implement safe recorder dispatch and protocol event builders; instrument process_request early tools/call failures, _handle_tools_call prepare failures, and execute_prepared_tool_call success/error/idempotency paths; verify focused/adjacent tests, Bandit, and whitespace.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented protocol-side MCP tool-use reporting capture. Added safe recorder dispatch, metadata-only event construction, process_request early failure recording, _handle_tools_call prepare failure recording, execute_prepared_tool_call success/error/idempotency recording, skip-marker handling for already observed calls, and focused protocol tests. Verification: focused pytest passed (35 passed, 4 warnings), Bandit exited 0 with JSON report at /tmp/bandit_mcp_tool_use_reporting_task4.json, and git diff --check passed.
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
