---
id: TASK-2266
title: Implement MCP tool-use recorder contracts and dependency fallback
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
- mcp_unified/tool_use_reporting/builders.py
- mcp_unified/tool_use_reporting/recorder.py
- mcp_unified/tool_use_reporting/__init__.py
- mcp_unified/interfaces/runtime.py
- mcp_unified/interfaces/__init__.py
- tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py
- tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the standalone tool-use recorder contract, no-op default, safe exception classification helpers, and runtime dependency fallback needed before protocol event capture is wired in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
TDD slice for Task 2: add failing tests for safe exception classification, context-dimension extraction, no-op recorder behavior, runtime dependency defaulting, duck-typed protocol fallback, explicit None fallback, and lightweight package imports; implement standalone builder/recorder contracts and runtime compatibility exports; verify focused tests, Bandit, and whitespace.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the MCP tool-use reporting builder and recorder contract slice. Added NoopToolUseRecorder, StoreBackedToolUseRecorder, safe exception classification, allowlisted context dimension extraction, runtime dependency defaulting, host compatibility exports, and protocol fallback for missing/None recorders. Verification: focused pytest passed (18 passed, 4 warnings), Bandit exited 0 with JSON report at /tmp/bandit_mcp_tool_use_reporting_task2.json, and git diff --check passed.
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
