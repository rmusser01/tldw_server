---
id: TASK-2266
title: Implement MCP tool-use recorder contracts and dependency fallback
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
Add the standalone tool-use recorder contract, no-op default, safe exception classification helpers, and runtime dependency fallback needed before protocol event capture is wired in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recorder contracts, dependency fallback, and safe metadata helpers are implemented and exported.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
TDD slice for Task 2: add failing tests for safe exception classification, context-dimension extraction, no-op recorder behavior, runtime dependency defaulting, duck-typed protocol fallback, explicit None fallback, and lightweight package imports; implement standalone builder/recorder contracts and runtime compatibility exports; verify focused tests, Bandit, and whitespace.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the MCP tool-use reporting builder and recorder contract slice. Added NoopToolUseRecorder, StoreBackedToolUseRecorder, safe exception classification, allowlisted context dimension extraction, runtime dependency defaulting, host compatibility exports, and protocol fallback for missing/None recorders. Verification: focused pytest passed (18 passed, 4 warnings), Bandit exited 0 with JSON report at /tmp/bandit_mcp_tool_use_reporting_task2.json, and git diff --check passed.
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
