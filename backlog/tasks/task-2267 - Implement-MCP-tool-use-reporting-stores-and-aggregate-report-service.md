---
id: TASK-2267
title: Implement MCP tool-use reporting stores and aggregate report service
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
- mcp_unified/tool_use_reporting/models.py
- mcp_unified/tool_use_reporting/store.py
- mcp_unified/tool_use_reporting/sqlite.py
- mcp_unified/tool_use_reporting/reporting.py
- mcp_unified/tool_use_reporting/recorder.py
- mcp_unified/tool_use_reporting/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Task 3 from the MCP tool-use evaluation reporting plan: store contracts, in-memory store, SQLAlchemy-backed SQLite store, bounded event queries, and aggregate report service.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
TDD slice for Task 3: add failing tests for in-memory and SQLite event ordering, filters, JSONL export, cursor pagination, retention cleanup, report grouping, truncation disclosure, and lightweight imports; implement query/report models, in-memory store, SQLAlchemy Core SQLite store with to_thread offload, and bounded report service; verify focused tests, Bandit, and whitespace.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the MCP tool-use reporting store/report slice. Added ToolUseEventQuery and report payload models, InMemoryToolUseEventStore, SQLAlchemy-backed SQLiteToolUseEventStore, cursor helpers, retention/export operations, ToolUseReportService, and package exports that avoid eager SQLite imports. Verification: focused pytest passed (19 passed, 4 warnings), Bandit exited 0 with JSON report at /tmp/bandit_mcp_tool_use_reporting_task3.json, and git diff --check passed.
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
