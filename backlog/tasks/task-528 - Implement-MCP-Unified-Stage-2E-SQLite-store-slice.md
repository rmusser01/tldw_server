---
id: TASK-528
title: Implement MCP Unified Stage 2E SQLite store slice
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 05:05'
labels:
  - mcp-unified
  - standalone
  - stage2
  - sqlite
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and implement the next reviewable MCP Unified standalone Stage 2E slice: gateway-local SQLite store and migration primitives for the split profile, assignment, approval policy, credential grant, external registry, and audit contracts. Keep the slice package-local and avoid runtime execution, FastAPI routes, external process lifecycle, or gateway entrypoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SQLiteMCPStore initializes an idempotent schema with explicit schema version metadata.
- [x] #2 Profiles, assignments, approval policies, credential grants, and external server definitions round-trip through SQLite with filtered listing and delete behavior.
- [x] #3 Audit events append and query by actor, profile, event type, and newest-first limit semantics.
- [x] #4 The standalone storage module remains free of tldw_Server_API imports and does not add FastAPI, runtime, gateway, or lifecycle wiring.
- [x] #5 Focused regression, Ruff, Mypy, Bandit, and diff whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a package-local SQLiteMCPStore in mcp_unified.storage using stdlib sqlite3 and JSON payload columns plus indexed filter columns. Added validation/copy boundaries through existing Pydantic models and allowlisted SQL identifier handling for filter queries while keeping SQL values parameterized. Exported the store from mcp_unified.storage and added contract tests for schema creation, future-schema rejection, CRUD/filter behavior, audit query semantics, copy isolation, and package-boundary isolation.

Verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py -v -> 50 passed, 3 warnings
- source .venv/bin/activate && python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -> All checks passed
- source .venv/bin/activate && python -m mypy mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -> Success, no issues in 15 source files
- source .venv/bin/activate && python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_stage2e_sqlite.json -> 0 findings, 0 skipped tests
- git diff --check -> clean
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 2E SQLite storage slice for MCP Unified standalone extraction. The new store covers schema-versioned SQLite persistence for profiles, profile assignments, approval policies, credential grants, external server definitions, and audit events without introducing host-package imports or runtime enforcement wiring. Known skips/blockers: none; gateway entrypoints, YAML import/export, runtime enforcement, and external MCP lifecycle remain out of scope for this slice.
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
