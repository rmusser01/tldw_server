---
id: TASK-528
title: Implement MCP Unified Stage 2E SQLite store slice
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 05:25'
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

Review-fix pass after rebasing on latest origin/dev:
- Rebased PR #2089 branch onto origin/dev at 4a48a0f6.
- Addressed async SQLite review feedback by routing async store methods through asyncio.to_thread, using a check_same_thread=False SQLite connection with timeout=30.0, and serializing connection access with a reentrant lock.
- Addressed foreign-key feedback by adding profile_id foreign keys with ON DELETE CASCADE for assignments, approval policies, and credential grants, plus regression coverage for orphan rejection and cascade cleanup.
- Addressed audit ordering feedback by normalizing persisted audit event timestamps to UTC and adding mixed-timezone ordering coverage.
- Addressed the redundant limit branch feedback by combining query LIMIT handling into one conditional.
- Evaluated the raw SQL / DB_Management comment and kept stdlib sqlite3 inside mcp_unified intentionally because this standalone storage package must not import host tldw_Server_API DB helpers; the package-boundary tests continue to enforce that.

Review-fix verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py -v -> 54 passed, 3 warnings
- source .venv/bin/activate && python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -> All checks passed
- source .venv/bin/activate && python -m mypy mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -> Success, no issues in 15 source files
- source .venv/bin/activate && python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_stage2e_review.json -> 0 findings, 0 skipped tests
- git diff --check -> clean
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 2E SQLite storage slice and completed the PR #2089 review-fix pass after rebasing on latest dev. The store remains standalone/package-local while async methods now offload SQLite work, the connection is configured for thread offload with a lock timeout, dependent profile rows use foreign-key cascades, audit timestamps are normalized for correct newest-first ordering across timezone offsets, and review coverage was expanded. Known skips/blockers: none; host DB_Management wiring remains intentionally out of scope because it would violate the standalone package boundary.
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
