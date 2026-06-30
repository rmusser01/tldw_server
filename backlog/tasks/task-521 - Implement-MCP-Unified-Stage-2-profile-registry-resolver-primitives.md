---
id: TASK-521
title: Implement MCP Unified Stage 2 profile registry resolver primitives
status: Done
assignee: []
created_date: '2026-05-27T07:20:16Z'
updated_date: '2026-05-27 07:19'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage2
  - profiles
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow MCP Unified Stage 2 slice after built-in profile presets: package-local profile store and resolver primitives that can load stored MCPProfile documents, apply an optional default profile for standalone gateway callers, and fail closed without changing host MCP route, approval, credential, or execution behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package-local profile registry/resolver primitives are available without importing tldw_Server_API.
- [x] #2 ProfileStore supports storing, listing, retrieving, and deleting MCPProfile documents in a package-local implementation suitable for tests and standalone bootstrap.
- [x] #3 Store-backed resolver returns explicit profiles by id, uses an optional default profile only when no profile id is provided, returns no profile when unavailable or disabled, and does not treat built-in presets as mutable runtime profiles unless explicitly duplicated/stored.
- [x] #4 Tests cover import isolation, explicit profile resolution, default standalone resolution, disabled/missing profile fail-closed behavior, and copy isolation.
- [x] #5 Focused tests, Ruff/Mypy for mcp_unified, Bandit touched scope, and diff whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Plan: Docs/superpowers/plans/2026-05-27-mcp-unified-profile-registry-resolver-implementation-plan.md
- Added package-local InMemoryProfileStore plus ProfileStoreUnavailableError for explicit fail-closed availability handling.
- Added StoreBackedProfileResolver for explicit/default profile lookup, disabled/missing fail-closed behavior, and copy-isolated returns.
- Updated ProfileStore protocol to return MCPProfile values and support list/upsert/delete operations without host imports.
- Added focused package-boundary, store copy-isolation, explicit/default resolution, disabled, missing, and unavailable-store tests.
- Verification: profile registry/preset/runtime package tests passed: 20 passed, 3 warnings.
- Verification: Ruff passed for mcp_unified and the new profile registry test.
- Verification: Mypy passed for mcp_unified/profiles, mcp_unified/interfaces/storage.py, and the new test.
- Verification: runtime Bandit scan passed with 0 findings for mcp_unified/profiles and mcp_unified/interfaces/storage.py. Full touched-scope Bandit produced only pytest assert B101 findings in the new test file.
- Verification: git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP Unified Stage 2 profile registry/resolver primitives in the standalone mcp_unified package. The slice adds an in-memory profile store, an explicit store-unavailable error signal, a store-backed resolver with optional default profile lookup and fail-closed disabled/missing/unavailable behavior, package exports, and a richer ProfileStore protocol. Focused tests cover package import isolation, copy isolation, list/delete behavior, explicit/default resolution, disabled profiles, and unavailable-store behavior. No FastAPI route, host MCP execution, approval, credential, or policy enforcement paths were changed.
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
