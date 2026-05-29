---
id: TASK-547
title: Implement MCP Unified Stage 3G catalog loader package seam
status: Done
labels:
- mcp
- mcp-unified
- standalone
- stage3
modified_files:
- Docs/superpowers/plans/2026-05-29-mcp-unified-stage3g-catalog-loader-package-plan.md
- mcp_unified/federation/catalog_loader.py
- mcp_unified/federation/__init__.py
- tldw_Server_API/app/core/MCP_unified/catalog_loader.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/tests/integration/test_first_run_setup_flow.py
- tldw_Server_API/tests/unit/test_mcp_catalog_endpoints.py
- tldw_Server_API/tests/unit/test_mcp_catalog_loader.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move MCP catalog YAML loader behavior into the standalone mcp_unified package while preserving the existing tldw_Server_API catalog_loader import path as a compatibility wrapper. Keep the slice scoped to catalog loading and cache behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone mcp_unified package exposes catalog loader functions for load/list/get behavior.
- [x] #2 Existing tldw_Server_API.app.core.MCP_unified.catalog_loader imports remain compatible and share the standalone cache behavior.
- [x] #3 Regression tests cover package ownership, host-wrapper compatibility, and existing catalog loader behavior.
- [x] #4 Focused pytest, Ruff, Bandit, and diff whitespace verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `mcp_unified.federation.catalog_loader` as the standalone owner of YAML catalog parsing, Pydantic validation, and in-memory catalog cache behavior.
- Re-exported catalog loader functions from `mcp_unified.federation` so standalone consumers can import load/list/get behavior without the host API package.
- Replaced `tldw_Server_API.app.core.MCP_unified.catalog_loader` with a compatibility wrapper that imports the standalone implementation and shares the same cache object.
- Added regression coverage for the host-wrapper package boundary, standalone exports, and shared host/package cache behavior.
- Addressed Qodo review findings by switching the standalone loader to Loguru, using slice assignment for cache replacement, skipping non-mapping YAML entries, and updating remaining cache-reset fixtures to mutate in place.
- Verification: RED boundary tests failed before implementation; focused pytest later reported 94 passed; Ruff passed; Bandit reported 0 findings; `git diff --check` was clean.
- PR: https://github.com/rmusser01/tldw_server/pull/2122
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Stage 3G catalog loader package seam and addressed Qodo's review findings. The standalone `mcp_unified` package now owns catalog load/list/get behavior, while the existing host import path remains compatible through a thin wrapper. PR: https://github.com/rmusser01/tldw_server/pull/2122
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
