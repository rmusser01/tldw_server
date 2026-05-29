---
id: TASK-546
title: Implement MCP Unified Stage 3F catalog schema seam cleanup
status: Done
labels:
- mcp
- mcp-unified
- standalone
- stage3
modified_files:
- Docs/superpowers/plans/2026-05-29-mcp-unified-stage3f-catalog-schema-seam-plan.md
- mcp_unified/federation/models.py
- tldw_Server_API/app/api/v1/schemas/archetype_schemas.py
- tldw_Server_API/app/core/MCP_unified/catalog_loader.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/tests/unit/test_mcp_catalog_loader.py
- backlog/tasks/task-546 - Implement-MCP-Unified-Stage-3F-catalog-schema-seam-cleanup.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the MCP external server catalog entry schema to the standalone MCP package boundary and keep tldw_server API schema imports working through compatibility re-exports. Keep the slice scoped to catalog schema ownership and import-boundary coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 catalog_loader.py no longer imports MCPCatalogEntry from tldw_Server_API API schemas.
- [x] #2 MCPCatalogEntry and MCPAuthType are owned by the standalone mcp_unified package and are re-exported by archetype_schemas for compatibility.
- [x] #3 Regression tests cover the import boundary and existing catalog loader/API schema behavior.
- [x] #4 Focused pytest, Ruff, and Bandit verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-29-mcp-unified-stage3f-catalog-schema-seam-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Stage 3F from origin/dev d8f092e553. Scope is limited to MCP catalog schema ownership and compatibility re-exports.

RED verification:
- `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_catalog_loader_uses_standalone_catalog_schema tldw_Server_API/tests/unit/test_mcp_catalog_loader.py::TestLoadMcpCatalog::test_catalog_entry_schema_is_standalone_package_model -q`
- Expected result: 2 failures while `catalog_loader.py` imported the API schema and `mcp_unified.federation.models` did not export `MCPCatalogEntry`.

Implementation:
- Added standalone `MCPAuthType` and `MCPCatalogEntry` to `mcp_unified.federation.models`.
- Re-exported the standalone names from `archetype_schemas.py` so existing API imports keep working.
- Updated `catalog_loader.py` to depend on `mcp_unified.federation.models`.
- Added extraction-boundary and loader compatibility regression coverage.

Verification:
- `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/tests/unit/test_mcp_catalog_loader.py tldw_Server_API/tests/unit/test_archetype_schemas.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 71 passed, 5 warnings.
- `.venv/bin/python -m ruff check mcp_unified/federation/models.py tldw_Server_API/app/api/v1/schemas/archetype_schemas.py tldw_Server_API/app/core/MCP_unified/catalog_loader.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/tests/unit/test_mcp_catalog_loader.py` -> all checks passed.
- `.venv/bin/python -m bandit -r mcp_unified/federation/models.py tldw_Server_API/app/api/v1/schemas/archetype_schemas.py tldw_Server_API/app/core/MCP_unified/catalog_loader.py -f json -o /tmp/bandit_mcp_stage3f_catalog_schema.json` -> 0 findings.
- `git diff --check` -> clean.

PR: https://github.com/rmusser01/tldw_server/pull/2115
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the MCP catalog entry contract to the standalone package boundary while keeping `archetype_schemas.py` as a compatibility export for API callers. The MCP catalog loader now imports `MCPCatalogEntry` from `mcp_unified.federation.models`, and regression tests cover both the package boundary and API re-export identity.

Known skips or blockers: none.
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
