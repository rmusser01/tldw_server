# MCP Unified Stage 3G Catalog Loader Package Seam Plan

**Goal:** Move MCP catalog YAML loading into the standalone `mcp_unified` package while preserving existing `tldw_Server_API` imports.

**Backlog:** TASK-547

## Stage 1: RED Boundary And Compatibility Tests
**Goal:** Capture the desired package boundary before implementation.
**Success Criteria:** Tests fail because `mcp_unified.federation.catalog_loader` does not exist and the host loader still owns implementation state.
**Tests:** Focused pytest for extraction contracts and catalog loader package compatibility.
**Status:** Complete

- [x] Add a boundary contract that requires the host catalog loader to delegate to `mcp_unified.federation.catalog_loader`.
- [x] Add catalog loader tests proving standalone and host import paths expose the same functions/cache state.
- [x] Run focused tests and confirm the expected RED failures.

## Stage 2: Move Loader Ownership
**Goal:** Put YAML loading/cache behavior in the standalone package and leave the host path as a compatibility wrapper.
**Success Criteria:** Existing host imports keep working while package imports are usable by standalone consumers.
**Tests:** RED tests turn green; existing catalog loader behavior remains unchanged.
**Status:** Complete

- [x] Add `mcp_unified.federation.catalog_loader`.
- [x] Re-export loader functions from `mcp_unified.federation`.
- [x] Replace host `catalog_loader.py` with a compatibility wrapper.
- [x] Re-run focused tests.

## Stage 3: Validation And Closeout
**Goal:** Verify the slice and record the result.
**Success Criteria:** Focused pytest, Ruff, Bandit, and diff whitespace checks pass or have documented non-slice blockers.
**Tests:** Focused pytest, Ruff, Bandit, `git diff --check`.
**Status:** Complete

- [x] Run focused pytest for extraction contracts, catalog loader, archetype schema, and package boundary tests.
- [x] Run Ruff and Bandit on touched code.
- [x] Update TASK-547 with verification, summary, and PR link.

## Verification Log

- RED: `python -m pytest ...::test_catalog_loader_delegates_to_standalone_package_loader ...::test_standalone_package_exports_catalog_loader_functions ...::test_host_and_standalone_loader_paths_share_cache -q` failed as expected before implementation because the standalone loader module did not exist and the host loader still owned YAML/Pydantic parsing.
- GREEN: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/tests/unit/test_mcp_catalog_loader.py tldw_Server_API/tests/unit/test_mcp_catalog_endpoints.py tldw_Server_API/tests/integration/test_first_run_setup_flow.py tldw_Server_API/tests/unit/test_archetype_schemas.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 94 passed.
- Ruff: `.venv/bin/python -m ruff check mcp_unified/federation/catalog_loader.py mcp_unified/federation/__init__.py tldw_Server_API/app/core/MCP_unified/catalog_loader.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/tests/unit/test_mcp_catalog_loader.py tldw_Server_API/tests/unit/test_mcp_catalog_endpoints.py tldw_Server_API/tests/integration/test_first_run_setup_flow.py` -> all checks passed.
- Bandit: `.venv/bin/python -m bandit -r mcp_unified/federation/catalog_loader.py mcp_unified/federation/__init__.py tldw_Server_API/app/core/MCP_unified/catalog_loader.py -f json -o /tmp/bandit_mcp_stage3g_catalog_loader.json` -> 0 findings.
- Whitespace: `git diff --check` -> clean.
- PR: https://github.com/rmusser01/tldw_server/pull/2122
- Qodo review pass: addressed Loguru logging, in-place cache reset fixtures, non-mapping YAML entry skipping, and slice-assignment cache replacement.
