# MCP Unified Stage 3F Catalog Schema Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MCP external catalog entry schemas into the standalone `mcp_unified` package while preserving tldw_server API schema compatibility.

**Architecture:** Define the catalog schema in `mcp_unified.federation.models`, update the MCP catalog loader to import from that package boundary, and re-export the same model/type from `archetype_schemas.py` so existing API imports remain stable.

**Tech Stack:** Python, Pydantic, pytest, Ruff, Bandit.

**Backlog:** TASK-546

---

### Task 1: RED Boundary And Compatibility Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Modify: `tldw_Server_API/tests/unit/test_mcp_catalog_loader.py`

- [x] Add a contract test that fails while `catalog_loader.py` imports `MCPCatalogEntry` from `tldw_Server_API.app.api.v1.schemas.archetype_schemas`.
- [x] Add/adjust catalog behavior coverage proving `catalog_loader.py` returns the same `MCPCatalogEntry` class exported by `mcp_unified.federation.models` and by `archetype_schemas.py`.
- [x] Run the focused tests and confirm the boundary test fails before implementation.

### Task 2: Move Catalog Schema Ownership

**Files:**
- Modify: `mcp_unified/federation/models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/archetype_schemas.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/catalog_loader.py`

- [x] Add `MCPAuthType` and `MCPCatalogEntry` to `mcp_unified.federation.models`.
- [x] Re-export those names from `archetype_schemas.py`.
- [x] Update `catalog_loader.py` to import from `mcp_unified.federation.models`.
- [x] Re-run the focused tests and confirm they pass.

### Task 3: Validation And Closeout

**Files:**
- Modify: `backlog/tasks/task-546 - Implement-MCP-Unified-Stage-3F-catalog-schema-seam-cleanup.md`

- [x] Run focused pytest for extraction contracts, catalog loader, archetype schema tests, and standalone package boundary tests.
- [x] Run Ruff and Bandit on touched files.
- [ ] Update TASK-546 with verification, final summary, and PR link.
