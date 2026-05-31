# MCP Unified Stage 3B Protocol Host Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the small remaining protocol-level host imports for telemetry, Redis factory defaults, and truthiness checks while preserving existing `tldw_server` behavior.

**Architecture:** Keep `MCPProtocol` dependent on `MCPRuntimeDependencies` for host services. Move dynamic current-telemetry behavior into the default `tldw_server` adapter bundle, make idempotency's no-injection fallback runtime-neutral, and replace testing-helper truthiness imports with a local neutral helper.

**Tech Stack:** Python 3.11, MCP Unified protocol/runtime dependency interfaces, pytest, Ruff, Bandit.

**Status:** Complete

---

### Task 1: Add Stage 3B Protocol Boundary Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`

- [x] **Step 1: Add a failing import-boundary test for protocol host imports**

Add an AST-based test that scans `tldw_Server_API/app/core/MCP_unified/protocol.py` and fails if it imports `tldw_Server_API.app.core.Infrastructure.redis_factory`, `tldw_Server_API.app.core.Metrics.telemetry`, or `tldw_Server_API.app.core.testing`.

- [x] **Step 2: Add a failing dynamic telemetry adapter test**

Update the existing telemetry test so `MCPProtocol()` gets current telemetry through the default dependency bundle, without monkeypatching `protocol.get_telemetry_manager`.

- [x] **Step 3: Run RED tests**

Run: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py -q`

Expected: fail because `protocol.py` still imports the three host modules and default telemetry is still resolved directly in `MCPProtocol`.

Observed: failed as expected with forbidden protocol imports and default telemetry resolving outside the runtime adapter.

### Task 2: Implement Runtime-Neutral Protocol Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`

- [x] **Step 1: Remove protocol imports of host Redis, telemetry, and testing helpers**

Delete the three direct imports from `protocol.py`.

- [x] **Step 2: Add local truthiness helper**

Add a small `_is_truthy()` helper in `protocol.py` and replace the three `is_truthy(...)` call sites.

- [x] **Step 3: Make idempotency fallback runtime-neutral**

Add an async no-op Redis factory in `protocol.py` that returns `None`, and use it only when `IdempotencyManager` is constructed without an injected factory.

- [x] **Step 4: Move dynamic telemetry behavior to the tldw adapter bundle**

Add a `TldwTelemetryProvider` proxy in `tldw_runtime.py` that calls `get_telemetry_manager()` on attribute access or `trace_context(...)`, and set `telemetry_provider=TldwTelemetryProvider()` in `build_default_runtime_dependencies()`.

- [x] **Step 5: Simplify `MCPProtocol.telemetry`**

Remove `_uses_default_telemetry_manager` and return `self.dependencies.telemetry_provider` from the telemetry property.

### Task 3: Verify And Close Out

**Files:**
- Modify: `backlog/tasks/task-542 - Implement-MCP-Unified-Stage-3B-protocol-host-dependency-cleanup.md`
- Modify: `Docs/superpowers/plans/2026-05-29-mcp-unified-stage3b-protocol-host-deps-plan.md`

- [x] **Step 1: Run focused pytest**

Run: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py tldw_Server_API/app/core/MCP_unified/tests/test_stage2_context_session.py tldw_Server_API/app/core/MCP_unified/tests/test_scope_and_fallbacks.py -q`

Result: 56 passed, 5 warnings.

- [x] **Step 2: Run Ruff**

Run: `.venv/bin/python -m ruff check tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces`

Result: all checks passed.

- [x] **Step 3: Run Bandit**

Run: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces -f json -o /tmp/bandit_mcp_stage3b_protocol_host_deps.json`

Result: completed with `"results": []`.

- [x] **Step 4: Update Backlog task and commit**

Record verification results in TASK-542, mark acceptance criteria and DoD complete, then commit the Stage 3B slice.
