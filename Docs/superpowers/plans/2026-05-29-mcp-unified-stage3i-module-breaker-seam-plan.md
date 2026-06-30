# MCP Unified Stage 3I Module Circuit-Breaker Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining direct host circuit-breaker dependency from MCP module base behavior while preserving tldw_server module runtime compatibility.

**Architecture:** Keep `BaseModule` and `ModuleConfig` import-compatible at the host path for this slice, but make the base module depend on a neutral circuit-breaker contract instead of importing `tldw_Server_API.app.core.Infrastructure.circuit_breaker`. Wire the tldw host circuit breaker through the existing runtime dependency bundle when modules are registered by `MCPServer`, and use a small host-neutral fallback breaker for direct module construction in tests or embedded package use.

**Tech Stack:** Python, asyncio, pytest, Ruff, Bandit.

**Backlog:** TASK-549
**PR:** https://github.com/rmusser01/tldw_server/pull/2128

---

### Task 1: RED Boundary And Injection Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`

- [x] Add a boundary contract proving `modules/base.py` does not import `tldw_Server_API.app.core.Infrastructure.circuit_breaker`.
- [x] Add a server registration regression proving `MCPServer` passes `dependencies.circuit_breaker_factory` into `ModuleConfig`.
- [x] Run the focused tests and confirm the expected RED failures.

### Task 2: Host-Neutral Base Fallback And Host Injection

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`

- [x] Replace the direct host circuit-breaker factory fallback with a local neutral async breaker contract that supports `can_attempt()`, `record_failure()`, `record_success()`, and `call_async()`.
- [x] Replace direct catching of host `CircuitBreakerOpenError` with a local neutral open exception that is also used by the fallback breaker.
- [x] Inject `self.dependencies.circuit_breaker_factory` into every `ModuleConfig` created by `MCPServer._register_default_modules()`.
- [x] Re-run RED tests and confirm they pass.

### Task 3: Compatibility Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-549 - Implement-MCP-Unified-Stage-3I-module-circuit-breaker-seam.md`
- Modify: `Docs/superpowers/plans/2026-05-29-mcp-unified-stage3i-module-breaker-seam-plan.md`

- [x] Run focused MCP module/server tests covering extraction contracts, basic module behavior, concurrency/breaker behavior, and server registration.
- [x] Run Ruff on touched Python files.
- [x] Run Bandit on touched implementation files.
- [x] Run `git diff --check`.
- [x] Update TASK-549 and this plan with verification results and final status.

## Verification

- Rebased cleanly onto `origin/dev` at `02a017e655` before final verification.
- RED focused tests: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_base_module_does_not_import_host_circuit_breaker tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_mcp_server_default_module_configs_use_injected_circuit_breaker_factory -q` -> failed as expected on the existing host import and missing factory injection.
- Focused pytest after rebase: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::TestBaseModule tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py -q` -> 38 passed, 3 warnings.
- Ruff after rebase: `.venv/bin/python -m ruff check tldw_Server_API/app/core/MCP_unified/modules/base.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py` -> All checks passed.
- Bandit after rebase: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/base.py tldw_Server_API/app/core/MCP_unified/server.py -f json -o /tmp/bandit_mcp_stage3i_module_breaker_after_rebase.json` -> 0 findings.
- Whitespace after rebase: `git diff --check` -> passed.
- Known skips/blockers: none.
