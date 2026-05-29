# MCP Unified Stage 3 Host Adapter Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first narrow Stage 3 host-adapter seams so `MCPServer` can use injected host services for auth, lifecycle, permission seeding, module config defaults, and policy-context flags while preserving current `tldw_server` behavior.

**Architecture:** Extend the neutral runtime dependency bundle with small host service protocols, implement default `tldw_server` adapters in `tldw_runtime.py`, and wire only the server paths covered by this slice through those adapters. Keep the standalone `mcp_unified` package free of `tldw_Server_API` imports and avoid gateway entrypoints.

**Tech Stack:** Python 3.11, FastAPI/WebSocket server code, Pydantic/dataclass protocols, pytest/pytest-asyncio, Ruff, Bandit.

---

### Task 1: Contract Tests For Stage 3 Host Seams

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`

- [x] **Step 1: Add failing extraction-contract assertions**

Require fake runtime dependencies to expose `auth_provider`, `lifecycle_guard`, `permission_seeder`, `module_config_provider`, and `policy_context_provider`. Assert `MCPServer` stores and uses the injected adapter objects.

- [x] **Step 2: Add failing behavior tests**

Add small async tests proving injected dependencies handle API-key scope extraction, policy-context flag lookup, shutdown-family registration, media DB default resolution, and permission seeding without calling the module-level tldw imports.

- [x] **Step 3: Run RED tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py -q`

Result: Failed before implementation with missing dependency fields/server wiring, then passed after implementation.

### Task 2: Runtime Protocols And Default tldw Adapters

**Files:**
- Modify: `mcp_unified/interfaces/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`

- [x] **Step 1: Add host service protocols**

Add protocols for auth token behavior, lifecycle guarding/shutdown registration, permission seeding, module config defaults, and policy-context feature flags. Keep types neutral and free of `tldw_Server_API` imports.

- [x] **Step 2: Add fields to `MCPRuntimeDependencies`**

Add the new protocol fields to the dataclass. Update test fake dependencies accordingly.

- [x] **Step 3: Implement default tldw adapters**

Move direct host calls behind `TldwAuthProvider`, `TldwLifecycleGuard`, `TldwPermissionSeeder`, `TldwModuleConfigProvider`, and `TldwPolicyContextProvider`. Preserve current fallback behavior and secret redaction.

### Task 3: Wire MCPServer Through Injected Dependencies

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`

- [x] **Step 1: Replace direct server auth/lifecycle calls**

Use injected dependencies for AuthNZ token detection, JWT manager access, API-key permission normalization, app lifecycle startup checks, and shutdown transport registration.

- [x] **Step 2: Replace direct permission/module config calls**

Use injected dependencies for wildcard permission seeding, default media DB path resolution in module registration, and MCP Hub policy-context enabled metadata.

- [x] **Step 3: Preserve legacy imports only where still required**

Keep compatibility imports only where the server still directly owns host behavior outside this slice. Do not move domain modules or start Stage 4 gateway work.

### Task 4: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-540 - Implement-MCP-Unified-Stage-3-host-adapter-shim-slice.md`

- [x] **Step 1: Run focused pytest**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py tldw_Server_API/app/core/MCP_unified/tests/test_stage2_context_session.py tldw_Server_API/app/core/MCP_unified/tests/test_scope_and_fallbacks.py -q`

- [x] **Step 2: Run lint and security checks**

Run: `source .venv/bin/activate && python -m ruff check mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`

Run: `source .venv/bin/activate && python -m bandit -r mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces tldw_Server_API/app/core/MCP_unified/adapters tldw_Server_API/app/core/MCP_unified/server.py -f json -o /tmp/bandit_mcp_stage3_host_adapters.json`

- [x] **Step 3: Update Backlog task and commit**

Record verification, check acceptance criteria/DoD when complete, and commit the Stage 3A slice.
