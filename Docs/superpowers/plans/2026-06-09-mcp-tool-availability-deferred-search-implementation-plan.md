# MCP Tool Availability And Deferred Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MCP profile tool discovery distinguish direct model-visible tools from deferred searchable tools.

**Architecture:** Extend the existing package-local `tool_discovery.py` catalog builder with exposure classification and availability summaries. Then update `ProfileAwareGatewayRuntime.list_tools()` to expose only direct installed tools plus discovery bridge tools, while preserving `tool_search`, `tool_describe`, and `tool_call` for deferred discovery and execution.

**Tech Stack:** Python 3.11+, Pydantic profile models, FastAPI gateway runtime tests, pytest, Bandit.

---

### Task 1: Add Exposure Classification To Tool Discovery

**Files:**
- Modify: `mcp_unified/gateway/tool_discovery.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py`

- [ ] **Step 1: Write failing tests**

Add tests that create a profile with direct category `code`, deferred category `browser`, and installed backend tools in both categories. Assert catalog entries include `exposure` values and category/global counts distinguish direct and deferred installed tools.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py -q
```

Expected: new tests fail because exposure/count fields do not exist.

- [ ] **Step 3: Implement classification**

Add normalized exposure constants and helpers in `tool_discovery.py`. Direct categories come from `profile.metadata.tooling.progressive_disclosure.direct_categories`; deferred categories come from `deferred_categories`. Installed tools outside direct categories should be deferred when the profile has any progressive-disclosure metadata.

- [ ] **Step 4: Run focused tests**

Run the same pytest command. Expected: all tests pass.

### Task 2: Hide Deferred Tools From Initial Runtime Tool List

**Files:**
- Modify: `mcp_unified/gateway/profile_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Write failing runtime tests**

Add coverage showing `ProfileAwareGatewayRuntime.list_tools()` hides deferred installed tools but exposes direct installed tools and bridge discovery tools. Also assert `tool_call` appears when deferred installed tools exist.

- [ ] **Step 2: Run focused tests**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: new tests fail because deferred installed tools are still directly listed.

- [ ] **Step 3: Implement runtime filtering**

Reuse discovery classification rather than duplicating category logic. Add a package-local helper that returns direct installed backend tool names for the active profile and backend tool descriptors. Keep collision behavior: an allowed backend tool with a bridge-reserved name still wins when it is direct.

- [ ] **Step 4: Run focused runtime tests**

Run the same pytest command or targeted test names. Expected: new and existing bridge tests pass.

### Task 3: Preserve Deferred Search And Delegated Calls

**Files:**
- Modify: `mcp_unified/gateway/tool_discovery.py`
- Modify: `mcp_unified/gateway/profile_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Add regression tests**

Assert `tool_search` and `profile.tools.list` include direct, deferred installed, and recommended-unavailable entries. Assert `tool_call` delegates a deferred installed tool through backend policy and still rejects recommendation-only tools.

- [ ] **Step 2: Run tests to verify behavior**

Run focused runtime tests. Expected: existing call behavior may pass, but direct/deferred metadata assertions fail until Task 1/2 helpers are wired through.

- [ ] **Step 3: Wire helper payloads**

Ensure catalog and search payloads include `exposure`, `availability_reason_code`, and counts without exposing denied tools.

- [ ] **Step 4: Run focused tests**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: pass.

### Task 4: Docs, Backlog, And Verification

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2295 - Add-tool-availability-and-deferred-tool-search-parity.md`

- [ ] **Step 1: Update user guide**

Document that direct categories are model-visible during `tools/list`, deferred installed categories are discoverable through bridge tools, and recommended unavailable tools are setup hints only.

- [ ] **Step 2: Run package boundary and security checks**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
source ../../.venv/bin/activate && python -m py_compile mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/profile_runtime.py
source ../../.venv/bin/activate && python -m bandit -r mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/profile_runtime.py -f json -o /tmp/bandit_mcp_tool_availability_search.json
git diff --check
```

Expected: pytest passes, py_compile passes, Bandit has zero findings, and diff check is clean.

- [ ] **Step 3: Update Backlog final notes**

Record touched files, validation commands, known skips, and final summary in `TASK-2295`.

- [ ] **Step 4: Commit**

Commit with a message such as:

```bash
git add mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/profile_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py mcp_unified/USER_GUIDE.md Docs/superpowers/specs/2026-06-09-mcp-tool-availability-deferred-search-design.md Docs/superpowers/plans/2026-06-09-mcp-tool-availability-deferred-search-implementation-plan.md "backlog/tasks/task-2295 - Add-tool-availability-and-deferred-tool-search-parity.md"
git commit -m "feat: add MCP deferred tool availability search"
```
