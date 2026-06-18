# MCP Tool-Call Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrow MCP Unified tool-call lifecycle hook seam that can block before execution and observe success/failure after execution.

**Architecture:** Extend the standalone MCP runtime dependency boundary with a small hook-manager protocol and no-op default. `MCPProtocol.prepare_tool_call()` runs pre-hooks only after existing RBAC/profile/path/governance gates pass, so hooks cannot bypass explicit denies. `MCPProtocol.execute_prepared_tool_call()` runs post-hooks after success and failure with bounded metadata and ignores post-hook return values so observers cannot convert failures into successes.

**Tech Stack:** Python dataclasses/protocols, existing MCP runtime dependency injection, pytest-asyncio, Bandit.

---

### Stage 1: Hook Contract And Red Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py`
- Modify: `mcp_unified/interfaces/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`

- [x] Write failing tests for pre-hook allow, deny, ask, post-hook success/failure, and policy-deny precedence.
- [x] Run the focused hook tests and confirm they fail because the hook runtime does not exist.
- [x] Add minimal hook dataclasses/protocol/default no-op manager to the standalone runtime interface and compatibility re-export.

### Stage 2: Protocol Integration

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py`

- [x] Inject the hook manager from runtime dependencies with a no-op fallback.
- [x] Build bounded hook context payloads from sanitized tool args, tool metadata, caller metadata, scope payload, status, duration, and error type.
- [x] Run pre-hooks after existing policy checks and before returning `PreparedToolCall`; map deny to `GovernanceDeniedError` and ask to `ApprovalRequiredError`.
- [x] Run post-hooks after both success and failure inside execution; log/suppress post-hook failures.
- [x] Re-run focused tests to green.

### Stage 3: Docs, Task Notes, And Verification

**Files:**
- Modify: `Docs/MCP/Unified/Developer_Guide.md`
- Modify: `backlog/tasks/task-2378 - Add-MCP-tool-call-lifecycle-hook-seam.md`

- [x] Document the hook lifecycle, ordering, and failure behavior.
- [x] Update Backlog notes and modified files.
- [x] Run focused pytest, `python -m compileall` on touched MCP Python files, Bandit on touched Python scope, and `git diff --check`.
