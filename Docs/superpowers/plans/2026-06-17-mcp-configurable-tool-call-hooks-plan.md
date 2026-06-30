# MCP Configurable Tool-Call Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a host-neutral, configurable MCP tool-call hook manager that makes the existing hook seam usable without changing default runtime behavior.

**Architecture:** Keep hook contracts in `mcp_unified.interfaces.runtime` and add concrete manager/config primitives under a new `mcp_unified.tool_hooks` package. The protocol remains the enforcement point; the concrete manager only orders hooks, normalizes results, captures safe hook metadata, and optionally records metadata-only hook events.

**Tech Stack:** Python dataclasses/protocols, existing MCP runtime dependency injection, existing tool-use reporting models, pytest.

---

## Stage 1: Hook Manager Package
**Goal**: Provide concrete host-neutral hook registration and deterministic execution.
**Success Criteria**: A configured manager can run ordered pre/post hooks, return the first blocking pre-hook decision, and never require tldw_server imports.
**Tests**: Unit tests for no hooks, ordering, deny/ask precedence, disabled hooks, and post-hook suppression.
**Status**: Complete

- [ ] Add failing tests in `tldw_Server_API/app/core/MCP_unified/tests/test_tool_hook_manager.py`.
- [ ] Create `mcp_unified/tool_hooks/models.py` for hook configs, callback protocols, and safe metadata result models.
- [ ] Create `mcp_unified/tool_hooks/manager.py` for `ConfiguredToolCallHookManager`.
- [ ] Export the package from `mcp_unified/tool_hooks/__init__.py`.
- [ ] Run focused tests and keep the existing protocol hook tests green.

## Stage 2: Reporting Metadata Contract
**Goal**: Surface hook decisions/failures in metadata-only reporting without leaking tool arguments or raw exception messages.
**Success Criteria**: Tool-use events can carry bounded hook metadata, and protocol events attach the hook summary when available.
**Tests**: Model validation tests plus protocol/tool-use tests for hook-denied and hook-post-failure summaries.
**Status**: Complete

- [ ] Add failing model tests for bounded hook decision metadata.
- [ ] Extend `mcp_unified/tool_use_reporting/models.py` with safe hook metadata fields.
- [ ] Update protocol event construction to read hook summaries from request metadata.
- [ ] Run reporting and protocol hook tests.

## Stage 3: Runtime Injection And Defaults
**Goal**: Let embedders opt into the configured manager while preserving no-op defaults.
**Success Criteria**: `MCPRuntimeDependencies` defaults remain no-op, and package callers can build a configured manager directly.
**Tests**: Extraction contract tests and manager tests verify default no-op behavior and explicit injection behavior.
**Status**: Complete

- [ ] Add tests proving default runtime dependencies still use `NoopToolCallHookManager`.
- [ ] Add tests proving `MCPProtocol` accepts the configured manager via dependencies.
- [ ] Keep tldw default dependency builder unchanged unless tests reveal a required package-level hook config.

## Stage 4: Docs And Verification
**Goal**: Document the new package-level hook manager and complete validation.
**Success Criteria**: User guide explains how to configure hooks, failure behavior, ordering, and reporting metadata.
**Tests**: Focused pytest suite and Bandit on touched Python paths.
**Status**: Complete

- [ ] Update `mcp_unified/USER_GUIDE.md` with the hook manager integration.
- [ ] Update `mcp_unified/README.md` if the package API surface needs a short mention.
- [ ] Run focused pytest commands.
- [ ] Run Bandit on touched Python package/test files.
- [ ] Update `TASK-2379` with verification and final summary.
