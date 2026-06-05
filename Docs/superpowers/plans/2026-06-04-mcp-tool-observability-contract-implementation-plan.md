# MCP Tool Observability Contract Implementation Plan

> **For:** TASK-2256
> **Spec Reference:** Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
> **Required Sub-Skills:** test-driven-development, verification-before-completion

## Stage 1: Central Tool Definition Metadata
**Goal:** Ensure every tool definition produced by the shared MCP helper carries safe, stable evaluation metadata while preserving explicit module metadata.
**Success Criteria:** `create_tool_definition()` adds `metadata.eval` by default, does not overwrite explicit `metadata.eval`, and derives deterministic defaults from tool name/category/read/write hints.
**Tests:** Extend `test_tool_observability.py` with failing tests for default inference, preservation of explicit eval metadata, and helper-created definitions.
**Status:** Complete

## Stage 2: Protocol Execution Event Enrichment
**Goal:** Attach safe execution eval metadata to protocol-level structured tool-call responses without changing direct module return contracts.
**Success Criteria:** `tools/call` responses include scalar eval metadata for dict/structured results, preserve module-provided eval metadata, and leave text/list content compatible.
**Tests:** Add protocol tests with a stub module returning a dict result and a result with existing eval metadata.
**Status:** Complete

## Stage 3: Manual And Federated Tool Surfaces
**Goal:** Cover tool surfaces that bypass `create_tool_definition()`, especially external/federated virtual tools and manually built descriptors.
**Success Criteria:** External virtual tool definitions expose `metadata.eval` with an external/federated prompt variant, and manually constructed module tools use the same helper where needed.
**Tests:** Extend external federation/runtime tests or add focused helper tests to verify virtual tool eval metadata without mutating external adapter DTOs.
**Status:** Complete

## Stage 4: Documentation And Operator Contract
**Goal:** Document the cross-tool observability/evaluation contract for standalone gateway operators and future tool authors.
**Success Criteria:** User-facing MCP module docs describe definition metadata, execution metadata, safe field constraints, and how profile/tool evaluations can consume the contract.
**Tests:** Documentation-only review plus targeted grep/inspection.
**Status:** Complete

## Stage 5: Validation And Cleanup
**Goal:** Run targeted tests, Bandit over touched Python scope, and record results in Backlog.md.
**Success Criteria:** Targeted pytest suite passes, Bandit has no new findings in touched code, task notes/final summary record verification and any intentional skips.
**Tests:** `python -m pytest` for touched MCP tests; `python -m bandit -r` on touched Python files.
**Status:** Complete
