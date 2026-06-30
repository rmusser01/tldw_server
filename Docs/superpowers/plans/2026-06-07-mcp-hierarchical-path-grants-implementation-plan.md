# MCP Hierarchical Path Grants Implementation Plan

## Stage 1: Compiler Contract
**Goal**: Add a standalone authored path-grant compiler in `mcp_unified` that turns org/workspace/folder/file authoring rules into flat runtime `path_grants`.
**Success Criteria**: Valid authored rules compile to normalized, de-duplicated flat grants with preview metadata; invalid prefixes/actions/effects produce diagnostics and are omitted.
**Tests**: New compiler tests for valid hierarchy, root grants, invalid rules, and duplicate action merging.
**Status**: Complete

## Stage 2: Runtime Bridge
**Goal**: Let the existing MCP Hub path enforcer consume compiled authored grants when explicit flat `path_grants` are absent, without changing the enforcement decision model.
**Success Criteria**: Authored broad allow plus child deny behaves like equivalent flat grants; malformed authored grants fail closed and do not fall back to legacy allowlists.
**Tests**: Extend path enforcement tests for allow/deny and invalid-authoring fallback behavior.
**Status**: Complete

## Stage 3: Documentation And Task Closeout
**Goal**: Document the authored hierarchy shape and record verification in Backlog.
**Success Criteria**: User guide explains authored hierarchy as an optional authoring layer that compiles into flat `path_grants`; `TASK-2299` DoD is updated.
**Tests**: `git diff --check`, focused pytest, and Bandit on touched Python code.
**Status**: Complete
