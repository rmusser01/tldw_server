---
id: TASK-14
title: Design native CodeGraph MCP module
status: Done
assignee:
  - Codex
created_date: '2026-05-03 20:35'
updated_date: '2026-05-03 20:48'
labels:
  - mcp
  - codegraph
  - design
dependencies: []
references:
  - 'https://github.com/colbymchenry/codegraph'
documentation:
  - Docs/MCP/Unified/Adding_Tools.md
  - Docs/MCP/Unified/Modules.md
  - Docs/MCP/Unified/External_Federation.md
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for a native Python CodeGraph-style Unified MCP module in tldw_server. The approved direction is a native service with optional .[codegraph] dependencies, in-depth, first-class support for Python plus JavaScript/TypeScript, and adapter-ready architecture for C, C++, C#, Java, and Kotlin. The design should cover storage, parser adapters, workspace safety, MCP tool surface, staging, testing, and operational constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design document is written under Docs/superpowers/specs with the native CodeGraph MCP architecture and staged rollout.
- [x] #2 The design explicitly scopes the first implementation slice to deep Python plus JavaScript/TypeScript support and adapter-ready placeholders for C, C++, C#, Java, and Kotlin.
- [x] #3 The design covers workspace-root safety, SQLite/FTS storage, parser adapter contracts, MCP tools, optional dependency packaging, indexing/sync behavior, and test strategy.
- [x] #4 The Backlog task is updated with the design document path and completion notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Ground the spec in the existing Unified MCP module framework and workspace-bound filesystem patterns.
2. Define the native CodeGraph core service boundary, SQLite/FTS storage model, parser adapter contract, and optional dependency packaging as .[codegraph].
3. Specify the first implementation slice as deep Python plus JavaScript/TypeScript extraction, with adapter-ready placeholders for C, C++, C#, Java, and Kotlin.
4. Define the MCP tool catalog, indexing/sync lifecycle, workspace safety rules, output limits, testing strategy, and staged rollout.
5. Save the design under Docs/superpowers/specs, update TASK-14 with the spec path and verification notes, then commit the design/task record if the worktree allows clean staging.

Review remediation pass: update the committed design spec to resolve four accepted review findings before implementation planning. Scope: make v1 indexing default to bounded foreground with Jobs deferred; add TS path alias resolution to JS/TS scope; define stable node IDs plus stale-edge cleanup; require tested dependency pins/compatibility matrix before adding .[codegraph].
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote design spec at Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md. Verification: rg confirmed requested language scope, optional .[codegraph] dependency model, workspace safety, SQLite/FTS storage, MCP tools, indexing/sync, and TASK-14 references are present; git diff --check passed for the design doc and task file. Bandit is not applicable because this task only created documentation and Backlog task metadata.

No implementation code was changed. Automated test suites were not run because the touched files are a design spec and Backlog task metadata only. No known blockers. Follow-up implementation should start from this spec and create separate Backlog tasks for reviewable implementation slices.

Applied review remediation to Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md. Changes: v1 indexing/sync now use bounded foreground mode with Jobs deferred; JS/TS scope now includes trusted-workspace tsconfig/jsconfig path aliases; schema and lifecycle now specify deterministic node/edge IDs plus stale-edge cleanup; dependency section now requires a tested Tree-sitter parser matrix and bounded compatible ranges rather than broad lower bounds. Verification: git diff --check passed for the spec and task file; rg confirmed no remaining Jobs-vs-foreground open question and confirmed the new path-alias, deterministic identity, and parser-matrix requirements. Bandit remains not applicable because this is documentation/task metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the native CodeGraph MCP design spec to address the four accepted review findings before implementation planning. The spec now resolves the indexing mode decision by making v1 bounded foreground-only and deferring Jobs integration to a later slice; adds TypeScript/JavaScript path-alias resolution for tsconfig/jsconfig aliases used by the frontend; defines deterministic node and edge IDs plus stale-edge cleanup rules for incremental sync; and turns .[codegraph] dependency pinning into a tested parser-matrix gate with bounded candidate ranges. Verification was documentation-focused: git diff --check passed and targeted rg checks confirmed the resolved decisions are present. Automated tests and Bandit were skipped as not applicable to documentation-only changes.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
