---
id: TASK-14
title: Design native CodeGraph MCP module
status: Done
assignee:
  - Codex
created_date: '2026-05-03 20:35'
updated_date: '2026-05-03 20:38'
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
Create a design spec for a native Python CodeGraph-style Unified MCP module in tldw_server. The approved direction is a native service with optional .[codegraph] dependencies, deep first support for Python plus JavaScript/TypeScript, and adapter-ready architecture for C, C++, C#, Java, and Kotlin. The design should cover storage, parser adapters, workspace safety, MCP tool surface, staging, testing, and operational constraints.
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote design spec at Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md. Verification: rg confirmed requested language scope, optional .[codegraph] dependency model, workspace safety, SQLite/FTS storage, MCP tools, indexing/sync, and TASK-14 references are present; git diff --check passed for the design doc and task file. Bandit is not applicable because this task only created documentation and Backlog task metadata.

No implementation code was changed. Automated test suites were not run because the touched files are a design spec and Backlog task metadata only. No known blockers. Follow-up implementation should start from this spec and create separate Backlog tasks for reviewable implementation slices.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the native CodeGraph MCP design spec at Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md and tracked it in TASK-14. The design chooses a native Python service exposed through Unified MCP, packaged behind .[codegraph], with first-slice depth for Python plus JavaScript/TypeScript and adapter-ready planning for C, C++, C#, Java, and Kotlin. It covers SQLite/FTS storage, trusted workspace-root safety, parser adapter contracts, MCP tool definitions, indexing/sync lifecycle, optional dependency behavior, testing strategy, rollout stages, risks, and open implementation questions. Verification was documentation-focused: rg checks confirmed the approved scope appears in the spec and git diff --check passed. Bandit and automated test suites were skipped as not applicable to documentation-only changes.
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
