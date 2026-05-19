---
id: TASK-46
title: Implement native CodeGraph context and impact tools
status: Done
assignee:
  - Codex
created_date: '2026-05-04 19:17'
updated_date: '2026-05-05 00:19'
labels:
  - codegraph
  - mcp
  - context
  - impact
dependencies:
  - TASK-38
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1259'
  - 'https://github.com/rmusser01/tldw_server/pull/1270'
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
  - >-
    Docs/superpowers/plans/2026-05-04-native-codegraph-context-impact-tools-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next native CodeGraph Stage 4 slice after PR #1264: read-only impact traversal and bounded context assembly for indexed workspaces. Keep the slice focused on codegraph.impact and codegraph.context with safe limits, no Jobs mode, no new language extractors, and no source output beyond bounded snippets requested by the context tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 codegraph.impact returns bounded incoming/outgoing/both graph neighborhoods with depth and limit controls.
- [x] #2 codegraph.context returns task-oriented nodes, files, relationships, and bounded source snippets with truncation metadata.
- [x] #3 Both tools validate selectors and limits, stay read-only, and offload repository/source IO from the async MCP handler.
- [x] #4 Repository/core helpers have focused tests for traversal, source snippet bounds, missing index behavior, and path safety.
- [x] #5 Focused CodeGraph/MCP tests, Ruff, Bandit on touched production scope, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan document: Docs/superpowers/plans/2026-05-04-native-codegraph-context-impact-tools-implementation-plan.md

Scope: implement Stage 4 read-only CodeGraph impact traversal and bounded context assembly only. The plan keeps traversal in the SQLite repository, source snippet assembly in a small CodeGraph context builder, and MCP module changes limited to tool definitions, validation, and to_thread execution.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented repository impact traversal with deterministic bounded traversal; removed dynamic SQL construction after Bandit flagged the first implementation.

Added CodeGraphContextBuilder for workspace-bounded snippets, file-size handling, path traversal protection, missing-file metadata, include_code=false metadata-only output, and truncation metadata.

Exposed codegraph.impact and codegraph.context through the Unified MCP CodeGraph module with readOnlyHint metadata, strict argument validation, missing-index read-only responses, limit bounds, and asyncio.to_thread offload for repository/source IO.

Verification on 2026-05-04 after rebasing onto origin/dev: pytest focused CodeGraph/MCP suite -> 92 passed, 5 warnings; Ruff touched scopes -> All checks passed; Bandit report /tmp/bandit_codegraph_context_impact.json -> zero findings; git diff --check -> clean.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1270

PR #1270 review follow-up started: address Gemini comments about duplicated CodeGraph node serialization and relationship-neighborhood connection cycles.

PR #1270 review fixes completed: centralized CodeGraphNode serialization in codegraph_node_to_dict and replaced per-node context relationship traversal with repository.traverse_impact_many batch traversal.

Review-fix verification on 2026-05-05: focused CodeGraph/MCP pytest suite -> 95 passed, 5 warnings; Ruff touched scopes -> All checks passed; Bandit /tmp/bandit_codegraph_context_impact.json -> zero findings; git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Stage 4 native CodeGraph context and impact slice. The branch adds deterministic repository graph traversal, a workspace-safe bounded context builder for source snippets, and read-only Unified MCP tools codegraph.impact and codegraph.context with validation, configured bounds, missing-index behavior, and async offload for blocking repository/source IO. Focused CodeGraph/MCP tests, Ruff, Bandit, and whitespace checks passed after rebasing onto origin/dev. PR review follow-up centralized CodeGraph node serialization in models.py and added batch impact traversal for context relationship collection, addressing Gemini review comments without changing MCP response shapes.
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
