---
id: TASK-46
title: Implement native CodeGraph context and impact tools
status: In Progress
assignee:
  - Codex
created_date: '2026-05-04 19:17'
labels:
  - codegraph
  - mcp
  - context
  - impact
dependencies:
  - TASK-38
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1259'
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
  - Docs/superpowers/plans/2026-05-04-native-codegraph-context-impact-tools-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next native CodeGraph Stage 4 slice after PR #1264: read-only impact traversal and bounded context assembly for indexed workspaces. Keep the slice focused on codegraph.impact and codegraph.context with safe limits, no Jobs mode, no new language extractors, and no source output beyond bounded snippets requested by the context tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 codegraph.impact returns bounded incoming/outgoing/both graph neighborhoods with depth and limit controls.
- [ ] #2 codegraph.context returns task-oriented nodes, files, relationships, and bounded source snippets with truncation metadata.
- [ ] #3 Both tools validate selectors and limits, stay read-only, and offload repository/source IO from the async MCP handler.
- [ ] #4 Repository/core helpers have focused tests for traversal, source snippet bounds, missing index behavior, and path safety.
- [ ] #5 Focused CodeGraph/MCP tests, Ruff, Bandit on touched production scope, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan document: Docs/superpowers/plans/2026-05-04-native-codegraph-context-impact-tools-implementation-plan.md

Scope: implement Stage 4 read-only CodeGraph impact traversal and bounded context assembly only. The plan keeps traversal in the SQLite repository, source snippet assembly in a small CodeGraph context builder, and MCP module changes limited to tool definitions, validation, and to_thread execution.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
