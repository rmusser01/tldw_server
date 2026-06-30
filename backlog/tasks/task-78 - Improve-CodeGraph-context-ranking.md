---
id: TASK-78
title: Improve CodeGraph context ranking
status: Done
assignee: []
created_date: '2026-05-05 17:11'
updated_date: '2026-05-05 17:22'
labels:
  - codegraph
  - mcp
  - context
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1314'
documentation:
  - Docs/MCP/Unified/CodeGraph.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve post-v1 native CodeGraph context assembly so codegraph.context selects and orders source context by task relevance and local graph relationships instead of relying only on repository search order. Keep the public MCP response shape, workspace safety, optional dependency contract, and bounded context limits intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 codegraph.context ranks candidate nodes using task-token matches in node names, qualified names, and file paths.
- [x] #2 codegraph.context boosts nodes that participate in relationships with other selected nodes so related call neighborhoods are more likely to fit inside max_nodes/max_files bounds.
- [x] #3 Existing context output shape, truncation behavior, path safety, and include_code=false behavior remain compatible.
- [x] #4 Focused CodeGraph context and MCP regression tests, Ruff on touched scope, Bandit on touched production scope, and git diff --check are run or blockers are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan added at Docs/superpowers/plans/2026-05-05-native-codegraph-context-ranking-implementation-plan.md. While mapping the MCP flow, noted that context currently searches the full task string, so the implementation will also collect token-level candidates before ranking while preserving public response shape.

Implemented context ranking and token-level candidate collection. Verification: context+MCP focused subset passed with 35 passed and 5 warnings; full CodeGraph plus MCP module suite passed with 163 passed and 5 warnings; Ruff touched scope passed; Bandit /tmp/bandit_codegraph_context_ranking.json reported 0 results; git diff --check passed.

Self-review added a regression for early common-token candidate starvation and changed context candidate collection to distribute the bounded candidate budget across whole-task and token searches before ranking.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Improved native CodeGraph context selection by adding deterministic task-token and relationship-aware ranking. codegraph.context now searches whole-task and token-level candidate terms, overselects bounded candidates, ranks them by symbol/file relevance plus graph proximity, and then clamps back to the requested max_nodes/max_files behavior without changing the public response shape. Added helper-level and MCP regression coverage for direct token ranking, relationship tie-breaking, multi-word tasks, and tight context bounds. Verification passed locally: context+MCP subset 35 passed and 5 warnings; full CodeGraph plus MCP module suite 163 passed and 5 warnings; Ruff touched scope passed; Bandit touched production scope reported 0 results; git diff --check passed. Known blockers: none.

Self-review also added and passed a regression where an early common token could otherwise exhaust the candidate budget before later task-specific terms were searched; candidate collection now gives each bounded search term a fair slice before ranking.
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
