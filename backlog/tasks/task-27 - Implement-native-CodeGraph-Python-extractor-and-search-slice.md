---
id: TASK-27
title: Implement native CodeGraph Python extractor and search slice
status: Done
assignee: []
created_date: '2026-05-04 02:38'
updated_date: '2026-05-04 02:49'
labels:
  - codegraph
  - mcp
  - python
dependencies:
  - TASK-16
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1244'
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
  - >-
    Docs/superpowers/plans/2026-05-03-native-codegraph-foundation-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next native CodeGraph slice after the merged foundation: index Python symbols into the existing graph tables, expose searchable symbol/node/caller/callee read tools through Unified MCP, and keep behavior bounded and truthful. This task builds on the Stage 1 foundation from PR #1244 and should not add JavaScript/TypeScript extraction, path-alias resolution, Jobs mode, impact/context tools, or planned-language extractors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Python files are parsed into deterministic module, class, function, method, import, and same-file call graph nodes or references without requiring optional tree-sitter dependencies at import time.
- [x] #2 The repository can persist, replace, search, and fetch graph nodes and edges without returning dangling relationships after re-index or sync.
- [x] #3 Unified MCP exposes Stage 2 read tools for symbol search, node lookup, callers, and callees while preserving Stage 1 status, index, sync, and files behavior.
- [x] #4 Indexing remains bounded by existing foreground file, byte, and wall-clock limits and does not write graph storage inside the source workspace.
- [x] #5 Focused CodeGraph and MCP tests cover the new Python extractor, repository query behavior, MCP tools, and Stage 1 regressions.
- [x] #6 Bandit on touched CodeGraph/MCP code and git diff whitespace checks are run before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-04-native-codegraph-python-search-implementation-plan.md

1. Add RED tests for Python AST extraction and repository graph persistence/search.
2. Implement deterministic CodeGraph node/edge/ref models, repository insert/query methods, and a stdlib-`ast` Python extractor.
3. Wire Python extraction into the bounded indexer while preserving Stage 1 inventory behavior for JS/TS and planned languages.
4. Add RED/GREEN MCP coverage for `codegraph.search`, `codegraph.node`, `codegraph.callers`, and `codegraph.callees`.
5. Run focused CodeGraph/MCP regression tests, Bandit on touched CodeGraph/MCP code, and `git diff --check`; then update this task and prepare the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED/GREEN evidence: extractor/repository tests first failed on missing CodeGraphNode/CodeGraphEdge/CodeGraphUnresolvedRef and missing PythonAstExtractor, then passed after implementation. Indexer tests first failed because Python files still had 0 graph nodes, then passed after wiring Python AST extraction. MCP tests first failed on missing Stage 2 tools, then passed after adding read tools and validation.

Verification: python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q passed with 43 passed, 5 warnings. Bandit touched scope wrote /tmp/bandit_codegraph_python_search.json with 0 results and 0 errors. git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the native CodeGraph Python extractor/search slice: Python files now index deterministic module/class/function/method/import nodes plus conservative same-file call edges or unresolved refs; repository graph persistence and query helpers support search, node lookup, callers, and callees without dangling relationship results; Unified MCP exposes the Stage 2 read tools while preserving bounded foreground indexing and Stage 1 inventory behavior for JS/TS/planned languages. Added the branch implementation plan and focused tests for extractor, repository, indexer, MCP tools, and protocol validation.
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

## Notes

<!-- SECTION:BASELINE_NOTES:BEGIN -->
- Baseline before Stage 2 edits: `python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q` passed with `31 passed`.
<!-- SECTION:BASELINE_NOTES:END -->
