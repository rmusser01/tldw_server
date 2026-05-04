---
id: TASK-38
title: Implement native CodeGraph JS/TS extractor slice
status: Done
assignee:
  - Codex
created_date: '2026-05-04 05:48'
updated_date: '2026-05-04 06:15'
labels:
  - codegraph
  - mcp
  - javascript
  - typescript
dependencies:
  - TASK-35
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1259'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-04-native-codegraph-js-ts-extractor-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next native CodeGraph epic slice after PR #1258: JavaScript/TypeScript/TSX extraction with conservative symbols, imports, exports, same-file calls, and trusted-workspace tsconfig/jsconfig path-alias resolution. Keep the slice focused: no context/impact tools, no Jobs mode, no C/C++/C#/Java/Kotlin extractors, and no full TypeScript compiler/type-check dependency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 JavaScript and TypeScript files index symbol nodes for functions, arrow-function variables, classes, methods, React-like components, interfaces, type aliases, and enums where supported by the parser.
- [x] #2 JS/TS import and re-export declarations create import nodes and unresolved refs for external package imports.
- [x] #3 Same-file calls by identifier or member expression are resolved conservatively or recorded as unresolved refs without false cross-file claims.
- [x] #4 Relative import targets and trusted tsconfig/jsconfig path aliases are resolved under the workspace root; escaping aliases are ignored with a clear unresolved reason.
- [x] #5 Indexer wires JS/TS extractors so TS/TSX/JS/JSX files no longer remain inventory-only when parser dependencies are available.
- [x] #6 Focused extractor, indexer, repository/MCP regression tests plus Ruff, Bandit, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm current Stage 3 requirements from the approved CodeGraph design and baseline tests. 2. Add RED extractor tests for JS, TS, TSX components, imports/exports, same-file calls, and parse failures. 3. Add RED resolver tests for relative imports and trusted/escaping tsconfig/jsconfig path aliases. 4. Implement a small parser loader plus JS/TS extractor modules using optional Tree-sitter dependencies with dependency-aware skip/error behavior. 5. Wire extractors into CodeGraphIndexer only when dependencies are available. 6. Add indexer/MCP regression coverage proving JS/TS files produce graph nodes/search results. 7. Run focused CodeGraph/MCP tests, Ruff touched files, Bandit touched CodeGraph/MCP scope, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-04-native-codegraph-js-ts-extractor-implementation-plan.md. Baseline focused CodeGraph/MCP tests passed with 53 passed and 5 warnings before Stage 3 work. Current shared venv is missing optional tree-sitter parser packages, so the plan explicitly gates implementation on installing/verifying .[codegraph] parser dependencies.

Installed pinned CodeGraph parser dependency set directly after .[codegraph] resolver conflict blocked full extra installation.

Task 1 red-green complete: added tree_sitter_loader tests, confirmed ModuleNotFoundError RED, implemented dynamic optional parser loader, and verified 5 passed.

Task 2 red-green complete: added resolver tests for relative imports, nearest config loading, frontend-style aliases, escaping aliases, and external package classification; verified 6 passed.

Task 3 red-green complete: added JavaScript extractor tests for modules, imports, re-exports, functions, arrow functions, classes, methods, JSX components, same-file calls, member-call unresolved refs, parse errors, and deterministic IDs; verified 3 passed.

Task 4 red-green complete: added TypeScript/TSX extractor tests for interface, type alias, enum, imports, function/class/method calls, TSX component detection, and deterministic IDs; verified 3 passed.

Task 5 red-green complete: indexer now wires optional JS/TS extractors, registry reports dependency-aware symbol extraction, MCP search finds indexed TSX components, and focused CodeGraph/MCP tests verified 72 passed.

Final verification: focused CodeGraph/MCP regression suite passed with 72 passed and 5 warnings; Ruff passed; Bandit JSON reported errors 0 and results 0; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the native CodeGraph JavaScript/TypeScript extractor slice. Added optional Tree-sitter loader, JS/TS import resolver with workspace-bounded path alias handling, JavaScript and TypeScript/TSX extractors, dependency-aware language metadata, and indexer/MCP wiring so JS/TS symbols are searchable. Verification: focused CodeGraph/MCP tests 72 passed, Ruff passed, Bandit reported errors 0/results 0, and git diff --check passed.
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
