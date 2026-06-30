---
id: TASK-49
title: Implement native CodeGraph Java/Kotlin extractor slice
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 00:52'
updated_date: '2026-05-05 03:20'
labels:
  - codegraph
  - mcp
  - java
  - kotlin
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1270'
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
  - 'https://github.com/rmusser01/tldw_server/pull/1277'
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
  - >-
    Docs/superpowers/plans/2026-05-05-native-codegraph-java-kotlin-extractor-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next Stage 5 native CodeGraph language slice after the merged context/impact tools: Java and Kotlin extractor support with conservative package/class/function/method/import symbols, same-file call edges, dependency-aware registry/indexer wiring, and focused MCP search coverage. Keep the slice narrow: no C#, no C/C++, no Jobs mode, no full type resolution or build-system-aware classpath analysis.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Java files index package, class/interface, method/constructor, and import nodes with deterministic IDs and conservative same-file call or unresolved references.
- [x] #2 Kotlin files index package, class/object/interface, function, and import nodes with deterministic IDs and conservative same-file call or unresolved references.
- [x] #3 Language registry and indexer expose Java/Kotlin as symbol-extraction capable only when optional Tree-sitter dependencies are available; otherwise they remain visible with dependency_missing metadata and safe skip/failure behavior.
- [x] #4 Focused extractor, loader, registry, indexer, and MCP search tests cover Java/Kotlin behavior and dependency-missing paths.
- [x] #5 Focused CodeGraph/MCP tests, Ruff, Bandit on touched production scope, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a focused Java/Kotlin implementation plan under Docs/superpowers/plans and verify current parser dependency state.
2. Add RED loader/registry tests for Java/Kotlin optional parser support and dependency-missing metadata.
3. Add RED extractor tests for conservative Java and Kotlin symbols/imports/calls.
4. Implement shared JVM-family helper code plus Java/Kotlin extractors using Tree-sitter when dependencies are present.
5. Wire registry/indexer/MCP search behavior and run focused verification before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created focused Stage 5 implementation plan at Docs/superpowers/plans/2026-05-05-native-codegraph-java-kotlin-extractor-implementation-plan.md. Local venv currently has tree_sitter but not tree_sitter_java/tree_sitter_kotlin; dependency install/verification is the first implementation gate.

Task 1 dependency gate complete: installed tree-sitter-java 0.23.5 and tree-sitter-kotlin 1.1.0 into the shared venv, verified both expose language(), added loader mappings and optional dependency bounds, and loader tests pass with 9 passed.

Task 3 Java extractor complete: added shared JVM helpers plus Java Tree-sitter extraction for package/import/type/method/constructor nodes, same-file bare method call resolution, unresolved imports/receiver calls, parse errors, and deterministic node IDs. Verified with python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_java_extractor.py -q (3 passed).

Task 4 Kotlin extractor complete: added Kotlin Tree-sitter extraction for package/import/class/object/interface/function nodes, same-file simple function call resolution, unresolved imports/receiver calls, parse errors, and deterministic node IDs. Verified with python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_kotlin_extractor.py -q (3 passed).

Task 5 registry/indexer wiring complete: promoted Java/Kotlin to foundation metadata with dependency_missing tracking, added dependency probe coverage, exported/register Java/Kotlin extractors when parsers are available, and added indexer coverage for persisted Java/Kotlin graph rows. Verified with python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py::test_indexer_extracts_java_kotlin_graph_rows_during_index -q (7 passed) and python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py -q (18 passed).

Task 6 MCP regression complete: added codegraph.index plus codegraph.search coverage for Java method and Kotlin function symbols. No MCP implementation changes were needed because the existing module delegates through the registry/indexer/repository path. Verified with python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py::test_codegraph_search_finds_java_kotlin_symbols_after_index -q (1 passed).

Final verification: python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q passed with 108 passed and 5 warnings. Ruff passed on touched CodeGraph/MCP scope. Bandit JSON at /tmp/bandit_codegraph_java_kotlin.json reported errors [] and results []. git diff --check exited 0 with no output. Dependency versions verified in the shared venv: tree-sitter-java 0.23.5 and tree-sitter-kotlin 1.1.0. Known limits remain intentional: no classpath, build-system, overload, inheritance, or full type-resolution semantics in this slice.

PR #1277 review-fix pass started. Active actionable items: avoid counting non-extractable optional JVM files in foreground size guards, keep dependency_available tied to core availability rather than all optional parser packages, preserve Java/Kotlin import syntax more faithfully, add type visibility for Java/Kotlin declarations, and harden Kotlin interface/navigation parsing.

PR #1277 review-fix pass completed. Addressed active review issues: core dependency health no longer fails when optional Java/Kotlin parsers are missing; non-extractable JVM files are skipped before foreground file/byte guards with dependency_missing_language_skipped; Java imports preserve static and wildcard syntax and type declarations record visibility; Kotlin imports preserve alias and wildcard syntax, type declarations record visibility, interface detection no longer depends on source-text prefix, and nested navigation call refs preserve the full expression.

Review-fix verification: focused review regressions passed (4 passed); focused config/indexer/Java/Kotlin extractor suite passed (31 passed); full CodeGraph plus MCP module/dynamic catalog suite passed (110 passed); Ruff passed on touched CodeGraph/MCP/test scope; Bandit on touched production scope reported 0 results and 0 errors in /tmp/bandit_codegraph_java_kotlin_review_fixes.json; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the native CodeGraph Java/Kotlin extractor slice. Added optional parser dependency pins and loader support, shared JVM Tree-sitter helpers, Java and Kotlin extractors, dependency-aware language registry/indexer wiring, and MCP search regression coverage. The implementation stays conservative: it extracts package/import/type/function or method symbols and same-file simple call edges while leaving imports, receiver calls, constructor calls without modeled targets, and build-system/type-resolution cases unresolved for later slices.

PR: https://github.com/rmusser01/tldw_server/pull/1277

PR #1277 review follow-up: fixed optional dependency health semantics, skipped unavailable JVM language files before foreground guard accounting, and tightened Java/Kotlin extraction to preserve import syntax, declaration visibility, Kotlin interface detection, and nested navigation call references.
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
