---
id: TASK-63
title: Implement native CodeGraph C/C++ extractor slice
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 04:09'
updated_date: '2026-05-05 04:41'
labels:
  - codegraph
  - mcp
  - c
  - cpp
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1288'
  - 'https://github.com/rmusser01/tldw_server/pull/1293'
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
  - >-
    Docs/superpowers/plans/2026-05-05-native-codegraph-c-cpp-extractor-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next native CodeGraph language slice for C and C++ after the merged C# work. Keep the slice conservative: optional tree-sitter-c/tree-sitter-cpp dependency awareness, parser loader wiring, foundation metadata, indexer registration, symbol extraction for common declarations, same-file simple call capture where practical, and MCP search visibility. Exclude full compiler semantics, preprocessor evaluation, include path resolution, overload resolution, templates beyond basic symbol names, and cross-file semantic resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 C and C++ move from planned to dependency-aware foundation metadata with symbol_extraction tied to optional parser availability.
- [x] #2 C/C++ extraction indexes common includes/imports, functions, methods, structs/classes/enums/unions/namespaces where supported by the grammar, with deterministic IDs and conservative unresolved refs for includes and receiver/qualified calls.
- [x] #3 The indexer skips C/C++ files when optional parser dependencies are missing without blocking other languages or foreground limits.
- [x] #4 MCP codegraph.search can find indexed C/C++ symbols after codegraph.index in a fixture workspace.
- [x] #5 Focused loader, registry, extractor, indexer, and MCP tests cover C/C++ behavior plus dependency-missing paths; Ruff, Bandit on touched production scope, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify parser package APIs and add optional dependency metadata for tree_sitter_c and tree_sitter_cpp.
2. Add RED loader/registry/indexer/MCP tests for C/C++ foundation behavior and missing dependency handling.
3. Implement conservative C/C++ Tree-sitter extraction using existing CodeGraph node/edge models and shared helper patterns.
4. Wire extractors into the indexer and MCP visibility tests.
5. Run focused CodeGraph/MCP tests, Ruff, Bandit on touched production scope, and git diff --check before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline in fresh worktree from origin/dev after PR #1288 merge: CodeGraph plus MCP focused suite passed with 118 passed and 5 warnings.

Started implementation in /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codegraph-c-cpp-extractor on branch codex/codegraph-c-cpp-extractor. Applying TDD for parser metadata, loader, extractor, indexer, and MCP behavior.

Implemented C/C++ dependency probing, parser loader mappings, foundation language metadata, c_family extractor, indexer registration, MCP search coverage, and C/C++ regression tests. Local parser versions verified and installed in shared venv for tests: tree-sitter-c 0.24.2 and tree-sitter-cpp 0.23.4. Verification: focused C/C++ tests passed with 14 passed and 5 warnings; full CodeGraph plus MCP focused suite passed with 129 passed and 5 warnings; Ruff passed on touched CodeGraph/MCP/test scope; Bandit JSON at /tmp/bandit_codegraph_c_cpp.json reported errors 0 and results 0; git diff --check passed.

PR #1293 review-fix pass started: Qodo requested docstrings for _node_name, _function_name, and _declarator_name in c_family_extractor.py.

PR #1293 review fix: added docstrings to _node_name, _function_name, and _declarator_name. Verification: C/C++ extractor tests passed with 6 passed and 5 warnings; Ruff passed on c_family_extractor and C/C++ extractor tests; Bandit JSON at /tmp/bandit_codegraph_c_cpp_review_fixes.json reported errors 0 and results 0; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the native CodeGraph C/C++ extractor slice. C and C++ are now dependency-aware foundation languages backed by optional tree-sitter-c/tree-sitter-cpp packages. The slice adds conservative include/import, type, namespace, function/method, same-file simple call, indexer, and MCP search coverage while intentionally excluding compiler-semantic features such as preprocessor evaluation, include path resolution, overload resolution, and cross-file semantic resolution. Verification passed locally: focused C/C++ tests 14 passed and 5 warnings; full CodeGraph plus MCP focused suite 129 passed and 5 warnings; Ruff clean; Bandit errors 0 and results 0; git diff --check clean. PR: https://github.com/rmusser01/tldw_server/pull/1293

PR #1293 review follow-up: added docstrings for the new C-family extractor helper functions requested by Qodo.
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
