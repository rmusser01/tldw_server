---
id: TASK-64
title: 'Address PR #1293 C/C++ parser guard review comments'
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 05:12'
updated_date: '2026-05-05 05:12'
labels:
  - codegraph
  - mcp
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1293'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the remaining PR #1293 review comments by adding parser-availability skip guards for optional C/C++ positive-path tests in CodeGraph indexer, extractor, and MCP coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 C and C++ extractor test modules skip gracefully when their optional parser package cannot be loaded.
- [x] #2 Indexer and MCP positive-path C/C++ tests skip gracefully when either optional C/C++ parser is unavailable.
- [x] #3 Focused tests and lint verification are recorded before pushing the PR update.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review-fix pass started for CodeRabbit parser-availability comments.

Added parser-availability guards using CodeGraph load_parser for C, C++, indexer, and MCP positive-path tests. Verification: focused parser guard pytest passed with 9 passed and 5 warnings; Ruff passed on touched test files; Bandit on touched test scope with B101 skipped reported errors 0 and results 0 at /tmp/bandit_codegraph_c_cpp_parser_guard_tests.json; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1293 parser-availability review comments by adding load_parser-based skips to the C/C++ extractor test modules and the C/C++ indexer/MCP positive-path tests. This keeps optional tree-sitter-c/tree-sitter-cpp dependency absence as a skip instead of a failure. Verification passed: focused pytest 9 passed and 5 warnings; Ruff clean; Bandit errors 0/results 0 with B101 skipped for test asserts; git diff --check clean.
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
