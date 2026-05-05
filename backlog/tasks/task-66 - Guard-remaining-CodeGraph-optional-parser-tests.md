---
id: TASK-66
title: Guard remaining CodeGraph optional parser tests
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 05:25'
updated_date: '2026-05-05 05:27'
labels:
  - codegraph
  - mcp
  - test-hardening
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish optional parser test hardening by adding load_parser-based skip guards for remaining parser-dependent CodeGraph positive tests covering JavaScript, TypeScript/TSX, and C# extraction, indexing, and MCP search paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 JavaScript extractor/indexer/MCP positive tests skip gracefully when tree-sitter-javascript cannot be loaded.
- [x] #2 TypeScript/TSX extractor/indexer/MCP positive tests skip gracefully when tree-sitter-typescript cannot be loaded.
- [x] #3 C# extractor/indexer/MCP positive tests skip gracefully when tree-sitter-c-sharp cannot be loaded.
- [x] #4 Focused CodeGraph tests, Ruff, Bandit on touched test scope, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1295 merged. This completes the optional parser guard pattern for the remaining JS/TS and C# parser-dependent positive tests.

Added load_parser-based skip guards for JavaScript, TypeScript/TSX, and C# extractor modules plus indexer and MCP positive-path tests. Verification: focused parser-dependent tests passed with 15 passed and 5 warnings; Ruff passed on touched test files; Bandit on touched test scope with B101 skipped reported errors 0 and results 0 at /tmp/bandit_codegraph_remaining_parser_guards.json; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed remaining optional parser guard hardening for CodeGraph tests. JavaScript, TypeScript/TSX, and C# parser-dependent extractor, indexer, and MCP positive tests now skip when their optional parser package cannot be loaded. Verification passed: focused pytest 15 passed and 5 warnings; Ruff clean; Bandit errors 0/results 0 with B101 skipped for test asserts; git diff --check clean.
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
