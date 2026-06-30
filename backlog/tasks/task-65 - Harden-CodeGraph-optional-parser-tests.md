---
id: TASK-65
title: Harden CodeGraph optional parser tests
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 05:18'
updated_date: '2026-05-05 05:20'
labels:
  - codegraph
  - mcp
  - test-hardening
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make CodeGraph parser-dependent positive tests skip gracefully when optional tree-sitter language packages are unavailable, matching the optional .[codegraph] dependency contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Java and Kotlin extractor/indexer/MCP tests skip gracefully when their optional parsers cannot be loaded.
- [x] #2 Tree-sitter loader parser smoke tests skip gracefully per optional parser instead of hard-failing on missing packages.
- [x] #3 Focused CodeGraph tests, Ruff, Bandit on touched test scope, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1293 merged. This applies the same parser-availability guard pattern to remaining optional-parser positive tests.

Implemented load_parser-based skip guards for Java/Kotlin extractor modules, Java/Kotlin indexer/MCP positive-path tests, and all Tree-sitter loader parser smoke tests. Added a regression check that the loader smoke helper raises pytest skip when an optional parser dependency is missing. Verification: focused parser tests passed with 21 passed and 5 warnings; Ruff passed on touched files; Bandit on touched test scope with B101 skipped reported errors 0 and results 0 at /tmp/bandit_codegraph_parser_test_guards.json; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened CodeGraph optional parser tests after the C/C++ slice merged. Parser-dependent Java/Kotlin extractor, indexer, and MCP tests now skip when optional parser packages cannot be loaded, and loader parser smoke tests use a shared per-language skip helper. Verification passed: focused pytest 21 passed and 5 warnings; Ruff clean; Bandit errors 0/results 0 with B101 skipped for test asserts; git diff --check clean.
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
