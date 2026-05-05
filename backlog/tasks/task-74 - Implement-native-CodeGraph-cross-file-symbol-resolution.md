---
id: TASK-74
title: Implement native CodeGraph cross-file symbol resolution
status: Done
assignee: []
created_date: '2026-05-05 14:55'
updated_date: '2026-05-05 16:28'
labels:
  - codegraph
  - mcp
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1259'
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add conservative same-workspace cross-file reference resolution for the native CodeGraph MCP module. The slice should resolve Python and JavaScript/TypeScript import-driven calls across indexed files, keep deterministic edge IDs and stale cleanup behavior, and surface the relationships through existing CodeGraph read tools without adding new public tools or broad language-server semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import and unresolved-reference data can resolve conservative same-workspace cross-file relationships without arbitrary host-path access.
- [x] #2 Python and JavaScript/TypeScript imports resolve to indexed module/export symbols where the target file is inside the trusted workspace.
- [x] #3 Repository and indexer behavior keeps deterministic IDs and stale-edge cleanup correct when source or target files are re-indexed or deleted.
- [x] #4 Unified MCP caller/callee/impact/context results include newly resolved cross-file relationships while preserving existing output caps.
- [x] #5 Focused CodeGraph/MCP tests, Ruff, Bandit on touched production scope, and git diff whitespace checks pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-05-native-codegraph-cross-file-resolution-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline from clean origin/dev worktree after PR #1304 merge: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q -> 142 passed, 5 warnings.

Implemented repository resolved-reference state, conservative Python/JS/TS import binding resolution, indexer resolution counters, stale cleanup, and MCP read-tool coverage for cross-file relationships.

Verification: pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q -> 152 passed, 5 warnings.

Verification: ruff check touched CodeGraph/MCP scopes -> All checks passed.

Verification: bandit touched production scopes -> zero findings in /tmp/bandit_codegraph_cross_file_resolution.json.

Verification: git diff --check -> clean.

Known skips/blockers: none for this focused slice. TypeScript alias tests use existing parser availability guards when tree-sitter TypeScript is unavailable.

PR #1308 review-fix pass: addressing unresolved Qodo/Gemini comments on resolver bounds, case-sensitive lookups, batched writes, read-only list helpers, NULL-safe stale cleanup, and binding lookup complexity.

PR #1308 review fixes implemented: bounded resolver source scopes/deadlines/item caps, exact-case resolution lookups, batched edge/reference writes, read-only reference listing, O(1) direct binding lookup, and explicit NULL handling for resolved_edge stale cleanup/counting.

Review-fix verification: targeted regression tests for repository/resolver/indexer review issues -> 8 passed, 5 warnings.

Review-fix verification: focused CodeGraph/MCP suite -> 160 passed, 5 warnings.

Review-fix verification: Ruff touched scopes -> All checks passed; Bandit touched production scopes -> zero findings; git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added conservative same-workspace cross-file CodeGraph resolution and addressed PR #1308 review feedback. The resolver now runs within explicit foreground bounds, batches SQLite writes, uses exact-case resolution lookups, keeps listing read-only, handles NULL resolved_edge stale state, and exposes truncation/scan counters. Focused CodeGraph/MCP tests, Ruff, Bandit, and whitespace checks passed.
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
