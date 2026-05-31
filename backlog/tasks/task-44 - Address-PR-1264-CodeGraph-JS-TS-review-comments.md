---
id: TASK-44
title: 'Address PR #1264 CodeGraph JS/TS review comments'
status: Done
assignee:
  - Codex
created_date: '2026-05-04 15:01'
updated_date: '2026-05-04 15:08'
labels:
  - codegraph
  - mcp
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1264'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #1264 for the native CodeGraph JavaScript/TypeScript extractor slice. Scope includes keeping optional parser and extractor failures contained, simplifying shared JS/TS helpers, handling invalid JSONC config gracefully, avoiding unnecessary TypeScript gating on TSX availability, and recording focused verification before updating the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Invalid tsconfig/jsconfig content does not abort JS/TS extraction or indexing.
- [x] #2 Optional Tree-sitter import-time failures are reported as unavailable parser results instead of raising during availability checks.
- [x] #3 TypeScript extraction is registered when the TypeScript parser is available even if TSX parser availability is false.
- [x] #4 Shared JS/TS extractor helpers no longer require cross-module imports of underscore-prefixed symbols.
- [x] #5 New review-fix behavior has focused regression tests and the CodeGraph test subset passes.
- [x] #6 Ruff, Bandit on touched production scope, and git diff whitespace checks pass before pushing the PR update.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved PR #1264 review feedback by adding regression coverage for invalid JSONC config fallback, optional parser import errors, non-ValueError extractor failures, TypeScript registration without TSX parser availability, and one-config-load-per-JS-file behavior. Implementation updates keep optional parser/import failures structured, cache project config per JS/TS extracted file, publicize the JS graph builder helpers consumed by the TypeScript extractor, and keep index runs alive when one extractor raises an expected boundary exception.

Verification: focused red tests failed before implementation and passed after; CodeGraph/MCP subset passed with 77 passed and 5 warnings; Ruff passed; Bandit JSON reported errors 0 and results 0; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all actionable PR #1264 review items for the CodeGraph JS/TS extractor slice. The PR now handles invalid JS/TS project config and optional native parser failures gracefully, registers TypeScript extraction without requiring TSX availability, avoids repeated config parsing per import, removes cross-module private helper imports, and records the review-fix verification.
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
