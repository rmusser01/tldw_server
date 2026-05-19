---
id: TASK-31
title: Address PR 1253 CodeRabbit docstring review
status: Done
assignee: []
created_date: '2026-05-04 03:06'
updated_date: '2026-05-04 03:16'
labels:
  - codegraph
  - mcp
  - review
dependencies:
  - TASK-27
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1253'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable PR #1253 review feedback from CodeRabbit. The review covered duplicate Backlog section markers, Python attribute-call false edges, atomic file plus graph persistence, extraction failure status reporting, MCP search normalization, ambiguous node selector validation, and docstring coverage. The PR description Change summary gate is intentionally left for the human requester because repository policy requires a human-written summary.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All six actionable CodeRabbit inline comments on PR 1253 are addressed in code tests or Backlog task metadata.
- [x] #2 Changed CodeGraph production code has focused docstrings and local AST docstring coverage remains above the reviewer threshold.
- [x] #3 Python attribute calls do not create false same-file call edges and are represented conservatively as unresolved refs.
- [x] #4 Indexer file inventory and graph replacement happen in one repository transaction and extraction failures persist a non-indexed status with run errors.
- [x] #5 MCP search and node selector validation normalizes whitespace and rejects ambiguous node_id plus symbol selectors.
- [x] #6 Focused CodeGraph/MCP tests Bandit touched scope and git diff whitespace check pass after the review fixes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #1253 review threads: renamed duplicate TASK-27 section markers; kept Python attribute calls unresolved instead of resolving by bare attribute name; added atomic file plus graph repository persistence; persisted extraction_failed file status and run error summaries; normalized MCP search and selector inputs; rejected node_id plus symbol ambiguity; added focused docstrings for changed production code.

Verification: focused CodeGraph/MCP pytest suite passed with 47 passed and 5 warnings; AST docstring inspection reported 111 definitions with 0 missing docstrings; Bandit touched scope reported 0 results and 0 errors; git diff --check origin/dev...HEAD passed.

Known non-code blocker: PR description Change summary still requires a human-written summary per repository policy.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all actionable PR #1253 CodeRabbit inline comments with focused CodeGraph fixes and tests. Attribute calls now remain unresolved instead of false-linking to same-file functions; file inventory and graph replacement persist in one repository transaction; extraction failures are reported as extraction_failed with run errors; MCP search and node selectors normalize/reject ambiguous inputs; duplicate Backlog markers were made unique; changed production definitions have docstrings. Verification passed locally.
<!-- SECTION:FINAL_SUMMARY:END -->
