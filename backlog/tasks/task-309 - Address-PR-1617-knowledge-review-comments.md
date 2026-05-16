---
id: TASK-309
title: Address PR 1617 knowledge review comments
status: Done
assignee: []
created_date: '2026-05-13 01:27'
updated_date: '2026-05-13 01:57'
labels:
  - knowledge
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1617'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the still-actionable review findings on PR #1617 for the /knowledge QA source workflow branch. Scope is limited to the PR review comments, preserving /knowledge as QA-only and keeping source-contract behavior aligned across backend, WebUI, and extension surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All valid unresolved inline review threads on PR #1617 are fixed or explicitly explained if not applicable.
- [x] #2 Knowledge QA localization and source label translation regressions are corrected without changing the product scope.
- [x] #3 Backend retrieval changes preserve prompts, world books, dictionaries, workspace filtering, source-status diagnostics, and consistent behavior across search variants.
- [x] #4 Frontend and extension review fixes are covered by focused tests where practical.
- [x] #5 Focused backend, frontend, extension, whitespace, and Bandit verification pass after the fixes are committed and pushed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented review fixes for PR #1617: localized knowledge/sidebar source picker strings and source labels, restored source picker item IDs, hardened saved profile ID validation/dependencies, preserved streaming source_status diagnostics, added extension import error context, wired prompts DB across stream/batch/resume endpoints, made new retrievers avoid blocking sync DB calls, centralized full source db path mapping, scoped cache namespace by workspace, and reapplied workspace artifact filtering/source_status after PRF/decomposition/final document mutation.

Verification passed: focused RAG review tests, compileall on touched Python files, UI Vitest package tests for KnowledgeContextBar/KnowledgeQAProvider/KnowledgePanelTabRouting/sourceMetadata, tldw-frontend knowledge route parity test, extension copilot entrypoint unit test, git diff --check, and Bandit on touched Python production files. The branch was force-pushed as a single amended commit and the addressed current PR review threads were resolved.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1617 review feedback by localizing reviewed UI labels, hardening saved source-profile behavior, preserving streaming source diagnostics, adding extension import error context, completing prompts DB wiring across unified RAG variants, moving synchronous retriever I/O into threadpool calls, tightening retriever scoring/adapter setup, centralizing source DB path mapping, and making workspace filtering/cache/source-status behavior consistent through final pipeline output.

Focused backend, frontend, extension, whitespace, compile, and Bandit verification passed locally. Remote CI was still pending immediately after the force push and is tracked on PR #1617 rather than blocked by a local failure.
<!-- SECTION:FINAL_SUMMARY:END -->
