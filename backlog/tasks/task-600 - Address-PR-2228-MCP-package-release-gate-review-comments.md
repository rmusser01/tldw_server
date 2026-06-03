---
id: TASK-600
title: Address PR 2228 MCP package release gate review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:32'
labels:
  - mcp-unified
  - packaging
  - pr-review
dependencies: []
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/2228'
  - >-
    Docs/superpowers/plans/2026-06-03-mcp-unified-package-release-readiness-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address still-valid PR #2228 review feedback after rebasing onto latest dev. Scope: dependency metadata version drift, package-name dependency checks, long-line style cleanup, and brittle test repo-root lookup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2228 is rebased onto latest origin/dev without losing TASK-599 work.
- [x] #2 Still-valid Qodo/Gemini review comments are fixed with minimal changes; invalid items are documented with a reason.
- [x] #3 Focused tests cover metadata dependency names, CLI/docs visibility, and import-boundary behavior after the fixes.
- [x] #4 Bandit on touched Python, focused pytest, and git diff --check pass before updating the PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2228 onto latest origin/dev and addressed all still-valid Qodo/Gemini comments. Dependency metadata now uses a names-only policy instead of duplicating pyproject version floors, and package-info/docs expose that policy. Tests now validate package names rather than substring matches, new assertions are PEP8-formatted, and docs/pyproject path lookup uses a repo-root helper instead of fixed parent depth. Verification passed: 91 focused tests, Bandit zero findings, and git diff --check.
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
