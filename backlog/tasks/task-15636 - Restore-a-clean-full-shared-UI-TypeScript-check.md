---
id: TASK-15636
title: Restore a clean full shared UI TypeScript check
status: In Progress
created_date: 2026-11-24 18:18
labels:
- frontend
- typescript
- tests
references:
- https://github.com/rmusser01/tldw_server/pull/3607
updated_date: 2026-11-24 18:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the existing full-package TypeScript diagnostics while retaining the checked application and test sources and preserving product behavior and test assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The full shared UI TypeScript project, including test files, exits successfully with zero diagnostics.
- [x] #2 Changed test fixtures and assertions model current contracts, and the focused affected tests pass.
- [x] #3 The normal WebUI and extension checks still pass or have their pre-existing unrelated failures recorded with evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish the locked dependency baseline and remove the ignored query option.
2. Repair test fixtures by subsystem and correct the shared runtime boundary types.
3. Check all changed tests and the complete compiler project, then review the diff.
ADR required: no. The change restores conformance to the existing toolchain and contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed 234 ignored role query options and updated typed fixtures for real chat, history, knowledge, and service contracts. The worker preserves the exact stored refresh identity and validates hosted targets when storage is empty. TypeScript and Vitest share React Router resolution. A clean frozen Bun install resolved both Tailwind and the shared UI to PostCSS 8.5.9. ADR required: no. The existing frontend contracts and dependency tooling govern this repair.
The full shared UI command checked 9,614 files with zero diagnostics. WebUI typecheck, extension compile, and scoped ESLint passed. Across the 171-file run and focused rerun, all 6,197 affected cases passed. One existing dialog focus assertion failed in the combined run and passed with the entire 44-case suite in isolation. Eight router and stylesheet suites also passed. This TypeScript-only patch uses the frontend lint and test gates. Bandit applies to Python.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The complete shared UI TypeScript project now passes with its application and test files included. The implementation is verified and ready for review in draft PR #3607.
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
