---
id: TASK-12970
title: Certify Skills browser-extension parity and fix shell-specific regressions
status: In Progress
labels:
- skills
- extension
- webui
- uat
- accessibility
- reliability
priority: high
references:
- TASK-12969
- 'PR #2732'
documentation:
- Docs/Design/2026-07-15-skills-extension-parity-design.md
- Docs/Plans/IMPLEMENTATION_PLAN_skills_extension_parity_TASK_12970.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise the merged /skills beginner, power-user, accessibility, responsive, persistence, and recovery workflows through the built browser-extension options shell. Add deterministic extension Playwright coverage and fix only defects reproduced in the extension runtime; do not redesign the shared Skills UI or expand MCP/backend behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The built extension options shell opens the Skills route through its production router, authentication bootstrap, and capability gate.
- [ ] #2 A deterministic beginner journey covers Skills discovery and at least one complete create-or-seed, detail, dry-render/test, and use-in-chat workflow.
- [ ] #3 A deterministic power-user journey covers search, filters, sorting, URL or hash-backed state, row management, export, Trash, and reload persistence where supported by the extension router.
- [ ] #4 Extension-width keyboard, focus-return, dialog, drawer, touch-target, and horizontal-overflow behavior is verified.
- [ ] #5 Offline, API failure, cancellation, retry, and refresh recovery behavior is verified without stale results or lost drafts.
- [ ] #6 Extension Playwright coverage runs without unconditional skips and any production changes are limited to defects reproduced by that coverage.
- [ ] #7 Focused extension/shared-UI tests, TypeScript checks for touched scope, diff hygiene, and applicable security checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The five-stage TDD implementation plan passed independent plan review with no blocking issues. The accepted advisory requires explicit empty assertions for page errors, console errors, request failures, and unexpected API requests. The advisory to retain a deleted plan link was not adopted because repository guidance requires removing completed task plans; the link will be removed with the file during finalization.

2026-07-15: Stage 1 started. Implementing the pre-navigation options-page preparation hook and targeted initial route in the built-extension launcher.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
