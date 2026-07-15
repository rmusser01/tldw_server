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

2026-07-15: Stage 1 complete. Added targeted initial options routing and an awaited pre-navigation preparation hook in the built-extension launcher. Focused Vitest verification passed 11/11 tests. Specification review passed. Code-quality review found one test-isolation issue; it was corrected with fresh launcher mocks and re-review passed with no remaining findings.

2026-07-15: Stage 2 complete. Added the packaged-extension beginner Skills journey with fail-closed API guarding, bounded/redacted diagnostics, direct-fetch fallback, exact bootstrap fixtures, seed/details/dry-run/run/chat assertions, and context cleanup. The strict browser run reproduced an MV3 CSP defect in the options theme bootstrap; the inline script was moved to a synchronous same-origin public script with a focused 3/3 unit regression. The full Skills Manager suite reproduced five timing-dependent row-action tests; each now waits for the existing `1 skill` readiness signal and the owning suite passes 81/81. Final verification: beginner Playwright 1/1, shell 1/1, CSP unit 3/3, diff check clean. Specification review passed. Code-quality review found one immediate seed-request race; polling was added, fresh Playwright passed, and re-review reported no remaining findings. The implementation plan was corrected to run extension and shared-UI Vitest files from their owning package roots because the original cross-root command silently skipped UI files.

2026-07-15: Stage 3 started. Adding isolated power-user hash/filter/export and Trash contracts.
2026-07-15: Stage 3A complete. Added the packaged-extension power-user contract and minimally extended the shared fixture for normalized model filtering plus deterministic binary exports. The approved design was rechecked after an initially over-constrained reload-selection assertion; bulk export correctly remains pre-filter and no unnecessary selection-persistence product behavior was added. Focused power run passed 1/1; beginner plus power passed 2/2 with zero skips. Specification review found two evidence gaps (post-reload request freshness and completed aggregate contents); both were fixed with request-count evidence and nested ZIP validation. Quality review hardened exhaustive export-attempt logging, nested payload validity, and backend-equivalent model normalization. Final specification and quality re-reviews reported no remaining findings. Commits: c1c15c00d3, c840c77d08, ed95cd0f17, a4f8915b83.
2026-07-15: Stage 3B and Stage 3 complete. Added a fresh packaged-extension Trash workflow that moves summarize to Trash, verifies the immediate Undo affordance without activating it, restores from the durable Trash view, returns to Library, and proves exact delete/restore fixture state plus empty diagnostics. Focused Trash passed 1/1; the combined beginner, power-user, and Trash file passed 3/3 with zero skips. No fixture or production defect was reproduced. Specification and quality reviews reported no findings. Commit: bc95d6ccdf.
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
