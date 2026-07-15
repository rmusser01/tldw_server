---
id: TASK-12970
title: Certify Skills browser-extension parity and fix shell-specific regressions
status: Done
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
modified_files:
- Docs/Design/2026-07-15-skills-extension-parity-design.md
- apps/extension/entrypoints/options/index.html
- apps/extension/package.json
- apps/extension/tests/e2e/skills.parity.spec.ts
- apps/extension/tests/e2e/utils/extension-build.test.ts
- apps/extension/tests/e2e/utils/extension-build.ts
- apps/extension/tests/unit/options-theme-bootstrap.test.ts
- apps/extension/tests/unit/skills-fixture-request-contract.test.ts
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
- apps/packages/ui/src/public/theme-bootstrap.js
- apps/tldw-frontend/e2e/utils/skills-fixtures.ts
- backlog/tasks/task-12970 - Certify-Skills-browser-extension-parity-and-fix-shell-specific-regressions.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise the merged /skills beginner, power-user, accessibility, responsive, persistence, and recovery workflows through the built browser-extension options shell. Add deterministic extension Playwright coverage and fix only defects reproduced in the extension runtime; do not redesign the shared Skills UI or expand MCP/backend behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The built extension options shell opens the Skills route through its production router, authentication bootstrap, and capability gate.
- [x] #2 A deterministic beginner journey covers Skills discovery and at least one complete create-or-seed, detail, dry-render/test, and use-in-chat workflow.
- [x] #3 A deterministic power-user journey covers search, filters, sorting, URL or hash-backed state, row management, export, Trash, and reload persistence where supported by the extension router.
- [x] #4 Extension-width keyboard, focus-return, dialog, drawer, touch-target, and horizontal-overflow behavior is verified.
- [x] #5 Offline, API failure, cancellation, retry, and refresh recovery behavior is verified without stale results or lost drafts.
- [x] #6 Extension Playwright coverage runs without unconditional skips and any production changes are limited to defects reproduced by that coverage.
- [x] #7 Focused extension/shared-UI tests, TypeScript checks for touched scope, diff hygiene, and applicable security checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The five-stage TDD implementation plan passed independent plan review with no blocking issues. The accepted advisory requires explicit empty assertions for page errors, console errors, request failures, and unexpected API requests. The advisory to retain a deleted plan link was not adopted because repository guidance requires removing completed task plans; the link will be removed with the file during finalization.

2026-07-15: Stage 1 complete. Added targeted initial options routing and an awaited pre-navigation preparation hook in the built-extension launcher. Focused Vitest verification passed 11/11 tests. Specification review passed. Code-quality review found one test-isolation issue; it was corrected with fresh launcher mocks and re-review passed with no remaining findings.

2026-07-15: Stage 2 complete. Added the packaged-extension beginner Skills journey with fail-closed API guarding, bounded/redacted diagnostics, direct-fetch fallback, exact bootstrap fixtures, seed/details/dry-run/run/chat assertions, and context cleanup. The strict browser run reproduced an MV3 CSP defect in the options theme bootstrap; the inline script was moved to a synchronous same-origin public script with a focused 3/3 unit regression. The full Skills Manager suite reproduced five timing-dependent row-action tests; each now waits for the existing `1 skill` readiness signal and the owning suite passes 81/81. Final verification: beginner Playwright 1/1, shell 1/1, CSP unit 3/3, diff check clean. Specification review passed. Code-quality review found one immediate seed-request race; polling was added, fresh Playwright passed, and re-review reported no remaining findings. The implementation plan was corrected to run extension and shared-UI Vitest files from their owning package roots because the original cross-root command silently skipped UI files.

2026-07-15: Stage 3 started. Adding isolated power-user hash/filter/export and Trash contracts.
2026-07-15: Stage 3A complete. Added the packaged-extension power-user contract and minimally extended the shared fixture for normalized model filtering plus deterministic binary exports. The approved design was rechecked after an initially over-constrained reload-selection assertion; bulk export correctly remains pre-filter and no unnecessary selection-persistence product behavior was added. Focused power run passed 1/1; beginner plus power passed 2/2 with zero skips. Specification review found two evidence gaps (post-reload request freshness and completed aggregate contents); both were fixed with request-count evidence and nested ZIP validation. Quality review hardened exhaustive export-attempt logging, nested payload validity, and backend-equivalent model normalization. Final specification and quality re-reviews reported no remaining findings. Commits: c1c15c00d3, c840c77d08, ed95cd0f17, a4f8915b83.
2026-07-15: Stage 3B and Stage 3 complete. Added a fresh packaged-extension Trash workflow that moves summarize to Trash, verifies the immediate Undo affordance without activating it, restores from the durable Trash view, returns to Library, and proves exact delete/restore fixture state plus empty diagnostics. Focused Trash passed 1/1; the combined beginner, power-user, and Trash file passed 3/3 with zero skips. No fixture or production defect was reproduced. Specification and quality reviews reported no findings. Commit: bc95d6ccdf.
2026-07-15: Stage 4A complete. Added the packaged-extension compact-width keyboard, focus-return, touch-target, and horizontal-overflow contract. It reproduced a real rapid-close defect: rc-drawer can unmount before its opening animation reports visible, so its normal post-close lifecycle never restores the trigger. The shared Manager now captures the exact details trigger and supplies a single state-driven focus fallback whose cleanup cancels any pending restore when details reopen; the temporary custom drawer after-close path was removed to avoid stale callback races. Focused coverage proves normal close and immediate close/reopen behavior, and the compact locators are exact. Verification: focused Manager 2 passed/81 skipped, SkillDetailsDrawer 4/4, compact extension 1/1, full extension parity 4/4 with zero skips, production Chrome build passed, and diff check clean. The prior 82-test full Manager suite passed before the new race regression was added; the current 83-test aggregate suite remains a Stage 5 release gate. Specification and quality reviews reported no findings. Commits: 9e442679ae, 7f09fc1f65.
2026-07-15: Stage 4B complete. Added deterministic packaged-extension list loading, one automatic retry, manual Try again success, redacted primary/expanded diagnostics, and unreachable connection-gate coverage. The fixture holds request one, returns exact 503 failures for requests one and two, succeeds only on request three, and rejects extra requests. Chromium's expected 503 console message is excluded only when both its full text and exact Skills list URL match in this single test; all four diagnostic arrays remain strict. Quality review identified that the background connection poller could restore connected state after five seconds. A focused RED proved checkOnce removed the unreachable UI; the disposable test context now pins checkOnce before forcing error_unreachable, and the browser test invokes it directly to prove stability. Verification: focused recovery 1/1, full current parity 5/5 with zero skips, and diff check clean. Specification review passed; quality re-review reported no remaining findings. Commits: f324eb4c9c, 873cf9a966.
2026-07-15: Stage 4C and Stage 4 complete. Added a fresh unseeded packaged-extension session draft contract. It captures the initialized summarizer template baseline, edits unique Name and Instructions values, proves the versioned sessionStorage draft entry, reloads the same #/skills tab, verifies the recovery notice and values, discards the recovered draft, proves storage removal and baseline restoration, closes without a discard confirmation, and reopens clean. Initial focused and full runs passed. Quality review identified baseline-readiness and persistence-evidence gaps; exact empty-state/template waits plus read-only storage polling resolved them. Final verification: focused draft 1/1, full extension parity 6/6 with zero skips, and diff check clean. Specification review passed; quality re-review reported no remaining findings. Commits: d8f35ca3ab, 7c80ebf945.

2026-07-15: Stage 5 package entry points complete. Added focused and strict one-worker Skills parity scripts pinned to the deterministic mock origin. Quality review requested existing `cross-env` usage and deletion of both JSON report locations before each strict run; both were implemented and re-review approved with no remaining findings. Verification: focused script 6/6, strict script 6/6, strict gate `passed=6 skipped=0 unexpected=0 flaky=0`, byte-identical ignored source/copied reports, and clean diff. Commits: 107ae02336, 3664b171e0.

2026-07-15: Final full-diff review found two evidence gaps. The shared path-only fixtures did not fail on wrong origin/method/auth contracts, and the theme source test allowed a no-op external script. Commit cdb801d231 added an opt-in exact request contract for extension-used Skills/capability routes, including all three recovery requests, while preserving default WebUI callers and the intentionally public no-auth docs-info route. It also executes the shipped theme bootstrap and proves dark-theme application precedes simulated app code. Validator TDD: initial RED 6/6 failed for missing validator, GREEN 6/6; public-contract RED 1 failed/6 passed, GREEN 7/7. Focused units 11/11, strict extension parity 6/6 with zero skips/unexpected/flaky, compile and diff check passed. Existing WebUI regression remained 13 mocked passed with three unavailable live-server skips. Specification re-review and original quality re-review approved with no remaining findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Certified the shared `/skills` experience in the packaged Chrome extension with six deterministic, skip-free contracts covering bootstrap/beginner use, power-user filtering and export, Trash, compact keyboard/focus/accessibility, retry and unreachable recovery, and session draft recovery. Added the targeted pre-navigation launcher hook, exact extension request contracts, MV3-safe synchronous theme bootstrap, the reproduced details-drawer focus-return fix, and focused/strict package entry points. Final verification: extension Vitest 15/15, shared Skills UI Vitest 125/125, WebUI mocked Skills Playwright 13/13, production Chrome build passed, strict extension parity 6/6 with `skipped=0 unexpected=0 flaky=0`, extension compile passed, report copies were byte-identical/ignored, and diff hygiene passed. Three pre-existing live-server WebUI checks skipped because no backend was running; they are not counted as parity evidence. The build retained existing duplicate-import, circular-chunk, bundle-size, and stale Browserslist warnings. Bandit is not applicable because no Python files changed. Final specification and quality reviews approved with no remaining findings.
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
