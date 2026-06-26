---
id: TASK-12046
title: Sanitize STT playground action error notifications
status: Done
created_date: 2026-06-26 07:12
labels:
- webui
- audio
- stt
- raw-error
references:
- TASK-418.8.2
- TASK-431
- TASK-420
- TASK-12045
documentation:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage17-stt-action-error-sanitization-plan.md
- apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
- apps/packages/ui/src/utils/server-error-message.ts
- apps/packages/ui/src/utils/__tests__/server-error-message.test.ts
updated_date: 2026-06-26 07:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI raw-error follow-up for STT playground action notifications. Preserve the existing STT recording, comparison, save, and history workflows, but prevent save-to-notes and history recording-load failures from exposing raw endpoint paths, filesystem paths, or inline secrets in user-facing notification descriptions and console output where applicable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 STT save-to-notes failures show sanitized user-facing notification descriptions instead of raw endpoint/path/secret details.
- [x] #2 STT history recording-load failures use the same sanitized formatter.
- [x] #3 Existing STT page readiness, preset apply, recording strip wiring, and comparison/history component mounting behavior continue to work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Create a stage-specific plan document for the STT action-notification cleanup.', 'Add failing SttPlaygroundPage regressions that trigger save-to-notes and history load failures with raw endpoint/path/secret details and expect sanitized notification copy.', 'Add a focused shared sanitizer regression for token-like secrets, then extend the shared server-error sanitizer to redact secrets while preserving endpoint/path redaction.', 'Wire STT save-to-notes and recompare recording-load failures through the shared sanitizer.', 'Run focused STT page and sanitizer tests, lint touched TS/TSX files, whitespace checks, and record Bandit applicability.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Scope refinement after inspection: STT can reuse the existing shared `sanitizeServerErrorMessage` utility for action notification descriptions. That utility already redacts endpoints and filesystem paths, but it does not yet redact token-like secrets, so this slice now includes a direct utility regression and a minimal shared sanitizer enhancement.
TDD/verification notes:
- RED: `bun run test:run ../packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx -t "sanitizes"` failed because save-to-notes and history recording-load notification descriptions still exposed raw `/api/v1/...` paths, local filesystem paths, and secret-like tokens.
- RED: `bun run test:run ../packages/ui/src/utils/__tests__/server-error-message.test.ts -t "redacts token-like secrets"` failed because `sanitizeServerErrorMessage` did not redact `token=...`, `api_key=...`, or `Bearer ...` secrets.
- GREEN: targeted STT sanitization tests passed after routing the STT action failures through the shared sanitizer.
- GREEN: targeted shared sanitizer secret-redaction test passed after extending `sanitizeServerErrorMessage`.
- GREEN: full focused test run passed: `SttPlaygroundPage.test.tsx` and `server-error-message.test.ts`, 20 tests.
- Lint: direct ESLint on touched STT and sanitizer files exited 0 after cleanup; only the known Next pages-directory notice was printed.
- Whitespace: `git diff --check` passed.
- Design-state verifier: `bun run verify:design-system-state` remains blocked by local `ERR_MODULE_NOT_FOUND: Cannot find package 'typescript'` from `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
- Bandit: not applicable; this slice touched TS/TSX, TS tests, Markdown, and Backlog metadata only.
Final verification refresh:
- After lint-warning cleanup, full focused Vitest still passed: `SttPlaygroundPage.test.tsx` and `server-error-message.test.ts`, 20 tests.
- After lint-warning cleanup, direct ESLint on touched files exited 0 with only the known Next pages-directory notice.
- After lint-warning cleanup, `git diff --check` passed.
- Re-ran `bun run verify:design-system-state`; it still fails before analysis because local `typescript` is missing from the UI package script dependency resolution.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routed STT playground save-to-notes and history recording-load failure notifications through the shared server error sanitizer so user-facing descriptions redact API endpoints, filesystem paths, and secret-like tokens. Extended `sanitizeServerErrorMessage` to redact `token=...`, `api_key=...`, `secret=...`, Bearer tokens, and common `sk_`/`sk-` style keys, with direct utility coverage plus STT page regressions. Existing STT readiness, preset, recording-strip, comparison, and history mounting behavior remains covered by the focused STT page suite.
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
