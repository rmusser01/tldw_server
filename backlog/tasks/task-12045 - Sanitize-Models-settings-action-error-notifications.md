---
id: TASK-12045
title: Sanitize Models settings action error notifications
status: Done
created_date: 2026-06-26 07:00
labels:
- webui
- capability-state
- models
references:
- TASK-420
- TASK-418.10.4
- TASK-12044
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage16-models-action-error-sanitization-plan.md
- apps/packages/ui/src/components/Option/Models/index.tsx
- apps/packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx
updated_date: 2026-06-26 07:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred /settings/model capability-error follow-up for action notifications. Keep the Models settings route behavior unchanged, but prevent refresh and OpenAI OAuth action failures from exposing raw endpoint paths, filesystem paths, or inline secrets in user-facing notification descriptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Models refresh failures show a sanitized notification description instead of raw endpoint/path/secret details.
- [x] #2 OpenAI OAuth action failures use the same sanitized user-facing formatter.
- [x] #3 Existing Models settings defaults, provider readiness, and catalog rendering behavior continue to work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Create a stage-specific plan document for the Models action-notification cleanup.', 'Add a failing ModelsBody regression that triggers a refresh failure with raw endpoint/path/secret details and expects sanitized notification copy.', 'Implement a small local formatter for Models action error notifications and reuse it for refresh and OpenAI OAuth actions.', 'Run the focused ModelsBody tests, lint touched TS/TSX files, and diff checks.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD/verification notes:
- RED: `bun run test:run ../packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx -t "sanitizes refresh failure notifications"` failed because the notification description still contained `/api/v1/llm/models/metadata`, `sk_secret_inline`, and `/Users/alice/...`.
- RED follow-up: after notification sanitization, the same test failed because `console.error` still logged the raw error details.
- GREEN: targeted regression passed after using the sanitized formatter for the notification and console log.
- GREEN: full `ModelsBody.test.tsx` suite passed: 4 tests.
- Lint: direct ESLint on `Models/index.tsx` and `ModelsBody.test.tsx` exited 0; only the known Next pages-directory notice was printed.
- Whitespace: `git diff --check` passed.
- Design-state verifier: `bun run verify:design-system-state` remains blocked by local `ERR_MODULE_NOT_FOUND: Cannot find package 'typescript'` from `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
- Bandit: not applicable; this slice touched TS/TSX and Markdown only.
Additional TDD/review notes:
- RED: expanded the refresh regression with a hyphenated `sk-...` style token; the focused test failed because that token still appeared in the notification description.
- GREEN: updated the Models action error formatter to redact both underscore and hyphen secret-like token forms.
- Coverage: added an OpenAI OAuth connect-action failure regression proving the same formatter redacts endpoint, `api_key`, and filesystem path details for OAuth action notifications.
- GREEN: full `ModelsBody.test.tsx` suite passed: 5 tests.
- Lint: direct ESLint on `Models/index.tsx` and `ModelsBody.test.tsx` exited 0; only the known Next pages-directory notice was printed.
- Whitespace: `git diff --check` passed.
- Design-state verifier: `bun run verify:design-system-state` remains blocked by local `ERR_MODULE_NOT_FOUND: Cannot find package 'typescript'` from `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
- Bandit: not applicable; this slice touched TS/TSX and Markdown only.
Final review cleanup:
- Switched the redaction placeholder from `[models-endpoint]` to `[server-endpoint]` so refresh and OAuth action details share neutral, accurate copy.
- Re-ran the full `ModelsBody.test.tsx` suite after the wording change: 5 tests passed.
- Re-ran direct ESLint and `git diff --check`; both exited 0. The design-state verifier remains blocked by the local missing `typescript` package as documented.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a Models settings action-error formatter that redacts API endpoint paths, filesystem paths, and secret-like tokens before showing refresh or OpenAI OAuth action failures to users. Refresh failures now notify and log only sanitized text, OAuth action failures reuse the same formatter with a neutral `[server-endpoint]` placeholder, and the ModelsBody regressions verify sanitized notification and console output while preserving the existing defaults/readiness/catalog behavior.
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
