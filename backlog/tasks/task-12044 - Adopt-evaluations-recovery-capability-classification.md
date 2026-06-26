---
id: TASK-12044
title: Adopt evaluations recovery capability classification
status: Done
created_date: 2026-06-26 06:51
labels:
- webui
- capability-state
- evaluations
references:
- TASK-420
- TASK-418.10.4
- TASK-12043
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage15-evaluations-recovery-classification-plan.md
- apps/packages/ui/src/components/Option/Evaluations/components/EvaluationRecoveryCallout.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationRecoveryCallout.test.tsx
updated_date: 2026-06-26 06:53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the Evaluations route recovery helper. Preserve the existing EvaluationRecoveryCallout API and diagnostics behavior, but classify auth, permission, unavailable, and generic failures through the shared capability-state helper instead of forcing every evaluations API failure into the same unavailable state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 403 Evaluations API failures render the shared permission-denied state while preserving caller-provided user-language title/message.
- [x] #2 Evaluations recovery diagnostics still include the request path and backend detail/status for operators.
- [x] #3 Existing EvaluationRecoveryCallout default message and route callers continue to work without API changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Create a stage-specific plan document for this route-helper slice.', 'Add a focused failing EvaluationRecoveryCallout test for a 403 response state classification.', 'Implement the minimal buildCapabilityState adoption inside EvaluationRecoveryCallout while preserving existing props and diagnostics labels.', 'Run the focused helper tests, lint touched TS/TSX files, and diff checks.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD/verification notes:
- RED: `bun run test:run ../packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationRecoveryCallout.test.tsx -t "classifies forbidden responses"` failed because a 403 response still rendered the `Unavailable` state label.
- GREEN: targeted regression passed after `EvaluationRecoveryCallout` began deriving its state from `buildCapabilityState`.
- GREEN: full `EvaluationRecoveryCallout.test.tsx` suite passed: 4 tests.
- Lint: direct ESLint on `EvaluationRecoveryCallout.tsx` and its test exited 0; only the known Next pages-directory notice was printed.
- Whitespace: `git diff --check` passed.
- Design-state verifier: `bun run verify:design-system-state` remains blocked by local `ERR_MODULE_NOT_FOUND: Cannot find package 'typescript'` from `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
- Bandit: not applicable; this slice touched TS/TSX and Markdown only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the shared Evaluations recovery helper so it derives its recovery state from `buildCapabilityState` instead of always rendering `unavailable`. A 403 response now shows the canonical permission-denied state while preserving route-provided user copy and the existing Evaluations diagnostics labels/details. The public `EvaluationRecoveryCallout` props and current route callers remain unchanged.
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
