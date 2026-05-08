---
id: TASK-45.18
title: Adapt TemplateCodeEditor MonacoLoading to shared LoadingState
status: Done
assignee: []
created_date: '2026-05-08 05:54'
updated_date: '2026-05-08 05:59'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplateCodeEditor.tsx
  - apps/packages/ui/src/components/ui/feedback/LoadingState.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the design-system product-state migration by routing the Watchlists TemplateCodeEditor Monaco loading fallback through the shared LoadingState primitive while preserving the editor loading, error, and ready behaviors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TemplateCodeEditor loading fallback renders through shared LoadingState.
- [x] #2 Existing Monaco editor loading, error, and content behavior remains covered.
- [x] #3 The product-state guard baseline no longer contains the TemplateCodeEditor MonacoLoading local-loading-state exception.
- [x] #4 Focused component tests and the design-system product-state verifier pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing test proving TemplateCodeEditor's Suspense fallback renders the canonical LoadingState primitive and preserves editor height.
2. Replace the local MonacoLoading div with a direct LoadingState return, extending LoadingState only as needed for surface sizing.
3. Remove the obsolete product-state baseline exception for TemplateCodeEditor MonacoLoading.
4. Run focused component tests, guard tests, the design-system verifier, whitespace checks, and record wider type-check status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: bunx vitest run src/components/Option/Watchlists/TemplatesTab/__tests__/TemplateCodeEditor.loading-state.test.tsx --reporter=dot failed because the Suspense fallback did not expose data-ds-component="LoadingState".

Implementation: MonacoLoading now directly returns the shared LoadingState primitive in spinner mode, with the existing editor height preserved via a new LoadingState style prop. The TemplateCodeEditor local-loading-state baseline entry was removed.

Verification: focused TemplateCodeEditor tests passed (2 files, 2 tests); product-state guard tests passed (42 tests); bun run verify:design-system-state passed with 518 baseline exceptions and local-loading-state reduced to 2; git diff --check passed.

Known wider check: bunx tsc --noEmit --pretty false still fails on existing repo-wide type errors in unrelated tests/services such as audio capture tests, Chat composer mocks, Flashcards deck fixtures, Playground tests, and service tests. The visible errors did not target this loading-state slice.

Bandit: skipped because touched implementation files are TypeScript/JSON/Backlog records only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted TemplateCodeEditor's Monaco Suspense loading fallback to render through the shared LoadingState primitive while preserving its editor-height layout. LoadingState now accepts an inline style prop for surface-specific sizing, and the obsolete TemplateCodeEditor MonacoLoading local-loading-state baseline exception was removed. Focused tests and the design-system verifier passed; the package-wide TypeScript check remains blocked by unrelated existing baseline errors.
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
