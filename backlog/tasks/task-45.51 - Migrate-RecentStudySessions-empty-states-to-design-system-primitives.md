---
id: TASK-45.51
title: Migrate RecentStudySessions empty states to design-system primitives
status: Done
labels:
- design-system
- product-state
- ui
- flashcards
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system product-state migration by replacing RecentStudySessions AntD Empty product-state surfaces with canonical design-system EmptyState/LoadingState primitives while preserving the existing loading, retryable error, no-sessions, and session-list behaviors. Remove the matching product-state baseline exceptions and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] RecentStudySessions uses the design-system LoadingState during recent session data fetches.
- [x] Retryable error state renders a design-system EmptyState with a Retry action and the dynamic error message as description.
- [x] Empty sessions state renders a design-system EmptyState with the no completed sessions message.
- [x] Focused tests assert the canonical design-system wrappers through data-ds-component attributes.
- [x] The two RecentStudySessions AntD Empty baseline exceptions are removed from design-system-product-state-baseline.json.
- [x] The design-system product-state guard passes with the reduced baseline.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verification: `bunx vitest run src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx --reporter=dot` passed (3 tests); `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed (52 tests); `bun run verify:design-system-state` passed with baseline exceptions now 323; `git diff --check` passed. `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on inherited repo-wide TypeScript debt unrelated to this slice. Bandit skipped because no Python was touched.
- Review follow-up: documented explicit Acceptance Criteria, removed redundant LoadingState padding override, moved dynamic load failure text into EmptyState description behind a stable title, and replaced raw `as any` hook-return mocks with a typed RecentSessionsQuery helper. Reverified the focused component test, product-state guard, design-system verifier, diff whitespace, and touched-file TypeScript status.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated RecentStudySessions loading, retryable error, and no-session product states from AntD Empty to canonical design-system LoadingState/EmptyState primitives. Added focused DOM coverage that asserts those states render through data-ds-component markers, removed the two matching RecentStudySessions product-state baseline exceptions, and documented explicit task Acceptance Criteria. Review follow-up removed redundant LoadingState padding, split the retryable error state into a stable title plus dynamic description, and typed the hook-return mocks instead of using raw `as any`. Verified the focused component test, product-state guard unit test, design-system state verifier, and git diff whitespace. Bandit was not run because this slice touched only TypeScript/TSX UI, JSON baseline, and Backlog metadata. Full local TypeScript still fails on inherited repo-wide debt unrelated to this slice; no diagnostics referenced the changed RecentStudySessions files.
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
