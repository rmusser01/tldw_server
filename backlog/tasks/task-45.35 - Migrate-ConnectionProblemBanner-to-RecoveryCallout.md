---
id: TASK-45.35
title: Migrate ConnectionProblemBanner to RecoveryCallout
status: Done
assignee: []
created_date: '2026-05-10 00:50'
updated_date: '2026-05-10 00:55'
labels:
  - design-system
  - ui
  - product-state
  - recovery
  - common
dependencies: []
references:
  - apps/packages/ui/src/components/Common/ConnectionProblemBanner.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining ConnectionProblemBanner local-recovery-banner debt to the canonical RecoveryCallout product-state primitive while preserving copy, optional details, action labels, and call-site behavior across common recovery surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ConnectionProblemBanner renders through RecoveryCallout with a matching recovery state and keeps the existing user-facing title/message/details content.
- [x] #2 Focused tests cover retry/settings actions and optional details rendering through the canonical primitive.
- [x] #3 The ConnectionProblemBanner local-recovery-banner baseline entry is removed and the design-system verifier passes with no remaining local-recovery-banner debt.
- [x] #4 Verification records focused tests, product-state guard tests, design-system verifier, syntax/whitespace checks, and known TypeScript/Bandit status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused tests that expect ConnectionProblemBanner to render the RecoveryCallout primitive while preserving actions and optional detail text. 2. Migrate the component to RecoveryCallout action semantics without changing its public props. 3. Remove the local-recovery-banner baseline exception. 4. Run focused tests, product-state guard tests, design-system verifier, git diff check, and touched-file TypeScript filtering.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ConnectionProblemBanner through RecoveryCallout with state=unavailable, preserving badge/title composition, description text, examples, primary action, secondary action, retry action, disabled retry state, and className forwarding. StatePanel primaryAction is now optional to match ActionGroup runtime semantics and allow canonical recovery surfaces that only have secondary actions.

Removed the ConnectionProblemBanner local-recovery-banner baseline exception. The design-system verifier now reports 511 allowed legacy exceptions and no local-recovery-banner bucket.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ConnectionProblemBanner from FeatureEmptyState plus custom retry markup to the canonical RecoveryCallout product-state primitive. This removes the final local-recovery-banner baseline exception while preserving the component public props used by Notes, Flashcards, Quiz, and Review recovery surfaces.

Verification: bunx vitest run src/components/Common/__tests__/ConnectionProblemBanner.test.tsx --reporter=dot passed 2 tests; bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx --reporter=dot passed 7 tests; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52 tests; bun run verify:design-system-state passed with 511 allowed legacy exceptions and no local-recovery-banner bucket; git diff --check passed. bunx tsc --noEmit --pretty false exited 2 on existing repo-wide TypeScript test debt, with no filtered output for ConnectionProblemBanner, StatePanel, RecoveryCallout, ActionGroup, baseline, product-state guard, or TASK-45.35. Bandit skipped because this is UI-only TS/TSX/JSON/Backlog work.
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
