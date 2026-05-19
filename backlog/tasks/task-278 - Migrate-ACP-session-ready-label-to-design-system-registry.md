---
id: TASK-278
title: Migrate ACP session ready label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 01:02'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the ACPSessionCreateModal hardcoded Ready product-state label from the guard baseline by routing the creation-progress ready step title through the design-system state registry without changing ACP session creation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ACP session creation ready step title is sourced from the design-system state registry instead of a hardcoded canonical label.
- [x] #2 The product-state guard baseline no longer includes the ACPSessionCreateModal Ready-label exception.
- [x] #3 Focused regression coverage proves the registry fallback is used for the ready step label.
- [x] #4 Design-system guard verification passes for this slice.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

- Updated `ACPSessionCreateModal` so the creation progress ready step uses `getDesignSystemState("ready").label` as the i18n fallback.
- Added a focused source guard to `ACPSessionCreateModal.modal-prop-guard.test.ts` for the ready-step registry fallback.
- Removed the matching `canonical-state-label` baseline exception for `src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx`.
- Refreshed current `ChatbooksPlaygroundPage` AntD Alert baseline IDs after the verifier reported unrelated current-dev drift.

## Verification

- PR: https://github.com/rmusser01/tldw_server/pull/1580
- `bunx vitest run src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts --reporter=dot`
- `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot`
- `bun run verify:design-system-state`
- `git diff --check`
- `bunx tsc --noEmit --pretty false 2>&1 | rg -n "ACPSessionCreateModal|design-system-product-state-baseline|ACPSessionCreateModal.modal-prop-guard"` returned no touched-path matches; full frontend `tsc` remains repo-noisy.
- Bandit skipped: frontend TypeScript/test-only slice with no Python runtime changes.

## Final Summary

Migrated the ACP session creation ready-step label to the design-system state registry and removed its product-state guard baseline exception. The focused guard test now prevents the ready step from regressing to a hardcoded canonical label.
