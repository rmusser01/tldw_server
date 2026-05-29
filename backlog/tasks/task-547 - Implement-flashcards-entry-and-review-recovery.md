---
id: TASK-547
title: Implement flashcards entry and review recovery
status: Done
labels:
- ux
- flashcards
- implementation
- webui
- extension
modified_files:
- apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx
- apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/FlashcardsWorkspace.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx
- apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/__snapshots__/ReviewTab.create-cta.test.tsx.snap
- apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx
- apps/packages/ui/src/public/_locales/en/option.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 1 from the narrow flashcards UX remediation plan: fix the direct extension /flashcards route, clean remaining Transfer copy, add selected Study deck to Create drawer handoff, keep Re-rate last card visible after rating, and verify Practice again remains absent when there are no cram cards.

References:
- Plan: Docs/superpowers/plans/2026-05-29-flashcards-narrow-ux-remediation-implementation-plan.md from planning commit d3bb1199ef in the source checkout
- Design: Docs/superpowers/specs/2026-05-29-flashcards-narrow-ux-remediation-design.md from planning commits e01a5da940 and 133bb5ec66

Scope is PR 1 only; do not implement PR 2 dashboard/session-history work in this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Extension sidepanel registry includes the real `/flashcards` route and local handoff component.
- [x] Direct flashcards import/export workflow no longer presents user-facing `Transfer` copy.
- [x] Study selected deck is passed one time into the Create drawer, without leaking stale URL deck or workspace state.
- [x] Re-rate remains visible after rating advances away from the answer branch and restores the reviewed card for re-rating.
- [x] `Practice again` is hidden when no cram cards exist and enabled only after caught-up cram availability is loaded.
- [x] Focused tests, package type check, and browser route check are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR 1 from the narrow flashcards UX remediation plan only. PR 2 dashboard-first Study and session-history deck-name work was intentionally left out.

The branch was rebased onto latest `origin/dev` after implementation. The pre-existing `ReviewTab.create-cta` active-card snapshot mismatch was refreshed to match the current design-system `Badge` markup so the focused suite passes on the rebased branch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented:
- Registered the real app-extension sidepanel `/flashcards` route and local handoff component.
- Replaced direct flashcards import/export user-facing `Transfer` labels with `Import / Export` and `Import/export summary`, while preserving internal transfer type/key names.
- Added a one-shot Study selected-deck Create handoff that preselects the drawer deck without leaking stale URL deck or workspace state after the Study selector is cleared.
- Kept `Re-rate last card` visible after rating advances away from the answer branch and hardened its regression test against countdown timing flake.
- Hid `Practice again` when no cram cards exist and enabled the cram queue query for caught-up due-mode completion availability.

Final verification after rebase:
- `cd apps/packages/ui && bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx` passed: 9 files, 85 tests.
- `cd apps/packages/ui && NODE_OPTIONS=--max-old-space-size=8192 bunx tsc -p tsconfig.json --noEmit` passed. The same command without the heap override failed with Node heap OOM before diagnostics.
- Browser route check: `http://127.0.0.1:18031/flashcards` loaded the Flashcards page with Study/Manage/Import / Export/Templates/Scheduler navigation and the expected credentials-required state.

Known skips and limits:
- Browser data-dependent flows for deck preselect, re-rate, and cram availability were not live-verified because `http://127.0.0.1:8000/api/v1/health` was unavailable and the WebUI reported missing credentials. These flows are covered by the focused React tests above.
- Bandit skipped: PR 1 touched frontend TypeScript/TSX, JSON, snapshot, and Backlog files only; no Python was modified.
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
