---
id: TASK-548
title: Implement flashcards entry and review recovery
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 20:33'
labels:
  - ux
  - flashcards
  - implementation
  - webui
  - extension
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 1 from the narrow flashcards UX remediation plan: fix the direct extension /flashcards route, clean remaining Transfer copy, add selected Study deck to Create drawer handoff, keep Re-rate last card visible after rating, and verify Practice again remains absent when there are no cram cards. Scope is PR 1 only; PR 2 dashboard/session-history work remains out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extension sidepanel registry includes the real `/flashcards` route and local handoff component.
- [x] #2 Direct flashcards import/export workflow no longer presents user-facing `Transfer` copy.
- [x] #3 Study selected deck is passed one time into the Create drawer, without leaking stale URL deck or workspace state.
- [x] #4 Re-rate remains visible after rating advances away from the answer branch and restores the reviewed card for re-rating.
- [x] #5 `Practice again` is hidden when no cram cards exist and enabled only after caught-up cram availability is loaded through a cheap availability probe.
- [x] #6 Focused tests, extension handoff test, package type check, and browser route check are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PR 1 from the narrow flashcards UX remediation plan only. PR 2 dashboard-first Study and session-history deck-name work was intentionally left out.

The branch was rebased onto latest origin/dev after implementation. The pre-existing ReviewTab.create-cta active-card snapshot mismatch was refreshed to match the current design-system Badge markup so the focused suite passes on the rebased branch.

Final review follow-up: Practice again availability now uses useCramQueueQuery(..., { limit: 1 }) while due-mode is caught up, and full cram queue loading is reserved for actual cram mode. The limit caps fetched backend cards before tutorial-residue filtering so availability probes cannot keep paginating. The extension sidepanel handoff is now covered by an app-level render test and uses sidepanel i18n keys with English resources.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented:
- Registered the real app-extension sidepanel /flashcards route and local handoff component.
- Replaced direct flashcards import/export user-facing Transfer labels with Import / Export and Import/export summary, while preserving internal transfer type/key names.
- Added a one-shot Study selected-deck Create handoff that preselects the drawer deck without leaking stale URL deck or workspace state after the Study selector is cleared.
- Kept Re-rate last card visible after rating advances away from the answer branch and hardened its regression test against countdown timing flake.
- Hid Practice again when no cram cards exist and changed caught-up availability to a 1-card cram probe before loading the full queue only in cram mode. The probe cap is applied to fetched backend cards before tutorial-residue filtering.
- Added extension route coverage for the sidepanel /flashcards handoff and localized the handoff copy.

PR: https://github.com/rmusser01/tldw_server/pull/2130

Final verification after rebase/final-review fixes:
- cd apps/packages/ui && bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx src/components/Flashcards/hooks/__tests__/useFlashcardQueries.cram-queue.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx passed: 10 files, 87 tests.
- cd apps/tldw-frontend && bunx vitest run __tests__/extension/sidepanel-flashcards-handoff.test.tsx passed: 1 file, 1 test.
- cd apps/packages/ui && NODE_OPTIONS=--max-old-space-size=8192 bunx tsc -p tsconfig.json --noEmit passed. Earlier, the same command without the heap override OOMed before diagnostics.
- git diff --check passed.
- Browser route check: http://127.0.0.1:18031/flashcards loaded the Flashcards page with Study/Manage/Import / Export/Templates/Scheduler navigation and the expected credentials-required state.

Known skips and limits:
- Browser data-dependent flows for deck preselect, re-rate, and cram availability were not live-verified because http://127.0.0.1:8000/api/v1/health was unavailable and the WebUI reported missing credentials. These flows are covered by focused React tests.
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
