---
id: TASK-508
title: Flashcards UX Phase 2 review recovery and completion actions
status: Done
labels:
- ux
- flashcards
- phase-2
modified_files:
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx
- apps/packages/ui/src/components/Flashcards/components/KeyboardShortcutsModal.tsx
- apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx
- apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/KeyboardShortcutsModal.rating-scale.test.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.analytics-summary.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.edit-in-review.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/__snapshots__/ReviewTab.create-cta.test.tsx.snap
- apps/packages/ui/src/public/_locales/en/option.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next flashcards UX remediation slice from the reviewed plan after PR #2065 merge. Scope is /flashcards review-loop recovery only: visible undo/re-rate, completion next actions, assistant collapsed/secondary until reveal, available-now progress copy, shortcut parity, and clearer completed-session snapshot labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After rating, Undo/Re-rate is visible in the review/completion area for the supported undo window.
- [ ] #2 Completion offers clear next actions such as Practice again/Cram this deck and Create card, with scheduler navigation only if supported by existing tab/route behavior.
- [ ] #3 Study assistant is collapsed or secondary until the user opens it, preserving recall-first review.
- [ ] #4 Progress copy distinguishes scheduled Due from cards available in the current study queue when new/learning cards are reviewable.
- [ ] #5 Shortcut hints do not advertise an action without a visible equivalent control.
- [ ] #6 Recent session snapshot labels distinguish completed snapshots from active continuation.
- [ ] #7 Focused unit/component tests cover the changed review, assistant, shortcut, and recent-session behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ReviewTab now preserves the supported undo/re-rate affordance into completion, gives clear post-session next actions, and distinguishes current review availability from scheduled due counts. The study assistant is secondary until opened or the answer is revealed, recent sessions read as completed snapshots, scheduler handoffs select the intended deck, and focused/broader ReviewTab tests cover the changed behavior. Draft PR: https://github.com/rmusser01/tldw_server/pull/2066. Verification: focused flashcards suite 50 passed; broader ReviewTab suite 54 passed; design-system product-state guard passed; git diff --check passed. Bandit not run because this slice touched TypeScript/JSON/test files only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
