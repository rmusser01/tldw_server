---
id: TASK-2404
title: Implement flashcards UX PR 3 review comprehension and recovery
status: Done
labels:
- ux
- flashcards
- frontend
ordinal: 2404
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx
- apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx
- apps/packages/ui/src/components/Flashcards/components/KeyboardShortcutsModal.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/ReviewProgress.test.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/KeyboardShortcutsModal.rating-scale.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 3 from the flashcards UX remediation plan: review comprehension and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- Assistant is secondary during recall and is hidden until the user opens help.
- Review completion offers recovery actions: practice again when applicable, create card, manage deck/cards, and open scheduler.
- Progress language explains the study queue without contradicting due/new/learning counts.
- Shortcut copy matches visible controls or clearly describes when accelerators are available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 3 from Docs/superpowers/plans/2026-06-23-flashcards-remaining-ux-remediation-plan.md: Review Comprehension And Recovery.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation notes:
- Added an explicit `Need help?` / `Hide help` review assistant toggle so the assistant stays secondary during recall.
- Reset the explicit assistant-open state whenever the active card UUID changes.
- Preserved the existing assistant query, reload, response, context, and feature-hint wiring; no `FlashcardStudyAssistantPanel` API changes were required.
- Strengthened completion recovery actions with tested Create card, Manage deck/cards, Open scheduler, and Practice again affordances when applicable.
- Updated review progress language to label the Study queue and show available, new, learning, and due bucket counts separately instead of relying on misleading scheduled-due copy.
- Updated keyboard shortcut copy so Ctrl/Cmd+Z refers to the visible Re-rate button availability.
- Spec review fix: suppressed the top-bar create CTA and deck study dashboard during the completed-review recovery card so the recovery state exposes a single action surface.
- Removed out-of-scope dependency artifact edits from interrupted worker attempts before committing.
- Bandit N/A: frontend-only TypeScript/React changes, no Python touched.
- Review follow-up 2026-06-23: rebased PR #2467 on latest origin/dev and narrowed completed-review recovery detection to actual completion states. Due mode now reuses `isDueModeCaughtUp`, while cram mode only uses the recovery card when the cram queue is exhausted without an active tag filter. Cram tag filters with no matching cards keep deck navigation and the top-bar create action available.

Verification:
- RED/PATCH: Focused Vitest initially failed the active-card snapshot after replacing the always-visible assistant panel with the explicit Need help toggle; updated the snapshot for the intended UI change.
- SPEC-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx src/components/Flashcards/components/__tests__/ReviewProgress.test.tsx src/components/Flashcards/components/__tests__/KeyboardShortcutsModal.rating-scale.test.tsx` (4 files passed, 36 tests passed).
- INITIAL PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx src/components/Flashcards/components/__tests__/ReviewProgress.test.tsx src/components/Flashcards/components/__tests__/KeyboardShortcutsModal.rating-scale.test.tsx` (4 files passed, 36 tests passed).
- PASS: `git diff --check`.
- REVIEW RED: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx -t "cram tag filter"` failed against the rebased branch because the cram tag-empty state hid `flashcards-review-create-cta`.
- REVIEW PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx -t "cram tag filter"` (1 test passed, 23 skipped).
- REVIEW PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx src/components/Flashcards/components/__tests__/ReviewProgress.test.tsx src/components/Flashcards/components/__tests__/KeyboardShortcutsModal.rating-scale.test.tsx` (4 files passed, 38 tests passed).
- REVIEW PASS: `git diff --check`.
- REVIEW Bandit N/A: frontend-only TypeScript/React and Backlog markdown changes, no Python touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the PR 3 review-comprehension slice. Review mode now stays recall-first by hiding the study assistant behind an explicit help toggle that resets on card changes. Completion recovery actions are pinned by tests, progress copy now separates queue buckets, and shortcut guidance now matches the visible Re-rate control availability. Review follow-up narrowed completion recovery gating so non-completion empty states, including cram tag filters with no matches, keep deck navigation available. Focused review, assistant, progress, and shortcut tests pass.
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
