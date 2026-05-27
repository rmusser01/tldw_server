---
id: TASK-517
title: Add flashcards extension repeat-capture queue
status: Done
labels:
- flashcards
- extension
- ux
priority: medium
ordinal: 517
documentation:
- Flashcards-UX-Fix-List.md
modified_files:
- Docs/superpowers/plans/2026-05-27-flashcards-extension-capture-queue-implementation-plan.md
- apps/packages/ui/src/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx
- Flashcards-UX-Fix-List.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next F12 follow-up: a native sidepanel queue for repeated page-selection captures, editable drafts, and save-one/save-all actions while keeping LLM generation/templates/import/review in the full Flashcards workspace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel Flashcards can hold multiple captured page-selection drafts at once without auto-opening the full workspace.
- [x] #2 Each draft shows source context and supports editing Front/Back fields before save.
- [x] #3 Users can delete individual drafts, save one draft, save all valid drafts, and clear saved drafts without losing unsaved failures.
- [x] #4 Bulk save surfaces clear success, partial failure, and failure status, preserving failed drafts for retry.
- [x] #5 Full Flashcards remains the explicit handoff for generation, templates, imports, and review; this slice does not add native sidepanel review or LLM generation.
- [x] #6 Focused sidepanel tests cover multi-capture queue behavior, per-draft deletion/editing, save-one, save-all partial failure, and handoff copy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-flashcards-extension-capture-queue-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: `Docs/superpowers/plans/2026-05-27-flashcards-extension-capture-queue-implementation-plan.md`.

Touched scope: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`, `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`, `Flashcards-UX-Fix-List.md`, and the implementation plan.

Implementation notes: converted sidepanel Flashcards from a single selected-text draft to a repeat-capture queue; added per-draft edit/delete, save-one, save-all, and partial failure preservation; kept full Flashcards as the handoff for LLM generation, templates, imports, and review.

Non-goals: native sidepanel LLM generation, template application, and in-extension review.

Verification:
- RED: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx` failed with 5 expected queue/save-all/preserve-on-capture-error failures after dependency setup.
- PASS: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts` passed 24 tests.
- PARTIAL: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` has only the unrelated baseline `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3): Type '"comfortable"' is not assignable to type 'GalleryCardDensity'.`
- PASS: `git diff --check`.
- Bandit not applicable: no Python files touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the F12 repeat-capture sidepanel queue. Users can capture multiple page selections, edit queued Front/Back drafts, remove individual drafts, save one card, or save all valid drafts. Successful saves are removed from the queue while failed drafts remain for retry, and the full Flashcards workspace remains the handoff for generation, templates, imports, and review.

Focused sidepanel and route-registry tests pass. TypeScript touched scope is clean; the package-wide `tsc` command still reports the unrelated pre-existing `CharacterListContent.design-system.test.tsx` density baseline.
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
