---
id: TASK-555
title: Implement flashcards dashboard and session history
status: Done
labels:
- ux
- flashcards
- implementation
- webui
references:
- Docs/superpowers/plans/2026-05-29-flashcards-narrow-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-29-flashcards-narrow-ux-remediation-design.md
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
- apps/packages/ui/src/services/flashcards.ts
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/api/v1/schemas/flashcards.py
- tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py
- tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 2 from the approved narrow flashcards UX remediation plan: make all-deck Study dashboard-first and preserve user-facing deck names in recent review session history. Scope is limited to /flashcards Study/session-history behavior and direct supporting backend/API fields only if current payload inspection proves a deck-name snapshot is missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All-deck Study shows a deck dashboard before starting review.
- [x] #2 Review all due starts all-deck due review without selecting a deck.
- [x] #3 Selected-deck Study remains a fast path into that deck review.
- [x] #4 Recent session history shows preserved or resolved user-facing deck names, not raw deck ids or scope keys.
- [x] #5 Payload inspection decision is recorded; backend/API schema is changed only if no adequate deck-name snapshot exists.
- [x] #6 Focused frontend tests pass; backend tests and Bandit run if backend changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Payload inspection found no preserved deck-name snapshot in `flashcard_review_sessions`, `FlashcardReviewSessionSummary`, or the UI `FlashcardReviewSessionSummary` type. This slice adds nullable `deck_name_snapshot` at review-session creation and returns it through the DB/API/client contract.
- All-deck due review now gates the fetched global card behind a dashboard launch panel. Selected-deck review continues to show the active card directly.
- Recent study sessions prefer `deck_name_snapshot`, then current deck names, then a non-technical unavailable fallback.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added an all-deck due-review launch gate in `ReviewTab` so power users see the deck dashboard first, can choose `Review all due`, and selected-deck review remains direct.
- Added nullable `deck_name_snapshot` persistence for flashcard review sessions and surfaced it through the API schema and UI client type.
- Updated recent session history to prefer preserved deck names, fall back to current deck names, and avoid raw deck IDs/scope keys when decks are unavailable.
- Verification: `python -m pytest tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py -v` -> 43 passed, 1 skipped; `bunx vitest run src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx` -> 30 passed; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc -p tsconfig.json --noEmit` -> exit 0; Bandit touched Python scope -> 0 findings; `git diff --check` -> exit 0.
- Browser smoke: `/flashcards` route loaded in the Next dev server after first-run skip, but the local backend was unavailable on the configured API port, so live dashboard data could not be browser-verified in this pass.
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
