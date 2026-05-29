---
id: TASK-483
title: Design narrow flashcards UX remediation split
status: Done
labels:
- ux
- flashcards
- planning
- docs
modified_files:
- Docs/superpowers/specs/2026-05-29-flashcards-narrow-ux-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved narrow design spec for remaining /flashcards UX fixes, splitting route/quick-win recovery work from dashboard/history behavior, and keeping scope limited to the user-listed findings plus the direct /flashcards extension route blocker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and revised Docs/superpowers/specs/2026-05-29-flashcards-narrow-ux-remediation-design.md for the approved narrow flashcards remediation scope. The spec splits work into PR 1 (extension /flashcards route blocker, import/export copy cleanup, visible Re-rate last card recovery, Study selected deck create prefill, Practice again regression, focused tests) and PR 2 (all-deck Study dashboard-first behavior, explicit Review all due/allDeckReviewStarted state, and preserved last-known deck names in recent session history). Ran the required spec review loop earlier: first review found a stale DeckStudyDashboard assumption; revised the spec to state no reusable dashboard exists in the current checkout and that PR 2 should adapt existing no-card UI or create the smallest dashboard component; follow-up reviews approved the spec. After user-requested plan review, addressed all five findings: added explicit all-deck review-start state and reset requirements, standardized recovery copy on Re-rate last card unless true rollback exists, added a review-session payload inspection gate and nullable/backward-compatible backend/API path for deck-name snapshots, changed the tab-label target from Create & Import to preserving Import / Export while cleaning user-facing Transfer copy, and called out create drawer initialDeckId ordering after form reset. Local checks found no TODO/TBD placeholders, no non-ASCII characters, and no git diff whitespace warnings. Bandit/tests skipped because this is docs/planning only and touches no executable code.
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
