---
id: TASK-13541
title: Protect Temporary tab closure and truthful compact bulk outcomes
status: To Do
labels:
- chat-ux
- extension
- data-loss
dependencies:
- TASK-13540
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compact New, switches and history opening retain Temporary snapshots. Close removes the target snapshot, including an inactive or last tab, and bulk closure currently does not await acceptance or count actual removals. Require an explicit keep-or-discard choice at target deletion and preserve cancellation and truthful aggregate outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cancelled close keeps exact active, inactive and last Temporary tab content; acknowledged close removes only the intended owned target.
- [ ] #2 Compact New and tab/history switching keep retained Temporary content reachable without a discard prompt.
- [ ] #3 Single and bulk close callers await accepted outcomes and do not report declined Temporary tabs as closed; existing relevant confirmation is not duplicated.
- [ ] #4 Causal checks and applicable client/static gates plus independent reviews are recorded; all-ten program remains In Progress.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only root trace during TASK-13540 identified sidepanel-chat.tsx handleCloseTab and compact Sidebar.tsx single/bulk close continuations as a distinct reviewable deletion unit. Highest task IDs just verified local13540/dev13530/openPR13526. Governing accepted Temporary lifetime and user-control design; no new architecture decision, ADR-049/050 and ADR-059 apply.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
