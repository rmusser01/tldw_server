---
id: TASK-15646
title: Remove the duplicate desktop chat sidebar toggle
status: Done
labels:
- webui
- navigation
references:
- TASK-15493
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the persistent sidebar is on screen, both its chevron and the header hamburger control the same collapsed state. Keep the chevron on desktop and use the header opener for drawers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A persistent sidebar has one toggle when collapsed or expanded.
- [x] #2 Mobile and legacy drawers remain reachable and can be dismissed.
- [x] #3 Sidebar shortcuts still work with the persistent layout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow-up to TASK-15493. Use the existing persistent-sidebar rendering predicate in the WebUI and shared layouts. Retain the existing drawer and keyboard behavior.
Two layout integration cases reproduced the duplicate before the change. The two layout conditions now keep only the desktop rail control whenever the persistent sidebar renders. Updated the shared research UAT script to locate the existing page-launcher button for geometry and the single accessible sidebar button for opening Chats. Verified 211 focused tests and scoped ESLint, plus local Chromium at 1440px for arrow expansion and collapse and at 390px for hamburger opening and keyboard dismissal. The live shared research UAT and hosted rollout were outside this check. Bandit is inapplicable to the TS/JS-only diff.
Renumbered from TASK-15494 while refreshing PR #3468. The original ID on dev belongs to the v0.1.47 release record, which is retained unchanged. The sidebar task still references TASK-15493. ADR-059 and ADR-084 govern the ID correction. ADR required: no. The navigation change reuses the existing header callbacks and persistent-sidebar predicate.
Independent review found that the only desktop toggle could lose focus when the compact and expanded sidebar controls exchange places. The existing two views now transfer focus to the replacement only when the outgoing control held focus and focus returned to the document. Pressing Enter in each state fails without the change and passes with it. The focused shared and WebUI selections passed 348 and 179 tests respectively, including existing Find conversations and mobile drawer behavior.
TASK-15635 was also assigned to the shared UI typecheck record that landed in #3607. This sidebar record has a fresh ID and still references TASK-15493.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the duplicate desktop hamburger in both shell layouts. Mobile and legacy drawers keep their header opener.
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
