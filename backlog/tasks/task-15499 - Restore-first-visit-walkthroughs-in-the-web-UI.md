---
id: TASK-15499
title: Restore first-visit walkthroughs in the web UI
status: In Progress
labels:
- frontend
- tutorials
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Offer the existing first-visit prompt on eligible web pages and add Settings and Collections walkthroughs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh web sessions see the first-visit prompt on eligible pages and can start a tour.
- [x] #2 Settings and Collections tours work on desktop and mobile.
- [x] #3 Notes and Research Workspace retain their page-owned first-visit guides.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new decision. Reuse the registered tutorials, prompt, and runner in the existing web shell.
Red: web Chat and Settings prompt tests and three registry cases reproduced absence. Green: 124 focused WebLayout, prompt lifecycle, registry, locale, Settings focus and Joyride runner tests passed. A single browser-level Settings prompt covers its subroutes. Page-owned Notes and Research guides remain in place.
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
