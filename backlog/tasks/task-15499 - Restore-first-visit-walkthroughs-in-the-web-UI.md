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
- [x] #3 Notes offers a short optional create/edit/save walkthrough and never starts by itself.
- [x] #4 Organization members serialize PostgreSQL timestamps without changing the string response contract.
- [x] #5 Research Workspace keeps its page-owned first-visit guide.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new decision. Reuse the registered tutorials, prompt, and runner in the existing web shell.
Red: web Chat and Settings prompt tests and three registry cases reproduced absence. Green: 124 focused WebLayout, prompt lifecycle, registry, locale, Settings focus and Joyride runner tests passed. A single browser-level Settings prompt covers its subroutes. Page-owned Notes and Research guides remain in place.
Code review caught early prompts on loading pages and trailing-slash URLs. Added failing cases first, then deferred new prompts until their first target mounts and honored page-owned guides using canonical routes. Final connected focused Vitest suite: 126 passed.
Signed-in hosted UAT: Media/Prompts first anchors absent in deployed variants, Knowledge skips unavailable secondary targets, handheld Settings spotlight can leave view, and generic offers hide the task. Follow-up aligns tours to actual stable controls and shows task descriptions before Start.
Candidate verification: 119 web-shell, registry, prompt, locale and Joyride tests passed, plus 12 offline demo unit tests. Browser fixtures opened the corrected Media and Prompts tours and kept the first Settings target visible at 320 px. Beta UAT verified Not now survives reload and Flashcards remains in Help.
Signed-in Team UAT reproduced HTTP 500. The production response model rejects a synthetic PostgreSQL datetime at added_at; retain the API string result with explicit normalization.
Signed-in beta UAT identified the Notes auto-tour and PostgreSQL timestamp failure. The fix preserves first-visit dismissal and Notes Help replay. Targeted frontend checks and all three schema regression cases passed.
Public PR #3473. Signed-in owner UAT also exposed an invitation list 500 after recovering the member list. The repository uses naive UTC parameters for PostgreSQL TIMESTAMP in list, create, and expiry cleanup. Eight focused backend cases passed.
PR #3473 CI reproduced the mobile parity guard failure against an unchanged dev source. dev already removed the decorative Composer pill in 157b04a3ac. Remove the stale source assertion; the sticky composer integration test covers the intended absence.
Verified all three local Playground CI suites (52 composer, 15 device matrix, and 63 accessibility tests), plus four sticky-composer integration tests.
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
