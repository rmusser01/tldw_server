---
id: TASK-15661
title: Finish the WebUI Quick Ingest review for blocked actions, mixed receipts, playlists
  and progress
status: Done
labels:
- webui
- ux
dependencies:
- TASK-15660
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete remaining priority items 2–4 from the full-modal review #3633 after the review-draft receipt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explain disabled submission and put the relevant provider or connection remedies beside it, preserve focus, show preset destination changes, and hide inapplicable overwrite settings.
- [x] #2 Put actionable failures before exclusions, distinguish repeats from Media matches, and keep full source identities and touch-sized result actions usable on phones.
- [x] #3 Offer one playlist add action with clear duplicate outcomes; lead progress with actual counts and active source and disclose technical details.
- [x] #4 Verify through the actual WebUI on desktop and phones, document behavior and publish the follow-up against dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. The changes clarify existing controls and server-reported outcomes in the one-screen modal.
Plan in three independently verifiable changes: submission blockers/settings, mixed result rows, and playlist/progress simplification. Follow the approved design review #3633. Validate focus and count transitions in the actual Media modal.
Verified 515 scoped Vitest tests, 27 Chromium cases and three mobile WebKit cases. A real 503-to-healthy fixture recovery preserved input and queue; a custom HTTP 422 produced Edit link using the status code. Browser checks at 320 and 390 CSS pixels confirmed complete source names and 44px row actions. Desktop and phone screenshot pairs cover missing provider, mixed results, playlist preview and progress. Frontend TypeScript, scoped ESLint, English catalog parity, mirrored guide comparison and git diff --check passed. Bandit scanned the frontend scope with no Python source.
Published against dev in https://github.com/rmusser01/tldw_server/pull/3641, stacked on the review draft receipt in #3640. The PR includes four desktop/mobile screenshot pairs and a comparison restricted to these follow-up commits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed blocked submission and focus, mixed result hierarchy and source repair, duplicate provenance, accessible phone rows, playlist selection and transparent progress. Eight screenshots and the fixture-backed regression suite document the updated WebUI.
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
