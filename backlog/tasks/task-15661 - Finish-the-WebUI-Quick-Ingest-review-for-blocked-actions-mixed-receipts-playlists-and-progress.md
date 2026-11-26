---
id: TASK-15661
title: Finish the WebUI Quick Ingest review for blocked actions, mixed receipts, playlists
  and progress
status: In Progress
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
- [ ] #1 Explain disabled submission and put the relevant provider or connection remedies beside it, preserve focus, show preset destination changes, and hide inapplicable overwrite settings.
- [ ] #2 Put actionable failures before exclusions, distinguish repeats from Media matches, and keep full source identities and touch-sized result actions usable on phones.
- [ ] #3 Offer one playlist add action with clear duplicate outcomes; lead progress with actual counts and active source and disclose technical details.
- [ ] #4 Verify through the actual WebUI on desktop and phones, document behavior and publish the follow-up against dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. The changes clarify existing controls and server-reported outcomes in the one-screen modal.
Plan in three independently verifiable changes: submission blockers/settings, mixed result rows, and playlist/progress simplification. Follow the approved design review #3633. Validate focus and count transitions in the actual Media modal.
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
