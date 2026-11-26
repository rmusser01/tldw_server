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
- [x] #5 Guide excluded sources back to inclusion and allow a deliberate single-source new attempt after a confirmed server cancellation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. The changes clarify existing controls and server-reported outcomes in the one-screen modal.
Plan in three independently verifiable changes: submission blockers/settings, mixed result rows, and playlist/progress simplification. Follow the approved design review #3633. Validate focus and count transitions in the actual Media modal.
Verified 515 scoped Vitest tests, 27 Chromium cases and three mobile WebKit cases. A real 503-to-healthy fixture recovery preserved input and queue; a custom HTTP 422 produced Edit link using the status code. Browser checks at 320 and 390 CSS pixels confirmed complete source names and 44px row actions. Desktop and phone screenshot pairs cover missing provider, mixed results, playlist preview and progress. Frontend TypeScript, scoped ESLint, English catalog parity, mirrored guide comparison and git diff --check passed. Bandit scanned the frontend scope with no Python source.
Published against dev in https://github.com/rmusser01/tldw_server/pull/3641, stacked on the review draft receipt in #3640. The PR includes four desktop/mobile screenshot pairs and a comparison restricted to these follow-up commits.
Remaining review follow-up on PR #3641: distinguish excluded, invalid and empty queues in the footer; provide an explicit per-source action only after a server-confirmed cancellation; verify real-worker status and available assistive technology. The saved-result handoff and its loaded contents were verified in #3631. No new ADR; the existing job status, receipt, and retry contracts govern this UI change.
Verified the all-excluded footer at 320, 390 and 1280 px. The confirmed-cancellation action submitted only the selected source a second time and preserved its saved peer in desktop and mobile Chromium and mobile WebKit. Added a fifth screenshot pair. The focused suite passed all 533 tests, scoped TypeScript and ESLint passed, and Bandit found no Python in the touched frontend scope.
A live API and worker cancelled one claimed file and one queued file. The minimized widget displayed Stop requested until confirmation, then 2 cancelled. The receipt after reload, Jobs API and job database agreed. Neither file appeared in Media after the downstream operation finished. In three runs, staging files for never-claimed jobs were still present 123 to 315 seconds after cancellation; claimed files were removed. Eventual staging cleanup remains unverified.
Final-build emulated iPhone WebKit and Pixel Chromium checks verified source-edit focus, Back and Undo preserving input, and retry of a failed IndexedDB draft write. The retry created one draft from one extraction. Native VoiceOver or TalkBack validation remains open pending access to a physical handset.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed blocked submission and focus, mixed results and source repair, duplicate handling, playlist selection and progress. All-excluded queues now explain inclusion. Only a confirmed cancelled source can start a new attempt from its row. Ten screenshots document the WebUI. Live worker cancellation agreed with both the minimized widget and the restored receipt. Physical screen-reader testing remains open.
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
