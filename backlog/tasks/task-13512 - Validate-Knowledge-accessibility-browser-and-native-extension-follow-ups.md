---
id: TASK-13512
title: Validate Knowledge accessibility browser and native extension follow-ups
status: In Progress
labels:
- ux
- accessibility
- knowledge-followup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend PR3196 evidence with available Safari, screen-reader and actual extension-control checks across first-use, multi-source review and saved-content workflows. Prepare real participant and mobile-device sessions; record unavailable hardware and participants honestly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exercise available real browser and native extension controls with retained evidence
- [ ] #2 Perform available screen-reader checks and fix observed accessibility issues
- [x] #3 Prepare first-time and power-user task protocol and explicitly record participant/device coverage
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Related TASK-13453 and Docs/Reviews/KNOWLEDGE_UX_REMEDIATION_2026_10_04.md. User availability for human testers and iOS hardware requested asynchronously. ADR required:no; validation and bounded UI fixes use existing platform conventions.
Native Chrome for Testing exposes real extension toolbar/context-menu capture actions. Save to Clipper did not open its sidebar. Traced launchWebClipperFromContextMenu: it awaits capture before requesting the gesture-restricted Chrome sidebar; openSidepanel also ignores the returned open Promise. Use the existing helper, open during the original context-menu callback, and observe acknowledgment/failure. Verify ordering and native capture on the final bundle; do not substitute a simulated onClicked event.
Available desktop browser and native menu controls exercised. Original actual Chrome menu capture failure reproduced; shared gesture/acknowledgment fix and all sibling callers have regression coverage. Final frontend affected set: 152 passing tests; both clients types/builds pass. Participant protocol and detailed device/VoiceOver/native-final limits retained in Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md. No participant, iOS keyboard or spoken VoiceOver session was completed.
Published bounded follow-up fixes/evidence as draft PR3205 against latest dev: https://github.com/rmusser01/tldw_server/pull/3205 . The separate human Change summary, unresolved device/participant qualifications, independent provenance design and existing CI integration remain explicit in the PR/report. Disposable task runtimes/browser profiles are stopped; managed worktree retained for review.
Native follow-up attempted in a newly isolated Chrome profile through cua-driver launch_app with focus guard. Owned article window was available on current Space, but branded Chrome did not load the flag-supplied extension (manifest expects background.js, debug targets had only built-in service workers). A second profile-bound new-window launch opened New Tab and left the owned browser active despite transient launch response active:false. AXShowMenu reported dispatch but post-snapshot showed no menu. Stopped this path after these three failures and closed only the owned Chrome instance/fixture. No final native capture success, VoiceOver, mobile-device or participant qualification claimed. Headless WebUI canonical history verification continues separately; foreground permission remains pending.
Cleanup verification: cmd+Q was acknowledged by the driver but windows remained, so stopped only the two recorded isolated-profile browser PIDs after verifying their exact --user-data-dir argument; no primary/user Chrome process was targeted. The owned native fixture exited. The background-driver native-menu result remains unqualified; test setup/driver behavior is not asserted as a Knowledge product defect.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented demonstrated sidebar gesture/acknowledgment defect and published a ready-to-run first-time/power-user study protocol. Keep In Progress for final native capture, VoiceOver and actual device/participant validation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
