---
id: TASK-13645
title: Qualify native sync storage faults for chat controls
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Toolbar preferences use native Plasmo sync while the existing fault helper reaches only local; several shared fixtures also omit the captured browser storage API. Extend the existing helper to the used areas and retain actual watches, faults, recovery and handoff assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual sync held reads and writes reach the selected API without intercepting sibling local storage
- [x] #2 Toolbar, owner-verification and OpenUI handoff fixtures retain real native SDK behavior and original assertions
- [x] #3 Both configurations and existing local helper consumers pass relevant regressions
- [x] #4 Scoped checks, independent review and normal hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 remainingnativeCI toolbar10, OpenUI3, owner-verification6. NativeSDKdefaultsync vsWebUIdefaultlocal verified reviewer; helperdefaultlocal retained. Highest13644 verifiedorigin/open/local. ADRrequired:no; fixtureIO declaration only, no productionstoragearea or readinessguard changes. Separatehandoffsubmit sourceauthority repairs notinthisunit.
RepresentativeRED4failed90skip andsync2failed9skip toinitialGREEN6passed99skip. Firstfullnative934=933pass1Prodetachedcontrols exposedsourcehydrationrace, tracked13646. Reviewerphysicalsync keyproof fixedactualprefix, ACKpending/settled andexactbytesusingexistingserde. Combinednative986passed10suites; WebUIactive. Defaultlocal preserved, WebUIactualthis retained, originalrefusal/watchassertionsintact. BoundedreviewCLEAR.
Finalnative986passed10suites; WebUI986passed10suites; affectednative7passed41unselected. Physicalsync actualprefix/ACK/bytes proofclear. Bothnormalclient sourcecompilers0 anddiffcheck0. BoundedreviewCLEAR. BanditN/A TS scope. Jointcommit with13646 required tokeep source regression green; exacthooks pending. ReviewDocs/Reviews/CHAT_NATIVE_STORAGE_AREAS_13645_2026_10_13.md.
Normalexact12filehooks allpasswithoutbypass. Owncompletedplan retired; jointreviewable source/fixturecommit13645+13646. WholePRremainingfindings/review/rebase/merge stillopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Qualified actuallocal/sync storagefaults and repaired Pro preference hydration viaexistingguardedlogic. Native986/WebUI986each10suites, affectednative7/WebUI5controls, bothnormal sourcecompilers0; boundedreview andnormalhooks clear. Preservedkeys/areas, watch precedence, visit-only storagefailure, originalrefusal/focusassertions. Physicalsync key/ACK/bytes verified. BanditN/A TS. Joint13645+13646 unit; wholePRstillopen.
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
