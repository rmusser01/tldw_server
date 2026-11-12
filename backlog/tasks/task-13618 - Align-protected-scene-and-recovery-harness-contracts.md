---
id: TASK-13618
title: Align protected scene and recovery harness contracts
status: Done
labels:
- chat
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair remaining PR3229 native Scene/recovery fixtures that fail before business boundaries: supply declared captured storage API, hold real creation-registry transport instead of a nonexistent Plasmo prototype method, bind real saved-roleplay helper into Compact AST harnesses, and assert current native capture/canonical path contracts. Preserve all CAS, owner, cancellation, late ACK and mismatch negatives; no production guard weakening.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual Scene/native first-create fixtures reach real guarded saves with captured API and unchanged Web Locks/CAS.
- [x] #2 Creation reservation cancellation holds acknowledged browser transport and preserves zero-POST/safe retry/original late ACK behavior.
- [x] #3 Compact AST harnesses bind the real saved-roleplay helper and retain input/source/scene/queue negatives.
- [x] #4 Native capture expectation and audio path assertions use current canonical contracts without dropping scope/CAS/path checks.
- [x] #5 Affected suites, normal client types, bounded review and normal hooks qualify the unit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; test contract repair under existing capturedstorage/sourceownership and ADR069 canonical paths decisions. Readonly diagnosticreview /root/pr3229_backend_review identifies7files; exact source/log evidence retained. Highest local/origin/openPR taskID13617checked. UserallremainingPR3229fixes/review/rebase/mergeauthorized; wholePRacceptancepending.
Final7suites227passed/noUnhandledErrors, bothnormalclienttypespassed, boundedsourceReviewCLEAR. RealtransportACKseamafterphysicalwrite/beforeSDKACK; no prototypepatch, onlycreation-registrykeys. AllCAS/WebLocks/originalinput/owner/source/cancel/lateACKnegativecontrolsretained. Review Docs/Reviews/CHAT_PROTECTED_HARNESS_CONTRACTS_13618_2026_10_12.md; owncompletedplanretired/private/tmp. ADRno:new, governingADR069canonicalpaths. BanditN/A tests/Markdown-only. Normalexacthooksnext; wholePR/CI/UX/H2/finalreview/rebase/merge remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repair nativeScene/recovery storage setup, realcreationACKbarrier, Compactrealhelperbindings and canonicalnativecapture/audio-path expectations.227affectedcases/typespass withoriginalownership/CAS/cancellationnegativeassertions; boundedreviewclear, no productionguards changed.
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
