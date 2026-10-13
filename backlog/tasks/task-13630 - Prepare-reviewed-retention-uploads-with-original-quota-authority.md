---
id: TASK-13630
title: Prepare reviewed retention uploads with original quota authority
status: Done
labels:
- chat
- assets
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved H2 private single-upload producer using typed candidate admission, actual AuthNZ quota reservation and the house upload validator. Check original source/lease after asynchronous boundaries; prepared replay keeps its charge and does not consume uploads. No claims/publication/public capability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact reviewed bytes prepare using original owner quota authority and existing validator outside chat transactions.
- [x] #2 Changed source, wrong quota owner, malformed upload and nested calls cannot return accepted preparation.
- [x] #3 Prepared replay does not consume input or double charge; interrupted staging retains conservative cleanup fences.
- [x] #4 Supported real-AuthNZ regression and connected tests, security and independent reviews recorded; publication remains open.
- [x] #5 Original lock ordering and separate bounded child startup/publication controls remain qualified.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved H2 Task2.3; ADR required:no, existing049/050 plus proposed070/071 govern lifecycle/quota/source. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_retention_upload_13610.md.
Private single-upload producer implemented through originaltypedcandidate→realAuthNZquota→sourceadmission→houseupload→finaladmission. Preliminary7runtime failures/7officialAuthPGskips317.52s withheld causes, notcausalproof. Properdirectmissingproducerred1failed13deselected4.42s; initialsuccess/replayfailed1/13deselect10.60s; safeValueErrordiagnostic31.47s. Sharedpreparedquota reconciliation added withoutintentcreation/allocation; 7SQLitegreen76.30s; expanded11green127.78s beforefixturefreshnesscorrection. PreservedoriginalNone promotion thenactualred1failed21deselect32.44s; boundedread-onlyrefresh fixes race; targeted3SQLitepass19deselect15.29s. Fullproducer11pass11deselect159.37s and connectedquota5pass45deselect89.05s. Connectedstore/upload running. Bothreviewers identified remaining classifier candidate-before-workspace ordering; correction + structuralcontrol pending after frozenrunner. No publication/capabilities/claims/runtime/profile/browser changes.
Final stable-source producer12pass12deselect122.31s; quota5pass45deselect56.27s; physicalIO58pass59skip137.19s,each12warnings. Four-fileRuff0/HEADexisting3files0; actual2prodBandit0/errors0/coveragechecked; scoped120columnASTBlack/compile/diff0; bothfinalsourceCLEAR. Canonical2129/3301 bd7997199762d42481de4c4b32536f3c82f279d695c86ff44afa427b36771212 fingerprint/pathunchanged. Lock-orderfixtureinitialwrongselection1pass excluded; correctedactual1fail23deselect48.38s→target3pass21deselect29.51s. Earlierconnected161pass163skip1fail508.56s atcombinedstartup/publicationcheckpoint; readinesssplit120sec init+unchanged15/5/8semanticbarriers; fourSQLiteprocesscontrols/fourPGskip23.22s. Physical7IOmethodsHEADASTsame. Record Docs/Reviews/CHAT_RETENTION_UPLOAD_13610_2026_10_12.md; onlyownplanretires to /private/tmp/IMPLEMENTATION_PLAN_chat_retention_upload_13610_completed.md. Publicproducer allocation/Sync/mapping/claims/readiness remainopen.
Tracking identity rekeyed from this branch original TASK-13610 during current-dev integration TASK-13629. Remote dev now owns TASK-13610 for Messages API usage. Used backlog-py canonical parser/frontmatter editor and atomic writer; all original metadata except id, full body, criteria and completion checks preserved. Original review/commit titles retain historical13610; current record is TASK-13630.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Private exact reviewed-upload preparation uses actual original AuthNZ quota and house validation with source checks after awaits. Shared prepared replay reconciles only existing confirmed active authority; bounded missing-receipt refresh and original lock order qualified. Final12producer/5quota/58physicalIO pass,59skips; contract/security/reviews clear. Cold child readiness separated from unchanged publication/death checks. No public readiness or broadUX completion claim.
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
