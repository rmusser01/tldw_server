---
id: TASK-13603
title: Preserve owned native asset reads after chat restore
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual native chat deletion permanently burns the original creation key while retaining claims and bytes for restoration. The private byte reader incorrectly requires replayable committed state and denies restored active chats. Allow exact owned restored claims with a gone key without reviving replay; retain active conversation/workspace/projection/generation/quota and secure physical read guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual soft-delete and restore permits verified exact original owned bytes while the complete gone operation remains unchanged and non-replayable.
- [x] #2 Deleted/purged/foreign/closed/invalid projection and retired claim/candidate/quota cases refuse; supported physical handle and byte-fencing checks remain unchanged.
- [x] #3 Causal and connected lifecycle/read checks, scoped lint/security/compile and two independent reviews are recorded with physical PostgreSQL and public streaming limits explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required no: ADR-050 and accepted H2 explicitly separate restored claim ownership from permanently burned creation replay. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_restored_asset_reads_13603.md. No originalhash inference, new storage, retention-backed claim/readiness/publicroute changes.
Actual restore root repaired: private read admits committed or existinggone/child_deleted with exact target/projection/owner/scope/activechild/generation/liveclaim/confirmedunreleasedquota. Keyreplay unchanged, physicalreader exactHEAD, lockorder unchanged; ASTproof only2conditions/docstring. Causal2fail11pass13PG6.19s, green51/51, final7suite218/218skip12warnings97.57s. Actualuploadprepare/adopt/delete/cleanuprefusal/restorebytes+goneimmutability, heldguard actualdeletion, corruptbytes and actualworkspaceclosure negative. Ruff2file0/0/Banditactual1prod0/0/errors0/newgateformat/compile/diff clear; bothfinalsourceCLEAR. Record Docs/Reviews/CHAT_RESTORED_ASSET_READS_13603_2026_10_12.md. PhysicalPG/publicstreaming/richpublisher/retentionkind andbroadergates unqualified.
All3AC/6DOD complete in boundedprivate exactread scope. Initial7normalexacthooks0/no bypass; own3stagescomplete andonlyown planretired /private/tmp/IMPLEMENTATION_PLAN_chat_restored_asset_reads_13603_completed.md. Final6filehooks/localcommit follow; broadparent remainsopen.
Final6file normalexacthooks0/no bypass; freshdev stilla744577 with0missingcommits. Only6owned code/test/review/trackingfiles forlocalcommit, no PR/push/merge. Broadmasteracceptance remainsopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repair actual restored native asset read refusal without reviving creation replay: permit only committed or gone/child_deleted provenance with exact current protected owner/scope/target/projection/claim/generation/quota and unchanged physical guards. Actualdelete/restore/exactbytes, cleanuprefusal, corruption/helddeletion/workspaceclosure and immutablegonekey qualified. Finalconnected218pass218skip/staticsecurity/compile andbothsourceCLEAR. PhysicalPG/publicstreaming/richretention andbroaderacceptance remainopen.
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
