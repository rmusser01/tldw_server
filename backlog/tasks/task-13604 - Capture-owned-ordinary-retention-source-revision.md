---
id: TASK-13604
title: Capture owned ordinary retention source revision
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide the approved private pre-retention source precondition for ordinary conversations using the existing coherent H1 snapshot. Domain-separate the owner/scope/conversation/history/settings/image/metadata state from native manifest readiness, lock workspace before conversation, and refuse protected/native-carrier sources until their canonical bridge qualifies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary source revision is stable for unchanged owned state, bound to original owner/scope/chat and changes with actual history, settings, metadata and inline image mutations.
- [x] #2 Active owner transaction and exact live owned open scope are required; native/protected/carrier sources are refused without writes or selected byte reads.
- [x] #3 Causal and connected checks, static/security and two independent reviews are recorded; actual retention reservation/publication/Sync admission/public capabilities remain pending.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required no: ADR-049/050 and accepted H2 require a source fence before publication. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_retention_source_revision_13604.md. This private domain-separated ordinary precondition is not a native manifest or ready bundle; no new column/publicroute/byteadmission.
Initialsourcecapture red14/14PG then green14/14 and connected188pass170skip97.06s are NOT final original-record qualification. Review found same-ID/same-clock harddelete/recreate reproduces oldtoken and Noneowner fallback. Actualidentity red2fail2PG1.88s confirmsboth; explicitNoneowner nowrefuses. Syncjournal rejected asincarnation authority: PGdropscreate triggers, sequencesresettable and selectiveprune canexposeoldrecord. Dedicatedserverissuedrowincarnation prerequisite required across add_conversation/upsert_conversation_from_sync; sourcecapture stays In Progress/unqualified, no publiccapability or commit. Banditfixedsuffix falsepositive annotated at actualqueryconstruction, not ignored. No runtime/browser/profile/sendchanges.
Final sourceprecondition bindsserverincarnation from TASK13605/ProposedADR071; no timestamp/journal fallback/ownerNone/nativeReadiness. ActualrecordsameclockABA reproduced/fixed. Finalconnected138/133skip72.19s + directSQL/rollback2/2PG2.08s, 5fileRuff0/0/actual2prodBandit0/0/errors0/compile clear/bothsourceCLEAR. Jointrecord Docs/Reviews/CHAT_RETENTION_SOURCE_IDENTITY_13604_13605_2026_10_12.md. Actualpublisher/reservation/recovery/Syncadmission/public gates remainpending.
All3AC/6DOD complete in boundedsource-incarnation prerequisite. Initial14normalexacthooks0/no bypass; bothown3stageplans complete andonlyown plansretired privately. Final12hooks/localcommit follow; ProposedADR071 notAccepted, public/physical/broaderH1 gates remainopen.
Final12normalexacthooks0/no bypass; latestdev a744577 with0missingcommits. Joint12ownedfiles onlyforlocalcommit; no PR/push/merge. Broaderparent staysopen; ProposedADR071 staysProposed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bind ordinary retention source to explicit owner/scope/coherent history and server-issued original-row incarnation, preventing same-ID/same-clock ABA and imported identity authority. Registered81/85 migration/guards preserve oldfields/update/restore; source capture selects no bytes/readiness. Finalconnected138pass133skip plus migrationrollback/directSQL2pass2skip, manualsecurity/static/compile andbothsourceCLEAR. Publicretention/recovery/Syncadmission/nativebridge/broaderH1/backup/physicalPG remainunqualified.
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
