---
id: TASK-13605
title: Persist server-issued conversation incarnation for source fences
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hard-deleted conversations can be recreated with the same ID and clock and reset counters. Persist a server-issued row incarnation on ordinary and Sync insertion; preserve updates and restore, ignore imported authority and migrate existing rows in new registered upgrades.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New and migrated row incarnation is stable across update, reopen and restore and regenerated on same-ID recreation.
- [x] #2 Both insert paths and registered migrations preserve existing data and refuse caller incarnation authority; published migrations unchanged.
- [x] #3 Causal and connected checks, static/security and two source reviews qualify the change with proposed ADR and public limits recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required yes: proposed Docs/ADR/071-conversation-source-incarnation.md. Governing ADR049/050 unchanged. Prerequisite for TASK13604; source capture remains unqualified until bound and verified.
Nonce issuedatdatabaseINSERT boundary coversordinary+Sync; newSQLite81/Pg85 migrations/all5paths, backfilloriginaldata preserved, immutable guards/copiedinputrefusal-or-serverreplacement andfixedclockrecreation. ExistingRLS/publicDTO/export/import unchanged. ProposedADR071 indexed, source13604bindsnonce. Finalconnected138/133+extra2/2PG andstaticsecurity/compile/bothsourceCLEAR; jointreview Docs/Reviews/CHAT_RETENTION_SOURCE_IDENTITY_13604_13605_2026_10_12.md. BroaderH1/backup/public/physicalPG unqualified.
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
