---
id: TASK-13644
title: Qualify genuine prior chat schemas and incarnation preservation
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migration tests incorrectly rewind current schemas and retain later columns, or omit newly assigned incarnation fields and expect unsanitized PostgreSQL errors. Build genuine prior heads with existing constructors, retain strict migration behavior, and qualify exact row and incarnation preservation on reopening.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce current SQLite/PostgreSQL migration qualification failures
- [x] #2 Provenance upgrades start from genuine prior heads and assert current heads
- [x] #3 Historical rows and new incarnation identity persist exactly through reopen and forbidden changes
- [x] #4 Actual connected engine tests, scoped checks and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 CI jobs115377536158/535615/536334. Highest13643 verifiedorigin/open/local. Scope notes_provenance, postgresmigrationv61, sourceincarnationfixtures only unlessproductiondefect proven. ADRrequired:no; fixturesimplement existingmigration/incarnationpolicy, proposedDocs/ADR/071-conversation-source-incarnation.md. OfficialPGfixtures available; no productionmigration duplicatecolumn waiver.
ActualRED7failed34passed76.93s→GREEN41passed0skips12warnings85.89s process0. Fourgenuinepriorconstructorfixtures76/77/80/81 replacecurrentmarkerrewinds. RLS/receiptcontrolsintact; notescontentpreserved. v61 historicalfieldsexact plusseparateassignedincarnation/stability; PGforgeryproperbackenderror and exactallrows/messagesunchanged. BoundedreviewCLEAR. BlackASTpreserved, oneimportsorted,Ruff0, testBandit0/0errors excludingintentionalassertions. No productionchange; normalhooks pending. Docs/Reviews/CHAT_PRIOR_SCHEMA_QUALIFICATION_13644_2026_10_13.md.
Normalexact5filehooks allpasswithoutbypass; owncompletedplanretired. Test-onlyunit doesnotchangeclientcompilerinputs. WholePRremainingfindings/review/rebase/merge remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replacedfourinvalidmarkerrewindfixtureswithgenuinehistoricalconstructors, retainedRLS/receipts and notecontent throughcurrentheadreopen. Preservedv61 historicalfields exactly plusnewstableincarnation; backend-correctforgeryerror withallrows/messagesunchanged. ActualSQLite/PostgreSQL41passed0skips; independentboundedreview, scopedstyle/security/hooksclear. No productionmigration relaxed; wholePRstillopen.
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
