---
id: TASK-13703
title: Repair native quota allocation on PostgreSQL
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Official CI native reservation and reviewed retention PostgreSQL cases fail inside the owning quota allocation transaction. Use the official AuthNZ fixture and private type-only diagnostics to identify and fix the shared boundary, retaining original ownership, quota, replay, rollback and concurrency behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce PostgreSQL failure and identify original cause without exposing runtime secrets
- [x] #2 Repair shared allocation root cause while preserving owner and transaction checks
- [x] #3 Actual PostgreSQL and SQLite quota reservation and retention controls pass
- [x] #4 Relevant checks, Bandit and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 publicf5d CI gap-verified8 job115377521080. Dockerdaemon available now; official AuthNZ fixture only, no custom DBsetup. WrapperTransactionError does not establish originalcause. ADR assessment pending sourceevidence.
ActualofficialPG retention successRED1failure16s; private typechain ProfileUserWriteRejected <- TokenError <- TokenError. ExactcompactINSERTparameters convertto$1,$2; existinghouse database.py375 documents comma-space requirement. Realrepository/adapter/guard2engineRED1failPG1passSQLite→GREEN2, actualPGsuccessGREEN1. Full76=65passed11failed (1840warnings), subsequent8fixturecompactorg/team/member setup lists plus adminreadonlyasyncpg instancepatch. AdminprobeAttributeError mock.__enter__/__exit__; fixture backendread method+connectionidentityfilteredclasspatch nowunderqualification. Productionguardunchanged. ADRrequired:no; correctionimplementsgoverningownership/quota policy ADR049/050/064 and proposed070, nonewdecision. ProductionBandit0findings/errors0, Black2filescheck0, Ruff0.
Final actualofficialfixture76passed0skips1837runtimewarnings209.89s process0; admin2engine2passed. ProductiononeSQLplaceholder-spacingfix+focused2engine realrepository/adapter/guardregression. Fixture8SQLspacings and originalconnection-onlyactualbackendread faultfix preserveallassertions. FinalRuff0, Blackall3filesclean (originalfixturealsoclean), embeddedSCRIPTparse0. ProductionBandit0/0errors; finaltestBandit0/0errors excludingintentionalB101assertions. BoundedindependentreviewCLEAR. Docs/Reviews/CHAT_NATIVE_QUOTA_POSTGRES_13641_2026_10_13.md. Normalhooks pending.
Normal exact5file hooks allpass without bypass; owncompletedplan retired. Finalprocessactual76/76noPGskips. CurrentwholePRCIheadf5d remainsfailed; notwholePRclear.
2026-10-15: Renumbered this PR's historical TASK-13641 to TASK-13703 after latest dev introduced an unrelated task with the old ID. The requester explicitly approved the proposed four manual task-ID renames and four report-heading updates. Backlog-py has no existing-task renumber operation; the approved exception changes filename/frontmatter ID, followed by this ordinary backlog-py note. Historical report filenames and original verification content remain unchanged. ADR assessment: no new architecture decision; metadata collision repair under ADR-059.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed managed PostgreSQL native reservation INSERT tokenization with spaced bound placeholders; SQL guard and allocation authority unchanged. Added realrepository/adapter/guard2engine regression. Corrected8fixtureboundINSERTlists and adminfault actualPGfetchrow vsSQLiteexecute on originalmanagedconnection. ActualSQLite/PostgreSQL76passed0skips, Ruff/Black/scopedBandit/hooks clear; 1837runtimewarnings recorded. ADR no newdecision, proposed070 governs existingdesign. RemainingwholePR failures/review/rebase/merge open.
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
