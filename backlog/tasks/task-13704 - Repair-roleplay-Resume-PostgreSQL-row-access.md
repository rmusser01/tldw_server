---
id: TASK-13704
title: Repair roleplay Resume PostgreSQL row access
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual roleplay Resume PostgreSQL uses mapping rows but reads prompt budget and original parent version positionally, rejecting admitted input before intended gates. Read named columns without changing byte limits, original parent identity or ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce real PostgreSQL append/replay failure with official fixture
- [x] #2 Named budget and parent version reads preserve SQLite and PostgreSQL semantics
- [x] #3 Actual two-engine roleplay admission, generation, byte-budget, replay and original-parent controls pass
- [x] #4 Scoped security, style, connected checks and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 publicf5d job115377554272; sourceKeyError0 conversation_resume_store.py630, secondpositionalparent row926. Highest13641 acrossremote/open/local beforecreate. No newarchitecturedecision; existingfrozenResume/sourcepolicy, assessmentlink pending. OfficialPG fixtures nowavailable.
ActualPGunchangedappendRED1failed249deselected3.39s→GREEN1passed2.18s. Source2numericrowreads mappedbudget/parent; bothSQLaggregates nowASprompt_bytes andversionnamed. Firstconnected302=282passed20failed(allfixtureKeyError0); fivefixturefunctions readnamedhistory_admission_json or aliasedconversation_count, originalcorruption/rollbackassertionskept. Reviewer requestedbyte-budgetbranch test: actual2engine4UTF8boundarycases cap3refusesbeforepayload/cap4reachesdeliberatepayloadfault, no input/history mutation. Selected46passed208deselected71.67s. Full306running. ProductionRuff0/Bandit0findings0scanerrors. WholefileBlackneedsformatalreadyatparent; fixednewlongconditionviaexplicitprompt_bytes scalar, no inheritedstylechurn. TestparentBlackclean; newtestBlackformattedASTunchanged. BoundedreviewCLEAR, completionpendingfull306/hooks.
Final actualcontract/endpoint306passed0skips1851runtimewarnings397.34s process0; selected46/46passed. BothchangedmethodsBlackclean; fullfileformatdebtconfirmedalsoremoteDev, no unrelatedstylechurn. FulltestfileBlackcleanASTpreserved; production/testBandit0/0errors (testassertionsonlyexcluded), Ruff0. BoundedfinalreviewCLEAR. Docs/Reviews/CHAT_RESUME_POSTGRES_ROWS_13642_2026_10_13.md. Normalexacthooks pending. Original9conversations28rows remainDeepEqual, immutablebaselineunrewritten.
Normalexact4filehooks allpass withoutbypass; owncompletedplan retired. Ready forincrementalpublication, wholePRstillnotclear.
2026-10-15: Renumbered this PR's historical TASK-13642 to TASK-13704 after latest dev introduced an unrelated task with the old ID. The requester explicitly approved the proposed four manual task-ID renames and four report-heading updates. Backlog-py has no existing-task renumber operation; the approved exception changes filename/frontmatter ID, followed by this ordinary backlog-py note. Historical report filenames and original verification content remain unchanged. ADR assessment: no new architecture decision; metadata collision repair under ADR-059.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed actualPostgreSQL Resume promptbudget and originalparent version reads via namedcolumns, preserving arithmetic/fences. Five corruption/rollbackfixtures useportable named rows withallassertionsintact. Added actual2engineUTF8historybyteboundary/refusal controls. Full306passed0skips1851runtimewarnings, selected46passed, scopedBandit/Ruff/touchedformat/hooks clear. RemoteDev wholefileformatdebt unchanged. FinalwholePRreview/rebase/merge andotherfindings remainpending.
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
