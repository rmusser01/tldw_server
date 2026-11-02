---
id: TASK-15421
title: Qualify Claims cluster Jobs recovery and operations
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15420
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cross-backend fault/property/security qualification, rollout/rollback/restore runbook and whole-branch review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 11: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Controller begins independent Task11 namespace prework while Task7 publication implementation runs in disjoint files; recovery/runbook/whole-branch qualification remain pending Tasks1-10 completion. Fresh2026-11-02 audit: origin/dev5937f33065c4b5b5b2d0f731d1914c0651532eb5 and all39openPR paginatedfileAPIs,290selectedpaths, remote taskmax15413/local15421. OUR planning15282/15283 collide with published Chunker records; OUR15411 collidesPR3398 sandbox task; OUR15412 collidespublishedVZ task; OUR15413 collidesPR3404 UserProfiles followup. Renumber onlyOUR five records to15422..15426 via backlog-py and repair active dependencies/references; preserve historical notes and allotherownersrecords. OUR private ClaimsADR085 collidespublishedAuthNZ085; metadata move to086 preserves accepted rationale. Proposed087 remains free in this fresh audit. No rebase/push/maincheckout changes. ADRassessment: identifier-only repair under existing decisions, no new architecture or acceptance.
Namespace repair uses CLI only, preserves selected statuses/criteria/history and fixes dependencies. ADR086 Decision-through-end diff against ebc5cd0f00 oldADR085 is byte-identical. Local task-ID checker exits0. First Backlog/format/ID suite164passed,1failed,32warnings2.39s: HEAD-aware changed-task gate still reads old filenames from committedHEAD while renames are unstaged, so FileNotFoundError for old15282. This is a measurement sequencing issue, not noncanonical content; explicit selected canonical check exits0. Commit isolated canonical metadata with normalhooks before rerunning unchangedHEAD-aware gate. Do not alter/skip gate or include activeTask7 production files.
Namespace-only commit20b97fad578cb5943248d7f6324230e6fdb55185 contains17metadata paths only; activeTask7 code/TASK15417 neither included nor stashed. Normal explicit-files hooks all applicable pass, diffcheck0, selected canonicalchecks0, localIDguard0. Unchanged full Backlog/tool/format/ID suite after committedrenames:165passed,32warnings,1.82s (/tmp/claims_stage2c_namespace_tests_after_commit.log), zero skips. AcceptedADR086 Decision-through-end byte-identical to old085. Initial HEAD/unstagedfilename sequencing failure resolved by commit, not test/gate change. RuntimeBandit not applicable to this metadata-only delta. Task11 remains InProgress for later recovery/runbook/wholebranchqualification; fresh publication audit still required.
Independent namespace-only review namespace-repair-review.md: specPASS/qualityAPPROVED, zeroCritical/Important findings, no runtime/Task11completion claim. Currentrecordmap and ADRmetadata repair verified against package20b97. NonblockingMinor:165-pass namespace log has32warnings but no category/location breakdown; retain for alreadyplannedTask11 final warning breakdown, no duplicate suite rerun/suppression solely for logs. Existing namespace audit mustrefresh before authorizedpublication. Reviewerclosed after controller read fullreport.
Task8reviewMinor carry-forward: eachrequiredcolumninventory test filtersmetadata, not a physicallyincompletePGtable. Beforefinalqualification add a focusedofficialrestricted-fixture case physicallyremoving one evaluatorrequiredcolumn (testsetup only, usingofficialfixtureDDLcapabilities), assert open_existingNone/evaluationno provisioning and unchangedpost-alterinventory. Do notcreatecustomPostgres/adminruntimepath orre-runalreadycoveredsuitessolelyforlogs. MissingPGtable/revokedSELECT alreadyactualcovered. InheritedTask8warning376/Ruff12baseline/nosecwarning qualifications remainexplicit, no newcodefinding. Task8codegateApproved; this is boundedTask11coverageimprovement, notanotherJobs/lifecyclefeature.
Begin bounded independent Task11 physicalpartial-columnPG coverage prework while Task9underreadonlyreview. Own only existing Task8testfile(s), tests/Claims/test_claims_cluster_watchlists_existing_only.py or tests/Watchlists/test_watchlists_existing_only.py, plus Task15421tracking. No Task9production/Task10/recoveryengine/wholequalification work. Use officialrestrictedPGfixture/testsetupDDL, no mockedmetadata/customserver/privilegedruntimepath; actual requiredcolumn absent, open_existingNone/evaluatorno provisioning and schema inventoryunchanged aftercall. Characterizationofalreadycorrectbehavior, no fakeRED/productionrevert. Keep changesUNSTAGED/UNCOMMITTED untilcontroller integrates aftercurrentTask9gate, disjointwriteownership. Run focusednewcase/configuredquality only, preserve all otherwork.
Bounded Task11 physical-column prework on original base556b226a298eecd140dfbaeab43ef85eb689bbd5: added only test_postgres_physical_missing_evaluator_column_preserves_schema_and_data in tldw_Server_API/tests/Claims/test_claims_cluster_watchlists_existing_only.py. Official pg_restricted_backend owns watchlist_clusters; same actual NOSUPERUSER/NOBYPASSRLS session performs test-setup ALTER TABLE DROP COLUMN created_at and accessor/evaluator operations. Real metadata proves column present before/absent after; post-ALTER public schema/actual row-value inventories are identical after open_existing(None) and evaluation(None), with seeded subscription, count17 and sentinel notification preserved and fail-fast constructor/bootstrap/global-routing/tag/count/notification tripwires. No fixture or production change, fake RED, admin runtime caller or custom PG. Initial focused case1passed/19warnings1.81s; final parsed-JSON assertion version1passed/0fail/error/skip,19warnings1.46s. Configured whole-file Ruff baseline0/final0; Black check added lines203-291, compilation to tmp and owned diffcheck pass. No production Bandit applicable: test-only delta, scanner not run/no scanner-pass claim. No shared fixture changed, so no neighbor/broad/Task9/Jobs/Chroma suites rerun. Full warning category/location breakdown, exact commands/logs and limits in ignored .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-11-physical-column-report.md; artifacts /tmp/claims-stage2c-physical-column-20261102-556b226a-*. Changes intentionally UNSTAGED/UNCOMMITTED; no git add/commit/stash/reset, nested agents, publication or Task9 scope inspection. Controller integrates/reviews after current gate; Task15421 remains In Progress with all AC/DoD unfinished, not final Task11 qualification.
Controllerreadphysicalcolumnreport+complete91lineownedtestdiff: actualrestrictedownerPGcatalog/DDL removeswatchlist_clusters.created_at; nonvacuouscount/subscription/notification/schema/data snapshots preservedafteropen_existingNone/evaluatorNone, bootstrap/routing/DMLtripwires. Focused1pass0skip19warnings1.46s withbreakdown, Ruff0baseline/current/compile/scopedformat/diffchecks pass, no productionBanditapplicable. Sidecarclosed, ownedtest+Task21notes remainpendingparentcheckpointintegration afterTask9cleangate. FormalTask11finalindependentreviewmustincludethiscoverage delta; no Task11completionclaim. PreserveTask9warningM1 historical1512/current585 qualification untilactualfinalbreakdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
