---
id: TASK-15419
title: Integrate cluster rebuild contract and shared Jobs worker
status: Done
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15418
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Immutable Claims cluster payload, configuration, scoped shared Jobs admission and existing WorkerSDK handler integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [x] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [x] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 9: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Active governance/reference update under TASK-15421: current Claims resource/publication ADR086 (metadata-only move from private085); ProposedSDKcoordinationADR087 remains Proposed. Planning/design are TASK15423/TASK15422; completed implementationTasks1/2/3 are TASK15424/TASK15425/TASK15426. Dependencies repaired viaCLI. Earlier note/commit IDs are historical evidence, not current aliases to other owners records. No accepted rationale, runtime behavior, status or verification rewritten.
BeginapprovedTask9 afterTask8independentlyapproved d62f897baa. Use task-9-brief.md/constraints.md/task-9-integration-notes.md plusreviewedTask2/3/4/5/6/7/8handoffs. ExistingWorkerSDK lifecycle ONLY: scopedadmission, strictstoredcontract, existingownerMediaDBnonadminscope, receiptbeforevectors, authoritybeforedispatch/publication, explicitdict(result), Task8onlynewly_published. Concreteconfigtable/mode/envprecedence innotes. ScopeincludesnarrowJobs/owned_status.py+tests/Jobs/test_jobs_owned_status_api.py explicitstablefailure/reasonallowlist integration mandatedTask4review; no auth/RLSweakening/wildcards/proxy. RealWorkerSDKSQLite+officialrestrictedPG testsfirst, RUN_JOBS=1, configuredquality/Bandit/hooks/scopedcommit, keepInProgressforreview. ADRassessment accepted086+003/058 governs, Proposed087unchanged; no secondworkerloop/newqueue/cancellationstate. No push/PR/merge/maincheckoutchanges.
Task9 implementation on clean base 63ba1bb81d54d7e1f3d1642c831021d309969216. Approved plan Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md; binding brief/interfaces consumed. RED /tmp/task9-red2.log: 173 expected missing-feature failures. Focused integration /tmp/task9-green2.log: 221 passed, zero skips, 478 warnings including actual official restricted Jobs/Media PostgreSQL coverage. Additional post-commit side-effect and real Chroma tests in progress. Keep In Progress for independent controller review; assigned report .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-9-report.md.
Task9 final implementation/evidence ready for controller review; retain In Progress. Existing handler signature/five branches and lifecycle preserved; explicit supported types, resource admission/nonadmin RLS, immutable settings validation, existing-only owner storage, receipt replay, generation fencing, bounded config, sanitized error/status vocabulary integrated. Final affected suite /tmp/task9-final-affected.log: 717 passed, 0 skips/xfails/xpasses, 1484 warnings; existing UUID/lease regression /tmp/task9-shared-fence-regression.log: 8 passed, 0 skips, 28 warnings. Configured whole-file Ruff: 71 inherited config.py findings vs same 71 at base, 0 new; compilation/scoped Black/applicable normal hooks pass. Bandit /tmp/bandit_task9.json: 0 findings/errors. Full report .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-9-report.md. Allowed extra paths Jobs/owned_status.py and tests/Jobs/test_jobs_owned_status_api.py only closed vocabulary; tests/Services/test_claims_jobs_worker.py only supported-types assertion. No Task10/11 implementation or unrelated files. Whole-branch/repository and warning-population qualification remain Task11.
IndependentTask9review556b226a29 Issues/Needsfixes, oneImportantI1/P1 validatedcontroller source+realWorkerSDKinjectedteardownprobe. Loadedexactreceipt alreadycommitted but newly_publishedFalse; managedMediaDBexiterror bypasses currentpostcommitsuccessguard, turns replayqueued/failed andcanexhaustretries despitecommittedoutput. Accept originalreplay/postcommitrequirement. Fix1/3 originalimplementer tracks loadedCOMMITTEDreceiptresult separately fromnewpublication; retaincapturedresultforlaterteardown/processingfailures, no vectors/calc/Watchlistsrepeat. DoNOTcallallnewly_publishedFalse durable because receiptlesssupersededexists. Precommitfaultsremainclassifiedfailures. Addexact/legitimate-skip retry+reactivationteardownregressionsactualWorkerSDK. Minor1512warningqualificationretainedTask11, no new lint/securitydebt. ConcurrentTask11testonlysidecarownsTask8testfiles/TASK15421unstaged; preserve/exclude them fromfix staging/hooks.
FIX ROUND1/3 for I1/P1 at FIX_BASE556b226a298eecd140dfbaeab43ef85eb689bbd5: capture known committed loaded receipt result independently of newly_published invocation metadata, and preserve it across later processing/teardown faults. Newly committed results retain same protection; receiptless superseded and precommit faults remain sanitized classified failures, from None. Parent finding note preserved. Actual WorkerSDK RED /tmp/task9-fix1-red2.log: 8 expected replay failures, 14 controls passed, 62 deselected, 97 warnings; GREEN /tmp/task9-fix1-green.log: 22 passed, 62 deselected, 97 warnings. Final scoped worker/config/owned-status/service run /tmp/task9-fix1-affected.log: 256 passed, zero skips/xfails/xpasses, 585 warnings. Exact+legitimate-skip retry/reactivation faults cover transient ConnectionError/nonretryable ValueError, receipt IDs/counts/output preservation, no vectors/calculation/Watchlists repeats, executing-thread two-scope restoration, first-publication and precommit/receiptless controls. Configured whole-file Ruff FIX_BASE0/current0, compile/scoped Black pass, touched-handler Bandit0 findings/errors. Keep In Progress for controller re-review. Concurrent TASK15421/Task8 watchlist tests excluded and preserved; only handler/assigned worker tests/this task staged. Full fix evidence appended to assigned task-9-report.md.
Task9complete556b226a29/57cda37a2f aftertask-9-fix1-review.md I1/P1ADDRESSED/no newbreakage. Controllerreadfullfixreview/report/evidence: knowncommittedreceipt+newpublication results surviveprocessing/teardown, receiptlesssuperseded/precommiterrorsremainclassified/staticfromNone, contextsrestore/no vector/calc/Watchlistsrepeat. Finalaffectedfix256pass0skip/xfail/xpass585warnings; earlierfull717+8fencecases0skips1512warnings qualifiedbeforelocalfix, notfalselypristine. ConfiguredfixRuff0baseline/current/compile/format/hooks/Bandit0findings/errors; originalconfig.py71exactinherited0new retained. ExistingWorkerSDKsolelifecycle, no secondqueue/runtime. NoTask10producerenabled.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated strict immutable cluster Jobs admission/config and realowner existing-only domain execution into existing WorkerSDK, preserving oldhandlers and adding six-typeallowlist. Receiptfirst replay, authoritychecks, generationpublication, same-thread vectorlimits, newlypublished-onlybest-effort Watchlists and closedstatusvocab verified. Replayed/newcommittedresults survive teardownfaults without hiding precommit failures. AllvalidatedTask9reviewfindingsclosed byscopedre-review; actualworker/backend/cancel/replay/config/fencetests pass0skips, newlint/securityfindings0; inheritedwarnings/lint qualified. APIrouting/finalqualification remain separate.
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
