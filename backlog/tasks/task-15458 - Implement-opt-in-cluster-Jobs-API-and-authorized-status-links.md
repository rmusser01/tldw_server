---
id: TASK-15458
title: Implement opt-in cluster Jobs API and authorized status links
status: Done
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Lazy authenticated routing preserves synchronous 200; opt-in 202/conflict contracts use owner-safe Jobs references without fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [x] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [x] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 10: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Active governance/reference update under TASK-15421: current Claims resource/publication ADR086 (metadata-only move from private085); ProposedSDKcoordinationADR087 remains Proposed. Planning/design are TASK15423/TASK15422; completed implementationTasks1/2/3 are TASK15424/TASK15425/TASK15426. Dependencies repaired viaCLI. Earlier note/commit IDs are historical evidence, not current aliases to other owners records. No accepted rationale, runtime behavior, status or verification rewritten.
BeginapprovedTask10afterTask9fix1independentreviewclean (556b226a29/57cda37a2f). Use task-10-brief.md/constraints/task-10-integration-notes +reviewedTask4/9handoffs. Capture routingonce, asyncbranchNOcreatingMediaDB dependency/lazyJobsmanagerAFTERClaimsowner/Jobsforeignlinkauth; explicitanyuser_idretainsplatform-adminrestrictionincluding self. ExistingQueryvalidationretained; exactvalidthresholdsemanticallyignored. DirectTask9enqueue neversecond-readsflags orlocalfallback, stabletyped202/409/200/OpenAPI/Location/quotamapping/sanitized503. Asyncingresseveryattempt inclreplay; disabledlegacy200/cleanupunchanged. Jobsowner/domaincontext insideactualexecutingthread, officialrestrictedPGendpointproof/restoreaftererrors. Routeavailabilitybeforeadmission. Testfirst/scopedquality/Bandit/hooks/commit, keepInProgressforreview. ADRassessment accepted086+003/058 governs,087Proposedunchanged,no newlifecycle/statusproxy. No push/PR/merge/maincheckout changes.
Task10 implementation complete pending controller independent review. Frozen routing context avoids all creating Media dependencies in Jobs mode; legacy service signature/200 fields preserved, sync and explicit scoped admission offloaded. Claims/strict-owner and foreign Jobs-role/domain authorization precede lazy manager/request lookup; actual status mount preflight, UUID-bound own/foreign links, typed202+Location/409, async-only keys and every endpoint async attempt ingress guard, sanitized unavailable/unknown/lost-ack503 without fallback. Final affected run RUN_JOBS=1 with importlib/strict-markers retained and warning display: 478 passed,0 skips,1030 warnings17.27s (/tmp/task10-green-final2.log); dedicated70 passed,0 skips,156warnings5.57s (/tmp/task10-api-final2.log), including actual official restricted jobs_pg_dsn/jobs_rls HTTP A202+ownedread/B404, sequential replay/error/target-B admission and executing-thread context restoration/non-superuser/non-bypassRLS. Configured7fileRuff baseline4/current4 zeroNEW, compile/scopedBlack pass; touched4productionBandit0findings/errors. InitialRED46failed2passed beforeproduction, additionalOpenAPI/ingressRED recorded. Report .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-10-report.md. ADR086Accepted+003/058 govern;087remainsProposed,no newarchitecture or acceptance. Minimal admin registration remains existing unconditional behavior; producer checks actual mounted routes rather than changing registration. No Task11 qualification/Chroma rerun, no push/PR/merge. Normal hooks and scoped localcommit follow; leaveInProgress for review.
Task10complete4db80b8fcbaac52aa87a93cf621b2d2a0f209d50. Independenttask-10-review.md SpecCompliant/QualityApproved, noCritical/Important. Controllerreadfullreview/report/savedevidence:478pass0skip incl70APIandactualrestrictedjobs_rlsHTTP/GUCthreadproof;1030warnings retained. ConfiguredRuff4exactbaseline/current0new,compile/scopedBlack/hooks pass,Banditfourprod0findings/errors. Frozenrouting/lazyJMafterauth/noasyncMediaProvision/synccleanup, explicitplatformadminoverrideinclself, awaitedforeignJobsreadguard/UUIDlinks/routegates,202/409/quota429/static503lostackno fallback/header/OpenAPI/ingress allcovered. Minorinheritednoise qualifiedTask11, no newdebt. APIcodeimplementedbutfeaturedefaultfalse; no deployment/publication/mergeauthorization.
Publication-time namespace correction under TASK-15465: only this private Claims record renumbered; historical IDs/commits in prior notes remain historical. Current accepted resource ADR094 and Proposed SDK ADR095 preserve prior rationale/status.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opt-in async cluster admission preserves default200syncresponse/cleanup, capturesbranchonce andavoidscreatingMediaDB/lazyJobsuntilauthorized. Typed202/409withUUID-pinnedowned/adminlinks andLocation, statusroute/typegates, asyncingress, strictkeys/settings, quotaanduncertain503handling useJobsdirectlywithnofallback/statusproxy. RealrestrictedPGendpointscope/restoration proven.478tests0skips; newlint/securityfindings0, allnormalquality/hooks pass. Independentreviewapproved; finalrecovery/runbook/wholebranchqualification separate.
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
