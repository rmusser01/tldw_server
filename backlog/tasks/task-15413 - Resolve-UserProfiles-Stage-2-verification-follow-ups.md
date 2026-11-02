---
id: TASK-15413
title: Resolve UserProfiles Stage 2 verification follow-ups
status: In Progress
labels:
- UserProfiles
- testing
- follow-up
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the explicitly recorded advisory shared-UI failure and non-green local verification controls after WP2 merge, using deterministic reproduction before minimal fixes. Keep test stabilization and local environment repair separate from WP3 runtime implementation and never hide failures with exclusions or gate changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CompanionHome setup-notice assertion has a deterministic delayed-layout reproduction and preserves all original expected titles and security behavior.
- [x] #2 Recorded shared-UI local failures are reproduced or accurately requalified, with narrow fixes only where evidence establishes a root cause.
- [x] #3 Root pytest collection and missing-Go ACP runner failures are investigated and corrected where justified without test skips or weakened runtime policy.
- [x] #4 Affected verification, touched-scope security checks when applicable and independent review are recorded; unresolved external constraints remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester authorized outstanding-item remediation and subsequent WP3-WP5 execution on November2. Work isolated on codex/userprofiles-stage2-verification-followups at authoritative dev3ab1ce5d3a49d5a7fd130e8939d55d7639e5a373. WP2 closeout is a separate two-file commit/PR; human-summary waiver applies only to that tracking-only PR. No runtime package has been started. Previous diagnostic evidence: PR2821 comments6569225730 and6568777659. Plan will be Docs/superpowers/plans/2026-11-02-userprofiles-verification-followups.md. Existing ADR085 governs unchanged membership behavior; a new ADR is not expected for test synchronization/environment corrections.
November2 current-dev RED/GREEN: Workspace API collection failed on nonexistent httpx2 import; corrected to the declared httpx Response annotation and applied existing Ruff import sorting, full file84pass. AuthNZ WebSocket auth matrix reproduced8multiplex unavailable-runner failures/174pass; test-only downstream available-runner fixture leaves real principal/JWT/revocation/scope/IP logic and all original assertions unchanged. Added valid/revoked loader-order controls; fullmatrix184pass, dedicated ACPwebsocket file41pass with unchanged runtime4404 handling. Full root collection now succeeds85935tests, not a full execution pass. PythonRuff/compile pass; Bandit baseline comparison pending final review. CompanionHome deterministic deferred-layout RED reproduces missing Resume Work in bothparametercases; minimal test-only synchronization GREEN19/19, preserved79originalassertions plus2barriers, independentreviewRamanujan approves with0Critical/Important/Minor findings. Fourfile broaderUI control reproduced9fail/31pass; separate implementation/review underway. No runtime/workflow/dependency/gate edits or retries. Closeout tracking PR3403 exactd71b679f3d independentlyapproved and published; requester waived summaryrequirement ONLY for that tracking-only PR, all automatedgates retained. WP3preflight is read-only and implementation remains unstarted.
CloseoutPR3403 is now MERGED as5937f33065c4b5b5b2d0f731d1914c0651532eb5 after exactheadref/rules/integration/feedback verification. User explicitly accepted unchanged docs-only E2E Critical not-applicable/SKIPPED result for3403 ONLY; allother7requiredresults succeeded. No newpolicy/settings/checkchanges or retries. Fullrootexecutionaftercollectionfix was deliberately interrupted on newfailures: exit130,2failed/1000passed/13skipped/18803warnings; nofullsuitepass. Exacttwo Admin/test_backup_storage_scope shared-audit controls passisolated but failafterfullrootcollection (root -k reproduction2fail); causeiscollection-time configreloaddetaching cached consumer settings fromnewconfig.settings. Minimaltest-onlyalignmentofbothAuditHTTPandbackupconsumeraliases is undergoing root-context+full57fileGREENqualification; alloriginalsafetyassertionsstay. UIcontrolsworkerproves genuineVitestjsdomStorage vsNodeglobal/fallbackmismatch andrealAntDmodalpreparationrace; original9failcontrol->41passNode22/46passNode25including5newbrowser-fidelitycontracts. Independentreviewfoundlocalquota-spycleanupgapunderfrontendconfig; sameimplementerfixround1underway, notclaimedapproved. WP3preflightidentifiesstale quota effect mapping vsneweracceptedADR064; narrowreconciliationapprovalrequestedasync, noWP3implementationyet.
Root-context backup controls nowGREEN7passed/12existingcollectionskips/85928deselected; entirebackupfile57passed. Alloriginalguards/storagebytes/artifact/ownership/restoreassertions retained; Banditoldandnew0findings/errors. IndependentreviewWegener approves6addedtestfixturelineswith0Critical/Important/Minorfindings; productioncacheclearingpreservesobjectidentity, noprodruntimeguarddefectmasked. User explicitlyapprovedpreservingneweracceptedADR064andreconcilingstaleStage2effectmapping, whilegenerationfencingremainsforactualevaluationsconfigwriters. RecordthatapprovedreconciliationbeforeWP3runtimework; do notrestoreprofilelimits-to-tierwrites. Pipelinepreflightreport /private/tmp/userprofiles-wp3-preflight.md has allTask9-13/interfacepairs and12coordinatorrulings requiringaWP3child/ADR/stagingbrief, notanimplementationapproval.
Final local checkpoint November2: scoped reviewers Ramanujan, Leibniz, Galileo and Wegener approve their slices with zero remaining Critical/Important/Minor findings. Composer prototype-spy leak reproduced under frontend config (4fail/13pass), fixed with local try/finally restoration, both frontend/shared composer17/17; no weakened assertions. Combined final four Python files366pass. Unchanged StandaloneHtmlWorkspace frontend111/111 Node22. Final six UI files65/65 V8 Node25 after stopping only owned competing jobs; initial same-command run63pass/2Studio15s timeouts remains non-green evidence, not declared harmless. Bare UI suite interrupted130 without full result; bare root suite interrupted130 at2fail/1000pass/13skip, both backup failures subsequently corrected and verified57file/7root-context. Complete root collection85935/exit0. Fulltreequality baseline unchanged and shardguard869/5349/4ignored/baseline44/new0 pass. Final Bandit464B101 (3new expected assertions),2B105+1B106 unchanged,zeroerrors; backupold/newzero. No production/runtime/workflow/dependency/timeouts/skips/policy changes; no compiled Go or localNode20 claim. All own test sessions ended. Plan updated; final exact-head whole-branch review/publication/remoteCI and human summary remain before merge. WP3 official child13001.3 is separate and records approved ADR064 reconciliation; no runtime edits yet.
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
