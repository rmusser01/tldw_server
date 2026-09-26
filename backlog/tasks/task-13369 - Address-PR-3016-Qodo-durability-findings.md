---
id: TASK-13369
title: Address PR 3016 Qodo durability findings
status: In Progress
assignee: []
created_date: 2026-09-26 00:52
updated_date: 2026-09-26 06:35
labels:
- vn-assets
- review
- durability
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/3016
- https://github.com/rmusser01/tldw_server/issues/2021
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR 3016 onto dev; address all verified Qodo and independent-review VN generation durability findings with regression coverage, preserve API and Jobs contracts, then merge only after exact-head review, required CI and human summary gates pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enqueue failure retries the original batch and deterministic parent Job.
- [x] #2 Storage handoff failures replay without terminalizing the variant.
- [x] #3 Concurrent deliveries use a fenced claim and cannot publish duplicate assets.
- [x] #4 Cancellation clears outstanding reservation capacity and preserves counters.
- [ ] #5 All remaining review comments are addressed with tests or reasoned thread replies.
- [ ] #6 PR checks and human summary gate pass before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_vn_pr_3016_review.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resumed after disk capacity was restored. Confirmed PR 3016 remains open and latest dev has four newer commits. Original eight Qodo threads remain outstanding. Independent review exposed missing-byte/quota accounting recovery and stale claim/publication windows; assigned non-overlapping Storage/AuthNZ and VN worker/DB implementers. Added and observed four new fencing regressions fail before implementation. Frontend verification underway; latest dev fetched; requester-provided Change summary preserved.

Frontend verification: 37 VN asset Vitest tests passed; TypeScript tsc --noEmit passed; scoped ESLint passed; 4 Chromium VN asset smoke tests passed in 58.7s (desktop/mobile and reload recovery). Temporary worktree UI dependency symlink removed after verification. Backend worker/storage fixes and independent re-review still underway.

Storage verification: 177 tests passed in sandbox; 15 PG cases initially skipped due Docker socket sandbox access. Required escalated official pg_temp_db run then passed all 15 PostgreSQL durability cases (100.53s), no PG skips. Independent final review found completed V1 cleanup fails because recipe item_id foreign key prevents item deletion after file cleanup; accepted as blocking and queued for final fix wave.

Final independent review (Rawls) verified three blocking regressions: completed V1 cleanup foreign-key failure; worker replay accepts invalid/truncated bytes and bypasses validation for attached planned items; definitive missing-slot 404 leaves browser pending receipt blocking future generation. Assigned one fresh implementer (Halley) all findings and full-suite failures for a final fix wave. Storage code finalized and PG15 cases passed. Worker code frozen while original full-suite report completes; not merge-ready.

Local implementation and independent re-review complete: all original eight Qodo findings implemented, plus three final-review regressions fixed. Final VN suite 361 passed; Storage/AuthNZ 177 passed; official required PostgreSQL 15 passed; adjacent tests 21 passed; frontend 44 passed; TypeScript/ESLint and four Chromium smoke tests passed. Production Bandit zero findings. Ruff only two pre-existing BLE001 warnings verified against baseline. Rawls independently passed 31 focused backend regressions and found final fix delta clean. Pending: commit/rebase, reply to external threads, fresh Qodo/CI gates, merge.

Rebased cleanly onto dev 3f909e133b and pushed head 87b808188f9599140e0f840da2f6b2e48da774e6. Source blobs unchanged by rebase; post-rebase 45 focused regressions passed. All eight original Qodo threads received individual evidence-backed replies and were resolved; Qodo edited summary 5836873877 marks all eight resolved and references current head. Requested fresh full Qodo review via comment 5842468613. Human requester Change summary preserved verbatim in PR body. Fresh GitHub CI remains queued and merge is blocked; AC6 and task completion remain pending. This tracking update is local pending final integration recording, not a new code change.

Fresh exact-head Qodo review 5324270842 completed at 02:48Z with eight NEW findings numbered 5-12: endpoint recovery ownership, AuthNZ SQL boundary, cleanup observability, PG fixture isolation, test tier/type annotations, checksum verification and missing-file retry behavior. Original eight resolved remain untouched. Reviewing new findings against current behavior and approved preservation/retry contracts before scoped fixes. CI is still queued; no merge attempt.

Fresh-review progress: Task8 implemented with DB_Management-owned VN lock/lookup SQL preserving transaction ownership; required shared isolated_test_environment now owns PostgreSQL durability cases. Agent verified all15SQLite+15PostgreSQL without skips,28 repository/boundary unit cases; productionBandit/Ruff clean. Task9 checksum and definitive-vs-transient failure handling implemented: completed approvals stay immutable, explicit regeneration produces a new draft, unfinished invalid variants fail once and release capacity. Main verified92focusedworker/helper tests,9integrity tests,4 checksum/transient regressions after3 valid checksum negative-control failures,1WorkerSDK nonretryable test; fullStorage142passed. Task7 core receipt workflow extraction and fullVN/independentreviews pending. No merge or new push yet.

Fresh eight findings implemented and independent re-review clean. Core receipt recovery now owns persistence; DB_Management owns new VN SQL on caller transaction; shared isolated_test_environment owns required PostgreSQL cases. Verification: full VN395 passed, final Storage/worker227 passed on previously failing seed, required SQLite15+PG15 passed without skips, AuthNZ28 unit passed, independent67 narrow cases passed. Explicit stat avoids Python3.14 predicate error suppression; off-loop assertions retained. Bandit8 productionfiles zero findings/errors; compileall/diffcheck pass; Ruff only2 verifiedbaselineBLE001. Missing/corrupt completed assets are nonretryable and preserve approvals; explicit regeneration creates new draft, no silent repair. Repository-wide attempt stopped at unrelated MCP flashcards KeyError rows, independently reproduced; source/test/exporter identicaldev, not whole-repo green. Dev advanced59bd584503 unrelatedMCP; final commit/rebase/push, new eight evidence replies, fresh Qodo and requiredCI remain pending. AC5/6 remain unchecked until actual external gates.

Rebased cleanly onto dev59bd5845038342013a2d84d0130f6164f14b54fd; range-diff confirms all6commit patches identical and changed VN source blobs unchanged. Post-rebase58focused passed; commit-stage pre-commit checks passed. Pushed9e5fb2fdfc9e77fb51745d72aa36924160ccc597 with exact force-with-lease87b80818. All8fresh comments received individual fix/test replies and are resolved; paginatedGraphQL confirms16threads total,noneunresolved,no remaining commentpages. Updated verification body preserving requester Change summary verbatim. Requested fresh fullQodo once via5842776017; pending. GitHub reportsOPEN/BLOCKED and actual CI queued; skipped admission checks are not passes. AC5checked;AC6/taskcompletion remainpending. Local integration records retained pending finalmerge recording.

Full Qodo review5324400373 completed on9e5fb2fd at03:33Z. Seven new actionable inline findings: cancelled/parallel slot-state reconciliation, stale-worker slot mutation, missing/retry-exhausted parent Jobs recovery, item visibility docstring, embedded runtime test annotations and positive persisted retry slot IDs. Summary omits seven historical lower-priority items, but paginated inline feedback supplies all seven new findings. Existing16resolved remain preserved. AC5reopened. No duplicate review request or merge; CI32queued/no failures. Extending approved review plan with bounded disjoint fixes and RED/GREEN independent review.

Task12 investigation: Jobs retry_now_jobs lacks owner admission and rejects exhausted failures; deterministic create returns same dead row. Recorded ruling to add the smallest explicit owner-scoped failed-job requeue admission in Jobs facade/DB_Management, same canonical identity with normal admission/counters and no cancelled/quarantined revival; no general admin API/queue change. Effective dev rulesets require7checks, strict base and merge-only method (legacy protection404 does not mean unprotected). Four disjoint implementers active Tasks11-14; independent reviews follow frozen reports. No new push/merge.

Task12 bounded API contract approved: same-row failed-job admission validates owner/UUID/domain/queue/type/payload/key under lock, queued/processing replays without double charge, resets only failed bounded retry budget. New DB_Management helper owns SQL; SQLite/PG quota helpers include explicit retry-admission events in submission-rate counts. Required real-backend race/quota/pause-drain/rollback tests; legacy admin retry unchanged, completed partial fanout fails closed and completed full fanout remains healthy. Task14 frozen for fresh independent review; browser RED6 failures thenGREEN76VN cases/TS/ESLint reported, controller validation pending.

Task13 embedded annotations/docstrings and Task14 positive persisted retry IDs independently approved with no actionable defect. Task13 validRED2 then4AST+SQLite15/requiredsharedPG15 no skips; Main4AST plus focusedSQLite/PG warning diagnostics passed. Existing Starlette/httpx,pytestpluginsconfig,passlibcrypt,Auditfixture shutdown warnings qualified (prior matrix also90); no warning-free claim. Task14 validRED6 then76VNfrontend; Main independently76/TypeScript/scopedESLint passed, temporary owned UI dependency symlink removed. Task11 frozen37new+124existing tests, independentreview underway; Task12 real Jobs recovery tests still active. No push/replies/merge yet.

Task12 frozen23SQLite+23requiredPG passed zero skips,49receipt+10API passed; freshPascalreview active. Main new+adjacent Jobs quota/admission122passed zero skips. Bandit expandedmanager has one confirmedbaselineB608 unchangedquery (validHEADfileexport, noerrors), no newfindings; initialstdininternalerror excluded. Task11 independentreview foundmixedV0/V1displaygap; four actualblockedadapterREDfailures confirmed bothdirections. RulingA approves boundedowner-scopedJobsreadcallback, smallservicewiringhandedfromfrozenTask12, no newqueue/persistedleaseauthority/aggregatepending-as-generating. Inlinehelperdisplayfinallycleanup+explicitlimits; scopedfix/re-review pending. No commit/push/replies/merge.

ConsolidatedTask12review accepted three fix-round1 findings: expired/exhausted processing parent incorrectly healthy; delayedoriginalsubmit overwritescompletedrecoverysnapshot; unindexedretryevents scansharedadmission locks. Copernicus resumed with Jobs-ownedread-onlyhealth/CASreceipt/indexmigrations+REDGREENcoverage, no separateVNlease/reaper. DisjointDBcomplete_idempotency_record ownership coordinatedwithAristotlelegacyhunks. MainJobswarningdiagnostic1passed4baselinewarnings classified. Task11mixedV0/V1R1 addressedoriginalgap,69focused+124existing passed; re-review confirmed newownfailed-deliveryhandoffstickygenerating bug, scopedexactjob+leaseexclusionfixround2 dispatched. Jobs-clockdelegationRED2 observed andminimalintegration underway; secondpost-outcome-reconcileerror risk beingverified, no unverifiedfix. Allcompletion/push/externalgates stillpending.

Task12 fix1 frozen: public Jobs-owned live lease health, first-completion receipt CAS, partial retry-admission event index through established SQLite/PG migration phases; Pascal scoped re-review of all3 findings underway. Main independently ran full new+adjacent Jobs admission/health/quota/migration215 tests with required official PostgreSQL fixture:215 passed,zero skips,4classifiedbaselinewarnings,51.84s (/tmp/vn3016-main-final-jobs.log). Task11 fixround2 preserves committed model outcome on display outage and excludes exact finishing delivery; additionally sanitize reproduced reader error inside VN transaction body before legacy rollback logger sees it, no broad ChaChaNotes edits. Actual SDK-facing RED4failures+1control, full safe sink coverage pending. New7externalfindings remainunresolved until verifiedpush/evidence, prior16resolved preserved; no commit/push/merge yet.

Task12fix1 independentreview: expired parent health and delayedoriginal receipt overwrite ADDRESSED; rateindex NOTFULLY due interruptedPGconcurrentbuild leavesinvalidindex skippedbyIFNOTEXISTS. Verified officialPGdocumentation and existing advisorylocked migrationpattern; Copernicusfix2 scopedownindexcatalogverification/concurrentrepair/failclosedcollisions withactualofficialPGfailedbuild regression. No genericframework/foreignindexdrops/newSQLoutsideDB_Management. Main215matrix predatesfix2, followup pending. Task11 targetGREEN9 includesfullDB+workersink secretredaction, actualSDKreturn/nonretryablefailure/cancellationpreservation andexactleaseexclusion; fullownedverification/report/re-reviewpending. No commit/push/merge.

Main final fullVN suite509passed,zero skips,10warnings,290.81s (/tmp/vn3016-main-final-vn.log),execsessioncompleted. Newslot/workerTask11fix2frozen79ownedcasespassedonce; Archimedes scopedrereviewactive, Aristotleclosed. MainproductionBandit4VNfileszeroresults/errors; Ruffonlysame2verifiedbaselineBLE001. Task12 PG invalidconcurrentindexrepairfix2active. Wholefreshdeltareview/commit/push/new7evidencereplies/freshfullQodo/currentdevrequiredCI/mergepending. No native3.14 or whole-repo-green claimed.

Task11 locallycomplete: Archimedesfix2spec/qualityAPPROVED,mixedV0/V1 originalgap andbothT11-FIX1handofffindingsADDR,nonewactionablefinding. Main509VNpassed,no skips,79owned,4prodBandit0,Ruff2baselinequalified. Task12fix2actualPGconcurrentinitializer test reproduced blockingadvisorylock SELECT/partial-index snapshotdeadlock. Approvedboundedpg_try_advisory_lock in separateautocommitstatements withconfiguredtimeout/30sfallback, finallyownunlock; scopedhelperonly,no genericframework. Indexrepair/finalreview/integrationexternalgatespending.

Tasks11-14 locallycomplete/independentreviewapproved. Pascalfix2all3Task12findingsADDR,canonicalPGinvalidindexrepair/foreigncollisionfailclosed/snapshot-freetrylockconfiguredtimeoutor30sfallback. Main finalindex+retryhealth82passedrequiredPGzero skips,4baselinewarnings,21.17s; earlierbroader215Jobs and509VN passed. FinalBandit10sourcefilesoneconfirmedbaselineB608/errors[],exactunchangedterminalarchivequery;Ruff17paths2baselineBLE001;compile/diffpass,freezehashesmatch. Task14Main76frontend/TS/ESLint andTask13requiredsharedSQLite15PG15+4AST remainapplicable. FreshRamanwhole-delta reviewactive, sourcefrozen; live devstill59bd584503. Commit/push/7evidencereplies/freshfullQodo/exactheadrequiredCI/mergepending; AC5/6notcheckedprematurely.

Task15finalRamanreview foundP2 publishedlegacycandidate hideslive replacement lease via bare(batch,index)settlement beforeexactleasecheck, independentlyboundedSQLite/sourceprobe. No otheractionablefinding. Accepted singlefinalfixwave, no push/resolve/merge. Ruling minimalopaque deliveryfingerprint inexisting itemprovenance onlysettlesexact canonicalcurrentJobsdelivery;unknown/mismatchedhistoryneverhidesreplacement, rawlease tokennotstored/exposed, no newqueue/table/authority/V0outcomechange. RealJobsblocked originalpublish+replacement/V1transition RED/GREEN andscopedrereviewrequired. Prior509VN/82Jobs/hookevidence predates thisfix; freshVN verificationpending.

Final single fix wave and scoped re-review approved: published V0 results now correlate only with the exact current delivery via opaque provenance, preserving live replacements and historical unknowns without exposing lease tokens or changing model inputs. Main final frozen VN suite: 520 passed, zero skips, 10 existing warnings, 307.06s. Final three-source Bandit zero findings/errors; expanded manager scope has one verified baseline B608. Ruff only two verified baseline BLE001; compileall and diff checks pass. Task11-14 independent reviews and Task15 scoped final re-review all clean. Live dev59bd584503 and remotehead9e5fb2fd unchanged; normal final commit hooks, push, seven evidence replies, fresh exact-head Qodo and required CI/merge still pending. Preserve worktree and reports; no whole-repository green claim.

Final normal commit-stage hooks passed on the 23 owned files (all applicable guards, syntax, whitespace and secret checks); inapplicable hooks skipped, existing deprecated-stage warnings remain qualified. Live paginated review state has 23 threads, seven unresolved, no remaining thread/comment pages. Task15 local gate complete; external evidence replies and exact-head gates still pending.

Pushed and verified head4666d4994b13b32e5fda8f4642cef9ca60f1e48f via normal fast-forward; fetched dev59bd584503 remains ancestor so no new rebase needed. All seven fresh findings now have individual fix/test evidence replies and resolved threads. Paginated GraphQL verifies 23 threads, zero unresolved and no remaining thread/comment pages. Human summary preserved verbatim and PR verification updated. One full exact-head Qodo request5843635886 posted at2026-09-26T05:46:59Z, pending; edited zero-findings summary reflects replies, not full new-head completion. Exact-head CI55runs:33queued22completed, no actionable failure, required gate contexts not present; skipped/cancelled checks not passes. No merge attempted; AC6 and task completion remain pending. These final integration records are local pending true finalization, not another code push.

Full Qodo review5324805354 completed on4666d4994b at05:50:11Z and adds four new findings: legacy fallback overrides approved review state; alleged PostgreSQL legacy table/index ordering; centralized display exception; public retry-admission contracts. Paginated GraphQL27threads4unresolved/no remaining pages; old23resolved preserved. Reopening AC5 and preparing one bounded Task16 fix/verification wave. Base DDL visibly creates job_events before index, so verify the migration allegation on a real legacy database before changing production or post a reasoned regression-backed rebuttal. CI unchanged33queued/no actionable failure; no duplicate review request or merge.

Task16 frozen implementation: 309 affected tests passed with zero skips/failures/errors and required official PostgreSQL. Valid RED11 then GREEN29; initial test-only API mistake and initial PG skip/setup failure are explicitly excluded as evidence. The alleged missing-events PG ordering failure does not reproduce: actual unmodified migration ensures events/index and preserves existing Job; production migration byte-identical, regression-backed rebuttal pending. Central exception move preserves executable class and safe logging; fallback now fills only planned/cancelled derived state. Main verified freeze hashes, production Bandit4files zero findings/errors, Ruff only unchanged worker BLE001, compileall/diff pass. Helmholtz independent Task16 spec/quality review active; Main full VN+central exception suite running, not yet passing. No new commit/push/replies or merge.

Task16 locally complete and independent Helmholtz spec/quality PASS, no actionable findings. Frozen affected309 passed required official PG zero skips; Main final VN+central units554passed zero skips10warnings327.30s. Post-run freeze hashes match including unchanged production PG migration; allegation not reproduced, regression-backed rebuttal pending. MainBandit4runtime0findings/errors, Ruff8onlybaselineworkerBLE001, compile/diffpass, applicable normalhooks11ownedfilespass; inapplicable hooks/skipped checks and existing stage warnings qualified. Reviewer closed; Task17 final fresh-wave interaction review/integration pending. No new commit/push/replies/resolutions/merge; AC5/6 still pending.
Task17 final fresh-wave interaction reviewer Harvey active (01a0dc64-c5ca-7d53-8a3b-6993e4ba77be), report task-17-final-review.md. Main live fetch verifies unchanged dev59bd584503 ancestor of4666d4994b. Paginated external review inventory27threads4unresolved/no remaining review/thread/comment pages, no new findings beyond known4; no review request pending. ActualCI55runs33queued22completed/no actionable failure, seven required contexts not yet present. Requester human summary remains exact. No merge attempt; all external gates pending.
Task17 final independent Harvey fresh-wave interaction review local spec/qualityPASS, no actionable findings/named code risks. Reviewer closed; all frozen source/test hashes reverified match. Boundaries/declined judgments remain qualified: prior completed matrices unchanged, instance-local legacy display not execution authority, global infrastructure logging outside bounded fix, actual PG absent-events probe not exhaustive historic installation certification, no exactly-once external execution promise. All local implementation/review/verification complete. Main normal commit/push and four evidence replies/resolutions next; exact-head fullQodo/requiredCI/currentbase/human gates still pending, no merge readiness claimed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
