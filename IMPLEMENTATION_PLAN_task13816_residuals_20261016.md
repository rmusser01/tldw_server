# TASK13816 residual correctness implementation plan

> For agentic workers: use subagent-driven-development for scoped implementation and independent review. Read-only diagnosis of independent domains may run concurrently; implementation writers run sequentially.

Goal: complete TASK13816 remaining acceptance criteria using current owning checks, actual bug fixes and accurate tracking.
Spec: User request 2026-10-16; existing Docs/Design/2026-10-08-knowledge-mechanism-correction.md; TASK13816 acceptance criteria; issue3117 current verified behavior.
Architecture: reuse existing DB, transport, scheduler, Notes and native test owners. No replacement frameworks, gate changes or diagnostic suppression. Existing ADR026/031/034/059/065/066 govern; no new ADR for repairs within these contracts.
Base: f87446f05d442146ee180277418b07a2e45b4683; prior PR3214 merged ce2f87bc; previous immutable evidence retained outside this workspace.

## Global constraints
- TASK13816 owns all changes; backlog-py only, activated existing project venv.
- Leave primary checkout, shared services/venvs and prior archived evidence untouched.
- No blanket casts/Any/ignores, retries, sleeps, assertion weakening, timeout increases or gate relaxation to clear failures.
- Historical640/33 are observations, not guaranteed current counts or causes. Freeze exact commands, dependency versions and current diagnostics.
- Failed reproduction remains qualified; do not infer historical cause from a pass.
- Independent reviews require actual diff, tests, command/outputs; normal configured hooks, Bandit for Python, owning types/builds.
- Three failed causal attempts trigger reassessment, no repetition lottery.
- Do not mark AC5/DoD1 or task Done before remaining obligations have actual evidence.

## Stage 1: Current owners and tracking
Goal: corrected summary and finite current inventory, including issue3117 subitems and existing frontend debt ownership.
Success Criteria: native commands/owners/dependency requirements, exact failure files/codes and evidence references recorded; independent diagnoses produce concrete fix units or specific missing evidence.
Tests: native current backend changed-file MyPy scope from prior32 bindings; inspect frontend coverage and issue3117 tests without changing config.
Status: Complete
- [x] Update stale TASK13816 summary through backlog-py.
- [x] Diagnose backend type families, shared UI33 and issue3117 independently.

## Stage 2: Backend type corrections
Goal: zero diagnostics in all previously failing owning files, original32-file scope and newly measured shared DB adapter scopes under unchanged native flags.
Success Criteria: current RED inventory, smallest corrections retaining runtime semantics, covering regression tests and owning MyPy GREEN; no suppressions or policy changes.
Tests: exact native MyPy --follow-imports=silent --ignore-missing-imports with the original32 files, relevant pytest owners, Ruff and Bandit touched scope.
Status: Complete
- [x] Convert diagnosis into per-family exact edit/test briefs.
- [x] Twenty-three backend units reviewed and committed through bdd0a416eca156058a9212e64ef20e5adc1f5642. Original32-file scope: 640 errors to zero; overlapping shared DB boundary: zero. Extra media rows/helper/wrapper scopes remain source-bound zero.
- [x] Independent actual-diff review for each unit; final fresh original32 and boundary5 MyPy commands both exit0 with no issues. Latest native preservation11pass/0skip; earlier genuine SQLite FTS267, Pg RLS sequence PK23505, missing-ID and version rollback RED/GREEN retained. No whole-repository type-green claim; inherited lint/format/security assertion and warning qualifications remain explicit.

## Stage 3: Coverage failure corrections
Goal: expose and correct reproducible failures behind historical33 without hiding non-reproduction.
Success Criteria: original native test:coverage owner, existing limit and reporter preserve assertion evidence; isolated vs package-suite distinction explicit; any fix has RED/GREEN and independent review.
Tests: RolePlayStarter214 and SkillsManager83 current tests, native coverage full relevant owner when needed for reproduction; no optional report success claim from selection skips.
Status: Complete (existing TASK13817 repair verified)
- [x] Reconcile current native owner and exact33-case identities with completed TASK13817; obsolete local measurement stopped.
- [x] Verify existing merged16shard repair/full hosted coverage; retain historical assertion cause UNKNOWN without duplicate repairs.

## Stage 4: Notes issue3117
Goal: correct currently reproduced print/editor/preview defects within shared Notes owners and both WebUI/extension callers.
Success Criteria: map all six issue items to current source, existing fixes and gaps; relevant deterministic RED/GREEN, platform safety, owning tests and browser behavior; no guessed broad redesign.
Tests: existing Notes export/stage45/UX and preview/editor owners, WebUI/extension types/builds; native browser print document content and lifecycle as applicable.
Status: Complete
- [x] Map current six items and write exact bounded repair briefs from source evidence.
- [x] Shared preview/code/Mermaid a5265c6c, print cc6e1087 and editor a1632648 independently reviewed and committed. Both normal client builds, nine-case native acceptance and fresh affected two-case native acceptance pass with all health gates. Selected-body review finding repaired with two behavior REDs, 64 owning passes and scoped re-review; historical unknowns retained.

## Stage 5: Integrated completion
Goal: satisfy all outstanding TASK13816 criteria with explicit residual dispositions.
Success Criteria: integrated types/tests/security/builds, independent whole-change review, normal commits; current final summary and AC/DoD reflect actual outcomes. No silent closure of unknown historical initiators or unrelated debt.
Tests: combined changed backend/MyPy owners, affected frontend/native coverage, WebUI/extension types/builds, Bandit/Ruff/hooks and tracking format.
Status: In Progress
- [x] Verify remaining obligations and existing TASK12116 ownership against measured results.
- [x] Complete whole-branch review plus one scoped final fix-wave re-review: ReadyYes locally, zero open Critical/Important/Minor findings. All affected source-bound checks and canonical tracking verification pass; AC5/DoD1 are complete. Source commit 3e8285df; no publication or cleanup claimed. Final report /private/tmp/task13816-final-verification-report-20261016.md; all failed receipts and rulings remain retained.

### PR3253 authorized landing continuation (2026-10-17)
Local completion through3b483dc4 remains historical source-bound evidence. The user now requests current-dev readiness and all hosted failures fixed.
- [x] Preserve original head/check failures, rebase only this unarmed PR onto actual latest dev, resolve conflicts and verify all source/base bindings. Actual rebase exit0/no conflicts onto696d7fde; 99branch/13dev paths no overlap and27730 source bindings preserved.
- [x] Correct EOF and stale tenant/import generated locations; strengthen the Notes coverage publication observations with unchanged gates/actions/deadlines. Original hosted Notes cause remains UNKNOWN; unchanged control4pass and observing4pass are local non-reproduction, requiring actual new-head hosted coverage. Native tenant52/cache9 pass, inventory10pass after preserved70pass1inventoryfail; normal hooks103files pass and scoped independent review recorded in continuation evidence.
- [x] Normal exact-lease publish; require actual new-head required statuses and relevant full-suite/coverage/Notes outcomes, review findings and trusted issuers.
- [ ] Final latest-dev/head/queue/rules/readiness verification; retain human-authored Change summary as required external input before merge. No merge is requested by this continuation.

ADR check for PR3253 CI continuation: ADR required: no; generated location refreshes, EOF and existing behavioral test observations introduce no durable architecture or workflow rule. Existing ADR-059 governs canonical task tracking.

### Actual changed-Python type scope continuation
- [x] Clear the actual hosted 63-file scope (28 app + 35 test files), locally reproduced 331 errors in23files using unchanged native flags; no whole-backend claim. Original32 means18app+14tests; boundary union36 means22app+14tests, not36productionfiles.
- [x] Unit A local source: independent review0C/I/M; actual2RED/224PASS and final2PASS; native64 removes39production diagnostics/292tests remain. Existing AllowedAction annotations, truthful optional generation models, narrowed credential runtime and native claims callback compatibility. Two canonical/full-positional regression REDs before implementation, deduplicated six owning suites, native type delta and security/quality review.
- [x] Unit B: source-backed existing static DB delegate signatures, then finite remaining fixture contracts; no runtime wrappers, blanket Any/casts/ignores or gate changes. Actual diagnostics and caller behavior govern each patch.
- [x] Independently review new units, normal commit/push, require new-head CI; publish supplied human Change summary verbatim after running gates settle, then final fresh dev/rules/reviews/readiness.
ADR assessment: no new ADR; corrections preserve existing delegated stores, callable and transaction architecture (existing ADR020/031/059 and prior plan authorities).

Unit B declaration substage complete locally:22native signatures/runtime AST preserved,101owningPASS/restrictedPG0skip, fresh74union231errors remaining, scopedreviewclear. Remaining test-contract/static Media boundary units remain open; exactfullchanged+acceptedunion required before final type-green claim.

- [x] Unit C: match actual SyncMaterializer optional-guard contract for chat/attachment domains by rejecting unsupported non-None guard before access, preserving normal projections; three native callback REDs, truthful Notes/Sync fixture contracts and five existing owning suites. Exact5pathV2 design, dataclass constructor retained with supported typed arguments; no guard/callback bypass or weaker validation.

Unit C local review complete:3trueguardRED/180PASS/newguardbehaviorapproved, native76 still193errors; core22contractfollowupmandatory/ACTIVE, no overallgreen.
- [x] Unit D: exact8existingtestfiles with truthful Optional/native fixture types, realMoodboardcleanup and observedWorkflow persistence/warning behavior; staticRED/native8ownerverification/security/scopedreview.

Unit D local scoped review complete:182PASS plus focused2PASS after three row-presence assertions; native76 193→154,39removed0added/owned8zero. Normal exact10path commit authorized; no new-head proof yet.
- [x] Unit E: exact5path existing Sync identity-boundary checks and native MessageStore declarations;18 actual corrupt-identity RED controls then six native owners including legacy attachment refs, native76/security/scoped review. No new ADR; existing Notes pattern and native stores govern.

Unit E local scoped review complete:18intendedRED/274PASS/native76 154→132 (22removed0added), safe native failure logs/productstate unchanged, three exact static delegates/runtimeASTsame. Reviewed0findings; normalexact7pathcommit next.
- [x] Unit F: exact9testpath same-shape native fixture batch (Native5V3 + backend/migration/transportV2/workflow-hardening), actual120diagnostic target notpassforecast; one deduplicated9owner runtime/native76/security/scopedreview.
- [x] Unit G: three-file native Media binding/cleanup unit, five truefaultREDs+positive subclass control, runtime323identity proof, finite native79 plus relevantSQLite/PG owners/security/review.

Unit F local complete scoped SpecPASS/QualityPASS0findings: actualinitial311PASS9FAIL4SQLite-onlyskip retained, corrected30PASS4skip+2PASS, native76 132→12 removes120/adds0/allowned9zero. Media native-class boundary12 plus separately recorded354implementation diagnostics addressed onlyinUnitG; notwholebackendgreen.

Unit G finite correction before production: initial6BASE2FAIL4PASS disproves ordinary tuple-crash premise. NEW native3.12 ExceptionGroup.split probe shows realnestedmatcherTypeError; exact13 controls characterize7ordinary/subclass passes and6group-boundary faults beforefix. PreserveALLfour existing suppress policies via unpacking, WITHDRAW outer-scope-removal proposal. Native323binding relocation remains; no newADR/securitypolicy. Allinitialfalsepremise/failedcontrols retained, rootactualdiffconcurrence required.

UnitG authoritative groupV2 refinement: THREEcleanup stars +scopeclassic except preserveordinarynoncritical suppression and ALLscopegroup failure outcomes/originalidentity; V1fourstars wouldnewlyhideknownscopegroups andiswithdrawnprivately. RootFULLread13testdiff/productiondiff/source/artifacthashes. Native7PASS6TypeError baseline required, then13GREEN/323identity/native79/owningSQLitePG/security/review. No newADR; sourcecurrentproductionstillunchanged.

All local affected79 Python types nowclean: UnitG348PASS/323nativeidentities/finalsubclass1PASS/scopedreview0findings, sourcequalifications retained. Stage5 integration stillInProgress: freshdev0566ornewer rebase/consequence tests/fullbranchreview/verbatimhumanbody/normalexactleasepush/newheadCI/trustedissuers/reviews/finalready. No merge requested.

Final current-dev integration locally complete: actual41commit rebase onto0566130c/no conflicts/109branch+11upstream paths/27736bindings verified; task filesystem-mode checkout difference restored with original failure retained. Fresh79MyPy0issues and four nativeWorkflowowners44PASS0skip/1369warnings. Final whole-branch review, humanbody publication, exact-lease push and actualNEW-headCI/trustedissuers/reviews/latestdev checks remain. Stage5/AC5/DoD1 InProgress; no merge requested.

Final whole-branch source review complete: SpecPASS/QualityPASS/0findings, all109paths/374evidencehashes; root109source+53evidencebindings0bad. Generated tenant/import freshness23PASS0skip8warnings. Source remains1a4eabf5; only metadata changes. Normal publication and current-head hosted readiness remain open.

### Unit H: actual published-head mobile MCP dialog close failure
Goal: diagnose and repair the new UX Smoke Gate first-attempt close failure through native owners without weakening no-flaky/timeout gates. Status: Complete. Actual4545d538 run39004678117/job117314622343: fourPASS/oneFLAKY; original footerClose leaves MCPdialogvisible, retrypass is not closure. Read-only source/artifact diagnosis first, then exact root-concurred discriminating native regression/fix only if supported; fresh scoped review and new-headCI mandatory. Artifact12090502943 SHA007493f2 native failed screenshot/errorcontext/video vs retry1trace preserved. No product/helper/runner/gate edits authorized by this plan until exact source-backed design concurrence.

UnitH controlled cause proved by ONE native A/B invocation: A PASS; B same hidden assertion FAIL, actual150tool delivery during press movesfooter335.28125px, hitBUTTON→DIV. Root3364baseline+10proposalbindings0bad/all27736trackedrestored/healthbeforeafterstoppedtrue. OriginalhostedpointertimingUNPROVED. Next finite stage: minimal permanenttest-only RED first, then sharedModal fixedavailableheight/flexscrollbody GREEN +sameA/B and unchanged focusedpackage now6cases, relevant quality/types/builds/scopedreview/newheadCI. Exactnewexecutionpackage concurrence pending; no product installation yet. Existing Modal pattern means no new ADR; close/security/API/runner unchanged. Stage5/AC5/DoD1 remain open.
UnitH permanent-regressionRED complete: exactly1newcase/retry0 fails solelyoriginalhiddenassertion after native actualresponsebetweenpress/release; root3000artifactbindings0bad/rawJSON+trace checked. Alltrackedrestored/produnchanged/healthstoppedtrue. SharedModalGREEN +nativeA/B +unchangedfocused6 and finitequality/build/scopedreview/publication/newheadCI remain; no productedit yet.

UnitH locally complete/reviewApproved0C/I/introducedMinor: ownpermanenthidden-onlyRED thenunchangedfocused6firstpass0skip/flaky, A/B2GREEN/footer768stable, sixowners48PASS/bothclienttypes/builds0. FinalnewblocksPrettier-normalized/TSASTsame/original21testsbytepreserved; ESLint0errors/old2+53warnings/wholefileformatdebtqualified. Root26559GREENbindings0bad/reviewer32923baselineREDGREEN0bad; finalsourceModal01dc3814/spec822b4d8f. Review unitH-independent-unit-review.md. Actualfreshdev0566130 unchanged/queueunset/remote4545unarmed. Stage5/AC5/DoD1 remainInProgress: normal4pathrepaircommit/push and actual NEW-headCI/trustedissuers/findings/readiness mandatory. No merge/cleanup. Humanparagraph remainsverbatimpublished; bodytechnicalupdate waits runninggates. No newADR, frontend-onlyBanditnotapplicable.

### Unit I: new-head PostgreSQL DB shard timeout
Goal: resolve the actual d05 full-suite db-management-m-r exit124 without weakening gates. Status: Complete. Native job117328521521 exceeded its unchanged55-minute command limit during PostgreSQL Notes fixture initialization. The abort stack reaches `_sync_postgres_sequences`; this alone proves neither deadlock nor cumulative bottleneck. Raw logs/artifact12094041477 preserved, no JUnit available. Read-only diagnosis first, followed only by an exact source-backed finite regression/repair design, owning native verification, security/quality and scoped review. All seven d05 required gates, native UX6, Notes29 and coverage16 pass; full-suite outcome remains blocked. No timeout/retry/skip/assertion/gate change or product repair from a terminal stack alone. Existing ADRs/fixture/transaction ownership must govern; assess any durable decision before edits. No merge/cleanup.

UnitI chosen minimal fix: split native m-r matrix owner into m/n/o-r across five literal lists/six effective matrices; existing two contract consumers follow. Root240designbindings0bad and independent134file partition89/20/25/all278rootDBfiles exactlyonce verified. All nativecommands/55m/300ssignal/markers/issuers/requiredstatuses/aggregate/concurrencycaps unchanged. Actualsource/product logic stays d05; the larger22production/192testline discovery prototype remains private/uninstalled. Finite affected contract RED/GREEN, native collection oldvsnew union, shardcoverage/actionlint, newactual70/accepted80 types, testquality/security and scopedreview required before publication/newheadCI. ADR required:no, existingpartition mechanism/TASK13696 creates no new durable gate/architecture rule; ADR002/059 tracking and ADR004humanSummary remain. Stage5/AC5/DoD1 InProgress; no merge/cleanup.

UnitI follow-up: exact21 newly exposed preexisting CI-contract type errors require native YAML mapping assertion and truthful list/set local annotation. Root read two-site4added/2removed diff and verified10bindings; original MyPy failure retained. Refresh affected two-contract owners/fresh80 and outstanding quality/security only; unchanged DB collection/coverage/actionlint evidence carries forward. No product change or new ADR.

UnitI local complete: two intended contractRED; final92PASS0skip6warnings; exact2497disjoint collection; native shardcoverage0newuncovered/actionlintPASS; corrected80MyPy0issues. Ruff0/Black inherited-onlydebt/Bandit2newLOWtestassertions qualified. Root364executionbindings0bad/independentSpecPASSQualityPASS0C-I-introducedMinor; review unitI-independent-unit-review.md. Normal exact4path commit/publication and actual final-head all208shards/aggregate/trusted7/fullnativeowners remain required; Stage5/AC5/DoD1 open.

Final functional source completion at dafd27c31caf3aef16f4eb4587c1a0f8e27088f4/dev0566130c: actual208/208Ubuntu3.12shards plusaggregate/downstreamSUCCESS; all7trustedrequiredgates,Notes29/UX6/coverage16/native70types/integrations/all8Workflows succeeded. NewDBowners2489PASS8explicitSKIP=2497; actualJUnit176NotesOwner/83PG and13newcleanupPASS0skip. Root6850hostedreceipts0bad; TASK13816AC5/DoD1complete/Done bounded scope. Final metadata keeps all source unchanged and is governed by its own native exact-head checks; separate readiness report follows those checks. Existing warnings/skips/Black debt/retentionfaults256/noUXpercaseJSON/OSprintdevice/historicalcause limits retained. No merge/arm/cleanup or wider debt closure.

### Unit J: actual final-head NotesGraph permission coverage failure
Goal: resolve actual4490 shared-ui-coverage6 failure at initial403alert beforeRefreshclick without weakening permission/privacy/error/test gates. Status: In Progress. Sourceidenticalfullyverifieddafd nevertheless1432PASS1FAIL/line177genericloaderrorvspermissionText; rawjob117373969810/verifiedartifact12098815128 retained. Read-only completeowner/service/query/mock/notification diagnosis first; exact source-backed finite control/fix design/rootactualdiff concurrence, meaningful RED/GREEN/security/quality/scopedreview and actualnewheadCI required. No blindwait/timeout/retry/assertion/gate relaxation or repeated oldunits; all other4490checkscontinue. ADR assessment pending actual design; no architecture change inferred. AC5DoD1 reopened/no merge/cleanup.

UnitJ exact finite design: rootread nativecached403/500control andminimalsharedhookclassificationdiff; publicQueryClient/realservice/offlinemount, no scheduler/private-statepatch. Execute2selectedREDcontrol thenconditionalfix2GREEN and3existingowningfilesonce, bothclienttypes/installedquality/scopedsecurityreview; old18testdefs/assertions/timers/config unchanged. Invariantproof doesnotidentify historicalfailedqueryerror or scheduler; native4490failure retained. No newADR(existing403boundary); TaskAC5DoD1openuntilactualverification/review/newheadCI.

UnitJ localverification complete: native403RED/500PASS then2GREEN, deduplicated3owners62PASS0skip; actualWebtypes0/extensionnativecompileconfig0/installedESLint0. Original18testsbytepreserved/finalformat3274nodeASTsame/newfragmentsconform/root95bindings0bad. Scopedindependentreview active; normalpublication andactualNEW-headCI mandatory. Historical4490scheduler/cause remainsunproved; AC5DoD1/Stage5InProgress. TS-onlyBanditNA/existing403boundary/noADR. No merge/cleanup.

UnitJ scopedreview complete: SpecPASSQualityApproved0Critical0Important; inheritedlintstderr/wholefileformatdebtMinorqualified. Reviewer95bindings0bad/root4reviewbindings0bad/original18testspreserved. Normalexact4pathcommit/push andactualNEW-headCI required; Stage5/AC5DoD1remainopen.
