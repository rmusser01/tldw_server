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
- [ ] Normal exact-lease publish; require actual new-head required statuses and relevant full-suite/coverage/Notes outcomes, review findings and trusted issuers.
- [ ] Final latest-dev/head/queue/rules/readiness verification; retain human-authored Change summary as required external input before merge. No merge is requested by this continuation.

ADR check for PR3253 CI continuation: ADR required: no; generated location refreshes, EOF and existing behavioral test observations introduce no durable architecture or workflow rule. Existing ADR-059 governs canonical task tracking.

### Actual changed-Python type scope continuation
- [ ] Clear the actual hosted 63-file scope (28 app + 35 test files), locally reproduced 331 errors in23files using unchanged native flags; no whole-backend claim. Original32 means18app+14tests; boundary union36 means22app+14tests, not36productionfiles.
- [x] Unit A local source: independent review0C/I/M; actual2RED/224PASS and final2PASS; native64 removes39production diagnostics/292tests remain. Existing AllowedAction annotations, truthful optional generation models, narrowed credential runtime and native claims callback compatibility. Two canonical/full-positional regression REDs before implementation, deduplicated six owning suites, native type delta and security/quality review.
- [ ] Unit B: source-backed existing static DB delegate signatures, then finite remaining fixture contracts; no runtime wrappers, blanket Any/casts/ignores or gate changes. Actual diagnostics and caller behavior govern each patch.
- [ ] Independently review new units, normal commit/push, require new-head CI; publish supplied human Change summary verbatim after running gates settle, then final fresh dev/rules/reviews/readiness.
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
