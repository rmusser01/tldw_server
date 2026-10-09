# Knowledge corrective PR3214 CI closure

TASK-13534. Scope: actual hosted backend failure and newly observed advisory diagnostics on9a2f916262. Existing user authorization covers required CI repairs. Human Change summary remains required; no merge or gate bypass.

ADR assessment: no new durable decision. Existing ADR026/031/034/042/059/065/066 and the repository ratchet procedure govern; no helper, workflow, rule or exception expansion.

## Stage 1: Diagnose
**Goal**: reproduce the ratchet failure and compare type diagnostics against exact dev sources with identical local rules.
**Success Criteria**: unchanged guard AST/finding-count proof; explicit introduced and retained diagnostic inventory with environment qualifications.
**Tests**: canonical ratchet RED; mypy2.4.0 common20 inputs on base/current and full22 current inputs.
**Status**: Complete

## Stage 2: Repair
**Goal**: apply only demonstrated owning corrections through existing mechanisms.
**Success Criteria**: canonical baseline maintenance changes only one existing line position; any introduced type diagnostics are resolved without ignores or policy changes.
**Tests**: exact baseline diff; all tenant-isolation ratchets and affected owning tests for any source changes; touched-scope security and canonical hooks.
**Status**: Complete

## Stage 3: Historical workstream diagnostics
**Goal**: repair demonstrated diagnostics owned by the earlier Knowledge PRs rather than treating current-dev matches as proof of pre-workstream debt.
**Success Criteria**: exact historical-source and line ownership receipts; existing Notes facade/storage, Sync validators and HTTP lifecycle contracts reused; all82assigned diagnostic instances resolved or specifically adjudicated with evidence, without ignores/casts or gate relaxation.
**Tests**: current identical-rule type receipt; owning Notes/Sync/probe/HTTP tests; targeted RED/GREEN for any behavioral validation change; touched-scope security/lint/format/hooks; one scoped review of this newly discovered unit.
**Status**: Complete

## Stage 4: Publish and verify
**Goal**: normal commit, actual latest-dev rebase and exact-head hosted verification.
**Success Criteria**: accurate audit/task13514 tracking, seven statuses and human-owned summary before merge; retain original failures and named follow-ups.
**Tests**: exact-head CI and actual comments; no stale-head acceptance or auto/admin workaround with queue unset.
**Status**: In Progress

CI scope continuation: consolidate6new HTML-prefix functions into the existing registered7-function security suite; preserve all cases/decorators/assertions. Actual unchanged contracts and owning tests255pass;21base/22current mypy504→501existing signatures with0introduced, not all-green. Fresh annotation owner1pass/bytecode-equivalent;51tenant tests and scopedRuff/Bandit/Black recorded. Original hosted/local failures and intermediate counts retained. Independent CI-unit review Approved at8822898f96 after root36entry hash verification. Actual fetch/rebase against dev97ea9cd5 and normal push/body update completed; fresh final-head CI, human-owned summary and normal merge remain pending. Existing501advisory diagnostics remain unresolved under TASK13534; no type-green or whole-task completion.

Historical extension after7775: exact unchanged-line/message plus AST-context mapping finds405pre-workstream matches.79suspect diagnostics last touch is in actualPR3205/3213/3214 exclusive commit sets;17others retain explicit provenance. Last-line membership is not causal proof; diagnose before fixes. Prior501baseline matching is limited to current dev, not proof of predating the workstream. Existing approved CI unit/client review stays approved; this is a new observed historical diagnostic unit, not a repeated broad/final UI review.

Historical repair candidate: all82 assigned instances removed; expanded645→522/0introduced and original22 projection501→391 are distinct nonadditive scopes. Current522 diagnostics remain red and individually retained (510exact pre-workstream/2moved historical signatures/10other-history). Final400connected before Notes-only changes and168finalNotes afterward pass with overlapping scope,9new RED/GREEN cases, scopedRuff0/seven-pathBandit0; four existing whole-file format failures and warnings remain qualified. Root29source/config+68receipt verification has0mismatches. Stage3 awaits its single scoped independent review; Stage4 awaits normal commit/publication, fresh dev/exactheadCI/human summary/merge.

Historical-unit review atc6c48b7dbb SpecCompliant/QualityApproved,0Critical/0Important/1Minor warning visibility retained; source/receipt/review binding independently verified. Root publication ratchet29exact exits0; no baseline edit. Configured canonical hooks explicitly passed (local Git pre-commit entrypoint absent); normal commit/no bypass. Stage3Complete. Stage4 remains InProgress for newhead publication/rebase/CI/human summary/merge; previous7775seven statuses allSUCCESS are historical.


## Task 5: Correct newly completed broader CI failures

**Goal**: fix demonstrated test ownership/configuration defects and the existing installer's missed active mirror-list file before publication. This is a new CI unit after the previously approved historical repair; the earlier product/client reviews remain source-bound and are not repeated.

**Spec**: Docs/Design/2026-10-08-knowledge-mechanism-correction.md. Associated TASK-13534 already records these failures and supplied human summary. ADR assessment: no new durable rule; existing egress, inventory and bounded installer mechanisms remain authoritative.

**Frozen BASE**: 3ebe3d82b699cfa8ceba1cd6f3865d5ad01f085e, actual clean rebase onto dev cd5160201cb32e05d254accdd8ac1377e1d164ad. All25 earlier patches range-match exactly;29 earlier reviewed source/config hashes unchanged.

**Observed failures**:
- integrations:6failed/3362passed/46skipped. Five new public HTTP tests use public.example while the actual workflow WORKFLOWS_EGRESS_ALLOWLIST admits93.184.216.34,does-not-resolve.invalid,example.com. Real policy rejects that fixture before send. The sixth failure is the canonical WebScraping import inventory: three WebSearch_APIs import line positions moved by eight after this workstream changed the owning source.
- media-ingestion-modification: apt update succeeds but FFmpeg/PortAudio install times out within the existing600-second deadline after three attempts. No Python/test execution. Existing actions normalize classic sources but miss the runner's active /etc/apt/apt-mirrors.txt. Original logs, bounds and skipped-test facts retained.

**Global Constraints**:
- Reuse existing monkeypatch/egress test configuration and canonical inventory generator. No production egress change, policy bypass, private-network allowance, relaxed host guard, or mocked policy success in place of real transport assertions.
- Regenerate only the owning canonical JSON/Markdown import artifacts with the existing helper. Compare precise record differences; retain the test and its assertions.
- Extend the existing Azure-to-archive normalization in both setup-ffmpeg and wait-for-postgres to the active apt-mirrors.txt file. Keep current classic-source/Microsoft handling, archive target, install packages, return/failure semantics, apt-bounded helper and every existing timeout/deadline/retry. No replacement installer, new helper abstraction, new config option, dependency or workflow/matrix/gate change.
- Add a bounded temporary-file regression in the existing CI test home that executes the actual owning action preflight against mirror-list and classic-source fixtures, with absent-file behavior and both callers. No writes to host /etc, actual apt installs or network in the unit test.
- Preserve all original failed receipts. Verify narrow RED/GREEN first, then finite owning tests and workflow contracts once. No broad client/browser/build replay: affected product inputs are unchanged. Run scoped lint/format/security applicability and existing hooks before committing; no bypass flags.
- Never modify the separate primary checkout, its rebase/conflict, shared services or shared venv.
- Implementer does not dispatch agents, stage, commit, rebase, push, publish or merge. Root owns tracking, commit and landing; report exact files, commands, outcomes, risks and SHA256 receipt bindings.

**Tests**: controlled exact-CI allowlist RED then GREEN for new public transport tests (keep body limit, pinning, credential clearance and response cleanup assertions); canonical import-artifact test RED then generated GREEN; actual action mirror-list RED/GREEN; owning transport/inventory and CI helper/workflow-contract scopes; shell syntax and lint/format applicability. Do not label hosted full-suite green until exact-head CI reaches pytest and completes.

**Status**: Complete; independent spec/quality review Approved at6f0db6dccc with0actionable findings. Final canonical owning166passed/0skips; actual new-head hosted installation/shards/checks remain Stage4 landing obligations. Single task-scoped independent review after the minimal correction; earlier broad/final product review is not repeated. Stage4 remains InProgress for correction, publication and actual exact-head latest-dev landing.

Task5 review at6f0db6dccc SpecCompliant/QualityApproved,0Critical/0Important/0Minor. Raw test-only Bandit assertions/fixed trusted preflight call and unchanged Black baseline explicitly reviewed; no suppression or gate change. Current human Change summary is supplied and posted verbatim. Stage4 remainsInProgress for actual publication, latest-base/exact-headCI and normal merge; residual522backend type diagnostics/criterion5remainopen.


## Task 6: Repair proved descriptor ownership defects in fixture tests

**Goal**: repair demonstrated test-owned descriptor corruption discovered while investigating the new exact-head integration setup error. This is a new finite CI unit; Task5 and all earlier product/client/historical reviews remain approved and are not repeated.

**Spec**: the original corrective spec's existing-mechanism and verified-CI requirements, plus the owning fixture suite's existing native close, descriptor ownership and exception-precedence contracts (commit29603c9b7f). Associated TASK-13534. ADR required:no: test instrumentation and cleanup corrections restore existing ownership rules; no public/runtime/persistence/provider/security/workflow architecture rule changes.

**Frozen BASE**: a77961c344a65cb3a44f5bc11071742133de2800. Both fixture generator and test are identical to fetched dev. Historical provenance is not proof of harmlessness or permission to ignore the failure.

**Observed evidence**: integrations3370passed/46skipped/11743warnings/1EBADF setup error; all7required statuses pass. Media hosted setup/test success is separately retained. Controlled original-body probes prove native scandir corruption while original assertions pass in five owning tests: released-slot reuse, no-dirfd fallback cleanup, root-close cleanup (both parameters), fdopen cleanup, recovery-write cleanup. Later two run after the failing hosted case and cannot explain its earlier setup. Exact historical interleaving remains unproven; the passing179-case module baseline does not close it. Preserve original probe v1 failures, corrected probes, hosted logs and baseline warning/logging qualifications.

**Global Constraints**:
- Only the owning test_phase4_fixture_generator.py and minimal behavioral regressions in the same existing test home may change for implementation. Root owns plan/audit/backlog records. No product generator, provider/transport/store/framework/dependency/config/workflow/matrix/gate change.
- Reuse existing _OwnedDescriptor tracking, closefd=False views, native dup2 on still-owned slots and module-confined SimpleNamespace OS spies where appropriate. Do not force reuse into a released slot or clean up a saved integer after ownership has been released.
- Preserve actual native close/resource-release, descriptor-number reuse, exception identity/precedence/traceback, sensitive-detail refusal and publication/rollback assertions. Replace any unsafe stale-integer fstat check only with existing owner-release/real-close evidence of equivalent behavior; do not delete its tested contract, skip cases or accept a mock-only substitute.
- Repair only the five proved unsafe owning paths; preserve recovery-read/fsync controls whose injected errors occur before native close and legitimately retain cleanup ownership. Narrow full-file caller classification must show other close/dup2 paths retain their ownership guards.
- Write behavioral interleaving regressions first and record expected RED before fixture repair, then GREEN. Exercise actual native unrelated directory/file ownership in deterministic bounded tests through existing pytest/monkeypatch mechanisms; no line-number tracing or new committed test runner/harness.
- Final verification: new regressions, entire owning module once, original hosted failing node and finite existing fixture architecture/security neighbors selected by actual changed contracts. Keep project importlib/strict-markers/addopts/300-second-thread-timeout and actual assertions; explicitly load existing asyncio/timeout plugins when autoload is disabled. Scope runtime/database/cache/log paths with existing env options to private temp; preserve original logging PermissionError diagnostic and verify existing SYSTEM_LOG_FILE_PATH relocation, do not disable logging.
- Run changed-scope Ruff/format/security applicability and configured hooks; no suppressions or bypass flags. Preserve raw baseline findings and qualifies; no false clean/type-green/broad-CI claim. No broad frontend/client/browser/build/type/history replays for unchanged inputs.
- Worker may not spawn agents, stage, commit, rebase, push, publish, merge or modify primary/shared services/venvs/Git GC. Stop after3failed attempts and reassess. Root performs one independent task-scoped spec/quality review before publication.

**Tests**: original-body native scandir interleaving probes retained as diagnosis; new finite regression RED/GREEN for all proved unsafe paths including both root-close cases; controls for cleanup before native close; owning module and original hosted failing node with canonical settings; unchanged generator/config/previous approved source bindings; exact-head hosted full-suite and required checks after normal latest-dev publication.

**Status**: Complete; independently SpecCompliant/QualityApproved at823922c480,0Critical/0Important/1Minor warning-visibility finding. Six suppressed warnings remain qualified; equal baseline counts do not prove identical categories or causes. Canonical owning union185passed/0skips/errors; source/config and executed receipts independently bound. Exact historical hosted interleaving remains unproven. Stage4 remainsInProgress for actual latest-dev publication and new-head hosted full-suite/required statuses before merge.
