# PR 3434 CI Monitoring Repairs

Tracked by TASK-15466. The requester approved CI/review repair and gated auto-merge;
the human-written Change summary is explicitly waived, not other quality gates.

ADR required: no. This restores existing test, publication, and shard contracts.
ADR-094 governs shared Jobs ownership; ADR-095 stays Proposed. No new architecture
or supported SDK/platform promise is introduced.

## Stage 1: Diagnose Current Failures
**Goal**: Establish the exact current-head CI failures and actionable feedback.
**Success Criteria**: Reproductions distinguish runtime defects from test/publication omissions.
**Tests**: Failed SQLite race case, docs manifest/strict build, shard coverage guard.
**Status**: Complete

Observed additional complete-CI failures: DB partition expected assignment list,
missing RLS registration for both publication tables, and ten test-quality
offenses across eight tests. These are included in Stage 2 without raising
baselines, adding exclusions, or weakening gates.

## Stage 2: Repair Validated Causes
**Goal**: Make the smallest corrections without weakening any gates or assertions.
**Success Criteria**: Original repros pass; race coverage controls actual transaction boundaries.
**Tests**: Red/green regression, applicable Jobs module, docs suite, all shard matrices.
**Status**: Complete

Evidence: deterministic two-case SQLite race RED (both admissions escaped), then
90-test module GREEN; dependency/active/archive state preserved. Docs 213 passed;
shard guard zero newly uncovered; five files exactly once across six matrices.
Workflow contracts 60 passed after aligning the explicit expected partition list.
Both backend poison gaps reproduced RED and now pass 207 coordination/prune/dependency
tests. Shared RLS repair passed 130 scoped tests, including 16 restricted-role
migration/repair cases. Quality cleanup passed all 350 selected SQLite cases;
the entire CI-contract suite passed 984 cases with six existing skips.

## Stage 3: Review And Publish
**Goal**: Independently review corrections, verify and publish a merge-ready head.
**Success Criteria**: Review findings resolved, evidence recorded, clean scoped commit, auto-merge rearmed with exact head.
**Tests**: Scoped suites, compilation, lint, Bandit, canonical task format and full PR-scoped hooks.
**Status**: In Progress

## Stage 4: Monitor Gated Merge
**Goal**: Follow new CI/review outcomes until GitHub confirms the final state.
**Success Criteria**: Record actual merge SHA/time and stop monitor on MERGED/CLOSED or required user action.
**Tests**: Fresh GitHub state, feedback, effective rulesets, exact head and base verification.
**Status**: In Progress

Automation `monitor-pr-3434-through-merge` is active. Auto-merge is disarmed
until the corrected, independently reviewed and verified head is published
and that exact head's ordinary hosted notes/chat UX regression job succeeds.
Actual MERGED state, SHA and time remain pending; do not claim them early.

### Latest Dev Requalification

On 2026-11-08, dev advanced to `7a020da14ab28e220fab652269723e3a2a4abbcb`.
Auto-merge was disarmed before the clean rebase. All 48 prior PR commits remain
patch-equivalent; the recovery branch retains published head `7e44324c2e`.
The dev change removes endpoint-local test-mode predicates and updates their
tests; repository instructions and test/build configuration are unchanged.
All 146 PR-owned paths are byte-identical before and after rebase; none overlaps
the 32 upstream paths. Existing independent reviews remain applicable.

Fresh qualification from the root venv completed:
- Claims/publication/RLS plus both-backend coordination/prune/dependency/TTL:
  1,923 passed, one existing spaCy skip.
- Full Jobs SQLite selection: 2,539 passed, two inherited crypto skips;
  1,560 PostgreSQL-marked cases deselected by the normal SQLite selection.
- Complete CI/Docs/Config, Audit/backup and endpoint regressions: 1,739 passed,
  six existing runner/environment skips.
- Additional official PostgreSQL MediaFiles and cluster publication/snapshot:
  105 passed, zero skips. No PostgreSQL availability skip in either cohort.
- Chroma manager: 33 passed, one legacy skip, 15 inherited xfails, two xpasses.
- Compile 108 files; full configured Ruff 113 versus dev 116 with no new/worse
  signatures; Bandit 58 production files, one identical inherited B608,
  zero errors/new findings; 15 PR-touched task records canonical.

Selections overlap and are not a combined-suite total. An initial invocation
with a nonexistent test path collected no tests; its corrected selection passed.
Bandit JSON was rerun with an explicit output file after its progress-prefixed
stdout defeated the collector. Neither required repository corrections.
The known full-root missing-pg_dump prerequisite was not rerun or called green.

Refetched dev and the remote PR head remain unchanged. All applicable full
146-path staged PR-scoped hooks passed; only no-matching TOML and wizard-only
Ruff/Black hooks skipped normally. Exact-lease publication and exact-head MERGE
rearming follow; their readback is recorded on PR #3434. Monitoring remains
active until GitHub reports the actual final state. No required CI gate or
user-summary waiver is changed.

### Scope Ratchet Address Refresh

Current-head `backend-required` failed the tenant-isolation step: one failure,
73 passes. Local RED reproduced the scope-ratchet failure with 12 other passes.
Both the live scan and baseline contain 28 entries. All ten reported additions
are existing owner guards at moved source lines; each complete `ast.If` subtree
and enclosing qualified function matches latest dev exactly. Ten stale addresses
map one-to-one to the ten current addresses. No new predicate was introduced.

Refresh only those ten line numbers. Keep all 28 predicates, the scanner, tests,
thresholds and runtime behavior unchanged. Re-run the exact 74-test tenant gate,
independently review the address mapping, check canonical tasks and full staged
PR hooks, then publish normally and rearm exact-head MERGE after fresh checks.
ADR required: no, this is non-behavioral ratchet maintenance. Bandit is skipped
for this text-only delta; unchanged production evidence remains applicable.

The exact CI step is GREEN: 74 passed, zero failures/errors/skips, 237 warnings
in 108.26 seconds. Standalone scanner reports all 28 entries exact. The proof
artifact `/tmp/claims-pr3434-scope-address-proof-20261108.json` records the ten
complete AST hashes and one-to-one address mappings; 18 entries are unchanged.
Independent review Zeno approved with no actionable findings and additionally
verified all 28 subtree/function identities and unchanged enclosing functions
for the ten moved entries. Report:
`/tmp/claims-pr3434-scope-ratchet-review-20261108.md`.

All applicable full 147-path staged PR hooks passed; 15 task records remain
canonical. Only the ten addresses and this task/plan changed; app/test/workflow
and scanner source remain identical to published head `d684526a20`. Existing
runtime approvals and qualification transfer unchanged, not a new runtime audit.
Publication and exact-head auto-merge readback will be recorded on PR #3434;
monitoring/task remain active pending actual merge. No required gate is waived.

### Current Code-Contract Gate Repair

On 2026-11-08T08:40Z, published head `6b4ced6123` failed the complete code-contract
step (five failures, 1,075 passes), after the repaired tenant gate passed.
Auto-merge is disarmed before edits. Reproduce the current function-size,
layering, DB-import-cycle, core/API-boundary and endpoint-getter ratchets; restore
their existing contracts without baseline growth or scanner/test exemptions.
Keep lifecycle and transaction fences in shared Jobs. Use scoped behavior tests,
official PostgreSQL fixtures, compile/lint/Bandit, canonical task checks, full
staged PR hooks and independent review before publication/rearming.
ADR required: no new decision; these corrections restore existing module
boundaries, DI and code-size constraints under the existing Jobs ownership rule.
Reassess if a durable architecture change is needed. Earlier runtime evidence is
historical, not certification of the forthcoming source changes.

The stable repair restores all original ceilings and import boundaries. Shared
Jobs failure orchestration is 185 lines; new helpers are at most 149 lines and
retain all 47 SQL call shapes. Test-mode policy stays in manager callbacks
(38 signal lines under the unchanged 41 ceiling); the new module has none.
Remove only the stale fail_job size exception. Seven existing owner guards moved
lines with identical complete ASTs and enclosing functions; all 28 entries stay.
The supported old import paths alias the same lower-level definitions. Lazy
manager construction and admission remain together in their executing worker.

An intermittent timeout-fixture input failure was also observed. Deterministic
global-config refresh reproduced the same 8.0 versus 0.01 symptom. Bind test-owned
consumed settings, preserving exact timeout/cancel/nonwaiting-shutdown assertions.
RED was one failure/five passes; its closure now passes 14 cases. The historical
intermittent trigger is not claimed proved; production ingestion is unchanged.

Final stable-source verification:
- Full Jobs SQLite: 2,543 passed, two inherited crypto skips; 1,560 normal PG
  marker deselections. Expanded official PostgreSQL and both-backend lifecycle,
  retention, TTL, prune, counters, schema and restricted RLS: 471 passed, no skips.
- Complete Claims/publication/snapshot/owned status/restricted PostgreSQL:
  1,804 passed, one existing spaCy skip.
- Exact backend code-contract step: 1,079 passed, one runner Bash skip, then
  separate WebSocket checks: 193 passed, no skips.
- Complete CI/Docs/Config: 1,613 passed, six existing runner/environment skips.
  Tenant gate plus migration/RLS/startup closure: 109 passed, no skips.
- Compile 117 Python paths; configured Ruff 113 versus dev 116, no new/worse
  signatures; Bandit 64 production files, one identical inherited B608 archive
  query, zero scan errors/new findings. All 15 tasks canonical; all applicable
  full 157-path staged hooks passed without modifying files. Selections overlap.

Five independent requesting-code-review seats approved the new scoped work with
no actionable findings. Parent-helper differential checks, failure SQL/exception
traces and metadata checks, boundary/cold-thread checks, migration/router parity,
and timeout-fixture review are recorded in TASK-15466. Live backend qualification
and final hooks remain parent evidence, not reviewer certification. Other PR
source retains earlier independent approvals. No new ADR or quality allowance.

Honest harness limits remain in the task: initial concurrent-source inspection
failures passed unchanged in a fresh process and final suite; one bad test path
collected zero; the diagnostic trace hit a Hypothesis generation health check,
then the unchanged 142-case module and final Claims suite passed. No health check
or test budget was weakened. The unchanged root pg_dump prerequisite was not
rerun or called green. Old-head E2E Critical, Jobs PostgreSQL and gap-verified-6
also hit GitHub uv-download HTTP 504 before install/tests; no workflow change,
manual dispatch or cancellation was used. The next head must pass its own CI.

Latest fetched dev and remote PR head are unchanged. Auto-merge remains disarmed
for publication; exact-head MERGE rearming/readback will be recorded on PR #3434.
Stages 3/4 and TASK-15466 remain active until actual merge, not merely publication.

### UX Browser Runner Compatibility

On current head `037c979499`, UX Smoke job `128710222829` failed before
browser/backend tests: `ubuntu-latest` supplied Ubuntu 26.04, unsupported by the
locked Playwright 1.58.0. The tagged upstream host-platform implementation
supports Ubuntu 22.04/24.04. Auto-merge was disabled and confirmed null before
edits. Pin only the two UX browser jobs to `ubuntu-24.04`; all parsed workflow
data except those two labels is identical, including every step, command,
condition, trigger, permission, admission dependency, timeout and budget.

Two new runner-contract cases failed RED against floating latest, then the
complete frontend workflow contract module passed all 16 tests. Compilation and
configured Ruff for the changed test module passed; Bandit on that test module
reported zero findings/errors with test assertions (B101) excluded from the scan.
No production code changed; earlier production/runtime qualifications remain
applicable, not newly rerun or relabeled. No quality baseline or scanner changed.
ADR required: no; this is a bounded CI platform compatibility correction.

Independent requesting-code-review Schrodinger approved this narrow delta with
no actionable findings. Report: `/tmp/claims-pr3434-ux-runner-review-20261108.md`.
The reviewer independently checked workflow parity and the regression's
red/green behavior. Prior full-PR reviews cover unchanged runtime. Parent owns
the broader CI/Docs/Config, static and full staged-hook evidence. Hosted Chromium
installation/browser execution remains for the new head's ordinary CI; it is
not certified by local YAML tests or this review.

Onboarding job `128710222834` separately failed Bun installation with one unnamed
package; its root cause is unproved and is not claimed fixed by the runner pin.
Neither job reached application assertions; later always-run health checks had
no started backend. No speculative dependency change, manual gate dispatch or
cancellation is used. Logs and proof use `/tmp/claims-pr3434-ux-*20261108*`.
The unchanged full-root missing-pg_dump prerequisite is not rerun or called green.

Complete CI/Docs/Config qualification passed 1,615 tests with six inherited
runner/environment skips and zero failures/errors. This overlaps the 16-test
module run and is not a combined total. All 15 PR-touched task records remain
canonical. All applicable final staged 159-path PR-scoped hooks passed without
file rewrites; only ordinary no-matching TOML and wizard-only Ruff/Black hooks
skipped. These qualifications are in TASK-15466. Exact-head publication/rearming
readback follows in the PR proof, not a tracking-only commit while waiting.

### E2E Tier Browser Host Closure

On head `a4f2a70f5a`, notes/chat E2E UX Regression job `128736888966`
failed before tests with the same explicit locked-Playwright/Ubuntu-26.04
Chromium-install rejection. Its log is
`/tmp/claims-pr3434-e2e-ux-128736888966-20261108T1327.log`.
All four browser jobs in `frontend-e2e-tiers.yml` invoke the same locked JS
toolchain. After disarming auto-merge and confirming null, extend the existing
runner contract to those jobs, then pin critical/features/admin/ux-regression
to `ubuntu-24.04`. Change detection and queue-tick remain `ubuntu-latest`.
Every other parsed workflow value is identical, including commands, gates,
conditions, permissions, admission, budgets, retries and timeouts. No dependency,
runtime, scanner or baseline changes. ADR required: no; bounded CI host repair.

RED reproduced four expected failures and two existing UX passes; the complete
frontend workflow module is GREEN with 20 passes. Changed test compilation,
configured Ruff and non-assert test-module Bandit (B101 excluded from that scan,
without configuration/suppression edits) passed with zero findings/errors.
Earlier production/runtime qualification remains applicable, not rerun or
relabeled. Hosted browser success still requires ordinary new-head CI. The
older unnamed Onboarding package-install failure and root pg_dump limit remain
unproved/unchanged, not claimed fixed or green. Evidence prefix:
`/tmp/claims-pr3434-tier-runner-*-20261108`.

Independent requesting-code-review Ampere approved the narrow tier-runner delta
with no actionable findings, independently reproducing four RED failures/two UX
passes, then all 20 GREEN cases and complete parsed-workflow parity. Report:
`/tmp/claims-pr3434-tier-runner-review-20261108.md`. Prior whole-PR reviews cover
unchanged runtime; parent owns broad CI/static/security/hooks and exact GitHub
state. Neither review nor local contracts certify hosted browser execution or
the older unnamed package-install failure. All 15 task records are canonical;
the full PR scope now has 160 existing paths.

Final stable-source CI/Docs/Config qualification: 1,619 passed, six inherited
runner/environment skips, zero failures/errors. This overlaps the 20-case
module and is not a combined total or full-root pass. All applicable final
160-path staged hooks passed without rewrites; all 15 tasks remain canonical.
Only ordinary no-matching TOML and wizard-only Ruff/Black hooks skipped.
Exact-head publication/rearming readback follows in the PR proof, not a
tracking-only commit while waiting for ordinary CI and actual gated merge.

### OpenAPI Artifact Publication And Dev Freshness

At 2026-11-08T15:20Z, seven required contexts passed on `ec326c5e88`, but
backend-required job `128761640996` failed its OpenAPI drift gate. The exported
contract has 2,141 paths and 3,319 schemas versus the committed 2,140/3,316.
The intended delta is the Jobs-owned status path, three typed response schemas,
and the already-reviewed rebuild/status operations. No runtime correction or
gate/budget change is proposed. Reproduce the exact gate, compare complete
base/head schemas, regenerate both artifacts and rich frontend types through
the existing generator, then verify and independently review the delta.

Auto-merge was disabled and confirmed null before edits. Dev advanced to
`2ec4e6434d6c6b73b402ee77a731911b85c307a2`; the dedicated branch rebased all
53 commits without conflicts. All 160 existing PR-owned blobs are identical
before/after; none overlaps upstream's 50 changed paths. Recovery ref:
`codex/claims-pr3434-before-dev-20261108t1520`. Existing scoped runtime reviews
retain their original meaning. Only an exact fresh remote-head lease may
publish this rebase after canonical task checks and staged full-scope hooks.

Native Python 3.14 reproduces the count/path drift but has a different full
schema hash from hosted Python 3.12. A temporary FastAPI patch-version comparison
produced identical native schemas, so no dependency upgrade is justified. Match
CI's Python version in an isolated temporary environment before choosing a new
fingerprint; do not copy an unverified hash or weaken canonicalization. Diagnostic
dev snapshots live only under `/tmp`; the first archive command used the wrong
cwd, and the next lacked the standalone MCP package. The completed snapshot
exports successfully after including its shared packages. These harness errors
are not green gate runs. The older missing-pg_dump root limit remains unchanged.

The isolated Python 3.12 environment reproduces hosted RED exactly. Its dev
export hashes to the old committed `f76f17d0de73...`; its PR export hashes to
hosted `a6a96d3dd761...`. Existing codegen regenerated both committed artifacts
and ignored rich types successfully; the unchanged exact gate is GREEN for
fingerprint and path union. Complete proof:
`/tmp/claims-pr3434-openapi-delta-proof-20261108.json`. Native/CI differences
are only 16 existing HTTP422 descriptions and equivalent anyOf ordering in
five schemas, not new API changes or grounds for normalization allowances.

Fresh qualification: both affected API modules 119 passed, zero skips,
including official PostgreSQL cases; complete CI/Docs/Config 1,619 passed with
six inherited runner/environment skips, zero failures/errors. The generated
rich types/path union pass a temporary consumer probe under repository-locked
TypeScript 5.9.3, including legacy 200, typed 202/409 and UUID status contracts.
Evidence prefix: `/tmp/claims-pr3434-openapi-*-20261108`. Selections are not a
combined/full-root result. No executable source changed, so prior compilation,
lint and 64-file production Bandit evidence remains applicable, not rerun or
relabeled; the artifact-only delta does not require a new Python security scan.

Independent requesting-code-review Archimedes approved with no actionable
findings: `/tmp/claims-pr3434-openapi-review-20261108.md`. Hash/delta and source
alignment checks were independent. Its restricted fresh export omitted two
MediaWiki routes and schemas with permission/logging limits and did not pass
the gate; cause unproved. Do not call that run green or substitute it for the
parent's successful full export. Declined scopes are dispositioned in TASK-15466:
unchanged runtime retains earlier reviews; parent owns environment provisioning,
codegen/consumer probe, API/official-PG and broad CI results, rebase parity,
canonical tasks, full staged hooks and exact GitHub publication state. Hosted
new-head CI still must pass normally, and actual merge remains pending.

Stage the verified canonical task/plan and generated artifacts before full
162-path PR-scoped hooks. Publish only against a freshly fetched explicit
remote-head lease, then rearm exact-head MERGE with fresh rules/dev/feedback
checks. Record actual hook/publication readback in the PR proof; never create
a tracking-only follow-up while waiting for unchanged CI.

All applicable full 162-path staged hooks passed without rewrites; all 15
selected task records are canonical. Only ordinary no-matching TOML and
wizard-only Ruff/Black hooks skipped. Fresh remote head and dev remain exact,
with auto-merge null and dev ancestry verified. Re-stage this final evidence
before the final full-scope rerun and normal-hook commit. Publication/rearming
readback belongs in the PR proof; actual merge is still pending.

### XS-07 Search Settlement Investigation

Current `da753fbd29` E2E UX Regression job `128783776748` reached application
assertions: 31 passed and XS-07 Delete failed. Server DELETE succeeded with 204,
the tab closed and GET returned 404, but the test immediately captured a cached
Delta search row at 16:17:46.033Z. The failure screenshot still says
"Searching chat history..."; the refreshed search completed at 16:17:46.748Z.
Artifacts: `/tmp/claims-pr3434-e2e-ux-artifacts-20261108T1630`; exact job log:
`/tmp/claims-pr3434-e2e-ux-128783776748-20261108T1630.log`.

Auto-merge is disabled and confirmed null before repository corrections. Dev,
rules and feedback are unchanged. Independent read-only diagnosis confirms the
snapshot was unsettled, not that eventual correctness or a production bug is
proved. The hook invalidation prefix matches; cached rows may render while a
refresh is pending. The test accepts either a cached row or no-matches before
snapshotting, so it has no positive settlement barrier. Reproduce the unchanged
case with the existing live-tier runner and real isolated backend/extension;
do not alter production invalidation or relax assertions without evidence.

The frozen Bun 1.3.5 workspace install succeeded without lock/source changes.
No sleeps, retries, skips, timeout/budget changes, baseline allowances, manual
dispatch or cancellation are authorized. Preserve real server-deleted, tab
closure, and final zero-result assertions. A test-only correction must positively
observe settled no-matches rather than merely observe zero rows during loading,
and must still fail for a persistently listed deleted chat. Obtain independent
requesting-code-review for any correction; unchanged runtime qualifications and
the known root pg_dump limit retain their original meaning.

### XS-07 Correction Qualification And Hosted UX Hold

Only the Delete case's permissive cached-row-or-empty wait changes: positively
observe settled no-matches, then require zero matching rows. Server polling,
tab closure and final server-deleted/no-match equality remain unchanged. The
other three tests, production code, helpers, workflow, Playwright configuration,
runner and reporter are unchanged. Assertions retain the configured 20-second
limit individually, not a shared deadline; the test retains its 120-second limit.

The unchanged real-backend/extension case passed once locally (23.5 seconds),
which is not RED or Linux certification. The controlled Chromium probe extracts
the actual assertion tail via TypeScript AST and uses actual page-object locators.
Valid old-source RED reproduces deleted-server/cached-row failure; corrected
GREEN accepts delayed settled-empty and rejects pending-empty, settled-stale,
false-empty-with-row and server-still-live. This is controlled DOM/server-oracle
assertion evidence, not autonomous backend certification. Initial module-path
and asynchronous matcher-wrapper probe errors were corrected before valid RED.
Artifacts: `/tmp/claims-pr3434-xs07-settlement-{red,green}-20261108.log` and
`/tmp/claims-pr3434-xs07-scope-proof-20261108.json`.

The complete unchanged live-tier UX runner finished 31 passed, one failed, zero
skips: corrected Delete passed on its first attempt, but unchanged NE-04 Print
timed out hovering its submenu. The Print cause is unproved, not claimed fixed.
Its old-head hosted pass and a subsequent bounded real-backend run do not make
the full local run green. That follow-up passed all four sidepanel cases plus
the same unmodified Print case: five first-attempt passes, zero skips; backend
health before/after and cleanup succeeded. Grep-mode is bounded diagnostic
evidence, not a full gate. The full run's after-health was not reached; backend
cleanup did succeed. All 13 temporary Deferred43 instrumentation targets report
restored and their live hashes match originals. No retained source changes.
Artifacts: `/tmp/claims-pr3434-xs07-{full-ux,followup}-20261108.log` and
`/tmp/claims-pr3434-xs07-browser-evidence-20261108.json`.

Focused frontend tests passed 134 across four files, zero skips/failures. The
changed TS module transpiles with zero syntax errors and configured ESLint has
zero errors, warnings or suppressions; this is not a whole-app semantic typecheck.
No executable Python changed. Retained production/runtime/lint/Bandit evidence
keeps its original scope and date, not rerun or relabeled for this TS/prose delta.
The old root missing-pg_dump and historical unnamed Bun-package limits persist.

Independent requesting-code-review Pascal approved the exact test-only delta
with no actionable P0-P3 findings: `/tmp/claims-pr3434-xs07-review-20261108.md`.
Reviewer inspected source, hosted screenshot/logs, AST parity and valid probe
RED/GREEN, not its own live test execution. Declined scopes: unchanged whole-PR
production/ownership/Trash/Undo retain earlier reviews; parent owns full/browser
and broad CI results, static/security/hooks, all 13 restoration hashes and Git
cleanliness, GitHub feedback/publication/merge state, recorded waiver, root
prerequisite limits and future tracking updates. Neither reviewer nor parent
claims universal future UI stability or a persistent production invalidation bug.

Dev advanced to `476527719a806e005225fd358b0f830f8b9253df` with five unrelated
docs/task paths and no overlap. With auto-merge confirmed null, all 54 committed
PR commits rebased conflict-free; all 163 captured PR-path HEAD blobs and the
reviewed dirty test hash are identical. Recovery ref:
`codex/claims-pr3434-before-dev-20261108t1700`. Fresh broad CI/Docs/Config passed
1,619 cases with six inherited skips, zero failures/errors (188.33 seconds),
on this base. Evidence: `/tmp/claims-pr3434-xs07-rebased-ci-docs-config-20261108`.
The pre-rebase 1,619/6 run remains separate evidence, not additive or relabeled.
All 15 explicitly selected tasks are canonical; staged full-scope hooks must pass
before exact fresh force-with-lease publication. No main checkout/dev edits.

The new published head must run its own ordinary CI. Keep auto-merge disabled
until its E2E UX Regression (notes/chat) succeeds; then inspect all feedback,
effective rules, queue mode and strict dev freshness before exact-head MERGE
auto-merge. This conservative monitor hold changes no workflow or required gate.
No sleeps/retries/skips, budget/baseline increases, manual dispatch/cancellation,
bot/billing changes or new architecture. TASK-15466 and this plan remain In
Progress until actual gated merge; record final qualification/publication in
the PR proof without a tracking-only follow-up commit.

All applicable full 163-path staged PR hooks passed without rewrites; all 15
selected task records are canonical. Log:
`/tmp/claims-pr3434-xs07-staged-hooks-20261108.log`. Only ordinary no-matching
TOML/wizard-only formatter hooks skipped. Restage this final evidence and rerun
the full scope before the normal-hook commit. Publication/readback belongs in
the PR proof; auto-merge stays disabled for the exact-head hosted UX hold.

### Shared UI Local Selection Investigation

At 18:45Z, current-head shared UI coverage shard 8/16 failed one accepted
LocalChatList selection assertion; 1,891 tests across 173 other files passed.
Job `128816245763`, run `42503741289`, failed waiting for model preference at
`useLoadLocalConversation.test.tsx:294`, after successful install/setup. Logs:
`/tmp/claims-pr3434-ui-coverage8-128816245763-20261108T1845.{json,log}`.
Auto-merge was disarmed and confirmed null before repository edits. No new
feedback or dev/rules changes. Prior exact-head hosted UX success remains real,
but does not override this new failure or transfer to another head.

The unchanged module passes locally: 26 tests without coverage, then 26 with
V8 coverage. These are not deterministic RED or Linux certification. The test,
hook and list match dev byte-for-byte. Investigate a controlled initial-owner
completion ordering: selecting while the reader scope changes must retire the
old read rather than weaken production ownership. This is a hypothesis, not
proof of the original trigger. Add a regression before choosing a correction;
preserve all selection/cancellation/failure assertions and configured budgets.
No production change, dependency upgrade, manual dispatch/cancellation or gate
relaxation is justified. Reopen only the validated-correction stage; independent
review and new-head qualification remain prerequisites for any publication.

Controlled owner ordering now reproduces two failures with 25 passes: the
ready-owner variant stops at the message-read wait, the pending-owner variant
at the model-preference wait. Neither is a final-callback failure. The first
wrong-root-cwd shell launch is a harness error, not RED. Valid evidence:
`/tmp/claims-pr3434-local-selection-owner-red-corrected-20261108.{json,log}`.
The minimal fixture correction completes controlled owner initialization in
awaited act and observes the real public provider lease before clicking. The
complete module then passes 27 tests. All 55 original assertion ASTs remain;
only one owner-readiness assertion is added. No production source changes.

Independent Boole requesting-code-review approves without actionable P0-P3:
`/tmp/claims-pr3434-local-selection-review-20261108.md`. Its precise limitation
governs: provider and list own independent owner hooks; shared/immediate scope
completion and awaited act establish this fixture's happy-path precondition,
not a general production synchronization contract. Pending clicks, independently
delayed production resolutions, later account/route/profile changes and account
unavailability are not recertified. Existing guards and adjacent profile-owned
browsing qualifications stay intact. The original hosted trigger is unproved.
Parent owns the complete shard/broad CI/static/security/hooks, prior runtime/PG
evidence, new hosted CI and Linux results, GitHub/waiver/cleanup and bookkeeping;
reviewer read saved RED/GREEN artifacts but ran no concurrent tests or scanners.

Changed TS syntax compiles; actual repository-locked shared UI ESLint is clean
with no suppressions. Initial outside-base ignored-file and temporary bunx-latest
incompatible-ESLint launches were harness errors, not lint passes. Corrected
explicit local CLI uses unchanged config/dependencies/suppressions; no lock edit.
No new executable Python security scope; retained production Bandit evidence
keeps its original meaning, not rerun or relabeled. Root pg_dump limit unchanged.

Full coverage shard 8/16 completed with the unchanged 4 GB/15-minute bound:
1,889 passed, four failed, zero skips; 172 passed/two failed files (174 total).
The corrected 27-case module passed. Complete CI/Docs/Config passed 1,619 with
six inherited skips, zero failures/errors (345.65s). Before publish, finish
canonical 15-task checks and staged 164-path PR hooks. Keep auto-merge off until
the next exact reviewed head's own
ordinary notes/chat UX AND shared UI coverage 8/16 succeed. This monitor hold
changes no workflow gate, budget or test. Then fresh rules/queue/dev/feedback
and all eight required contexts govern exact-head MERGE. No tracking-only commit.

The native full shard is NOT green: unchanged workspace.persist-diff failed
the unchanged-keys, exactly-one-serialization and legacy-revision-rewrite cases;
unchanged Studio.model-targets failed legacy output visibility. A complete
five-module native closure passed 258 and failed four (63.95s): the same three
workspace cases plus Studio canonical output visibility instead. All 243 actual
selection/reader cases passed: target 27, H1 controller 172, on-device list 44.

Activation selects Node 25.6.0 locally. The same closure under already-installed
Node 22.16.0 passed 260 and failed two (46.44s): workspace all four pass and all
243 selection cases pass; Studio canonical and legacy PRE visibility still fail.
Neither Studio nor workspace or their production/config/lock files changed.
Both complete exceptional modules passed hosted c474f3c8e9 (15 and four cases).
Record all outcomes; no unrelated speculative fixes or broad-green claim. The
runtime comparison does not prove specific causes or equal hosted Node20/Bun1.3.2
qualification; local Bun remains 1.3.5 and no dependency/lock upgrade was made.

Logs/JSON: `/tmp/claims-pr3434-local-selection-{full-coverage8,closure-coverage,node22-closure}-20261108`;
CI evidence: `/tmp/claims-pr3434-local-selection-ci-docs-config-20261108`.
Fresh remote head/dev are unchanged and dev is an ancestor; no rebase needed.
Auto-merge remains null and no new feedback; preserve the eight effective
required contexts, queue-unset and MERGE-only policy. Publish only this reviewed
test delta plus linked tracking via a normal fast-forward, after final hooks.

All applicable full 164-path staged hooks passed without rewrites; all 15
explicitly selected task records are canonical. Reviewed test hash is unchanged.
Log: `/tmp/claims-pr3434-local-selection-staged-hooks-20261108.log`.
Restage this evidence and rerun the full scope before normal-hook commit and
normal fast-forward publication. Record actual readback in PR proof, without
another tracking-only commit. Auto-merge remains off for both new-head holds.

### Research Workspace Parity Host Repair

At 19:50Z, exact-head e95eac2a4d run 42513937161 failed both Research Workspace
Parity jobs (128837900036 and 128837899955) at JS Playwright 1.58.0 Chromium
dependency installation on ubuntu26.04-x64, before either browser test command.
Both frozen installs and the extension production build passed. Logs:
`/tmp/claims-pr3434-research-parity-20261108T1950-{128837900036,128837899955}.{json,log}`.
Auto-merge was disabled and freshly confirmed null before edits. Extend the
existing supported-host test with these two explicit cases; watch RED, then pin
only these two browser jobs to Ubuntu 24.04. Compare the entire parsed workflow
after restoring only the two runner labels. Commands, admission/security rules,
triggers, dependencies, retries, timeouts and budgets must remain unchanged.

ADR required: no; this restores locked-tool host compatibility under the same
earlier host correction, with no new architecture or workflow gate. Independent
review, full CI/Docs/Config, changed-test compile/configured Ruff/scoped Bandit,
canonical selected tasks and staged full PR hooks precede normal publication.
No speculative nightly, dependency, application or scanner changes. Retained
production/runtime/PostgreSQL/security evidence keeps its original scope/date.
The next exact head must independently pass ordinary coverage 8/16 AND notes/chat
UX before rearming, then all effective required checks and freshness still gate
actual MERGE. Existing local Studio/workspace and root pg_dump limits are not
rerun or relabeled. Stage 3/4 and TASK-15466 stay In Progress until actual merge.

The expanded supported-host contract produced exactly two expected failures and
six existing passes; after the two-label correction both full workflow modules
pass 25 tests, zero skips. Full CI/Docs/Config passes 1,621 with six inherited
runner/environment skips, zero failures/errors (192.27s, 1,627 cases). Selections
overlap, not additive or a full-root pass. Changed test compilation and configured
Ruff pass; scoped Bandit reports zero findings/errors with B101 assertions excluded
from that invocation only. Existing nosec-comment parser warnings are unchanged;
no scanner/config/suppression edits. Whole parsed workflow parity and reviewed
source hashes are recorded in `/tmp/claims-pr3434-parity-runner-scope-proof-20261108.json`.

Independent Euclid APPROVE/no actionable P0-P3:
`/tmp/claims-pr3434-parity-runner-review-20261108.md`. It independently inspected
exact source/hashes, whole-workflow byte and parsed parity, hosted logs, tagged
upstream platform code and saved RED/GREEN; it ran no tests/builds/scanners.
Declined scopes remain parent-owned: broad qualification/static/security/hooks,
canonical tasks, GitHub feedback/rules/freshness/waiver and task/plan bookkeeping.
Unchanged runtime/PostgreSQL/security retain earlier evidence; existing local
limits remain unproved/unfixed. Nightly behavior is unchanged, not certified.
Future exact-head hosted success and the dual hold remain required; no universal
Linux/browser/image/download or dependency-health guarantee is inferred.
All 15 selected tasks are canonical. Stage verified four-path corrections before
full 165-path hooks, normal-hook commit and normal fast-forward publication;
record exact GitHub readback in the PR proof, not a tracking-only commit.

All applicable full 165-path staged hooks passed without rewrites or bypass;
all 15 explicitly selected tasks remain canonical. Only the ordinary no-matching
TOML and wizard-only Ruff/Black hooks skipped. Log:
`/tmp/claims-pr3434-parity-runner-staged-hooks-20261108.log`.
Restage this final evidence and rerun the full scope before the normal-hook
commit. Exact new-head hosted holds, publication readback and actual merge are
still pending, not inferred from local contract success.

### MCP Default-Selection Settlement Investigation

At 22:08Z, ad1a5a8dbe shared UI coverage 1/16 job 128867956306 failed one
McpToolsStep default-selection assertion: Research was present but unchecked.
The other 1,962 cases and 174 files passed. Both current-head hosted holds and
both repaired parity jobs succeeded, but this new failure keeps auto-merge off;
it was disabled and freshly confirmed null before edits. Logs:
`/tmp/claims-pr3434-shared-ui-coverage1-128867956306-20261108T2210.{json,log}`.
The unchanged real module passes all 11 locally with V8 coverage (14.15s), not
deterministic RED or Linux certification. Component and test match dev.

Hypothesis: the catalog exposes inputs before the separate passive defaults
effect settles; findByLabelText waits only for presence. A temporary probe uses
the actual component and source-extracted assertion tail/catalog, holding only
that initialization effect. Release after its first real checked-predicate
sample; preserve the matcher return/throw behavior and reject permanently
unchecked Research, missing catalog and incorrectly selected Writing controls.
This is controlled scheduling evidence, not proof of the original hosted trigger
or a persistent application defect. Remove the temporary probe before final scope.
Only a validated awaited checked-predicate correction is in scope; no production,
helper, workflow, dependency, skip/retry/sleep or timeout/budget changes. ADR
required: no for test-settlement repair. TASK-15466/Stage 3/4 stay In Progress.

The valid controlled probe reproduces the exact unchecked-Research assertion:
old source RED one failed/three passed, then corrected source plus the complete
real module GREEN 15 passed/zero skips (11 real, four probe cases). The original
unchanged real module's 11-pass run remains separate, not RED. The probe's first
jsdom URL/createRequire setup error ran no tests and is not RED. Probe source and
all outcomes are retained under `/tmp/claims-pr3434-mcp-defaults-*`; the temporary
repository file is removed. No instrumentation or helper changes remain.

Replace only the Research assertion statement with an awaited waitFor around
the same checked predicate. Complete TS-AST/scope proof records all 11 tests and
45 assertions retained, every byte outside that statement identical, no waitFor
options or test-budget changes, and production source identical. New source
SHA256: `2c9936abea2b8d61d8f35a98ae06c4109043618b6edda257d7fe2000b1ef672d`.
Syntax transpilation has zero errors (not whole-app semantic TSC); the established
apps-base locked ESLint command has zero diagnostics, suppressions or stderr.
The initial root-cwd lint had no file diagnostics but emitted a pages-path
environment warning; only the corrected invocation is clean qualification.
Full native coverage shard 1 with unchanged 4 GB/15-minute bounds, broad
CI/Docs/Config and independent review remain in progress. No production Python
security scope changed; earlier production Bandit/runtime/PostgreSQL evidence
retains its original scope/date, not rerun or relabeled.

Complete native V8 coverage shard 1 is GREEN: 1,963 passed, zero failures/skips,
175 files, 365.34s (366.50s wrapper elapsed) under the unchanged 4 GB/900-second
bound. JSON/default reporters are local evidence only; workflow/report settings
are unchanged. Corrected McpToolsStep passes all 11 there. Complete CI/Docs/Config
is GREEN: 1,621 passed, six inherited runner/environment skips, zero failures/errors,
234.59s. These selections and the 15-case probe/module run overlap, not additive
or full-root totals. Independent Laplace APPROVE/no actionable P0-P3:
`/tmp/claims-pr3434-mcp-defaults-review-20261108.md`. It personally inspected
source/hash, real component/effects, installed Testing Library behavior and saved
probe/hosted evidence; it ran no tests/scanners or hosted commands.

Reviewer-declined scopes: unchanged whole-PR/runtime/Jobs/Claims/PostgreSQL/security
retain prior reviews at original meaning. Parent owns broad/full-shard/static/
security/hooks/canonical-task verification, task/plan bookkeeping, source/probe
cleanup, live feedback/rules/queue/dev freshness and the existing summary waiver.
Original natural scheduler cause, early production interactions, broader onboarding
behavior and universal future checked-state stability are unproved/not recertified.
Existing dependency/config/workflow policy adequacy and root pg_dump/native local
Studio/workspace limits are unchanged, not re-reviewed, rerun or called green.

Before normal publication, stage only the three verified owned paths and run
canonical selected-task checks plus full 166-path staged hooks. Keep auto-merge
OFF until the exact next head's ordinary shared-ui-coverage 1/16 AND 8/16 AND
notes/chat UX all succeed; then freshly inspect feedback, effective rules,
MERGE_QUEUE and strict dev freshness before exact-head MERGE auto-merge. This
extends only the conservative monitor hold, not a workflow gate or budget.
All eight effective required contexts still control actual merge. Prior ad
coverage8/UX/parity success does not transfer; no all-eight-required-success claim.

All applicable full 166-path staged hooks passed without rewrites/bypass; all 15
explicitly selected tasks are canonical. Only ordinary no-matching TOML and
wizard-only Ruff/Black hooks skipped. Log:
`/tmp/claims-pr3434-mcp-defaults-staged-hooks-20261108.log`.
Laplace's report-only remote-status wording now explicitly distinguishes shard8
success from the eight required contexts; code/hash approval is unchanged.
Restage this final evidence and rerun full scoped hooks before normal commit and
fast-forward publication. Record actual GitHub readback in PR proof, without a
tracking-only follow-up commit. New-head hosted holds and actual merge remain
pending; TASK-15466 and this plan stay In Progress.

### Llama.cpp Downloads Retry Settlement

Exact df1d0b2ba3 ordinary frontend-required shard 3/16 failed one
LlamacppAdminPage stale-download-list-error test at the third request assertion:
two calls, 1,873 passes and two skips. Unchanged local V8 module passed all 32
and is not RED. The real Ant Design Button blocks clicks via its inner loading
state, which settles in a passive effect in tests. Error visibility alone
does not guarantee that the retry button will accept a click.

A temporary probe delays only the real loading-clear effect until the retry
query, retaining the real click guard and request assertions. The old sequence
failed at the same two-versus-three call assertion. This demonstrates ordering
sensitivity, not the original hosted scheduling trigger. Add a readiness
assertion before the retry, preserve all existing assertions, remove the probe,
then run the complete module, shard, broad CI/Docs/Config and full staged hooks.
No production, dependency, workflow, timeout or scanner changes. ADR not needed.
Auto-merge remains off. Stage 3/4 and TASK-15466 remain In Progress.

The same controlled probe passes with the added readiness wait. The 31
non-selected cases are diagnostic filter exclusions, not gate skips.
Removing the temporary probe leaves a five-line insertion and preserves all
32 tests, fixtures, assertions and every other source byte. Exact reviewed
SHA256: `322d64587f97fea70fffffbe2467533b5af12be23ce21340df9bb673d3710437`.
Native V8 shard 3 passes 1,874 tests with two existing WorldBooks skips
(175 files, 247.22s) using the unchanged 4 GB and 900s command bounds.
Local Node 25.6/Bun 1.3.5 and JSON/default reporting are not exact hosted
qualification. Full CI/Docs/Config passes 1,621 with six skips and zero
failures/errors (202.05s). Selections overlap, not a full-root result.
Locked TS syntax transpile has zero diagnostics; configured ESLint has zero
errors, zero suppressions and the same four warnings as the unchanged file.
Initial CLI/module/config discovery errors were harness errors, not passes.
No executable Python changed. Prior production runtime, official PostgreSQL
and Bandit evidence retains its original scope; old root/local limits remain.

Independent Carson APPROVE/no actionable P0-P3. The reviewer verified the
actual Button guard, probe, exact source and retained assertions, and ran no
tests. Report: `/tmp/claims-pr3434-llamacpp-review-20261109.md`. Declined scope
is recorded in TASK-15466; parent owns broad qualification, hooks, tracking
and live GitHub state. No natural hosted trigger or wider runtime claim.
After full staged hooks and normal fast-forward publication, keep auto-merge
off until exact new-head ordinary coverage shards 1 and 8 and notes/chat UX
all succeed; also verify the repaired shard 3 before rearming. All effective
required contexts still gate actual merge. Prior df1 results do not transfer.

### Notes Outline Prompt Ordering

Exact c37428752c ordinary UX run 42540493637, job 128922316590, had 31
passes, one failure and no skips/flaky attempts. Extension NE-03 failed
before outline assertions at the 20s prompt visibility check. The trace
confirms the same-path note address change, not the natural timing trigger.
The global prompt's route cleanup can close a prompt already shown for
that pathname; the seeded Notes marker only suppresses a separate tour.

Controlled real-browser RED observed the real prompt before clicking the
real seeded row. The address changed and the prompt was hidden within
1.167s of click start, before nominal auto-dismiss. The original positive
wait then failed with the same 20s not-found assertion. This establishes
an ordering failure mode, not its frequency or the exact hosted trigger.
Remove the probe and move only the NE-03 note click after the unchanged
visible / Not now / hidden checks. A comment explains the ordering.
All 55 assertions, five test declarations, fixtures, WebUI order and other
bytes are preserved. SHA256:
`b00353d68a3f68380904ef9a5d529b8176c341b9cd9922004f6acf8081798d0d`.

Full unchanged local real Chromium/backend UX: 32 first-attempt passes,
zero skips/failures/flaky; NE-03 WebUI 6.347s and extension 13.819s.
Healthy before/after and stopped; all 13 temporary source hashes restored.
Deferred43 retention was incomplete (fault 256), separately from test
accounting. Both Print cases passed here; earlier local failures remain
historical. Existing prompt/Notes contracts 22 pass. Complete CI/Docs/Config
1,621 pass, six inherited skips, zero failures/errors (188.08s), not
additive/full-root. Locked TypeScript syntax transpile only and configured
ESLint pass. No production Python/security scope change; retain previous
scoped runtime/PG/Bandit results with their original meaning.

Independent Arendt APPROVE/no actionable P0-P3 findings:
`/tmp/claims-pr3434-ne03-review-20261109.md`. Reviewer checked exact source
and saved traces, not independent execution. Parent-owned verification
and all declined scopes are dispositioned in TASK-15466. Evidence:
`/tmp/claims-pr3434-ne03-*-20261109.*`. Before normal publication, verify
all 15 selected tasks and applicable final 168-path pre-commit hooks.
Auto stays off pending exact new-head ordinary shards 1/3/8 and notes/chat
UX success, with parity and new failures inspected before fresh feedback,
rules, queue and dev checks. Old outcomes never transfer; all eight
effective required contexts still gate merge. Stages 3/4 remain open.

### Required Frontend Chromium Runner

Exact d540 ordinary frontend-required run 42545967988, job 128976634564,
passed setup, lint/typechecks and WebUI coverage (2,492 passed, 14 existing
skips). The Ubuntu 26.04 image selected by ubuntu-latest was rejected by
locked JS Playwright 1.58.0 during Chromium install, before browser tests.
That head passed all four hosted holds and both parity jobs. MERGE auto-merge
was armed on d540 at 04:34:32Z and verified off before this repair.

Pin only the frontend-required job to Ubuntu 24.04. Tagged Playwright host
support and three exact-d540 Ubuntu 24.04 UX/parity installs corroborate
compatibility. Both parsed YAML loaders and byte reconstruction prove the
job runner label is the only workflow difference. Both browser installs,
check identity, triggers, admission, tests, gates, dependencies and budgets
are unchanged. Extend the existing supported-host test by one case only.

Contract RED: one expected unsupported-label failure, eight passes. GREEN:
146 workflow, queue and license contracts. Complete CI/Docs/Config: 1,622
passed, six inherited skips, zero failures/errors (195.64s); selections
overlap and are not full-root qualification. Test compilation and configured
Ruff passed. Unfiltered Bandit comparison: 63 identical inherited B101
assertions, zero scanner errors or new findings; no scanner changes.
Feynman independently approved the exact source with no actionable findings
and did not rerun tests. Artifacts: `/tmp/claims-pr3434-required-runner-*-20261109.*`.

The separate d540 Jobs PostgreSQL job 128962907227 hit its existing one-hour
limit with last reported progress 95%; no final suite/coverage report or
demonstrated assertion failure. Underlying runtime cause is not proved.
No Jobs source or budget change. Prior production and official PostgreSQL
qualification keeps its original scope and date.

Publish with auto-merge off after canonical-task and full staged PR hooks.
Require the new head's own ordinary shards 1/3/8 and notes/chat UX successes,
inspect parity and fresh failures, then recheck feedback, rules, queue and
dev freshness. All effective required contexts still gate actual merge.
Stages 3/4 and TASK-15466 stay In Progress.

### VN Storage Resolver Fixture

Ordinary 5b904b00 CI run 42564927350, auth-integration-b-z job 129006619658,
reached tests: 534 passed, six org/team/user SQLite/PostgreSQL variants
failed before the storage assertions with child diagnostics withheld.
Exact isolated head and dev 47652771 controls both reproduced 6 failed,
6 passed using the official PostgreSQL fixture. A separate unchanged SQLite
child run identified the required keyword-only org membership context
TypeError without printing raw diagnostics; both membership methods require
that context on head and dev. This is an inherited fixture failure.

Supply the existing trusted BOOTSTRAP context to the two fixture membership
writes only. Byte reconstruction preserves all other source, including
all 25 assertions across the test module and its embedded storage script.
No production authorization, quota precedence, transaction, test selection,
workflow, dependency, scanner, retry or timeout change. No new ADR.

Patched exact-source temporary snapshot: 12 focused cases passed and the
complete VN module (73), membership integration (2) and runtime selection
(15) passed together (90, no skips, 259.04s). Both official PostgreSQL and
SQLite were exercised. Complete CI/Docs/Config: 1,622 passed, six inherited
skips, zero failures/errors (218.65s). Selections overlap; neither is the
full hosted shard or full root. Compile (including embedded script), Ruff
and unfiltered Bandit passed comparison (25 identical inherited B101
assertions, zero new findings/errors). Parfit independently approved the
exact fixture diff, not independent execution or full-PR certification.
Evidence: `/tmp/claims-pr3434-vn-control-20261109.dM8PzP` and
`/tmp/claims-pr3434-vn-fixture-20261109*`; independent review:
`/tmp/claims-pr3434-vn-fixture-review-20261109.md`.

The 5b904b00 Jobs PostgreSQL job passed 1,557 with three skips within its
unchanged limit. Its Bun install failures and GitHub Releases uv HTTP 504s
are separate, not fixed by this patch. Auto-merge remains off; repeat exact
ordinary hosted holds, parity and required-check verification on the
published head. TASK-15466 and Stages 3/4 remain In Progress.
