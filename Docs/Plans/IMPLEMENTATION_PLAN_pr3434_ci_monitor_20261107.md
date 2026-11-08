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
until the corrected, independently reviewed and verified head is published.
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
