# VN PR 3016 Review And Merge

Task: TASK-13369. Parent [#3016](https://github.com/rmusser01/tldw_server/pull/3016);
stacked follow-up [#3067](https://github.com/rmusser01/tldw_server/pull/3067).
Designs: Docs/Design/2026-09-25-vn-pr-3016-review.md and
Docs/Design/2026-10-02-vn-pr-3067-review-tests.md.

## Current State
The human explicitly approved the presented engineering fixes, publication,
safe current-dev parent integration and fresh exact-head review/CI.
Tasks64/65/67 and Task66 are published on child
6909f5f7c4c501de191faf8d5d5ef558ba15f4e9; its 15 current checks completed
without failures. This is not parent or required-dev-context green.
Parent GitHub head remains68861f5a229365c867db8deb3432028207cd8848,
OPEN/unmerged with the known old-head Jobs SQLite failure.
Local parent branch is on reviewed393c365796434e717edae072fbb9802c7f256181
with an uncommitted merge of actualdev86e287fee7bfa1a1588639232e35db3666851ded.
New actualdev1c8491ff341053468afb9ffde9b379707c9e3bad is fetched but not yet
integrated. Seven reproduced conflicts are textually resolved; behavioral
integration verification and publication remain in progress.
Frontend and Main service/storage slices have independent scoped review PASS.
DB D1 flat-only V1 precedence correction is independently SPEC/QUALITY PASS;
the complete DB file passed100 native cases. Complete VN v5 then stopped with
6 failures and638 passes, not a complete PASS. Fresh bounded repairs cover
legacy start-display outages/privacy, nonretryable legacy failure disposition,
and execution pinning before rejected generating admission. Original assertions
are retained. Changed worker/DB hashes and the native exhausted-retry handoff
fixture have independent SPEC/QUALITY PASS. Complete slot-state v7 passed122
with no failures/errors/skips. Complete VN-domain v7 passed977, zero failures,
errors or skips, 27 warnings, 1071.53s. All16 frozen Python hashes matched.
This existing Python3.11/asyncio1.1 environment is below declared dependency
floors; local evidence is not supported-Python/native CI, PG or whole-repo green.
Live Task68 below and the SDD ledger govern, not historical approval holds.
Tasks1-63 are complete/reviewed/published and must not be redispatched.

## Global Constraints
Preserve Jobs sole queue/lease authority, public outcomes, original counters,
approved bytes and explicit regeneration/new-draft semantics. No external
exactly-once execution promise. Preserve checkout/main/backups/already-applied
stashes and all frozen evidence. No cleanup or historical manifest refresh.
No PR body edit or cleanup inferred from this approval. Parent reconciliation
and publication are now explicitly approved; normal merge remains gated.
New CI fix plans require explicit approval under gh-fix-ci.
Child own human-written Change summary and parent guarded Verification-only
body approval remain separate. Normal merge needs exact-head full Qodo with no
actionable findings, all seven required dev passes, current strict actual-dev
integration/rules and human gate. No bypass/admin/autoqueue.
Root venv before Python/pytest/Bandit; VN pytest basetemp only under the approved
native macOS temporary root. Focused verification, not whole-repo green claims.

## Stage 1: Approved Test Fixes (Task64)
**Goal**: Correct SQL fixture placement, outcome/rollback tests and raw copy hook.
**Success Criteria**: No public corruption API; sensitive real SQLite outcomes
and rollback coverage; no raw runtime copy/finally masking.
**Tests**: Focused fixture consumers, receipt recovery/rollback, native deletion,
copied omission sensitivity, compile, scoped baseline Ruff/Bandit.
**Status**: Complete
Brief/report: .superpowers/sdd/IMPLEMENTATION_PLAN_vn_pr_3016_review/human-fixes-20261002/task-64-brief.md
Implemented locally: 45 focused passes, three native rollback passes and three
intended omission failures; compile/Ruff clean, six inherited non-B101 findings.
Independent Jason SPEC/QUALITY PASS/no actionable findings, reviewer closed.
Baseline busy-timeout failure
is qualified separately; local SQLGlot29 does not cover CI30.

## Stage 2: Tracking Archival (Task65)
**Goal**: Keep current state readable without losing historical records.
**Success Criteria**: Exact pre-condensation plan/task bytes retained and hashed;
task notes condensed only through official Backlog tooling.
**Tests**: Archive byte/hash equality, task metadata/AC/DoD/outside-notes
preservation, link and diff checks.
**Status**: Complete
Independent SPEC/QUALITY PASS; task-65-report.md and task-65-review.md retained.

## Stage 3: CI Diagnosis And Approved Fixes (Task66)
**Goal**: Resolve all verified CI causes without disabling tests or guards.
**Success Criteria**: Reproduction establishes root cause, concrete fix approved,
sensitive regression and independent review pass.
**Tests**: Narrow reproductions matching CI environment; affected regression scope.
**Status**: Complete
Task59 nested asyncio-plugin fix already exists in child; parent old-head failure
remains evidence until new-head CI actually runs. Child CI logs sanitize the
native cause. A matching-SQLGlot30.20.0 isolated run now reproduces UsersDB
bootstrap rejection: standalone AUTOINCREMENT rendering differs from29.0.1.
The narrow AST-validation fix is implemented and independently SPEC/QUALITY
PASS after correcting the bootstrap test's preservation-observation order.
SQLGlot30 RED2intendedfail, GREEN127pass;29GREEN127pass. Final corrected native
bootstrap1pass on each version; managed-boundaries30 21pass. Scoped Ruff,
compile/diff checks pass; Bandit49/49 unchanged findings, not blanket clean.
Evidence/reviews: task-66-approved-20261002. Tasks64/65/67 committed9a5271aa2e.
Diagnosis: human-fixes-20261002/task-66-ci-diagnosis.md.
Additional Task67 local test diagnosis: shared sqlite3 instrumentation samples
other backend handles before their 10-second configuration. Intended read-only
VN handles already use timeout10. Human explicitly approved the bounded
observation-test fix; Boole implementation is complete, three native/adjacent
passes and one intended missing-timeout failure. Lovelace independent
SPEC/QUALITY PASS/no actionable findings; implementer/reviewer closed.
No production timeout change. Task66 approval is now recorded in TASK-13369.
Diagnosis: human-fixes-20261002/task-67-timeout-diagnosis.md.

## Stage 4: External Gates
**Goal**: Publish approved source fixes, close verified findings, obtain fresh
exact-head review/CI and merge normally only when all gates pass.
**Success Criteria**: AC5/AC6/DoD truthful; GitHub MERGED independently verified.
**Tests**: Paginated exact-head review/check/status and actual-dev/rules reads.
**Status**: In Progress
Publication and safe parent integration approved. Parent body approval and child
human summary remain separate; exact-head external gates still pending.
No tracking-only push.

## Records And Qualifications
Ledger: .superpowers/sdd/IMPLEMENTATION_PLAN_vn_pr_3016_review/progress.md.
Complete pre-condensation plan/task snapshots:
.superpowers/sdd/IMPLEMENTATION_PLAN_vn_pr_3016_review/human-fixes-20261002/plan-before.md
.superpowers/sdd/IMPLEMENTATION_PLAN_vn_pr_3016_review/human-fixes-20261002/task-before.md
These ignored local archives preserve historical working-tree bytes; prior
published history remains in Git. They are not recovered missing /tmp evidence.
Historical raw loss remains qualified: 535 missing references among 1077
selected historical manifest entries; do not refresh old manifests.
Task68 automatic integration audit is source-scoped, not native CI proof.
AM1 is an inherited manual features-tier live-UAT prerequisite gap; AM2 is
an overbroad incoming Sandbox documentation guarantee for optional Redis.
Neither establishes the old parent Jobs CI cause or a VN union regression.
New evidence: task-66-approved-20261002/task-68-auto-merge-audit.md.
ADR required: no for test/tracking changes; ADR002/004/006 apply.
Tasks64/65/67 and Task66 are committed, reviewed and published on child6909.
Parent current-dev integration and fresh external review/CI remain in progress. Automation prompt condensed from84725to6860 characters; exact
before/after TOML snapshots retained. Status/schedule/target preserved and
stored prompt equality verified. It remains ACTIVE/QUIET on known blockers.
New human-fixes-20261002/SHA256SUMS verifies68selected immutable inputs,
including current source/design snapshots and relevant SQLGlot generator/version
metadata. Manifest/check log and mutable task/plan/ledger are excluded from its
input inventory; old manifests remain untouched. No historical recovery claim.

## Task68 Approved Integration: 2026-10-03

This live section supersedes historical completion/approval statements above.
The human approved publication, Task66 and safe current-dev parent integration.
Design: `Docs/Design/2026-10-03-vn-parent-dev-integration.md`.
Original associated task: TASK-13369, Address PR 3016 Qodo durability findings.
Current dev introduces duplicate TASK-13369 records; official CLI selects an
unrelated task, new-task CLI failed and official MCP creation timed out without
a confirmed result. Do not manually edit/renumber ambiguous tasks. New ignored
evidence and the live SDD ledger retain progress. External AC5/AC6/DoD pending.

### Stage 1: Reproduce And Preserve
**Goal**: Reproduce conflicts and preserve both accepted histories.
**Success Criteria**: Exact inputs recorded; backups and applied stashes retained.
**Tests**: Real merge diagnostics and focused preservation controls.
**Status**: Complete

### Stage 2: Repair The Union
**Goal**: Retain authored Retry/reload behavior and durable variant outcomes.
**Success Criteria**: Narrow native regressions precede each demonstrated repair.
**Tests**: Cancellation replay, integrity/legacy provenance, journal precedence,
Retry capacity/rollback, zero-work receipt recovery and world-book error safety.
**Status**: Complete
All accepted bounded union repairs have RED/GREEN and independent scoped review.

### Stage 3: Verify Frozen Files
**Goal**: Complete coherent VN validation and independent SPEC/QUALITY review.
**Success Criteria**: Reviewed hashes; scoped compile/lint and Bandit baseline;
automatic source/workflow integration audited with qualified evidence limits.
**Tests**: Complete VN-domain run, frontend contracts, native bootstrap guards.
**Status**: Complete
Frozen86e integration: complete VN977pass/no skips, frontend297pass, scoped
native bootstrap/guard tests and independent SPEC/QUALITY reviews. Source and
workflow audit qualifications remain. New actual-dev1c Sandbox integration
requires its separately prepared native shard and unchanged-owned-hash proof.

### Stage 4: Publish And Merge
**Goal**: Publish safely on actual current dev, then normally merge PR3016.
**Success Criteria**: Full exact-head Qodo completion without actionable findings,
seven required dev contexts pass, strict actual-dev integration and human gate.
**Tests**: Fresh complete GitHub arrays and independent MERGED verification.
**Status**: In Progress
Local86e integration verified; checkpoint commit and normal1c integration next.
Exact-head GitHub review/required CI, strict actual-dev and normal merge pending.

Child PR3067 has its own human-written summary/base/review/CI gates and is not
auto-merged. Parent Change summary remains verbatim; separate Verification-only
body approval remains pending. No bypass, tracking-only push or cleanup.
