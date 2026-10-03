# VN PR 3016 Review And Merge

Task: TASK-13369. Parent [#3016](https://github.com/rmusser01/tldw_server/pull/3016);
stacked follow-up [#3067](https://github.com/rmusser01/tldw_server/pull/3067).
Designs: Docs/Design/2026-09-25-vn-pr-3016-review.md and
Docs/Design/2026-10-02-vn-pr-3067-review-tests.md.

## Current State
Human explicitly replied "APPROVED" after the presented Task66 bootstrap fix,
publication of Tasks64/65/67, parent current-dev integration and fresh review/CI.
This supersedes the implementation/publication/integration holds below.
Latest actual dev is86e287fee7bfa1a1588639232e35db3666851ded; parent and child
published heads remain68861/393c. Main now executes Task66 with bounded TDD.
Human "address all the ci failures and review findings then!" approves the
presented four child test/docs fixes. Tasks1-63 are locally complete,
independently reviewed and published; do not redispatch them.
Parent exact68861f5a229365c867db8deb3432028207cd8848 and child
393c365796434e717edae072fbb9802c7f256181 remain OPEN/unmerged.
Fresh 2026-10-02 01:08 UTC: review/body/check/thread arrays unchanged;
parent97/child15 completed checks, known parent Jobs and child macOS/Ubuntu E2E
failures. Parent92threads/5unresolved; child2/2, all pages complete.
Protected devd81c13fddd1dac1948b30401af0388932a0af8f2 read twice stable.
No safe current-dev integration or local conflict reproduction is credited.

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
**Status**: In Progress
Task59 nested asyncio-plugin fix already exists in child; parent old-head failure
remains evidence until new-head CI actually runs. Child CI logs sanitize the
native cause. A matching-SQLGlot30.20.0 isolated run now reproduces UsersDB
bootstrap rejection: standalone AUTOINCREMENT rendering differs from29.0.1.
The narrow AST-validation fix plan is explicitly approved and in implementation.
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
Integration path risk: dev-88f8b8-integration-diagnosis.md; workflow/source
changes remain unaudited, not verified causes of bootstrap failures.
ADR required: no for test/tracking changes; ADR002/004/006 apply.
Tasks64/65/67 complete locally and independently approved, uncommitted and
unpublished. Automation prompt condensed from84725to6860 characters; exact
before/after TOML snapshots retained. Status/schedule/target preserved and
stored prompt equality verified. It remains ACTIVE/QUIET on known blockers.
New human-fixes-20261002/SHA256SUMS verifies68selected immutable inputs,
including current source/design snapshots and relevant SQLGlot generator/version
metadata. Manifest/check log and mutable task/plan/ledger are excluded from its
input inventory; old manifests remain untouched. No historical recovery claim.
