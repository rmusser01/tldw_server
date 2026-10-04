# VN PR 3016 Review And Merge

Task: TASK-13369. Parent [#3016](https://github.com/rmusser01/tldw_server/pull/3016);
stacked follow-up [#3067](https://github.com/rmusser01/tldw_server/pull/3067).
Designs: Docs/Design/2026-09-25-vn-pr-3016-review.md and
Docs/Design/2026-10-02-vn-pr-3067-review-tests.md.

## Current State
The latest direct human instruction authorizes rebasing PR3016 onto latest dev,
addressing all PR findings/comments and normal gated merge. The previously
presented Task69 four-source-fix and two CI-test-double designs are now approved
for execution under original TASK-13369. Local merge-preserving rebase onto
actualdev4c4f197 completed at2f3d3e09c0. At that checkpoint all production/test/configuration bytes
matched the preserved current-dev union tree c3342dca64; only this plan drifted
while replaying historical merges, and its qualified checkpoint text is restored
below. The approved source/CI fixes and independent reviews are complete locally;
current-dev quota controls, publication and fresh exact-head review/CI remain
In Progress. No old-head result transfers to the new head.
Incoming mixed-version RG TTL disposition is separate. Child merge/body rewrite
are not authorized; all frozen evidence and existing backups remain preserved.

The human explicitly approved the presented engineering fixes, publication,
safe current-dev parent integration and fresh exact-head review/CI.
Tasks64/65/67 and Task66 are published on child
6909f5f7c4c501de191faf8d5d5ef558ba15f4e9; its prepublication snapshot had15
completed checks without failures. This is not parent/required-dev green.
Parent prepublication baseline68861f5a229365c867db8deb3432028207cd8848 was
OPEN/unmerged with the old-head Jobs SQLite failure. Latest published-head and
external-gate state belongs to the live SDD ledger, not that historical baseline.
Local parent checkpoint bdf99343019a95a53785cedb32eec6e0f4b37431 commits the
reviewed86e integration and preserves393c and published68861 ancestry. New
actualdev1c8491ff341053468afb9ffde9b379707c9e3bad is normally merged without
conflicts. All16 frozen VN Python files are unchanged. Prepared native Sandbox
shard r2 passed95/no failures/errors/skips,6warnings85.99s; explicit Bash wrapper
recorded pytest_rc0 and session87721 exited0/reaped,42 frozen inputs unchanged.
R1's95 test outcomes remain qualified by post-pytest zsh wrapper exit1; its
frozen report/log/XML are preserved. Existing fixture/application startup
mutated checkout default DB/logs; no isolation/unchanged-runtime-state claim.
Integration publication and exact-head external gates remain in progress.
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
Local86e checkpoint bdf9934301 committed; normal1c merge has no conflicts and
no owned VN hash drift. Sandbox r2 passed95/no skips and captured exit0/reaped;
r1 wrapper failure remains qualified. Independent publication checkpoint reply/
evidence review PASS; r2 harness evidence is recorded separately. Latest commit,
publication and gate reads belong to the live ledger; no tracking-only push.
Exact-head GitHub review/required CI, strict actual-dev and normal merge pending.

Child PR3067 has its own human-written summary/base/review/CI gates and is not
auto-merged. Parent Change summary remains verbatim; separate Verification-only
body approval remains pending. No bypass, tracking-only push or cleanup.

## Task69: Requested Rebase And Remaining Findings

2026-10-04 requester explicitly directed immediate publication and again
requested latest-dev rebase, exact-head Qodo follow-up and normal merge.
Publish the independently reviewed source/CI repairs and explicit quota opt-in
for existing enforcement tests. Preserve the new NOT-GREEN quota matrix in
the ignored publication archive and restore it locally after rebase/publication;
it is not passing evidence or part of this publication. No production quota/RG
change, integration waiver, child action or merge-gate bypass is inferred.
Current actual dev is e70e0abb129ea8010f8096d1080b8ea23570f442.

### Stage 1: Preserve And Rebase
**Goal**: Rebase onto actual current dev without losing reviewed union repairs.
**Success Criteria**: Published-head backup, merge topology and exact union bytes.
**Tests**: Actual branch endpoints, merge-tree comparison and ancestry checks.
**Status**: Complete
Backup codex/vn3016-before-rebase-76a8-20261003 retains published76a8. Local
merge-preserving rebase2f3d3e09c0 contains actualdev4c4f197; source/tests/config
matched the preserved union before new repairs. Complete r77 still verifies
unchanged dev4c, parent76a8 OPEN and child6909 OPEN. No new head is published.

### Stage 2: Approved Source And CI Repairs
**Goal**: Address four source findings and two diagnosed CI test-double failures.
**Success Criteria**: Bounded RED/GREEN, preserved contracts, independent review.
**Tests**: Native publication/drain and typed-health controls, nine-file frontend
suite, cursor-lifecycle and migration-driver companions, touched-scope Bandit.
**Status**: Complete
Backend R1:97 passed/48 unavailable-PG skips; SPEC PASS, QUALITY identified one
new docstring line shifting the scope baseline. A line-neutral correction is
implemented without relocating baseline entries. Final health48 passed/48 PG
skips, existing four-file CI ratchet51 passed; R2 SPEC/QUALITY PASS. A docword
changed during ratchet startup; no whole-run input-equality claim is made.
Frontend R1:365 passes and scoped lint/types; independent SPEC/QUALITY PASS.
Its P3 expected-warning capture is corrected with final two-case passes and
clean scoped static checks; R2 SPEC/QUALITY PASS. The365 outcome remains
prior-test-hash evidence, not a final365 rerun.
CI doubles:3 driver cases and34 backend/boundary cases pass; independent
SPEC/QUALITY PASS. No real-PG, native CI or supported-runtime credit inferred.

### Stage 3: Current-Dev Integration Controls
**Goal**: Verify incoming Jobs completion and quota policy against VN contracts.
**Success Criteria**: Exact Jobs identity/transaction behavior, quota default/off/on
first publication/replay/accounting and applicable user/team/org admission.
**Tests**: Existing completion controls, native quota matrix, required ratchets.
**Status**: In Progress
Jobs completion controls:51 passed/14 PG cases deselected, input hashes stable.
Existing quota enforcement tests reproduce incoming default-off expectations;
their private runtime now explicitly opts in without weakening assertions.
New native quota controls expose an inherited three-argument call to the
two-argument QuotaExceededError constructor. The source mismatch exists at
published76a8, actualdev4c and rebasedHEAD; narrow correction approval was
requested once and remains pending. Current quota verification is NOT green.
Separate mixed-version RG TTL disposition is also pending; no risk waiver.

### Stage 4: Publish, Review And Normal Merge
**Goal**: Publish the reviewed rebase/fixes, obtain exact-head Qodo and CI, merge.
**Success Criteria**: Guarded exact-remote-head rewrite; full new-head Qodo with
no actionable findings; all seven required contexts; strict actual current dev
and requester Change summary; independently verified GitHub MERGED.
**Tests**: Complete paginated external arrays and actual branch endpoints.
**Status**: In Progress
No old-head CI/review result transfers. No admin/autoqueue/bypass, body PATCH,
child merge or cleanup. Original TASK-13369 association remains; ambiguous
duplicate task records and unknown task creation are untouched. AC5/AC6/DoD
remain pending real external gates. NEW evidence is task-69-approved-rebase-20261003-r1.

## Tasks70-74: Approved Post-Publication Repairs

Direct human approval2026-10-04: "address them and continue until the PR is
merged. All approvals granted". Original TASK13369 association remains. Earlier
pending-approval notes are historical; exact scopes and preservation still bind.

### Stage 1: Task70 Qodo Repairs
**Goal**: Resolve4178073234/4178073236/4178073240 without contract drift.
**Success Criteria**: Drained owning-thread setup/release with cancellation-safe
cleanup; current scoped journal smoke expectation; test-owned Jobs DB connections.
**Tests**: Thread responsiveness/setup cancellation, retry persisted counters,
reload scoped receipt assertions, touched-scope Bandit, independent SPEC/QUALITY.
**Status**: In Progress

### Stage 2: Task71 Canonical Task Formatting
**Goal**: Remove the five exact-path task-format violations blocking backend CI.
**Success Criteria**: Official normalize/check, task identities/status/text retained.
**Tests**: Exact-path check and semantic before/after comparison, normal hooks.
**Status**: Complete
Exact-path official check RED1 -> GREEN0; all5 parser frontmatter/checklist/
non-structural text comparisons PASS. Independent SPEC/QUALITY PASS with no
actionable findings. Normal commit hooks and new-head backend CI still pending.

### Stage 3: Task72 Quota And RG Integration
**Goal**: Correct inherited quota exception arity and explicitly configure audio
test caps; dispose of mixed-version RG expiry risk without silently losing charges.
**Success Criteria**: Typed quota denial, strict WIP matrix unchanged, persisted
30-minute test limit with own-user cache invalidation; conservative rollout safety.
**Tests**: Default/off/on controls, numeric DATE assertions, official PG fixture,
source-checked quiesced rollout protocol, scoped Bandit and independent SPEC/QUALITY.
**Status**: Complete
Approved implementation complete; independent SPEC/QUALITY review underway.
Storage155passed, audio8real-PGpassed, unchanged strict matrix48passed with all18
default/off/on controls. No new non-assert Bandit findings. Below-floor runtime
evidence does not transfer to supported-runtime CI. RG deployment docs require
quiesced homogeneous cutover/rollback; no production RG change or live rollout.
Independent SPEC/QUALITY PASS on the frozen40-input scope, no actionable findings.
Task72 local stage is complete; normal publication and exact-head CI remain Task74.

### Stage 4: Task73 Remaining CI Root Causes
**Goal**: Fix verified privilege snapshot, offline email guard and preflight deadline
causes rather than masking assertions or blindly regenerating snapshots.
**Success Criteria**: Reproduction/caller evidence, smallest explained changes.
**Tests**: Failing scenario before repair, covering green suite after, bounded runtime
qualification, Bandit and independent SPEC/QUALITY. Stop the failed diff-parser approach.
**Status**: In Progress
Peirce is the sole implementation agent after Task70 R2 completion. Verified
privilege/deadline fixes and strict email guard diagnostics are in scope; the
unidentified offline caller is not declared repaired.
Implementation is now frozen and Peirce closed.26focusedpassed; four exact quota
dependency additions match real served routes with enforcement on/off. The full
default profile has four extra Notes scopes, while canonical minimal-test-app
83buckets equal the fixture and the existing full snapshot test passes1case.
Native deadline mutant fails as intended, Bandit has0newnonassert findings.
Independent review remains; email caller is unknown and strict diagnostic guard
is retained for exact-head CI evidence rather than declaring a speculative fix.

### Stage 5: Task74 Publish, Review And Merge
**Goal**: Publish the reviewed union on current actual dev and normally merge3016.
**Success Criteria**: WIP/archive preservation, normal hooks, exact lease if rewrite
needed, complete exact-head Qodo with no actionable findings, seven required contexts
PASS, strict current rules/dev and human Change summary, independently verified MERGED.
**Tests**: Complete paginated external evidence, ancestry/tree checks, bounded scoped
verification and final independent review. No admin/autoqueue/bypass/child mutation.
**Status**: In Progress
Task70R3 adds only the current-principal profile mock. Controller exact Chromium
reload recovery smoke passes1case8.7s with source hashes unchanged; task-owned
server and client sessions reaped. Existing original frontend build inputs remain
unchanged. Byte-exact strict quota matrix is included with approved Task72 repair;
all original archives remain retained. No commit/publication yet; full new-head
review, seven required contexts and current-dev/human-summary gates still apply.
