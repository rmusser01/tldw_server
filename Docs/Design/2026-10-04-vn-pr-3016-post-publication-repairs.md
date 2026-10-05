# PR3016 Post-Publication Repair Design

Association: original TASK13369; plan: IMPLEMENTATION_PLAN_vn_pr_3016_review.md.
The requester authorized the presented repairs on2026-10-04 with all approvals.
This design supplements the completed VN durability design, not a new generation,
quota, Jobs lease or receipt protocol.

## Qodo Repairs

Move inline legacy-display begin and claim release through the existing drained
owning-thread repository boundary. A cancelled setup drains the operation and
cleans its acquired marker/claim before propagating cancellation. Keep attempt
fencing, Jobs authority and legacy overlap semantics; use real blocking barriers
to prove event-loop responsiveness and cancellation cleanup.

The reload smoke test observes the current server/principal-scoped command
journal. Do not reintroduce a duplicate legacy receipt writer. Failed-job retry
tests own their SQLite/official isolated-PG connections instead of relying on
private JobManager helpers; persisted admission and rollback assertions remain.

## Quota Repairs

`QuotaExceededError` takes two arguments. Correct only the inherited combined
quota call's arity; preserve typed denial and all admission/accounting decisions.
The preexisting strict VN matrix stays byte-exact and must not accept arbitrary
transaction failures as quota denial.

Audio profile caps now come from persisted user/team/org limits, not an implicit
30-minute tier. Set the test user's30-minute override with the existing repository,
invalidate only that user's resolver cache, explicitly opt in to enforcement and
retain DATE and numeric expectations. Verify disabled/no-limit controls through
the official isolated-PG fixture; unavailable fixtures are skips, not PG passes.

## Redis Rollout Restriction

New RG writers refresh a shared window key's `ceil(window)+5` expiry; old writers
do not. If an old writer adds late to a new writer's expiring key, the whole key
can disappear while that old charge is still live. Changing script hashes does
not protect the shared key; changing namespaces would split admission counters.

The approved disposition is a homogeneous, quiesced cutover and rollback. Stop
all API/worker/background RG writers, freeze policies, wait out the largest
configured live request/token window plus the five-second expiry margin, retain
the same Redis namespace/state, then start only the chosen version together.
Do not resume traffic while mixed versions can write. Deployment documentation
records this restriction explicitly. No production TTL/accounting edit, key
deletion, real deployment or mixed-version compatibility claim is made.

## CI And Merge

Normalize only the five exact approved task paths with the official formatter,
preserving frontmatter, prose, IDs/status and all checklist state. Remaining
privilege/email/deadline failures need verified runtime causes before minimal
repairs; neither snapshot regeneration nor assertion weakening is a substitute.

Every code repair receives bounded TDD, touched-scope Bandit and independent
SPEC/QUALITY review. Normal hooks, full new exact-head Qodo, all seven required
contexts, strict current dev/rules and the verbatim human Change summary govern
normal merge. No old-head result, unsupported-runtime suite or unavailable-PG
skip certifies those external gates. PR3067 remains untouched.

## Task75 Timeout Compatibility Follow-Up

Exact-head Python3.12 CI passed219 governed preflight tests but rejected the new
test's direct `asyncio.timeout(None)` call through the existing architecture
guard. The guard remains unchanged. The selected-stage synchronization, genuine
native cancellation, FakeClock advance, budget and cleanup assertions also remain.

Expose finite absolute-deadline `reschedule` on the existing compatibility context.
Delegate to native `reschedule`, or legacy async-timeout4 `update`, in loop clock
coordinates; preserve original factory/enter/exit/expired and error propagation.
The test uses its already imported compatibility factory and public context API,
not private native fields, AST-evasion tricks or fake cancellation. This small
extension keeps supported and legacy timeout interfaces behind one boundary.

Verify the existing guard fails before repair, new native and forced-legacy
forwarding/error controls fail before implementation, then the complete three-file
governed CI scope passes. Scoped Bandit and independent SPEC/QUALITY apply. Local
Python3.11 and a forced legacy double do not establish native Python3.12/3.10 CI.

## Task76 Cursor And Verification Feedback

Full exact5cd Qodo review adds three verified findings. Close the SQLite cursor
owned by the transaction-bound quota lookup through its existing public async
lifecycle, preserving returned rows, misses, error/cancellation propagation and
the borrowed connection's outer transaction. PostgreSQL selection is unchanged.

Annotate only the newly added offline diagnostic test's fixtures and return;
preserve its executable body and strict forbidden-call/privacy assertions.

On terminal initial principal-verification failure, end the pack-list loading
state and retain the existing error and Retry recovery check. Do not list packs
before verified authority or weaken generation/replay/account boundaries. Cover
network/inactive/unverifiable failures, pending verification and successful retry
with sensitive tests. Avoid restarting list loads on unrelated recovery errors.

Bounded TDD, scoped baseline Bandit/static checks and independent SPEC/QUALITY
precede publication. Future exact-head Qodo/CI still govern normal merge.

## Task77 Offline Daily-Accounting Isolation

Exact a2d9 CI job111546965676 reports240passed/1skipped/1teardownerror.
The strict argument-free diagnostic now identifies Redis DNS from AuthNZ
migration locking, not email/model work. The upload's positive synthetic user
reaches daily accounting, which resolves quotas and initializes the shared
AuthNZ ledger. The fixture declares auth/quota/billing outside its native email
scope but isolates only storage quotas. Disabling enforcement alone would not
isolate the unlimited-user shadow record.

Under the requester's all-engineering-approvals authorization, substitute only
the existing daily-accounting boundary inside this offline fixture. Preserve
real upload/form validation, parser, persistence, search/detail and every strict
network/model tripwire. Do not alter production quota accounting, migration
locking, Redis settings or CI workflows, and do not allowlist or clear the
observed forbidden call. Add a real-upload regression that detects AuthNZ pool
entry with fresh daily-ledger state and usage quotas enabled, without touching
operator databases. Observe RED before the fixture repair, then GREEN for all
fixture consumers and existing guard/privacy controls. Scoped static/Bandit
and fresh independent SPEC/QUALITY review precede publication.

This diagnoses the new exact-head CI caller; it does not retroactively certify
the earlier unlabelled call, a whole-repository result or current-dev integration.

Task77 implementation and independent SPEC/QUALITY are complete. One fixture
substitution and one actual-upload regression preserve every original guard/test
AST. RED records two intercepted pool entries; all103 consumers and26 separate
accounting controls pass locally. Independent4GREEN and in-memory RED confirm
sensitivity; no new nonassert Bandit findings. Both agents are closed. Local
runtime remains below declared floors; fresh actual-dev integration and new-head
external review/CI remain mandatory before normal merge.

## Task78 Finite Property Generation Budget

Current-dev integration exposed a local Hypothesis input-generation health check,
not a scheduler assertion counterexample. The complete31file run reports1229pass/
1fail/21Jobs-gated skips; unchanged exact-seed replay repeats the health failure.
The third and final profiled attempt attributes4.320s to cold local-constant
discovery and4.107s to scanning599 imported modules. The profiler wrapper exits0
but the nested pytest result is FAILED; it is not green. No further unchanged
reproduction loop is permitted. The test and scheduler match actualdev bytes.

Under all engineering approvals, give only test_real_window_derivation a finite
two-second Hypothesis deadline. Its derived generation health budget is ten
seconds, above the observed cold scan; no health check is suppressed. Keep the
default example count, phases, datetime/interval/lookback ranges and all three
slot/window assertions exactly. Retain pytest's outer timeout. This property
checks correctness, not a200ms performance contract; no production code changes.
Verify exact-seed GREEN plus an in-memory future-slot fault that still fails the
original assertion, scoped static/Bandit, and fresh independent SPEC/QUALITY.
Retain all three original failures and below-floor qualifications; no whole-run,
supported-runtime, PG or future-CI certificate is inferred from narrow GREEN.

Task78 independent SPEC/QUALITY PASS/no actionable findings: unchanged source
hashes,100examples and in-memory future-slot original-assertion RED confirmed.
Final controller current-dev31file matrix1230passed/21official Jobs-gated skips/
44warnings250.12s, JUnit1251tests0failure0error21skip. Finite timing tradeoff is
explicit; installed-version draw-budget coupling is not a future-version promise.
All original failures, warnings and below-floor/nonPG/nonCI qualifications remain.
