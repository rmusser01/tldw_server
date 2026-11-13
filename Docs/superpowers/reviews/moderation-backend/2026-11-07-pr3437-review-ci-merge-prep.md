# PR3437 Review and CI Corrections

Associated task: TASK-15477. Request: address comments and failed CI, use the
requesting-code-review skill instead of Qodo, and prepare for merge. No merge or
auto-merge was requested. Existing ADR-097/ADR-021 govern; no new ADR is required.

## Review Inventory

Initial head: `510b00da4c91e526b95cb873cc1372d33c04482f`.
Reviewed base: `dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c`.
Paginated inline comments and GraphQL review threads: none. The only issue
comments were Qodo billing-blocked and CodeRabbit base-branch-disabled notices;
neither supplied a code finding. No paid review was retriggered.

Independent reviewer Raman reviewed the full immutable PR with the
requesting-code-review template. No Critical or Minor findings. One Important
finding: synchronous text_transform StopIteration crosses the new coroutine
helper and becomes RuntimeError, turning a previously clean stop into a terminal
stream_transform_error. Benign immutable-source probes confirmed the difference
in both the helper and actual plain-text process_line path.

The correction carries that synchronous signal as a private
_TextTransformStopped exception, caught by both plain and structured streaming
owners. It does not reinterpret arbitrary RuntimeError or moderation failures.
Actual stream regressions verify the accepted prefix, omitted stopped/later
content, successful prefix save and finish marker, and no error finalization.
RED: two expected failures; GREEN: two passed. Adjacent streaming regressions:
126 passed, one preexisting skip, 386 warnings, exit 0. Logs:
`/tmp/pr3437-stopiteration-{red,green}.log`,
`/tmp/pr3437-streaming-regression.log`.

An initial regression edit accidentally attached an existing stream_kind
decorator to the new test, causing collection failure. Restored it to the old
test before the meaningful RED/GREEN runs; no test or parameter was removed.

Fix-diff re-review accepted P2 and all exact ratchet exceptions, and identified
two Minor test-determinism findings. The reporter test could assert thread exit
after its coroutine had already stopped waiting; cleanup now boundedly joins
the actual owner. Reader-start/kill assertions were incorrectly unconditional
inside a tightened 20 ms timeout that can expire before IO starts; retain the
original deadline/elapsed check and verify bounded reads under the existing
fixed-clock transport test instead. Both files pass all 45 tests (147 warnings,
exit 0). Scoped independent re-review accepted both corrections with no new
findings: all findings in the reviewed scope are cleared. Technical acceptance
does not replace CI, integration or the human Change summary gate.

Declined-to-judge decisions from the reviewer: alternative engines, compiler
and admin lint isolation, inherited scan geometry, newly moderated LLM surfaces
and arbitrary trusted override implementations are explicitly outside the
approved slice. Windows/full-project/runtime-wide guarantees remain unverified,
not silently approved. Latest dev synchronization and CI are coordinator-owned.

## CI Root Causes

1. coverage-required: 15 packaging setup errors because the offline
   `build --wheel --no-isolation` fixture required wheel in the test environment,
   but the dev extra did not declare build/backend tools. Declare build,
   setuptools (same requirement as build-system) and wheel in dev, not runtime.
   A metadata regression failed before the correction; all 17 installed-wheel
   tests now pass (63 warnings, exit 0).
2. onboarding-docs-gate and full-suite gap-verified-9: generated Published ADR-097
   was absent from the tracked inventory and the ADR index snapshot was stale.
   Official refresh script generated both; all 212 Docs tests pass (653 warnings,
   exit 0). Strict MkDocs build and public/private, command and endpoint boundary
   guards pass. Existing INFO-level anchor notices remain visible.
3. backend-required and full-suite gap-verified-12: seven new test-quality offense
   keys. Strengthened six tests to assert terminal/error/ownership/metadata and
   real persistence outcomes rather than only fake internals. No detector or
   quality-baseline change. Focused six-file suite: 243 passed, 738 warnings;
   quality ratchet: 18 passed, 66 warnings; exits 0. Scoped code review retains
   every prior safety assertion and adds observable contracts.

CI logs retained under `/tmp/pr3437-{coverage,backend,onboarding,gap9,gap12}-failed.log`.
Local logs: `/tmp/pr3437-build-deps-{red,green}.log`,
`/tmp/pr3437-docs-{red,green}.log`, `/tmp/pr3437-mkdocs-strict.log`,
`/tmp/pr3437-six-tests-green.log`, `/tmp/pr3437-quality-ratchet-green.log`.

A disposable build-only virtual environment also built and installed the real
tracked release offline, then ran an actual installed redact_many child and
checked packaged operator notes. Exit 0:
`/tmp/pr3437-fresh-offline-build-resolved.log`. The first diagnostic built and
installed successfully but compared macOS /var and /private/var path aliases in
its ad hoc factory. Resolving that factory path, as pytest does, fixed the
diagnostic without weakening the installed-path assertion or changing tests.

## Additional Required Contracts

The first backend failure prevented later required checks from running. Running
the exact next workflow step locally found ten function-size keys and three
global-declaration keys. All other 1,076 checks passed (one skip); this was not
reported as a green contract run.

The documented _ceilings.py security-exception path permits a new reason and an
exact measured ceiling, never reserved headroom. Per-entry PR3437/TASK-15477
reasons identify fail-closed branches in existing large Chat, Audio and Workflow
owners. The nested streaming owner crosses 300 lines while preserving terminal
failure through fallback, cleanup and save/finalization paths. The three module
global counts describe the ADR-097 PID-wide four-slot orchestration/worker
ownership, quarantine and fork-publication boundary, and independent one-slot
reporting lease across all default/injected instances. No production state or
function growth was introduced to satisfy these exceptions. Broad decomposition
and changing ownership representation remain separate work.

Targeted size/owner ratchets: 25 passed, 90 warnings, exit 0. Independent reviewer
must assess the reasons, not merely accept a raised baseline. The strict
test-quality baseline is unchanged.

## Local Verification Caveats

The first local global-coverage invocation had 1,072 passing tests and one
preexisting Python-3.14-sensitive sandbox test failure. Its coverage report was
invalid: overlapping local AuthNZ coverage reused the default .coverage file,
causing "no such table: tracer". This is an orchestration mistake, not a genuine
zero-coverage regression. The global rerun uses its own /tmp coverage data file
and XML; the original failed log is retained, not counted as success.

The sandbox fixture assumed two Path.stat calls and therefore did not inject
the intended vanished-file race on Python 3.14. The test-only correction checks
the real file then actually unlinks it between is_file and size-stat, retaining
the run-id/zero-size assertions and asserting deletion. Sandbox production code
is unchanged; no test skip, weakened expectation or count baseline was added.

Deployment-shaped minimal-env smoke: /health 200, exit 0. Production Bandit on
streaming_utils: zero findings/errors. Controller-owned edited Python files are
Ruff-clean and compile with SyntaxWarning as error. Test security scans retain
only existing fixed-vector subprocess, dummy-secret and /tmp path fixtures,
with B101 the only globally skipped rule.

Final isolated global coverage selection: 1,073 passed, 3,689 warnings, exit 0,
198.62 seconds; 18.09 percent against the unchanged 12 percent floor. Report:
`/tmp/pr3437-global-isolated.xml`; log `/tmp/pr3437-global-coverage-isolated.log`.
Later P3 changes alter only test cleanup/readiness assertions, not production.
Sandbox fixture suite: 20 passed, 72 warnings, exit 0. Exact tenant ratchets:
74 passed, 236 warnings, exit 0. Exact backend CI-contract step: first batch
1,079 passed/one preexisting skip/2,948 warnings, then 193 passed/1,304 warnings;
both exit 0. These overlap other runs and are not summed as unique coverage.

The separate local AuthNZ floor invocation used the default shared home lock
path and finished with five failures and 416 passes (45.00 percent coverage,
above 35 percent). This is a failed suite, not a successful floor check. Two
failures were BYOK rotation lock acquisition; three were chat503s after repeated
app lifespans. An unchanged two-test BYOK reproduction failed similarly. Setting only the
supported OPENAI_OAUTH_REFRESH_LOCK_DIR override to an isolated /tmp directory
makes both tests pass (440 warnings, exit 0, 2.10 seconds). This is local sandbox
and shared-state qualification, not a reason to weaken assertions or change
AuthNZ production code. Its final floor/result and the real CI floor remain
separate gates. The supported override does not select the weaker in-memory
lock backend: both tests still exercise native file locking and rotation.

## Sequential Lifespan Recovery

A native reproduction with isolated native OAuth lock files confirmed a real
Moderation lifecycle defect: the first AuthNZ chat request passed, then later
app lifespans returned moderation_unavailable503. Shutdown closed offload and
singleton-runner admission permanently in the same PID. The restart regression
failed on the second lifespan before the correction. Additional RED tests
covered a real blocked native offload and a retained quarantined child.

Trusted startup now reopens only the same settled singleton owner under its
lock, and only when actual orchestration ownership is zero. It does not replace
an owner, reset counters, drop entries, spawn a child or reopen on request paths.
Admission stays closed while native work or quarantined resources remain owned.
The async global ceiling changes exactly 4 to 5 for trusted startup, with a
reviewed rationale; the strict test-quality baseline remains unchanged.

Independent fix review caught a second lifecycle gap: incident reporting also
stayed closed after restart. A second-lifespan oversized-input regression failed
because no review item was persisted. Successful trusted store preparation now
reopens reporter admission under its existing lock without clearing a live
reporting-thread lease. A live-thread regression failed before the correction,
then verified retained single-slot ownership, rejection of another report and
real persistence after release. The exact reporter global ceiling changes 5 to
6 with that reason. No capacity or timing limit changes.

The reviewer also removed a new test race: quarantine is now established by an
immediate malformed frame and unreapable fake child under the default scope,
not an artificial 20 ms spawn assumption. Scoped re-review accepted all changes
with no further actionable findings; concurrent independent lifespans remain
unvalidated. Initial regression construction used a wrong WorkerResponse field;
that was corrected before the meaningful RED reproduction, not counted as proof.

Final lifecycle/async/runner/reporter suite: 84 passed, 267 warnings, exit 0,
25.50 seconds. All three native AuthNZ chat cases pass (523 warnings, exit 0,
18.14 seconds). Production Bandit on all four lifecycle modules has zero
findings/errors; all eight touched lifecycle/test Python files are Ruff-clean.
Logs: `/tmp/pr3437-restart-red.log`, `/tmp/pr3437-restart-ownership-red.log`,
`/tmp/pr3437-reporter-restart-red.log`, `/tmp/pr3437-reporter-lease-red.log`,
`/tmp/pr3437-lifecycle-final-green.log`,
`/tmp/pr3437-authnz-chat-native-green.log`, `/tmp/bandit_pr3437_restart.json`.

Full native AuthNZ floor: 421 passed, 46,210 warnings, exit 0, 611.98 seconds;
45.22 percent against the unchanged 35 percent floor. Native OAuth lock files
use an isolated /tmp directory, not an in-memory bypass. This run began during
merge preparation before the rebase; backend content is unchanged by the rebase.
Log: `/tmp/pr3437-authnz-coverage-native.log`.

An intervening global rerun overlapped rebase and had one cleanup-unconfirmed
refusal in test_probe_output_is_bounded_and_nonzero_exit_preserved (1,075 passed).
The isolated benign reproduction passed. Independent static review found no
supported code/deadline change: the controller explicitly refuses success until
both process-group cleanup and pipe readers are confirmed, retains ownership
and retries. Subsequent absence of a controller demonstrates eventual retirement,
not that the initial refusal was false. Its exact cause is not established.
Retain `/tmp/pr3437-global-restart.log` and
`/tmp/pr3437-harness-overflow-repro.log`; this failed run is not a pass.

Final isolated global coverage reran on the stable published source head
52efef9d1b6f43b6df82f52f6a22518815fe2b63: 1,076 passed, 3,677 warnings,
exit 0, 181.33 seconds; 18.09 percent against the unchanged 12 percent floor.
Log `/tmp/pr3437-global-final-head.log`, XML `/tmp/pr3437-global-final-head.xml`.
Coverage used distinct /tmp data paths throughout these final reruns. Final
quality/size/global ratchets: 43 passed, 146 warnings, exit 0. Lifecycle test
Bandit: zero findings/errors with only B101 excluded. All eight changed
lifecycle/test paths compile under both Python3.12 and3.14 with SyntaxWarning
as error. No runtime, test or packaging edit follows these final checks.

## Merge Gates

MERGE_QUEUE is unset. Latest fetched dev initially advanced seven UI-only
commits to `cb70f0c5d134ac585c70fc30b5a99bf0ced98562`; only this PR is being
prepared for serial integration, not every open PR. Required statuses must
genuinely pass on the final head. No manual gate dispatch, fabricated status,
disabled test or hook bypass is authorized. A requester-owned Change summary
was requested asynchronously and remains separate from AI-authored summaries.
Windows runtime remains unverified. No whole-project pass is claimed.

Published reviewed source52efef9d1b6f43b6df82f52f6a22518815fe2b63 on latest
fetched devcb70f0c5d134ac585c70fc30b5a99bf0ced98562 with an exact lease against
remote510b00da4c91e526b95cb873cc1372d33c04482f. Rebase had no conflicts;
all33 range-diff entries are equal patches, and backend/tests/pyproject content
matches backup codex/moderation-pre-merge-prep-rebase-20261107. Worktree is clean
before this evidence-only update. Current PR is MERGEABLE, auto-merge null;
required checks have not finished. The trusted license audit is queued, with
downstream workflows awaiting its verdict. No current-head failed check is
reported. Active rules require seven strict statuses plus the trusted dev
license status; an empty gh pr checks --required response while jobs queue does
not remove those rules. No gate dispatch, status forgery or paid bot invocation.
