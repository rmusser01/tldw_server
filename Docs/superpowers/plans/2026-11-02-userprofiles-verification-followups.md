# UserProfiles Stage 2 Verification Follow-Ups

Tracking: TASK-15413. Base: dev `3ab1ce5d3a49d5a7fd130e8939d55d7639e5a373`.
WP2 runtime is merged; these follow-ups do not start WP3 or alter merge policy.
The requester authorized these outstanding repairs on November 2.

## Stage 1: Reproduce Recorded Failures
**Goal**: Establish the boundary and cause of each recorded failure.
**Success Criteria**: Deterministic delayed-layout reproduction for the Companion
Home notice; fresh focused controls for recorded UI assertions and Python/ACP
environment failures. Prior isolated passes do not qualify a full suite.
**Tests**: CompanionHomePage, useSpeechHistory, useComposerVariantPreference,
workspace.persist-diff, Studio.model-targets, StandaloneHtmlWorkspace, Workspaces
collection, and the previously failing ACP/WebSocket tests.
**Status**: Complete

## Stage 2: Narrow Repairs
**Goal**: Correct established test synchronization, fixture, import or environment
defects without weakening product behavior or verification.
**Success Criteria**: Original assertions and security boundaries remain effective;
no timeout inflation, blanket skips, baseline changes, test exclusions, required
gate dispatch or workflow/policy changes. Each change has RED/GREEN evidence.
**Tests**: Affected files and adjacent scopes, lint, compile, diff, and applicable
touched-scope Bandit. Use official fixture-managed PostgreSQL if needed, never
concurrent suites sharing fixed databases.
**Status**: Complete (dev integration and subsequent VN fixture repair qualified below)

## Stage 3: Independent Review and Publication
**Goal**: Publish a coherent, independently reviewed follow-up patch.
**Success Criteria**: Validate spec and code quality, record accurate separate test
runs and remaining environmental constraints, and require all current protected
checks plus applicable human-summary policy before merge.
**Tests**: Canonical task formatting, review findings, exact-head CI and merge
readbacks. No unchanged whole-PR review or unnecessary broad-suite repetition.
**Status**: In Progress

## ADR Assessment
No new ADR is required for behavior-preserving test synchronization or local
verification environment repair. ADR085's membership transaction/version decision
is unchanged. Any proposed runtime behavior change must be assessed separately
before implementation, not folded into a test-only correction.

## Scope Boundaries
WP3-WP5 retain their approved Stage 2 plan and sequential package review gates.
WP2 tracking closeout is a separate two-file PR with the requester's explicit
human-summary waiver; that waiver does not apply to runtime changes here or later.

## Verification Checkpoint (November 2)

The patch changes tests and their environment only. Companion Home waits for
the independently persisted layout; shared UI uses the real jsdom Storage
object; Studio waits for actual modal visibility; the composer quota spy is
restored even on failure. Workspace collection uses declared httpx, the auth
matrix isolates only downstream runner startup, and backup fixtures align both
real settings consumers after collection reloads. Original behavior and safety
assertions remain; no product, dependency, workflow, timeout or skip changes.

Separate final controls: six affected UI files pass 65/65 with v8 on Node 25;
the unchanged frontend StandaloneHtmlWorkspace file passes 111/111 on Node 22;
the four Python files pass 366/366; complete root collection succeeds with
85,935 tests. The backup controls additionally pass seven cases after full
root collection. Focused RED/GREEN and scoped independent review evidence is
linked in TASK-15413. Python Bandit adds only three expected test assertions,
with no new non-assert findings; backup Bandit remains empty. Full-tree test
quality enforcement and the 869-shard guard pass without baseline changes.

Non-green evidence is retained: bare root execution was interrupted (exit 130)
after 1,000 passes and two backup failures, subsequently repaired and verified
in the root-collection context. Bare shared-UI execution was interrupted
(exit 130), without a complete suite result. An initial six-file UI run had
63 passes and two Studio 15-second timeouts while other UI jobs ran; after
stopping only owned competing jobs, the unchanged command passed 65/65. That
does not establish harmlessness or full-suite correctness. CI uses Node 20,
which was not executed locally. No compiled Go runner/RPC integration was
claimed. Exact-head remote CI and the human-summary gate still govern merge.

## Published-Head UI10 Follow-Up

**Status**: Independently Approved and Published in 4d1649; Later CI Requires Further Repairs
Published-head CI job125454102604 ran 2,359 tests: 2,358 passed and the Playground
no-usable-model assertion failed at line972 while the UI still showed Checking
model readiness. Its wait observes the fetch call, not settled query rendering.
Controlled delayed-response RED reproduces the same checking rail; the same
real query/render path becomes GREEN when the three original rail assertions
are awaited together. Commit6da9b1c868 changes only that observation point,
preserving all227 original assertions and restoring the original immediate
response fixture after diagnosis. The temporary barrier is released in finally;
no global scheduler or final state is mocked. Final Node22 V8 full-file66/66
passes; targeted ESLint has zeroerrors/fourbaselinewarnings. No precise remote
event-loop, harmlessness, Node20 or full-suite correctness claim is made.
Independent bounded review and new-head CI remain mandatory. Do not change
runtime, mocks to force a result, timeouts, skips or policy. The separate UI7
Tailwind install failure occurred before tests and is not repaired by such a
test synchronization change.

The requester subsequently waived the human-written Change summary for PR3404
only; that waiver is recorded in its body. All actual required gates remain.

Report and controlled RED/GREEN artifacts:
/private/tmp/userprofiles15413-playground-ui10-report.md and
/private/tmp/userprofiles15413-playground-*.

## Further Exact-Head CI Repairs

Published4d1649 remains blocked: required frontend unit shards report a
VisualPack candidate-review count failure and an uncaught real Joyride/Popper
callback after environment disposal. Broad backend shards also report Skills
collection lifetime, Slides policy, DB fixture/status/session-setting and
Embeddings OAuth failures. These are not waived by earlier gates or review.
Evidence: PR3404 comment6586881626 and retained per-job diagnostic logs.

Local sourcec37c4fa351 adds the missing direct unit marker to the existing
webhook lifecycle test. The enforcing catalog RED becomes GREEN without any
test-body or assertion change. Final non-PG controls119pass/threePGdeselected;
an earlier file command inadvertently selected threePGcases, all skipped, and
does not constitute PG verification. Bandit90B101 plus1B105 is unchanged, with
zero scan errors. Python output retains1746 warnings, not a pristine-output claim.

Local source6d6196d5bf fixes a reproduced pending-owned-timeout invariant in
the real Joyride suite. File-local pass-through real timer ownership drains
callbacks and promise descendants before restoring APIs; no cancellation sweep,
fake clock, fixed sleep, private-field filter or shared setup change. All five
original scenarios are byte-identical; one chained-work/restoration regression
is added. Drain-disabled negative control fails; final threefile46/46 Node22
controls, focused TypeScript baseline/current0/0, ESLint and diff checks pass.
This is not a local reproduction or closure of the exact CI exception/Node20.
Independent Newton approves exact6d source with zero Critical/Important and one
Minor inherited-warning observation; no new source defect. Report:
repair-4d-review.md in this plan's ignored SDD workspace.

Further narrow repairs remain under TDD: VisualPack render readiness and
validated backend fixtures. Three fake-PostgreSQL resume cases reproduce an
omitted construction-time owner; Slides official-host policy reproduces the
inherited CI allowlist; malformed regeneration returns its documented400 before
writes rather than stale409/422 expectations; the official PostgreSQL benchmark
fixture reproduces losing session SETs on deliberately secure pool RESET ALL.
Use a test-owned startup budget, not weaker pool resets or service defaults.
Separate Skills90/90, real-Joyride5/5 and OAuth focused/collection controls pass
locally but do not explain or dismiss the remote failures. No speculative
runtime fix, source exclusion, dependency install, CI retry or policy change.

These sources are not yet published. Task15413 stays In Progress; affected
verification/review acceptance is reopened until new-head evidence is recorded.
Only successful actual required gates, fresh refs/rules/feedback and exact-head
review permit merge. Existing whole-branch approvals carry unchanged content;
new remediation receives bounded independent review.

## Fresh Dev Integration

The source batch through3b213e79ff is independently approved for its original
base. Backend controls222selectedpass and separate official PostgreSQL runs
2benchmark/16regeneration pass without skips; warning-bearing evidence remains
explicit (366/45/70 local and16/45 PG warnings), not pristine or full-suite proof.
Source reviews have no Critical/Important finding; inherited evidence-noise
Minor observations remain recorded rather than expanded into unrelated cleanup.

Authoritative dev advanced tob2b7d5d2b04256458fe6f33e02a36019e2c46795 from
mergedPR3187:142paths, shared HTTP/AuthNZ/accounting changes and overlapping
fixture repairs. Before publication, qualify the combined test boundaries,
preserve upstream runtime/workflows and stronger real-ACP/shared-Slides/startup-
budget fixtures, and drop only demonstrably superseded patch portions. Preserve
all discriminating assertions and byte-identical reviewed UI. Fresh affected
Python/PG/security/ratchet and bounded integration/records review are required;
old-base success is not fresh qualification. No manual CI retry or bypass.

Detailed integration rulings and intended scopes are recorded in this plan's
ignored SDD dev-b2b7-integration-notes.md. Task15413 stays In Progress and its
new verification acceptance remains open. This checkpoint is not publication,
merge approval, or a claim that unrelated Skills/OAuth failures are harmless.

## Dev b2b7 Qualification

Rebased onto authoritative b2b7d5d2b04256458fe6f33e02a36019e2c46795 with a
pre-rebase backup ref, other-ref updates disabled and auto-merge/queue unarmed.
Five previous patches replay equal, two replay contextually and the duplicate
webhook marker disappears because it is upstream. Workspace and regeneration
files now exactly match dev. Upstream's real Python ACP transport, common
Slides fixture and explicit connection-startup budget are preserved; our loader-
order and discriminating owner/settings assertions remain.

Fresh integration exposed one new refusal-control interaction: upstream's common
fixture adds an official-host workflow allowlist, which unions with the global
allowlist. Original control failed1/175; sourceb5e480aaaeb79fbdd9e9eb4ef3059b66619b0779
isolates only the global-list case by removing that additive test input.
The same hostile-policy command then passes175/175, retaining refusal and
zero-request assertions. No production policy or shared fixture is changed.

Separate fresh runs, not an aggregate full suite:366four-file Python controls
pass (2635warnings);175Slides/owner controls pass (362warnings);2official PG
benchmark cases pass (16warnings);32regeneration cases pass (77warnings), with
16actual PG parametrizations and16SQLite cases. No skips. PG runs are serial,
required-PG/no-Docker, using only official UUID temporary-database fixtures.
Ruff/compile/diff pass. Fulltree quality5364 uses the new upstream scanner with
unchanged baseline; shard870/5364/fourignored/baseline44/new0 passes.

Unfiltered Bandit versus exact b2b7 test blobs reports621->636B101 assertions,
unchanged4B105/3B106/2B608 contexts and zero scan errors. No new non-assertion
finding. Original broad-suite and unresolved Skills/OAuth limits still apply;
none is declared harmless or silently excluded.

Tree/mode checks preserve137/142upstream paths exactly; only five test paths
combine both sides. All28844other pre-rebase entries, including all eight
reviewed UI artifacts, match their reviewed objects/modes before this final
two-record checkpoint. Candidate changes remain tests, test setup and tracking
only: no runtime, workflow, dependency, schema or policy delta from dev.
Logs /private/tmp/userprofiles15413-devb2b7-* and the retained range-diff record
support this qualification. Final bounded integration/record review and fresh
published-head CI remain before merge; Task15413 stays In Progress.

## VN Resolver Fixture Qualification

Published376's eight actual required checks succeeded, but a newly completed
nonrequired AuthNZ shard failed six VN resolver org/team/user cases across
SQLite and PostgreSQL (534 other passes, 42404 warnings). The conservative
merge guard stopped before sending any merge request; PR3404 remains open.
Local org/SQLite RED and a private exception-type/stack-only diagnostic identify
two fixture membership calls missing the current mandatory context argument.
The test and relevant runtime inputs match dev b2b7; that does not make the
failure harmless or waive it.

Source a941e4f50815af3285508823beabd769fc58e9f1 changes only the embedded
STORAGE_SCRIPT: import the existing trusted-context types, construct one
BOOTSTRAP fixture context and supply it to the two existing membership calls.
Outer AST is otherwise identical, REGISTRATION_SCRIPT is byte-identical, and
all 18 outer and seven embedded Storage assertions, cases, results, mocks and
private-environment controls remain unchanged. No runtime shim, raw-write
bypass, new trusted reason, policy, dependency, workflow or timeout change.

Separate overlapping runs, not an aggregate suite: same org/SQLite case
RED1fail -> GREEN1pass (12 warnings each); full SQLite selection35pass/
38deselected/148warnings; adjacent non-PG controls28pass/2deselected/120warnings.
Controller-owned official fixture run of all12 resolver cases passes with no
skips: six SQLite plus six actual PostgreSQL cases, 61deselected/366warnings,
42.54seconds, exit0. Existing configured cluster was inspected privately;
official fixtures own isolated databases, with no container lifecycle action.

Outer Ruff/compile/diff pass. Embedded Ruff retains exactly 3I001+7BLE001, not
a clean embedded-lint claim. Unfiltered Bandit baseline/current outer18B101
and embedded8B101 contexts match exactly, zero new findings or scan errors.
Fulltree quality5364 baseline unchanged and shard870/5364/fourignored/
baseline44/new0 pass. Herschel independently approves exact a941 source with
zero Critical/Important/Minor findings; no unchanged whole-PR review restart.
Logs /private/tmp/userprofiles15413-vn-fixture-* and
/private/tmp/userprofiles15413-vn-resolver-backends.log retain warnings and
limits. Prior broad-suite/Skills/OAuth limits remain explicit. Task15413 stays
In Progress until final closeout. Final records require bounded exact-head
review, then authorized source publication and fresh automatic required gates;
376's successful checks do not qualify the new candidate.

## Plugin Discovery and Audio Patch Ownership Qualification

Two later nonrequired shards on published376 exposed separate fixture defects:
platform MCP had4303passes/3skips/36missing-mocker setup errors/1765warnings;
media audio had1failure/4316passes/37skips/1xfail/1xpass/6319warnings. Both
relevant test/runtime inputs match dev b2b7, not newly introduced product code.
This neither excuses those failures nor changes any actual required gate.

Source9960d3eb6082053dec42ea5fb96e33f3ebe1f46f changes exactly two lines of
test setup: explicitly register the already-declared pytest_mock root plugin
when CI disables autoload, and patch the Parakeet adapter class rather than its
cached instance. Instance monkeypatch undo had restored a bound method as a
direct instance attribute, shadowing the later auth test's class replacement
and reaching its original Whisper-refusal assertion. A safe method-owner/
exception-type/stack-only diagnostic and ordered RED reproduce that mechanism.
All eight audio assertions, parameterizations, refusal/private-path/zero-call/
minutes controls and accounting/victim files remain exact. No product, schema,
dependency, workflow, timeout, skip, policy or global registry reset changes.

Separate overlapping local evidence, not a sum/full shard: disabled-autoload
MCP RED36errors -> fullfileGREEN63pass/4warnings; ordinary discovery63pass/
2warnings. Ordered audio RED1fail/15pass -> GREEN16pass/93warnings. Exact
registration-removed and ownership-reverted negative controls reproduce both
failures and were restored before final controls. A private teardown oracle
also rejects instance method shadowing without invoking ASR. Both complete
affected audio files23pass/360warnings; final ordinary combined three-file
control86pass/360warnings, no skips. Python3.14.3/local pytest-mock3.15.1 is not
claimed identical to CI Python3.12/pytest-mock3.17. No PG/model/download/network
or full-shard execution was performed.

Ruff/compile/diff pass. Unfiltered Bandit ignoring nosec is byte-identical:
eight existing LOW B101 test assertions, zero scan errors/new findings, root
setup zero findings. Quality5364 baseline unchanged and shard870/5364/
fourignored/baseline44/new0 pass. Warning inventories retain dependency/INI/
Starlette/Pydantic/Python deprecations, not a warning-free claim. Reports
plugin-audio-* in this plan's ignored SDD workspace and logs
/private/tmp/userprofiles15413-plugin-audio-* retain exact commands and limits.
Hume independently approves exact source9960d3 for spec and quality with zero
Critical/Important/Minor findings; approved VN source/records are unchanged and
carry their existing reviews. The later full-suite aggregate failed only its
Check Python3.12 shard result step, not a new executable test. Final exact-head
record review, authorized source publication and fresh automatic required checks
remain. Task15413 is In Progress; prior broad-suite/Skills/OAuth limitations
remain explicit. No merge, CI retry/dispatch or old-head check transfer.
