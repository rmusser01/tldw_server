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

## November 7 Same-Key Draft Persistence Follow-Up

Reviewed4b239's required frontend unit7 job128499323509 and aggregate
frontend-required job128513398512 failed in run42401177625. The test observed
visible SECOND_EDIT before actual validation/persistence, then read the old
EDITED recovery record. A separate cold focused control failed earlier at
initial storage; these are different observations, not exact reproduction.

Source-only50f2cbb1d25f5fc1e733de66e2cb09a6874d5178 changes one test file.
It awaits a connected current editor and actual same-key stored payload,
preserving all nine original assertions and Save/blocked-cleanup/revoke/restore
ordering. A companion real-WebCrypto call-through gate establishes a reachable
pending-visible/old-stored interval: the premature storage assertion is RED;
released real validation replaces the same key under v8 and retains the newer
recovery across cleanup and authority restoration. Finally drains/restores
the held work even on failure. No runtime/setup/timeout/skip/policy edit.

Separate final local controls: affected workspace/editor/recovery files158/158,
target-file TypeScript BASE/current0/0; ESLint0errors/57identical inherited
warnings. Original other module/test AST statements are unchanged. Local
Node25.6/Bun1.3.5 differs from CINode20.20.2/Bun1.3.2. The engineered schedule
does not prove CI's unrecorded timing or a harmlessness claim; no whole-suite
green claim. No new Python/Bandit scope; earlier security evidence applies to
unchanged Python blobs only. Report and logs are retained in the ignored
SDD workspace and /private/tmp/userprofiles15413-standalone-*.

Authoritative dev advanced dfac3ca7 tocb70f0c5d134ac585c70fc30b5a99bf0ced98562
through PR3400's twelve header/account/Watchlists/browser/tracking paths,
without Standalone overlap. Controller must preserve those upstream objects,
qualify the mechanical rebase, independently review the exact final candidate,
and wait for entirely fresh exact-head required checks. The PR3404-only human
summary waiver remains; no waiver of CI, review, latest-dev or merge gates.
No remote retry/dispatch, body rewrite, settings/policy change or merge.

Mechanical integration checkpoint77d800d21e30158d7dc308835ea91bd5f49f1656
replays all19 patches range-diff equal onto cb70. All12 upstream paths match
dev exactly in object/type/mode; all29,069 other pre-rebase entries are exact.
The tested Standalone blob remains6b4381ad76bc0cc8faef1f8f289ba6159de1b5ee.
Fresh integrated six-file UI control passes220/220 with no skips: the three
draft/editor/recovery files plus current-dev AccountMenu, ChatHeader notification
and Watchlists PipelineWizard tests. No full-frontend/browser/CI equivalence is
inferred. Bounded exact-head independent review and fresh required CI remain.

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

## November 3 Latest-Dev Integration

Published d2cbae's eight actual required checks succeeded, but authoritative dev
advanced to576feef820e6521f021e6f57c82c24df2429b901. The PR is still open,
unmerged and unarmed; no stale-head merge was attempted. Only this next-to-merge
verification branch was rebased, not unfinished WP4 runtime work. Backup
codex/userprofiles-verification-pre-576feef preserves exact d2cbae; ref updates
and autostash were disabled. Local560451aeae608174f4c0fe3a33852bd6d1a4ec05
is unpublished and requires a bounded independent integration review.

All13 patches replay range-diff equal. Of406 upstream paths,405 match dev by
object/type/mode; the combined AuthNZ WebSocket matrix preserves upstream's
container-owned JWT reset and the reviewed real-ACP/credential controls. All
28613 other approved leaf entries, including eight UI artifacts, remain exact.
No new runtime, workflow, dependency, schema or policy change is introduced by
this integration. Full proof is retained at
/private/tmp/userprofiles15413-dev576-preservation.mjs and its log.

Separate overlapping fresh controls, not one aggregate suite: four affected
HTTP/AuthNZ/ACP/backup files366pass/2998warnings; Slides/database-owner224pass/
684warnings; complete plugin/audio controls86pass/383warnings; official isolated
VN resolver12pass (six actual PostgreSQL plus six SQLite),61deselected/
379warnings; two official PostgreSQL benchmark cases2pass/18warnings. No skips
in those runs. PostgreSQL runs were serial against the existing configured
cluster with official temporary databases, no lifecycle or server-setting edits.
An initial incorrect ACP path stopped exit4 before collection; the corrected
four-file command is the366-pass result, not a hidden product failure.

All nine touched Python paths pass Ruff/compile/diff. Unfiltered Bandit ignoring
nosec against exact dev reports647->662B101 test assertions, unchanged4B105/
3B106/2B608 contexts and zero errors/new nonassert findings; scans exit1 for
retained findings, not a clean scan. Full quality5368 baseline unchanged and
shard870/5368/fourignored/baseline44/new0 pass. The byte-identical embedded VN
script carries its earlier reviewed security evidence, not a newly claimed scan.

New nonrequired d2cbae infrastructure shard125613616538/run41543048234 failed
the registry weakref lifetime case (488 other passes,128warnings); full-suite
aggregate failure is consequent. Test/runtime inputs match current dev. Local
unchanged focused4pass/20warnings and configured full shard489pass/1030warnings
do not explain the remote failure or establish harmlessness. No speculative
runtime fix, collection loop, sleep, skip, retry, exclusion or policy change.

Independent latest-dev/exact-head review is not yet available: the agent review
attempt hit account usage limits before producing a report. Existing own-source
approvals carry unchanged content but do not approve this unseen integration.
No review is fabricated or replaced by author self-review, and no credits or
reviewer service were purchased. This checkpoint stays local until that gate
is met. Fresh automatic published-head checks remain necessary afterward;
d2cbae's green statuses cannot qualify the rebased head. TASK15413 stays In
Progress, AC4 open. Earlier broad-suite and Skills/OAuth limits remain explicit.
Logs use /private/tmp/userprofiles15413-dev576-*; no unchanged UI suites or
whole-PR review were restarted merely for pending CI.

## November 7 Integration And Qualification

Authoritative dev advanced to dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c;
published d2cbae remains open/unmerged/unarmed. Backups preserve d2cbae and
the unpublished8ff5 checkpoint. Only this next-to-merge verification branch
was replayed; runtime WP4/WP5 was not rebased. At local349a6dfd,12of14 prior
patches are range-diff equal. Upstream already contains the reviewed Parakeet
class patch, so that replay retains only explicit plugin registration. The
one Playground conflict keeps upstream's stronger deferred-response controls
and our settlement comment; Slides retains upstream's removed test bypass
and all owned policy/refusal cases. No runtime/policy/dependency change.

Complete object/type/mode proof counts459 upstream path endpoints (415
rename-aware changes),455 exact upstream entries/four overlaps,28666 other
approved entries unchanged and18 own paths. Initial proof omitted old rename
endpoints and falsely flagged a renamed task; corrected --no-renames inventory
passes. James independently reproduces both this and earlier dev576 proof,
approving frozen integration spec/static quality with no source findings,
but explicitly withholds publication for fresh qualification failures below.

Fresh four-file backend control:364pass/2fail/2998warnings, positive first-work
assertions for audio chat/TTS saw no work after a one-second silent probe.
Fermat causally reproduced the fixture gap using delayed REAL credential
verification: silence is not completed authentication; disconnect can precede
work. Source3cbd98fec903d984f64116eb180deab677f1cf39 changes only that test
file. Real guard emits a test-owned acknowledgement after success; voice uses
its public auth_ok. Positive checks await the existing frame-reply budget,
not a longer silent probe. Invalid-first-frame4400/zero-work, real credentials,
revocation/scope checks and exact positive route identity remain. No runtime,
timeout constant, auth-result mock or guard bypass is changed. Deterministic
RED2fail -> focused7pass; full affected file186pass/1367warnings/no skips.
Original four-file command on3cbd then368pass/3008warnings/no skips. These
overlap, not a sum, and do not establish the original host schedule's cause.

Fresh UI seven-file V8 run initially191pass/4Studio15s timeouts. Unchanged
Studio file16pass in isolation; one final unchanged seven-file run195pass/
zero failures. Initial failures remain evidence; no contention causality or
harmlessness is inferred. No UI source, mock, timeout, assertion or dependency
change was made. Local runtime/version is not remote CI equivalence.

Other separate overlapping controls: Slides/database owner224pass/684warnings;
disabled-autoload plugin/audio87pass/457warnings; official VN resolver12pass
(six actualPG/sixSQLite),61deselected/437warnings; official PGbenchmark2pass/
2deselected/18warnings. No selected-case skips. Existing configured cluster,
official isolated task/UUID databases, serial runs, no container/server changes.
An initial wrong MCP path invocation stoppedexit4 before collection; corrected
documented paths produce87pass. Logs /private/tmp/userprofiles15413-devdfac-*.

Touched eight Python Ruff/compile/diff pass; unfiltered Bandit ignoring nosec
639->656B101 at3cbd, unchanged4B105/3B106/2B608 contexts and zero scan errors.
Both scans exit1 for retained findings, not clean scans. Shards871/5381/
fourignored/baseline44/new0 pass. Full-tree quality after3cbd catches one NEW
mock_density offense in its added regression. That is a mandatory blocker:
Fermat owns bounded fixround1/3 to reduce actual test-design complexity without
baseline/suppression/policy changes or heuristic evasion. Source review,
covering tests, full quality and exact final records remain outstanding; no
publication approval is inferred. Earlier broad-suite/Skills/OAuth/weakref
limits remain. Task15413 InProgress/AC4open, old checks do not qualify this SHA.

Quality fixround1/3 source878cc0145a176f2caa688485660393e36bbc39fb splits
the existing five-route invalid-frame security test from the distinct two-case
delayed-real-auth positive regression. Shared exchange helper performs only
the real operation/reply-budget selection, with no patching; explicit patch
sites are two in the security test and three in the delay regression. Original
security assertions/route matrix remain exact. The regression still fails
deterministically against the old silent exchange, then passes with explicit
work acknowledgement: RED2fail -> focused7pass/270warnings; full affected file
186pass/1367warnings/no skips. Full-tree quality5381 passes the unchanged
baseline; mock_density returns347. Scanner/baseline hashes unchanged; no alias,
helper-hidden mocking, suppression or policy change. Ruff/compile/diff/hooks
pass. Bandit103->106B101 in this fix, unchanged2B105/1B106 contexts/errors0.
The368-pass four-file control is pre-split3cbd evidence, not a claimed rerun
at878; final186 covers every amended source path. Source and final-record
scoped review remain pending. No publication or merge approval is inferred.

## November 7 Copy-Draft Readiness Qualification

Publishedba74 exacthead frontendunit5 job128551726965/run42414634295 failed
one VisualPackEditor copy-default selection assertion: expected copied-draft,
received undefined. Head347files3873pass/1fail/1skip; exactbasecb70 failedfile
and fullcontext347files3874pass/1skip, then strict identity/order comparison
rejects. No harmlessness, exactCI scheduling cause or gate waiver is inferred.
Latest authoritative dev remains cb70f0c5d134ac585c70fc30b5a99bf0ced98562.

Source d838852656a6a6172a25bb5bbe876c831adf094c changes only four lines in
VisualPackEditor.test.tsx's existing local copy helper: observe the real Copy
button enabled before its original single click. Actual StarterChoice disables
copying until artwork readiness, including the existing test stub's passive
effect. A controlled first-commit Profiler probe proves disabled/noPOST, original
click dropped and original copied-value assertion RED; identical probe plus
the default enabled wait GREEN/onePOST/alloriginaloracles. Probe/diagnostic
mutations removed. This establishes a reachable synchronization defect, not
the exact remote schedule or harmlessness. No runtime/mock/setup/dependency/
workflow/timeout/skip/policy change, no retry or manually dispatched gate.

Removing the four inserted lines restores the entire exactba74 file bytes:
all66cases/names/order,375originalexpects and prior review fix unchanged.
Separate focused original1pass, controlledRED1fail, matchedGREEN1pass and final
restoredfocused1pass; complete editor+real-artwork files68/68pass/no skips/
warnings,8.32s. TargetTypeScript0BASE/0current; ESLint0errors/50sameinherited
warnings. Configuredhooks/diffpass. LocalNode25.6/Bun1.3.5 differsCI20.20.2/
1.3.2; no wholefrontend/shard/equivalence claim. No newPython/Banditscope;
prior Pythonsecurity evidence applies only to identical blobs, not rerun.

Report copy-draft-ci-report.md and /private/tmp/userprofiles15413-copy-* retain
raw logs/patches/static/oracle/process evidence. Sourceauthorfinished/closed,
controllerrecords separate. Bounded exactfinalhead independentreview and fresh
automatic exactheadrequiredCI still govern; oldba74sevenrequiredsuccesses do
not qualify this unpublishedrepair. TASK15413 staysInProgress/AC4open until
verifiedmerge; verification-onlyhumanwaiver preserved, never runtime-transferred.

## November 7 Dev7a Qualified Integration

Before publication, authoritative dev advanced to
7a020da14ab28e220fab652269723e3a2a4abbcb through mergedPR3432's32 endpoint/
test-mode paths. Unpublished reviewedeb6 was not pushed stale. FreshPR3404
readback remained OPEN atba74, unarmed/noqueueentry/MERGE_QUEUEunset. Only this
next-to-merge branch was rebased, preserving backup
codex/userprofiles-verification-pre-7a020d=eb6; runtime branch stayed untouched.

Local f5be9844b59e276480ff93466b861109436d10c9 replays22/22patches range-diff
equal with no conflict, dropped patch or compatibility edit. Full29,081path
union proof:32upstream-only exactnewdev (including2deletions),19own-only exact
acceptedeb6,0combined,29,030untouched exact;29,079finalpresent. Entireapp subtree
matchesnewdev, apps subtree matcheseb6; copytest234fe33 andStandalone6b4381ad
blobs unchanged. Unrelated refs/worktrees/FETCH_HEAD preserved, no sourcepush.

Nine separate bounded compatibility runs pass822/no failures/errors/skips,
5351warnings: endpoint errors202, evaluations113, PromptStudio125, audio38,
credentialmatrix186, sharedHTTPcredentials120, writing18, testmodeguards17,
authMFA3. These are not a fullsuite or CI-equivalence claim. Actual fixture
closures inspected; SQLite media case explicit, no PGexecution/provisioning,
no unchanged UI rerun. LocalPython3.14 differsCI, warnings remain visible.

Full37existingPythoncompilepass; Ruff23exactinheritedupstream findings, own8
files0. UnfilteredBandit --ignore-nosec allcurrent1732findings incl1669B101,
35B105/22B106/3B311/3B608;0scanerrors,0newcontexts againstexactdev/acceptedeb6.
ThreeMEDIUMand63nonassertfindings remain inherited, not pristine or waived.
Fullquality5379baselineunchanged/shard871/5379/4ignored/baseline44/new0,
normalconfiguredhooks/diffpass. Reports/raw manifests andactualcommands in
dev7a-integration-report.md and /private/tmp/userprofiles15413-dev7a/.

Finaltwo-recordcheckpoint/independent bounded exacthead review then fresh
externalrefs/body/queue/rules checks and authorizedleasedsourcepublication
remain mandatory. Oldba74successes and its frontendfailure do not qualify this
unpublishedrebasedcandidate. TaskInProgress/AC4open untilverifiedmerge; no CI
retry/dispatch, policy/settings/gatechange or runtime-summary-waiver transfer.

## November 8 Dev4765 Qualified Integration

Published `eedf340645` genuinely passed all eight required checks, but
authoritative dev advanced to `476527719a806e005225fd358b0f830f8b9253df`.
The stale candidate was not merged. Current rules still require the seven
strict Actions checks plus trusted license, all from integration 15368, and
allow merge commits only. The PR remained open and unarmed, MERGE_QUEUE unset,
with no review threads or changes-requested reviews. The PR-specific human
summary waiver and body outside Cubic's generated block remain byte-identical.

The qualified local rebase is `a88b7f28ddaafc3e94e3080336e869e8d207955e`:
all 23 patches replay equal, no conflict, dropped patch or compatibility edit.
Complete 29,094-entry object/type/mode proof preserves 55 upstream-only paths,
19 own-only paths, zero combined paths and 29,020 untouched entries. New dev
contains accepted WebUI W3 storage/search/cache work and two Phase 2b planning
records; those plans were integrated, not implemented here. All 9,552 Python
entries and the backend application subtree are unchanged from approved eedf.
The VisualPackEditor `234fe33` and StandaloneHtmlWorkspace `6b4381ad` blobs remain
exact. Only the owned branch and unique backup moved during the rebase window;
later unrelated shared-ref movements were observed and left untouched.

Five separate, non-overlapping frontend selections pass with no failures or
skips: DB 525, eight owned UI files 307, folder/cache 230, mirror/hooks 1008,
caller supplement 21. These 60 files are not a whole-frontend or CI-equivalent
result. Default two-worker concurrency and budgets were retained. Application
WebUI typecheck exits 0. Actual tests used Node 25.6/Bun 1.3.5, not CI's Node
20.20.2/Bun 1.3.2; warnings and intentional diagnostic output remain explicit.

Broader touched TypeScript diagnostics remain 50 inherited errors: four own
errors reproduce byte-identically on approved eedf; 46 upstream errors reproduce
on exact dev, with three rendered union-order/truncation differences explicitly
retained alongside matching location/code/semantic contexts. Scoped ESLint
retains the exact accepted 31 errors/656 warnings, no fatal errors or ignored
files. No suppression, source repair or baseline change hides this debt. No new
Python scope exists; prior Bandit applies only to proven identical blobs and is
not claimed rerun. Full quality/shard guards and configured cached hooks pass,
no newly uncovered tests, no dependency or hook installation.

Old-eedf advisory CI remains non-green: six dependency-setup HTTP504 failures;
an AuthNZ shard's 55-minute outer abort during unchanged TestClient shutdown,
without established root cause; one WebScraping inventory assertion reproduced
identically by the official scanner on complete exact base/head archives; and
a report-only UI summary's frozen Bun install failure before report execution.
These are not declared harmless, repaired or required-gate failures. No source
fix was invented from these diagnostics, no retry/dispatch or policy change
occurred, and no old outcome predicts the new candidate's CI.

ADR assessment: no new durable decision or ADR for an object-preserving rebase;
accepted upstream storage behavior and existing ADR-085/059/084 remain intact.
Evidence: `dev4765-integration-report.md` in the verification SDD workspace,
`/private/tmp/pr3404-dev4765/`, `/private/tmp/pr3404-nov8-ci-triage.md`, and the
exact base/head inventory proof. Source author handed off with a clean tree
and no owned process. Controller records and independent bounded exact-head
review precede any guarded lease push. Entirely fresh current-head required CI
and authoritative latest-dev ancestry remain mandatory before merge; no old
green result, publication approval or PR3404 waiver transfers to runtime WP5.
