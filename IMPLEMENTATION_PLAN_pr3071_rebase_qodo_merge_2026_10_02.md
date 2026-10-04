# PR3071 Latest Dev Rebase, Qodo Review, and Merge

Task: TASK-13421.1. PR: https://github.com/rmusser01/tldw_server/pull/3071.

Goal: Rebase the existing PR, resolve verified review/CI findings, and merge only
the qualified final head. Preserve the requester-authored Change summary.

ADR check: No new ADR required; integration and corrective review work do not
change a durable architecture rule. ADR-002, ADR-004, and ADR-006 govern tracking,
human ownership, and security verification.

## Current Continuation: 2026-10-03

The requester approved cleanup and continuation. Chatbook companion PR2968 is
merged at efea5e45cf2348dd0da50fedac215cd849143822. Its exact final head passed all
four hosted gates and Qodo with no active findings; actual authenticated GET-only
scope/history verification passed without mocks. No client release is claimed.
The server PR is published at2683d4fd28c3b950105c91e02aafe30ca1f7126e and current dev is4c4f197f68481664c58d4553bbfbb45dae157e28.
Use the clean attached isolated worktree; preserve the original live UAT checkout,
all protected services/data, historical evidence and the human Change summary.

### Stage A: Preserve and Reconcile
**Goal**: Integrate current dev without losing either side's qualified behavior.
**Success Criteria**: Preserve the published head and original stash inventory;
inspect all overlapping upstream paths; retain topology and compare the rebased
tree with an independently reconciled expected integration tree.
**Tests**: Ancestry, tree/range comparisons and focused conflict regressions.
**Status**: Complete

Integration findings: retain unconditional retrieval-only RAG from dev while
preserving the PR's protected admission tests. The real pipeline exposed a stale
whole-module unit mock; preserve actual exports and complete default option keys.
Frontend typecheck reproduced four missing dispatch-field errors in the shared
history helper; its parameter contract and RAG forwarding now include those
existing fields. Migration restoration uses complete canonical hydration and the
existing owner-qualified installer instead of a second unqualified snapshot path.
The initial body stays pending, transitions are fenced, retained empty-ID content
(including keywords) is not overwritten, and discard requires explicit consent.
Four initial restoration regressions and three fallback cases failed before their
fixes. The confirmation's initial declaration-order failure is retained separately.
Fresh combined frontend295 tests pass, including mounted real-store StrictMode
switch/unmount/account/success and canonical note-domain regressions; these are
unit/integration doubles, not UAT. CI-helper/route-auth29 pass. The requested
topology-preserving rebase finishes at0126148, exactly equal to independently
reconciled d7fc3cc/tree54432; dev4c4f197f ancestry and all69stashes are preserved.
The incoming Redis window-growth reload limitation is already documented under
TASK-13430 Known limits; it is not new to this PR and is not claimed remediated.

### Stage B: Fresh Qualification
**Goal**: Qualify changed runtime and incoming integration on the new source.
**Success Criteria**: Owning/incoming tests, TypeScript/lint/build and touched-scope
Bandit pass; actual Chrome raw-CDP acceptance uses live authenticated services,
databases, Gemma and embeddings without mocks or automatic resends.
**Tests**: Durable chat, RAG/history/prompt recovery, Research hydration and
workspace switching, desktop/mobile send/stream/stop/reload/source handling.
**Status**: Complete

Incoming backend485pass/5backend-specific skips includes all57 PostgreSQL HTTP
lifecycle cases unskipped. Owning527pass/1inheritedskip includes six durable
PostgreSQL cases; realRedis4 and docs212 pass. Production TypeScript exits0;
actual ESLint has0errors/34warnings; production build/token/budget checks pass
without raised budgets; fourteen backend/helper production files have Bandit0.
These results qualify tree54432, not subsequent frontend changes.

Fresh native Chrome exposed a cookie-session timeout preference bug: actual
Settings Save persisted60000ms, but reload resolved the default10000ms because
cookie authority omitted saved preferences. Bounded corrective scope copies only
the eight existing positive finite timeout fields from exact same-origin cookie
configuration, retaining credential precedence and rejecting foreign/manual
authority. Independent review verifies number-only consumers and native timer
overflow: normalize numeric strings and use the existing Settings2147483000ms
ceiling. These regressions produce expectedRED4 before their correction. Expanded
owning tests expose stale whole-router fixtures in two Settings suites; reuse the
actual neighboring data-router pattern. All14 owning suites289pass with a20-second
runner budget and unchanged assertions; production typecheck0, actual four-file
ESLint0errors/0warnings/0ignored, and fresh fourteen-file production Bandit0.
Initial failed runs, five-second runner expirations and ignored-path lint are
preserved. Fresh cookie-aware build and complete live Chrome UAT remain required;
no full acceptance or merge is claimed yet. The fresh cookie-aware build passes
unchanged token/bundle budgets and binds7225 tracked app files to2683d4f. Actual
native Settings fresh loaders retain60seconds. One new grounded native send has
retrieval-only200 and chat200; its original CDP completion collector timed out and
remains failed. A separate zero-send continuation verifies the actual protected
input/result pair, citations and restored draft without fabricating raw SSE.
Desktop/mobile fresh reload and explicit unavailable-workspace Retry pass with
zero sends. Workspace switching and fresh raw-stream recovery remain required.

The next workstream audit identifies an introduced checkpoint-readiness defect:
settled restoration rejection can leave the controller unavailable while both
rails report Ready. TASK-13421.1 covers the bounded corrective edit. Add a failing
mounted regression, project existing checkpoint status/error into the existing
history readiness gate, preserve valid empty-chat readiness, and verify retained
draft/recovery with no automatic send. Missing-model readiness is inherited from
dev and remains a separate follow-up, not claimed fixed by this correction.
The bounded correction has expected mounted RED8 and GREEN23; hook/panel163 and
rail/page/runtime64 pass (250 tests/seven suites total), with actual two-file
ESLint0errors/0warnings/0ignored. Independent review finds no new actionable
defect; actual controller publication through both rails remains for native UAT.
Archived2683 Chrome reproduces false Ready for an invalid native history URL,
fails the expected-unavailable assertion with0sends, preserves all checkpoint
rows and restores the original conversation/draft. Four native workspace-switch
cycles also pass without sends. Production TypeScript exits0. The new correction
changes only TypeScript; the fourteen-file production Bandit0 scan remains bound
to unchanged backend source. Fresh production build and native GREEN remain
required before this correction is accepted.

Committed correctioncb6b9a9703/treee893b4be passes fresh production build, token sync
and unchanged budgets (shared540.7KB/600KB, heaviest844.4KB/900KB). Native invalid
history-address GREEN projects unavailable through the actual controller and both
rails, disables sends, preserves checkpoints and restores the original valid
conversation/draft with0sends. Real prepared-context Stop/reload/reprepare/explicit
Send completes all three core assertions: actual admission/result stream frames,
one canonical logical input and exact prepared text. Its final style-reset driver
assertion fails and remains failed; zero-send cleanup/verification continuation
passes all three checks, including the actual stopped-input protected GET200,
original UI preference restoration and fresh-loader draft/conversation restoration.
It does not rewrite the original failed run. Earlier hidden-target,
retained-own-draft and invalid-warning-wait
drivers remain failed; accepted turns are inspected, never automatically resent.

Dev advances to d7997bc2052ac52c77427157fe3c5b2e2ca1843f with only five CI matrix
max-parallel caps at20. Application sources are unchanged by that delta. Integrate
and verify that exact workflow after final rebase. The matching original issue2033
task TASK-12135 is reopened through official CLI edit for inherited missing-model
readiness; prior hydration/offline criteria remain complete and new criteria are
unchecked. CLI task creation overflows twice; official MCP explicit-ID creation
times out300seconds and creates no record. No further blind retries or manual
Backlog file edits. The bounded next-item design is presented for approval; no
missing-model implementation is bundled into this checkpoint correction.

Final d799 rebase finishes at78207b2965/treeaf6ef2ee, exactly equal to the
independent expected integration tree. Application tree27a5b022 is byte-identical
to the qualifiedcb6 build; all3817 backend/config and7225 app files verify before
API restart. The historical replay briefly exposes older intermediate files;
only ownAPI22401 is stopped during that replay and replaced by38726 after final
verification. Pre-existing services remain untouched. Fresh actual workflow
contracts83pass/4warnings. Desktop/mobile reload and four native workspace-switch
cycles pass with0sends. Original10/eight protected rows, served contracts including
12negative controls, and all69original stashes pass read-only preservation.
Canonical workspace-notes GET200 returns an actual empty list; nonempty note body
mapping is unit/integration coverage, not claimed as native UAT.

The original browser-target preservation check fails: baseline18092page is absent
and Chromium browser-ui IDs have changed; no tab-close command is issued in this
continuation. Both own18099tabs remain. A separate data/contracts/stash check passes
and explicitly retains false target-preservation status; it does not relabel the
original failed browser check. Final publication, Qodo/CI and protected merge remain
unfinished. Subsequent evidence-only edits must retain this exact application tree.

### Stage C: Review and Hosted CI
**Goal**: Publish only a verified integration and address actual final-head review.
**Success Criteria**: Preserve the human summary; use an exact publication lease;
document merged companion disposition; fresh exact-head Qodo/CI and all review
threads are qualified, with no hidden or bypassed failures.
**Tests**: Per-finding regression/security checks and actual hosted readback.
**Status**: In Progress

Exact-lease publication2683d4f and body readback preserve the human Change summary.
Qodo's updated dashboard for that exact head has zero active bugs/rule violations/
cross-repository conflicts; all eight review threads are resolved. Hosted checks
remain queued as of the latest snapshot. New checkpoint edits require renewed
publication/review/CI qualification; the2683d4f review does not qualify a later head.

### Stage D: Protected Merge and Tracker Reconciliation
**Goal**: Land the qualified server work before starting another workstream.
**Success Criteria**: Latest head/base and human summary are rechecked; normal
protected merge is verified by commit parents/tree; Backlog and epic statuses
reflect actual completed criteria, leaving any genuine residual work open.
**Tests**: Fresh GitHub/repository readback and evidence-linked issue review.
**Status**: Not Started

## Historical Qualification Before This Continuation

## Stage 1: Preserve and Rebase
**Goal**: Rebase on latest fetched dev9958110df2a9011e19f48b0eae821353e19d4af8,
retaining the qualified8140 integration and the new request-scoped ChaCha reuse fix.
**Success Criteria**: Backup refs retain published31a and tracking3554. Rebased
HEAD940fd21b exactly equals clean integration tree4d696befb04ddd3c26fd5307a4ecf54218624713,
before subsequent task/plan qualification updates. Incoming eight paths match
dev9958 byte-for-byte; prior reviewed fixes remain intact.
**Tests**: Git ancestry/tree comparison; original stash inventory unchanged.
**Status**: Complete

## Stage 2: Verify and Publish
**Goal**: Verify the rebased implementation and resolve published-ADR drift.
**Success Criteria**: Docs refresh and affected owning regressions pass; normal
hooks pass; publish the9958 integration using an exact lease bound to current
published head31a94f5dbadf6f7da6e776225f43a6ef88fe02c8.
**Tests**: Docs refresh suite, changed CI/test-isolation regressions, owning Chat
Workspace regressions, touched-scope Bandit, and exact production-source binding
to prior real Chrome desktop/mobile acceptance. Unit doubles are not UAT.
**Status**: In Progress

Verified: docs33; combined owning273; incoming fixtures52; Chat180passed/24skipped;
CI helper/workflow56 and formatted helper5. Independent review's child-PATH
lookup defect is fixed with a red/green regression; no runtime changes. Dev
advanced to38b09af8e92b3d9a2a00aced22d6b992dbae9435 with only relay task metadata;
include that final delta before publishing. Retain initial path-guard test
invocation failures and raw Bandit test-assert diagnostics in the evidence.
The final rebase completes cleanly and matches expected treea082f5f9. Every
applicable configured pre-commit check passes. Published head5860186188 is
verified OPEN and ready with the exact prepared body and unchanged human summary.

## Stage 3: Qodo and CI
**Goal**: Mark ready, evaluate every PR finding, and qualify the final head.
**Success Criteria**: Each actionable finding fixed with regression coverage or
answered with verified evidence; no unresolved blocking finding; exact-head CI
successful. Do not interpret missing Qodo review as approval.
**Tests**: Per-fix red/green tests, security checks, final GitHub checks/threads.
**Status**: In Progress

Qodo completed review in comment5958856767: eight findings and eight inline
comments. Bounded fixes cover absent-snapshot activation, canonical preview
identity (including null URL), SSE control-frame forwarding, streamed error
propagation through the existing parser, and schema-helper types/docs/unit marker.
Frontend transport70, activation278 (independent review290), preview53, backend
175, scope/marker17 pass. Typecheck, ESLint, Python lint/format and runtime Bandit
pass; inherited Node/i18n/test warnings remain recorded, not suppressed.
The current Chatbook client lacks diagnostics scope arguments; the server's
explicit owner/scope checks remain intact. Companion-repository approval is
pending; do not call this compatibility gap fixed.
Dev advanced during review to8140e493f2d0a79e2039084930151eba6565df82 with
AuthNZ/resource-governor changes. Integrate and qualify that actual backend base
before publication. The new frontend build/token/budget checks pass; preliminary
38b backend availability is not UAT of8140. Final no-mock Chrome UAT, Qodo
disposition and exact final-head hosted CI remain required before merge.

Fresh rebased8140 owning/incoming regressions266pass. Live Chrome proved one
retrieval without generation and one verified Gemma input/result pair, but dropped
heartbeats; that failed acceptance is retained, not called passing. Config enables
the unified transport, which re-filtered pipeline-generated heartbeats. The minimal
existing-queue preservation fix has expected RED then GREEN3; full owning507pass,
1skip, runtime Bandit0. Independent scoped review found no actionable regression.
Do not resend the completed original turn.

Final independent review found three further bounded recovery defects, tracked as
TASK-13421.1.5/.6/.7: suppress unsupported selected-durable legacy regeneration,
preserve prepared recovery text exactly once, and provide router-aware manager
navigation after failed activation. Tests100/97/101 pass respectively; docs33pass.
F1/F2 scoped independent reviews pass. F3's raw-link extension defect was
confirmed RED through real hash/memory navigation and corrected with existing Link.
Frontend TypeScript8GB passes; the initial default-heap OOM is retained. Actual
ESLint analyzes all touched shared files with zero errors and11 inherited ChatPane
warnings; out-of-base ignored-file invocation is not lint qualification. Expanded
upstream Bandit retains19 low token-type literals in files byte-identical to dev,
not new findings or a zero-findings expanded scan. Final production build/source
binding and native desktop/mobile acceptance remain under qualification.

Published e7d0013 native Chrome acceptance captures three real heartbeat frames,
one retrieval-only request and one protected Gemma input/result pair. Citation
expansion, qualified draft restoration after fresh reload, mobile keyboard access,
canonical previews, and real 404 manager navigation pass without inference on reads.
Native Stop/Verify/Reprepare/explicit Send preserves one canonical logical input;
unsupported legacy retry controls are absent. Earlier driver failures and the
accepted result-unknown input remain recorded and are not resent automatically.
Prepared full-source/preset recovery is still under bounded fresh-conversation UAT.

Exact-head CI found two stale behavior-copy expectations and unqualified legacy
model-picker wording in the guide. TASK-13421.1.7 repairs only documentation and
tests: expected RED2 then GREEN7; published docs refreshed; docs35pass; production
Chat Bandit0. Application runtime files are unchanged. The new head still requires
fresh hosted CI and Qodo disposition; companion SDK scope approval remains pending.

Final31a native acceptance closes bounded prepared-input recovery with a separate
zero-send protected-row/draft continuation; its original focus-failure artifact
remains failed. Four settled B/A cross-surface switches pass with zero sends.
The earlier switch assertion sampled during actual history loading; read-only
investigation proved natural restoration and preserved checkpoint contents except
the expected session reference. Independent copy/UAT review finds no issue.
Hosted31a auth149/0skips includes six durable PostgreSQL passes; chat861passed/
30skips includes all60 image-recovery passes and its strict PG snapshot unskipped.
The RAG shard's initial apt-mirror failure occurred before tests; the targeted
same-head retry passes. These results qualify8140, not the newly advanced base.

Dev advanced during hosted CI to9958110 with eight changed paths: two ChaCha
request-operation runtime helpers, their regression, RG replay/identity tests and
task metadata. Frontend and both PR-owned runtime paths remain nonoverlapping.
Preserve31a, rebase, verify exact expected integration and incoming/owning tests,
then bind a fresh actual backend before UAT. Current imported73ada backend must
not be claimed as UAT of9958. Companion SDK authorization remains pending.

Rebased9958 head940fd21b matches the complete expected integration tree exactly;
all68stashes retained. Frontend bytes remain identical to published31a. New API
PID13156:18096 runs this actual source in the existing real non-test environment,
with3817 tracked application/config hashes checked before native Chrome acceptance.
Official incoming regressions107pass, including all57 PostgreSQL HTTP lifecycle
cases unskipped; owning chat507pass/1skip. Four-production-path Bandit0findings/
0errors. Both initial test invocations incorrectly placed basetemp beside TMPDIR;
their path-guard failures are retained. Corrected paths nest beneath TMPDIR,
without changing application guards or test expectations.

Fresh native Chrome acceptance on9958 captures one retrieval without generation,
one deliberate durable Gemma send, four live heartbeat frames, canonical protected
input/result and source metadata, citation expansion, previous history/draft
reload, and new draft reload. Desktop/mobile fresh reload and native keyboard
access pass with zero inference requests and zero horizontal overflow. The runner
initially waited for enabled Send after the composer correctly cleared. A recorded
zero-send native draft continuation kept its original receipt collector attached;
the wait is corrected for future runs, with no resend or fabricated receipt.
An independent incoming9958 scoped review found no actionable issue. Published31a
CI finished293success/35skip/1neutral; these historical checks do not qualify the
new head. Fresh published-head Qodo/CI and SDK disposition remain merge gates.

Fresh9958 four settled B/A/B/A Research/Chat transitions also pass with zero
sends; A restores its exact conversation/draft and B's checkpoint remains
byte-identical. Read-only preservation against the new API passes original10/eight
protected rows, served contracts/12negative controls, six current baseline Chrome
targets and all68stashes. Already-absent historical target IDs are not claimed
preserved. Original shared services, unrelated files and backup refs remain intact.

## Stage 4: Merge and Read Back
**Goal**: Merge the verified PR into dev without bypassing checks.
**Success Criteria**: Human summary byte-identical, latest head/base rechecked,
GitHub reports MERGED and its merge commit is verified on dev; task finalized.
**Tests**: Fresh PR/review/check readback and GitHub merge outcome. Preserve live
services, profiles, tabs, drafts, databases, unrelated files, and backup refs.
**Status**: Not Started

## Stage 5: Actual Latest Dev Quota Integration
**Goal**: Integrate actual dev3700e2d6e7 while preserving parent8c8509b6 and
separate model PR3159, live services, tabs, data, stashes and backup refs.
**Success Criteria**: Topology-preserving rebase matches the independently
reconciled tree except the canonical regenerated OpenAPI fingerprint and scoped
review corrections. Persistence-only receipts do not inflate monthly estimates;
rejected selected-durable projections still account for consumed provider tokens
exactly once, without gaining result authority. Final head passes owning tests,
security, source-bound no-mock qualification, exact-head Qodo and all seven actual
dev ruleset gates before normal merge. No close/reopen or bypass is needed.
**Tests**: Focused quota RED/GREEN HTTP regressions; incoming Usage/UserProfile,
durable Chat/RAG, official PostgreSQL operation lifecycle and quota repositories;
OpenAPI drift/types, production frontend verification and touched-scope Bandit.
**Status**: In Progress

The actual dev ref advances to3700 while PR metadata still reportsd799. A temporary
source clone isolates this integration from the live model worktree. Rebase
HEAD54d617eb matches independent tree48e709f5 for all implementation files; only
fingerprint and our tracking note differ. The canonical 2107-path/3264-schema
fingerprint293312e270f0 matches a fresh independent export, and ignored generated
frontend API types are regenerated. Independent review verifies two P2 quota
interactions above. Initial temporary-path-guard failures and the bare --check
argument error remain failed artifacts; corrected isolated invocations do not
change guards. No new-source UAT or hosted merge qualification is claimed yet.

Corrective verification: valid bounded citation fixtures produce2expected402
failures and2uncited passes before estimator correction. Rejected projection
tests produce3missing-accounting failures before their correction. Fresh valid
targeted7pass; full owning498cases finishes497pass/1backend-specific skip with
ambient credential fallback explicitly absent in isolated real-usage tests.
Official PostgreSQL68 and legacy history114 pass without skips; actual frontend
157 plus Research stage3 53 pass. Production typecheck exits0. Incoming26 and
corrected2production Python paths have Bandit0findings/0errors; changed production
and regression Ruff checks pass, with26 inherited shared-fixture diagnostics
verified unchanged. Stable configured hooks pass. Initial invalid source fixtures,
temporary-path/socket failures, parallel shared-counter interference, cleared
credential fixture failures, wrong frontend selector and concurrent-task hook
file-change detection remain separate failed artifacts. Independent final review
finds no actionable issue in corrected production code or isolated fixture.
Real19-database snapshots preserve shared originals. Fresh production build,
Chrome acceptance and exact final-head hosted gates remain required.
