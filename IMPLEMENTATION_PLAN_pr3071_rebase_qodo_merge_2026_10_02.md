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
The server PR remains published at5b6258f909 and current dev is4c4f197f68481664c58d4553bbfbb45dae157e28.
Use the clean attached isolated worktree; preserve the original live UAT checkout,
all protected services/data, historical evidence and the human Change summary.

### Stage A: Preserve and Reconcile
**Goal**: Integrate current dev without losing either side's qualified behavior.
**Success Criteria**: Preserve the published head and original stash inventory;
inspect all overlapping upstream paths; retain topology and compare the rebased
tree with an independently reconciled expected integration tree.
**Tests**: Ancestry, tree/range comparisons and focused conflict regressions.
**Status**: In Progress

### Stage B: Fresh Qualification
**Goal**: Qualify changed runtime and incoming integration on the new source.
**Success Criteria**: Owning/incoming tests, TypeScript/lint/build and touched-scope
Bandit pass; actual Chrome raw-CDP acceptance uses live authenticated services,
databases, Gemma and embeddings without mocks or automatic resends.
**Tests**: Durable chat, RAG/history/prompt recovery, Research hydration and
workspace switching, desktop/mobile send/stream/stop/reload/source handling.
**Status**: Not Started

### Stage C: Review and Hosted CI
**Goal**: Publish only a verified integration and address actual final-head review.
**Success Criteria**: Preserve the human summary; use an exact publication lease;
document merged companion disposition; fresh exact-head Qodo/CI and all review
threads are qualified, with no hidden or bypassed failures.
**Tests**: Per-finding regression/security checks and actual hosted readback.
**Status**: Not Started

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
