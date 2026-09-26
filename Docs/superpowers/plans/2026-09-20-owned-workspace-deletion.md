# Owned Workspace Deletion

Tracking: TASK-12020.50. The user approved the six deletion-review corrections
and a separate review of the existing unmerged workspace stream.

This plan does not enable the owned route, implement Duplicate, introduce local
Undo for server deletion, or erase library media, native files, Git roots, sandbox
volumes, independently created copies, or detached study materials.

## Decisions From Design Review

- The deletion boundary is an owner-authenticated, version-conditional soft
  deletion of the canonical workspace and its scoped conversations/messages.
  Keep workspace notes, artifacts and source associations as retained database
  records hidden by the deleted parent; do not describe this as secure erasure.
- The entire local cascade must commit or roll back together, including message
  search/sync/history side effects and quiz/deck detachment. Claim/lock the parent
  before walking children. Never swallow a child deletion error.
- Local atomicity does not prevent a writer that was admitted before deletion
  from adding new content afterward. Relevant child writers must lock/recheck the
  parent in their write transaction. SQLite and PostgreSQL need concurrency tests,
  not only mocked lock-call assertions. In-flight external work must not publish
  new workspace content after deletion; stopping its external execution is a
  separate contract, not an implied promise of Delete.
- Preserve existing versionless DELETE clients for compatibility; add an optional
  positive expected_version query. The new owned UI must always send its observed
  version and expected-user header using the captured connection. Never upgrade a
  stale version implicitly. Existing callers do not acquire the stronger client
  precondition guarantee merely because the server supports it.
- A minimal owner-only, no-store deletion-status read returns workspace_id,
  deleted and version from the canonical row, including tombstones. A missing row
  still returns 404. A 404, failed authentication, aborted request or network error
  is not evidence that this client deleted anything. A tombstone confirms current
  deletion state, not which client caused it. Do not expose retained content.
- Sharing data is in another database. Recipient access must deny deleted parents
  independently of revocation cleanup. Cleanup needs an idempotent retry and
  observable pending/failed state before the complete deletion UX is certified.
  A committed deletion must never be described as rolled back because subsequent
  cleanup failed. No distributed transaction is implied.
- Execute the synchronous local DB command off the event loop, retaining one
  operation-owned connection/transaction. Measure large-history behavior before
  introducing a new Jobs workflow solely for the local cascade.
- Before dispatch, retain outgoing scoped drafts durably. After an uncertain
  result, keep recovery state; allow explicit status inspection, not automatic
  DELETE replay. A still-active workspace requires fresh metadata review and
  confirmation. Account/activation/editor changes invalidate old callbacks.
- Confirmation must distinguish inaccessible workspace content from retained
  library/native/sandbox data. State that the UI cannot undo deletion. Preserve
  legacy local behavior separately; do not reuse local snapshot undo for owned
  records. Do not discard the current view/drafts before confirmed outcome.

## Stage 1: Local Transaction And API Foundation
**Goal**: Fix partial local deletion, accept client version preconditions, and
provide authenticated tombstone reads without enabling the UI.
**Success Criteria**: Failure injection rolls back parent, messages, conversations
and study detachments; conflicts do not alter content; account mismatch rejects
before DB access; status distinguishes active/deleted/missing; DB work is off-loop.
**Tests**: Existing workspace DB/API tests plus rollback, version, wrong-account,
tombstone, unavailable DB, and threadpool behavior regressions.
**Status**: Complete (bounded foundation; SQLite/PostgreSQL regression verified
after current-dev integration; not full deletion acceptance)

## Stage 2: Writer Fencing And Sharing Cleanup
**Goal**: Prevent post-delete publication and make cross-database cleanup truthful.
**Success Criteria**: Enumerated writer paths lock/recheck the parent; in-flight
operations cannot publish after deletion; sharing reads deny tombstones during
cleanup failure; cleanup retry/state is explicit and cannot resurrect content.
**Tests**: Real SQLite/PostgreSQL concurrent writers and readers, cleanup failure,
repeated cleanup, and recipient access/operation completion races.
**Status**: In Progress (direct content, ordinary chat creation and restore verified;
remaining writers and sharing cleanup pending)

2026-09-25 bounded slice: fence all 13 direct source/note/artifact mutations
inside their write transaction, including bulk source changes and artifact export
references. Preserve staged clone targets and SQLite device-client semantics.
SQLite transactions already start with BEGIN IMMEDIATE; PostgreSQL requires a
parent row lock. Verify deletion-first and writer-first races on both backends,
retained-content immutability and unchanged parent metadata. Chat/messages,
runtime/root/inventory/membership/study/migration writers and durable sharing
cleanup remain subsequent slices; this does not enable the owned route.

Slice implementation/review checkpoint: the initial runs passed 531 affected
regressions with 98 PostgreSQL-unavailable skips and 14 rollback checks with 14
skips. After an approved Docker recovery, all 69 real PostgreSQL fencing checks
pass without skips. A pre-existing source-selection BOOLEAN incompatibility was
reproduced and fixed with four passing dual-backend regression cases. Independent
review found no actionable issue in that fix. The final full affected regression
passes 649 tests with zero skips and four warnings (exit 0). See
`../../Development/workspace-content-writer-fencing-2026-09-25.md` for test-first
evidence, scope, and the remaining validation/implementation boundaries.

Next bounded slice: ordinary workspace-scoped chat creation now acquires the
same parent fence before INSERT, including caller-owned transactions. The chat
creation endpoint maps a lost deletion race to 409 instead of a generic 500.
Test-first coverage extends the live SQLite/PostgreSQL race matrix and verifies
Persona/Character API failures leave no partial conversation, messages or
settings. All 36 database cases (SQLite and real PostgreSQL, zero skips), two
API cases and four focused compatibility cases pass; independent read-only
review is clear. Earlier broad attempts were interrupted under host resource
pressure; after approved stale-test cleanup and fixture recovery, all 889
affected cases pass in eight groups and in one combined invocation, with zero
skips and identical case identities, including 118 PostgreSQL-named cases.
TASK-12020.50.1 repairs duplicate AuthNZ conftest registration: six guard tests
pass, 205 affected tests collect with AuthNZ, and full-project collection exits
0 with 67,556 tests. The ordinary creation slice is regression-verified.
See the direct-content evidence report's recovery section for failed harness
attempts, successful logs and limits. Full-project test execution is not claimed.
This is not Sync v2 upsert or existing-message/settings/restore certification.
Those paths require parent-before-child locking without inverting existing
message/conversation locks, including when a sync operation changes scope.

Restore slice: `ConversationStore.restore_conversation` now locks an active
workspace parent before taking the PostgreSQL conversation row lock. It rereads
scope under that lock, including for initially global conversations, and rejects
scope changes rather than acquiring a different parent after a child lock.
SQLite uses its existing BEGIN IMMEDIATE transaction. Version checks and
already-active idempotency remain intact for eligible chats. Missing/deleted
workspace parents reject without changing the retained chat or its projections.
The existing restore endpoint maps that conflict to HTTP 409. This does not
change the endpoint's already-active fast-path read or certify broader read
denial after deletion. Real-backend regression and review evidence is recorded
in the fencing report. Final affected regression passes 563 cases, including
132 PostgreSQL-named cases, with zero failures/errors/skips and four warnings.
Independent review is clear after adding the initially-global scope race.

Remaining chat implementation inventory (not covered by creation/restore):
- `ConversationStore.upsert_conversation_from_sync` can insert, resurrect, and
  replace scope/workspace identity. The Sync v2 chat materializer calls it
  directly. The workspace-scoped creation endpoint bypasses Sync via
  `_active_chat_sync_service`; enabling Sync does not change that endpoint path.
  Both the prior and destination workspace and concurrent scope changes must be
  considered; guarding only the requested destination is not enough.
- A parent ConflictError alone does not give Sync a conflict result: the chat
  materializer currently catches exceptions as failed/chat_projection_failed.
  Cover explicit error classification, accepted-envelope apply state, unchanged
  object-state projections, and repair/same-key replay after parent deletion.
  Do not imply atomicity between the Sync store and the product database.
- Existing conversation update/settings and message content, image and
  metadata publication need the same deletion boundary. Preserve deletion's own
  child tombstoning after the parent has been marked deleted.
- The character-message edit endpoint explicitly locks message, then metadata,
  then resume/conversation state. Other completion/greeting paths lock resume
  state before persistence. Parent admission must precede these locks, not be
  added only inside the final update. Preserve global and recipient-owned shared
  chat behavior, which must not require a local owner workspace row.
- Verify losing operations leave history versions, search/sync projections and
  settings unchanged, including caller-owned transactions and failures. Reuse
  real PostgreSQL concurrency fixtures rather than mocked lock-call assertions.

## Stage 3: Owned UI And Live Acceptance
**Goal**: Replace legacy Delete behavior only for server-owned activations.
**Success Criteria**: Account/version-pinned DELETE, exact typed confirmation,
durable drafts, explicit uncertain-result recovery and no fake Undo. Keyboard and
late callbacks cannot bypass recovery. Keep route disabled until all outstanding
owned mutation boundaries, not merely deletion, are certified.
**Tests**: Store/transport/hook/Header regressions and real backend plus WebUI/CDP
account-switch, conflict, response-loss, repeated-click and storage-failure cases.
**Status**: Not Started

## Stage 4: Unmerged Stream Review
**Goal**: Independently review the existing unmerged .49/.50 work for the same
correctness, recovery, authorization and maintainability concerns.
**Success Criteria**: Findings distinguish patch regressions, integration risks,
known disabled-route gaps and verification gaps; each has a concrete disposition.
**Tests**: Targeted reproductions for findings, scope checks and recorded evidence.
**Status**: Complete

Report: `../../Development/workspace-unmerged-review-2026-09-20.md`.
Three independent review scopes completed. Parent reran all six counterexamples
and the existing frontend regression set (1328 passing tests/51 files). Findings
were open at that checkpoint and subsequently corrected in the P1/P2 follow-ups.
Review completion does not mean the implementation is merge-ready.

## Verification

The Stage 1 real-backend probe passed 23 HTTP checks using the actual FastAPI app,
API-key authentication and isolated SQLite databases; lifespan orchestration was
off. It covered version/account preconditions, active/deleted/missing state,
no-store responses (including 401/412/404), legacy no-version behavior and rejected
repeat deletion. The final rerun includes the response-send wrapper that preserves
registered exception handlers and their headers.
This is not full startup, PostgreSQL, concurrent-writer or WebUI/CDP certification.
The server stopped after the probe. Script: `/tmp/tldw-owned-delete-live.py`;
log: `/tmp/tldw-owned-delete-http-final.log`; synthetic evidence directory:
`/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/tldw-owned-delete-http-t66w9sdm`.

OpenAPI export/type generation and fingerprint recheck passed: 2095 paths,
3204 schemas. Logs: `/tmp/tldw-owned-delete-openapi.log`,
`/tmp/tldw-owned-delete-codegen.log`, `/tmp/tldw-owned-delete-openapi-check-final.log`.

Independent review found and corrected two foundation defects: worker-thread
connections were retained after operations, and no-store headers omitted dependency
failures. New cancellation tests prove release occurs only after worker execution
finishes. The route-scoped response wrapper retains Starlette's exception-handler
dispatch rather than replacing application error handling. Separate tests cover
registered status/exception/validation handlers, retained headers and cross-owner
tombstones. Final wrapper review is clear: 13 targeted tests passed, 138 deselected;
log `/tmp/workspace-deletion-final-wrapper-review.log`.

Final production Bandit has zero findings/errors:
`/tmp/task12020-deletion-handlers-bandit.json`. Test-scope Bandit reports only the
pre-existing B106 fixture literal; existing endpoint Ruff BLE001 findings remain.
No clean full-project lint or build claim is made. The final backend regression
passed 186 tests, with 16 PostgreSQL-unavailable skips and 10 warnings; log
`/tmp/task12020-deletion-handlers-regression.log`. Scope includes workspace API,
rate-limit contracts, expected-user routing, sharing deletion hooks, message and
conversation stores, and PostgreSQL fallback-pool unit behavior. Pool unit tests
are not live PostgreSQL execution. All test sessions and the probe server stopped.
Temporary frontend dependency links used for verification were removed. Do not
equate this checkpoint with full deletion acceptance or completion of .49/.50.

## Integration Baseline

Fetched origin/dev on 2026-09-20: d72b1d2850ea947b6d12cac19f6b95867b68a580.
Worktree HEAD: c70387f496d82fcee92926bf3715bf5cd240ba88, 446 commits behind.
Upstream changes the database connection ownership machinery; the workspace
deletion implementation itself remains unchanged. No rebase/stash of the large
uncommitted .49/.50 patch was performed. Verification on this base is not proof of
integration with the newer operation-scope implementation; reconcile and rerun
before proposing a merge.

Follow-up: integration now uses a separate worktree at that fetched dev revision;
the original worktree remains untouched. Real PostgreSQL execution uncovered
request-checkout ownership and message-deletion sync parity defects, with bounded
fixes and test-first regressions. See
`../../Development/workspace-dev-integration-2026-09-20.md` for the final run and
remaining limits. Stage 2 had not started at that integration checkpoint. Its
direct-content writer slice is now in progress; the owned route remains disabled.
