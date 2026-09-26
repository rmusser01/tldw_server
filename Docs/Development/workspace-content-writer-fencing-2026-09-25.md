# Workspace Content Writer Fencing

Tracking: TASK-12020.50, deletion Stage 2, first bounded slice.

Worktree: `codex/shared-workspace-dev-integration`, HEAD
`d72b1d2850ea947b6d12cac19f6b95867b68a580`. This continues the existing
uncommitted integration work; it is not a new rebase or a merge-ready checkpoint.

## Changes

All 13 direct source/note/artifact mutation methods now check and lock their
parent inside the same write transaction. This covers creation, updates,
deletion, bulk source selection/review/reordering and artifact export references.
A missing/deleted parent raises the existing domain conflict before child writes.
The source retry path cannot return a retained row after parent deletion.

SQLite reuses the existing `BEGIN IMMEDIATE` transaction boundary. PostgreSQL
uses `SELECT ... FOR SHARE`, so deletion must wait for admitted writers and
blocked writers recheck the parent after deletion commits. The helper does not
modify parent version/timestamps or introduce owner checks based on SQLite
device client IDs. Internal staged clone targets remain writable; existing
public visibility/authorization checks remain the callers' responsibility.

This does not stop external execution or fence every workspace writer. Scoped
chat/messages, runtime/root/inventory/membership/study/migration publication and
durable cross-database sharing cleanup remain subsequent Stage 2 work. It does
not certify recipient read races or enable owned-workspace routes.

## Verification

- Test-first run: 22 failed, 39 passed, 61 PostgreSQL-unavailable skips. Failures
  demonstrate deleted-parent writes succeeding, late publication after a deletion
  commit, and raw constraint errors for missing-parent note/artifact creation.
  `/tmp/workspace-content-fence-red.log`.
- Focused implementation run: 61 passed, 61 PostgreSQL-unavailable skips. Covers
  all direct mutations, staged clone compatibility, retained content/version
  immutability, parent metadata preservation, deletion commit/rollback races,
  and deletion waiting for the writer's outer transaction.
  `/tmp/workspace-content-fence-green.log`.
- Four additional checks passed: source/artifact/note API admission-to-write
  deletion races return 409, do not enqueue ingestion, and do not write content;
  SQLite writes still work when the device client ID changes.
  `/tmp/workspace-content-fence-api.log`. These use the real DB and FastAPI
  router with test authentication overrides, not full-server HTTP acceptance.
- Ruff: touched production/test files clean after import ordering.
- Bandit: changed database module has zero findings/errors.
  `/tmp/workspace-content-fence-bandit.json`.
- Broader affected regression: 531 passed, 98 skipped, four warnings, exit 0.
  Covers workspace API/subresources, artifact validation, clone target lifecycle,
  clone service/operations and output Jobs API/worker behavior.
  `/tmp/workspace-content-fence-regression.log`.
- Independent read-only review found no actionable defect in this slice or new
  lock-order cycle in inspected callers. It is not whole-stream merge approval.
  The suggested writer-first rollback and artifact-version failure tests were
  added: 14 passed, 14 PostgreSQL-unavailable skips, four warnings.
  `/tmp/workspace-content-fence-rollback.log`. This follow-up overlaps six
  original writer-first cases; counts should not be added as unique tests.
- Concurrent coverage exercises source/note/artifact add and update. Bulk,
  delete and export-reference methods have sequential tombstone coverage.
- Final touched-file Ruff and whole-worktree `git diff --check` pass. No frontend
  changes, full-server startup, browser acceptance, commit or push in this slice.

## PostgreSQL Recovery And Parity Fix

The official fixture could not reach local PostgreSQL. Its Docker auto-start
stalled in `docker rm`; a separate read-only `docker ps` also stalled. Only this
session's test/Docker CLI processes were interrupted. No daemon/container was
restarted and no CI checks were cancelled. Subsequent runs use
`TLDW_TEST_NO_DOCKER=1` and retain the fixture's explicit unavailable skips.
No listening PostgreSQL service was found for the fixture to reuse.
The selected context is `desktop-linux`; the default socket points to the same
daemon. A bounded read of its `/_ping` endpoint returned
`Docker Desktop is unable to start`, confirming a service-level blocker rather
than an alternate-context selection issue.

The preceding blocker was resolved after the user approved restarting Docker
Desktop. The graceful restart timed out; the supported force-stop/start commands
recovered the daemon. Its historical log reported an engine crash due to disk
exhaustion; the host had approximately 74 GiB free at recovery. No Docker pruning,
data reset or CI cancellation was performed. The existing PostgreSQL fixture
service is reachable on port 5434. Runs use `POSTGRES_TEST_PORT=5434`,
`TLDW_TEST_NO_DOCKER=1` and `TLDW_TEST_POSTGRES_REQUIRED=1` with the official
per-test database fixtures, rather than recreating a container.

Real PostgreSQL execution then exposed a pre-existing source-selection defect:
the insert bound an integer to a BOOLEAN column, and bulk selection assigned
integer literals. Bind a Python boolean and use SQL FALSE/TRUE instead; SQLite
retains the same logical behavior. A dual-backend regression covers selected and
unselected inserts, individual updates, bulk selection and clearing selection.

- Selection regression RED: two PostgreSQL failures and two SQLite passes.
  `/tmp/workspace-selection-pg-red.log`.
- Selection regression GREEN: four passes, zero skips.
  `/tmp/workspace-selection-pg-green.log`.
- Real PostgreSQL direct-content fencing, staged-parent, concurrency and rollback
  verification: 69 passed, 243 deselected, four warnings, zero skips, exit 0.
  `/tmp/workspace-content-fence-pg-verified.log`.
- Focused independent read-only review of the boolean fix and regression found
  no actionable issues. The reviewer did not independently execute tests.
- Touched-file Ruff passes; production Bandit reports zero findings/errors.
  `/tmp/workspace-selection-pg-bandit.json`.
- Final full affected regression: 649 passed, zero skips, four warnings, exit 0
  in 348.45 seconds. Includes workspace API/subresources, artifact validation,
  SQLite/PostgreSQL clone target lifecycle, clone service/operations and output
  Jobs API/worker tests. `/tmp/workspace-content-fence-full-pg.log`.

The earlier skipped runs above are historical evidence, not PostgreSQL parity
evidence. The recovered runs close that specific validation gap; they do not
constitute full-server startup or browser acceptance.

The Backlog record remains In Progress. Its previously duplicated empty final
summary markers were normalized with the official Backlog edit tool, without
claiming this larger work item complete. Scoped chat/message publication fencing
is the next implementation slice. Owned-workspace routes remain disabled.

## New Chat Creation Follow-up

The next bounded increment fences `ConversationStore.add_conversation` before
INSERT when the normalized scope is workspace. Caller-owned transactions retain
the parent lock through their own commit/rollback. Global chats and internal
staged-clone creation remain supported. The chat creation endpoint now maps
`ConflictError` through the existing 409 handler, rather than its generic 500.

Verification:
- SQLite test-first failure: creating a chat through a caller-owned transaction
  after workspace deletion did not raise a conflict. One failure, exit 1.
  `/tmp/workspace-chat-create-red-bounded.log`. The earlier mixed-backend RED
  was interrupted during slow teardown and is not terminal verification evidence.
- API test-first failure: character-backed creation after preflight deletion
  returned 500 rather than 409. `/tmp/workspace-chat-create-api-red.log`.
- Final creation matrix: 36 passed (18 SQLite, 18 real PostgreSQL), zero skips,
  four warnings, exit 0, 1309.34 seconds under heavy host load.
  `/tmp/workspace-chat-create-green.log`. Covers missing/deleted parents,
  deletion-first/writer-first commit and rollback, caller-owned transactions,
  parent metadata, staged targets, global scope and cascade of the winning chat.
- Persona/Character API races: two passed, zero skips. No residual conversation,
  message or settings rows. `/tmp/workspace-chat-create-api-green.log`.
- Scoped conversation and recipient-owned shared-chat compatibility: four
  passed. `/tmp/workspace-chat-create-compatibility.log`.
- Independent read-only review of this increment found no actionable issue or
  lock inversion in the inspected creation paths. It did not run tests itself.
- Bandit: zero findings/errors in both changed production files.
  `/tmp/workspace-chat-create-bandit.json` and
  `/tmp/workspace-chat-create-api-bandit.json`.
- Both changed test files pass Ruff. The production files retain six
  ConversationStore and four chat-endpoint Ruff findings also present at HEAD;
  none are introduced by this increment. No clean whole-module lint claim.

Broader verification remains incomplete. Repository-wide `python -m pytest -x
-q --tb=short` stops at AuthNZ collection because unchanged Admin_Webhooks modules
register `AuthNZ.conftest` as a plugin before pytest discovers it as a conftest.
See `/tmp/workspace-chat-create-project-suite.log`. The separate affected run
selected 771 tests but was interrupted after 149 passes, 118 deselected and five
warnings as the host load exceeded 50 and free disk space fell to about 18 GiB
(100% displayed capacity). `/tmp/workspace-chat-create-regression.log` is partial
evidence, not a green suite result. No other task's processes or CI were stopped.

This increment does not fence Sync v2 conversation upserts, existing-chat edits,
settings, restoration, or late message publication. The workspace-scoped chat
creation endpoint explicitly bypasses Sync v2, even when Sync is enabled;
direct Sync materialization uses a separate, still-unfenced upsert path. The plan records the required
parent-before-child lock ordering and concurrent scope-change considerations.
Complete broader verification before advancing this slice; the larger task and
owned-route enablement remain unfinished. No commit or push was performed.

### Collection Repair And Resumed Verification

TASK-12020.50.1 restores the existing `authnz_full_fixtures` plugin bridge in
three Admin Webhooks modules and `Ingestion_Sources/test_service_postgres.py`.
Each had registered `AuthNZ.conftest` directly, conflicting with pytest's own
conftest discovery. The four-line repair changes no fixtures or test assertions.

- Existing plugin-isolation guard RED: one failed, five passed, exit 1.
  `/tmp/workspace-authnz-collection-red.log`.
- Combined affected-suite/AuthNZ collection RED: duplicate plugin registration,
  exit 1. `/tmp/workspace-authnz-combined-red.log`.
- Guard GREEN: six passed, four warnings, exit 0.
  `/tmp/workspace-authnz-collection-green.log`.
- Combined collection GREEN: 205 tests collected, exit 0.
  `/tmp/workspace-authnz-combined-green.log`.
- Ruff passes on all four changed test modules. Bandit reports 403 existing
  findings (401 test assertions and two test constants), identical by file,
  line, rule and message to archived HEAD files; both scans have no errors.
  `/tmp/workspace-authnz-collection-bandit.json` and
  `/tmp/workspace-authnz-bandit-head.json`. No new findings or suppressions.
  The earlier stdin baseline attempt had a Bandit plugin error and was replaced
  by this file-based comparison. Self-review confirms only plugin names changed.
- Full-project collection printed 67,556 tests collected in 183.63 seconds,
  then exited 130 after interruption during shutdown. This demonstrates progress
  past the original failure but is not a clean full-project invocation.
  `/tmp/workspace-authnz-project-collection.log`.
- The resumed 771-case affected regression did not report a failure before
  interruption, but did not reach a terminal test summary. SIGINT did not stop
  it; SIGTERM ended only that identified pytest process (exit 143).
  `/tmp/workspace-chat-create-regression-resume.log`. No partial pass count is
  promoted to suite evidence.

During the resumed run, available disk fell from about 28 GiB to 10 GiB and
system load rose above 40. Broad verification remains capacity-blocked; avoid
another unchanged broad retry until sufficient headroom is available. No other
task's process, container or CI run was stopped, and no data was pruned. Existing
chat/message and Sync v2 fencing remain pending, with owned routes disabled.

### Read-only Continuation Under Capacity Pressure

Available disk subsequently fell to 2.6 GiB. No further tests were launched.
The positively identified logs and baseline artifacts from this task total
about 27 MiB; removing them would not resolve the capacity shortage. Shared
pytest cleanup paths mentioned in logs do not prove this task owns their data,
so they were not deleted. No repository runtime changes were made.

Source inspection corrected the earlier Sync-enabled creation claim above:
`character_chat_sessions._active_chat_sync_service` returns None for workspace
scope, and `create_chat_session` uses that helper. This narrows the outstanding
Sync gap; it does not certify the existing-conversation or message paths.

The next Sync slice must also test error classification and durable apply state.
`ChatConversationMaterializer.apply` currently catches DB exceptions generically
and stores `failed/chat_projection_failed`. Simply adding a parent ConflictError
would therefore not automatically produce a Sync conflict outcome. Verify the
accepted envelope remains truthful, object state is not advanced, and repair or
same-key replay cannot resurrect content after parent deletion. These are
source-derived requirements, not executed counterexamples or completed fixes.

When capacity is restored, run the affected regression in sequential fresh
processes with explicit suite-specific temporary directories and aggregate all
results. Preserve every selected case and the already-required real PostgreSQL
matrix; do not treat partitioning as evidence of cross-file isolation. Run a
combined invocation afterward when capacity permits. The earlier monolithic
retry grew beyond 6 GiB RSS before termination, so another identical retry is
not a useful first recovery step.

### Approved Cleanup And Bounded Regression Recovery

The user approved removal of exactly `pytest-2175` and `pytest-2176` under the
macOS `pytest-of-macbook-dev` temporary root. Their recorded owner PIDs 9608 and
50589 were absent, and an open-file/working-directory check found no users.
Removed only those two directories, including their test-created unreadable
subdirectories. The active `pytest-2193` and `pytest-2194` were not touched.
Available disk subsequently reached roughly 290 GiB; other concurrent cleanup
may account for part of that recovery.

The first bounded attempt used `/tmp` for basetemp, outside the configured
database path allowlist, and the previous PostgreSQL container was absent.
It ended with 57 passes, one failure and 54 setup errors, exit 1. These were
harness/environment failures, not passing evidence or product regressions.
Preserved `batch-1.log`, `report-1.xml` and `fixture-probe.log` under
`/tmp/workspace-chat-regression-batches.RiHClX`.

Moved task-owned basetemp directories to the approved macOS temporary root,
without changing the allowlist. The official PostgreSQL fixture provisioned
`tldw_workspace_12020_50_pg` on port 5434; four dual-backend selection checks
passed with zero skips. Subsequent runs disable Docker auto-start and require
PostgreSQL, preserving explicit failure rather than silently skipping it.

Eight sequential fresh-process groups then passed using pytest-split
`least_duration` and fixed seed 47850. JUnit reconciliation found 889 cases,
889 unique identities, 118 PostgreSQL-named cases, zero duplicates, zero failures,
zero errors and zero skips. Group counts are 112 then seven groups of 111.
Reports: `verified-report-1.xml` through `verified-report-8.xml`; logs:
`verified-batch-1.log` through `verified-batch-8.log` in that same directory.

Full-project collection now exits 0: 67,556 tests collected in 58.18 seconds,
`project-collection.log`. This closes the collection gap, not full-project test
execution. Fresh Bandit on ChaChaNotes_DB, ConversationStore and the chat
endpoint reports zero findings/errors (`production-bandit.json`). Ruff passes
on the six touched regression/fixture-registration test modules.

The final combined invocation passes all 889 affected cases, zero failures,
zero errors, zero skips, seven warnings, exit 0 in 2262.66 seconds. Its JUnit
case identities exactly match the eight batch reports, including all 118
PostgreSQL-named cases. Evidence: `combined.log` and `combined-report.xml` in
the same directory. This closes the ordinary chat-creation slice's affected
regression gap, including cross-file execution; it does not certify the
remaining deletion writers or live UI acceptance.

All verification processes finished. Free disk remained approximately 283 GiB.
The task-specific PostgreSQL fixture remains available for subsequent tests.
No production code, route enablement, commit or push accompanied this recovery.

## Existing Conversation Restoration

The next Stage 2 slice fences `ConversationStore.restore_conversation`. Its
production API caller does not enter an outer child-locking transaction and
already maps `ConflictError` to HTTP 409. The store locks the active parent
first, then locks/rereads the PostgreSQL conversation and rejects scope changes.
The reread is unconditional for PostgreSQL, including initially global chats.
SQLite already serializes the transaction with BEGIN IMMEDIATE. This preserves
nested transaction lifetime and does not add a parent metadata write.

Test-first evidence:
- Restoration after parent deletion reproduced on SQLite and PostgreSQL: two failures,
  both DID NOT RAISE ConflictError, `/tmp/workspace-chat-restore-red.log`.
- Parent fencing alone passed 14 version/idempotency and deletion-first/restore-
  first commit/rollback cases, `/tmp/workspace-chat-restore-green.log`.
- Concurrent Sync scope reassignment still bypassed the parent-only fence:
  two PostgreSQL failures, `/tmp/workspace-chat-restore-scope-red.log`.
- Locked scope reread then passed all 17 focused DB/API cases, zero skips,
  `/tmp/workspace-chat-restore-final.log` and corresponding `.xml`.
- Review found no concrete implementation defect but requested the missing
  initially-global race. Added a real PostgreSQL global-to-deleted-workspace
  case; it passes, `/tmp/workspace-chat-restore-global.log`.

Added explicit missing/deleted-parent checks for active and trashed retained
Sync projections. The first broad run finished with 558 passes and four setup
failures: nonexistent-parent inserts violate the actual workspace foreign key.
Confirmed that constraint in the task-owned PostgreSQL log and corrected the
fixture to create the parent first, then remove it and assert ON DELETE SET NULL.
All eight corrected edge cases pass (`/tmp/workspace-chat-restore-projection-fixed.log`).
The final broad workspace/chat/transaction/error-mapping regression passes
563 tests in 581.21 seconds (exit 0, four warnings). JUnit confirms zero failures,
errors or skips and 132 PostgreSQL-named cases. Evidence:
`/tmp/workspace-chat-restore-verified.log` and `.xml`. This includes the added
initially-global race and corrected projection fixtures in the same invocation.
The earlier failed run is retained in `/tmp/workspace-chat-restore-regression.log`
and `.xml`.
Production Bandit: zero findings/errors, `/tmp/workspace-chat-restore-bandit.json`.
Touched tests pass Ruff. ConversationStore has six unchanged baseline Ruff
findings, reproduced from HEAD; no clean whole-module lint claim is made.

Limits: the already-active API fast-path is a read, not restoration, and remains
outside this mutation slice. Direct Sync upsert can still publish or reparent
after deletion; these tests use that current behavior to verify restoration's
independent boundary, not to certify Sync. Conversation updates/settings,
messages and other writer families, sharing cleanup and owned UI remain pending.
No route enablement, commit, push or merge is part of this slice.

Independent read-only review found no implementation defect; its initially-global
coverage finding is addressed, and the corrected fixture semantics were reviewed.
The reviewer and all verification processes finished. This is restore-specific
certification, not full-project execution or completion of Stage 2.

## Settings Endpoint Parent Admission

After publishing draft PR #3020 at checkpoint `1408c78dba`, continued with the
primary settings endpoint. It already has a single write transaction and checks
requested ownership/scope again after locking its resume state. The change adds
the existing workspace fence before that child lock; global requests keep their
current path. This avoids adding a late parent lock to the shared settings store,
whose other callers have different upstream lock ordering.

Two SQLite/PostgreSQL RED cases demonstrated that settings could be updated on
an active Sync projection whose parent was deleted, `/tmp/workspace-chat-settings-red.log`.
The fix passes 14 focused cases, zero skips, `/tmp/workspace-chat-settings-green.log`:
deleted-parent rejection without graph/settings side effects, active/global
compatibility, deletion-first and settings-first transactions, commit and rollback.
The settings/error-mapping/selected HTTP regression passes 175 cases, zero
failures/errors/skips, six warnings, exit 0 in 573.36 seconds:
`/tmp/workspace-chat-settings-regression.log` and `.xml`.

Review found no implementation defect and suggested a stronger lock-order test.
Added a PostgreSQL test that pauses deletion after its parent claim but before
any child lock, starts settings admission, then resumes deletion. It requires
deletion success and settings HTTP 409, without a deadlock or settings row.
That additional case passes (`/tmp/workspace-chat-settings-lock-order.log`), and
the reviewer confirmed the gap is closed. All verification processes and the
reviewer have finished.

Production Bandit has zero findings/errors, `/tmp/workspace-chat-settings-bandit.json`.
Touched tests pass Ruff. Four endpoint Ruff findings (two I001, F401, B904)
are also present in checkpoint HEAD; no new lint finding is introduced.

This does not fence the shared DB settings primitive or other settings callers,
nor make direct Sync upsert safe. The owned route remains disabled; durable
sharing cleanup and the other Stage 2 writer families remain outstanding.

## Metadata Endpoint Parent Admission

After rebasing PR #3020 onto dev `59bd5845038`, fenced the primary metadata
endpoint, not the shared DB update primitive. The endpoint now admits the
workspace parent before locking/rechecking the conversation's owner and scope.
Metadata mutation and response construction share that transaction. Global
non-Sync chats skip the parent check; the existing global Sync path is unchanged.
No late parent lock is added underneath message-edit callers that already hold
child locks.

Test-first evidence: two deleted-parent cases failed with DID NOT RAISE
HTTPException on SQLite/PostgreSQL (`/tmp/workspace-chat-metadata-red.log`).
The first extended race run passed 18 cases but exposed unmapped NotFoundError
when ownership changed after preflight. Added explicit 404 mapping and reran.

| Check | Result | Evidence |
| --- | --- | --- |
| Reviewed metadata fence/identity/concurrency matrix | 29 passed, no skips, 4 warnings | `/tmp/workspace-chat-metadata-reviewed.log`, `.xml` |
| Error mapping and FastAPI HTTP deletion conflict (fixture authentication) | 130 passed, no skips, 4 warnings | `/tmp/workspace-chat-metadata-api.log`, `.xml` |
| Production Bandit | 0 findings/errors | `/tmp/workspace-chat-metadata-bandit.json` |
| Three touched test files Ruff | Passed after import ordering fix | Scoped Ruff command |
| Diff whitespace | Passed | `git diff --check` |

Identity races use actual Sync projection writes after the endpoint preflight,
without advancing its version, to prove owner/scope reread rather than merely
optimistic-version rejection. Deletion races cover commit and rollback in both
orderings. A separate PostgreSQL case pauses deletion between parent and child
locks and requires deletion completion followed by metadata 409.

Independent review found no actionable code defect. Its version-race,
response-failure rollback, empty-payload and exact identity-error coverage
recommendations are addressed in the final 29-case matrix. An active global
Sync-service route test remains a coverage limit; that branch is unchanged and
this slice does not certify Sync. The wider writer model is still
unfinished: generic DB updates, messages, direct Sync, other settings writers,
runtime/root/inventory/membership/study/migration and sharing cleanup are not
certified by this slice. No owned route enablement or full UI/live acceptance is
claimed.

Verification correction: the Character_Chat_NEW `test_client` fixture overrides
`get_request_user` and `get_current_user`. The historical settings/metadata HTTP
cases using it test endpoint ownership and error dispatch, not actual credential
authentication. Earlier descriptions of these cases as authenticated were
incorrect. This does not invalidate their endpoint assertions or the separate
real-process authentication probes in the dev-integration report. The new
identity-loss HTTP cases reuse the isolated AuthNZ bootstrap fixture instead,
with only the product database overridden and an explicit no-key 401 check.

## Settings Identity-Loss Error Boundary

The primary settings endpoint's locked owner-filtered resume read can raise
unified `NotFoundError` if the chat is trashed or its owner changes after
preflight. That class is not covered by the endpoint's previous handler set.
Map it through `map_db_error_to_http` to 404; no locking, authorization or
publication behavior is loosened.

The RED run reproduced four failures (owner/trash on each backend), while the
four scope-change cases already returned 404:
`/tmp/workspace-settings-identity-red.log`, `.xml`. After the two-line handler
change, the combined real SQLite/PostgreSQL settings/metadata matrix passes
52 cases with no skips and four warnings:
`/tmp/workspace-settings-identity-green.log`, `.xml`.

The first combined unit/HTTP run passed 160 cases but used the auth-overriding
chat fixture. Review identified that evidence limitation. Removing the overrides
rejected its unregistered legacy test key with 401, confirming it was not a
real credential test. The corrected cases reuse the existing isolated AuthNZ
`single_user_client` fixture and override only the product DB; their focused
run passes two cases (seven warnings):
`/tmp/workspace-settings-identity-auth-bootstrap.log`, `.xml`.
The earlier failed credential run remains in `/tmp/workspace-settings-identity-auth.log`.

Identity changes are deterministic committed preflight interleavings, not
independent-thread owner races. Product graph/settings are unchanged after
rejection. HTTP authentication coverage is SQLite-only. Independent review
confirms the corrected harness keeps production authentication and restores its
fixture state. Production Bandit has zero findings/errors:
`/tmp/workspace-settings-identity-bandit.json`; touched tests pass Ruff.

Final combined regression with the corrected AuthNZ fixture: 160 passed, no
failures/errors/skips, eight warnings, exit 0 in 118.86 seconds:
`/tmp/workspace-settings-identity-api-final.log`, `.xml`. This includes two
credential-authenticated HTTP cases, not 160 credential tests. After tightening
the unauthenticated assertion to full conversation/settings rows (including
versions/timestamps), the final focused HTTP rerun passes both cases:
`/tmp/workspace-settings-identity-auth-final.log`, `.xml` (seven warnings).
The reviewer confirms the full-row assertion closes that coverage gap; no
additional defect was identified. All verification processes have finished.

## Message Edit And Pin Admission

The September 26 follow-up fences the primary `edit_message` endpoint. Workspace
parent admission precedes message, optional metadata and conversation locks.
The locked message must still belong to the preflight conversation; the locked
conversation must retain the requested owner/scope. Scope changes, including a
global chat moved into a workspace, are rejected without taking a late parent
lock. Already-deleted messages preserve their existing 409 response.

Message content, pinned metadata/settings, conversation metadata and response
construction share one transaction. Pin settings advance the conversation
version, so a combined content/pin update rereads that version before the
additional metadata bump. Message/history and settings fences retain their
existing once-per-write semantics. Parent metadata is not changed.

Independent review identified two additional gaps, both addressed: combined
content/pin version assertions, and response hydration that swallowed SQL errors.
The endpoint now uses strict attachment reads and propagating connection-owned
metadata reads. Legacy non-transactional metadata reads retain their nullable
fallback. Real failing response SQL statements must produce an error and roll
back every edit, rather than returning success after PostgreSQL rolls back an
aborted transaction. The second review found no remaining actionable issue.

Verification:
- Initial RED: 14 failed and 2 passed across both backends. Failures show missing
  parent/scope/message-identity admission and post-commit write/response failures.
  `/tmp/workspace-message-edit-red.log`.
- Review SQL-failure RED: four failures, all DID NOT RAISE, with actual bad SQL
  executed during response hydration after the writes.
  `/tmp/workspace-message-edit-sql-red-response.log`.
- Final real-backend message/settings/metadata boundary run: 105 passed, no
  skips, four warnings, exit 0 in 167.36 seconds.
  `/tmp/workspace-message-edit-reviewed.log`, `.xml`.
- Final affected message-store/settings/error-mapping and edit/pin HTTP run:
  192 passed, no skips, ten warnings, exit 0 in 158.93 seconds.
  `/tmp/workspace-message-edit-final-regression.log`, `.xml`. This includes two
  real API-key HTTP tests using isolated AuthNZ bootstrap and only a product DB
  override: missing-key 401 and deleted-parent 409 leave full product rows
  unchanged. Credential cases are SQLite-only; the other HTTP cases use the
  established fixture-auth client.
- Broader full behavior-snapshot HTTP suite: 147 passed, no skips, fifteen
  warnings, exit 0 in 1877.40 seconds. `/tmp/workspace-message-edit-http-final.log`,
  `.xml`. This invocation started before the last strict response-read change;
  it is broader regression evidence, not final-head hydration verification.
  The final 105/192 runs above cover that hardening. Counts overlap and should
  not be added as unique tests.
- Touched production and tests pass Ruff and Python compilation; diff whitespace
  passes. Bandit reports zero findings/errors in both production modules.
  `/tmp/workspace-message-edit-reviewed-bandit.json`.
- Independent review's two findings are closed. Review was read-only and did not
  execute tests itself.

The Backlog task remains In Progress; owned routes remain disabled. Send/completion,
direct Sync, generic message writers, other settings/runtime/root/inventory/
membership/study/migration writers and durable sharing cleanup are not certified
by this edit-only slice. No WebUI/CDP or full-project acceptance is claimed.
All verification processes and the reviewer have finished.

## Primary Workspace Message Send Admission

The next September 26 slice fences only workspace-scoped, non-Sync
`POST /api/v1/chats/{chat_id}/messages`. The endpoint checks transaction ownership
before preflight reads: SQLite transactions, ChaCha/native PostgreSQL managed
depth, and non-IDLE PostgreSQL driver state all reject as borrowed. This avoids
committing another caller's pending work, including raw `BEGIN`. HTTP's existing
middleware provides a fresh operation; direct endpoint tests now model that
ownership, while borrowed-transaction tests deliberately retain the caller's
checkout. PostgreSQL cannot distinguish an existing implicit read transaction
from raw BEGIN, so both reject at entry; the endpoint's own later preflight
reads are not classified as borrowed.

Publication takes the process limit lock before its database transaction, then
the workspace parent, an optional existing reply message, and the conversation.
Owner/scope and reply identity are checked under locks. The configured cap is
rechecked for every accepted role under the conversation lock, without awaiting
inside the critical section. Independent test locks model separate workers and
prove that the database, not just process serialization, protects the final slot.

`post_message_to_conversation(conn=...)` is an additive strict persistence mode.
It requires a current-operation transaction connection before inserting; native,
idle and foreign connections outside that contract reject. Its metadata reads
and nested updates are safe only under that verified ownership. It does not
reacquire the process lock, commit, or schedule enrichment. Callers must propagate
errors to their outer owner rather than catch and commit partial SQLite writes.
Legacy calls retain the string-ID return and best-effort behavior. Global Sync,
completion, and recipient-owned shared chat contracts are not changed.

Message/attachments, history advancement, conversation metadata and strict
response hydration share the transaction. The response model is constructed
before commit; this is not a guarantee of successful network delivery or
idempotent retry. The non-Sync path still does not implement Idempotency-Key.
Enrichment is triggered only after commit and process-lock release, and trigger
errors cannot report a committed message as an HTTP failure. **Moving the
trigger does not fence enrichment's own writes**: tagging/keyword coordination
and clustering remain a separate writer family. Tagging is not silently removed.
Owned routes remain disabled until these and the other outstanding boundaries
are certified.

Review reproduced and addressed native-backend ownership bypass (two failures)
and raw-BEGIN bypass (one failure). Initial domain/rollback/cap/reply contracts
produced 34 expected failures before implementation. Final boundary verification
passes 142 SQLite/PostgreSQL cases without skips (four warnings, 156.96 seconds):
`/tmp/workspace-send-final.log` and `.xml`. This includes an actual PostgreSQL
driver commit fault, explicit operation scopes, invalid connection rejection,
and rollback after metadata or response failure. Final real-authentication HTTP
verification passes four cases (147 deselected, eleven warnings, 19.14 seconds):
`/tmp/workspace-send-http-reviewed.log` and `.xml`. These credential cases use
SQLite and isolated AuthNZ bootstrap; they are not PostgreSQL/JWT or live CDP
acceptance.

Three legacy endpoint failures reproduce at the exact pre-change commit
`32d6c9072f8b32c51d9773847e56c26b6d764f02`. Two target a removed edit helper;
the third expects absent settings even though creation initializes them. The
tests now inject faults at the actual database update, retain exact HTTP error
assertions and verify unchanged stored content/version, and assert the canonical
initialized settings response. The focused correction passes three cases:
`/tmp/workspace-send-baseline.log` and `/tmp/workspace-send-legacy-fixed.log`.
Final affected regression passes 124 cases, with one pre-existing module-level
skip and 24 warnings, exit 0 in 1588.36 seconds:
`/tmp/workspace-send-regression-final.log` and `.xml`. It covers legacy HTTP
endpoints, message storage, limiter units, completion prechecks, stream lookup,
enrichment, and SQLite/PostgreSQL operation/transaction lifecycle. The dedicated
legacy rate-limit module requires `TEST_MODE=0` and is intentionally skipped by
the established `TEST_MODE=1` fixture; this slice's configured cap and independent
worker races were exercised in the separate real-backend boundary suite. Counts
overlap across runs and are not a unique-test total. Background teardown was
slow, but the process finished normally; no checks were cancelled or disabled.

Independent production and test review has no remaining findings. Ruff and
Python compilation pass for all seven touched Python files; scoped production
Bandit reports zero findings/errors (`/tmp/workspace-send-final-bandit.json`).
No full workstream completion or WebUI/CDP acceptance is claimed by this
checkpoint.
All verification processes and the reviewer have finished. TASK-12020.50
remains In Progress and the owned route remains disabled.
