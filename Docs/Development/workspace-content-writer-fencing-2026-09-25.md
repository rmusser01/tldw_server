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
| Error mapping and authenticated HTTP deletion conflict | 130 passed, no skips, 4 warnings | `/tmp/workspace-chat-metadata-api.log`, `.xml` |
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
