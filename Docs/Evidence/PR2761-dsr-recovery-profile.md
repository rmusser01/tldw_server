# PR2761 DSR storage checks and controlled recovery

Tracking: TASK-13013.8. Base source:
`08946442af85f58f8078c2e3df1e4dc42bb40876`. This report covers the bounded
DSR correction and an existing-API recovery procedure. It does not complete
the broader tenant-isolation and lifecycle task.

## Stage 1: Reproduce the two boundaries

**Goal:** Check actual normalized storage and persisted interrupted execution.
**Success criteria:** Reproduce false erasure success and determine whether a
fresh repository connection can retry an interrupted request.
**Tests:** `test_dsr_release_recovery_regressions.py`.
**Status:** Complete. Two initial probes failed as intended: storage remained
under the manager's normalized `~` path while erasure returned zero; a real
notes deletion committed before cancellation, and the persisted original DSR
still returned HTTP 409 `request_already_executing` after reopening AuthNZ.
Log: `/tmp/pr2761-dsr-release-probes.log` (2 failed, 6 warnings).

## Stage 2: Minimal repair and existing recovery contract

**Goal:** Reject unknown embedding storage and prove controlled replacement
recovery without changing execution concurrency or retention policies.
**Success criteria:** Zero is accepted only for confirmed absence; replacement
intake and execution preserve durable linkage and complete remaining notes work.
**Status:** Complete. Expanded controls reproduced three storage failures and
four passing controls before the source repair. Log:
`/tmp/pr2761-dsr-release-controls-red.log` (3 failed, 4 passed, 6 warnings).

`_erase_embeddings` now uses the same non-creating `DatabasePaths` resolver and
`stat` policy as preview when its Chroma manager is unavailable. Existing
storage, unresolved paths and failed inspection raise fixed errors. Confirmed
absence returns zero without creating directories. This corrects supported
path normalization; it adds no collection-deletion or retention behavior.

The execution protocol is unchanged. An interrupted request can retain
`executing`; there is no automatic stale-execution reclaim. The recovery tests
exercise the existing create/execute endpoint functions with real migrated
SQLite AuthNZ repositories, canonical seeded users, actual notes deletion and
fresh database pools. They inject cancellation either before deletion or
after the thread-backed deletion has completed, then prove that a new request
can finish the selected notes work or idempotently delete zero remaining notes.

Replacement intake uses the authoritative preview, a new `client_request_id`,
the same subject and categories, and the existing `notes` field to reference
the original ID and operator's completion check. Repeating the replacement key
returns the same replacement ID. After another pool restart, its original-ID
link, operator ID, subject, categories and completed status remain present.
The original stays `executing`, and its execute endpoint still returns 409.
The test supplies a platform-admin principal directly and substitutes only
repository dependency construction and the independent audit-delivery call;
it does not claim HTTP authentication or unified-audit delivery coverage.

## Operator recovery using the existing API

1. Preserve the original request ID, resolved subject, selected categories and
   current status. Do not interpret `executing` as proof that work is stale.
2. Stop the process/worker that owned the interrupted request and conclusively
   confirm its work has ended before starting a replacement. Wait for any
   thread-backed operation to finish, or confirm the owning process exited.
   A client disconnect, elapsed timeout or canceled coroutine alone is
   insufficient: database deletion can continue in its worker thread. Keep
   subject writes quiescent through recovery so new data is not swept into it.
3. As an authorized administrator, call
   `POST /api/v1/admin/data-subject-requests/preview` for the exact original
   subject and categories. Resolve unavailable coverage or storage failures;
   do not replace them with fabricated zero counts.
4. Call `POST /api/v1/admin/data-subject-requests` with `request_type: erasure`,
   the same subject/categories and a new unique `client_request_id`. In its
   `notes`, record the original DSR ID, why recovery is needed, the operator,
   and how/when original execution was confirmed ended. The returned record's
   own ID and original-ID reference establish the initial old/new linkage.
   Before executing, preserve both IDs, subject/categories, operator and the
   evidence of confirmed quiescence in an external recovery receipt. This
   receipt is required because failed execution can overwrite API notes.
   Do not include private subject content or credentials in these notes.
5. Call `POST /api/v1/admin/data-subject-requests/{replacement_id}/execute`.
   Check each category and final status. Inspect the replacement record through
   the request listing and retain its outcome. A successful replacement does
   not relabel the original stale `executing` record; retain it as interrupted
   history. Same-request execution remains blocked with 409.

This is a controlled operator procedure, not an enforced stop-check or an
automatic retry mechanism. The system does not prevent an administrator from
creating a replacement while the original still runs. Operators must satisfy
step 2. Failed replacement execution can overwrite its `notes` with the error
summary; preserve the old/new linkage externally before execution, and include
it again in any subsequent replacement's notes. No schema or audit mechanism
was added for permanent bidirectional recovery relations.

The demonstrated recovery profile is selected **notes** erasure against SQLite
without attachment restrictions, including an already-deleted notes set. A
failed restricted attachment deletion must keep its data intact and needs the
appropriate existing attachment lifecycle operation; this procedure does not
authorize bypassing RESTRICT, shortening retention or purging registry rows.
Other categories retain their current contracts and need their own recovery
evidence before extending this profile. Existing related evidence is in
[DSR erasure](PR2761-dsr-erasure.md) and
[selected preview coverage](PR2761-dsr-preview-coverage.md).

## Stage 3: Verification and scope

**Goal:** Verify the focused DSR suites and scoped security/style checks.
**Success criteria:** All selected tests pass with no new security findings.
**Status:** Complete. The seven-file focused selection passed **61 tests,
0 failures, 0 skips, 2 warnings** in 57.72 seconds. Logs:
`/tmp/pr2761-dsr-release-green.log` and
`/tmp/pr2761-dsr-release-green.xml`. The final formatted new-test file was
rechecked separately: **7 passed, 6 warnings** in 5.18 seconds, log
`/tmp/pr2761-dsr-release-formatted.log`. The normal pytest configuration
suppresses the detailed warning summary; no warning-free result is claimed.

Run from the release worktree with the root virtual environment:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
PYTHONPATH=. python -m pytest \
  tldw_Server_API/tests/Admin/test_dsr_release_recovery_regressions.py \
  tldw_Server_API/tests/Admin/test_dsr_preview_coverage.py \
  tldw_Server_API/tests/Admin/test_dsr_embeddings_erasure.py \
  tldw_Server_API/tests/Admin/test_dsr_sqlite_erasure_integration.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_data_subject_requests_api.py \
  tldw_Server_API/tests/Admin/test_admin_data_ops_dsr_sanitizers.py \
  -q --basetemp="$TMPDIR/pr2761-dsr-release-green" \
  --junitxml=/tmp/pr2761-dsr-release-green.xml
```

Ruff passes for the service and new test file. Black passes for the full new
test file and changed service lines 511–533. Bandit reports **0 findings,
0 errors** on the service and new tests; only test assertions are excluded
with `-s B101` for the test report. Reports:
`/tmp/bandit_pr2761_dsr_release_service.json` and
`/tmp/bandit_pr2761_dsr_release_tests.json`.

Whole-account deletion, backup erasure, physical SQLite/WAL sanitization,
live Chroma physical deletion, PostgreSQL recovery and general worker-crash
recovery are outside this evidence. No automatic lease/reclaim feature,
schema change, execution-status change or retention-policy change is included.

Source hashes (SHA-256):

- `tldw_Server_API/app/services/admin_data_subject_requests_service.py`:
  `710217d841ac81d897f8c6c8e43ce8629a0f36f4c8c804126aa5b17c9da2e3aa`
- `tldw_Server_API/tests/Admin/test_dsr_release_recovery_regressions.py`:
  `23a0a902cd9ab0fe5e76c9e474ab1fb9dad423bf245130fcc659d1ffe8ade0da`
