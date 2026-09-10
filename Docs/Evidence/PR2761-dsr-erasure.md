# PR2761 SQLite DSR erasure verification

Tracking: TASK-13013.8. This bounded correction supports acceptance criteria 2
(deletion and partial-failure recovery) and 3 (safe logs); it does not certify
whole-account erasure or complete either broad criterion.

## Stage 1: Reproduce against migrated storage

**Goal:** Exercise the existing notes erasure handler against the real current
ChaChaNotes schema for two isolated users. Verify dependent projection cleanup,
attachment foreign-key restrictions, rollback/retry, and failure log privacy.
**Success criteria:** Regression tests expose missing foreign-key enforcement
and raw exception content in logs. A failed delete leaves preceding graph-edge
deletions rolled back.
**Tests:** `tldw_Server_API/tests/Admin/test_dsr_sqlite_erasure_integration.py`.
**Status:** Complete. Three regressions failed on the original implementation;
the existing transaction rollback/retry behavior passed.

## Stage 2: Minimal correction

**Goal:** Apply the existing DB connection policy to the DSR SQLite connection
so schema cascades and restrictions are enforced. Keep database exception
content out of shared DSR failure logs.
**Success criteria:** The same tests pass. Attachments which prevent parent
deletion cause a reported failure with data intact; no false erasure success.
**Status:** Complete. Reused `configure_sqlite_connection` with WAL and
synchronous changes disabled. No new SQL or schema changes were introduced.

## Stage 3: Verify and document limits

**Goal:** Run new and existing DSR tests, lint and scoped Bandit; retain exact
evidence and unresolved erasure coverage.
**Success criteria:** No new findings and explicit partial certification.
**Status:** Complete. The focused four-file DSR suite passed 26 tests, with six
existing warnings. Ruff, Black on the new test and changed service ranges, and
scoped Bandit pass (zero findings; B101 excluded only for test assertions).

## Reproduction and verification

The tests initialize the actual current schema using `CharactersRAGDB`, seed
two separately routed user databases with notes and wikilink projections, and
exercise the real `_erase_notes` handler. The attachment test creates the real
registry row using `note_attachment_store`; a schema restriction must preserve
its parent and roll back earlier edge deletion. The transient failure test uses
a SQLite abort trigger and proves a later retry succeeds. The log test uses a
real database error containing a synthetic private marker and verifies that the
marker is absent from captured logs and the returned result.

Initial test setup required explicitly setting `USER_DB_BASE_DIR` to the
pytest temporary directory: macOS's default approved temp root does not include
the selected `/tmp` basetemp. This fixture uses the supported path configuration;
no path-validation or repository test guard is disabled.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Admin/test_dsr_sqlite_erasure_integration.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_dsr_embeddings_erasure.py \
  tldw_Server_API/tests/Admin/test_admin_data_ops_dsr_sanitizers.py \
  -q --basetemp=/tmp/pr2761-dsr-green
```

Red evidence: `/tmp/pr2761-dsr-red3.log` (three failed, one passed).
Green evidence: `/tmp/pr2761-dsr-green.log` (26 passed).
Security evidence: `/tmp/bandit_pr2761_dsr_service.json` and
`/tmp/bandit_pr2761_dsr_tests.json`, each with zero findings/errors.
The source revision is the release worktree base `decdf9db77` plus the service
and test diff associated with this evidence; the final integrated commit will
be recorded by the owning release task.

## Stage 4: Prove external blob cleanup and restart recovery

**Goal:** Verify the existing Sync retention primitive actually removes file
bytes and recovers from interruption while preserving another owner's objects.
**Success criteria:** Test normal GC, interruption before unlink, and interruption
after unlink but before metadata finalization. Close the original SQLite pool,
create a fresh backend/service, and complete the persisted `deleting` operation.
**Status:** Complete. The full retention and blob-store suites pass 68 tests,
with four existing warnings; log `/tmp/pr2761-retention-final2.log`.

The former metadata-only assertion now also verifies the target file disappears.
Each case seeds identical content under another owner's independently allocated
storage namespace and verifies both its bytes and metadata remain unchanged.
Failure injection surrounds the real `delete_namespace_blob` operation; both
the filesystem and SQLite registry are real. Existing device acknowledgement,
tombstone and restore-window eligibility guards remain active. No retention
production code changed. A fresh backend is constructed explicitly because the
factory caches backends and otherwise returns the deliberately closed pool.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  -q --basetemp="$TMPDIR/pr2761-retention-final"
```

Ruff and Black on changed ranges pass. Bandit on the retention test reports one
existing B106 test signing-secret finding, also reproduced from the untouched
HEAD file; zero new findings. Reports:
`/tmp/bandit_pr2761_retention_tests.json` and
`/tmp/bandit_pr2761_retention_baseline.json`. B101 is excluded for test assertions.
The initial broader run's four unchanged notes/task fixtures required using
macOS's approved `$TMPDIR` rather than `/tmp`; the final normal suite uses that
root and disables no guards.

## Remaining limits

The DSR correction cannot itself erase externally stored attachment objects. Their
restricting registry rows must be cleaned up by an appropriate lifecycle
operation before notes erasure can succeed. It deliberately honors the schema
restriction instead of discarding registry references and orphaning objects.
The separate Sync evidence proves eligible physical blob garbage collection and
recoverability, but does not add whole-account or canonical registry hard purge.
Backups, deleted SQLite pages/WAL, sync history, shared PostgreSQL storage,
jobs, caches, and complete account erasure still require separate evidence.
The test that captures orchestration logs uses an inert mocked status repository;
its database erasure handler and failing SQLite trigger are real.
