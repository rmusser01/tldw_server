# Reading-safe output file mutations implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user has selected inline execution; do not ask again or delegate implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make activated generic output writers and readers preserve Reading-owned bytes through concurrency, crashes, delayed history delivery, and bounded large-file production.

**Architecture:** Reuse the Collections revision fence and verified descriptor-relative POSIX storage lock. Add one bounded generic operation journal, a persisted per-user activation binding, and narrowly scoped services for file operations and opened-descriptor responses. Filesystem completion releases reservations independently of history delivery; no public jobs API or generic locking/outbox framework.

**Tech Stack:** Python, existing FastAPI/Starlette, SQLite and PostgreSQL adapters, stdlib `os`/`stat`/`uuid`, existing worker lifecycle and pytest fixtures. No new dependency.

**ADR required:** yes — existing ADR-003 applies and is amended.
**ADR path:** `backlog/decisions/003-reading-atomic-hard-delete.md`.
**Reason:** Implements the approved storage, cross-writer and external-delivery contract.
**Task:** TASK-13153, In Progress. These are checkpoints within that task, not newly allocated Backlog IDs.
**Approved spec:** `Docs/superpowers/specs/2026-09-05-reading-output-file-reservations-design.md`, user approved after checkpoint `8dc255fcca`.
**Parent plan:** `Docs/superpowers/plans/2026-09-04-reading-atomic-hard-delete.md`; this plan replaces its generic-file-writer gap, not its remaining Reading DTO/HTTP/reconciliation/release tasks.
**Status:** Inline execution in progress. Task 1, Task 2a, Task 2b and Task 3a staging/write checkpoints verified on SQLite/PostgreSQL. Publication/recovery and later tasks remain pending. See checkpoint evidence below.

---

## Execution boundaries and verification convention

Work in `/private/tmp/task-13153-reading-hard-delete`, branch `codex/task-13153-reading-hard-delete`. Preserve unrelated work. Last production checkpoint is `2207a84fc1`; its 146 passes are historical, not evidence for this plan. Read `AGENTS.md`, the task, ADR, approved spec and `backlog/docs/lessons-testing-evidence.md` before execution.

Follow @superpowers:test-driven-development for every behavior change and @superpowers:verification-before-completion for checkpoint claims. Use @superpowers:systematic-debugging when a test fails unexpectedly. Use @ponytail to reuse existing helpers rather than invent infrastructure. Each numbered task ends with review and an exact-file commit; do not leave multiple unverified tasks accumulated. A checklist test-writing step is a small test group; implement and run one behavior at a time when a group exceeds a short work interval.

For every command involving Python, activate the existing Server environment first:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
export TLDW_TEST_NO_DOCKER=1
```

New Collections tests import `db` from `tldw_Server_API.tests.Collections.test_reading_revision_mutations`; it parametrizes real SQLite/PostgreSQL. Import the existing endpoint `client` only for HTTP tests. Use a narrowly scoped new fixture for a provisioned temporary root/activated binding; do not change the existing fixture to activate every legacy test implicitly. Database-only tests must not require a mounted filesystem.

Run each task's listed pytest command first with `-k sqlite` while establishing DB failures, then without that filter for its final SQLite/PostgreSQL check. Filesystem-only tests without backend parametrization run unfiltered. For PostgreSQL, use the already-running local instance, escalation if required, and `TLDW_TEST_NO_DOCKER=1`; unavailable/skipped PostgreSQL is a verification blocker, never permission to replace a container. Expected RED is the newly asserted missing contract, not an import/environment error; expected GREEN is all selected tests passing with no required-backend skips. Never run a full sweep without user permission.

Before every implementation commit, run Ruff, Black `--check`, scoped Bandit, the listed focused regression tests, and `git diff --check`. Substitute only that task's explicit changed Python paths into these commands:

```bash
python -m ruff check <changed-python-paths>
python -m black --check <changed-python-paths>
python -m bandit <changed-production-python-paths> -f json -o /private/tmp/task-13153-output-bandit.json
git diff --check
```

Record actual counts and commands with `backlog task edit 13153 --plan ...` and the checkpoint record below. Do not convert unchecked full-task acceptance criteria to Done. Existing unrelated lint debt must be identified against HEAD rather than silently fixed or called clean. Stage explicit paths with `git add -- path...`; commit messages below include TASK-13153. Never bypass hooks. No push/PR/merge is authorized by this plan.

## File ownership map

Paths below are repository-relative, rooted at `/private/tmp/task-13153-reading-hard-delete`.

| File | Responsibility |
| --- | --- |
| `tldw_Server_API/app/core/DB_Management/Collections_DB.py` | Bootstrap binding/journal schema; authoritative token transitions, collision checks, row guards, immutable output incarnation, quota in existing transaction |
| `tldw_Server_API/app/services/reading_artifact_cleanup_service.py` | Reuse/expose existing verified directory descriptor context; preserve Reading storage behavior |
| **New** `tldw_Server_API/app/services/output_file_operations.py` | Bounded prepared/committed/aborting file lifecycle, admission, chunking, recovery and delivery orchestration |
| **New** `tldw_Server_API/app/services/output_file_response.py` | Protected lookup/open and bounded descriptor response, existing HTTP semantics |
| `tldw_Server_API/app/services/outputs_service.py` | Shared activated producer/delete integration; retain genuinely inactive behavior |
| `tldw_Server_API/app/api/v1/endpoints/outputs.py` | Dispatch PATCH/GET/download/HEAD/create/delete/purge through shared boundaries; sanitized errors |
| `tldw_Server_API/app/api/v1/endpoints/watchlists.py` | Output/briefing generation, failed-generation cleanup, protected downloads and output-content loaders |
| `tldw_Server_API/app/services/audiobook_jobs_worker.py`, `tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py` | Audio/subtitle producer adaptation and atomic output quota |
| `tldw_Server_API/app/core/Collections/reading_service.py` | Reject unactivated production ownership/staging; integrate existing reserve/adopt boundary, not a fake generic Reading parent |
| `tldw_Server_API/app/services/outputs_purge_scheduler.py`, `tldw_Server_API/app/services/startup_cleanup_workers.py` | Retention delegates safe deletion; recovery lifecycle independent of retention flag |
| `tldw_Server_API/app/core/DB_Management/media_db/runtime/tts_history_ops.py` | Internal incarnation-qualified, idempotent artifact-disposal effect |
| Media migration files enumerated in Task 5 | Minimal receiver identity/tombstone storage; no public history schema change |
| **New** `tldw_Server_API/scripts/output_storage_protocol.py` | Offline dry-run/explicit activation and blocked-operation inspection; never automatic destructive repair |

Do not split the entire large Collections adapter. Keep its transaction-bearing methods there and filesystem orchestration out of it. The two new services have different owners: file lifecycle versus HTTP response lifetime.

## Task 1: Inert schema and journal state machine

**Files:** Modify `tldw_Server_API/app/core/DB_Management/Collections_DB.py`; create `tldw_Server_API/tests/Collections/test_output_file_operations_db.py`.

- [x] Add failing repeated-bootstrap tests for `output_storage_bindings` and `output_file_operations`, with explicit NOT NULL identities, user scope, no output cascading FK, valid kinds/phases, nonnegative counts/budgets, and `fs_done` only on committed/aborting rows. Schema installation must not create/provision paths or activate a user.
- [x] Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py -q -k sqlite`; confirm new behavior fails.
- [x] Extend `_ensure_reading_revision_schema` using its existing transaction/advisory bootstrap lock and explicit connection. Bindings contain user, namespace, protocol version and validated finite policy. Journal stores the spec's fixed source/stage/destination fields and collision keys, snapshot/intended changes, source/stage/publication identity, lease/retry/error, bounded history payload/status, and byte reservation. Use compact bounded JSON only for finite structured fields, never bodies/credentials. Add due-work and active-user lookup indexes; no general lock table.
- [x] Add internal immutable `outputs.file_incarnation` (random UUID on insert, retained on rename, never copied on new output/ID reuse). Existing rows receive new tokens in an idempotent stopped-writer backfill; never reset a nonempty token. This is not a public patchable metadata key. `create_output_artifact` remains the ordinary allocator; the existing trusted Reading adoption insert also allocates its own fresh token.
- [x] Implement transaction-owned methods `prepare_output_file_operation`, `get_output_file_operation`, `validate_output_file_operation`, `commit_output_file_operation`, `abort_output_file_operation`, `finish_output_file_operation`, and `ack_output_file_effect`. Internal optional `connection` propagates the same DB transaction; public HTTP payloads never accept operation tokens. Fresh lease time is read after the fence, not supplied pre-wait.
- [x] Test conditional transitions, unknown token, other user, expiry, unchanged source snapshot, and rollback. Core transition predicates are:

```sql
-- All statements additionally run under the existing revision fence.
UPDATE output_file_operations SET phase = 'aborting'
WHERE token = ? AND user_id = ? AND phase = 'prepared' AND fs_done = 0;
UPDATE output_file_operations SET fs_done = 1, reserved_bytes = 0
WHERE token = ? AND user_id = ? AND phase IN ('committed', 'aborting') AND fs_done = 0;
DELETE FROM output_file_operations
WHERE token = ? AND user_id = ? AND fs_done = 1 AND effects_pending = 0;
```

- [x] Run the new module unfiltered and `python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -q`. Check migration twice on both backends, including reconstructed adapter instances. Executed as separate complementary backend selections below. Review/commit: `feat(reading): persist inert output operation journal (TASK-13153)`.

## Task 2: Enforce shared row/path claims and resource admission

**Files:** Modify `tldw_Server_API/app/core/DB_Management/Collections_DB.py`; extend `tldw_Server_API/tests/Collections/test_output_file_operations_db.py` and `tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py`.

Task 2b keeps its cross-writer matrix in new
`tldw_Server_API/tests/Collections/test_output_file_claims_db.py` to avoid further
expanding the journal transition/admission module. Existing adoption/deletion
modules remain focused regressions, not rewritten fixtures.

Execution split: Task 2a implements and verifies capacity admission plus explicit
same-transaction audiobook accounting first. Task 2b then covers shared row/path
guards across all writers. This keeps each checkpoint independently reviewable;
neither checkpoint enables runtime file operations or the public capability.
Task 2a reuses the persisted policy and existing revision fence, counts every
unfinished filesystem phase, and excludes delivery-only records. Validate policy
before admission and prove exact bounds, user isolation, rollback and concurrent
admission on both databases. Zero reserved bytes remains a zero-byte ceiling,
never unlimited storage; byte-producing orchestration and physical free-space
checks are still Task 3. Accounting connection propagation is internal and must
not change existing standalone calls or claim that legacy producers are migrated.

- [ ] Add RED races for all three path columns and the reserved output identity against generic create/update/delete/retention, Reading registration/reserve/adopt/reconciliation and guarded item delete, in both commit orders. Include metadata-only mutations, absolute/case aliases, managed-source alias, shared-unowned source, namespaces and foreign user. No writer may bypass through a direct adapter method.
- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py -q -k sqlite`.
- [ ] Extend `_assert_output_path_not_reserved` and shared row checks at the existing transaction boundaries. Query generic claims only where `fs_done = 0`; check all source/stage/destination collision keys and exact recorded identity. A private token exception validates kind, row snapshot and the recorded transition, not merely equality to any journal token. Reading journal checks remain mandatory. Row guards also cover ID allocation/reuse before `fs_done`.
- [ ] Validate finite positive per-operation/per-user byte and active-operation caps, text input/output caps and free-space margin before activation/admission. Under the fence sum active reserved bytes and count active operations; include blocked/pending file work, exclude delivery-only rows. Reject before a producer/file exists with `output_storage_capacity` or policy-unavailable. Different users do not spend one another's budget; physical free-space checks are advisory and happen in Task 3 outside DB transactions.
- [ ] Prove two concurrent reservations cannot overspend; `fs_done` releases row/path/count/bytes together despite pending history, and old completion/ack never touches a recycled output. Keep same-store audiobook delta in the output commit transaction with explicit connection; no best-effort additive replay.
- [ ] Run new DB module and `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q`. Review/commit: `fix(reading): fence output writers with durable claims (TASK-13153)`.

## Task 3: Bounded file protocol and restart recovery

**Files:** Create `tldw_Server_API/app/services/output_file_operations.py`, `tldw_Server_API/tests/Collections/test_output_file_operations_storage.py`; modify `tldw_Server_API/app/services/reading_artifact_cleanup_service.py` only to expose/reuse its verified directory descriptor context.

Execution split: Task 3a establishes immutable source/stage evidence and guarded
write-offset updates in `Collections_DB.py`, plus exclusive staging and bounded
offloaded writes. Each call uses at most one 1 MiB chunk and closes/syncs before
unlocking (stricter than the maximum 8 MiB interval). Real-file failures and
SQLite/PostgreSQL adapter checks precede its separate review/commit. Task 3b then
adds publication, recorded DB mutation and phase-specific recovery, including
uncertain commit acknowledgements and the full crash matrix below. No runtime
caller or activation is enabled by Task 3a; incomplete/ambiguous files retain
their journal claims, with automatic cleanup still pending Task 3b.

- [ ] Write RED tests against real temporary files for copy-before-commit, occupied destination, absent source, symlink/special/unexplained hardlink, changed source fingerprint, and same DB/different root. Start with service-level replace; no routes yet. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py -q`.
- [ ] Implement `prepare`, `write_chunk`, `publish_and_commit`, and `recover_due` as a small service using Task 1 methods. Lock order is verified storage context then DB fence. Capture original fingerprint before first read. Reserve before exclusive stage creation; persist stage identity before leaving the first lock interval. No filesystem/network call occurs inside DB mutation transactions. Never reopen the root path after verifying its descriptor.
- [ ] Chunk with buffers at most 1 MiB and lock intervals at most 8 MiB or 50 ms measured between chunks. Close writable FDs before releasing exclusion; no background write survives cancellation/unlock. Reacquire/revalidate token, lease, source identity, stage identity and exact written offset before continuing. Persist expected offset after each completed interval. A crash between file-write and DB offset acknowledgement leaves ambiguous work blocked, not blindly truncated/resumed. Run blocking I/O off the event loop and bound nonstream converters before loading input.
- [ ] Check actual free space before admission and periodically, count hardlink allocation once, and enforce declared limits before each write. ENOSPC safely aborts; source survives. Reject overrun with `output_size_limit`. Reader-held unlinked files remain physical allocations despite logical budget release.
- [ ] Publish with `os.link(stage, destination, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)` and directory fsync, retaining the private witness. Existing destination always conflicts. Commit metadata + committed phase + quota on one connection. On an uncertain commit acknowledgement reread phase on a fresh connection; committed wins, unreachable DB preserves every file/claim and returns `output_update_unconfirmed`.
- [ ] Implement phase-specific cleanup: abort removes verified destination then fsyncs before witness removal; commit preserves destination, removes witness and proven unreferenced original source. Recheck exact source identity/fingerprint before unlink. Persist `fs_done` only after required fsync/preservation; no-effects aborts retire immediately. Unproved pre-identity stages and changed files stay blocked for operator inspection.
- [ ] Add crash injection at reservation/create/identity record/write/fsync/link/commit/unlink/fs_done/retirement and lost commit acknowledgements. Add process-kill lock release, late producer after expiry/retirement, two-process contention, bounded-memory large copy, cooperative reader/writer progress and cancellation during offloaded writes. Do not signal a multiprocessing Event after terminating its waiter.
- [ ] Run new module plus `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q`. Review/commit: `feat(reading): recover bounded reserved output writes (TASK-13153)`.

## Task 4: Protected lookup and opened-descriptor HTTP responses

**Files:** Create `tldw_Server_API/app/services/output_file_response.py`, `tldw_Server_API/tests/Collections/test_output_file_responses.py`; modify `tldw_Server_API/app/api/v1/endpoints/outputs.py` and `tldw_Server_API/app/api/v1/endpoints/watchlists.py`; extend `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`.

- [ ] Characterize current download-by-ID, by-name and HEAD success/missing/auth/Content-Disposition/range/conditional behavior with real endpoint tests before replacing FileResponse. Add race tests paused before protected lookup and after descriptor open. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_responses.py -q` and confirm new race assertions fail.
- [ ] Implement protected current-row lookup under the verified generic storage lock. Compare structural ownership namespace to the locked volume; missing/ambiguous/mismatched owned namespace fails closed before open. No filename fallback, no cross-volume search, no path normalization writes. Unowned rows use reconciled generic provenance. Open relative to that same directory with nofollow and verify `fstat`; only journal-proven committed witness links permit expected extra links.
- [ ] Route Watchlist `download_output` and every output-content-loading helper through that same protected lookup/open primitive. Audit `read_text`, `read_bytes`, `open` and `FileResponse` call sites in `watchlists.py`: classify templates/config reads separately, but actual output bytes must use the protected descriptor. Bounded text helpers may read/close the descriptor instead of returning an HTTP response; do not duplicate pathname resolution. Add real Watchlist dispatch and content-loader regressions paused before lookup/open, then delete/replace/reuse the path, plus wrong-volume protection.
- [ ] Return a small descriptor-owning response using existing Starlette response primitives; do not pass a pathname to FileResponse. Reuse installed HTTP/range helpers where possible after inspecting their source locally. Length, ETag/Last-Modified and range offsets derive from the opened file. Close on HEAD, 304, 416, normal completion, exception and disconnect; streaming is bounded and offloads blocking reads. A descriptor opened before deletion may finish serving that old inode.
- [ ] Prove path reuse cannot serve another file; two same-user volumes with identical names/different bytes reject wrong-volume reads; committed witness works; pending publication does not. Existing inactive-route behavior remains characterized. Mutation worker health must not deny safe reads when namespace verification succeeds.
- [ ] Run new response module plus `python -m pytest tldw_Server_API/tests/Collections/test_items_and_outputs_api.py tldw_Server_API/tests/Collections/test_reading_output_updates.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py -q`. Review/commit: `fix(outputs): bind downloads to protected file descriptors (TASK-13153)`.

## Task 5: Original-instance history delivery without file claims

**Files:** Modify `tldw_Server_API/app/core/DB_Management/media_db/runtime/tts_history_ops.py`, `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`, `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`, `tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_tts_source_hash_structures.py`; create `tldw_Server_API/app/core/DB_Management/migrations/027_output_history_incarnation.sql` and `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/postgres_output_history_incarnation.py`. Extend `tldw_Server_API/app/services/output_file_operations.py`, `tldw_Server_API/tests/DB_Management/test_media_db_tts_history_ops.py`; create `tldw_Server_API/tests/Collections/test_output_file_history_delivery.py`. Modify history-link callers `tldw_Server_API/app/core/TTS/tts_jobs_worker.py` and `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py`.

- [ ] Add RED tests for post-fs_done Media outage, acknowledgement lost after successful update, newer file/path/ID reuse, and history insertion arriving after deletion delivery. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_history_delivery.py -q`.
- [ ] Implement the approved original-output-incarnation option, not replay by numeric ID. Media v27 adds nullable internal `tts_history.output_incarnation` and one narrow tombstone relation keyed by `(user_id, output_incarnation)`, containing the stable disposal operation token and timestamp. Register both migrations and bump the current version from 26 to 27 (if the branch has changed, resolve the next unoccupied version before editing). Fresh and upgraded schemas must match; no silent migration failure. No public DTO or general-purpose delivery table.
- [ ] Receiver relation uses explicit `live`/`disposed` state. Disposal transaction inserts a disposed row if absent, or atomically transitions an existing live row to disposed, recording the first disposal token/timestamp; already-disposed replays preserve that evidence. Then clear links only for the same user's matching incarnation. Do not use conflict-do-nothing as the entire disposal transition. History creation inserts/locks that same keyed receiver row before checking disposal state and holds that lock through its history insert; disposed instances always produce cleared links. Never resurrect disposed. This narrow monotonic receiver state is necessary because a SELECT-then-INSERT test alone races on PostgreSQL. Test history creation → disposal → delayed history insertion, and both concurrent commit orders, including an existing live receiver row.
- [ ] TTS callers capture the internal incarnation when the output is produced, not by fetching a potentially recycled ID later. Pass it only internally to history creation. Non-output file-ID history remains unchanged. If an old activated association lacks proven incarnation, leave a sanitized blocked-delivery effect; do not guess. Offline activation reconciles existing known output/history links with writers stopped and leaves ambiguity explicit.
- [ ] At logical output commit record a bounded `dispose_history` effect with token, incarnation and fixed timestamp. File recovery marks fs_done/releases claims without Media access. Delivery pass processes fs_done effects outside the OS lock; stable receiver identity makes newer output updates safe. Acknowledge conditionally only after receiver commit; retire only after all bounded effects acknowledge. Separate backoff/status from filesystem health. Preserve tombstones while delayed history writers could still reference an old incarnation; no age-only retirement.
- [ ] Run new delivery tests and `python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_tts_history_ops.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_history_artifact_purge.py -q`. Test migration twice and disposal/late-insert commit orders on real SQLite/PostgreSQL. Review/commit: `fix(outputs): replay history effects for original output instances (TASK-13153)`.

## Task 6: PATCH, explicit deletion and retention integration

**Files:** Modify `tldw_Server_API/app/api/v1/endpoints/outputs.py`, `tldw_Server_API/app/services/outputs_service.py`, `tldw_Server_API/app/services/outputs_purge_scheduler.py`; extend `tldw_Server_API/tests/Collections/test_reading_output_updates.py`, `tldw_Server_API/tests/Collections/test_reading_output_disposal_routes.py`, `tldw_Server_API/tests/Services/test_outputs_service.py`, `tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py`.

- [ ] Add RED full-dispatch tests reproducing the three recorded failures: ownership registered after unmanaged dispatch, unowned row aliasing a managed source, and destination collision overwriting managed bytes. Assert both files and all relevant rows unchanged on rejection. Run `python -m pytest tldw_Server_API/tests/Collections/test_reading_output_updates.py -q`.
- [ ] Activated PATCH goes through one prepared replacement with final compound title/format fields; no intermediate rename, hidden normalization write or file-first fallback. Managed title/retention stays DB-only and managed conversion409 stays unchanged. Case-only title may retain spelling; exact no-op allocates no file operation. Shared-unowned copy preserves source while another reference exists.
- [ ] Route activated unowned delete-with-file through remove intent and atomic quota/history commit. Managed explicit disposal still uses Reading-owned intents. Metadata-only retention/soft-delete semantics remain; no file permission is inferred. Remove old duplicate additive quota/history effects on the activated branch only. Purges recheck eligibility under the same fence and count only actual completed unlinks, not queued work.
- [ ] Map known failures to spec errors: 409 busy/path conflict/source unavailable, 503 unavailable/unconfirmed, 507 capacity, 413 overrun; retain auth/missing semantics and path-free logs. Valid pure metadata updates do not acquire file locks. Inactive + ownership/reservations is inconsistent and cannot enter legacy path; unknown binding also fails closed.
- [ ] Run all four listed test modules and `python -m pytest tldw_Server_API/tests/Collections/test_items_and_outputs_api.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q`. Review/commit: `fix(outputs): reserve physical updates and deletion end to end (TASK-13153)`.

## Task 7: Text, briefing and Reading producer integration

**Files:** Modify `tldw_Server_API/app/api/v1/endpoints/outputs.py`, `tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `tldw_Server_API/app/services/outputs_service.py`, `tldw_Server_API/app/core/Collections/reading_service.py`; create `tldw_Server_API/tests/Collections/test_output_file_producers.py`; extend `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py` and `tldw_Server_API/tests/Collections/test_reading_service.py`.

- [ ] Trace every create/write/rename/unlink into the authoritative output root from these entry points. Record each caller and chosen shared boundary in the checkpoint. Include failed multi-variant generation and cancellation. Any additional writer found is a blocker until mapped/tested; do not claim the file list exhausts the repository.
- [ ] Extend the inventory to every reader of activated output files, including internal ingestion/content-loading paths reached by producers. Reuse Task 4's protected descriptor primitive and assign each discovered reader an explicit path-reuse regression; no unreviewed pathname-read exception. Unrelated template/config files are not output artifacts.
- [ ] Add RED producer tests for reservation before first private byte, unknown-length budget, cancellation, publication collision, idempotency replay and inactive Reading creation. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_producers.py -q`.
- [ ] Adapt text/briefing producer callbacks to bounded bytes, admitted before rendering/production. Network/rendering happens outside storage exclusion and writes use Task 3 chunks. Compound/multi-variant operations retain their existing response policy but every produced file has its own bounded journal authority. Failed generation schedules conditional abort, never an unreserved unlink. No separate named scratch artifact.
- [ ] Production Reading creation continues using existing item-owned reserve/adopt, with generic cross-claims and explicit activation prerequisites. Do not manufacture a Reading parent for generic outputs. Parent/revision changes during capture reject stale adoption. Keep legacy reconciliation's unchanged-record and namespace checks; this plan does not replace the parent plan's offline manifest implementation.
- [ ] Run new producer module, Reading service tests and Watchlist audio delivery tests. Review/commit: `fix(outputs): stage generated text under durable reservations (TASK-13153)`.

## Task 8: Audio/subtitle producers and large-file resource checks

**Files:** Modify `tldw_Server_API/app/services/audiobook_jobs_worker.py`, `tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py`, `tldw_Server_API/app/services/outputs_service.py`; extend `tldw_Server_API/tests/Collections/test_output_file_producers.py`, `tldw_Server_API/tests/Audiobooks/integration/test_audiobook_worker_pipeline.py`, `tldw_Server_API/tests/Audiobooks/integration/test_audiobook_subtitle_export_endpoint.py`.

- [ ] Add RED audiobook/TTS/subtitle cases for byte overrun, exhausted capacity, cancellation during a blocked producer, and post-output-commit quota failure. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_producers.py -q`.
- [ ] Move output-specific production to Task 3's reserved private stage with bounded streams/in-memory results. Network waits do not own the OS lock. Any provider pipeline that requires named scratch outside this lifecycle remains disabled for activated stores until its ownership/recovery is covered; do not silently drop the provider or claim rollout complete. Stop and report this if encountered rather than invent an unreviewed scratch framework.
- [ ] Ensure source-to-stage audiobook rename never loads the whole file; committed artifact quota is separate from temporary admission. Subtitle/audio output insertion and quota update share a DB connection, and retry cannot add usage twice. Existing optional provider dependencies remain optional.
- [ ] Run new producer tests plus both listed audiobook integration modules with deterministic bounded producer fakes at network boundaries and real file/DB storage. Add a large streamed fixture and process cancellation test; no real provider charges needed. Review/commit: `fix(outputs): bound audio artifact staging and accounting (TASK-13153)`.

## Task 9: Offline activation, recovery lifecycle and fail-closed rollout

**Files:** Create `tldw_Server_API/scripts/output_storage_protocol.py`, `tldw_Server_API/tests/Collections/test_output_storage_activation.py`; modify `tldw_Server_API/app/services/output_file_operations.py`, `tldw_Server_API/app/services/startup_cleanup_workers.py`; extend `tldw_Server_API/tests/Services/test_startup_cleanup_workers.py` and `tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py`.

- [ ] Add RED tests for missing/invalid finite policy, binding/marker mismatch, wrong node volume, unknown version, unavailable locking, legacy records outside root, ambiguous history links, and repeated dry run with no changes. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_storage_activation.py -q`.
- [ ] Implement offline dry-run-first activation with explicit user/root/policy input. Require stopped writers, verify provenance and resource policy, provision through existing explicit namespace helper, then persist binding under clock. Never activate from schema bootstrap, HTTP or missing-marker recovery. Errors/ordinary logs contain no private filenames; any operator manifest is explicitly sensitive. Ambiguous ownership/history requires operator resolution, not matching by title.
- [ ] Register bounded filesystem-recovery and delivery passes through existing `WorkerSpec` cleanup lifecycle. Do not gate them on OUTPUTS_PURGE_ENABLED. Enumerate users through existing deployment discovery, not only numeric on-disk directories when DB users are remote/non-numeric. Track per-user validated namespace and recovery health; wrong-volume work remains pending. Shutdown signals/stops/joins owned tasks and does not abandon writes while releasing locks.
- [ ] Expose separate sanitized blocked-file versus pending-history counts using existing operational status/metrics. Recovery failure denies new activated file mutations; history outage alone does not. Safe descriptor reads remain available. Retry bounds and per-user due batches prevent one blocked operation starving others.
- [ ] Test startup/termination/retention-disabled/readiness restoration, genuine process death and activation with all production entry points. Do not advertise `hasReadingOptimisticDeletesV1`: parent-plan HTTP/DTO/legacy reconciliation and remaining writer checks are still required.
- [ ] Run activation tests and both listed lifecycle/retention modules. Review/commit: `feat(outputs): activate reserved storage with independent recovery (TASK-13153)`.

## Task 10: Integrated failure matrix, documentation and parent-plan handoff

**Files:** Extend the new modules from Tasks 1–9; update `Docs/API-related/Reading_List_API.md`, `Docs/Published/API-related/Reading_List_API.md`, `backlog/decisions/003-reading-atomic-hard-delete.md`, this plan, the parent plan and the TASK-13153 record via CLI.

- [ ] Execute the full approved-spec failure matrix across new tests: migration twice; both-backend cross-writer races; all fsync/commit/unlink/ack boundaries; exact aliases and multiple volumes; resource exhaustion/cancellation; descriptor/range/disconnect; pending history after ID reuse; all mutation entry points and inactive/unknown activation. Add the smallest missing behavior test before fixing any uncovered gap.
- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_responses.py tldw_Server_API/tests/Collections/test_output_file_history_delivery.py tldw_Server_API/tests/Collections/test_output_file_producers.py tldw_Server_API/tests/Collections/test_output_storage_activation.py -q`, then the exact touched existing suites from each task. This is a focused combined sweep, not the entire repository.
- [ ] Document activated versus inactive behavior, explicit upgrade/resource policy, error codes, open-download semantics, wrong-volume rejection, cleanup versus history status, and blocked-operation recovery. Preserve managed immutability and explicit file permission contracts. Document user action GET-before-retry for uncertain commits; no automatic destructive retry.
- [ ] Review every file-reading and file-writing caller found in Tasks 4/7 for an active-path bypass. Any unmapped reader/writer, deferred pathname reopen, untracked scratch, missing real PostgreSQL evidence or unavailable lock semantics blocks rollout. Keep capability absent and resume the remaining parent-plan tasks only after this checkpoint passes.
- [ ] Run scoped Ruff/Black/Bandit and diff checks, review the final diff, update task evidence/ADR links via CLI, and commit: `docs(outputs): record reserved storage verification and rollout (TASK-13153)`. Record only actually verified outcomes. Do not mark TASK-13153 Done at this sub-plan boundary.

## Review and checkpoint record

Independent plan review found two gaps: disposal of an already-live receiver row,
and output readers outside the Outputs endpoint. Both were corrected with explicit
monotonic receiver transitions, delayed-insertion tests, Watchlist/internal-reader
ownership and path-reuse coverage. Whole-plan re-review approved with no remaining
serious implementation blockers. Checked boxes now reflect the dated execution
checkpoint below, not the earlier plan-review outcome.
At plan approval, validation was document consistency, existing-path checks and
`git diff --check`; no production tests were claimed for the documentation-only plan.

### Task 1a: Inert schema/identity foundation (2026-09-05)

Split Task 1 at its verified schema boundary before implementing transition APIs.
No transition/claim enforcement, filesystem operation, activation or capability is
implemented by this checkpoint. Table DDL reuses the existing schema fence and
never inserts a binding. Legacy output tokens remain NULL through ordinary
bootstrap; explicit user-scoped offline backfill assigns missing UUIDs in one
transaction with bounded read batches. Generic creation and trusted Reading
adoption allocate fresh tokens; existing output DTOs do not expose them.

The initial eight schema/identity tests and separate adoption test failed on the
missing tables/column before implementation. One first GREEN-run fixture failed
because it reused an existing unique title; a distinct title corrected that setup,
without changing production uniqueness behavior. Additional migration/idempotency
and valid-kind checks complete 12 foundation tests per backend.

Final evidence: 12 SQLite foundation passes; 98 non-PostgreSQL targeted revision,
adoption and idempotency regression passes; 106 PostgreSQL foundation/regression
passes with two isolated workers (375.06 seconds). Total: 216 distinct passes.
The PostgreSQL keyword selection also selected the SQLite parameter of
`test_postgres_search_path_cannot_reset_public_revision_clock`, which intentionally
skipped; collection confirmed it is the only SQLite case selected. No required
PostgreSQL case skipped. Independent read-only review found no actionable
foundation issues. Python compile checks also pass.
New test module Ruff/Black and changed production-range Black checks pass; the
adapter's nine preexisting Ruff findings and whole-file Black debt are unchanged
against HEAD. Scoped Bandit reports no findings or scanner errors.

Verification commands (after environment activation):

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py -q -k sqlite
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_output_artifact_idempotency.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_output_artifact_idempotency.py -q -k postgres -n 2
```

Future PostgreSQL selection can use `-k 'postgres and not sqlite'` to exclude that
inapplicable case explicitly; also set `TLDW_TEST_POSTGRES_REQUIRED=1` to turn
cluster unavailability into an error rather than a fixture skip.

Logs: `/private/tmp/task-13153-journal-schema-{red,green,pg,regressions}.log`,
`/private/tmp/task-13153-journal-adoption-red.log`,
`/private/tmp/task-13153-journal-schema-bandit.json`.
Next at that checkpoint: remaining Task 1 transition methods with fresh failing tests. All full-task
acceptance criteria remain unchecked and TASK-13153 stays In Progress.

### Task 1b: Dormant journal transitions (2026-09-05)

Implements scoped prepare/get/validate/commit/abort/fs_done/ack methods under the
existing revision fence. The original-row snapshot stores a complete-row SHA-256
digest and immutable incarnation, not raw output content. Intended changes are a
bounded field allowlist. Validation checks the current protocol binding and reads
fresh lease time after the fence. Removal supports soft-deleted originals while
replacement remains active-only; deletion-state changes invalidate the snapshot.

Implementation clarification: `commit_output_file_operation` is a transaction
context, not a CRUD method. It yields the same connection for the trusted caller's
recorded output mutation and accounting, then updates the journal phase after the
body succeeds. An exception rolls both back; an explicitly supplied transaction
must be rolled back by its owner on propagated exceptions. Bodies must be DB-only.
Later orchestration must supply filesystem proofs and enforce cross-writer claims;
these primitives alone do not authorize any file operation or runtime activation.

File completion releases reserved bytes independently of pending history effects.
History acknowledgement is idempotent and cannot retire unfinished file work.
Committed work cannot become aborting or run its mutation body again.

The first 12 transition tests failed on absent methods with all 12 foundation
tests still passing. Review then found soft-deleted removal was incorrectly
excluded. Three further RED cases reproduced that gap and a protocol-version
change after preparation; the active-only replacement control passed. Fixes now
pass with additional expiry-after-fence and both-order threaded commit/abort
coverage. Independent follow-up review found no remaining serious issues within
this deliberately dormant primitive scope.

Final evidence: 109 non-PostgreSQL targeted passes (31 journal/foundation and 78
existing revision tests), plus 110 required real PostgreSQL passes with two
isolated workers (388.89 seconds). Total: 219 distinct passes, no skips. The
complementary backend selections exclude only the intentionally inapplicable
SQLite parameter of the PostgreSQL search-path test.
New test module Ruff/Black, changed production-range Black, compile and diff checks
pass. Adapter Ruff still reports exactly the same nine findings as HEAD; existing
whole-file formatting debt is not reformatted. Scoped Bandit has no findings or
scanner errors. Exact-file checkpoint commit follows this verification.

Verification commands (after environment activation):

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -q -k 'postgres and not sqlite' -n 2
```

Logs: `/private/tmp/task-13153-journal-transitions-{red,green,sqlite,pg}.log`,
`/private/tmp/task-13153-journal-review-red.log`, and
`/private/tmp/task-13153-journal-transitions-bandit.json`.
Next: Task 2 shared row/path claims and resource admission. Task 1 does not provide
those guards. TASK-13153 remains In Progress and the public capability stays off.

### Task 2a: Admission capacity and transactional accounting (2026-09-05)

This is the capacity/accounting portion of Task 2, not its cross-writer guards.
Preparation reads a supported policy and verifies all six stored limits are
finite positive integers. SQLite integer affinity admits fractions despite a
positive-range CHECK, so runtime validation is explicit. The same reader is used
when validating an existing operation. Under the existing revision fence,
preparation checks the per-operation ceiling and aggregates same-user pending
bytes/count before inserting the journal record. Every `fs_done=0` phase counts,
including expired and blocked work; zero-byte operations still consume a slot.
No source bytes, filesystem free space or producer limits are measured here.

The existing audiobook usage get/set/update helpers now accept an optional
internal connection. Their standalone behavior is preserved. A caller can compose
the output write, quota change and journal phase on one transaction; rollback
restores them together, and the journal rejects replay before a second delta can
execute. The additive helper itself is not idempotent. Legacy production callers
have not been migrated and their later integration remains required.

TDD evidence: ten initial admission cases failed for missing capacity/policy
checks, then the 41-test SQLite journal module passed. Three accounting tests
failed on missing connection support before that change. Additional tests cover
user isolation, admission rollback, abort retaining capacity until file completion,
and file completion releasing capacity despite pending history acknowledgement.
Independent read-only review found no actionable issues in this checkpoint.

Final verification: 74 non-PostgreSQL passes (48 journal tests, 26 existing
output-deletion regressions), plus 73 required real PostgreSQL passes with two
isolated workers (213.39 seconds). Total: 147 distinct passes, no skips.
The fractional-policy regression is explicitly SQLite-only because PostgreSQL's
column type does not have that affinity behavior; it is not a skipped PG test.
New tests pass Ruff/Black; modified production ranges pass Black. Adapter Ruff's
nine preexisting findings remain unchanged. Compile/diff checks pass and scoped
Bandit reports no findings or scanner errors.

Verification commands (after environment activation):

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q -k 'postgres and not sqlite' -n 2
```

Logs: `/private/tmp/task-13153-output-admission-{red,green}.log`,
`/private/tmp/task-13153-output-accounting-red.log`,
`/private/tmp/task-13153-output-capacity-{sqlite,pg}.log`, and
`/private/tmp/task-13153-output-capacity-bandit.json`.
Next: Task 2b row/path claims across generic and Reading writers, both-order races,
recorded-transition exceptions, and ID-reuse protection. Activation stays off;
the complete Task 2 checklist and full-task acceptance criteria remain unchecked.

### Task 2b: Shared database row/path claims (2026-09-05)

Adds one shared unfinished-operation guard under the existing revision fence.
Generic creation/retargeting checks all three path claims; metadata/link changes,
soft/hard deletion and retention check the original row/path. Newly allocated IDs
are checked before commit for both generic insertion and Reading adoption, so
SQLite cannot reuse an ID still held by unfinished cleanup. All queries are scoped
to the user and `fs_done=0`; history-only records impose no row/path claim.

Reading registration, reserve, staging validation/adoption, owned-output disposal,
and cleanup preparation/retirement honor generic claims. Known ownership/volume
namespaces remain distinct; unowned generic references stay conservative. Incoming
legacy absolute paths are compared by conservative basename, and stored output
references and Reading intents also match escaped absolute suffixes. Exact names
remain unchanged for future filesystem identity checks.

Generic preparation and precommit validation check both journals and surviving
outputs. Managed source aliases and occupied destinations reject; shared unowned
sources remain eligible for a later copy/preserve lifecycle. Only the internal
validator, after checking user/namespace, phase, lease and the original snapshot,
excludes its own journal token. No public writer accepts a bypass token. The
existing trusted caller-owned commit context remains a DB primitive, not an
implemented filesystem operation or runtime producer integration. Future legacy
reconciliation must use the guarded registration boundary.

TDD: 14 initial writer/ID cases failed; one foreign-user control passed. Ten more
admission/Reading cases failed with the shared-unowned control passing. Tests then
exposed missing cleanup guards (two RED cases). Independent review found stored
absolute Reading intent aliases; three RED cases covered attachment, prepare and
revalidation. Two further RED cases corrected over-conservative path checks on
owned rows whose known volume differs. Both-order threaded races exercise reserve
against ownership, destination attachment, metadata and deletion. The older stale
snapshot test now verifies normal writes reject first and uses explicit incompatible
SQL to retain its stale-commit check. One test fixture was corrected to compare
persisted Reading state rather than transient upsert `is_new` flags.

The initial PostgreSQL run failed early on untyped optional-null predicates and
was interrupted, not counted as verification. An isolated read-only probe proved
SQLSTATE 42P18; explicit text casts fixed the probe and isolated regression.
This incident is recorded in `backlog/docs/lessons-testing-evidence.md`.
Independent follow-up review found no remaining serious DB-guard findings.

Verified evidence: 134 non-PostgreSQL combined passes plus one focused history-only
claim-release test, and 133 required PostgreSQL passes in 332.56 seconds with no
required-backend skips: 268 distinct targeted passes. New/changed tests pass
Ruff/Black, modified production ranges pass Black, and the adapter's nine
preexisting Ruff findings remain unchanged. Compile and diff checks pass; scoped
Bandit reports no findings or scanner errors. No full-suite, filesystem lifecycle
or activation claim.

Verification commands (after environment activation):

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_claims_db.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_claims_db.py -q -k 'sqlite and file_completion_releases_claims'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_claims_db.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q -k 'postgres and not sqlite' -n 2 -x
```

Logs: `/private/tmp/task-13153-output-claims-writers-{red,green}.log`,
`/private/tmp/task-13153-output-claims-admission-red.log`,
`/private/tmp/task-13153-output-claims-races.log`,
`/private/tmp/task-13153-output-claims-review-red.log`,
`/private/tmp/task-13153-output-claims-{sqlite-final,release,pg-final}.log`,
and `/private/tmp/task-13153-output-claims-bandit-final.json`.
Next: Task 3 bounded file protocol and restart recovery. The exhaustive writer/
path/order integration matrix and recorded runtime mutation boundary remain
rollout obligations; this checkpoint does not mark the complete Task 2 checklist
or full-task acceptance criteria complete. No storage activation or public
capability is enabled.

### Task 3a: Durable evidence and bounded offloaded staging (2026-09-05)

Adds internal `OutputFileOperations.prepare`/`write_chunk`, reusing the existing
verified directory descriptor without changing Reading storage primitives. No
runtime callers, publication, unlink or activation. The adapter now exposes its
validated policy and an immutable source/stage evidence + compare-and-set offset
transition under the existing revision fence. Source fingerprints include device,
inode, type, link count, size and modification/change timestamps; stage authority
uses device/inode/type/link count plus the durable expected length. All source
bytes and output metadata remain unchanged during this checkpoint.

Preparation checks physical capacity and reserves source-size plus declared
maximum staging bytes before exclusive private creation. It records source
evidence before stage creation and syncs the empty stage/directory before recording
stage evidence. Every write reacquires volume exclusion, checks live phase/lease,
source/stage identity and exact offset, enforces cumulative bounds, syncs/closes
the writable descriptor, then records the new offset. One call writes at most
1 MiB, yielding exclusion more frequently than the plan's 8 MiB maximum. There
are no producer/network waits or filesystem calls inside DB mutation transactions.

Cancellation drains the offloaded asyncio worker before returning, then attempts
conditional abort under verified storage exclusion. AnyIO task-group and direct
asyncio cancellation are both covered. I/O errors are sanitized, including stat/
free-space preflight calls. Failed writes or lost identity/offset acknowledgements
retain source bytes and durable claims; they do not truncate/recreate ambiguous
stages or delete files. Automatic recovery/blocked-category reporting is still
Task 3b, so this is not yet a usable production file-operation lifecycle.

TDD: 15 initial assertions failed on the missing service/DB transition, then
passed. Fault tests exposed raw preparation I/O errors and detached direct-asyncio
cancellation; follow-up cases exposed AnyIO cancellation not scheduling abort.
Independent review found soft-deleted removal lookup and preflight error privacy.
Both findings were reproduced and fixed. The removal regression initially used
unsupported test APIs/fields; after inspecting the adapter it uses real soft
deletion plus both supported lookup modes. Those fixture failures and the
interrupted PostgreSQL run are not counted as verification. Follow-up read-only
review confirmed the fixes with no remaining serious checkpoint findings.

Verified evidence: 173 SQLite/non-PostgreSQL cases pass in 17.60 seconds and
103 required PostgreSQL cases pass in 266.99 seconds, with no required-backend
skips: 276 distinct targeted passes. New service/tests pass Ruff and Black;
modified adapter ranges pass Black. Compile/diff checks pass; scoped Bandit
reports zero findings and zero scanner errors. Adapter Ruff retains exactly the
nine baseline findings, verified against `2583f3e443`.

Verification commands, after activating the Server virtual environment:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_output_file_claims_db.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'postgres and not sqlite' -n 2 -x
```

Logs: `/private/tmp/task-13153-staging-{red,green,faults-red,faults-green}.log`,
`/private/tmp/task-13153-staging-review-red-valid.log`,
`/private/tmp/task-13153-staging-{sqlite,pg}-verified.log`,
`/private/tmp/task-13153-staging-bandit.json`.

Next: Task 3b bounded source-copy orchestration, no-clobber publication, recorded
DB mutation and phase-specific restart recovery with the remaining crash matrix.
The complete Task 3 checklist/full-task AC remain unchecked. No push, PR, merge,
full-suite claim or public capability activation.
