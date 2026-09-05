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
**Status:** Independently reviewed and ready for inline execution. No implementation or new runtime evidence.

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

- [ ] Add failing repeated-bootstrap tests for `output_storage_bindings` and `output_file_operations`, with explicit NOT NULL identities, user scope, no output cascading FK, valid kinds/phases, nonnegative counts/budgets, and `fs_done` only on committed/aborting rows. Schema installation must not create/provision paths or activate a user.
- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py -q -k sqlite`; confirm new behavior fails.
- [ ] Extend `_ensure_reading_revision_schema` using its existing transaction/advisory bootstrap lock and explicit connection. Bindings contain user, namespace, protocol version and validated finite policy. Journal stores the spec's fixed source/stage/destination fields and collision keys, snapshot/intended changes, source/stage/publication identity, lease/retry/error, bounded history payload/status, and byte reservation. Use compact bounded JSON only for finite structured fields, never bodies/credentials. Add due-work and active-user lookup indexes; no general lock table.
- [ ] Add internal immutable `outputs.file_incarnation` (random UUID on insert, retained on rename, never copied on new output/ID reuse). Existing rows receive new tokens in an idempotent stopped-writer backfill; never reset a nonempty token. This is not a public patchable metadata key. `create_output_artifact` remains the only ordinary allocator.
- [ ] Implement transaction-owned methods `prepare_output_file_operation`, `get_output_file_operation`, `validate_output_file_operation`, `commit_output_file_operation`, `abort_output_file_operation`, `finish_output_file_operation`, and `ack_output_file_effect`. Internal optional `connection` propagates the same DB transaction; public HTTP payloads never accept operation tokens. Fresh lease time is read after the fence, not supplied pre-wait.
- [ ] Test conditional transitions, unknown token, other user, expiry, unchanged source snapshot, and rollback. Core transition predicates are:

```sql
-- All statements additionally run under the existing revision fence.
UPDATE output_file_operations SET phase = 'aborting'
WHERE token = ? AND user_id = ? AND phase = 'prepared' AND fs_done = 0;
UPDATE output_file_operations SET fs_done = 1, reserved_bytes = 0
WHERE token = ? AND user_id = ? AND phase IN ('committed', 'aborting') AND fs_done = 0;
DELETE FROM output_file_operations
WHERE token = ? AND user_id = ? AND fs_done = 1 AND effects_pending = 0;
```

- [ ] Run the new module unfiltered and `python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -q`. Check migration twice on both backends, including reconstructed adapter instances. Review/commit: `feat(reading): persist inert output operation journal (TASK-13153)`.

## Task 2: Enforce shared row/path claims and resource admission

**Files:** Modify `tldw_Server_API/app/core/DB_Management/Collections_DB.py`; extend `tldw_Server_API/tests/Collections/test_output_file_operations_db.py` and `tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py`.

- [ ] Add RED races for all three path columns and the reserved output identity against generic create/update/delete/retention, Reading registration/reserve/adopt/reconciliation and guarded item delete, in both commit orders. Include metadata-only mutations, absolute/case aliases, managed-source alias, shared-unowned source, namespaces and foreign user. No writer may bypass through a direct adapter method.
- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py -q -k sqlite`.
- [ ] Extend `_assert_output_path_not_reserved` and shared row checks at the existing transaction boundaries. Query generic claims only where `fs_done = 0`; check all source/stage/destination collision keys and exact recorded identity. A private token exception validates kind, row snapshot and the recorded transition, not merely equality to any journal token. Reading journal checks remain mandatory. Row guards also cover ID allocation/reuse before `fs_done`.
- [ ] Validate finite positive per-operation/per-user byte and active-operation caps, text input/output caps and free-space margin before activation/admission. Under the fence sum active reserved bytes and count active operations; include blocked/pending file work, exclude delivery-only rows. Reject before a producer/file exists with `output_storage_capacity` or policy-unavailable. Different users do not spend one another's budget; physical free-space checks are advisory and happen in Task 3 outside DB transactions.
- [ ] Prove two concurrent reservations cannot overspend; `fs_done` releases row/path/count/bytes together despite pending history, and old completion/ack never touches a recycled output. Keep same-store audiobook delta in the output commit transaction with explicit connection; no best-effort additive replay.
- [ ] Run new DB module and `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -q`. Review/commit: `fix(reading): fence output writers with durable claims (TASK-13153)`.

## Task 3: Bounded file protocol and restart recovery

**Files:** Create `tldw_Server_API/app/services/output_file_operations.py`, `tldw_Server_API/tests/Collections/test_output_file_operations_storage.py`; modify `tldw_Server_API/app/services/reading_artifact_cleanup_service.py` only to expose/reuse its verified directory descriptor context.

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
serious implementation blockers. No boxes above represent completed implementation.
Current validation is document consistency, existing-path checks and
`git diff --check`; no production tests were run for this documentation-only plan.
