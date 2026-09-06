# Reading-safe output file mutations implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user has selected inline execution; do not ask again or delegate implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make activated generic output writers and readers preserve Reading-owned bytes through concurrency, crashes, delayed history delivery, and bounded large-file production.

**Architecture:** Reuse the Collections revision fence and verified descriptor-relative POSIX storage lock. Add one bounded generic operation journal, a persisted per-user activation binding, and narrowly scoped services for file operations and opened-descriptor responses. Filesystem completion releases reservations independently of history delivery; no public jobs API or generic locking/outbox framework.

**Tech Stack:** Python, existing FastAPI/Starlette, SQLite and PostgreSQL adapters, stdlib `os`/`stat`/`uuid`, existing worker lifecycle and pytest fixtures. No new dependency.

**ADR required:** yes — existing ADR-003 applies and is amended.
**ADR path:** `backlog/decisions/003-reading-atomic-hard-delete.md`.
**Reason:** Implements the approved storage, cross-writer and external-delivery contract.
**Task:** TASK-13153, In Progress. These are checkpoints within that task, not newly allocated Backlog IDs.
**PR:** https://github.com/rmusser01/tldw_server/pull/2903 (draft against `dev`,
opened at `f43549c209` on user request; implementation continues on the same branch).
**Approved spec:** `Docs/superpowers/specs/2026-09-05-reading-output-file-reservations-design.md`, user approved after checkpoint `8dc255fcca`.
**Parent plan:** `Docs/superpowers/plans/2026-09-04-reading-atomic-hard-delete.md`; this plan replaces its generic-file-writer gap, not its remaining Reading DTO/HTTP/reconciliation/release tasks.
**Status:** Inline execution in progress. Task 1, Task 2a, Task 2b, Task 3a staging/write and Task 3b's recorded commit, copy/publication, recovery and real-process kill/progress checkpoints verified on SQLite/PostgreSQL. Immediate post-commit cleanup reuses recovery under publication exclusion. Task 4a descriptor responses, Task 4b protected generic downloads and Task 4c registered Watchlist readers are implemented; checkpoint verification is recorded below. Unregistered evidence sidecars remain blocked for activated reads pending producer/reconciliation provenance. History/producer integration and Task 9 background lifecycle/activation remain pending.

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

Task 3b begins with a separately verified DB-owned recorded-mutation checkpoint
in `Collections_DB.py` and `test_output_file_operations_db.py`: publication
evidence, intended create/replace/remove changes, allocated-ID claims and atomic
quota/history updates. Services must not issue raw SQL or use a generic writer's
unvalidated token bypass. Descriptor-relative publication/recovery follows this
boundary; no file lifecycle is claimed complete by the DB-only checkpoint.

- [ ] Write RED tests against real temporary files for copy-before-commit, occupied destination, absent source, symlink/special/unexplained hardlink, changed source fingerprint, and same DB/different root. Start with service-level replace; no routes yet. Run `python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py -q`.
- [ ] Implement `prepare`, `write_chunk`, `publish_and_commit`, and `recover_due` as a small service using Task 1 methods. Lock order is verified storage context then DB fence. Capture original fingerprint before first read. Reserve before exclusive stage creation; persist stage identity before leaving the first lock interval. No filesystem/network call occurs inside DB mutation transactions. Never reopen the root path after verifying its descriptor.
- [ ] Chunk with buffers at most 1 MiB and lock intervals at most 8 MiB or 50 ms measured between chunks. Close writable FDs before releasing exclusion; no background write survives cancellation/unlock. Reacquire/revalidate token, lease, source identity, stage identity and exact written offset before continuing. Persist expected offset after each completed interval. A crash between file-write and DB offset acknowledgement leaves ambiguous work blocked, not blindly truncated/resumed. Run blocking I/O off the event loop and bound nonstream converters before loading input.
- [ ] Check actual free space before admission and periodically, count hardlink allocation once, and enforce declared limits before each write. ENOSPC safely aborts; source survives. Reject overrun with `output_size_limit`. Reader-held unlinked files remain physical allocations despite logical budget release.
- [ ] Publish with `os.link(stage, destination, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)` and directory fsync, retaining the private witness. Existing destination always conflicts. Commit metadata + committed phase + quota on one connection. On an uncertain commit acknowledgement reread phase on a fresh connection; committed wins, unreachable DB preserves every file/claim and returns `output_update_unconfirmed`.
- [ ] Implement phase-specific cleanup: abort removes verified destination then fsyncs before witness removal; commit preserves destination, removes witness and proven unreferenced original source. Recheck exact source identity/fingerprint before unlink. Persist `fs_done` only after required fsync/preservation; no-effects aborts retire immediately. Unproved pre-identity stages and changed files stay blocked for operator inspection.
- [ ] Add crash injection at reservation/create/identity record/write/fsync/link/commit/unlink/fs_done/retirement and lost commit acknowledgements. Add process-kill lock release, late producer after expiry/retirement, two-process contention, bounded-memory large copy, cooperative reader/writer progress and cancellation during offloaded writes. Do not signal a multiprocessing Event after terminating its waiter.
- [ ] Run new module plus `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q`. Review/commit: `feat(reading): recover bounded reserved output writes (TASK-13153)`.

## Task 4: Protected lookup and opened-descriptor HTTP responses

**Files:** Create `tldw_Server_API/app/services/output_file_response.py`, `tldw_Server_API/tests/Collections/test_output_file_responses.py`, and `tldw_Server_API/tests/Collections/test_output_download_compatibility.py`; modify `tldw_Server_API/app/api/v1/endpoints/outputs.py` and `tldw_Server_API/app/api/v1/endpoints/watchlists.py`; extend `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`.

Execution split: Task 4a characterizes existing generic HTTP downloads and the
response-start/path-reuse race, then establishes the bounded descriptor-owning
response and its lifetime/HTTP tests. Reuse installed Starlette range/header
helpers, not a pathname response or a new HTTP parser. Task 4b adds protected
DB lookup, namespace/publication-witness checks, and migrates generic ID/title/HEAD
downloads. Task 4c then audits and migrates Watchlist downloads/content loaders
and sidecar output reads. Neither the response foundation alone nor
its compatibility tests establish protected lookup or activated-reader readiness.

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
- [ ] Resolve the Task 4c evidence-sidecar blocker before activation: `_write_report_snapshot_for_user` currently creates unregistered JSON referenced only by `report_snapshot_path` metadata, shared across generated variants. Establish and review explicit publication/ownership/reconciliation authority for these sidecars, then migrate their reads through the protected boundary. Until that prerequisite is met, activated sidecar reads return sanitized 503 without filesystem access; do not treat metadata filenames as authority or mark reader rollout complete.
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

### Task 3b database boundary: Recorded output commit (2026-09-05)

PR #2903 was opened as a draft against `dev` at `f43549c209`, as requested,
before continuing implementation. The PR includes the accumulated checkpoints;
its description explicitly retains the human-written merge-summary gate and does
not claim end-to-end readiness.

Adds `CollectionsDatabase.apply_output_file_operation`, a DB-only boundary for
the forthcoming publication orchestrator. It rechecks the prepared token, live
lease, original snapshot, scoped ownership/path claims and immutable file evidence
under the existing fence. Create/replace require caller-provided publication
identity matching the private stage with its expected two links; remove requires
source evidence and no stage/publication. These are checks of trusted recorded
evidence, not filesystem proof: the service must still verify actual files and
directory durability before invoking this method.

The transaction applies only recorded title/format/retention/path changes,
preserves other replacement fields and incarnation, stores actual staged byte
size in metadata, and records publication identity plus committed phase. Creation
allocates a new incarnation and checks reused IDs against other unfinished claims
before binding the journal's output ID. Removal records original-incarnation
history disposal in the same transaction. Same-store audiobook deltas use the
same connection; absent/underflowing/overflowing usage fails closed instead of
guessing or doing filesystem recomputation. Existing byte-size metadata supplies
the previously accounted size, falling back only to the recorded source size.
Soft-deleted removal does not subtract already-decremented usage again. Activated
store initialization/reconciliation must establish usage before these writes.

TDD: 14 initial recorded-commit assertions failed on the missing boundary and then
passed. Additional scoped-authority, reused-ID, accounting, and two-thread duplicate
commit regressions passed. Independent read-only review found no serious DB-only
findings. Self-review then added two failing database-suppressed-write cases;
requiring exactly one updated/deleted output row now rolls those transactions back
instead of marking an unchanged output committed.

Verified evidence: final SQLite/non-PostgreSQL combined run passes 199 cases in
24.28 seconds. Broader PostgreSQL regression passes 126 cases in 342.31 seconds;
that run preceded the final row-count guard. All 25 PostgreSQL recorded-commit
cases were then rerun after the guard and passed in 76.54 seconds. These runs
cover 128 distinct PostgreSQL cases (23 repeated), hence 327 distinct targeted
cases across both backends, with no required-backend skips. Modified tests pass
Ruff/Black; touched adapter ranges pass Black; compile/diff checks pass and scoped
Bandit reports zero findings/scanner errors. The adapter's existing nine unrelated
Ruff findings remain. Independent follow-up review found no concern with the
final row-count guard.

Verification commands after Server virtual-environment activation:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_output_file_claims_db.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'postgres and not sqlite' -n 2 -x
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_db.py -q -k 'postgres and recorded_ and not sqlite' -n 2 -x
```

Logs: `/private/tmp/task-13153-recorded-commit-{red,green,guards,rowcount-red}.log`,
`/private/tmp/task-13153-recorded-commit-sqlite-verified.log`,
`/private/tmp/task-13153-recorded-commit-pg-final.log`,
`/private/tmp/task-13153-recorded-commit-pg-focused.log`,
`/private/tmp/task-13153-recorded-commit-bandit-final.json`.

Next: the Task 3b filesystem orchestrator (copy, no-clobber publication, uncertain
commit acknowledgement and phase-specific recovery). Producer-specific metadata/
idempotency mapping, route integration and all rollout gates remain pending.
No runtime wiring, activation, full-task completion or merge is claimed.

### Task 3b filesystem boundary: Bounded copy and no-clobber publication (2026-09-05)

Adds internal `copy_source` and `publish_and_commit` methods. Copy uses at most
1 MiB buffers, closes source descriptors between intervals, and revalidates the
token, source fingerprint, stage identity and acknowledged offset on resume,
including at EOF. Existing bounded writes provide capacity enforcement, syncing,
lease refresh and cancellation handling. Publication verifies and syncs the private
file, links without replacement through the verified directory descriptor, checks
both links and their lengths, then syncs the directory and rechecks the source
before invoking the recorded DB mutation. Create, replace and remove share this
logical commit boundary. No file is unlinked and no reservation is released here.

Uncertain commit acknowledgements are resolved on a new non-pooled connection,
under the existing revision fence. Committed state always wins. If the state
cannot be established, all files and claims remain and the service reports
`output_update_unconfirmed`. A failed conditional abort requires another outcome
read rather than assuming rejection. No new ADR is required: this directly
implements the approved ADR-003 contract.

TDD: 13 initial real-volume cases failed on missing methods, then passed. Expanded
fault tests exposed an EOF resume that skipped stage revalidation; the added guard
made the 22-case group pass. Independent review reproduced a delayed commit that
completed after the first outcome read: conditional abort returned false but the
service reported a definite conflict. Both new regressions failed, then passed
after fencing the fresh read and respecting the abort result. The follow-up review
confirmed resolution with no further checkpoint findings. The incident is recorded
in `backlog/docs/lessons-testing-evidence.md`.

Verified evidence: the final SQLite/non-PostgreSQL regression run passes 223
cases in 21.21 seconds. The broader PostgreSQL run passes 150 cases in 384.59
seconds, using code loaded before the final review fix. All 24 new PostgreSQL
copy/publication cases were rerun after that fix and passed in 76.32 seconds.
These cover 152 distinct PostgreSQL cases (22 repeated), hence 375 distinct
targeted cases across both backends, with no required-backend skips. Service and
tests pass Ruff/Black; touched adapter ranges pass Black; compile/diff checks
pass; scoped Bandit reports zero findings and scanner errors. Adapter Ruff
retains exactly the nine baseline findings, compared by code/message to HEAD.
No full suite was run and no Docker instance was provisioned.

Verification commands after activating the Server virtual environment:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_output_file_claims_db.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'postgres and not sqlite' -n 2 -x
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_operations_storage.py -q -k 'postgres and (publication or source_copy)' -n 2 -x
```

Logs: `/private/tmp/task-13153-publication-{red,green,faults-red,faults-green}.log`,
`/private/tmp/task-13153-publication-review-{red,green}.log`,
`/private/tmp/task-13153-publication-sqlite-verified.log`,
`/private/tmp/task-13153-publication-pg-{final,focused}.log`,
`/private/tmp/task-13153-publication-bandit-final.json`.

This is not filesystem completion or runtime readiness. Next is phase-specific
cleanup/recovery: prove witness ownership, sync abort destination removal before
witness removal, preserve committed publication, fingerprint/reference-check old
sources, and release reservations only after durable cleanup. Due-work selection,
bounded retry/blocked reporting, remaining process-crash matrix and all later
producer/reader/activation tasks remain pending. The complete Task 3 checklist
and full-task acceptance criteria remain unchecked; PR #2903 remains a draft.

### Task 3b cleanup boundary: Phase-specific recovery (2026-09-05)

Adds internal `recover_due`, bounded to 20 operations by default (validated range
1–100), with one verified storage interval per operation. Selection excludes live
prepared leases, future retries, operator-blocked identities and `fs_done` rows.
After acquiring storage exclusion, the adapter rechecks due state under the
revision fence and changes only expired prepared work to aborting. Surviving
source references are checked on that same transaction, including soft-deleted
rows and case/legacy absolute/Windows aliases, scoped to the user and known
namespace. Existing reservations prevent new compliant attachments during file
cleanup outside the transaction.

Aborting cleanup never touches source bytes. A destination requires the recorded
stage identity and live private witness, with the expected two links and length.
Destination unlink and directory fsync precede witness unlink; a failed sync
retains the witness. On restart with destination already absent, directory sync
still precedes witness removal. Committed cleanup preserves the identity-verified
destination, removes only its proven private link, and unlinks an unreferenced
source only after checking the original full fingerprint. A committed destination
with its witness already removed must have exactly one link. Final directory
sync precedes `fs_done` and reservation release. No-effects work retires; removal
history remains independently pending, with no later filesystem access.

Missing/wrong volumes, busy exclusion and I/O failures retain claims with a
sanitized 60-second retry. Unproved identity/length stays operator-only blocked
using the existing maximum retry sentinel; automated failure updates cannot
downgrade that state. Attempts saturate instead of overflowing. Database failures
return `output_update_unconfirmed`; cleanup never guesses that an unacknowledged
final DB transition succeeded. Cancellation drains the active cleanup interval
without conditionally aborting another live producer. No new ADR or schema is
needed: these are direct implementations of ADR-003 and the existing journal.

TDD: 15 initial cases failed on the missing recovery entry point; two legacy-path
fixtures instead failed at the modern insertion boundary. After correcting those
fixtures to model historical persisted paths, all 17 basic cases passed. Expanded
restart/fault coverage passed 33 cases, including interruption after each unlink
and before/after final DB completion. Three subsequent RED cases exposed raw DB
errors at selection, phase recheck and completion; sanitizing them made the
40-case group pass. Independent review reproduced a delayed busy/unavailable
failure report clearing an identity block. Both added regressions failed, then
passed after making automated identity blocking monotonic; follow-up review found
no remaining checkpoint issue. The incident is recorded in the testing lessons.

Final evidence: 265 SQLite/non-PostgreSQL cases pass in 37.13 seconds; 122 required
PostgreSQL cases pass in 394.22 seconds after the final review fix. This is 387
distinct targeted cases, with no required-backend skips. An earlier focused
PostgreSQL run passed 40 cases before the sticky-block fix; it is superseded by
the final run, not counted again. Service and new tests pass Ruff/Black, touched
adapter ranges pass Black, compile/diff checks pass, and scoped Bandit reports
zero findings and scanner errors. The adapter retains exactly its nine baseline
Ruff findings, compared by code/message to `8f547266d1`. No full sweep or Docker
provisioning was performed.

Verification commands after Server virtual-environment activation:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_recovery.py tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_operations_db.py tldw_Server_API/tests/Collections/test_output_file_claims_db.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_recovery.py tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'postgres and not sqlite' -n 2 -x
```

Logs: `/private/tmp/task-13153-recovery-{red,green,green-verified,faults}.log`,
`/private/tmp/task-13153-recovery-db-errors-red.log`,
`/private/tmp/task-13153-recovery-review-{red,green}.log`,
`/private/tmp/task-13153-recovery-sqlite-final.log`,
`/private/tmp/task-13153-recovery-pg-{focused,final}.log`,
`/private/tmp/task-13153-recovery-bandit-final.json`.

The 42 new recovery cases live in
`tldw_Server_API/tests/Collections/test_output_file_recovery.py`, separate from
the staging/publication tests. No runtime caller, background-worker scheduling,
automatic post-publication cleanup integration, or capability activation is added.
Real process-kill/two-process and remaining end-to-end fault coverage are still
required before marking the complete Task 3 checklist done. Later descriptor
readers, history receiver, producer/route adaptation and rollout gates also remain
pending. The task stays In Progress and PR #2903 stays draft.

### Task 3b process evidence: Actual kill and cross-process progress (2026-09-05)

Adds `tldw_Server_API/tests/Collections/test_output_file_process_recovery.py`.
This is characterization/fault evidence for the existing implementation, not a
new production behavior change; no service or adapter code changed. Spawned
workers construct their own backend/Collections adapter from the test fixture's
configuration, avoiding inherited live database connections. Parent and worker
coordinate over a pipe. Parent sends SIGKILL while the worker is paused inside
the real storage interval, checks its negative signal exit code, and reacquires
the same persistent lock inode before recovery. Bounded teardown kills only the
test's worker and never signals a terminated process's semaphore.

The 12 kill boundaries are: durable reservation, private file creation before
identity recording, recorded stage identity, file sync before offset recording,
no-clobber link before directory sync, committed DB mutation, aborted destination
unlink, aborted witness unlink, committed witness unlink, source unlink, and
before/after final filesystem-completion DB transaction. Recovery preserves
unproved pre-identity/unacknowledged files as blocked; confirmed aborts leave the
original bytes and no orphan private files; commits retain the published bytes
and clean the old source. Retirement replay does not reopen disposed authority.

Two additional tests pause a separate producer before a write, abort or expire/
recover/retire its operation, and prove it cannot recreate the private file after
resuming. A 12 MiB source-copy test checks each write is at most 1 MiB, traced
Python allocation peak during copy is below 8 MiB (not an RSS claim), and another
process acquires storage exclusion to read between all twelve chunks. In the
first gap, a conflicting mutation is rejected by durable claims while an
unrelated create/write/commit/recovery succeeds and its bytes are checked.

Initial SQLite and PostgreSQL process runs passed. Review identified two evidence
gaps: successful retirement did not explicitly check for orphan private files,
and a rejected competing writer did not prove successful writer progress. Both
assertions/flows were added and follow-up review found no remaining issue. An
isolated parent-side negative-control pytest plugin deliberately skipped aborted
cleanup: the real stage-recorded kill test then failed specifically with
`retired operation left an orphan private file`. The plugin is only in
`/private/tmp`, was not used in final runs, and is not part of the repository.
No new ADR is needed: test-only verification of ADR-003's approved contract.

Final verification after review changes: 140 SQLite/non-PostgreSQL process,
recovery, publication and lock regressions pass in 36.00 seconds; all 15 new
PostgreSQL process cases pass in 71.29 seconds. This is 155 distinct targeted
cases, with no required-backend skips. Earlier overlapping runs are superseded,
not counted again. Ruff, Black, compile and diff checks pass. Scoped Bandit on
the new test module reports no findings or scanner errors with only B101
(ordinary pytest assertions) excluded; the first unfiltered scan reported only
those assertions. No production code was modified, no full sweep was run, and
the existing PostgreSQL fixture was used without Docker provisioning.

Commands after Server virtual-environment activation:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_process_recovery.py tldw_Server_API/tests/Collections/test_output_file_recovery.py tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_process_recovery.py -q -k postgres -n 2 -x
python -m bandit tldw_Server_API/tests/Collections/test_output_file_process_recovery.py -s B101 -f json -o /private/tmp/task-13153-process-bandit-reviewed.json
```

Logs: `/private/tmp/task-13153-process-sqlite-verified.log`,
`/private/tmp/task-13153-process-pg-reviewed.log`,
`/private/tmp/task-13153-process-negative-control.log`,
`/private/tmp/task-13153-process-bandit-reviewed.json`.

This closes the specifically recorded process-kill, independent-connection,
late-producer and cross-process progress evidence gap. It does not simulate host
power loss, verify network-filesystem semantics, or establish runtime activation
readiness. Immediate post-publication cleanup/background lifecycle integration,
remaining producer/reader/history contracts and rollout gates are still pending;
the full Task 3 checklist and TASK-13153 acceptance criteria are not marked done.

### Task 3b completion: Immediate cleanup under publication exclusion (2026-09-05)

`publish_and_commit` now attempts phase-specific cleanup after every confirmed
commit path, including recovered lost acknowledgements, without releasing the
verified storage directory lock. It reuses the existing recovery interval rather
than introducing a second disposal implementation. Successful create/replace
retires no-history journal rows after file and directory sync; remove releases
filesystem claims while retaining its independent history effect. Unknown commit
outcomes still preserve every file and claim.

Post-commit filesystem/identity failures retain retryable or blocked authority,
respectively. Database failures and failures while recording cleanup status keep
the known logical result successful and log only fixed diagnostic categories.
Cancellation drains cleanup before exclusion is released. Recovery tests now
seed committed-but-unfinished work by interrupting the real DB apply boundary,
not by disabling cleanup on every service fixture. Actual process-kill tests
remain intact; unrelated writer progress now verifies immediate completion.

The initial nine focused SQLite cases failed at the missing cleanup boundary,
then passed with the implementation. Three further cases cover cleanup-status
write failure, unexpected cleanup errors with sanitized logs and successful
later recovery, and cancellation during unlink. Independent checkpoint review
found no actionable correctness/security or test findings. Existing ADR-003
applies; no new architectural boundary or dependency was introduced.

This is still service-level work. Runtime routes, automatic worker registration,
descriptor readers, history delivery, remaining producers and activation are not
enabled. Task 9 owns background lifecycle registration after its prerequisites;
the next implementation checkpoint is Task 4's protected descriptor readers.
TASK-13153 remains In Progress, full-task ACs unchecked, and PR #2903 draft.

Final checkpoint verification: 172 SQLite/non-PostgreSQL cases passed in 26.55
seconds and 129 required PostgreSQL cases passed in 333.70 seconds, for 301
distinct targeted cases with no required-backend skips. Earlier overlapping
runs are not counted again. Ruff, Black, compile and diff checks pass for all
five changed Python files. Bandit reports zero findings and scanner errors on
the production service without exclusions, and on changed tests with only B101
(pytest assertions) excluded. The DB adapter is unchanged in this checkpoint.
No full sweep or Docker provisioning was performed.

Commands after activating the existing Server virtual environment:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_completion.py tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_recovery.py tldw_Server_API/tests/Collections/test_output_file_process_recovery.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_completion.py tldw_Server_API/tests/Collections/test_output_file_operations_storage.py tldw_Server_API/tests/Collections/test_output_file_recovery.py tldw_Server_API/tests/Collections/test_output_file_process_recovery.py -q -k postgres -n 2
```

Logs: `/private/tmp/task-13153-completion-{red,green,expanded}.log`,
`/private/tmp/task-13153-completion-sqlite-final.log`,
`/private/tmp/task-13153-completion-pg.log`,
`/private/tmp/task-13153-completion-bandit.json`, and
`/private/tmp/task-13153-completion-tests-bandit.json`.

### Task 4a: Descriptor response foundation and HTTP characterization (2026-09-05)

Added `OpenedOutputResponse` in `app/services/output_file_response.py`. It takes
ownership of an already-authorized descriptor, captures metadata via fstat and
never transmits or reopens a pathname, including with ASGI pathsend advertised.
It reuses the installed Starlette stat-header, If-Range, range parsing and
multipart helpers, and StreamingResponse disconnect handling. Reads use bounded
64 KiB pread intervals off the event loop, reusing the existing cancellation-drain
helper before closing the descriptor. Construction belongs in the later protected
open worker; the response itself performs no lookup or authorization.

The 14 new compatibility cases per backend exercise real generic download-by-ID,
by-name and HEAD routes with real Collections adapters: media types, filenames,
body/length, ranges, validators, missing/deleted/foreign rows and authentication
before lookup. Current generic HEAD emits only type/length; current GET supports
If-Range but ignores If-None-Match/If-Modified-Since (200, not 304). Task 4b must
preserve inactive route behavior rather than silently introduce a cache policy.
No 304 response path is introduced by this foundation.

The 37 descriptor cases are filesystem/ASGI tests, independent of DB backend.
A negative-control pathname response reproduces recycled bytes after headers;
the new full/single/multipart responses retain the original inode and metadata.
Coverage includes path deletion/reuse, Unicode filename headers, range merging,
empty/bounded reads, changed descriptor offsets, short/read/send errors, HEAD,
400/416, abandoned/replayed responses, nonregular descriptors and failed setup.
Direct asyncio and AnyIO cancellation drain active reads with both ASGI 2.3 and
2.4; disconnect-message and send-error paths both close the descriptor.

Initial verification had 15 characterization/negative-control passes and three
expected failures for the missing descriptor response; the three then passed.
Expanded HTTP comparison caught a missing 416 content type and passed after
reusing PlainTextResponse for that error. A separate failing fstat fault proved
constructor OSError details needed sanitization; fixed-category errors now
preserve descriptor cleanup. Independent review requested failed header-setup
coverage; it was added and follow-up review found no residual findings.

Final verification: 63 descriptor/SQLite compatibility/completion regressions
passed in 31.06 seconds, plus all 14 required PostgreSQL compatibility cases in
107.36 seconds: 77 distinct targeted cases, no required-backend skips. Earlier
overlapping runs are superseded, not counted again. All three changed Python
files pass Ruff, Black, compile and diff checks. Scoped production Bandit reports
zero findings/errors without exclusions; test Bandit is clean with only B101
(pytest assertions) excluded. Existing local PostgreSQL was reused; no Docker
provisioning or full sweep occurred. Existing ADR-003 applies.

Commands after Server virtual-environment activation:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_file_responses.py tldw_Server_API/tests/Collections/test_output_download_compatibility.py tldw_Server_API/tests/Collections/test_output_file_completion.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_download_compatibility.py -q -k postgres -n 2
```

Logs: `/private/tmp/task-13153-response-{red,green,expanded,expanded-green}.log`,
`/private/tmp/task-13153-response-construction-red.log`,
`/private/tmp/task-13153-response-final-verified.log`,
`/private/tmp/task-13153-response-pg.log`,
`/private/tmp/task-13153-response-bandit-final.json`, and
`/private/tmp/task-13153-response-tests-bandit-final.json`.

The Task 4 checklist remains incomplete: protected DB lookup, namespace and
committed-witness checks, generic/Watchlist download and content-loader dispatch,
and their end-to-end races are next. No production caller uses the new response
yet. No runtime activation/background-worker change; TASK-13153 is In Progress
and PR #2903 remains draft with the human-written Change summary gate pending.

### Task 4b: Protected generic download lookup (2026-09-05)

Generic ID/title/HEAD routes now select a protected response when a per-user
storage binding exists. DB-owned snapshots reject unsupported or inconsistent
bindings, wrong structural ownership namespaces and ambiguous/uncommitted
publication evidence. An absent binding permits legacy dispatch only when no
Reading ownership/path or output-operation authority exists. Mutation-worker
health is intentionally not a read prerequisite.

The existing verified storage lock covers current-row lookup and nofollow open
relative to that same directory descriptor. Path resolution does not provision
directories: a missing mount fails closed. Regular single-link files are accepted;
extra links require exact committed publication identity and witness evidence.
The response retains the opened inode across path deletion/reuse. HEAD preserves
the existing type/length-only header policy, and cancellation drains protected
open work before closing an unreturned response.

Fourteen initial real-route tests failed before implementation, then passed.
Expanded tests cover ownership, special files, hardlinks, committed witness
states, current-row deletion/retarget before lock acquisition, root replacement,
HEAD cleanup and direct cancellation. Two additional fault tests exposed a
malformed recorded link-count acceptance and a metadata-error descriptor leak;
requiring recorded nlink=2 and calculating media type before opening fixed both.
Independent follow-up review found no remaining actionable checkpoint findings.

Verification commands after Server virtual-environment activation:

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_output_protected_downloads.py tldw_Server_API/tests/Collections/test_output_download_compatibility.py tldw_Server_API/tests/Collections/test_output_file_responses.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_items_and_outputs_api.py tldw_Server_API/tests/Collections/test_reading_output_updates.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Collections/test_output_protected_downloads.py tldw_Server_API/tests/Collections/test_output_download_compatibility.py -q -k postgres -n 2
```

The first two runs passed 97 cases in 54.12 seconds and 57 cases in 98.69 seconds.
All 60 required PostgreSQL cases passed in 307.48 seconds: 214 distinct targeted
cases overall, with no required-backend skips. Earlier overlapping runs are not
counted again. The already-running local PostgreSQL instance was reused.
Ruff/Black pass for the response service and new tests, and scoped DB Black,
compile and diff checks pass. Whole-file lint is not clean: the DB's nine and
endpoint's one pre-existing Ruff findings match HEAD by code/message. Scoped
production Bandit reports no findings or scanner errors; test Bandit also passes
with only B101 (pytest assertions) excluded.

Logs: `/private/tmp/task-13153-lookup-{red,green,expanded,fault-red,reviewed}.log`,
`/private/tmp/task-13153-lookup-existing.log`,
`/private/tmp/task-13153-lookup-pg.log`,
`/private/tmp/task-13153-lookup-bandit.json`, and
`/private/tmp/task-13153-lookup-tests-bandit.json`.

Existing ADR-003 applies; no new dependency or architecture decision. Task 4c's
Watchlist download/content-loader and sidecar audit is next, not completed here.
The full Task 4 checklist and TASK-13153 ACs remain incomplete. No activation or
background-worker registration, full sweep, Docker provisioning or merge; PR
#2903 remains draft and the human-written Change summary gate remains pending.

### Task 4c: Registered Watchlist output readers and sidecar audit (2026-09-05)

`open_protected_output` now transfers the current DB row and its opened response
together; generic downloads remain a small wrapper over the same primitive.
Watchlist downloads and inline content in list/detail/delivery consumers use this
pair, validate origin/expiry and derive titles/metadata from the same protected
snapshot. All production callers pass their authenticated/authorized Collections
adapter, including delegated delivery retries. Cancellation drains pending reads
before closing, and response construction/validation failures close the descriptor.

MP3 retains descriptor streaming and range/validator semantics. Markdown/HTML
retain their existing full-text response policy (no ranges/validators), UTF-8 and
universal-newline handling. Activated text materialization is capped at 8 MiB,
returning `413 output_content_too_large` before reading larger files; this is a
bounded-memory safeguard for text DTO/download consumers, not a generic-download
size limit. Inactive text behavior is unchanged. The content helper always closes
the descriptor, including limit rejection and cancellation.

File-read audit of `watchlists.py`:

- `_load_output_content`, `_row_to_output` and `download_output`: registered output
  bytes now pass through the shared protected opener. Legacy pathname reads remain
  reachable only for genuinely inactive storage.
- `_load_report_snapshot_for_user`, reached by evidence/readiness: metadata-only
  JSON sidecars are **not registered artifacts**. Activated reads fail closed with
  sanitized 503, before pathname resolution. Inactive snapshot success and missing/
  invalid-path behavior are preserved. Task 7 explicitly owns the provenance and
  producer/reconciliation prerequisite; no sidecar ownership was guessed here.
- `_read_log_chunk`, `_read_log_tail` and run-detail `read_text`: run logs resolved
  under the user's log scope, not output artifacts; unchanged.
- Snapshot/output `write_text` and cleanup unlinks are producer work, not silently
  migrated here. No template/config file reads were classified as output bytes.

Initial tests produced 18 expected protected-reader failures and six HTTP
characterization passes. Additional failures established the sidecar guard and
bounded-text requirement. Edge tests caught the missing-download error category.
Independent review found stale recipient metadata in delivery retry and changed
inactive evidence error translation. Three real-Collections regressions reproduced
old-recipient delivery, a removed plan still sending, and invalid-path 400 instead
of legacy 404. Delivery now uses/revalidates the metadata accompanying protected
content; evidence propagates only the fixed protected-storage 503. Follow-up
review found no remaining actionable checkpoint issue.

Existing ADR-003 applies. Task 4's full reader rollout remains incomplete until
the sidecar provenance prerequisite is fulfilled. The next implementation unit is
Task 5's original-instance history delivery; Task 7/9 must not activate around the
recorded sidecar blocker. TASK-13153 remains In Progress, full-task ACs unchecked,
and PR #2903 stays draft with its human-written Change summary gate pending.

Checkpoint verification commands (after activating the existing Server venv):

```bash
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Watchlists/test_output_protected_readers.py tldw_Server_API/tests/Collections/test_output_protected_downloads.py tldw_Server_API/tests/Collections/test_output_download_compatibility.py tldw_Server_API/tests/Collections/test_output_file_responses.py -q -k 'not postgres'
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Watchlists/test_output_protected_readers.py tldw_Server_API/tests/Collections/test_output_protected_downloads.py tldw_Server_API/tests/Collections/test_output_download_compatibility.py -q -k postgres -n 2
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Watchlists/test_output_protected_readers.py -q -k 'postgres and invalid_path'
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py -q -k 'output or evidence or readiness or retry_delivery'
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q -k retry_run_delivery
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py -q
```

The final local combined run passed 137 cases in 86.24 seconds. Existing
Watchlist output-route checks passed 10 in 95.78 seconds; existing retry checks
passed three in 2.62 seconds; full-app report/evidence checks passed five in 57.02
seconds. The final evidence-error correction passed its targeted PostgreSQL
regression in 31.03 seconds. Earlier RED/overlapping runs are not counted again.
The broader PostgreSQL run, whose workers loaded code before that final error
translation patch, finished with 99 passes and that one known failure in 573.35
seconds. The fresh one-case run above replaces that failed result; it is not an
additional distinct case. Overall coverage is 155 SQLite/non-backend cases and
100 required PostgreSQL cases (255 distinct passing cases across the final runs),
with no required-backend skips. The broad PostgreSQL invocation itself is not
reported as a clean run. No Docker provisioning or full sweep was performed.

Service/new-test Ruff and Black pass; scoped Black for modified Watchlist lines,
compile and diff checks pass. Watchlist whole-file Ruff has six pre-existing
findings, unchanged by code/message against HEAD; it is not claimed clean.
Production Bandit has no findings/errors without exclusions, and test Bandit
passes with only B101 (pytest assertions) excluded. No dependency was added.

Logs: `/private/tmp/task-13153-watchlist-{red,bounds-red,green,edges,review-red}.log`,
`/private/tmp/task-13153-watchlist-local-verified.log`,
`/private/tmp/task-13153-watchlist-pg.log`,
`/private/tmp/task-13153-watchlist-pg-evidence-final.log`,
`/private/tmp/task-13153-watchlist-existing.log`,
`/private/tmp/task-13153-watchlist-retry-existing.log`,
`/private/tmp/task-13153-watchlist-reports-existing.log`,
`/private/tmp/task-13153-watchlist-bandit-final.json`, and
`/private/tmp/task-13153-watchlist-tests-bandit.json`.
