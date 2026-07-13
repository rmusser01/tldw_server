# YouTube Playlist Ingest Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the owner-scoped backend contracts that inspect a YouTube playlist completely, materialize selected occurrences, and track every selected occurrence through one ingest run without allowing an opaque playlist job.

**Architecture:** Reuse the existing Jobs SQLite/PostgreSQL database, leases, events, and media-ingest worker. Add focused playlist-ingest tables and a repository beside Jobs, keep yt-dlp in a bounded child process, and let a small service translate run occurrences into existing media jobs or terminal duplicate actions. Keep the synchronous preflight endpoint only as a compatibility surface.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL Jobs backends, yt-dlp, existing `JobManager`, Media DB and Collections DB abstractions, pytest, Hypothesis, Bandit.

**Backlog:** `TASK-12110`

**Spec:** `Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md`

---

## File map

**Create**

- `tldw_Server_API/app/api/v1/schemas/media_playlist_ingest.py` — version-2 preflight, materialization, run, item, event, retry, and structured submission models.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_store.py` — owner-filtered persistence, cursor encoding, expiry cleanup, and atomic state transitions over the Jobs database.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight_runner.py` — bounded child-process extraction and cancellation/timeout termination.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_service.py` — preflight scheduling, materialization, run validation, duplicate-action resolution, and job/run reconciliation.
- `tldw_Server_API/app/api/v1/endpoints/media/playlist_ingest.py` — version-2 HTTP and SSE routes.
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py` — SQLite repository and state-machine tests.
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py` — orchestration, duplicate, idempotency, and metadata-patch tests.
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py` — route, auth, cursor, and error-contract tests.
- `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_store_postgres.py` — repository parity against the existing isolated PostgreSQL fixture.
- `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_run_workflow.py` — real Jobs DB/worker workflow with media processing faked.
- `tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py` — pagination, identity, action-resolution, and retry invariants.

**Modify**

- `tldw_Server_API/app/core/Jobs/migrations.py` — SQLite tables/indexes for preflights, items, materializations, runs, run items, and run events.
- `tldw_Server_API/app/core/Jobs/pg_migrations.py` — PostgreSQL-equivalent schema and indexes.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py` — configured-limit-plus-one extraction and availability/occurrence normalization.
- `tldw_Server_API/app/api/v1/endpoints/media/__init__.py` — register `playlist_ingest` beside the compatibility preflight router.
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py` — reject playlist candidates; accept aligned run/occurrence/attempt bindings; return structured per-occurrence acceptance.
- `tldw_Server_API/app/services/media_ingest_jobs_worker.py` — handle the internal preflight job and require exactly one media result for a concrete occurrence.
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py` — atomic allowlisted metadata-only patch.
- `tldw_Server_API/app/core/DB_Management/media_db/repositories/media_lookup_repository.py`, `api.py`, and `runtime/query_ops.py` — one owner-database bulk URL lookup shared by preflight enrichment and run validation.
- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py` — bind the metadata-patch method.
- `tldw_Server_API/app/core/DB_Management/Collections_DB.py` — transactional optional collection plus planned-item creation.
- `tldw_Server_API/app/api/v1/endpoints/config_info.py` — advertise contract version/readiness.
- Existing tests under `tldw_Server_API/tests/Jobs/`, `MediaIngestion_NEW/unit/`, `DB_Management/`, and `Media/` — extend migration, worker, collection, compatibility, and capability coverage.

## Stage 1: Durable owner-scoped storage

**Goal:** Establish portable persistence and strict state transitions before adding routes.

**Success Criteria:** Both Jobs backends create the same schema; owner-scoped reads, immutable ordering, expiry, and cursors are deterministic.

**Tests:** SQLite/PostgreSQL migration tests, repository unit tests, Hypothesis pagination/identity tests.

**Status:** Complete

### Task 1: Add contract models and Jobs migrations

- [x] **Step 1: Write failing migration and schema tests**

Add table assertions to `tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py` and `test_jobs_migrations_postgres.py`. Add model-validation tests to `test_playlist_ingest_store.py` proving duplicate policies are explicit and `file_reattach_required` is rejected as a server state.

```python
EXPECTED_PLAYLIST_TABLES = {
    "playlist_preflights",
    "playlist_preflight_items",
    "playlist_materializations",
    "playlist_materialization_items",
    "media_ingest_runs",
    "media_ingest_run_items",
    "media_ingest_run_events",
}

def test_run_state_rejects_client_only_file_reattach_state():
    with pytest.raises(ValidationError):
        RunItemSnapshot(occurrence_id="occ-1", ordinal=1, state="file_reattach_required")
```

- [x] **Step 2: Run the tests and verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py -q`

Expected: FAIL because the tables and `media_playlist_ingest` schemas do not exist.

- [x] **Step 3: Add the minimum portable schema**

Define Pydantic enums/models in `media_playlist_ingest.py`. Add matching `CREATE TABLE IF NOT EXISTS` DDL to both Jobs migration modules. Store bounded display metadata and patches as JSON, but keep owner, status, ordinal, occurrence ID, normalized source ID, job ID, attempt, event ID, and expiry as indexed columns. Add uniqueness constraints for `(preflight_id, ordinal)`, occurrence IDs, `(run_id, occurrence_id)`, and `(run_id, occurrence_id, attempt)` job mappings.

- [x] **Step 4: Run SQLite and PostgreSQL migration tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py -q`

Expected: PASS (PostgreSQL test skips only through its existing fixture when unavailable).

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/media_playlist_ingest.py tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py tldw_Server_API/tests/Jobs tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py
git commit -m "feat: add playlist ingest persistence schema (TASK-12110)"
```

### Task 2: Implement the focused repository and cursor contract

- [x] **Step 1: Write failing repository tests**

Cover creation, owner isolation, ready-only materialization, selected occurrence copying, immutable ordering, cursor tampering, expiry, event replay, compare-and-set state transitions, and one-query Media DB lookup of normalized URL candidates.

```python
def test_materialization_copies_identity_but_not_review_policy(store):
    ready = store.seed_ready_preflight(owner_id="1", item_count=2)
    materialized = store.create_materialization(
        owner_id="1", preflight_id=ready.preflight_id, occurrence_ids=["occ-2"]
    )
    item = store.list_materialization_items("1", materialized.id)[0]
    assert item.occurrence_id == "occ-2"
    assert item.source_url.endswith("v=2")
    assert "duplicate_policy" not in item.display_metadata
    assert "metadata_patch" not in item.display_metadata
```

- [x] **Step 2: Run the repository tests and verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py tldw_Server_API/tests/MediaDB2/test_dedupe_url_normalization.py -q`

Expected: FAIL because `PlaylistIngestStore` is missing.

- [x] **Step 3: Implement `PlaylistIngestStore`**

Accept the existing `JobManager` and use its package-owned `_connect`/`_pg_cursor` helpers so backend/DSN selection is not duplicated. The playlist store itself must not open the Media DB or AuthNZ DB. Every public method requires `owner_user_id`. Use one transaction for snapshot replacement, materialization copying, run creation, event append/version bump, and cleanup. Sign opaque cursors with `AuthNZ.crypto_utils.derive_hmac_key()` and bind them to owner, resource, ordering, and last ordinal.

In the same red/green cycle, add `get_media_by_urls(urls)` through `media_lookup_repository.py` → `media_db/api.py` → `runtime/query_ops.py` → `MediaDatabase`. Normalize all URL candidates and execute one parameterized query against the already owner-specific Media DB. This helper is shared by preflight duplicate enrichment and Start Processing validation.

- [x] **Step 4: Add and run property tests**

For arbitrary unique occurrence lists and page sizes, concatenated pages must equal the source ordering exactly once. Invalid owner/cursor pairs must never disclose whether a resource exists.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_store_postgres.py tldw_Server_API/tests/MediaDB2/test_dedupe_url_normalization.py tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_store.py tldw_Server_API/app/core/DB_Management/media_db/repositories/media_lookup_repository.py tldw_Server_API/app/core/DB_Management/media_db/api.py tldw_Server_API/app/core/DB_Management/media_db/runtime/query_ops.py tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py tldw_Server_API/tests/MediaDB2/test_dedupe_url_normalization.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_store_postgres.py tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py
git commit -m "feat: persist playlist ingest resources (TASK-12110)"
```

## Stage 2: Asynchronous inspection and materialization

**Goal:** Replace optional synchronous preview with a bounded, cancellable version-2 resource.

**Success Criteria:** The API returns 202, worker extraction is terminable, complete snapshots paginate, oversize playlists block, and materialization survives preflight expiry.

**Tests:** Runner process tests, worker tests, endpoint tests, configured-limit-plus-one tests.

**Status:** In Progress

### Task 3: Add the bounded preflight child-process runner

- [x] **Step 1: Write failing runner tests**

Test successful normalized extraction, child timeout termination, cancellation termination, malformed child payload, `configured_limit + 1` producing `playlist_too_large` without a partial-ready snapshot, owner-library duplicates being marked `duplicate_existing`/counted/deselected before ready, and a failed library lookup producing `unknown` evidence plus a warning rather than falsely reporting `new`.

- [x] **Step 2: Verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py -k "process or configured_limit" -q`

Expected: FAIL because the process runner and hard-ceiling behavior are absent.

- [x] **Step 3: Implement the runner and tighten extraction**

Use `multiprocessing.get_context("spawn")`, one result pipe, and a small polling loop. On timeout/cancel, call `terminate()`, then `kill()` only if the child remains alive after a bounded join. The child calls the existing yt-dlp normalizer. Request at most `limit + 1`, return unavailable entries visibly, generate opaque occurrence IDs server-side, and never return truncation as success.

- [x] **Step 4: Extend the media worker**

Handle `job_type == "playlist_preflight"` before `media_ingest_item`; call the runner, open the owner's Media DB, bulk-resolve extracted URLs with `get_media_by_urls`, and merge `duplicate_existing` evidence with in-snapshot duplicates before atomically storing the complete snapshot. Recompute duplicate/selected counts from the enriched items. A library lookup failure marks otherwise-new evidence `unknown` and adds a typed warning; extraction, capacity, or snapshot-write failure blocks the resource with a safe error code. Keep Jobs leases as the only cross-process extraction claim.

- [x] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight_runner.py tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/unit
git commit -m "feat: run playlist inspection as bounded jobs (TASK-12110)"
```

### Task 4: Expose preflight, pages, cancellation, and materialization

- [x] **Step 1: Write failing route tests**

Cover POST 202, summary polling, paginated pages, delete/cancel, ready-only materialization, cross-owner 404-equivalent behavior, `preflight_busy`, expiry, and sanitized errors.

- [x] **Step 2: Verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py -k preflight -q`

Expected: FAIL because `/playlist-preflights` routes are missing.

- [x] **Step 3: Implement the routes and service methods**

`POST /playlist-preflights` validates the trusted YouTube boundary, reserves capacity transactionally, creates the resource, and enqueues the internal job. Item routes return bounded pages. Materialization accepts only selected occurrence IDs and returns compact identity records; it never accepts policies or patches. `DELETE` requests job cancellation and expires the resource.

- [x] **Step 4: Register the router and keep compatibility explicit**

Add `playlist_ingest` to `_MEDIA_ENDPOINT_MODULES`. Leave `/playlists/preflight` working for older clients, but add compatibility tests proving version-2 clients are advertised separately.

- [x] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/api/v1/endpoints/media/playlist_ingest.py tldw_Server_API/app/api/v1/endpoints/media/__init__.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_service.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py
git commit -m "feat: expose playlist preflight resources (TASK-12110)"
```

## Stage 3: Run creation and duplicate actions

**Goal:** Resolve every selected occurrence exactly once before any media job is accepted.

**Success Criteria:** Mixed materialized URLs/direct URLs/file stubs validate atomically; stale duplicate choices return `review_required`; non-processing policies terminate without jobs.

**Tests:** Service tests, Media DB patch tests, Collections DB transaction tests, action-resolution properties.

**Status:** Not Started

### Task 5: Validate and create ingest runs atomically

- [ ] **Step 1: Write failing service tests**

Test mixed input unions, playlist rejection in `direct_url`, expired materialization, unique occurrence IDs, file `awaiting_upload`, missing/extra review overrides, and a fresh duplicate appearing after Review.

```python
def test_fresh_duplicate_requires_review_without_side_effects(service, media_db):
    materialized = service.seed_materialized_video("youtube:video:abc")
    media_db.seed_existing(url="https://www.youtube.com/watch?v=abc")
    with pytest.raises(ReviewRequiredError) as exc:
        service.create_run(inputs=[materialized], review_overrides={})
    assert exc.value.items[0].occurrence_id == materialized.occurrence_id
    assert service.count_runs() == 0
    assert service.count_media_jobs() == 0
```

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py -k create_run -q`

Expected: FAIL because run creation is absent.

- [ ] **Step 3: Implement validation and fresh duplicate lookup**

Resolve materialized identity server-side, canonicalize non-playlist direct URLs, and create file stubs without bytes. Reuse the Stage 1 `get_media_by_urls(urls)` owner-database bulk lookup to refresh library evidence. Resolve in-run repeats by normalized source ID plus occurrence order. Validate Review overrides only after this fresh evidence. Return structured `review_required` before opening the run transaction when choices are stale.

- [ ] **Step 4: Persist the run and initial events in one transaction**

Store immutable ordinal/identity plus mutable state/outcome/attempt. Initial action is `ingest`, `overwrite`, `skip`, `include_existing`, or `update_metadata_only`; do not create jobs yet. Append one initial event per occurrence and one summary version bump.

- [ ] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/api/v1/schemas/media_playlist_ingest.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_service.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py
git commit -m "feat: validate playlist ingest runs (TASK-12110)"
```

### Task 6: Execute non-processing duplicate policies and optional collection planning

- [ ] **Step 1: Write failing atomicity tests**

Add tests proving `skip`, `include_existing`, and `update_metadata_only` create zero media jobs and distinct outcomes. Test allowlisted title/author/keyword union, empty/forbidden patches, optimistic conflict rollback, and collection-plus-items rollback.

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py -k "metadata or collection or duplicate" -q`

Expected: FAIL because the atomic patch and action executor are missing.

- [ ] **Step 3: Add the narrow Media DB patch**

Implement `apply_media_metadata_patch(media_id, title=None, author=None, keywords_add=())` in `media_item_update_ops.py`. In one Media DB transaction, fetch the active row/version and current keywords, validate non-empty values, union keywords case-insensitively, update title/author/version, update FTS when title changes, update keyword links with the same connection, and log one sync event. Do not expose content/type/analysis mutation.

- [ ] **Step 4: Add transactional collection planning and action execution**

Add one `CollectionsDatabase.create_media_collection_with_items(...)` method using its existing transaction helpers. In the service, execute non-processing actions after the run transaction exists: set terminal outcome/media ID, resolve optional planned items, and append events. On error, use `metadata_update_failed`; never silently fall back to a media job.

- [ ] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_service.py tldw_Server_API/tests
git commit -m "feat: resolve playlist duplicate actions (TASK-12110)"
```

## Stage 4: Processing jobs, status, cancellation, and retry

**Goal:** Bind processing-required occurrences to existing Jobs with durable run reconciliation.

**Success Criteria:** One accepted job per occurrence attempt, structured partial acceptance, dynamic run events, real cancellation, and reconciled retry.

**Tests:** Endpoint/worker tests, event replay tests, cancellation races, integration workflow.

**Status:** Not Started

### Task 7: Tighten media-job submission and worker boundaries

- [ ] **Step 1: Write failing endpoint and worker tests**

Cover aligned URL/file occurrence arrays, length mismatch, owner/run/state validation, derived idempotency, repeated ambiguous submit, opaque playlist 422, client URL mismatch, canonical server-authoritative payload, and worker rejection when processing returns zero or multiple items.

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py -q`

Expected: FAIL on the new occurrence contract.

- [ ] **Step 3: Extend `/media/ingest/jobs` minimally**

Accept `run_id`, aligned `occurrence_ids`, `attempts`, and optional planned IDs for URLs; use equivalent `file_*` arrays for uploads. Derive the Jobs idempotency key as an HMAC/hash over authenticated owner, run, occurrence, and attempt. Validate run membership/state before staging files. Resolve the authoritative concrete URL from the run item; reject any non-matching client URL with `occurrence_source_mismatch`, then write only the stored URL into the job payload. Return one structured accepted/rejected record per occurrence while retaining legacy response fields during deprecation.

- [ ] **Step 4: Enforce one concrete worker result**

Reject `classify_playlist_url(source).is_playlist` before job creation and again defensively in the worker. Replace `results[0]` projection with a length check: exactly one dict is required for a media occurrence. Include `run_id`, `occurrence_id`, and attempt in the payload/result for reconciliation.

- [ ] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py tldw_Server_API/tests/Media/test_media_ingest_jobs_endpoint_sanitization.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/unit tldw_Server_API/tests/Media/test_media_ingest_jobs_endpoint_sanitization.py
git commit -m "feat: bind media jobs to ingest occurrences (TASK-12110)"
```

### Task 8: Add run routes, reconciliation, event replay, cancellation, and retry

- [ ] **Step 1: Write failing run-route tests**

Cover run POST, summary, paginated items, SSE initial snapshot/replay/resync, a stream-only client observing job progress without polling, later chunk jobs appearing in the same stream, occurrence-scoped cancellation of unsent/accepted work, whole-run cancellation, `status_unavailable`, and retry after media reconciliation.

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py -k run -q`

Expected: FAIL because run routes are missing.

- [ ] **Step 3: Implement run routes and reconciliation**

Run POST calls the service, executes terminal duplicate actions, and returns processing occurrences ready for bounded client chunks. Summary/items reconcile Jobs by stored mappings and append occurrence events only on actual changes. On every SSE cycle, call `reconcile_run_jobs(owner_id, run_id)` before reading new run events; it must query the current run/job mappings dynamically, compare state/progress/result, and transactionally append occurrence events/version changes. SSE then reads events by monotonically increasing ID; expired replay emits `resync_required`.

Define `POST /ingest/runs/{run_id}/cancel` with optional body `{ occurrence_ids?: string[], reason?: string }`. Before a run exists, cancellation is client-local. Once a run exists, supplied occurrence IDs terminalize unsent items and cancel their accepted jobs; an omitted list cancels the whole run. Repeated requests are idempotent and completion may win the race.

- [ ] **Step 4: Implement retry with media-first reconciliation**

Before incrementing attempt, query the current user's Media DB by normalized URL and planned item. If media exists, resolve terminally without a new job. Otherwise increment once with compare-and-set, clear the prior job mapping, and return the occurrence for resubmission.

- [ ] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_run_workflow.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/api/v1/endpoints/media/playlist_ingest.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_service.py tldw_Server_API/tests/MediaIngestion_NEW
git commit -m "feat: track playlist ingest runs (TASK-12110)"
```

## Stage 5: Capability rollout and release gates

**Goal:** Make version negotiation, cleanup, compatibility, security, and operational limits explicit.

**Success Criteria:** Version-2 capability is truthful, expired resources clean up, compatibility remains covered, all focused gates and Bandit pass.

**Tests:** Config capability tests, cleanup tests, owner isolation, `/process-videos` compatibility, full focused suite.

**Status:** Not Started

### Task 9: Finish rollout, cleanup, and verification

- [ ] **Step 1: Add failing capability and cleanup tests**

Assert `mediaPlaylistIngestContractVersion == 2` only when preflight/run routes and worker readiness are enabled. Test bounded cleanup of expired preflights/materializations/runs/events and lease release after worker crash. Keep `/process-videos` multi-result playlist behavior covered.

- [ ] **Step 2: Implement capability and cleanup wiring**

Update `config_info.py` with granular flags. Invoke bounded cleanup from preflight/run mutations and worker startup; do not add a new scheduler. Emit only counts/error codes in logs and metrics, never full playlist URLs.

- [ ] **Step 3: Run the complete focused backend gate**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_store_postgres.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_service.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_run_workflow.py tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py -q`

Expected: PASS with only fixture-declared PostgreSQL skips.

- [ ] **Step 4: Run security and diff gates**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/media/playlist_ingest.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_store.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight_runner.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_service.py tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/services/media_ingest_jobs_worker.py -f json -o /tmp/bandit_task_12110.json`

Run: `git diff --check`

Expected: Bandit exits 0 with no new findings; diff check exits 0.

- [ ] **Step 5: Update docs/task and commit**

Record exact test counts, PostgreSQL skips, Bandit result, touched files, and compatibility result in `TASK-12110`.

```bash
git add tldw_Server_API Docs backlog/tasks
git commit -m "test: verify playlist ingest backend contract (TASK-12110)"
```
