# API Boundary Remediation Design

Task: TASK-500
Date: 2026-06-01

## Summary

This design addresses three maintainability findings in `tldw_Server_API`:

- Media update endpoints duplicate durable Media DB update invariants.
- Jobs event endpoints query `job_events` through `JobManager` private storage internals.
- Document workspace endpoints create and migrate storage tables lazily from route handlers.

The selected direction is a repository and migration rewrite for long-term stability. Public HTTP routes, response schemas, authorization behavior, status codes, and client-visible payloads should remain stable unless an implementation slice explicitly identifies and tests a broken behavior.

## Goals

- Make API endpoints thin transport adapters for auth, request parsing, response mapping, and HTTP error translation.
- Move durable storage rules behind owning Media DB and Jobs APIs.
- Remove backend-specific SQL and private helper access from endpoint modules.
- Move document workspace schema ownership into Media DB bootstrap/migrations and repositories.
- Preserve external API compatibility while reshaping internal APIs.

## Non-Goals

- Do not redesign public REST routes or response models.
- Do not introduce a new persistence technology.
- Do not rewrite unrelated Media DB, Jobs, or document workspace behavior.
- Do not collapse all endpoint modules into one service layer.

## Architecture

### Media DB Write API

Add a public Media DB operation for user-facing media item updates, for example `update_media_item` or `apply_media_item_update`. This operation owns the complete durable write choreography:

- Fetch current active/non-trash media state.
- Apply metadata-only updates.
- Detect content-present, content-changed, and identical-content cases.
- Compute and persist content hashes.
- Increment media sync versions with optimistic concurrency.
- Reset derived-state flags such as `chunking_status` and `vector_processing` when content changes.
- Create document versions according to the existing endpoint contract.
- Refresh FTS state when title or content changes.
- Write sync log entries.
- Trigger or report best-effort post-commit hooks for highlight staleness and RAG/vector invalidation.

The endpoint should call this public operation and then return the existing rich media detail response.

### Jobs Event Query API

Extend `JobManager.list_job_events_after` into the authoritative public event-read API. It should support filters needed by current callers:

- `after_id`
- `limit`
- `domain`
- `queue`
- `job_type`
- `job_id`
- `owner_user_id`
- `event_types`

`JobManager` owns postgres/sqlite placeholder differences, selected columns, ordering, bounded limits, row normalization, and connection lifecycle. Endpoints keep authorization decisions and pass only the allowed filters to the manager.

### Document Workspace Repository And Migrations

Move reading progress, annotations, and parsed-reference cache storage into Media DB-owned repositories and schema setup.

The repository surface should cover current endpoint needs:

- Reading progress: get, upsert, delete.
- Annotations: list, create, update, sync, soft delete.
- Parsed-reference cache: get by media/user/parser/content hash and upsert.

Schema creation and upgrades should run through Media DB bootstrap/migration paths, not endpoint `_ensure_*_table` helpers. The bootstrap must be idempotent so old per-user SQLite databases open cleanly.

## Component Data Flow

### Media Update

1. Endpoint validates auth and parses `MediaUpdateRequest`.
2. Endpoint calls the public Media DB update operation.
3. Media DB performs the transaction and returns update metadata.
4. DB-owned post-commit hooks run best-effort or return hook metadata for the endpoint to complete user-scoped invalidation.
5. Endpoint fetches and returns `MediaDetailResponse` through the existing rich detail path.

### Jobs Events

1. Endpoint validates auth and domain/owner scope.
2. Endpoint calls `JobManager.list_job_events_after(...)` with allowed filters.
3. Manager returns normalized event dictionaries.
4. List endpoints map dictionaries into response schemas.
5. SSE endpoints keep the existing streaming loop, cursor advancement, snapshot events, and termination rules while polling through the manager API.

### Document Workspace

1. Media DB initialization ensures document workspace schema exists and applies idempotent upgrades.
2. Endpoint validates media existence and auth as it does today.
3. Endpoint calls document workspace repository methods.
4. Repository owns SQL and row normalization.
5. Endpoint maps repository results into existing response schemas.

## Compatibility And Error Handling

Public API compatibility is mandatory by default.

### Media Update

- Reuse existing domain errors where possible: `InputError`, `ConflictError`, and `DatabaseError`.
- Keep endpoint HTTP mapping via `map_db_error_to_http`.
- Preserve existing missing-media, conflict, and unexpected-error behavior.
- Define and test the identical-content case before implementation changes it. Today the endpoint appears to create a document version whenever `content` is present, even if the content hash is unchanged.
- Best-effort hooks should log and continue, matching current cache/highlight invalidation behavior.

### Jobs Events

- Authorization remains in endpoints.
- Storage filtering moves into `JobManager` only after endpoints compute allowed filter values.
- SSE loops should continue swallowing only the same classes of noncritical read/stream exceptions they currently handle.
- Admin/list endpoints should preserve current error surfacing semantics.

### Document Workspace

- Old user databases without document workspace tables must be upgraded idempotently.
- Missing media remains a 404.
- Storage failures remain 500-level responses.
- Corrupt or missing progress rows should keep the current response behavior unless an implementation slice explicitly changes and tests it.

## Rollout Stages

### Stage 1: Media DB Update Ownership

Move media item update persistence from `app/api/v1/endpoints/media/item.py` into a public Media DB operation. The endpoint becomes a thin adapter.

Acceptance checks:

- Metadata-only, content-changing, identical-content, missing-media, and conflict cases preserve public behavior.
- Content-changing updates reset derived-state flags.
- Document versions, FTS, sync log entries, highlight staleness, and RAG/vector invalidation are covered.
- Endpoint code no longer calls private Media DB helpers such as `_update_fts_media` or `_log_sync_event`.

### Stage 2: Jobs Event Query Ownership

Expand `JobManager.list_job_events_after` and migrate admin, media ingest, audio jobs, and prompt-studio event reads to it.

Acceptance checks:

- All existing event-list and SSE filters are supported.
- SQLite and postgres paths normalize to the same event dictionary shape.
- SSE payloads and cursor behavior remain stable.
- Endpoint code no longer calls `jm._connect()` or `jm._pg_cursor()` for event reads.

### Stage 3: Document Workspace Storage Ownership

Add document workspace repositories and Media DB schema bootstrap/migrations for reading progress, annotations, and parsed-reference cache tables.

Acceptance checks:

- Old DBs without these tables are upgraded idempotently.
- Existing endpoint responses remain stable.
- Repository methods own SQL and row normalization.
- Endpoint code no longer creates or alters document workspace tables directly.

## Testing Strategy

Run focused tests for each stage before broad backend verification.

Stage 1 tests:

- Media endpoint regression tests for response compatibility.
- Media DB unit tests for content update invariants.
- FTS, sync-log, document-version, `chunking_status`, and `vector_processing` assertions.
- Best-effort hook tests for highlight staleness and RAG/vector invalidation.

Stage 2 tests:

- `JobManager.list_job_events_after` filter combinations.
- SQLite path coverage and postgres SQL/behavior coverage where existing fixtures allow.
- Admin jobs, media ingest SSE, audio jobs SSE, and prompt-studio status regressions.

Stage 3 tests:

- Media DB bootstrap against old schemas.
- Reading progress get/upsert/delete regressions.
- Annotation list/create/update/sync/delete regressions.
- Parsed-reference cache get/upsert regressions.

Final verification:

- Run focused pytest modules for touched Media, DB_Management, Jobs, AudioJobs, Prompt Studio, and document workspace tests.
- Run Bandit on touched backend paths.
- Confirm no endpoint-owned private storage calls or lazy table DDL remain in the targeted paths.

## Risks And Mitigations

- Risk: moving endpoint SQL into repositories changes subtle behavior.
  Mitigation: write endpoint regression tests before moving logic.
- Risk: old per-user SQLite databases miss new schema bootstrap.
  Mitigation: add old-schema fixture tests before removing lazy endpoint DDL.
- Risk: Jobs event filters diverge between SQLite and postgres.
  Mitigation: centralize SQL generation in `JobManager` and test normalized event output.
- Risk: content update post-commit hooks become hidden side effects.
  Mitigation: keep hooks explicit in the Media DB update result or a named helper with tests.

## Definition Of Done

- Public HTTP contracts remain stable unless explicitly documented and approved.
- Endpoints no longer own targeted durable storage rules.
- Backend-specific Jobs event SQL lives behind `JobManager`.
- Document workspace table creation and migration are removed from route handlers.
- Regression tests cover the identified drift risks.
- Bandit passes for touched backend paths or any non-code skip is documented.
