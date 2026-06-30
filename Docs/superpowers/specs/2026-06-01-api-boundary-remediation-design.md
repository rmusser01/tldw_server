# API Boundary Remediation Design

Task: TASK-500
Date: 2026-06-01

## Summary

This design addresses five maintainability findings in `tldw_Server_API`:

- Minimal-test router specs duplicate production router metadata and can bypass route policy.
- Media update endpoints duplicate durable Media DB update invariants.
- Jobs event endpoints query `job_events` through `JobManager` private storage internals.
- Document workspace endpoints create and migrate storage tables lazily from route handlers.
- Prototype promotion review splits authorization and state transitions across endpoint and service layers.

The selected direction is a repository, service, and registry rewrite for long-term stability. Public HTTP routes, response schemas, authorization behavior, status codes, and client-visible payloads should remain stable unless an implementation slice explicitly identifies and tests a broken behavior.

## Goals

- Make API endpoints thin transport adapters for auth, request parsing, response mapping, and HTTP error translation.
- Define router metadata once and derive minimal-test router sets from production `RouterSpec` definitions.
- Move durable storage rules behind owning Media DB and Jobs APIs.
- Remove backend-specific SQL and private helper access from endpoint modules.
- Move document workspace schema ownership into Media DB bootstrap/migrations and repositories.
- Keep prototype promotion authorization and review state transitions inside public service APIs.
- Preserve external API compatibility while reshaping internal APIs.

## Non-Goals

- Do not redesign public REST routes or response models.
- Do not introduce a new persistence technology.
- Do not rewrite unrelated Media DB, Jobs, router, prototype workspace, or document workspace behavior.
- Do not collapse all endpoint modules into one service layer.
- Do not address worker lifecycle state or WorkerRegistry consolidation in this series; that cleanup is a separate task.

## Architecture

### Router Spec Metadata Ownership

Make the production router group specs the authoritative source for router metadata. Minimal-test routing should select from, filter, or explicitly override those shared `RouterSpec` objects instead of rebuilding partial `ImportedRouterSpec` copies.

The derived minimal router set should preserve production metadata by default:

- `route_key`
- `default_stable`
- `module_path`
- `attribute_name`
- `prefix`
- `tags`

Minimal-test-specific behavior should be represented as explicit overrides near the minimal router group code. Test-only routes with no production analogue may still define local specs, but they should make the routing-policy choice obvious by setting `route_key`/`default_stable` intentionally or documenting why the route is always enabled.

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
- Run DB-local best-effort post-commit hooks, such as collection highlight staleness.
- Return explicit effect metadata for endpoint-scoped/user-scoped invalidation, such as RAG cache invalidation.

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

The canonical internal event dictionary returned by `JobManager` should include raw event storage fields:

- `id`
- `event_type`
- `attrs_json`
- `job_id`
- `domain`
- `queue`
- `job_type`
- `owner_user_id`
- `request_id`
- `trace_id`
- `created_at`

Endpoints remain responsible for endpoint-specific response mapping, including parsing `attrs_json` into `attrs` for SSE payloads where that is the existing client contract.

### Document Workspace Repository And Migrations

Move reading progress, annotations, and parsed-reference cache storage into Media DB-owned repositories and schema setup.

The repository surface should cover current endpoint needs:

- Reading progress: get, upsert, delete.
- Annotations: list, create, update, sync, soft delete.
- Parsed-reference cache: get by media/user/parser/content hash and upsert.

Schema creation and upgrades should run through Media DB bootstrap/versioned migration paths, not endpoint `_ensure_*_table` helpers. The bootstrap must be idempotent so old per-user SQLite databases open cleanly. It should not become ad hoc DDL on every request or every DB open; implementation should attach these tables to the existing schema initialization/migration mechanism used by Media DB.

### Prototype Promotion Review Service

Add a public prototype workspace service method for the complete review decision, for example `review_promotion_request(...)`. The method should own:

- Loading the workspace and promotion request.
- Applying promoter authorization rules through public service behavior rather than endpoint access to `_is_promoter`.
- Rejecting a pending request, including status, reviewer, timestamp, reason, and audit behavior.
- Approving a pending request by delegating to the existing promotion workflow or a shared internal helper.
- Returning a domain result that the endpoint can map into the existing `PrototypePromotionReviewResponse`.

The endpoint should remain responsible for HTTP auth dependency resolution, request parsing, and exception-to-HTTP translation only.

## Component Data Flow

### Media Update

1. Endpoint validates auth and parses `MediaUpdateRequest`.
2. Endpoint calls the public Media DB update operation.
3. Media DB performs the transaction and returns update metadata.
4. Media DB runs DB-local post-commit hooks best-effort and returns explicit effect metadata for endpoint/user-scoped invalidation.
5. Endpoint fetches and returns `MediaDetailResponse` through the existing rich detail path.

### Jobs Events

1. Endpoint validates auth and domain/owner scope.
2. Endpoint calls `JobManager.list_job_events_after(...)` with allowed filters.
3. Manager returns canonical event dictionaries with raw `attrs_json`.
4. List endpoints map dictionaries into response schemas.
5. SSE endpoints keep the existing streaming loop, cursor advancement, snapshot events, and termination rules while polling through the manager API and preserving existing parsed `attrs` payloads.

### Document Workspace

1. Media DB initialization ensures document workspace schema exists and applies idempotent upgrades.
2. Endpoint validates media existence and auth as it does today.
3. Endpoint calls document workspace repository methods.
4. Repository owns SQL and row normalization.
5. Endpoint maps repository results into existing response schemas.

### Router Registration

1. Production router group modules expose canonical `RouterSpec` definitions.
2. Minimal-test router group code requests named specs or filters canonical specs by route identity.
3. Minimal-specific overrides are applied through a small helper that preserves unspecified metadata.
4. `register_router_specs` receives specs with production `route_key` and `default_stable` metadata intact.

### Prototype Promotion Review

1. Endpoint validates auth and parses the review request.
2. Endpoint calls the public service review method with workspace, request, reviewer, decision, and reason data.
3. Service validates promoter authorization and request state.
4. Service applies reject or promote behavior through one domain boundary.
5. Endpoint maps the service result into the existing response schema.

## Compatibility And Error Handling

Public API compatibility is mandatory by default.

### Router Specs

- Minimal-test mode should keep the same route inclusion set unless an implementation slice explicitly documents a route that was incorrectly included.
- Existing route policy semantics should become closer to production, not looser.
- Production `route_key` and `default_stable` values should flow into minimal specs by default.
- Any intentional minimal-only override must be named and covered by a focused test.

### Media Update

- Reuse existing domain errors where possible: `InputError`, `ConflictError`, and `DatabaseError`.
- Keep endpoint HTTP mapping via `map_db_error_to_http`.
- Preserve existing missing-media, conflict, and unexpected-error behavior.
- Preserve the current identical-content contract by default: when a non-null `content` field is present in the request, create a document version with the provided content/prompt/analysis even if the content hash is unchanged. Only change this if a regression test proves the behavior is broken and the behavior change is explicitly approved.
- Best-effort hooks should log and continue, matching current cache/highlight invalidation behavior.
- Media DB should own DB-local side effects. Endpoint code should own request/user-scoped RAG cache invalidation using the effect metadata returned by the Media DB update operation.

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

### Prototype Promotion Review

- Preserve existing approve and reject response payload shapes.
- Preserve existing authorization failure, missing workspace, missing request, stale request, and validation failure status codes.
- The public service method should expose domain errors that the endpoint can map using current endpoint conventions.
- Reject and approve paths should use the same promoter authorization rule source and compatible audit/state-transition behavior.

## Rollout Stages

### Stage 1: Router Spec Metadata Single Source

Derive minimal-test router specs from production router group specs and keep minimal-only overrides explicit.

Acceptance checks:

- Minimal-test route inclusion remains stable for the intended test app.
- Shared production metadata, especially `route_key` and `default_stable`, is preserved by default.
- Minimal router group code no longer duplicates production router metadata for core, content, and admin routes.
- Route-policy tests cover at least one route whose previous minimal spec omitted `route_key`.

### Stage 2: Media DB Update Ownership

Move media item update persistence from `app/api/v1/endpoints/media/item.py` into a public Media DB operation. The endpoint becomes a thin adapter.

Acceptance checks:

- Metadata-only, content-changing, identical-content, missing-media, and conflict cases preserve public behavior.
- Content-changing updates reset derived-state flags.
- Document versions, FTS, sync log entries, highlight staleness, and RAG/vector invalidation are covered.
- Endpoint code no longer calls private Media DB helpers such as `_update_fts_media` or `_log_sync_event`.

### Stage 3: Jobs Event Query Ownership

Expand `JobManager.list_job_events_after` and migrate admin, media ingest, audio jobs, and prompt-studio event reads to it.

Acceptance checks:

- All existing event-list and SSE filters are supported.
- SQLite and postgres paths normalize to the same event dictionary shape.
- SSE payloads and cursor behavior remain stable.
- Endpoint code no longer calls `jm._connect()` or `jm._pg_cursor()` for event reads.

### Stage 4: Document Workspace Storage Ownership

Add document workspace repositories and Media DB schema bootstrap/migrations for reading progress, annotations, and parsed-reference cache tables.

Acceptance checks:

- Old DBs without these tables are upgraded idempotently.
- Existing endpoint responses remain stable.
- Repository methods own SQL and row normalization.
- Endpoint code no longer creates or alters document workspace tables directly.

### Stage 5: Prototype Promotion Review Ownership

Move promotion review authorization and state transitions behind a public prototype workspace service method.

Acceptance checks:

- Owner, designated promoter, non-promoter, approve, reject, stale request, and missing request cases preserve public behavior.
- Endpoint code no longer calls `service._is_promoter()` or `repo.update_promotion_request(...)`.
- Reject and approve paths share the same service-owned authorization and state-transition boundary.
- Existing promotion workflow behavior remains covered by endpoint or service regression tests.

## Testing Strategy

Run focused tests for each stage before broad backend verification.

Stage 1 tests:

- Router group unit tests for canonical spec selection and metadata preservation.
- Minimal-test app registration tests covering route policy for a route with `route_key`.
- Regression check that intended minimal route names/prefixes/tags remain present.

Stage 2 tests:

- Media endpoint regression tests for response compatibility.
- Media DB unit tests for content update invariants.
- FTS, sync-log, document-version, `chunking_status`, and `vector_processing` assertions.
- Best-effort hook tests for highlight staleness and RAG/vector invalidation.

Stage 3 tests:

- `JobManager.list_job_events_after` filter combinations.
- SQLite path coverage and postgres SQL/behavior coverage where existing fixtures allow.
- Admin jobs, media ingest SSE, audio jobs SSE, and prompt-studio status regressions.

Stage 4 tests:

- Media DB bootstrap against old schemas.
- Reading progress get/upsert/delete regressions.
- Annotation list/create/update/sync/delete regressions.
- Parsed-reference cache get/upsert regressions.

Stage 5 tests:

- Endpoint regressions for approve and reject decisions.
- Service unit tests for promoter authorization and request state transitions.
- Negative tests for non-promoter review attempts and missing/stale promotion requests.

Final verification:

- Run focused pytest modules for touched router groups, Media, DB_Management, Jobs, AudioJobs, Prompt Studio, document workspace, and prototype workspace tests.
- Run Bandit on touched backend paths.
- Confirm no endpoint-owned private storage calls or lazy table DDL remain in the targeted paths.
- Run explicit smoke checks for the targeted endpoints, such as:
  - `rg -n "ImportedRouterSpec\\(" tldw_Server_API/app/api/v1/router_groups/minimal.py`
  - `rg -n "_update_fts_media|_log_sync_event" tldw_Server_API/app/api/v1/endpoints/media/item.py`
  - `rg -n "jm\\._connect|jm\\._pg_cursor" tldw_Server_API/app/api/v1/endpoints/jobs_admin.py tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py`
  - `rg -n "CREATE TABLE IF NOT EXISTS|ALTER TABLE|PRAGMA table_info" tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py tldw_Server_API/app/api/v1/endpoints/media/document_references.py`
  - `rg -n "_is_promoter|repo\\.update_promotion_request" tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`

## Risks And Mitigations

- Risk: moving endpoint SQL into repositories changes subtle behavior.
  Mitigation: write endpoint regression tests before moving logic.
- Risk: deriving minimal router specs changes test-app route gating.
  Mitigation: snapshot intended minimal route inclusion and add route-policy coverage before removing duplicate specs.
- Risk: old per-user SQLite databases miss new schema bootstrap.
  Mitigation: add old-schema fixture tests before removing lazy endpoint DDL.
- Risk: Jobs event filters diverge between SQLite and postgres.
  Mitigation: centralize SQL generation in `JobManager` and test normalized event output.
- Risk: content update post-commit hooks become hidden side effects.
  Mitigation: Media DB owns only DB-local post-commit hooks and returns explicit effect metadata for endpoint/user-scoped invalidation.
- Risk: service-owned prototype review changes reject response shape or error mapping.
  Mitigation: pin approve/reject endpoint behavior with regressions before moving logic.

## Definition Of Done

- Public HTTP contracts remain stable unless explicitly documented and approved.
- Minimal-test router specs derive shared production metadata instead of duplicating partial copies.
- Endpoints no longer own targeted durable storage rules.
- Backend-specific Jobs event SQL lives behind `JobManager`.
- Document workspace table creation and migration are removed from route handlers.
- Prototype promotion review decisions are owned by a public service method.
- Regression tests cover the identified drift risks.
- Bandit passes for touched backend paths or any non-code skip is documented.
