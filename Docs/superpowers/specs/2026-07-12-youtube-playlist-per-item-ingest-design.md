# YouTube Playlist Per-Item Ingest Design

**Date:** 2026-07-12

**Status:** Three-iteration written-spec review complete; final issues resolved with requester approval; pending requester file review

**Backlog:** TASK-12109

**Scope:** Shared WebUI/browser-extension Quick Ingest plus the backend contracts required to expose every playlist item throughout ingestion

## Summary

When a user adds a YouTube playlist URL, the WebUI can currently represent it as one queue row and one ingest job even though video processing later expands the URL into many videos. The frontend therefore reports the lifecycle of the original playlist URL rather than the videos actually being processed.

The target behavior is fail-closed and per-item:

1. Every playlist-shaped URL is inspected by the server before it can enter the queue.
2. The complete, ordered playlist is shown to the user.
3. Every selected video becomes one concrete queue occurrence. Each occurrence resolves exactly once, and only an action that requires media processing creates one media ingest job per attempt.
4. WebUI and extension show the same per-video queue, progress, cancellation, recovery, and result states.
5. Large playlists remain bounded, virtualized, resumable, and honest about server limits.

The frontend must not parse YouTube playlists. The server owns URL classification, metadata extraction, duplicate lookup, snapshot consistency, and job execution.

## Current-State Evidence

The repository already contains most of the product skeleton, but the ordinary URL-add path can bypass it:

- `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx` detects playlist candidates and offers `PlaylistPreflightPanel`, but `handleAddUrls` still converts every input line directly into one `WizardQueueItem`.
- The extension active-tab action in `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx` seeds the shared playlist preflight flow, so extension capture is safer than ordinary WebUI paste.
- `apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx` already renders metadata returned by the server, but preview is optional and the panel eagerly renders a short list.
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py` creates one job per submitted URL. A playlist URL is therefore initially treated as one item.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py` expands YouTube playlists inside video processing.
- `tldw_Server_API/app/services/media_ingest_jobs_worker.py` then projects only `results[0]` into the job result, losing the remaining child-result identities at the Jobs boundary.
- The WebUI direct path in `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts` submits and waits on URL entries sequentially. The extension background path submits entries before polling, so the two clients also differ in scale behavior.
- `apps/packages/ui/src/store/quick-ingest-session.ts` persists Quick Ingest state in `sessionStorage`, whose quota is unsuitable for large playlists and whose failures are currently silent.

The earlier bulk conference workflow established a useful metadata-only preflight and durable collection model. This design tightens the active playlist path and addresses the remaining identity, scale, and recovery gaps without creating a separate playlist product.

## Product Decisions

| Decision | Required behavior |
|---|---|
| Expansion timing | Expand before confirmation; never after the UI has committed to one playlist row. |
| Failure policy | Block playlist ingestion when inspection is unavailable or incomplete. Do not offer an opaque fallback. |
| Large playlists | Load the complete playlist through a bounded server snapshot and paginated client reads. Never silently truncate. |
| Row presentation | Show playlist position and video title first; show channel, duration, availability, duplicate state, and progress second. Keep the concrete URL in row details. |
| Client parity | WebUI paste, extension active-tab import, Add, and Enter use the same shared controller and state model. |
| Execution identity | Every selected occurrence resolves exactly once. A processing-required action maps to one media ingest job per attempt; duplicate reuse, skip, and metadata-only actions terminate without a processing job. |
| Collections | A media collection is optional. Run tracking cannot depend on collection creation. |

## Goals

- Make every selected video visible before submission and throughout its ingest lifecycle.
- Preserve stable mapping among preflight occurrence, queue row, planned collection item when present, job, media result, and retry attempt.
- Support multiple playlists and ordinary URLs in the same staged Quick Ingest session.
- Keep large playlist rendering and network behavior bounded.
- Recover accepted work after WebUI reload or extension service-worker restart.
- Give users truthful statuses for inspection, submission, processing, cancellation, and recovery.
- Preserve the documented multi-result playlist behavior of the no-database `/process-videos` endpoint.

## Non-Goals

- Parsing YouTube pages or playlist APIs in the frontend.
- Creating a general workflow engine or replacing the existing media Jobs system.
- Guaranteeing access to private, deleted, region-blocked, or authentication-gated videos.
- Claiming that all playlist sizes are safe. The server retains an administrator-configurable hard ceiling.
- Automatically loading third-party thumbnails for every row.
- Requiring a media collection for ordinary playlist ingestion.

## Target Architecture

```mermaid
flowchart LR
    A["WebUI paste or extension active tab"] --> B["Shared playlist detector"]
    B -->|"ordinary URL"| C["Staged ordinary queue row"]
    B -->|"playlist context"| D["Owner-scoped preflight resource"]
    D --> E["Immutable metadata snapshot"]
    E --> F["Paginated virtual preview"]
    F --> G["Confirmed concrete occurrences"]
    G --> H["Owner-scoped ingest run"]
    H --> I["Bounded job-submission chunks"]
    I --> J["One media job per processing occurrence"]
    J --> K["SSE or run-level polling"]
    K --> L["Per-video progress and results"]
    H -. "optional" .-> M["Media collection and planned items"]
```

The new contracts are deliberately narrow:

- an asynchronous, temporary playlist-preflight resource;
- a small owner-scoped playlist ingest run manifest;
- structured occurrence-aware submission fields and results on the existing media Jobs endpoint;
- one shared client controller and normalized run-status transport.

The existing Jobs worker remains responsible for processing each concrete video URL.

## Playlist Detection and Mandatory Inspection

The shared client classifies every submitted line before mutating the queue.

A URL is a playlist candidate when it is a trusted YouTube/youtu.be host and contains a non-empty `list` parameter. This includes `/playlist`, `/watch?...&list=...`, and youtu.be URLs with playlist context. Lookalike and suffix-appended hosts are ordinary web URLs.

Rules:

- Add and Enter invoke playlist inspection instead of `handleAddUrls` for candidates.
- The extension active-tab action invokes exactly the same shared controller.
- The original input remains visible until inspection succeeds or the user removes it.
- Ordinary URLs may appear as staged rows while playlist inspection runs, but Configure, Quick Process, and Start Processing remain disabled while any candidate is unresolved.
- Multiple playlist candidates may inspect concurrently only up to a small client-side bound; the backend applies its own global/per-owner capacity limits.
- No client path may convert a candidate directly into an opaque `WizardQueueItem`.

## Preflight Resource Contract

The current synchronous endpoint remains available as a compatibility surface during rollout. The version-2 client uses an asynchronous resource contract advertised by `mediaPlaylistIngestContractVersion >= 2`.

Required version-2 routes:

- `POST /api/v1/media/playlist-preflights`
- `GET /api/v1/media/playlist-preflights/{preflight_id}`
- `GET /api/v1/media/playlist-preflights/{preflight_id}/items?cursor=...&limit=...`
- `POST /api/v1/media/playlist-preflights/{preflight_id}/materializations`
- `DELETE /api/v1/media/playlist-preflights/{preflight_id}`

`POST` validates the playlist URL, creates an owner-scoped resource, starts bounded extraction, and returns promptly with HTTP 202. It does not create media, collections, runs, or media ingest jobs. The implementation may use an internal `playlist_preflight` Jobs-domain task for multi-worker-safe leasing and execution; that task is not a media ingest job and has no library side effects.

The resource state is one of:

- `pending`
- `ready`
- `blocked`
- `cancelled`
- `expired`

The summary exposes:

- `preflight_id`
- `status`
- `source_url`
- `source_kind`
- `playlist_id`
- `playlist_title`
- `total_count`, when trustworthy
- `loaded_count`
- `ingestible_count`
- `unavailable_count`
- `duplicate_count`
- `expires_at`
- typed warnings or error code

The item endpoint returns an immutable ordered page plus an opaque `next_cursor`. A cursor is bound to owner, preflight ID, and snapshot version. Tampered, cross-owner, expired, or mismatched cursors fail safely.

### Extraction and capacity

- Extraction runs outside the API event loop in a bounded, terminable process so timeout or cancellation can reclaim capacity.
- The extractor requests at most `configured_limit + 1` entries. Seeing the extra entry produces `playlist_too_large`; it must not first materialize an unbounded playlist.
- If YouTube provides a trustworthy total, the UI may show it. Otherwise the oversized error says the playlist contains more than the configured limit rather than inventing an exact count.
- Preflight resource metadata and normalized snapshot items are stored in a shared owner-scoped temporary repository backed by the deployment's configured SQLite/PostgreSQL data layer. A process-local TTL/LRU may be only a read-through optimization, never the source of truth.
- Extraction claims use database-backed leases and enforce global and per-owner capacity across API/worker processes. A local child process performs the blocking yt-dlp call only after a worker owns the lease.
- Successful snapshots are immutable and TTL-bound. Cleanup releases leases and removes expired snapshot rows or temporary artifacts.
- Server restart may expire temporary preflights. That is a recoverable state, not data loss, because no ingest has begun.
- Deleting a preflight terminates or marks extraction cancelled, releases temporary artifacts, and makes future item reads unavailable.

### Preflight item fields

Each item includes:

- `occurrence_id`
- `ordinal`
- `occurrence_index_for_source`
- `source_url`, when ingestible
- `normalized_source_id`
- `source_kind`
- `title`
- `channel_or_uploader`
- `duration_seconds`
- `published_at`
- `thumbnail_url`
- `availability`
- `duplicate_status`
- `duplicate_of_occurrence_id`, when applicable
- `selected_by_default`

Unavailable entries remain visible but cannot be selected. A response is `ready` only when the snapshot is complete. An interrupted extraction with unknown missing entries is `blocked`, not partial-ready.

### Queue materialization

`Add N videos` calls the materialization route with selected occurrence IDs. The server re-reads the completed owner-scoped snapshot and copies only the selected source identity: concrete URLs, occurrence IDs, normalized source IDs, and compact display metadata. It must not freeze duplicate evidence, duplicate-policy choices, or metadata patches because library state and user choices can change through Review. The route returns `materialization_id`, an owner-bound opaque materialization token, compact occurrence records, and `expires_at`.

The client creates queue rows only after this request succeeds and persists the materialization reference with them. A preflight snapshot may then expire without invalidating those rows because Start Processing resolves playlist inputs from the materialization record, not from the deleted preflight snapshot and not from client-supplied URLs.

Queue materialization retention is at least as long as the supported Quick Ingest draft-retention window and may be renewed while the draft remains active, subject to an administrator-configured maximum. If the materialization itself expires before Start Processing, playlist rows become blocked and require reinspection; the client must not submit its cached URL copies as authoritative replacements.

## Identity and Duplicate Semantics

Identity must distinguish an occurrence from its underlying video:

- **Occurrence ID:** an opaque server-generated identifier returned on every preflight item. It is unique within the authenticated owner's preflight namespace, stable for the lifetime of that immutable snapshot, and copied unchanged into the queue and run. Clients must not derive it. It is used for queue, job, progress, cancellation, and result mapping.
- **Normalized source ID:** canonical video identity, such as a YouTube video ID, used for duplicate detection.

The same video may appear more than once in one playlist or across several staged playlists. Those occurrences remain independently visible while sharing a dedupe identity.

Duplicate evaluation occurs across:

1. repeated items in one snapshot;
2. all playlists and ordinary concrete URLs staged in the current Quick Ingest session;
3. the authenticated user's existing Media DB records through an owner-scoped bulk lookup.

If the library lookup cannot establish existing state, the status is `unknown`, not `new`.

Default selection keeps the first new occurrence and deselects later duplicates. Existing duplicate policies remain explicit: skip, include existing, update metadata only, or overwrite. Overwrite is never selected implicitly.

Refresh reconciliation uses normalized source ID plus occurrence index among repeats. Unambiguous selection is retained. Ambiguous additions, removals, or reordering are disclosed before confirmation.

## Ingest Run Contract

Confirmation creates a lightweight owner-scoped run before any job submission. The run is not a replacement for Jobs; it is a manifest that preserves the relationship between selected occurrences and the job batches that execute them.

Required routes:

- `POST /api/v1/media/ingest/runs`
- `GET /api/v1/media/ingest/runs/{run_id}`
- `GET /api/v1/media/ingest/runs/{run_id}/items?cursor=...&limit=...`
- `GET /api/v1/media/ingest/runs/{run_id}/events/stream?after_id=...`
- `POST /api/v1/media/ingest/runs/{run_id}/cancel`
- `POST /api/v1/media/ingest/runs/{run_id}/retry`

`Add N videos` materializes selected preflight items into the owner-scoped queue materialization described above and then creates concrete client queue records plus compact IndexedDB persistence. It does not create the server run. The server run is created only when the user chooses Start Processing, after all playlists, ordinary URLs, files, configuration, and review choices are final. Preflight expiry after queue materialization therefore does not invalidate the queued playlist rows.

Run creation receives normalized processing options, optional collection metadata, a list of selected input records, and `review_overrides` keyed by occurrence ID. An override contains the final duplicate policy when duplicate evidence requires a choice and an optional metadata patch conforming to the allowlist below. At Start Processing, the server refreshes owner-scoped duplicate evidence, then validates that every override belongs to an input occurrence, that every selected duplicate has an explicit valid policy, and that metadata patches are allowed for that policy. Unknown, missing, conflicting, or extra overrides fail run creation before any collection, metadata mutation, or job submission. If refreshed evidence changes a required choice, the API returns structured `review_required` details and the client returns to Review without side effects.

Input records are a discriminated union:

- `materialized_playlist_item`: owner-bound materialization ID/token plus the server-issued occurrence ID;
- `direct_url`: a client-generated opaque occurrence ID, concrete non-playlist URL, normalized type, and compact display metadata;
- `file_stub`: a client-generated opaque occurrence ID plus file name/type/size metadata, later bound to the staged upload job without storing file bytes in the run manifest.

For `materialized_playlist_item`, the server resolves the occurrence from the owner-scoped unexpired queue materialization and treats its stored source identity as authoritative, refreshes duplicate evidence, and applies only the validated Review-time override supplied at Start Processing. For `direct_url`, the server validates and canonicalizes the URL, rejects any playlist candidate, requires occurrence IDs to be unique within the run, and validates its Review-time override against the same fresh duplicate lookup. For `file_stub`, the server records metadata and an `awaiting_upload` lifecycle state but no bytes. This gives mixed sessions one identity contract while keeping local file bytes in the existing upload path.

The run stores:

- owner and `run_id`
- playlist summaries
- occurrence IDs, normalized source IDs, concrete URLs, compact display metadata, and selection
- optional collection and planned-item IDs
- submission chunk states and batch IDs
- job-to-occurrence and job-to-planned-item mappings
- attempt number and endpoint-derived idempotency identity
- per-occurrence and aggregate states
- retention timestamps

A collection is optional. When requested, collection plus planned-item creation is one transactional bulk operation. Failure leaves the run unsubmitted and does not create a partial collection.

Run retention outlives active jobs and remains long enough for reload, extension restart, retry, and result inspection. Cleanup removes only run metadata; it never deletes referenced media or collections.

### Run status snapshot and events

`GET /ingest/runs/{run_id}` returns the owner-scoped run summary, aggregate counts, current version, update timestamp, optional collection ID, and all known batch IDs. It does not embed an unbounded item list.

`GET /ingest/runs/{run_id}/items` returns an ordered occurrence page with lifecycle `state`, terminal `outcome` when present, progress percentage/message when known, job and media IDs when available, attempt, retryability, and an opaque `next_cursor`. Occurrence ordering is immutable after run creation, so the cursor is bound to owner, run ID, sort order, and the last stable occurrence position rather than the frequently changing status version. Each page reports the current run version; clients merge by occurrence ID and refresh active rows if versions changed during a multi-page read.

The run event stream emits an initial summary snapshot followed by occurrence-aware events with:

- monotonically increasing `event_id`
- `run_id`
- `occurrence_id`
- optional `job_id` and `batch_id`
- event type, lifecycle state, and terminal outcome when present
- progress percentage/message when known
- `occurred_at`

Clients resume with `Last-Event-ID` or `after_id`. The producer queries run/job events by owner and `run_id` on every cycle; it must not freeze the tracked job-ID set at connection time. Jobs accepted by later submission chunks therefore appear in the same stream. If replay history has expired, the server emits `resync_required` and the client reloads the summary and paginated item snapshot before continuing.

### Lifecycle state and terminal outcome

Run items expose two separate axes. `state` describes current lifecycle position:

- `staged`
- `preparing`
- `awaiting_upload`
- `submit_pending`
- `queued`
- `running`
- `cancellation_requested`
- `status_unavailable`
- `terminal`

`outcome` is null until `state=terminal` and is then exactly one of:

- `completed`
- `included_existing`
- `metadata_updated`
- `skipped_existing`
- `submit_failed`
- `processing_failed`
- `metadata_update_failed`
- `cancelled`

Progress phase/message is optional evidence from the worker, not a third lifecycle state. This separation is used consistently by the run snapshot, events, IndexedDB record, filters, counts, and result groups.

`file_reattach_required` is a client presentation state, not a backend lifecycle value. The client derives it only when the server reports `awaiting_upload` and the current browser runtime no longer has the corresponding local file bytes.

## Job Submission and Idempotency

Selected concrete URLs are submitted to the existing `/api/v1/media/ingest/jobs` endpoint in bounded chunks. The initial implementation should use a configurable chunk size with a conservative default rather than one request per video or one unbounded request.

Each submitted occurrence carries:

- `run_id`
- `occurrence_id`
- optional planned collection item ID
- attempt number
- a client attempt token used only as input to endpoint-derived idempotency

The endpoint derives the actual Jobs idempotency key from authenticated owner, run, occurrence, and attempt. Raw client keys are never trusted as globally scoped Jobs keys.

The submit response returns a structured record for every occurrence:

- `occurrence_id`
- `accepted` or `rejected`
- `job_id`, when accepted
- `batch_id`
- safe error code and message, when rejected
- `retryable`
- attempt reference

String-only error lists are insufficient because they cannot safely map failures back to repeated URLs.

URL chunks use aligned occurrence/attempt/planned-item arrays. File chunks use multipart `files` plus aligned `file_occurrence_ids`, `file_attempts`, and optional `file_planned_item_ids`; array lengths must exactly match the uploaded file count. The server validates that each file occurrence belongs to the owner/run and is currently `awaiting_upload`, stages the upload, writes `run_id`, `occurrence_id`, and attempt into the job payload, and returns the same structured per-occurrence acceptance record used for URLs.

If the UI/runtime restarts before a file job is accepted, browser file bytes are not assumed recoverable. The server item remains `awaiting_upload`; the client presents `file_reattach_required` while the bytes are absent. The user may reselect the file, preserving occurrence identity, or cancel it. Once the job is accepted, normal run/job reattachment applies.

Global failures stop later chunks: authentication/authorization, quota, worker unavailability, shutdown/draining, invalid run ownership, and rate limiting. Rate-limited clients honor `Retry-After`. Isolated invalid, unavailable, or duplicate entries do not stop unrelated occurrences.

Ambiguous network failure retries the same occurrence attempt. Jobs idempotency returns the original job rather than creating another. A deliberate processing retry uses a new attempt number after reconciling whether the prior attempt already created media.

### Duplicate policy actions

Duplicate policies have distinct server actions whether or not a collection exists:

| Policy | Job action | Terminal outcome | Optional collection behavior |
|---|---|---|---|
| `skip` | Do not submit a job or mutate media metadata. | `skipped_existing` with the resolved existing media ID when available. | Do not add membership. |
| `include_existing` | Do not submit a job. Resolve and reuse the existing media item. | `included_existing`. | Resolve the planned item/membership to the existing media ID. |
| `update_metadata_only` | Do not submit a media-processing job. Apply only the reviewed metadata patch contract through the Media DB abstraction. | `metadata_updated` or `metadata_update_failed`. | Resolve membership to the existing media ID when the update succeeds. |
| `overwrite` | Submit one concrete job with overwrite enabled. | Normal completed/processing-failed outcome. | Resolve the planned item to the resulting media ID. |

Without a collection, `include_existing` still creates a run result linking the existing media item, while `update_metadata_only` still performs and reports the metadata operation. These policies must not collapse into the same `skipped_existing` state.

The metadata patch is built at Review time only from fields the user explicitly edited or explicitly applied as shared tags. Extracted playlist metadata that the user did not edit is not permission to overwrite an existing record.

The allowed patch schema is:

- `title`: non-empty validated string; replace the existing title;
- `author`: validated string sourced from an explicitly edited speaker/author field; replace the existing author;
- `keywords_add`: validated keyword list sourced from explicitly applied shared/item tags; case-insensitive union with existing keywords.

Content, media type, analysis, prompt, deletion state, and keyword set/remove operations are forbidden. An empty patch makes `update_metadata_only` unavailable in the UI and invalid at the API. The backend applies title/author plus keyword union through one Media DB abstraction transaction with normal version/conflict handling; a conflict produces `metadata_update_failed` and does not silently partially apply the patch.

## Worker Boundary

Every selected run occurrence resolves to exactly one action and terminal outcome. The media Jobs endpoint enforces one job per attempt only for occurrences whose resolved action requires media processing. `skip`, `include_existing`, and `update_metadata_only` resolve through the run without creating a media-processing job. An opaque playlist candidate submitted directly to the Jobs endpoint fails with HTTP 422 and `playlist_preflight_required`.

The worker receives one concrete video URL and returns one media result. It must not expand a playlist within a media ingest job.

The no-database `/process-videos` endpoint may continue accepting a playlist and returning multiple results because its response already represents a processing batch and does not require one job or one planned-item identity.

## Shared Frontend UX

### Inspection

Pasting a playlist changes the primary action to **Inspect playlist**. Extension active-tab import begins the same inspection automatically because the extension action itself is explicit.

The preflight card shows:

- inspecting progress, such as `Loaded 100 of 742`;
- playlist title and availability summary;
- a virtualized ordered item list;
- title and playlist position as primary text;
- channel, duration, availability, and duplicate state as secondary text;
- Select all, Select none, Select new, and per-row controls;
- typed blocking guidance, retry, cancel, and refresh.

`Add N videos` remains disabled until the snapshot is complete and every selected occurrence has a concrete URL.

An expired snapshot can be restarted. Selection reconciliation is shown only after explicit refresh or expiry recovery; immutable snapshots do not continuously claim the source changed.

### Queue

Confirmation converts selected occurrences into ordinary `WizardQueueItem` records. The queue uses flat per-video rows with a lightweight playlist heading rather than a collapsible parent that hides active work.

Each row preserves:

- occurrence ID and normalized source ID
- title
- playlist ID/title and ordinal
- channel/uploader and duration
- concrete URL
- duplicate state and chosen policy
- optional planned collection item

The title/ordinal is primary. The URL and optional thumbnail appear only in details. Thumbnails are not eagerly loaded; explicit loading uses no-referrer behavior and failure is cosmetic.

### Processing

The first state is **Preparing N videos** while run/collection records and jobs are created. Rows do not claim to be processing before job acceptance.

The processing UI uses the lifecycle `state` and terminal `outcome` axes defined by the run contract. In particular, included-existing and metadata-only results are terminal outcomes and `status_unavailable` is a recoverable server lifecycle state. `file_reattach_required` is a recoverable client presentation of server `awaiting_upload` when local bytes are missing.

Known backend progress and messages are displayed. The UI must not fabricate precise Analyze or Store stages when the server supplied only generic processing evidence.

Preflight, queue, processing, and result lists are virtualized and offer useful filters for large runs. Virtual rows preserve keyboard navigation, stable focus, `aria-setsize`, `aria-posinset`, and live summary announcements.

### Cancellation

- Before submission, cancelling a row removes it from unsent chunks.
- After job acceptance, cancelling a row invokes the actual job-cancel endpoint.
- The row remains `cancellation_requested` until the server reports a terminal state.
- If completion wins the race, the final state is completed.
- Cancelling the whole run stops unsent chunks and requests cancellation for all accepted jobs.

### Results and retry

Terminal result groups are Completed, Included existing, Metadata updated, Skipped existing, Not submitted, Failed during processing, Metadata update failed, and Cancelled.

`status_unavailable` is not terminal. It retains Check again and Reconnect actions and becomes interrupted only when the user abandons recovery or the server proves the job record is no longer recoverable.

Retry first reconciles by normalized source/media identity and optional planned item because a failed job may have created media before failing later. Only eligible occurrences receive a new attempt.

## Persistence and Runtime Recovery

Large Quick Ingest sessions move from silent `sessionStorage` persistence to compact IndexedDB records shared by the common UI package. The WebUI and extension keep separate origin-local databases; the design does not assume cross-origin storage.

Persisted state includes compact display metadata, occurrence/run/job mappings, selections, and terminal summaries. It excludes thumbnail bytes and other bulky artifacts.

Requirements:

- migrate the current session-storage record safely;
- make migration interruption recoverable;
- surface quota or write failure instead of silently losing resume guarantees;
- retain active/interrupted runs and recent terminal results for a bounded period;
- clean expired preflight state and stale terminal runs;
- coordinate multiple tabs so one run is not submitted twice.

Status transport is normalized but platform-aware:

- WebUI prefers run-scoped SSE and falls back to paginated run polling.
- Extension treats polling/reattachment as first-class because a service worker may be suspended; SSE is opportunistic.
- Reopening either UI rehydrates the local run record, asks the server for the owner-scoped run/job snapshot, and reconciles by occurrence ID.

Temporary status-fetch failure produces `status_unavailable`, not failure.

## Error Model

Stable public preflight errors include:

- `invalid_playlist_url`
- `playlist_not_found`
- `playlist_private_or_auth_required`
- `playlist_metadata_unavailable`
- `playlist_too_large`
- `preflight_busy`
- `preflight_timeout`
- `preflight_expired`
- `preflight_cancelled`
- `preflight_incomplete`
- `materialization_expired`
- `server_unreachable`

Submission/run errors distinguish authentication, authorization, quota, rate limit, worker unavailable, server draining, invalid ownership, structured per-occurrence rejection, and terminal processing failure.

The UI preserves the source input and gives typed retry/troubleshooting guidance. It never exposes raw yt-dlp output. Unknown provider failures use a generic safe public message with sanitized operator diagnostics.

## Security and Privacy

- Validate trusted YouTube host boundaries; reject lookalike domains.
- Bind preflights, queue materializations, cursors, runs, and jobs to the authenticated owner.
- Derive Jobs idempotency keys with owner scope.
- Redact complete playlist URLs and query secrets in logs and metrics.
- Never persist browser cookies or credentials in preflight/run records.
- Never return another owner's existence through cursor, run, batch, or idempotency behavior.
- Apply existing rate, billing, storage, and worker-readiness controls before accepting work.
- Record counts and typed outcome categories in metrics, not complete source URLs.

## Compatibility and Capability Signaling

The server advertises `mediaPlaylistIngestContractVersion` and granular readiness for asynchronous preflight resources, run tracking, media Jobs, worker availability, and event streaming.

Version-2 clients require the new preflight/run contract for playlist candidates. Older clients that submit a playlist directly to Jobs receive a structured 422 with actionable preflight guidance. The existing synchronous preflight may remain during a deprecation window, but it is not the version-2 path.

The `/process-videos` playlist contract remains separate and covered by compatibility tests.

## Verification Strategy

### Fast PR gate

- Backend unit tests for classification, snapshot state, queue materialization/expiry, capacity, pagination, duplicate lookup and Review-time refresh, metadata-patch validation, error sanitization, Jobs rejection, run transitions, file binding, state/outcome separation, and owner-derived idempotency.
- Hypothesis property/state-machine tests for pagination completeness, occurrence uniqueness, chunk coverage, exactly-once action resolution, zero media jobs for non-processing duplicate actions, at-most-one accepted job per processing attempt, idempotent ambiguous retry, cancellation/completion races, and terminal-count invariants.
- Shared Vitest tests for mandatory inspection, mixed input blocking, metadata persistence, duplicate selection, filters, virtualization, cancellation dispatch, and transport normalization.
- TypeScript checks and Bandit on touched backend scope.

### Integration gate

- Temporary real Jobs database and worker with only expensive media processing replaced by a deterministic fake.
- Preflight process timeout, cancellation, crash, shutdown, capacity release, and orphan cleanup.
- Owner isolation across preflight, queue materializations, cursors, runs, jobs, optional collections, and idempotency.
- Structured partial acceptance, global chunk-stop behavior, retry reconciliation, run-level list/cancel, and optional transactional collection planning.
- IndexedDB migration, interrupted migration, quota failure, cleanup, retention, multi-tab coordination, and runtime recreation.

### Browser gate

- One complete WebUI journey using the existing 34-item conference fixture: paste, inspect, select, queue, submit, progress, reload/reattach, and results.
- One extension journey: active-tab handoff into the same inspection state, background runtime recreation, polling reattachment, and completion.

Large 500-item behavior is verified below full E2E through bounded mounted-row counts, bounded chunk requests, run-level status retrieval, and the absence of per-item polling fan-out. The configured-limit-plus-one case is a backend contract test.

### External and benchmark checks

Required CI uses deterministic sanitized yt-dlp-shaped fixtures. An optional/nightly external test may verify current YouTube compatibility. Non-blocking benchmarks may track extraction and rendering timings, but normal PR gates assert structural bounds rather than flaky wall-clock thresholds.

Accessibility checks combine axe with explicit tests for virtual-list position metadata, keyboard selection, focus recovery, filter announcements, and live-region deduplication.

## Acceptance Criteria

- No WebUI or extension entry path can queue a YouTube playlist as one opaque item.
- A complete ordered preview appears before confirmation, or ingestion remains blocked with typed guidance.
- Every selected occurrence becomes one concrete queue row and resolves exactly once; only processing-required actions create at most one accepted job per attempt.
- Queue, processing, cancellation, recovery, and results preserve occurrence identity and title-first presentation.
- WebUI and extension expose the same normalized states and outcomes.
- Large playlists use bounded extraction, virtualized rendering, chunked submission, and run-level status retrieval.
- Reload, extension runtime restart, partial submission, ambiguous retry, and cancellation do not lose or duplicate accepted work.
- Owner isolation, safe idempotency, redacted diagnostics, and configurable limits are verified.
- `/process-videos` retains its documented multi-result playlist behavior while Jobs ingestion fails clearly when preflight is bypassed.
