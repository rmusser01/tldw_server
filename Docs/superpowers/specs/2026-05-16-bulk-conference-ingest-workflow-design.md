# Bulk Conference Ingest Workflow Design

## Summary

This design defines a staged, PR-sized path from the current Quick Ingest and Media review experience to a first-class workflow for ingesting and reviewing many related conference videos, such as a 34-talk YouTube conference playlist.

The target user is a first-time researcher who wants to ingest a whole conference, preserve conference-level metadata, monitor long-running processing, recover from failures, and later review or ask questions across the conference as one coherent source set.

The design is intentionally reuse-first. It does not introduce a parallel conference-ingest product. It extends the existing WebUI and extension surfaces:

- Quick Ingest for intake and configuration
- media ingest jobs for durable background processing
- Media review for library and collection review
- Knowledge QA for scoped questions over the collection
- extension sidepanel/context detection for browser-originated playlist capture

## Goals

- Make YouTube playlist and watch-with-list URLs understandable before processing starts.
- Let users create durable conference collections during ingest, not after the fact.
- Carry shared metadata such as conference name, source playlist, shared tags, event date, and optional per-talk overrides through the ingest workflow.
- Make 34-item ingest runs recoverable through persistent status, retry, cancel, and failure export.
- Turn completion into an obvious review workflow centered on the conference collection.
- Reuse shared UI so WebUI and extension behavior stays aligned.
- Keep each stage small enough to ship and review as a vertical PR.

## Non-Goals

- Rebuilding Quick Ingest from scratch.
- Creating an extension-only ingest path.
- Replacing the existing Media DB, Jobs, RAG, or Knowledge QA systems.
- Building full transcript editing or draft review as part of this program. The existing Content Review PRD remains adjacent and can integrate later.
- Guaranteeing automatic speaker/date extraction from video titles. The initial contract should support editable fields and optional suggestions only.
- Downloading or processing a whole playlist during preflight.

## Current-State Evidence

The current Quick Ingest wizard accepts files and pasted URLs, supports presets, per-item progress, cancellation, and results actions. However, a pasted YouTube watch URL with a playlist parameter is represented in the UI as one queued video item, not as an expanded playlist preview.

The backend video ingestion documentation says playlist URLs are already expanded server-side before processing begins. That means the right product change is not frontend playlist parsing. The needed contract is a metadata-only preflight endpoint or mode that exposes the expansion before processing.

Media bulk review already supports post-hoc selection, tagging, collection naming, opening a multi-review selection, and export. The current Media collection state is client-side, stored under `media:collections:v1`, so the ideal conference workflow should not depend on it as the durable source of truth.

Media ingest jobs already expose batch submission, batch status, per-job status, cancellation, and SSE events. The worker is controlled by environment flags, so the UI must detect worker availability and communicate degraded durability when jobs are unavailable.

The WebUI and extension share core source under `apps/packages/ui/src`; extension entry points should hand off to the shared preflight flow instead of implementing separate ingest logic.

## Product Spine

The end-to-end workflow should become:

1. User pastes a YouTube playlist URL or uses the extension from a playlist page.
2. Quick Ingest detects playlist context and opens a metadata preflight.
3. The server expands the playlist metadata without downloading videos.
4. User reviews the item list, deselects unwanted videos, sees duplicates, and sets conference metadata.
5. User starts a durable ingest run.
6. The run is tracked as a batch with per-item status, cancel, retry, and failure export.
7. Completed items are linked into a durable conference collection.
8. Results route the user into the collection.
9. The user reviews talks, tracks transcript/summary status, compares selected talks, and asks Knowledge QA scoped to the conference.

## Architecture Principles

- **Server-owned playlist expansion**: The backend owns URL interpretation and playlist metadata extraction. The frontend displays and edits the resulting preflight model.
- **Durable collections before rich review**: The collection/grouping model must be persistent before the UI relies on it.
- **Jobs-aware but not jobs-only**: Use media ingest jobs when available. If unavailable, preserve current ingestion behavior but show reduced recovery guarantees.
- **Shared UI first**: Implement core state, components, and services in shared UI packages so WebUI and extension remain consistent.
- **Progressive enhancement**: Each PR must improve the current workflow even if later PRs are not yet merged.
- **No hidden automation claims**: Speaker/date extraction is treated as suggestion, not truth.

## Cross-Cutting Contracts

### Preflight Is Read-Only

Playlist preflight must not enqueue jobs, download media, create media rows, mutate collections, or write user library state. It may create only short-lived preflight/cache records needed to return metadata and support pagination or retry. Any durable user-visible mutation starts only after the user confirms ingest.

Required verification:

- API tests prove preflight does not create media ingest jobs.
- API tests prove preflight does not create media records or collection membership.
- UI tests prove leaving or cancelling preflight leaves the library unchanged.
- Browser-observed QA confirms the user sees preview state before processing begins.

### Collection Source of Truth

The conference collection source of truth must be server-owned or stored in durable per-user media metadata before any workflow depends on it. The existing client-side `media:collections:v1` state can be migrated, bridged, or left as a local convenience, but it cannot be the authoritative model for conference ingest.

PR 2 must begin with a collection-contract inventory and end with a chosen contract. Later PRs must use that contract and cannot create a second grouping model.

### Metadata and Job Sequencing

Batch metadata is committed before or atomically with job submission, not only stored in transient Quick Ingest state.

Rules:

- A selected preflight item gets a stable planned item record or equivalent metadata binding before processing starts.
- A collection item may be unresolved or resolved. Unresolved items are keyed by a stable planned/source item ID and retain source URL plus metadata. Resolved items also carry a stable media ID.
- A collection may contain planned, processing, completed, skipped-existing, failed, and cancelled item states.
- A successfully created media item resolves its planned item into a stable media ID without losing the original source-item history.
- A skipped existing item may join the collection only when the user explicitly includes existing duplicates.
- A failed or cancelled item keeps enough metadata and source URL for retry or export.
- Synchronous fallback must preserve the same collection metadata semantics even if status tracking is less durable.

### Knowledge QA Scope

Knowledge QA scoped to a conference collection means retrieval is constrained to ready media IDs in that collection unless the user explicitly broadens scope.

Rules:

- Ready items are media records with indexed transcript/text available to Knowledge QA.
- Not-ready items remain visible in collection status but are excluded from retrieval.
- The QA surface must show readiness counts, such as "24 of 34 talks searchable."
- If no items are ready, the scoped QA action is disabled or opens an explanatory empty state.
- Summaries may be displayed in collection review, but retrieval scope must be explicit about whether it searches transcript chunks, summaries, or both.

## Data Concepts

### Playlist Preflight

Represents metadata discovered before processing.

Fields:

- `preflight_id`
- `source_url`
- `source_kind`: `youtube_playlist`, `youtube_watch_playlist`, `manual_urls`, or future source types
- `title`
- `description`
- `item_count`
- `items[]`
- `warnings[]`
- `created_at`

Item fields:

- `source_url`
- `normalized_source_id`
- `position`
- `title`
- `duration_seconds`
- `thumbnail_url`
- `channel_or_uploader`
- `published_at`
- `existing_media_id`
- `duplicate_status`: `new`, `existing`, `duplicate_in_batch`, `unknown`
- `selected`
- `metadata_suggestions`
- `warnings[]`

### Conference Collection

Represents the durable review set created from the batch.

Fields:

- `collection_id`
- `name`
- `kind`: `media_collection` or equivalent server-owned grouping type
- `conference_name`
- `event_date` or `event_year`
- `source_playlist_url`
- `source_playlist_id`
- `shared_tags[]`
- `description`
- `created_from_batch_id`
- `items[]`
- `created_at`
- `updated_at`

Collection item fields:

- `collection_item_id` or `planned_item_id`
- `status`: `planned`, `processing`, `completed`, `skipped_existing`, `failed`, or `cancelled`
- `media_id`, present after the item is resolved to an existing or newly created media record
- `source_url`
- `normalized_source_id`
- `position`
- `job_id`, when processed through media ingest jobs
- `metadata`
- `error_summary`, when failed or cancelled

### Batch Metadata

Metadata inherited by queued items unless overridden.

Fields:

- `collection_name`
- `conference_name`
- `event_date`
- `event_year`
- `shared_tags[]`
- `source_playlist_url`
- `default_media_type`
- `processing_preset`
- `item_overrides`

Item override fields:

- `title`
- `speaker`
- `talk_date`
- `track`
- `tags[]`
- `selected`

### Ingest Run

Represents the durable processing state.

Fields:

- `batch_id`
- `collection_id`
- `source_preflight_id`
- `status`: `queued`, `running`, `completed`, `completed_with_errors`, `cancelled`, `failed`
- `job_ids[]`
- `counts`: queued, running, succeeded, skipped, failed, cancelled
- `started_at`
- `updated_at`
- `completed_at`
- `warnings[]`

## Staged PR Plan

### PR 1: Playlist Preflight + Basic Dedupe

Goal: Make playlist URLs visible and controllable before ingest.

Scope:

- Detect YouTube playlist and watch-with-list URLs in Quick Ingest.
- Add a server-owned metadata-only preflight contract around existing playlist expansion capability.
- Show expanded item list with count, title, duration, URL, position, thumbnail when available, and duplicate status.
- Let users deselect/remove items before continuing.
- Add basic duplicate indicators:
  - already in library
  - duplicate within this preflight
  - unknown when lookup fails
- Add item-count guardrails and partial metadata failure handling.

Out of scope:

- Persistent conference collection creation.
- Durable ingest run dashboard.
- Advanced overwrite/update choices.
- Extension playlist entry point.

Acceptance criteria:

- Pasting the reference conference playlist no longer presents one opaque video row.
- The user sees a multi-item preview before processing.
- The user can deselect playlist items.
- Duplicate-in-batch and already-ingested states are visible when known.
- Preflight does not enqueue jobs, download videos, create media records, or mutate collections.
- Cancelling or closing preflight leaves the user's library unchanged.
- Single URL and file ingest flows remain unchanged.

Risk:

- Metadata extraction may be slow or brittle. Mitigate with timeout, partial results, and a fallback to manual URL queue.

### PR 2: Durable Conference Collection Contract

Goal: Establish a persistent grouping model for conference review before building rich collection UX.

Scope:

- Inventory existing collection/grouping models and choose one server-owned or durable media-metadata-backed source of truth.
- Define or reuse the chosen media collection/grouping contract.
- Store collection name, conference name, source playlist URL, shared tags, event date/year, and membership.
- Represent planned, processing, completed, skipped-existing, failed, and cancelled item states.
- Link collection items to media records as they are created or explicitly resolved from duplicates.
- Provide list/get/update collection APIs or extend an existing media collection API if one exists.
- Bridge current client-side Media collections where useful, without treating localStorage as authoritative.

Out of scope:

- Full conference review UI.
- Advanced folder hierarchy.
- Cross-user or shared collections.

Acceptance criteria:

- The selected collection source of truth is documented with rejected alternatives.
- A conference collection survives refresh and is visible across WebUI and extension contexts.
- Completed collection membership is based on stable media IDs, while planned, failed, and cancelled items remain represented by durable planned/source item IDs.
- Planned, failed, and cancelled items retain source URL and metadata needed for retry or export.
- Existing one-off Media usage is not forced into collections.
- The plan for localStorage Media collections is explicit: migrate, ignore, or bridge.

Risk:

- There are several existing "collection" concepts in the repo. Mitigate with a short collection-contract inventory before implementation and pick the smallest existing model that can support media grouping.

### PR 3: Batch Metadata in Quick Ingest

Goal: Let users define conference-level organization during ingest.

Scope:

- Add batch metadata fields to Quick Ingest:
  - collection name
  - conference name
  - event date/year
  - shared tags
  - source playlist URL
- Add per-item editable overrides:
  - title
  - speaker
  - talk date
  - track
  - selected
  - item tags
- Make inherited metadata scope clear.
- Ensure changes apply to selected/all queued items explicitly, not ambiguously to future items only.
- Preserve existing presets and advanced processing controls.

Out of scope:

- Automatic speaker extraction as a source of truth.
- Rich metadata validation beyond basic field validation.

Acceptance criteria:

- A user can set conference metadata once for all 34 talks.
- A user can override title/speaker/date for one item without editing every row.
- The review step states which metadata will be applied to all selected items.
- Metadata is persisted through the chosen collection/planned-item contract before or atomically with job submission.
- Failed, cancelled, and synchronous-fallback items retain batch metadata and source URLs for later retry/export.

Risk:

- The Quick Ingest modal could become too dense. Mitigate with a progressive disclosure layout: batch fields visible, per-item overrides in a table/drawer.

### PR 4: Jobs-Backed Ingest Run Tracking

Goal: Make long-running bulk ingest recoverable.

Scope:

- Submit batch ingest through media ingest jobs when server capabilities indicate support.
- Bind job submission to the existing collection/planned-item state so status updates can resolve planned items to final media IDs.
- Show a persistent ingest-run panel/page with:
  - batch status
  - per-item job status
  - progress counts
  - elapsed time
  - cancel job
  - cancel batch
  - retry failed
  - export failed URLs
- Subscribe to SSE events when available and fall back to polling.
- Detect worker disabled/unavailable state and show a clear degraded-mode message.
- Preserve current synchronous path for environments without jobs.

Out of scope:

- Changing worker deployment defaults.
- Perfect cancellation of third-party processing.
- Background OS notifications.

Acceptance criteria:

- Refreshing during a jobs-backed 34-item run restores the run state.
- Failed items can be exported and retried.
- Job status updates preserve collection membership and item metadata across success, skip, failure, cancellation, and retry.
- Users can tell whether the run is durable or using a less recoverable fallback.
- One-off ingest remains possible.

Risk:

- Jobs availability varies by deployment. Mitigate with server capability checks and explicit fallback copy.

### PR 5: Results + Collection Handoff

Goal: Turn completion into the next review workflow.

Scope:

- Replace generic completion emphasis with collection-centered next actions.
- Show grouped results:
  - succeeded
  - skipped/existing
  - failed
  - cancelled
- Primary action: open the created conference collection.
- Secondary actions:
  - ask this collection in Knowledge QA
  - review failed items
  - retry failed
  - export failed URLs
  - ingest more
- Link successful and skipped duplicate items to the collection when appropriate.

Out of scope:

- Full collection review redesign.
- Transcript editing.

Acceptance criteria:

- A completed batch routes naturally to the conference collection.
- The user can recover failed URLs without copying from logs.
- Duplicate/skipped items do not disappear from the user's mental model.
- Results are understandable for mixed success/failure batches.

Risk:

- Linking skipped existing items could surprise users if duplicates are unrelated. Mitigate with explicit "include existing items in collection" behavior in preflight/results.

### PR 6: Conference Collection Review

Goal: Make many related videos efficient to review as a set.

Scope:

- Add or enhance a collection review page for media collections.
- Show a talk list with:
  - title
  - speaker
  - date/year
  - transcript status
  - summary status
  - tags
  - ingest status
- Add next/previous talk navigation.
- Add a V1 "compare selected talks" action limited to selected talk metadata plus available summaries/transcript excerpts. It should not create a new chat mode or broad synthesis workspace in this PR.
- Add Knowledge QA scoped to the collection.
- Show Knowledge QA readiness counts and explain which talks are searchable.
- Preserve access to individual media detail, transcript, summary, notes, and chat actions.

Out of scope:

- A full NotebookLM-style workspace.
- Multi-user collaboration.
- Transcript editing as a required step.

Acceptance criteria:

- A user can move through the conference talks without returning to global Media search.
- A user can see which talks are ready for review and which failed or are still processing.
- Knowledge QA can be launched scoped to ready media IDs in the conference collection.
- If only some talks are ready, the QA surface states the ready/not-ready counts.
- Selection-based comparison works for a small set of talks using metadata and available summary/transcript snippets only.

Risk:

- Review scope can expand quickly. Mitigate by limiting V1 to navigation, status, scoped QA, and compare selected.

### PR 7: Extension Playlist Capture

Goal: Reduce manual URL collection from the browser extension.

Scope:

- Detect YouTube playlist pages and watch pages with a `list` parameter.
- Show an "Import playlist to tldw" action in the extension sidepanel or relevant quick action surface.
- Pass URL and detected context to the shared Quick Ingest preflight.
- Keep expansion, metadata editing, and ingest submission in shared UI/services.

Out of scope:

- Extension-specific playlist parser.
- Extension-only collection model.
- Processing inside the content script.

Acceptance criteria:

- The same playlist URL produces the same preflight state from WebUI paste and extension capture.
- Extension users can start the conference workflow without manually copying 34 URLs.
- Unsupported pages do not show misleading playlist import actions.

Risk:

- Browser permissions and sidepanel availability vary. Mitigate by keeping the handoff URL-based and using existing extension configuration/error patterns.

### PR 8: Duplicate and Failure Recovery

Goal: Make duplicate and failed-item recovery explicit before final polish.

Scope:

- Advanced duplicate policies:
  - skip
  - overwrite
  - update metadata only
  - include existing in collection
- Better YouTube failure classifications:
  - unavailable/private
  - cookies/auth required
  - network/provider timeout
  - unsupported URL
  - server capability missing
- Cookies/auth guidance where appropriate.
- Retry selected subsets.

Out of scope:

- Background notifications.
- Full-path visual QA.
- Solving every yt-dlp provider edge case.
- New telemetry.

Acceptance criteria:

- Mixed duplicate/failure batches leave the user with actionable next steps.
- Failure messages explain likely cause and recovery path.
- Retry selected applies only to selected retryable failures.
- Duplicate actions preserve collection membership semantics and do not create surprise overwrites.

Risk:

- Error classification can become over-specific. Mitigate with a small taxonomy and conservative unknown fallback.

### PR 9: Notifications and Full-Path QA

Goal: Validate and polish the ideal workflow end to end.

Scope:

- Background completion notification within WebUI/extension.
- Full-path QA for 34-video-scale flows using mocked playlist metadata and mocked job events.
- Browser-observed QA across WebUI and extension entry points.
- Documentation updates for the conference playlist workflow.
- Final copy and empty-state polish for degraded jobs, unavailable metadata, partial readiness, and failed item recovery.

Out of scope:

- New core ingest behavior.
- Additional collection model changes.
- New telemetry.

Acceptance criteria:

- The mocked 34-video happy path passes in automated coverage.
- Mixed duplicate/failure/jobs-unavailable cases pass in automated coverage.
- A browser-observed QA record confirms the end-to-end WebUI path.
- Extension capture and WebUI paste produce the same preflight state.
- User-facing docs explain playlist preflight, collection creation, durable runs, and recovery.

Risk:

- Full-path QA can become flaky if it depends on real YouTube or real downloads. Mitigate with mocked metadata and mocked job events.

## Cross-PR Requirements

- Preserve ordinary one-file and one-URL Quick Ingest behavior.
- Keep shared implementation in `apps/packages/ui/src` unless platform-specific browser APIs are required.
- Use existing backend/client API helpers and OpenAPI guard patterns.
- Prefer server-owned persistent state for anything that must survive refresh, device changes, or WebUI/extension transitions.
- Keep localStorage-only state limited to temporary UI preferences or draft state.
- Avoid hidden downloads during preflight.
- Avoid storing secrets or auth cookies in job payloads.
- Make degraded modes visible.

## Testing Strategy

Each PR should include focused tests matching its blast radius.

Recommended coverage:

- Unit tests for YouTube URL classification and preflight normalization.
- API tests for metadata-only preflight, dedupe lookup, collection creation, and job status behavior.
- UI tests for Quick Ingest playlist preflight, batch metadata, results handoff, and collection review.
- Extension tests for playlist context detection and shared preflight handoff.
- Browser-observed QA for:
  - one normal URL
  - one playlist URL
  - a mocked 34-item playlist
  - duplicate items
  - failed items
  - jobs unavailable fallback

## Rollout and Feature Flags

- Gate playlist preflight behind a server capability until the endpoint is stable.
- Gate jobs-backed run tracking behind existing media-ingest job capability checks.
- Keep synchronous ingest fallback for deployments without media ingest workers.
- Prefer progressive rollout by route/surface rather than a single all-or-nothing flag.

## Open Questions

- Which existing backend collection model should PR 2 select as the durable media collection source of truth after the inventory?
- Should playlist preflight use a new endpoint or an option on an existing media ingest endpoint?
- What is the maximum playlist size for V1 preflight before requiring pagination or truncation?
- How should server-side duplicate lookup normalize YouTube watch URLs, playlist URLs, and shortened URLs?
- Should "include existing duplicate items in this collection" default on or off?
- Where should the collection review page live: Media, Knowledge, or a dedicated Collections route?
- What exact server capability field should communicate media ingest worker availability?
- Should scoped Knowledge QA search transcript chunks only in V1, or transcript chunks plus generated summaries when available?

## Definition of Done for the Program

- A first-time user can paste or capture a YouTube conference playlist.
- The app expands the playlist into an editable preflight without downloading videos.
- Preflight is proven read-only with no library, job, media, or collection mutation.
- The user sets conference metadata once.
- The collection source of truth is durable and documented.
- The ingest run is durable when jobs are available and honest about fallback when not.
- Completed, skipped, and failed items are clearly reported.
- Successful and included existing items appear in a durable conference collection.
- The user can review talks sequentially, compare selected talks, and ask Knowledge QA scoped to the conference.
- The extension starts the same workflow instead of creating a separate one.
- The 34-video workflow has automated coverage and a browser-observed QA record.
