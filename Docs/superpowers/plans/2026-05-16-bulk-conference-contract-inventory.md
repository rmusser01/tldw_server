# Bulk Conference Contract Inventory

Date: 2026-05-16
Backlog: TASK-400
Parent plan: `Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md`

## Purpose

This inventory resolves the source-of-truth question before implementing playlist preflight, durable conference collections, retry behavior, and collection-scoped review. The target workflow is a first-time user ingesting 34 related conference videos as one durable collection, not 34 independent URLs.

## Candidate Stores

| Candidate | Supports stable collection ID | Supports ordered membership | Supports planned items | Supports media resolution | Collision risk | Decision |
|---|---:|---:|---:|---:|---|---|
| Media DB item metadata/keywords | No | No | No | Yes | Medium: tags/metadata can conflate unrelated media and cannot represent unprocessed rows | Reject as collection source of truth; keep as canonical ingested media store |
| Existing `CollectionsDatabase.content_items` only | No | No | Weak | Yes | High: `upsert_content_item()` selects existing rows by `canonical_url`, `content_hash`, or `url`, so planned playlist rows can collide with resolved content or each other | Reject as sole collection model; keep as resolved searchable item index |
| Existing `collection_tags`/`content_item_tags` | Weak | No | No | Indirect | High: tag names are global labels, not collection instances, and cannot carry run/order/status metadata | Reject for conference grouping; keep for user-visible labels |
| JobManager `batch_id` and media ingest job payloads | Weak | No | Only after submit | Yes after terminal result | Medium: batch IDs model a processing run, not a durable conference collection or later review object | Use only as run/execution state linked from collection items |
| WebUI `media:collections:v1` localStorage | Browser-local only | Yes | No | Yes for existing IDs | High: not server-owned, not extension/WebUI durable, not recoverable across clients | Reject as durable source; keep as legacy/manual review convenience with optional migration |
| Extend `content_items` with collection fields | Possible | Possible | Weak | Yes | High: mixes pending source rows with resolved content and inherits URL/hash upsert ambiguity | Reject for planned rows; only add bridges/filters if needed for resolved review |
| New narrow media collection tables inside `CollectionsDatabase` | Yes | Yes | Yes | Yes via `media_id`/`content_item_id` after completion | Low: collection item identity and idempotency are separate from URL/hash content dedupe | Select |
| Vector-store/RAG collection names | No | No | No | Partial | High: retrieval grouping is not an ingestion workflow contract | Reject; derive RAG scope from durable media collection membership |

## Selected Contract

Add a narrow durable media collection layer in `tldw_Server_API/app/core/DB_Management/Collections_DB.py`, adjacent to but separate from `content_items`.

Minimum tables:

- `media_collections`
  - Owner-scoped stable collection entity.
  - Stores user-facing name, collection kind, description, source playlist URL, conference-level metadata, default tags, created/updated timestamps, and soft-delete state if consistent with nearby tables.
- `media_collection_items`
  - Planned and resolved membership rows.
  - Stores `collection_id`, `ordinal`, `source_url`, `normalized_source_id`, `source_kind`, title/speaker/date/track/tag overrides, duplicate status, current ingestion status, `media_id`, `content_item_id`, latest `job_id`, latest `collection_run_id`, idempotency key, retry count, terminal error/warnings, and timestamps.
  - Represents the user's intended 34-item conference list before any download/transcription starts.
- `media_collection_runs`
  - One ingestion attempt across selected collection items.
  - Stores `collection_id`, `batch_id`, run status, requested/queued/processing/completed/failed/skipped/cancelled counts, safe ingest options, started/completed timestamps, and terminal summary.

The collection item row is the stable unit for retry, cancellation, duplicate marking, per-talk metadata edits, and later review navigation. `content_items` remains the resolved searchable/reviewable item after successful ingestion, linked by `content_item_id` and `media_id`.

## Evidence From Current Code

- `CollectionsDatabase.content_items` already includes `media_id`, `job_id`, and `run_id` fields and supports list filters for `job_id`, `run_id`, tags, origin, status, and FTS search.
- `CollectionsDatabase.upsert_content_item()` looks up existing rows by `canonical_url`, `content_hash`, and `url`, then updates the matched row. That is useful for deduped resolved content, but unsafe for planned playlist rows where the product needs stable membership and retry state even before media exists.
- `/api/v1/items` first queries `CollectionsDatabase.list_content_items()` and falls back to Media DB only when the collections layer has no rows. It is a unified content item list, not a collection/run orchestration endpoint.
- `/api/v1/media/ingest/jobs` already creates one job per URL/file, returns a `batch_id`, lists by `batch_id`, streams SSE by `batch_id`, and supports batch cancellation. It does not store collection identity, planned item IDs, or inherited conference metadata in a first-class contract.
- `media_ingest_jobs_worker.py` processes one job payload source and returns `media_id`/`media_uuid`/warnings/errors. It does not currently update collection item state after terminal job status.
- `sync_media_add_results_to_collections()` dual-writes successful synchronous `/media/add` results into `content_items` with a canonical `media://{media_id}` URL and source metadata. This is the right bridge from ingested media into the review/search layer.
- `apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts` stores manual media collections in `media:collections:v1` localStorage. Those collections have browser-local IDs, names, and `itemIds`, but no server identity, run state, planned source rows, inherited metadata, or extension/WebUI durability.
- `apps/packages/ui/src/services/tldw/domains/collections.ts` exposes `/api/v1/items` list and bulk update wrappers. It has no durable collection CRUD or item membership API.
- `apps/packages/ui/src/services/tldw/domains/media.ts` exposes `/api/v1/media/add` and `/api/v1/media/ingest/jobs` wrappers. It has no playlist preflight or durable collection/run API.

## Rejected Alternatives

Do not use only tags to group a conference. A tag such as `pycon-2010` is useful for search and filtering, but it cannot represent playlist order, planned-but-not-yet-ingested talks, retries, duplicate state, or collection-level metadata.

Do not create planned playlist rows directly in `content_items`. The current upsert contract is designed for deduped content, not intended membership. A repeated URL, a content hash collision, or a later successful ingest could overwrite the planned row and lose the distinction between "this source is part of the conference plan" and "this media item exists."

Do not treat `batch_id` as the collection ID. A batch is an execution run. The same conference collection needs multiple runs for retry, partial selection, cancellation/resume, and later review after the original batch is gone from the user's immediate task.

Do not depend on `media:collections:v1` for durability. It should remain a legacy/manual client-side convenience until a migration affordance exists, but it cannot support extension-to-WebUI continuity or server-side review/RAG scope.

Do not expose collection-scoped Knowledge QA from frontend-only filtering. The backend must resolve the selected collection to authorized `media_id` values and constrain retrieval there before the UI offers scoped QA.

## API Placement

Use the media API namespace for playlist and conference workflow primitives:

- `POST /api/v1/media/playlists/preflight`
  - Metadata-only playlist inspection.
  - Returns normalized source IDs, proposed item rows, duplicate-in-batch flags, warnings, and safe playlist metadata.
  - Does not download media or persist browser cookies, auth headers, videos, audio, or secrets.
- `/api/v1/media/collections`
  - Collection create/list/get/update.
  - Item create/update/reorder/select/status operations.
  - Run creation, run status, retry failed/skipped items, cancel active run, and export failed sources.
- Existing `/api/v1/media/ingest/jobs`
  - Stays the low-level execution mechanism.
  - Accepts optional `collection_id`, `collection_item_id`, `collection_run_id`, and idempotency metadata when jobs come from a durable collection run.
  - Continues to support plain one-off URL/file ingestion without collection metadata.
- Existing `/api/v1/items`
  - Remains the resolved content-item list/bulk-update API.
  - May later gain collection filters only as a review/query convenience, not as the owner of run orchestration.
- Knowledge/RAG endpoints
  - Accept collection scope only after backend authorization and media-ID resolution are implemented.

Capability discovery should expose playlist preflight, durable media collections, ingest jobs endpoint, ingest worker availability, job SSE, and collection-scoped Knowledge QA as separate booleans. Endpoint existence and worker availability must remain separate states.

## Migration/Bridge Notes

- Successful synchronous `/media/add` and async ingest jobs should continue to write or update resolved `content_items`; media collection items should link to those resolved rows after completion.
- Planned collection items should not require a `content_items` row. They become reviewable content only after successful ingestion or an explicit duplicate-existing resolution.
- Duplicate detection should be layered:
  - preflight duplicate-in-batch by normalized source ID,
  - duplicate-existing by source URL/media/content item lookup,
  - optional content hash dedupe after media processing.
- Retry should reuse the same `media_collection_items.id` and create a new `media_collection_runs` row or update latest run/job references. The user's row identity and edited metadata must survive failed attempts.
- Existing localStorage collections can be offered an explicit migration path into backend media collections, but migration is not required before bulk conference ingest. Local collections without source URLs should migrate as manual resolved-media collections, not playlist ingestion plans.
- Task 2 decision: `media:collections:v1` in `apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts` remains a local-only manual review collection store. Durable playlist/conference ingestion now uses `/api/v1/media/collections`; migration or side-by-side labeling of local review collections is deferred to a later UX slice.
- The extension and WebUI should submit the same collection/run payloads through shared services so a playlist detected in the sidepanel can be continued and reviewed in the WebUI.
