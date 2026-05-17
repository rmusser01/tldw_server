# Bulk Conference Playlist Ingest

This guide covers the WebUI and browser-extension workflow for ingesting many related conference talks from one playlist, then reviewing them as one collection.

## When To Use This

Use this workflow when you have a playlist of related videos, such as a conference, course, workshop, or event track, and you want:

- one collection for the whole event
- shared metadata applied once
- per-talk corrections when needed
- durable progress tracking
- clear recovery for duplicates, failed downloads, and retries
- a review page ordered like the playlist

## Start From The WebUI

1. Open **Media** and choose **Quick Ingest**.
2. Paste the playlist URL into the URL input.
3. If the server advertises playlist preflight support, Quick Ingest shows **Playlist detected**.
4. Select **Preview**.

Preflight is metadata-only. It should not create media rows, collections, ingest jobs, or downloaded files. It reads the playlist structure, normalizes item URLs, and marks duplicate items before you commit to ingesting anything.

## Start From The Extension

When the active browser tab is a supported playlist URL, the extension can open Quick Ingest with the playlist already seeded. The same shared Quick Ingest preflight panel is used; the extension should not parse the playlist itself.

If the extension opens Quick Ingest but no preview appears, check that:

- the active tab URL contains a playlist id
- the WebUI is connected to a tldw server
- the server advertises playlist preflight capability

## Review The Playlist Preflight

The preview shows:

- playlist title
- total item count
- selected item count
- duplicate count
- ordered talk list
- item source URLs

Review the selected items before continuing. For large conference playlists, deselect trailers, intro videos, duplicates, or talks you do not want to process.

Duplicate policies:

| Policy | What happens |
|---|---|
| Skip duplicates | Duplicate items are not selected for processing. |
| Include existing | Existing duplicate items are included in the collection but are not reprocessed. |
| Update metadata only | Existing duplicate items are kept as collection members with metadata updated where supported. |
| Overwrite | Duplicate items are submitted with overwrite enabled. |

## Add Conference Metadata Once

After adding previewed items, fill the conference batch section:

- Collection name: the review collection name, for example `Conference 2010 Review`
- Conference name: the event name
- Event year or date: used for inherited item metadata
- Shared tags: tags applied to each planned item

Use per-item overrides for talk title, speaker, track, talk date, or extra tags when the playlist metadata is incomplete.

## Durable And Degraded Modes

Durable mode is available when the server supports media collections and ingest jobs. In this mode, Quick Ingest creates a collection and planned collection items before submitting jobs. This gives you stable progress, retry metadata, and a collection review target even when some jobs fail.

Degraded mode is used when the server does not support the durable collection or job path. Quick Ingest can still process items, but recovery is weaker because run state is local to the browser session.

## Monitor Processing

The processing step shows:

- queued, processing, completed, failed, and cancelled item counts
- durable collection tracking when available
- collection id, planned item count, job count, and batch count
- failed item export when failures are present
- cancel controls for active runs

If you minimize the wizard, the floating progress widget summarizes the collection name and mixed outcomes such as succeeded, skipped, and failed counts. It does not claim the collection is searchable until readiness is confirmed in the results or review surfaces.

## Recover From Problems

Use the results step for recovery:

- **Skipped existing**: duplicates that were included but not reprocessed.
- **Not submitted**: an item failed before a job was accepted.
- **Failed during processing**: a job was accepted but the ingest process failed.
- **Export failed list**: copies or downloads failed source URLs, collection item ids, retry attempt, and error summaries.
- **Retry All**: retries durable, retryable failures using collection item ids and new idempotency keys.

For private or removed videos, retrying may not help until the source is accessible. Export the failed list before closing if you need to inspect source URLs externally.

## Review The Collection

Open the collection from the results step. The collection review page shows:

- conference collection name
- total talk count
- ready talk count
- in-progress count
- items needing attention
- ordered talk navigation
- status badges per talk
- summary and transcript excerpt areas when available
- compare checkboxes for reviewing multiple talks side by side

Ready talks are completed items and skipped-existing items that have a resolved media id. Failed, cancelled, planned, or processing items are visible but not counted as ready for scoped QA.

## Scoped Knowledge QA Readiness

The **Ask this collection** action should only be enabled when at least one talk is ready and the server advertises backend-enforced collection scope. Scoped QA must be enforced by the backend using the collection id or resolved media ids, not by a frontend-only tag filter.

If the button is disabled, wait for more talks to complete or retry failed items. If the button is missing, verify that the server advertises scoped Knowledge QA support.

## Common Mistakes

- Previewing a playlist is not ingestion. You still need to add selected items and start processing.
- Skipped duplicates may still be useful collection members if they point to existing media.
- A completed ingest job does not always mean the talk is ready for QA; readiness depends on resolved media ids and available transcripts/chunks.
- Closing the wizard before exporting failed items is safe in durable mode, but export is useful when debugging unavailable videos.
- Using tags alone is not enough for conference QA scope; use the collection review or scoped QA handoff.
