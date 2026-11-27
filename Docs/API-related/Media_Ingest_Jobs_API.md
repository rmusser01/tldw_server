# Media Ingest Jobs API

Submit background media ingestion jobs with cancellation support.

Base path: `/api/v1/media`

## Related Connector Sync Jobs

External file-hosting sync does not use the `/api/v1/media/ingest/jobs` endpoints directly.
Google Drive and Microsoft OneDrive source sync runs are queued through the connectors API and the core Jobs domain `connectors`.

- Connector endpoints:
  - `GET /api/v1/connectors/sources/{source_id}/sync` - source-level sync status, cursor state, webhook status, and active job summary
  - `POST /api/v1/connectors/sources/{source_id}/sync` - queue a manual `incremental_sync` job
  - `POST /api/v1/connectors/providers/{provider}/webhook` - webhook callback that validates, dedupes, and queues `incremental_sync`
- Connector job types currently used for file-hosting sync:
  - `import` - source creation / legacy bootstrap path
  - `incremental_sync` - cursor-backed delta sync
  - `subscription_renewal` - webhook renewal before expiry
  - `repair_rescan` - scheduled replay/full-rescan recovery when a source is marked `needs_full_rescan`
- Sync job result payloads include `processed`, `skipped`, `failed`, and, for file-hosting sync, `degraded` counts.

## Resuming a Quick Ingest collection

The WebUI keeps its plan key with the draft and uses a stable source key for each queue row. The authenticated account owns the plan. A fresh draft may import the same URL into a separate collection.

| Request | Effect |
| --- | --- |
| `PUT /api/v1/media/collections/quick-ingest-plans/{plan_key}` | Atomically create or resume the collection and the requested rows. The body contains `collection` and an `items` array of `{source_key, item}`. The response contains `collection_id` and the current row and `dispatch_state` for each requested key. |
| `GET /api/v1/media/collections/quick-ingest-plans/{plan_key}/sources/{source_key}` | Read a source receipt. An unknown source returns HTTP 404. This operation creates no collection rows or jobs. |
| `POST /api/v1/media/collections/quick-ingest-plans/{plan_key}/sources/{source_key}/claim` | Reserve the initial source or an explicit retry of a confirmed terminal failure. A reservation can survive the browser closing before it sends the work. |
| `POST /api/v1/media/collections/quick-ingest-plans/{plan_key}/sources/{source_key}/dispatch/{transport}` | Submit the source using `jobs`, `web`, or `add` and the source attempt's `retry_count` and `idempotency_key` query parameters. The receiver records the canonical job or Media ID before acknowledging a successful dispatch. |

If the setup response is lost, repeat the `PUT` with the same keys. A completed source returns its saved Media ID. An accepted job returns its existing job ID. Replaying setup preserves edited metadata and source order. After a dispatch with an uncertain outcome, read the source receipt before taking further action. A Jobs admission rejection with HTTP 429 and the code `plan_jobs_quota_rejected` changes the source to `fallback_ready`. Only then can the client use the planned `add` fallback. A native file rejected during upload validation has a failed receipt. The user can correct it and request a new attempt on the same source. A confirmed analysis-quota refusal before processing also leaves that source eligible for an explicit retry.

## Endpoints

- POST `/ingest/jobs` - Submit jobs (one job per item)
  - Body: multipart form-data matching `/media/add` fields plus optional `files`
  - Response: `{ batch_id, jobs: [{ id, uuid, source, source_kind, status }], errors: [] }`
  - Retrying a full set of collection URL jobs returns the original job IDs and cancellation batch for that owner. Submit new sources separately. A known mismatch returns HTTP 409 before job creation. If two requests race after one has accepted a source, `errors` identifies the affected source and `jobs` lists the jobs accepted into the returned batch.
  - New uploads with idempotency keys and new collection-bound URL retries use the primary media worker queue. A retry can recover an existing job from its original queue when heavy-worker availability changes.
  - Notes: `errors` contains per-item staging failures; if every item fails, response uses HTTP 207.

- GET `/ingest/jobs?batch_id=...` - List jobs for a batch (owner or admin)
  - Response: `{ batch_id, jobs: [MediaIngestJobStatus...] }`
  - Notes: Uses indexed `batch_group` lookup with legacy payload fallback for older rows.

- GET `/ingest/jobs/{job_id}` - Get job status (owner or admin)
  - Response includes progress fields (`progress_percent`, `progress_message`), result summary,
    and payload metadata (`media_type`, `source`, `source_kind`, `batch_id`).
  - A job that asks for analysis while its owner is over an LLM quota still stores the item, without
    analysis. Its `result` has `status: "Warning"`, the warning
    `Analysis skipped: LLM quota exceeded (<category>).` and
    `analysis_error: {"code": "llm_quota_exceeded", "message": ...}`.

- DELETE `/ingest/jobs/{job_id}` - Cancel a job (owner or admin)
  - Response: `{ success, job_id, status, message }`

- POST `/ingest/jobs/cancel?batch_id=...` - Cancel jobs for an entire batch (owner or admin)
  - Alias: `session_id` can be provided instead of `batch_id`
  - Optional query: `reason`
  - Response:
    - `{ success, batch_id, requested, cancelled, already_terminal, failed, message }`

- GET `/ingest/jobs/events/stream` - Stream ingest job events via SSE (owner or admin)
  - Optional query: `batch_id` to scope to one batch
  - Optional query: `after_id` to resume from a previous event id
  - Stream events:
    - `snapshot` event containing current job statuses
    - `job` events containing incremental `event_id`, `job_id`, `event_type`, and `attrs`

## Cancellation Semantics

- Cancellation is cooperative and best-effort.
- Queued jobs are cancelled immediately.
- In-flight jobs check cancellation before persistence and finalize as `cancelled` without DB writes.
- Audio/video ingestion attempts to preempt long-running FFmpeg/STT work when cancellation is requested.
- Batch/session cancellation applies the same semantics to each matched non-terminal job.

## Worker

- Service: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Env flags:
  - `MEDIA_INGEST_JOBS_WORKER_ENABLED`: `true|false` (default follows the `media` route policy; set `false` to disable the in-process worker)
  - `MEDIA_INGEST_JOBS_QUEUE`: queue name (default `default`)
  - `JOBS_DB_URL` or `JOBS_DB_PATH`: Jobs backend (Postgres DSN or SQLite path)

## Staging

- Uploads are staged into a per-file temp directory.
- The temp dir is stored in the job payload and cleaned up by the worker on completion/cancel.
