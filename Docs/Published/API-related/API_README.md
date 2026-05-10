# API Documentation

## Overview

API uses FastAPI framework.
Designed to be simple and easy to use.
Generative endpoints follow openai API spec where possible.
See [API Design](API_Design.md) for more details.

See also:
- `Docs/Code_Documentation/Ingestion_Pipeline_Audio.md` for the audio processing endpoint (`POST /api/v1/media/process-audios`).
- `Docs/API-related/Email_Processing_API.md` for the email processing endpoint (`POST /api/v1/media/process-emails`) and email ingestion via `/media/add`.
- `Docs/API-related/Reminder_Notifications_API.md` for reminder tasks, inbox notifications, SSE stream, and related env flags.

### URLs
- **URLs**
    - Main page: http://127.0.0.1:8000
    - API Documentation page: http://127.0.0.1:8000/docs




### Endpoints

#### Chat

- Chat API (OpenAI-compatible): `POST /api/v1/chat/completions`
  - Streaming (SSE), persistence toggle, multi-provider support
  - See: [Chat API Documentation](Chat_API_Documentation.md)

- Anthropic Messages API (Anthropic-compatible): `POST /api/v1/messages` and `POST /v1/messages`
  - Includes `POST /api/v1/messages/count_tokens` and `POST /v1/messages/count_tokens`
  - Native providers: `anthropic`, `llama.cpp`; other providers use OpenAI conversion
  - See: [Anthropic Messages API](Anthropic_Messages_API.md)

- Character Chat API: sessions and messages under `/api/v1/chats` and `/api/v1/messages`
  - Create/list/update/delete chats, send/edit/delete/search messages
  - Export chat history; fetch messages formatted for completions
  - Use Chat API for LLM replies with `conversation_id`/`character_id`
  - See: [API Design](API_Design.md) for character chat endpoints overview
- Conversation metadata API: list/search, tree view, and analytics under `/api/v1/chat/conversations` and `/api/v1/chat/analytics` (alias: `/api/v1/chats/conversations`)
  - Includes ranking modes (`bm25|recency|hybrid|topic`) and topic/state filters
- Knowledge-save API: `POST /api/v1/chat/knowledge/save`

#### RAG (Retrieval-Augmented Generation) - `/api/v1/rag`

Unified RAG endpoints provide hybrid search (FTS5 + vectors + reranking), optional answer generation, citations, and streaming.

##### Core Endpoints

- `POST /api/v1/rag/search` - Unified search (all features via parameters)
- `POST /api/v1/rag/search/stream` - NDJSON streaming of answer chunks (set `enable_generation=true`)
- `GET /api/v1/rag/simple` - Convenience query param search with sensible defaults
- `GET /api/v1/rag/advanced` - Convenience endpoint with common features enabled (citations/answer)
- `POST /api/v1/rag/batch` - Batch multiple queries concurrently

Notes:
- “Agent” endpoints are not exposed in the current server. Use `/rag/search` with `enable_generation=true` or `/rag/search/stream`.
- Health and ops endpoints are available under `/api/v1/rag/health*` and `/api/v1/rag/cache*`.

For comprehensive documentation, see:
- [RAG API Consumer Guide](RAG-API-Guide.md) - Complete API reference with examples
- [RAG Developer Guide](../Development/RAG-Developer-Guide.md) - Architecture and implementation details

#### Text2SQL - `/api/v1/text2sql`

- `POST /api/v1/text2sql/query` - Run guarded read-only SQL retrieval
  - Request body: `query`, `target_id` (currently `media_db`), optional `max_rows`, `timeout_ms`, `include_sql`
  - RBAC: requires `sql.read`
  - ACL: target must pass connector ACL checks via `sql.target:*` or `sql.target:<target_id>` (returns `403` with `unauthorized_target` when denied)
  - SQL policy: single read-only statement with guardrails and deterministic limit enforcement

Notes:
- Unified RAG can include SQL retrieval by setting `sources` to include `sql` and optionally `sql_target_id` in `POST /api/v1/rag/search`.

#### Media Ingestion - `/api/v1/media`

- `POST /api/v1/media/add` - ingest and persist media (synchronous)
- `POST /api/v1/media/ingest/jobs` - async ingest (one job per item)
- `GET /api/v1/media/ingest/jobs?batch_id=...` - list jobs for a batch
- `GET /api/v1/media/ingest/jobs/{job_id}` - job status
- `DELETE /api/v1/media/ingest/jobs/{job_id}` - cancel job
- `POST /api/v1/media/ingest/jobs/cancel?batch_id=...` - cancel jobs for a batch (supports `session_id` alias)
- `GET /api/v1/media/ingest/jobs/events/stream` - SSE stream for ingest events (supports `batch_id`, `after_id`)

See: [Media Ingest Jobs API](Media_Ingest_Jobs_API.md)

#### Slides / Presentation Studio - `/api/v1/slides`

Slides now back both generated presentations and the Presentation Studio editor surfaces.

- `POST /api/v1/slides/presentations` - create a presentation, including `studio_data`
- `GET /api/v1/slides/presentations/{presentation_id}` - fetch a saved presentation
- `PATCH /api/v1/slides/presentations/{presentation_id}` - optimistic-locking updates with `If-Match`
- `GET /api/v1/slides/presentations/{presentation_id}/export` - slide-native exports (`revealjs`, `markdown`, `json`, `pdf`)
- `POST /api/v1/slides/presentations/{presentation_id}/render-jobs` - enqueue versioned `mp4` or `webm` render jobs
- `GET /api/v1/slides/render-jobs/{job_id}` - inspect render job status
- `GET /api/v1/slides/presentations/{presentation_id}/render-artifacts` - list render outputs for a presentation

Capability notes:

- `hasSlides` indicates the slide/presentation surface is available
- `hasPresentationStudio` indicates the structured Presentation Studio editor should be exposed
- `hasPresentationRender` indicates render-job powered video publishing is available

Presentation Studio clients should treat saved presentation content as the source of truth and
rendered `mp4`/`webm` artifacts as versioned derivatives.

#### Reading List - `/api/v1/reading`

Reading List supports URL capture, clean text extraction, tagging, import/export, and actions (summarize/TTS).

- `POST /api/v1/reading/save` - save a URL
- `GET /api/v1/reading/items` - list/search items
- `GET /api/v1/reading/items/{id}` - item detail
- `PATCH /api/v1/reading/items/{id}` - update metadata
- `DELETE /api/v1/reading/items/{id}` - delete (soft/hard)
- `POST /api/v1/reading/items/{id}/summarize` - summarize
- `POST /api/v1/reading/items/{id}/tts` - TTS audio
- `POST /api/v1/reading/import` - Pocket/Instapaper import
- `GET /api/v1/reading/export` - JSONL/ZIP export

See: [Reading List API](Reading_List_API.md)

#### Chatbooks - `/api/v1/chatbooks`

Chatbooks provide portable backup, restore, sharing, and migration workflows. The same import and preview endpoints also support OpenWebUI "Export Chats" JSON files when `source_format=openwebui_json`.

- `POST /api/v1/chatbooks/export` - create a `.chatbook` archive
- `POST /api/v1/chatbooks/import` - import a `.chatbook` archive or OpenWebUI chat export JSON
- `POST /api/v1/chatbooks/preview` - inspect a chatbook or OpenWebUI JSON export before importing
- `GET /api/v1/chatbooks/export/jobs` and `GET /api/v1/chatbooks/import/jobs` - list background jobs
- `GET /api/v1/chatbooks/download/{job_id}` - download a completed export

See: [Chatbook API Documentation](Chatbook_API_Documentation.md)

#### Reminder Tasks and Notifications

- Tasks: `POST /api/v1/tasks`, `GET /api/v1/tasks`, `PATCH /api/v1/tasks/{task_id}`, `DELETE /api/v1/tasks/{task_id}`
- Notifications: `GET /api/v1/notifications`, `GET /api/v1/notifications/unread-count`, `POST /api/v1/notifications/mark-read`, `POST /api/v1/notifications/{id}/dismiss`, `POST /api/v1/notifications/{id}/snooze`
- Realtime inbox stream: `GET /api/v1/notifications/stream` (SSE with `Last-Event-ID` / `after` cursoring)

See: [Reminder Tasks and Notifications API](Reminder_Notifications_API.md)

#### Collections Feeds - `/api/v1/collections/feeds`

Collections Feeds wraps Watchlists sources/jobs to ingest RSS/Atom into Collections items with `origin="feed"`.

- `POST /api/v1/collections/feeds` - create a feed subscription
- `GET /api/v1/collections/feeds` - list feed subscriptions
- `GET /api/v1/collections/feeds/{feed_id}` - get a feed subscription
- `PATCH /api/v1/collections/feeds/{feed_id}` - update a feed subscription
- `DELETE /api/v1/collections/feeds/{feed_id}` - delete a feed subscription

See: [Collections Feeds API](Collections_Feeds_API.md)

#### Ingestion Sources - `/api/v1/ingestion-sources`

Ingestion Sources provides source-based sync/import for local directories and archive snapshots into `media` or `notes`.

- `POST /api/v1/ingestion-sources` - create a source
- `GET /api/v1/ingestion-sources` - list sources
- `GET /api/v1/ingestion-sources/{source_id}` - get source details and last successful sync summary
- `PATCH /api/v1/ingestion-sources/{source_id}` - update mutable source settings
- `POST /api/v1/ingestion-sources/{source_id}/sync` - enqueue a manual sync
- `POST /api/v1/ingestion-sources/{source_id}/archive` - upload a new ZIP or tar-family snapshot
- `GET /api/v1/ingestion-sources/{source_id}/items` - inspect tracked source items
- `POST /api/v1/ingestion-sources/{source_id}/items/{item_id}/reattach` - reattach a detached notes item

See: [Ingestion Sources API](Ingestion_Sources_API.md)
