# Reading List API

Reading List endpoints support capture, extraction, organization, import/export, and actions (summarize/TTS).

## Endpoints (MVP)

- `POST /api/v1/reading/save` - save a URL (or inline content)
- `GET /api/v1/reading/items` - list items with filters/search
- `GET /api/v1/reading/items/{id}` - item detail with text/clean_html
- `PATCH /api/v1/reading/items/{id}` - update metadata (status/tags/notes/title)
- `DELETE /api/v1/reading/items/{id}` - soft delete (archived) or hard delete
- `POST /api/v1/reading/items/{id}/summarize` - summarize an item
- `POST /api/v1/reading/items/{id}/tts` - generate TTS audio for an item
- `POST /api/v1/reading/items/{id}/archive` - create an archive snapshot (HTML/MD)
- `POST /api/v1/reading/import` - Pocket/Instapaper import (multipart, async job)
- `GET /api/v1/reading/import/jobs` - list reading import jobs
- `GET /api/v1/reading/import/jobs/{job_id}` - get reading import job status
- `GET /api/v1/reading/export` - JSONL or ZIP export
- `POST /api/v1/reading/digests/schedules` - create a digest schedule
- `GET /api/v1/reading/digests/schedules` - list digest schedules
- `GET /api/v1/reading/digests/schedules/{schedule_id}` - get schedule
- `PATCH /api/v1/reading/digests/schedules/{schedule_id}` - update schedule
- `DELETE /api/v1/reading/digests/schedules/{schedule_id}` - delete schedule
- `GET /api/v1/reading/digests/outputs` - list digest output artifacts

## Core object: ReadingItem

Key fields:
- `id`, `url`, `canonical_url`, `title`
- `status`: `saved|reading|read|archived`
- `processing_status`: `processing|ready`
- `tags`, `favorite`
- `media_id` (link to Media DB when ingested)
- `created_at`, `updated_at`, `read_at`

## Save

`POST /api/v1/reading/save`

Request:
```
{
  "url": "https://example.com/article",
  "title": "Example Article",
  "tags": ["ai", "reading"],
  "notes": "Why it matters"
}
```

Response (ReadingItem):
```
{
  "id": 123,
  "title": "Example Article",
  "url": "https://example.com/article",
  "canonical_url": "https://example.com/article",
  "status": "saved",
  "processing_status": "processing",
  "favorite": false,
  "tags": ["ai", "reading"],
  "created_at": "2025-10-19T09:15:00Z",
  "updated_at": "2025-10-19T09:15:00Z"
}
```

Notes:
- Use `content` in the request body to supply inline text (offline/testing).
- Non-HTML URLs (PDF, EPUB, etc.) are routed into the document ingestion pipeline.

## List

`GET /api/v1/reading/items`

Filters: `q`, `tags`, `status`, `favorite`, `domain`, pagination (`page`/`size`), `offset`/`limit`, and `sort`.

Sort options: `updated_desc` (default), `updated_asc`, `created_desc`, `created_asc`, `title_asc`, `title_desc`, `relevance`.

Example:
```
GET /api/v1/reading/items?status=saved&tags=ai&favorite=true&q=vector&sort=updated_desc&limit=20&offset=0
```

Response:
```
{
  "items": [...],
  "total": 42,
  "page": 1,
  "size": 20,
  "offset": 0,
  "limit": 20
}
```

## Detail

`GET /api/v1/reading/items/{id}`

Response (ReadingItemDetail) includes:
- `text` (plain text)
- `clean_html` (sanitized HTML)
- `metadata` (extra fields like `reading_time_seconds`, `media_uuid`)

## Update

`PATCH /api/v1/reading/items/{id}`

Request:
```
{
  "status": "read",
  "favorite": true,
  "tags": ["ai", "priority"],
  "notes": "Follow up later"
}
```

Response: updated ReadingItem.

## Delete

`DELETE /api/v1/reading/items/{id}`

- Soft delete (default) marks the item as `archived`.
- Hard delete: `?hard=true`.

Response:
```
{ "status": "archived", "item_id": 123, "hard": false }
```

## Highlights

Highlights belong to saved Reading captures, not to external Media IDs.

- `POST /api/v1/reading/items/{item_id}/highlight` creates a highlight. The parent
  must be an existing Reading item owned by the authenticated user; otherwise the
  response is 404 (`content_item_not_found`).
- `GET /api/v1/reading/items/{item_id}/highlights` lists the user's highlights.
- `PATCH /api/v1/reading/highlights/{highlight_id}` edits a highlight, and
  `DELETE /api/v1/reading/highlights/{highlight_id}` removes it. Missing or
  inaccessible highlights/parents return 404 on mutation.

Changing highlights updates the owning capture's modification time; an equivalent
patch does not. Reanchoring follows changes to the saved capture and does not
apply results computed before a newer capture/highlight edit. Editing linked
external Media does not reanchor or stale the capture's highlights.

## Summarize

`POST /api/v1/reading/items/{id}/summarize`

Request:
```
{
  "provider": "openai",
  "model": "gpt-4o",
  "prompt": "Summarize for a product brief."
}
```

Response:
```
{
  "item_id": 123,
  "summary": "Short summary text...",
  "provider": "openai",
  "model": "gpt-4o",
  "citations": [
    {
      "item_id": 123,
      "url": "https://example.com/article",
      "canonical_url": "https://example.com/article",
      "title": "Example Article",
      "source": "reading"
    }
  ],
  "generated_at": "2025-10-19T09:20:00Z"
}
```

## TTS

`POST /api/v1/reading/items/{id}/tts`

Request:
```
{
  "model": "kokoro",
  "voice": "af_heart",
  "response_format": "mp3",
  "stream": true,
  "text_source": "text",
  "max_chars": 12000
}
```

Response:
- Streaming audio bytes (Content-Type based on `response_format`)
- Non-streaming mode returns raw bytes in the response body

## Import (Async)

`POST /api/v1/reading/import` (multipart)

Form fields:
- `file`: Pocket JSON or Instapaper CSV
- `source`: `auto|pocket|instapaper`
- `merge_tags`: `true|false`

Example:
```
multipart form
  file = (pocket.json)
  source = pocket
  merge_tags = true
```

Response (202 Accepted):
```
{
  "job_id": 42,
  "job_uuid": "b7fbd5a0-3b37-4a2d-b6c2-0d0b9d3a6c1c",
  "status": "queued"
}
```

Import normalization behavior:
- URLs are canonicalized (tracking params removed where possible) before dedupe/upsert.
- Equivalent URLs in one import batch are merged with tag union and strongest status precedence (`read > archived > reading > saved`).
- Missing titles are backfilled from URL path/domain.
- If merged status is `read` and no timestamp is provided, `read_at` is set automatically.

## Reading Digests

### Create schedule

`POST /api/v1/reading/digests/schedules`

Request:
```
{
  "name": "Morning Digest",
  "cron": "0 8 * * *",
  "timezone": "UTC",
  "format": "md",
  "filters": {
    "status": ["saved", "reading"],
    "tags": ["ai"],
    "limit": 50
  }
}
```

Optional suggestions block:
```
{
  "filters": {
    "status": ["saved"],
    "tags": ["ai"],
    "limit": 50,
    "suggestions": {
      "enabled": true,
      "limit": 5,
      "status": ["saved", "reading"],
      "exclude_tags": ["ignore"],
      "max_age_days": 90,
      "include_read": false,
      "include_archived": false
    }
  }
}
```

Response:
```
{ "id": "5a6f0e9d3d1b4b2f8c3d5a9c7b1a2e3f" }
```

### List schedules

`GET /api/v1/reading/digests/schedules`

Response:
```
[
  {
    "id": "5a6f0e9d3d1b4b2f8c3d5a9c7b1a2e3f",
    "name": "Morning Digest",
    "cron": "0 8 * * *",
    "timezone": "UTC",
    "enabled": true,
    "require_online": false,
    "format": "md",
    "filters": {
      "status": ["saved", "reading"],
      "tags": ["ai"],
      "limit": 50
    },
    "last_run_at": null,
    "next_run_at": "2025-10-20T08:00:00+00:00"
  }
]
```

### Digest suggestions (optional)

If `filters.suggestions.enabled` is true, the digest job adds a `suggestions` list to the template context.

Fields:
- `enabled`: turn suggestions on/off.
- `limit`: max suggestions (default 5).
- `status`: candidate statuses (default `["saved", "reading"]`).
- `exclude_tags`: tags to exclude from suggestions.
- `max_age_days`: drop items older than N days.
- `include_read`: allow `read` items even if status list excludes them.
- `include_archived`: allow `archived` items even if status list excludes them.

### Digest template context

The output template receives:
- `items`: digest items.
- `item_count`: number of digest items.
- `filters`: the schedule filters.
- `suggestions`: suggestion items (optional).
- `suggestions_meta`: `{count, scores, reasons}` keyed by item id (optional).

Sample template snippet:
```
{% if suggestions %}
## Suggested reads
{% for item in suggestions %}
- {{ item.title }} ({{ item.url }})
{% endfor %}
{% endif %}
```

### List digest outputs

`GET /api/v1/reading/digests/outputs`

Response:
```
{
  "items": [
    {
      "output_id": 11,
      "title": "Morning Digest",
      "format": "md",
      "created_at": "2025-10-20T08:00:01Z",
      "download_url": "/api/v1/outputs/11/download",
      "schedule_id": "5a6f0e9d3d1b4b2f8c3d5a9c7b1a2e3f",
      "item_count": 42
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

Notes:
- Output metadata includes `suggestions_count`, `suggestions_item_ids`, and `suggestions_config` when enabled.

## Import Jobs

`GET /api/v1/reading/import/jobs`

Response:
```
{
  "jobs": [
    {
      "job_id": 42,
      "job_uuid": "b7fbd5a0-3b37-4a2d-b6c2-0d0b9d3a6c1c",
      "status": "completed",
      "created_at": "2025-10-19T09:15:00Z",
      "completed_at": "2025-10-19T09:15:05Z",
      "result": {
        "source": "pocket",
        "imported": 12,
        "updated": 3,
        "skipped": 1,
        "errors": []
      }
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

`GET /api/v1/reading/import/jobs/{job_id}`

Response:
```
{
  "job_id": 42,
  "job_uuid": "b7fbd5a0-3b37-4a2d-b6c2-0d0b9d3a6c1c",
  "status": "completed",
  "created_at": "2025-10-19T09:15:00Z",
  "completed_at": "2025-10-19T09:15:05Z",
  "result": {
    "source": "pocket",
    "imported": 12,
    "updated": 3,
    "skipped": 1,
    "errors": []
  }
}
```

## Export

`GET /api/v1/reading/export`

Query params:
- `format`: `jsonl` (default) or `zip`
- filters: `status`, `tags`, `favorite`, `q`, `domain`
- optional flags: `include_metadata`, `include_clean_html`, `include_text`, `include_highlights`

JSONL line example:
```
{"id":123,"url":"https://example.com/article","canonical_url":"https://example.com/article","domain":"example.com","title":"Example Article","summary":"...","notes":null,"status":"saved","favorite":0,"tags":["ai"],"created_at":"2025-10-19T09:15:00Z","updated_at":"2025-10-19T09:15:00Z","read_at":null,"published_at":null,"origin_type":"manual","metadata":{"import_source":"pocket"}}
```

## Archive

### Managed archive output deletion

For structurally managed Reading archives, `DELETE /api/v1/outputs/{id}?hard=true`
also requires `delete_file=true`. Without it the API returns 409
`reading_file_deletion_required` without changing the archive or its file. Soft
deletion keeps the file and its ownership record. Unowned outputs retain their
existing file-retention options; a surviving shared reference protects its file.

Explicit managed deletion commits durable file cleanup before removing metadata.
The returned `file_deleted=false` means no file was synchronously unlinked, not
that the logical deletion failed; do not repeat DELETE to finish cleanup. API and
scheduled output purges with `delete_files=false` skip managed archives. Purge
maintenance triggered automatically by Watchlist reads also preserves managed
archives; reading a Watchlist does not authorize their file deletion. Purge
counts include only records actually removed and files actually unlinked, not
queued cleanup. Both retention and the selected soft-delete grace period are
rechecked when deletion commits.

This managed lifecycle is being rolled out under TASK-13153. It does not by itself
enable the optimistic Reading-delete capability or reconcile legacy archives.

### Managed archive output updates

`PATCH /api/v1/outputs/{id}` changes a managed archive's title and retention
metadata without renaming or rewriting its file. A changed format returns 409
`reading_archive_file_immutable` and rejects the entire request, including any
title/retention edits. Repeating the existing format is allowed. The database
also rejects direct managed path changes. Unowned outputs, including legacy
archives not yet structurally managed, retain their rename/conversion behavior.

This is a partial lifecycle checkpoint, not a readiness claim: generic file
operations still need fencing against late ownership registration and writes
through shared managed paths before the optimistic-delete capability can ship.

### Create an archive

`POST /api/v1/reading/items/{id}/archive`

Request:
```
{
  "format": "html",
  "source": "auto",
  "retention_days": 30
}
```

Response:
```
{
  "output_id": 101,
  "title": "Example Article (archive 20251021_120000)",
  "format": "html",
  "storage_path": "reading_archive_123_example_article_20251021_120000.html",
  "created_at": "2025-10-21T12:00:00+00:00",
  "retention_until": "2025-11-20T12:00:00+00:00",
  "download_url": "/api/v1/outputs/101/download"
}
```
