# Data Tables Backend (Media DB + Async Jobs)

## Summary
Add backend support for Data Tables with async generation jobs, Media DB storage, reproducible RAG snapshots, and server-side exports. This design aligns with existing Media DB patterns, JobManager workflows, and AuthNZ scoping.

## Related ADR

- `Docs/ADR/023-data-tables-backend-storage-jobs-and-exports.md` records the accepted, bounded backend storage, Jobs, source snapshot, UUID table identity, and server-side export decision backfilled from this design.

## Goals
- Store all Data Tables in the per-user Media DB.
- Generate tables via background jobs with progress and cancellation.
- Snapshot RAG query sources so regeneration is reproducible.
- Provide server-side exports (CSV, XLSX, JSON) with streaming or async jobs.
- Enforce AuthNZ, owner scoping, and rate limits on all endpoints.

## Non-Goals
- UI or frontend implementation details.
- Full editing workflow (row/column editing is a separate feature).
- Historical versioning beyond the latest generated table (can be added later).

## Decisions And Constraints
- Data tables live in the per-user Media DB (SQLite default or shared Postgres backend).
- Generation is async via JobManager, not synchronous.
- RAG queries persist chunk snapshots (ids and content) for reproducibility.
- Exports are server-side only for security and auditability.
- External API identifiers use `uuid`; internal numeric `id` is DB-only.
- Export tracking reuses File_Artifacts (Collections DB) instead of a new Media DB table.

## Data Model (Media DB)
Use Media_DB_v2 schema patterns: UUIDs, client_id, version, deleted flag, and last_modified timestamps. JSON is stored as TEXT.

### Tables

#### data_tables
Stores table metadata and generation status.

- id INTEGER PRIMARY KEY
- uuid TEXT UNIQUE NOT NULL
- name TEXT NOT NULL
- description TEXT
- prompt TEXT NOT NULL
- column_hints_json TEXT
- status TEXT NOT NULL DEFAULT 'queued'  -- queued|running|ready|failed|cancelled
- row_count INTEGER NOT NULL DEFAULT 0
- generation_model TEXT
- last_error TEXT
- created_at TEXT NOT NULL
- updated_at TEXT NOT NULL
- last_modified DATETIME NOT NULL
- version INTEGER NOT NULL DEFAULT 1
- client_id TEXT NOT NULL
- deleted BOOLEAN NOT NULL DEFAULT 0
- prev_version INTEGER
- merge_parent_uuid TEXT

Indexes:
- idx_data_tables_status on (status)
- idx_data_tables_updated on (updated_at DESC)

#### data_table_columns
Stores column definitions for a table.

- id INTEGER PRIMARY KEY
- table_id INTEGER NOT NULL
- column_id TEXT NOT NULL
- name TEXT NOT NULL
- type TEXT NOT NULL  -- text|number|date|url|boolean|currency
- description TEXT
- format TEXT
- position INTEGER NOT NULL
- created_at TEXT NOT NULL
- last_modified DATETIME NOT NULL
- version INTEGER NOT NULL DEFAULT 1
- client_id TEXT NOT NULL
- deleted BOOLEAN NOT NULL DEFAULT 0

Indexes:
- idx_data_table_columns_table on (table_id, position)
- ux_data_table_columns_table_column on (table_id, column_id)
- ux_data_table_columns_table_position on (table_id, position)

#### data_table_rows
Stores rows as JSON blobs for paging and export.

- id INTEGER PRIMARY KEY
- table_id INTEGER NOT NULL
- row_id TEXT NOT NULL
- row_index INTEGER NOT NULL
- row_json TEXT NOT NULL  -- JSON object keyed by column_id
- row_hash TEXT
- created_at TEXT NOT NULL
- last_modified DATETIME NOT NULL
- version INTEGER NOT NULL DEFAULT 1
- client_id TEXT NOT NULL
- deleted BOOLEAN NOT NULL DEFAULT 0

Indexes:
- idx_data_table_rows_table on (table_id, row_index)
- ux_data_table_rows_table_row on (table_id, row_id)
- ux_data_table_rows_table_index on (table_id, row_index)

#### data_table_sources
Stores source references and snapshots (especially for rag_query).

- id INTEGER PRIMARY KEY
- table_id INTEGER NOT NULL
- source_type TEXT NOT NULL  -- chat|document|rag_query
- source_id TEXT NOT NULL
- title TEXT
- snapshot_json TEXT  -- rag_query snapshot or captured source text
- retrieval_params_json TEXT
- created_at TEXT NOT NULL
- last_modified DATETIME NOT NULL
- version INTEGER NOT NULL DEFAULT 1
- client_id TEXT NOT NULL
- deleted BOOLEAN NOT NULL DEFAULT 0

Indexes:
- idx_data_table_sources_table on (table_id)

#### exports (File_Artifacts)
Exports are tracked via File_Artifacts in the Collections DB (`file_artifacts` table). No `data_table_exports` table.
The data tables export endpoint returns `file_id` (and optional `job_id`) so clients can poll
`GET /api/v1/files/{file_id}` and download via `GET /api/v1/files/{file_id}/export`.
Use the File_Artifacts adapter `data_table` for table exports (csv/json/xlsx formats).

### RAG Snapshot Payload
For rag_query sources, store the retrieval config and chunk snapshot so regeneration is deterministic.

Example snapshot_json (rag_query):
```
{
  "query": "compare pricing across vendors",
  "retrieval": {
    "top_k": 20,
    "bm25": true,
    "vector": true,
    "rerank": true,
    "filters": {"tags": ["pricing"]}
  },
  "chunks": [
    {
      "chunk_id": "chunk_abc123",
      "media_id": 42,
      "chunk_text": "...",
      "chunk_hash": "sha256...",
      "score": 0.87,
      "rank": 1,
      "source_uri": "media://42#chunk_abc123"
    }
  ]
}
```

## API Endpoints
All endpoints use AuthNZ deps and owner scoping. Media DB access uses `get_media_db_for_user`.

### Create Generation Job
`POST /api/v1/data-tables/generate`
- Body: prompt, sources, column_hints, model, max_rows, retrieval overrides
- Response (202): job_id, table_uuid, status=queued

### Job Status
`GET /api/v1/data-tables/jobs/{job_id}`
- Returns job status, progress stage, table_uuid

### Cancel Job
`DELETE /api/v1/data-tables/jobs/{job_id}`
- Cancels queued or running job (owner/admin)

### List Tables
`GET /api/v1/data-tables`
- Returns summaries only (no row payloads)
- Params: limit, offset, status, search

### Get Table
`GET /api/v1/data-tables/{table_uuid}`
- Returns metadata + rows with pagination
- Params: limit, offset (or cursor)

### Update Metadata
`PATCH /api/v1/data-tables/{table_uuid}`
- Updates name/description only (no row edits in v1)

### Regenerate
`POST /api/v1/data-tables/{table_uuid}/regenerate`
- Creates a new generation job using stored sources/snapshots
- Optional prompt override

### Delete
`DELETE /api/v1/data-tables/{table_uuid}`
- Soft delete in Media DB

### Export
`GET /api/v1/data-tables/{table_uuid}/export?format=csv|xlsx|json`
- For small payloads: stream response with Content-Disposition
- For large payloads: return 202 with file_id and export job_id (File_Artifacts)

## Job Worker Flow
Create a dedicated worker similar to media ingest and chatbooks jobs.

### Job Domain
- domain: `data_tables`
- queue: `default`
- job_type: `data_table_generate`
- export jobs use File_Artifacts (`domain=files`, `job_type=file_artifact_export`)

### Payload
- table_uuid
- user_id
- prompt
- sources
- column_hints
- model
- max_rows
- retrieval_config

### Stages
1. **resolve_sources**
   - Fetch chat/document text.
   - For rag_query, run RAG retrieval and store snapshot_json + retrieval_params_json.
   - Persist sources in data_table_sources.
2. **build_prompt**
   - Assemble prompt with source text, hints, and schema.
   - Enforce token and row limits.
3. **llm_generate**
   - Call LLM with structured output schema.
   - Use strict JSON parsing and retry on invalid output.
4. **validate_and_normalize**
   - Normalize column names and types.
   - Validate row count, null handling, and type coercion.
5. **persist**
   - For regenerate, soft-delete prior columns/rows before insert (same transaction).
   - Write data_table_columns and data_table_rows in a transaction.
   - Update data_tables status, row_count, updated_at, last_modified.
6. **finalize**
   - Update job result and status.
   - Capture last_error on failure.

### Cancellation
- Check `jm.should_cancel(job_id)` at each stage boundary.
- If cancelled, mark job cancelled and set table status to cancelled.

### Progress
- Update job progress with stages: resolve_sources, build_prompt, llm_generate, validate, persist, finalize.

## Export Flow
Server-side export only.

- Small export: generate in-memory and stream.
- Large export: create a File_Artifacts export job and write to user outputs dir.
- XLSX uses existing openpyxl stack (see File_Artifacts adapters) to avoid new dependencies.
- CSV/JSON/XLSX exports use the File_Artifacts adapter `data_table`.
- Export permissions enforced by owner scope and table ownership.

## Security And Rate Limits
- Enforce AuthNZ on every endpoint.
- Validate source ids against user ownership.
- Apply rate limits on generate/regenerate/export endpoints.
- Sanitize prompts and source text in logs (no secrets).

## Testing
- Unit tests for MediaDatabase CRUD (tables, columns, rows, sources).
- Unit tests for RAG snapshot creation and storage.
- Integration tests for job lifecycle: submit -> status -> completion -> get/export.
- Export tests for CSV, JSON, XLSX (openpyxl optional skip).
- Integration tests for File_Artifacts export path (file_id returned, job status, download).
- Negative tests for ownership and invalid sources.

## Migration Notes
- Add new tables to Media_DB_v2 schema and migration routines.
- Ensure schema_version bump and test migrations for both SQLite and Postgres backends.

## Follow-up Tasks
- Update schemas/routes to accept `table_uuid` and return `uuid` everywhere; resolve to internal `id` server-side.
- Add schema constraints for `status=cancelled`, unique `position` and `row_index`, and enforce `row_json` keys by `column_id`.
- Implement regenerate soft-delete for prior columns/rows inside the persist transaction.
- Ensure job updates set both `updated_at` and `last_modified`.
- Wire export path to File_Artifacts (`POST /api/v1/files/create`, `GET /api/v1/files/{file_id}`, `GET /api/v1/files/{file_id}/export`).
- Add a File_Artifacts adapter `data_table` supporting csv/json/xlsx exports and document the behavior.

## Staged Implementation Plan

### Stage 1: Media DB Schema + Migrations
**Goal**: Add data tables schema to Media_DB_v2 with migrations and version bumps.
**Success Criteria**: New tables (data_tables, data_table_columns, data_table_rows, data_table_sources, optional data_table_exports) exist in SQLite and Postgres backends; schema_version updated.
**Tests**: Migration tests for SQLite/Postgres; smoke test that tables exist and are writable.
**Status**: Complete

### Stage 2: MediaDatabase CRUD Layer
**Goal**: Implement MediaDatabase helpers for table CRUD, paging, and source snapshot persistence.
**Success Criteria**: CRUD methods for tables/columns/rows/sources exist; row pagination works; rag_query snapshot JSON persists and round-trips.
**Tests**: Unit tests for create/update/get/list/delete; pagination tests; snapshot serialization tests.
**Status**: Complete

### Stage 3: API Schemas + Endpoints
**Goal**: Add API schemas and endpoints for table CRUD and job submission/status.
**Success Criteria**: Endpoints implemented with AuthNZ dependencies and owner scoping; OpenAPI updated; table list and detail endpoints return paginated rows.
**Tests**: Integration tests for create/list/get/update/delete; auth enforcement tests.
**Status**: Complete

### Stage 4: Jobs + Worker Pipeline
**Goal**: Implement async generation/regeneration via JobManager worker.
**Success Criteria**: Job submission returns 202 with job_id/table_id; worker stages update progress; cancellation respected; regeneration uses stored snapshots.
**Tests**: Integration tests for job lifecycle (submit -> status -> completion); cancel mid-run; regeneration determinism using stored snapshots.
**Status**: Complete

### Stage 5: Server-side Export
**Goal**: Implement server-side CSV/JSON/XLSX export with streaming or async export jobs.
**Success Criteria**: Export endpoint returns streamed content for small tables and 202 for large tables; export jobs write to outputs dir; download endpoint works.
**Tests**: Export integration tests for csv/json/xlsx; size threshold tests; permissions tests.
**Status**: Complete
