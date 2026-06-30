# Data Tables

Data_Tables contains the Jobs worker that generates and regenerates structured tables from prompts and user sources. The worker resolves media, chat, and RAG source text, calls the configured LLM adapter for JSON table output, normalizes columns and rows, and persists generated table content through the media database APIs used by the Data Tables endpoint.

## Start Here

- `jobs_worker.py` is the module's runtime implementation for `data_table_generate` Jobs work.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/data_tables.py`, declared under `/data-tables`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/data_tables_schemas.py`.
- Related tests: `tldw_Server_API/tests/DataTables/`.
- Covering ADR: `Docs/ADR/023-data-tables-backend-storage-jobs-and-exports.md`

## Responsibilities

- Consume Jobs entries in the `data_tables` domain for table generation and regeneration.
- Normalize job payload fields such as user id, prompt, sources, column hints, model, max rows, and regenerate mode.
- Resolve source text from media records, document versions, transcriptions, chat history, and RAG queries.
- Build bounded prompts for LLM table generation.
- Parse structured JSON output into normalized columns and rows.
- Persist generated content and snapshots through Media DB APIs.
- Cache per-user Media DB and ChaChaNotes DB connections with explicit close-on-eviction behavior.

## Module Map

- `jobs_worker.py`: Jobs worker entrypoint, payload normalization, source resolution, LLM invocation, structured-output parsing, table normalization, persistence, and worker lifecycle.
- `__init__.py`: package marker.

## How It Connects

- `data_tables.py` exposes generate, list, get, update, delete, export, content replacement, regenerate, job status, and job cancel routes.
- `data_tables_schemas.py` defines source inputs, generate and regenerate requests, table summaries, table details, table content, export responses, and job responses.
- The endpoint uses AuthNZ media permissions, RBAC rate limits, Jobs, Media DB, and File Artifacts for export behavior.
- The worker uses ChaChaNotes DB for chat sources, Media DB for media and document sources, RAG for query sources, and LLM_Calls adapters for model execution.
- Sidecar deployments can run the worker via `tldw_Server_API.app.core.Data_Tables.jobs_worker` and `DATA_TABLES_JOBS_WORKER_ENABLED`.

## Architecture Notes

### Core Flow

- Generate and regenerate routes create or reuse a table record in Media DB, store the selected sources, and enqueue a Jobs entry with `job_type="data_table_generate"`.
- `jobs_worker.py` normalizes the Jobs payload, resolves source text, builds a bounded table-generation prompt, invokes the selected LLM_Calls adapter, parses JSON output, then persists normalized columns, rows, status, and snapshots.
- Regeneration relies on the stored table/source snapshot instead of requiring the worker to trust fresh endpoint state.
- Export routes either render table content directly or hand structured table payloads to File Artifacts when generated-file metadata is required.

### State And Data

- Media DB owns table metadata, source rows, column definitions, generated rows, status transitions, and regeneration snapshots.
- Jobs owns lifecycle state such as queued, running, cancellation, failure, and completion; the Data Tables worker should not invent parallel lifecycle state.
- Per-user Media DB and ChaChaNotes DB connections are cached by the worker and closed on eviction to avoid leaking file handles in sidecar processes.

### Security And Operations

- Endpoint permissions, rate limits, and user scoping are enforced before work is queued; the worker assumes the payload already carries the authenticated user boundary.
- Source text and prompt size are truncated by environment-backed limits before provider calls. Keep those limits in place when adding new sources.
- Provider keys and model-specific behavior belong in LLM_Calls. Data_Tables should pass provider configuration through the adapter layer and avoid logging provider secrets.
- Cancellation and failure should flow through Jobs so the endpoint status and admin controls stay accurate.

### Extension Checklist

- New source type: update `jobs_worker.py`, `data_tables_schemas.py`, endpoint validation, and DataTables worker/API tests.
- New output shape: update column normalization, row validation, export behavior, and `test_data_tables_export.py`.
- New worker lifecycle behavior: update Jobs integration tests and sidecar worker settings.

## Extension Points

- Add a source type by updating source resolution in `jobs_worker.py`, then add endpoint/schema support if the API accepts the new source.
- Add a column type by changing `_ALLOWED_COLUMN_TYPES`, `_COLUMN_TYPE_ALIASES`, schemas, and export handling.
- Change prompt sizing or row limits through the environment-backed constants in `jobs_worker.py`.
- Change export behavior by inspecting the Data Tables endpoint and File Artifacts adapters together.
- Add provider-specific LLM behavior through the existing LLM_Calls adapter registry rather than calling a provider directly.

## Testing

- Direct tests live under `tldw_Server_API/tests/DataTables/`.
- Use `test_data_tables_worker.py` for worker behavior.
- Use `test_data_tables_api.py` for route behavior.
- Use `test_data_tables_export.py` for export behavior.
- Use `test_data_tables_jobs_integration.py` for Jobs integration.

## Gotchas

- The core module only contains the worker; most HTTP orchestration is in `endpoints/data_tables.py`.
- The worker truncates sources and prompts according to environment-backed limits before calling the LLM.
- Generated output must parse as structured JSON with matching column and row lengths.
- Database connections are cached per user and closed on eviction; avoid bypassing the cache helpers when extending source resolution.
