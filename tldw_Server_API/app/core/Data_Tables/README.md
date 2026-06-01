# Data Tables

Data_Tables contains the Jobs worker that generates and regenerates structured tables from prompts and user sources. The worker resolves media, chat, and RAG source text, calls the configured LLM adapter for JSON table output, normalizes columns and rows, and persists generated table content through the media database APIs used by the Data Tables endpoint.

## Start Here

- `jobs_worker.py` is the module's runtime implementation for `data_table_generate` Jobs work.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/data_tables.py`, declared under `/data-tables`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/data_tables_schemas.py`.
- Related tests: `tldw_Server_API/tests/DataTables/`.

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
