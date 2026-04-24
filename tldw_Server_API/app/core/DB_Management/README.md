# DB_Management

Central data stores and database abstractions for content, prompts, notes, evaluations, workflows, and per-user DB paths. Provides a unified backend interface for SQLite and PostgreSQL, full-text search helpers, migrations, and factories used across the API.

## Wave 1 Contract Notes

- The current tree already closes the April 2026 RLS, migration-loader, UserDatabase_v2, trusted-path, media_db API, and backend-FTS findings covered by the Wave 1 rebaseline.
- Shared content-backend cache replacement and reset must close superseded backends deterministically.
- `verify_migrations()` reports missing migration files, checksum mismatches, and non-contiguous available-version gaps.

## 1. Descriptive of Current Feature Set

- Purpose: Provide consistent, secure, and scalable database access for content (Media DB v2), ChaCha Notes/Characters, Prompts/Prompt Studio, Evaluations, Workflows, Collections, Watchlists, and related utilities (paths, backups, migrations).
- Capabilities:
  - Backend abstraction for SQLite and PostgreSQL (pooling, transactions, FTS, schema management).
  - Per-user database layout and helpers (paths, structure validation, backups).
  - Media DB v2 with soft-deletes, versioning, sync logs, FTS5, chunking support, claims, templates, and scope columns (org/team).
  - ChaChaNotes DB for chat, characters, messages, and note-taking.
  - Prompt Studio DB (projects, prompts, iterations, tests) + FTS and indices.
  - Evaluations DB for unified evaluation flows and metrics.
  - Workflows DB for job orchestration and scheduler state.
  - Factories and helpers for content backend detection and initialization.
- Inputs/Outputs:
  - Inputs: SQL queries via backend adapters, Pydantic-validated payloads at endpoints.
  - Outputs: dict-like row results, higher-level DTOs from module methods, exported artifacts (e.g., backups, chatbooks).
- Related Endpoints (selected; all under `/api/v1`):
  - Media/RAG: `tldw_Server_API/app/api/v1/endpoints/media.py:1`, `.../rag_unified.py:1`, `.../chunking.py:1`, `.../chunking_templates.py:1`, `.../paper_search.py:1`, `.../media_embeddings.py:1`, `.../sync.py:1`, `.../vector_stores_openai.py:1`, `.../claims.py:1`
  - Notes/Characters/Chat: `.../chat.py:1`, `.../characters_endpoint.py:1`, `.../character_chat_sessions.py:1`, `.../character_messages.py:1`, `.../notes.py:1`, `.../flashcards.py:1`
  - Prompts/Prompt Studio: `.../prompts.py:1`, `.../prompt_studio_projects.py:1`, `.../prompt_studio_prompts.py:1`, `.../prompt_studio_test_cases.py:1`, `.../prompt_studio_optimization.py:1`, `.../prompt_studio_status.py:1`
  - Evaluations/Workflows/Other: `.../evaluations_unified.py:1`, `.../workflows.py:1`, `.../health.py:1`, `.../watchlists.py:1`, `.../items.py:1`, `.../reading.py:1`, `.../outputs_templates.py:1`
- Related Schemas (selected):
  - Media: `tldw_Server_API/app/api/v1/schemas/media_request_models.py:1`, `tldw_Server_API/app/api/v1/schemas/media_response_models.py:1`, `tldw_Server_API/app/api/v1/schemas/chunking_schema.py:1`, `tldw_Server_API/app/api/v1/schemas/chunking_templates_schemas.py:1`
  - Notes/Prompts: `tldw_Server_API/app/api/v1/schemas/notes_schemas.py:1`, `tldw_Server_API/app/api/v1/schemas/prompt_studio_base.py:1`, `.../prompt_studio_project.py:1`, `.../prompt_studio_schemas.py:1`
  - Evaluations/Watchlists/Other: `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py:1`, `.../watchlists_schemas.py:1`, `.../outputs_schemas.py:1`, `.../outputs_templates_schemas.py:1`, `.../research_schemas.py:1`, `.../reading_schemas.py:1`

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Backend abstraction: `backends.base` defines `DatabaseBackend`, `ConnectionPool`, `QueryResult`, `FTSQuery`, and `BackendFeatures` implemented by `backends.sqlite_backend` and `backends.postgresql_backend`.
  - Content backend config: `content_backend.py` resolves `sqlite` vs `postgresql` via env/config and returns a shared backend for Postgres content mode; SQLite uses per-user file paths instead of a shared pool.
  - Factories: `DB_Manager.py` creates `MediaDatabase`, `CharactersRAGDB` (ChaCha), `PromptStudioDatabase`, `EvaluationsDatabase`, `WorkflowsDatabase`, wiring the right backend.
  - Scope: `scope_context.py` records per-request user/org/team scope for row-level filtering and Postgres RLS policies.
  - Path management: `db_path_utils.py` centralizes per-user DB locations under `<USER_DB_BASE_DIR>/<user_id>/...` (defaults to `Databases/user_databases` under repo root; `USER_DB_BASE` is a deprecated alias for rewrite cache resolution).
- Key Classes/Modules:
  - Legacy media database module and runtime wrappers — content store with schema versioning, FTS, chunking, claims, sync logs, soft deletes, and versioned entities.
  - `ChaChaNotes_DB.CharactersRAGDB` — notes/characters/messages with search and content helpers.
  - `Prompts_DB.PromptsDatabase`, `PromptStudioDatabase.PromptStudioDatabase` — prompt storage and Prompt Studio artifacts with FTS and migrations.
  - `Evaluations_DB.EvaluationsDatabase` — evaluation runs, datasets, metrics.
  - `Workflows_DB.WorkflowsDatabase` — workflow/job orchestration state and scheduler.
  - Backends: `backends.sqlite_backend`, `backends.postgresql_backend`, helpers `fts_translator.py`, `query_utils.py`, `pg_rls_policies.py`.
- Data Models & DB:
  - Media v2 tables include `Media`, `Keywords`, `MediaKeywords`, `Transcripts`, `MediaChunks`, `UnvectorizedMediaChunks`, `DocumentVersions`, `DocumentStructureIndex`, `ChunkingTemplates`, `sync_log`, `Claims`, plus indices and FTS tables.
  - Prompt Studio migrations under `app/core/DB_Management/migrations/` (schema, indices, FTS, templates, watchlists, scope columns).
  - For Postgres content mode, RLS policies are expected (see `pg_rls_policies.py`) and validated by `validate_postgres_content_backend()`.
- Configuration:
  - Content backend selection: `TLDW_CONTENT_DB_BACKEND=sqlite|postgresql` (defaults to `sqlite`).
  - SQLite path: `TLDW_CONTENT_SQLITE_PATH` or `[Database].sqlite_path`; backups: `TLDW_DB_BACKUP_PATH` or `[Database].backup_path` (defaults to `./tldw_DB_Backups/`).
  - Postgres (content) envs: `TLDW_CONTENT_PG_DSN` (or `POSTGRES_TEST_DSN`), `TLDW_CONTENT_PG_HOST|PORT|DATABASE|USER|PASSWORD|SSLMODE` (fallback to `TLDW_PG_*` or `PG*`).
  - Per-user base dir: `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`) used by `db_path_utils.DatabasePaths` for media/notes/prompts/evals/workflows trees; defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.
  - General app config merges env + config files via `load_comprehensive_config`.
- Concurrency & Performance:
  - SQLite: WAL mode, busy_timeout, thread-local pooled connections; memory DBs keep a persistent connection to retain state.
  - Postgres: pooled connections via backend factory; transactions via context managers.
  - FTS: SQLite FTS5 with translator; Postgres FTS/ranking via backend support.
- Error Handling & Security:
  - Custom `DatabaseError`/`ConflictError`/`InputError` in content DBs; exceptions surface as HTTP errors in endpoints.
  - Parameterized queries throughout; strict path handling in `db_path_utils`; row-level scope (`scope_context`) feeds RLS.
  - Soft deletes and versioning minimize data loss and aid synchronization.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - Backends: `backends/` (base, sqlite/postgresql implementations, FTS/query helpers, RLS policies).
  - Content DBs: legacy media DB module, `ChaChaNotes_DB.py`, `Collections_DB.py`, `Watchlists_DB.py`.
  - Prompting/Studio: `Prompts_DB.py`, `PromptStudioDatabase.py`, `migrations/` (SQL/JSON migrations).
  - Evaluations/Workflows: `Evaluations_DB.py`, `Workflows_DB.py`, `Workflows_Scheduler_DB.py`.
  - Utilities: `DB_Manager.py`, `db_path_utils.py`, `db_migration.py`, `migration_tools.py`, `transaction_utils.py`, `async_db_wrapper.py`, `content_backend.py`, `scope_context.py`, `DB_Backups.py`.
- Extension Points:
  - Add a new content DB: create a module with a clear public API, define schema/migrations (SQLite JSON or SQL), implement FTS/indexes, and optionally Postgres DDL + RLS. Expose factory wiring in `DB_Manager.py` if app-wide.
  - Extend Media v2: bump `_CURRENT_SCHEMA_VERSION` and add migration; keep soft-delete/versioning and sync_log semantics consistent; add indices where read paths need them.
  - Postgres content mode: ensure `pg_rls_policies.py` and `validate_postgres_content_backend()` cover new tables and policies.
- Coding Patterns:
  - Use context managers for transactions; rely on `DatabaseBackend.transaction()` or module-provided helpers.
  - Prefer backend-agnostic helpers (`query_utils`, `fts_translator`) and avoid dialect-specific SQL unless guarded.
  - Do not store secrets or PII; never build SQL from untrusted strings.
- Tests:
  - Locations: `tldw_Server_API/tests/DB_Management`, plus feature suites (Media, ChaChaNotesDB, Prompt_Management, Workflows, Claims).
  - Examples: `tests/DB_Management/test_media_postgres_support.py:10`, `.../test_users_db_sqlite.py:8`, `.../test_migration_cli_integration.py:12`, `.../test_db_paths_media_prompts_env.py:17`, `tests/Claims/test_ingestion_claims_sql.py:4`.
  - When adding migrations, include both SQLite and Postgres paths in tests when applicable.
- Local Dev Tips:
  - Content backend quick switch: set `TLDW_CONTENT_DB_BACKEND=sqlite` (default) or `postgresql`, then run feature flows.
  - Create a Media DB for the single-user path:
    ```python
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
    db = create_media_database(client_id="dev-client")
    ```
  - Validate Postgres content backend state:
    ```python
    from tldw_Server_API.app.core.DB_Management.DB_Manager import validate_postgres_content_backend
    validate_postgres_content_backend()
    ```
- Pitfalls & Gotchas:
  - In-memory SQLite DBs are process-local; keep a persistent connection (handled internally) or use file-backed DBs in tests requiring multiple connections.
  - Ensure per-user DB paths are used (avoid legacy root-level Media_DB paths) via `DatabasePaths`.
  - Postgres content mode requires RLS policies and up-to-date schema; use `validate_postgres_content_backend()` on startup.
  - FTS tokenization and LIKE queries differ between backends; use `fts_translator` and helper functions to keep behavior consistent.
- Roadmap/TODOs:
  - Expand property-based tests for Media v2 synchronization and conflict resolution.
  - Add more Postgres integration tests for FTS, ranking, and RLS policy coverage.
  - Unify docstrings across DB modules; remove references to obsolete params where noted in code comments.

---

Example Quick Start (optional)

```python
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
mdb = create_media_database(client_id="example-client")
# Use mdb methods to insert media, update keywords, and search via FTS
```
