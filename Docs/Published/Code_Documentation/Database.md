# Database Overview

This document describes the persistence layer used by tldw_server: relational databases (SQLite by default, PostgreSQL where applicable), per-user storage layout, vector storage (ChromaDB), and where each module keeps its data. It links to focused docs for detailed schemas and API usage.

## At a Glance
- Default relational store: SQLite (production supports PostgreSQL for AuthNZ and is wired for content DB backends).
- Vector store: ChromaDB per user, on disk.
- Per-user data root: `USER_DB_BASE_DIR` (defined in `tldw_Server_API.app.core.config`, defaults to `Databases/user_databases/` under the project root; override via environment variable or `Config_Files/config.txt`; `USER_DB_BASE` is a deprecated alias for rewrite cache resolution only).
- Soft deletes, versioning, and sync logging are first-class across content DBs.
- FTS5 is used for full-text search in multiple databases (Media, Prompts, Prompt Studio).

Related docs:
- Media DB v2: Docs/Code_Documentation/Databases/Media_DB_v2.md
- ChaChaNotes DB: Docs/Code_Documentation/Databases/ChaChaNotes_DB.md

## Storage Locations and Backends

Environment and config determine paths. Defaults are created on startup if missing.

- `USER_DB_BASE_DIR` (base for per-user data)
  - Default: `Databases/user_databases` (resolved from repo root; `~` expands; normalized for Windows)
  - Defined in `tldw_Server_API.app.core.config` (see `tldw_Server_API/app/core/config.py:403`)
  - Override via environment variable or `Config_Files/config.txt` as needed.
  - `USER_DB_BASE` is deprecated and treated as an alias for rewrite cache resolution.

- AuthNZ main database (`DATABASE_URL`)
  - Default (single-user): `sqlite:///<USER_DB_BASE_DIR>/<SINGLE_USER_FIXED_ID>/tldw.db`
  - Multi-user recommended: PostgreSQL `postgresql://...`
  - Code: tldw_Server_API/app/core/AuthNZ/database.py:1, tldw_Server_API/app/core/config.py:408

- Media database (content)
  - Default path (per user): `<USER_DB_BASE_DIR>/<user_id>/Media_DB_v2.db`
  - Root-level DB paths are deprecated; prefer per-user files for isolation and portability
  - Factory: tldw_Server_API/app/core/DB_Management/DB_Manager.py:33
  - Engine/backends: SQLite (default), PostgreSQL support via backends layer
  - Library seam: tldw_Server_API/app/core/DB_Management/media_db/native_class.py:1

- ChaChaNotes (notes, chats, characters)
  - Per user: `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db`
  - Dependency: tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:1

- Prompts and Prompt Studio
  - Per user: `<USER_DB_BASE_DIR>/<user_id>/prompts_user_dbs/user_prompts_v2.sqlite`
  - Dependency: tldw_Server_API/app/api/v1/API_Deps/Prompts_DB_Deps.py:1
  - Library: tldw_Server_API/app/core/DB_Management/Prompts_DB.py:180
  - Prompt Studio extension: tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:1
  - Prompt Studio DB path: `<USER_DB_BASE_DIR>/<user_id>/prompt_studio_dbs/prompt_studio.db`

- Evaluations (OpenAI-compatible + internal/unified)
  - Default DB: `Databases/evaluations.db`
  - Used by background workers (ephemeral cleanup) and APIs
  - Code: tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:1, tldw_Server_API/app/main.py:343
  - Per-user audit/evaluations paths available via db_path_utils for some DI flows

- Vector store (ChromaDB + meta SQLite per user)
  - Chroma storage: `<USER_DB_BASE_DIR>/<user_id>/chroma_storage/`
  - Meta/Jobs DBs: `<USER_DB_BASE_DIR>/<user_id>/vector_store/` with:
    - `vector_store_meta.db`, `vector_store_batches.db`, `media_embedding_jobs.db`
  - Code: tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py:130, vector_store_meta_db.py:1, vector_store_batches_db.py:1, media_embedding_jobs_db.py:1

- Per-user storage (non-DB assets)
  - Outputs: `<USER_DB_BASE_DIR>/<user_id>/outputs/`
  - Voices: `<USER_DB_BASE_DIR>/<user_id>/voices/`
  - Rewrite cache: `<USER_DB_BASE_DIR>/<user_id>/Rewrite_Cache/rewrite_cache.jsonl`
  - RAG personalization: `<USER_DB_BASE_DIR>/<user_id>/rag_personalization.json`

- MCP Unified module
  - `MCP_DATABASE_URL` (default `sqlite+aiosqlite:///./Databases/mcp_unified.db`)
  - Code: tldw_Server_API/app/core/MCP_unified/config.py:1

Notes:
- Elasticsearch/OpenSearch paths are present in legacy wiring but not implemented for content DB operations in current code (placeholders raise NotImplemented). Prefer SQLite/PostgreSQL backends.

## Key Databases and Capabilities

### AuthNZ (Users)
- Backend: SQLite (single-user) or PostgreSQL (multi-user recommended).
- Pooling/transactions: tldw_Server_API/app/core/AuthNZ/database.py:1
- Schema files: `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`, `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql`
- Settings: `DATABASE_URL`, `AUTH_MODE`, `SINGLE_USER_FIXED_ID`

### Media DB v2
- Purpose: Media items, transcripts, chunk queues, document versions, keywords, sync log, and chunking templates.
- Features: soft delete, versioning, sync logging, FTS5 (`media_fts`, `keyword_fts`), triggers for optimistic concurrency.
- Current schema version: 5 (MediaDatabase._CURRENT_SCHEMA_VERSION)
- Backends: SQLite default; backend layer supports PostgreSQL for many operations.
- Doc: Docs/Code_Documentation/Databases/Media_DB_v2.md
  - Sections: Claims API, Chunking Templates API, Troubleshooting
- Code: tldw_Server_API/app/core/DB_Management/media_db/media_database.py:1, tldw_Server_API/app/core/DB_Management/media_db/native_class.py:1

### ChaChaNotes DB
- Purpose: Notes, characters, chats, tags, flashcards, decks, reviews, sessions.
- Per-user file: `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db`
- Current schema version: 7
- DI: `get_chacha_db_for_user`
- Doc: Docs/Code_Documentation/Databases/ChaChaNotes_DB.md
- Code: tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:1

### Prompts / Prompt Studio DB
- Purpose: Prompt library, keyword FTS, projects, signatures, test cases, evaluations, optimizations.
- Per-user file: `<USER_DB_BASE_DIR>/<user_id>/prompts_user_dbs/user_prompts_v2.sqlite`
- FTS5 enabled for prompts/keywords; Prompt Studio migrations under `.../migrations`.
- DI: `get_prompts_db_for_user`
- Code: tldw_Server_API/app/core/DB_Management/Prompts_DB.py:180, PromptStudioDatabase.py:1

### Evaluations DB (Unified)
- Purpose: OpenAI-compatible evaluations, evaluation runs, datasets; internal/unified evaluations; webhook registrations; embedding A/B tests; ephemeral collections registry.
- Default file: `Databases/evaluations.db`
- Unified migration: `migrations_v5_unified_evaluations.py`
- Code: tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:1

### Vector Store and Jobs (Per user)
- ChromaDB storage: `<USER_DB_BASE_DIR>/<user_id>/chroma_storage` with per-user `ChromaDBManager`.
- Meta and batch/job tracking: `vector_store_meta.db`, `vector_store_batches.db`, `media_embedding_jobs.db`.
- Code: tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py:130, vector_store_meta_db.py:1, vector_store_batches_db.py:1, media_embedding_jobs_db.py:1

## Data Management Patterns
- Soft delete and versioning on content tables (many tables have `deleted`, `version`, `prev_version`, `uuid`).
- Sync logging: `sync_log` records create/update/delete/link/unlink for synchronization/export.
- FTS5 virtual tables for search; kept in sync by library code.
- Concurrency: SQLite uses WAL/busy timeouts in some modules (e.g., Prompts DB, Users SQLite schema initialization). Triggers enforce optimistic concurrency where applicable.

## Dependency Injection (FastAPI)
- ChaChaNotes: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`:1 (`get_chacha_db_for_user`)
- Prompts/Prompt Studio: `tldw_Server_API/app/api/v1/API_Deps/Prompts_DB_Deps.py`:1 (`get_prompts_db_for_user`)
- Unified audit service: `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`:1 (`get_audit_service_for_user`)

## Configuration Reference
- `USER_DB_BASE_DIR`: Base dir for per-user SQLite and ChromaDB storage.
- `DATABASE_URL`: AuthNZ DB (SQLite or PostgreSQL). Defaults to per-user `tldw.db` in single-user mode.
- `SINGLE_USER_FIXED_ID`: User ID used to compute single-user default paths.
- `MCP_DATABASE_URL`: MCP unified DB (SQLite+aiosqlite or PostgreSQL).
- Content DB settings: `tldw_Server_API/Config_Files/config.txt` (Media DB path, backend selection).

## Backups and Migrations
- Media DB v2 exposes `backup_database(path)`; automated/incremental helpers are placeholders.
- Prompt Studio applies migrations from `tldw_Server_API/app/core/DB_Management/migrations/`.
- Evaluations DB applies unified migration v5 automatically at init.
- General backup helpers exist in DB_Management, but prefer per-library backup functions where available.

## Testing Pointers
- Media DB v2: tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py:35
- ChaChaNotes: tldw_Server_API/tests/Notes/test_notes_library_unit.py:12
- Evaluations: tldw_Server_API/tests/Evaluations/integration/test_api_endpoints.py:125
- Prompt Studio: tldw_Server_API/tests/prompt_studio/test_database.py:32
- Vector store admin/meta: tldw_Server_API/tests/VectorStores/test_vector_stores_admin_users.py:8

## Notes on Elasticsearch/OpenSearch
Legacy wiring mentions Elasticsearch/OpenSearch; current code paths for these backends are placeholders and raise `NotImplementedError` in DB manager functions. Use SQLite/PostgreSQL paths described above.
