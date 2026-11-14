# Database layer

This document describes the server's storage spine: the backend abstraction (SQLite/PostgreSQL) with its factory and per-user path resolution, the three independent backend-selection axes, migrations, RLS, and what deliberately does *not* live here (vectors, Elasticsearch).

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Backend abstraction | `app/core/DB_Management/backends/base.py` | `DatabaseBackend`, `ConnectionPool`, `QueryResult`, `FTSQuery`, `BackendFeatures`, `DatabaseConfig` |
| Implementations | `backends/sqlite_backend.py`, `postgresql_backend.py` | SQLite: WAL, busy_timeout, thread-local/pooled connections; PG: pooling |
| Factory | `backends/factory.py` | `create_backend` / `get_backend` — shared-backend caching with eviction (5 s grace) |
| FTS translation | `backends/fts_translator.py`, `query_utils.py` | FTS5 → PG translation; parameter helpers |
| PG extras | `backends/pg_rls_policies.py`, `pg_sharing_schema.py`, `postgresql_locks.py` | RLS ensure functions, sharing schema, advisory locks |
| Content backend | `content_backend.py` | `load_content_db_settings`, `get_content_backend` — SQLite vs PostgreSQL for all content stores |
| Store factories | `DB_Manager.py` | `create_media_database` / `create_chacha_database` / `create_prompt_studio_database` / `create_workflows_database` / `create_chat_workflows_database` / `create_evaluations_database`, `validate_postgres_content_backend`, `reset_content_backend` |
| Paths | `db_path_utils.py` | `DatabasePaths` — every per-user path; `get_user_db_base_dir`, `get_media_db_path`, `get_chacha_db_path`, … |
| Migrations | `db_migration.py`, `migrations/` (SQL), `migrations.py`, `migrate_db.py` | `DatabaseMigrator.verify_migrations()` (checksums/gaps); JSON migration runner; SQLite→PostgreSQL content migrator |
| Media DB v2 | `media_db/` package | native `MediaDatabase`, runtime factory, schema, repositories, `get_media_repository` |
| AuthNZ DB | `app/core/AuthNZ/database.py`, `settings.py` | `DatabasePool` / `get_db_pool()`; `DATABASE_URL` default `sqlite:///./Databases/users.db` |
| Users DB | `DB_Management/Users_DB.py` | `UsersDB` over the pool (Postgres detected via pool presence) |

## Backend selection — three independent axes (verified)

1. **Content stores** (media / ChaCha / prompt studio / workflows / evaluations): `TLDW_CONTENT_DB_BACKEND` env overrides config.txt `[Database] type`; accepted values `sqlite|sqlite3` and `postgres|postgresql`. SQLite means per-user file paths; Postgres means one shared pooled backend, validated at startup with RLS ensured (`ensure_chacha_rls`, `ensure_prompt_studio_rls`). PG DSN comes from `TLDW_CONTENT_PG_DSN` (or host/port/database/user/password/sslmode with fallbacks to `TLDW_PG_*`).
2. **AuthNZ/users**: `DATABASE_URL` (SQLite default). Postgres when the DSN starts with `postgres`; single-user mode deliberately ignores non-SQLite URLs unless `TLDW_USER_DB_BACKEND=postgres(ql)`.
3. **Jobs / Scheduler**: `JOBS_DB_URL` / `JOBS_DB_PATH` and `SCHEDULER_DATABASE_URL` (see [jobs-background.md](./jobs-background.md)).

Per-user layout: `USER_DB_BASE_DIR` (env or config.txt) default `<project_root>/Databases/user_databases/`; user ids normalized with invalid ids rejected; tests get isolated tmp dirs tagged by run/worker id. Per-user ChromaDB lives at `<base>/<user_id>/chroma_storage/` — vectors are **not** this layer; pgvector exists only behind the RAG adapter factory.

## Request → database (dataflow)

1. An endpoint dependency (e.g. `API_Deps/DB_Deps.py`) resolves `DatabasePaths.get_*_path(user_id)`.
2. A `DB_Manager` factory constructs the store class over a backend from the factory (SQLite: per-user file cached by signature; PG: shared pool).
3. Store classes execute SQL via the backend; on PG, `scope_context` sets the user/org/team context that RLS policies consume.
4. Startup (PG content mode) auto-ensures RLS; a content-backend reset deterministically closes superseded backends.

## Boundaries

- **Raw SQL is forbidden outside `DB_Management`** (AGENTS.md contract, ADR-020).
- Elasticsearch/OpenSearch are explicitly not wired (a dedicated not-supported raise exists).
- The module README at `core/DB_Management/README.md` is current and is the in-tree reference.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Path traversal / unavailable storage | Typed `InvalidStoragePathError` / `StorageUnavailableError` |
| PG RLS ensure failure | Startup warning registry; can block startup |
| Migration file missing / checksum mismatch / version gap | `verify_migrations()` reports; media schema/FTS init is lazy per user, not at startup |
| SQLite strict-schema | Missing required columns detected by the AuthNZ strictness check |

## Verified gotchas

1. Two "Databases" directories: `tldw_Server_API/Databases/` ships Postgres/SQLite SQL bootstrap assets; the repo-root `Databases/` is **gitignored runtime data**, absent from fresh clones.
2. `Docs/Code_Documentation/Database.md` is partly stale: it claims the AuthNZ default is `sqlite:///<USER_DB_BASE_DIR>/<SINGLE_USER_FIXED_ID>/tldw.db` — the code default is `sqlite:///./Databases/users.db` — and its line anchors for config resolution have drifted.
3. `TLDW_CONTENT_DB_BACKEND` docs say `sqlite|postgres`; the code also accepts `postgresql`/`sqlite3`.
4. Media DB schema and FTS initialize lazily on first per-user use — a fresh user's first request pays initialization.

## Governing decisions

ADR-020 (`Docs/ADR/020-db-management-per-user-paths-and-content-backend.md`), ADR-055 (prompt studio single implementation). Current references: `Docs/Code_Documentation/Database-Backends.md` (broadly current, minor value-vocabulary drift).

## Related docs

- [app-shell.md](./app-shell.md) — startup RLS auto-ensure
- [authnz.md](./authnz.md) — the users DB pool
- [jobs-background.md](./jobs-background.md) — the jobs/scheduler DBs
