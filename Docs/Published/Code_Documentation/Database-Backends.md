Database Backends & Migrations (tldw_server)

Overview
- The project uses a backend abstraction (`DatabaseBackend`) to support SQLite (default) and PostgreSQL (select modules).
- Content stores (Media/ChaChaNotes/Evaluations) default to SQLite with WAL, tuned pragmas, and FTS where applicable.
- Prompt Studio and Workflows include first-class PostgreSQL support and ship their own schema/migration logic.

Backends
- SQLite (default): optimized with WAL, `synchronous=NORMAL`, and `busy_timeout=5000` to reduce lock contention.
- PostgreSQL (optional): available where adapters are implemented; connection managed via lightweight pooling.

Configuration
- Content database backend selection comes from config file/env, normalized via:
  - `TLDW_CONTENT_DB_BACKEND=sqlite|postgres`
  - Additional vars: `TLDW_CONTENT_SQLITE_PATH`, `TLDW_CONTENT_PG_*` (or compatible `POSTGRES_TEST_*`).
- See `tldw_Server_API/app/core/DB_Management/content_backend.py` for resolution rules.

Migrations
- Media Database (SQLite):
  - Uses internal schema versioning with optional JSON-based migrations via `DatabaseMigrator`.
  - If no JSON migration files are present, the system proceeds (no-op) and ensures FTS structures are initialized.
  - Entry points: `media_db.native_class.MediaDatabase` (initializes/migrates on first use).

- ChaChaNotes (SQLite):
  - Managed version upgrades inside `ChaChaNotes_DB.py` with FTS triggers and sync logs.

- Evaluations (SQLite):
  - Dedicated migrations: v5 unified evaluations table, v6 audit logging enhancements.
  - See `migrations_v5_unified_evaluations.py` and `migrations_v6_audit_logging.py`.

- Prompt Studio (PostgreSQL):
  - Schema and indexes defined in SQL files under `DB_Management/migrations/`.
  - Adapter applies migrations and FTS setup (`PromptStudioDatabase`).

- Workflows (SQLite/PG):
  - SQLite schema included in adapter; PostgreSQL schema applied via adapter migrations.
  - SQLite→PostgreSQL migration utility: `DB_Management/migration_tools.py`.

CLI Utilities
- Generic JSON-file migration runner: `tldw_Server_API/app/core/DB_Management/migrate_db.py`.
- SQLite→PostgreSQL content/workflows migrator: `tldw_Server_API/app/core/DB_Management/migration_tools.py`.

Backups
- Centralized DB path helpers: `db_path_utils.DatabasePaths` (per-user). Backup helper: `DB_Backups.setup_backup_config()`.
- Recommend using SQLite `backup` API (full) or `VACUUM INTO` (incremental) for safe copies.

Notes
- Keep raw SQL within `DB_Management` modules per project guidelines.
- Prefer parameterized queries; avoid logging secrets or full SQL with sensitive values.
