Database Migrations Overview

This project maintains multiple logical databases with distinct migration registries:

- Content databases (Media/ChaCha/etc.)
  - Registry: `schema_version` (single-row integer) and `schema_migrations` (for legacy tracking via the DatabaseMigrator).
  - Location: Managed primarily within the package-native `tldw_Server_API/app/core/DB_Management/media_db` package and the `db_migration.py` helper.
  - Backends: SQLite (default) and PostgreSQL (optional). PostgreSQL migrations are applied inline via helper methods on `MediaDatabase`.

- Evaluations/Audit databases
  - Registry: `schema_version` and `migrations` tables for applied steps and verification.
  - Location: `migrations_v5_unified_evaluations.py` and `migrations_v6_audit_logging.py` maintain their own registries to reflect the independent lifecycle of these stores.

Why two registries?

- Content schema updates are tightly coupled to runtime application boot (for both SQLite and PostgreSQL) and often need inline verification (FTS, RLS policies). The single version integer plus inline checks keep the boot path fast and predictable.
- Evaluations/Audit introduce separate, self-contained features and may be operated as distinct databases. Their modules maintain a `migrations` history to aid exporting, auditing and verification tasks without coupling to content store versioning.

Guidance

- When changing Media/ChaCha schema: update `MediaDatabase._CURRENT_SCHEMA_VERSION` and add inline migration helpers or SQLite migration SQL files (under `DB_Management/migrations/`).
- When changing Evaluations/Audit schema: update the appropriate migration module and record the step in the module’s registry tables.
- Keep tests aligned with the registry logic: content tests should validate `schema_version` and FTS/RLS; evaluation tests should validate both `schema_version` and `migrations` integrity.

## SQLite migration transactions

`DatabaseMigrator` runs each migration body, its successful `schema_migrations`
entry, and its `schema_version` update in one `BEGIN IMMEDIATE` transaction.
A failure rolls back that migration's schema and data changes before recording
the failure. Earlier successful migration versions remain applied; a sequence
of versions is not one transaction. Backups remain available for recovery.

New SQL migrations should omit transaction controls. For compatibility, the
runner accepts one outer `BEGIN` / `COMMIT` (or `END`) pair in existing scripts
and owns that boundary itself. Shipped migration files stay unchanged so their
checksums continue to match previously recorded upgrades. Transaction controls
and foreign-key setting changes inside the body are rejected by SQLite's
authorizer, including statements containing comments. Trigger bodies and
semicolons in string literals remain valid.

Place `PRAGMA foreign_keys = OFF` / `ON` (or the equivalent `foreign_keys(OFF)` /
`foreign_keys(ON)` syntax) before and after the migration body, outside any
legacy wrapper. The runner executes these connection settings outside its
transaction. Other setting spellings in the body fail instead of being silently
ignored. These scripts are intended to run through `DatabaseMigrator`.

## Legacy SQLite Media DB recovery

Automatic upgrades of existing file-backed Media DBs start at schema version
22. Versions 1–21 are rejected before invoking the packaged migrator, because
the package does not contain a supported earlier Media DB migration chain.
An empty database is initialized directly at the current version. This boundary
does not describe other logical databases or PostgreSQL migrations.

If startup reports an unsupported legacy version:

1. Stop the server and workers using that database. Preserve an untouched backup
   of the database and any SQLite `-wal` / `-shm` sidecars, plus the associated
   files and configuration. Work from a separate copy.
2. Use the previous server release that successfully opened this database to
   export the content it supports. If that release provides Chatbooks, see the
   [Chatbook guide](User_Guides/WebUI_Extension/Chatbook_User_Guide.md). Export
   coverage varies by release; inventory media, transcripts, notes, metadata,
   and source files before choosing a format. Keep the original backup even
   when export succeeds.
3. Start the current server with a separate, empty destination database/data
   directory. Import the supported export, or re-ingest preserved source files.
   Do not point the current server at the only copy of the legacy database.
4. Compare record counts and representative content, transcripts, relationships,
   and files with the original. Switch to the destination only after checking
   the data you need. Retain the old database for anything the export omitted.

This is a recovery procedure, not a guaranteed lossless automatic conversion.
If no compatible release can export the required data, preserve the backup and
seek a schema-specific migration before proceeding. Do not manually increase
`schema_version` to bypass the check or apply migrations for another database.
